"""Run Dead Cells faster/slower from Python (replaces Cheat Engine's speedhack).

    python speedhack.py 2      # 2x speed
    python speedhack.py 1      # back to normal
    python speedhack.py        # show current speed

The first call loads speedhack_dll/dqn_speedhack{32,64}.dll into the game
process; after that only a shared-memory number is changed. The speed stays
in effect until you change it or restart the game. Don't run Cheat Engine's
speedhack at the same time.
"""
import ctypes
import ctypes.wintypes as wt
import mmap
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DLL_DIR = os.path.join(HERE, "speedhack_dll")
WINDOW_TITLE = "Dead Cells"

PROCESS_ALL_ACCESS = 0x1F0FFF
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE = 0x1000, 0x2000, 0x8000
PAGE_READWRITE = 0x04
LIST_MODULES_32BIT, LIST_MODULES_64BIT = 0x01, 0x02

_maps = {}  # keep the shared-memory views alive for this Python process


def _k32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wt.HANDLE
    k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k.VirtualAllocEx.restype = ctypes.c_void_p
    k.VirtualAllocEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
    k.VirtualFreeEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
    k.WriteProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_size_t)]
    k.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_size_t)]
    k.CreateRemoteThread.restype = wt.HANDLE
    k.CreateRemoteThread.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                     ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
    k.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    k.GetExitCodeThread.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
    k.IsWow64Process.argtypes = [wt.HANDLE, ctypes.POINTER(wt.BOOL)]
    k.CloseHandle.argtypes = [wt.HANDLE]
    return k


def _psapi():
    p = ctypes.WinDLL("psapi", use_last_error=True)
    p.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_void_p), wt.DWORD,
                                       ctypes.POINTER(wt.DWORD), wt.DWORD]
    p.GetModuleFileNameExW.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.LPWSTR, wt.DWORD]
    return p


def find_game_pid(title=WINDOW_TITLE):
    import win32gui
    import win32process
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        raise RuntimeError("window '%s' not found - is the game running?" % title)
    return win32process.GetWindowThreadProcessId(hwnd)[1]


def _is_32bit(k, h):
    if struct.calcsize("P") == 4:
        return True
    wow = wt.BOOL()
    if not k.IsWow64Process(h, ctypes.byref(wow)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(wow.value)


def _modules(p, h, flag):
    arr = (ctypes.c_void_p * 1024)()
    need = wt.DWORD()
    if not p.EnumProcessModulesEx(h, arr, ctypes.sizeof(arr), ctypes.byref(need), flag):
        raise ctypes.WinError(ctypes.get_last_error())
    out = {}
    buf = ctypes.create_unicode_buffer(1024)
    for i in range(min(need.value // ctypes.sizeof(ctypes.c_void_p), 1024)):
        if arr[i] and p.GetModuleFileNameExW(h, arr[i], buf, 1024):
            out[os.path.basename(buf.value).lower()] = arr[i]
    return out


def _read(k, h, addr, size):
    buf = ctypes.create_string_buffer(size)
    n = ctypes.c_size_t()
    if not k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, size, ctypes.byref(n)):
        raise ctypes.WinError(ctypes.get_last_error())
    return buf.raw


def _remote_export(k, h, base, func):
    """Find an exported function's address inside ANOTHER process by reading
    its PE export table (needed when the game is 32-bit and Python is 64-bit)."""
    e_lfanew = struct.unpack_from("<I", _read(k, h, base, 64), 0x3C)[0]
    nt = _read(k, h, base + e_lfanew, 0x120)
    magic = struct.unpack_from("<H", nt, 0x18)[0]
    dd = 0x18 + (96 if magic == 0x10B else 112)
    exp_rva, exp_size = struct.unpack_from("<II", nt, dd)
    blob = _read(k, h, base + exp_rva, exp_size)
    n_names, rva_funcs, rva_names, rva_ords = struct.unpack_from("<IIII", blob, 24)
    get = lambda rva, fmt: struct.unpack_from(fmt, blob, rva - exp_rva)
    names = get(rva_names, "<%dI" % n_names)
    ords = get(rva_ords, "<%dH" % n_names)
    target = func.encode()
    for i, r in enumerate(names):
        off = r - exp_rva
        if blob[off:off + len(target) + 1] == target + b"\0":
            frva = get(rva_funcs + 4 * ords[i], "<I")[0]
            if exp_rva <= frva < exp_rva + exp_size:
                raise RuntimeError("%s is forwarded, cannot use it" % func)
            return base + frva
    raise RuntimeError("%s not found in remote module" % func)


def _dll_for(is32):
    path = os.path.join(DLL_DIR, "dqn_speedhack32.dll" if is32 else "dqn_speedhack64.dll")
    if not os.path.exists(path):
        raise RuntimeError("missing " + path)
    return path


def is_injected(pid=None):
    pid = pid or find_game_pid()
    k, p = _k32(), _psapi()
    h = k.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    try:
        flag = LIST_MODULES_32BIT if _is_32bit(k, h) else LIST_MODULES_64BIT
        return any(n.startswith("dqn_speedhack") for n in _modules(p, h, flag))
    finally:
        k.CloseHandle(h)


def inject(pid=None, timeout_s=10):
    pid = pid or find_game_pid()
    k, p = _k32(), _psapi()
    h = k.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not h:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        is32 = _is_32bit(k, h)
        if not is32 and struct.calcsize("P") == 4:
            raise RuntimeError("the game is 64-bit but this Python is 32-bit")
        flag = LIST_MODULES_32BIT if is32 else LIST_MODULES_64BIT
        mods = _modules(p, h, flag)
        if any(n.startswith("dqn_speedhack") for n in mods):
            return False
        load_library = _remote_export(k, h, mods["kernel32.dll"], "LoadLibraryW")
        path = _dll_for(is32)
        data = ctypes.create_unicode_buffer(path)
        size = ctypes.sizeof(data)
        mem = k.VirtualAllocEx(h, None, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not mem:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not k.WriteProcessMemory(h, mem, data, size, None):
                raise ctypes.WinError(ctypes.get_last_error())
            th = k.CreateRemoteThread(h, None, 0, ctypes.c_void_p(load_library), mem, 0, None)
            if not th:
                raise ctypes.WinError(ctypes.get_last_error())
            k.WaitForSingleObject(th, timeout_s * 1000)
            k.CloseHandle(th)
            # (the thread exit code is only the low 32 bits of the module handle,
            #  so confirm by listing the game's modules instead)
            if not any(n.startswith("dqn_speedhack") for n in _modules(p, h, flag)):
                raise RuntimeError("LoadLibraryW failed inside the game (path: %s)" % path)
        finally:
            k.VirtualFreeEx(h, mem, 0, MEM_RELEASE)
        time.sleep(1.5)  # let the DLL patch the game's modules
        return True
    finally:
        k.CloseHandle(h)


def _shared(pid):
    if pid not in _maps:
        _maps[pid] = mmap.mmap(-1, 8, tagname="Local\\DQNSpeedhack_%d" % pid)
    return _maps[pid]


def get_speed(pid=None):
    pid = pid or find_game_pid()
    v = struct.unpack("<d", _shared(pid)[:8])[0]
    return v if 0.05 <= v <= 20 else 1.0


def set_speed(speed, pid=None):
    """Inject if needed, then set the game speed factor (1.0 = normal)."""
    if not (0.05 <= speed <= 20):
        raise ValueError("speed must be between 0.05 and 20")
    pid = pid or find_game_pid()
    m = _shared(pid)  # create the shared value first, so the DLL picks it up on load
    m[:8] = struct.pack("<d", float(speed))
    injected = inject(pid)
    m[:8] = struct.pack("<d", float(speed))
    return injected


if __name__ == "__main__":
    if len(sys.argv) > 1:
        s = float(sys.argv[1])
        newly = set_speed(s)
        print("%s; game speed set to %.2fx" % ("speedhack loaded into the game" if newly else "already loaded", s))
    else:
        pid = find_game_pid()
        print("loaded:", is_injected(pid), "| speed: %.2fx" % get_speed(pid))
