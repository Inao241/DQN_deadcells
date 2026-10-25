"""Run Dead Cells faster/slower from Python with Cheat Engine 7.7's own speedhack.

    python speedhack.py 2      # 2x speed
    python speedhack.py 1      # back to normal
    python speedhack.py        # show status

Cheat Engine 7.x no longer ships a speedhack DLL: the speedhack is autorun/SpeedhackV3.lua,
which compiles C hooks for GetTickCount64 / GetTickCount / timeGetTime /
RtlQueryPerformanceCounter into the game and reads the speed from a float in the game,
`speedhack_wantedspeed`. This module uses that code unchanged:

  first call for a game process:
    1. make sure speedhack_dll/dqn_speedhack.lua is in Cheat Engine's autorun folder
       (it does nothing unless CE was started by this module);
    2. start Cheat Engine minimized with DQN_SPEEDHACK_PID/SPEED/OUT set; the Lua opens
       the game, calls speedhack_setSpeed(speed) and reports where speedhack_wantedspeed is;
    3. close Cheat Engine (hooks and the float live in the game, so the speed stays).
  later calls: write the float directly (what CE's own speed box does).

Cheat Engine is found at %DQN_CE_DIR% or C:/Program Files*/Cheat Engine*.
Don't also turn on the speedhack in the CE window for the same game process
(set --game-speed 0 / GAME_SPEED = 0 if you prefer to do it by hand).
The speed stays until you change it or restart the game.
"""
import ctypes
import ctypes.wintypes as wt
import glob
import os
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LUA_SRC = os.path.join(HERE, "speedhack_dll", "dqn_speedhack.lua")
WINDOW_TITLE = "Dead Cells"
CE_EXE = {True: "cheatengine-x86_64.exe", False: "cheatengine-i386.exe"}
CE_TIMEOUT_S = 90  # first start can be slow (symbols + compiling CE's C hooks)
OLD_DLL_PREFIX = "dqn_speedhack"  # our earlier home-made DLL; can't share a process with CE's hooks

PROCESS_ALL_ACCESS = 0x1F0FFF
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE = 0x1000, 0x2000, 0x8000
PAGE_READWRITE, PAGE_EXECUTE_READWRITE = 0x04, 0x40
LIST_MODULES_32BIT, LIST_MODULES_64BIT = 0x01, 0x02


# --------------------------------------------------------------------------- win32 plumbing
def _k32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wt.HANDLE
    k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k.VirtualAllocEx.restype = ctypes.c_void_p
    k.VirtualAllocEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
    k.VirtualFreeEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
    k.VirtualProtectEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD,
                                   ctypes.POINTER(wt.DWORD)]
    k.WriteProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_size_t)]
    k.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_size_t)]
    k.FlushInstructionCache.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t]
    k.CreateRemoteThread.restype = wt.HANDLE
    k.CreateRemoteThread.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                     ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
    k.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    k.WaitForSingleObject.restype = wt.DWORD
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
    hwnd = win32gui.FindWindow(None, title)  # exact title match
    if not hwnd:
        raise RuntimeError("window '%s' not found - is the game running?" % title)
    return win32process.GetWindowThreadProcessId(hwnd)[1]


class _Proc:
    """Open handle + helpers for one target process."""

    def __init__(self, pid):
        self.pid = pid
        self.k, self.p = _k32(), _psapi()
        self.h = self.k.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not self.h:
            raise ctypes.WinError(ctypes.get_last_error())
        if struct.calcsize("P") == 4:
            self.is32 = True
        else:
            wow = wt.BOOL()
            if not self.k.IsWow64Process(self.h, ctypes.byref(wow)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.is32 = bool(wow.value)
        self.ptr = 4 if self.is32 else 8

    def close(self):
        self.k.CloseHandle(self.h)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def modules(self):
        flag = LIST_MODULES_32BIT if self.is32 else LIST_MODULES_64BIT
        arr = (ctypes.c_void_p * 1024)()
        need = wt.DWORD()
        if not self.p.EnumProcessModulesEx(self.h, arr, ctypes.sizeof(arr), ctypes.byref(need), flag):
            raise ctypes.WinError(ctypes.get_last_error())
        out = {}
        buf = ctypes.create_unicode_buffer(1024)
        for i in range(min(need.value // ctypes.sizeof(ctypes.c_void_p), 1024)):
            if arr[i] and self.p.GetModuleFileNameExW(self.h, arr[i], buf, 1024):
                out[os.path.basename(buf.value).lower()] = arr[i]
        return out

    def read(self, addr, size):
        buf = ctypes.create_string_buffer(size)
        n = ctypes.c_size_t()
        if not self.k.ReadProcessMemory(self.h, ctypes.c_void_p(addr), buf, size, ctypes.byref(n)):
            raise ctypes.WinError(ctypes.get_last_error())
        return buf.raw

    def write(self, addr, data, code=False):
        old = wt.DWORD()
        if code:
            if not self.k.VirtualProtectEx(self.h, ctypes.c_void_p(addr), len(data),
                                           PAGE_EXECUTE_READWRITE, ctypes.byref(old)):
                raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.k.WriteProcessMemory(self.h, ctypes.c_void_p(addr), data, len(data), None):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            if code:
                tmp = wt.DWORD()
                self.k.VirtualProtectEx(self.h, ctypes.c_void_p(addr), len(data), old.value, ctypes.byref(tmp))
                self.k.FlushInstructionCache(self.h, ctypes.c_void_p(addr), len(data))

    def alloc(self, size, prot=PAGE_EXECUTE_READWRITE):
        mem = self.k.VirtualAllocEx(self.h, None, size, MEM_COMMIT | MEM_RESERVE, prot)
        if not mem:
            raise ctypes.WinError(ctypes.get_last_error())
        return mem

    def free(self, mem):
        self.k.VirtualFreeEx(self.h, ctypes.c_void_p(mem), 0, MEM_RELEASE)

    def run_thread(self, start, param=None, timeout_s=10):
        th = self.k.CreateRemoteThread(self.h, None, 0, ctypes.c_void_p(start),
                                       ctypes.c_void_p(param) if param else None, 0, None)
        if not th:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if self.k.WaitForSingleObject(th, int(timeout_s * 1000)) != 0:
                raise RuntimeError("remote thread did not finish in %ss" % timeout_s)
        finally:
            self.k.CloseHandle(th)

    def allocation_base(self, addr):
        """AllocationBase of the region containing addr (None if the query fails)."""
        class MBI(ctypes.Structure):
            _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                        ("AllocationProtect", wt.DWORD), ("PartitionId", wt.WORD),
                        ("RegionSize", ctypes.c_size_t), ("State", wt.DWORD),
                        ("Protect", wt.DWORD), ("Type", wt.DWORD)]
        mbi = MBI()
        self.k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
        self.k.VirtualQueryEx.restype = ctypes.c_size_t
        if not self.k.VirtualQueryEx(self.h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            return None
        return mbi.AllocationBase or 0

    def read_ptr(self, addr):
        return struct.unpack("<I" if self.is32 else "<Q", self.read(addr, self.ptr))[0]

    def export(self, base, name, allow_forward=False):
        """Address of an exported symbol inside the game (reads the PE export table
        of the loaded module, so it works for a 32-bit game from 64-bit Python).
        Returns None if the export is a forwarder (and allow_forward is False)."""
        e_lfanew = struct.unpack_from("<I", self.read(base, 64), 0x3C)[0]
        nt = self.read(base + e_lfanew, 0x120)
        magic = struct.unpack_from("<H", nt, 0x18)[0]
        dd = 0x18 + (96 if magic == 0x10B else 112)
        exp_rva, exp_size = struct.unpack_from("<II", nt, dd)
        blob = self.read(base + exp_rva, exp_size)
        n_names, rva_funcs, rva_names, rva_ords = struct.unpack_from("<IIII", blob, 24)
        get = lambda rva, fmt: struct.unpack_from(fmt, blob, rva - exp_rva)
        names = get(rva_names, "<%dI" % n_names)
        ords = get(rva_ords, "<%dH" % n_names)
        want = name.lower().encode()
        for i, r in enumerate(names):
            off = r - exp_rva
            end = blob.index(b"\0", off)
            if blob[off:end].lower() == want:  # CE's symbol lookup is case-insensitive too
                frva = get(rva_funcs + 4 * ords[i], "<I")[0]
                if exp_rva <= frva < exp_rva + exp_size and not allow_forward:
                    return None
                return base + frva
        raise KeyError("%s not exported by module at 0x%x" % (name, base))


# --------------------------------------------------------------------------- Cheat Engine
def find_ce_dir():
    cands = []
    if os.environ.get("DQN_CE_DIR"):
        cands.append(os.environ["DQN_CE_DIR"])
    for pf in (os.environ.get("ProgramFiles", r"C:\Program Files"),
               os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
               r"C:\Program Files", r"C:\Program Files (x86)"):
        cands += sorted(glob.glob(os.path.join(pf, "Cheat Engine*")), reverse=True)
    for c in cands:
        if os.path.isfile(os.path.join(c, CE_EXE[struct.calcsize("P") == 8])) and \
                os.path.isfile(os.path.join(c, "autorun", "SpeedhackV3.lua")):
            return os.path.abspath(c)
    raise RuntimeError("Cheat Engine 7.x (with autorun/SpeedhackV3.lua) not found; "
                       "set DQN_CE_DIR to its folder")


def _install_lua(ce_dir):
    dst = os.path.join(ce_dir, "autorun", "dqn_speedhack.lua")
    src = open(LUA_SRC, "rb").read()
    try:
        if os.path.isfile(dst) and open(dst, "rb").read() == src:
            return
        with open(dst, "wb") as f:
            f.write(src)
    except OSError as e:
        raise RuntimeError("could not copy %s to %s (%s). Copy it there by hand once (as administrator)."
                           % (LUA_SRC, dst, e))


def _state_file(pid):
    return os.path.join(tempfile.gettempdir(), "dqn_speedhack_%d.txt" % pid)


def _known_addr(pr):
    """Address of speedhack_wantedspeed in this game process, if we set it up earlier."""
    try:
        addr = int(open(_state_file(pr.pid)).read().strip(), 16)
        v = struct.unpack("<f", pr.read(addr, 4))[0]
        return addr if 0.01 <= v <= 100 else None
    except (OSError, ValueError, struct.error):
        return None


def _already_hooked(pr):
    """True if RtlQueryPerformanceCounter already starts with a jmp (CE window, old DLL...)."""
    mods = pr.modules()
    if any(n.startswith(OLD_DLL_PREFIX) for n in mods):
        return True
    try:
        qpc = pr.export(mods["ntdll.dll"], "RtlQueryPerformanceCounter")
        return qpc is not None and pr.read(qpc, 1) == b"\xE9"
    except Exception:
        return False


def _hook_target(pr, func):
    """Where a hooked function jumps to (None if it doesn't start with jmp rel32)."""
    if func is None:
        return None
    b = pr.read(func, 5)
    if b[0] != 0xE9:
        return None
    return (func + 5 + struct.unpack("<i", b[1:5])[0]) & 0xFFFFFFFF


def _abs_refs(pr, start, size=768):
    """Every 4-byte value inside the first `size` bytes of compiled hook code that
    could be an absolute address (32-bit code addresses globals with imm32/disp32)."""
    code = b""
    for off in range(0, size, 256):  # stop at the first unreadable page
        try:
            code += pr.read(start + off, 256)
        except OSError:
            break
    return {struct.unpack_from("<I", code, i)[0] for i in range(len(code) - 3)}


def find_wanted_speed(pr, verbose=False):
    """Find CE SpeedhackV3's `speedhack_wantedspeed` in an already sped-up game, without
    any saved state: it is the one global that both CE hooks read (new_RtlQueryPerformanceCounter
    and new_gettickcount), and it holds the current speed. Returns the address or None."""
    if not pr.is32:
        return None
    mods = pr.modules()

    def ex(mod, name):
        try:
            return pr.export(mods[mod], name) if mod in mods else None
        except KeyError:
            return None

    qpc_hook = _hook_target(pr, ex("ntdll.dll", "RtlQueryPerformanceCounter"))
    gtc_hook = None
    for mod, name in (("kernel32.dll", "GetTickCount64"), ("kernelbase.dll", "GetTickCount64"),
                      ("kernel32.dll", "GetTickCount")):
        gtc_hook = _hook_target(pr, ex(mod, name))
        if gtc_hook:
            break
    if not (qpc_hook and gtc_hook):
        return None
    found = []
    for a in sorted(_abs_refs(pr, qpc_hook) & _abs_refs(pr, gtc_hook)):
        # CE's globalalloc(speedhack_wantedspeed,4) is the first variable of its own
        # VirtualAllocEx block, so it sits at the start of an allocation (64K-aligned).
        # This rules out 4-byte windows taken from the middle of instructions.
        if a < 0x10000 or a % 0x1000:
            continue
        try:
            v = struct.unpack("<f", pr.read(a, 4))[0]
        except OSError:
            continue
        if not 0.01 <= v <= 100:
            continue
        base = pr.allocation_base(a)
        if base is not None and base != a:
            continue
        found.append(a)
    if verbose:
        print("speedhack: candidates after filtering: %s" % (", ".join("0x%X" % a for a in found) or "none"))
    if len(found) == 1:
        return found[0]
    if found:
        print("speedhack: several candidates for speedhack_wantedspeed: %s" % ", ".join("0x%X" % a for a in found))
    return None


def _save_addr(pid, addr):
    with open(_state_file(pid), "w") as f:
        f.write("%X" % addr)


def _focus_game():
    try:
        import win32gui
        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
    except Exception:
        print("speedhack: click the Dead Cells window to give it focus again")


def _start_with_ce(pr, speed):
    ce_dir = find_ce_dir()
    _install_lua(ce_dir)
    out = os.path.join(tempfile.gettempdir(), "dqn_speedhack_out_%d.txt" % pr.pid)
    for f in (out, out + ".tmp"):
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ, DQN_SPEEDHACK_PID=str(pr.pid), DQN_SPEEDHACK_SPEED=repr(float(speed)),
               DQN_SPEEDHACK_OUT=out)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 7  # SW_SHOWMINNOACTIVE: don't steal focus from the game
    exe = os.path.join(ce_dir, CE_EXE[struct.calcsize("P") == 8])
    print("speedhack: starting Cheat Engine to turn on its speedhack ...")
    ce = subprocess.Popen([exe], cwd=ce_dir, env=env, startupinfo=si)
    t0 = time.time()
    while not os.path.exists(out):
        if ce.poll() is not None:
            raise RuntimeError("Cheat Engine exited before turning on the speedhack (code %s)" % ce.returncode)
        if time.time() - t0 > CE_TIMEOUT_S:
            raise RuntimeError("Cheat Engine did not report back in %ds; it is left open so you can "
                               "see any error dialog in its window" % CE_TIMEOUT_S)
        time.sleep(0.25)
    lines = open(out).read().splitlines() + ["", "", ""]
    os.remove(out)
    if lines[0] != "ok":
        ce.kill()
        raise RuntimeError("Cheat Engine's speedhack failed: %s" % lines[2])
    addr = int(lines[1], 16)
    # Kill instead of closing normally: on a normal exit CE resets the speed to 1.
    ce.kill()
    ce.wait(10)
    _save_addr(pr.pid, addr)
    print("speedhack: CE's speedhack is on (speedhack_wantedspeed at 0x%X); Cheat Engine closed" % addr)
    _focus_game()
    return addr


# --------------------------------------------------------------------------- public API
def is_injected(pid=None):
    """True if CE's speedhack is already on in this game process (by this module or a CE window)."""
    pid = pid or find_game_pid()
    with _Proc(pid) as pr:
        return _known_addr(pr) is not None or (_already_hooked(pr) and find_wanted_speed(pr) is not None)


def get_speed(pid=None):
    """Current speed factor of the game (1.0 if the speedhack isn't on)."""
    pid = pid or find_game_pid()
    with _Proc(pid) as pr:
        addr = _known_addr(pr) or (find_wanted_speed(pr) if _already_hooked(pr) else None)
        return struct.unpack("<f", pr.read(addr, 4))[0] if addr else 1.0


def set_speed(speed, pid=None):
    """Turn on Cheat Engine's speedhack for the game if needed, then set the speed
    (1.0 = normal). Returns True if Cheat Engine had to be started for it."""
    if not (0.01 <= speed <= 100):
        raise ValueError("speed must be between 0.01 and 100")
    pid = pid or find_game_pid()
    with _Proc(pid) as pr:
        addr = _known_addr(pr)
        started = False
        if addr is None and _already_hooked(pr):
            # sped up before without our saved state (CE window, or %TEMP% was cleared):
            # find CE's speed variable in the game instead of hooking a second time
            addr = find_wanted_speed(pr)
            if addr is None:
                raise RuntimeError("the game's timers are already hooked, but not by Cheat Engine's "
                                   "SpeedhackV3 as far as this script can tell (old home-made DLL?). "
                                   "Restart Dead Cells, or use --game-speed 0 / GAME_SPEED = 0.")
            _save_addr(pid, addr)
            print("speedhack: game was already sped up; reusing CE's speedhack_wantedspeed at 0x%X" % addr)
        if addr is None:
            addr = _start_with_ce(pr, speed)
            started = True
        pr.write(addr, struct.pack("<f", float(speed)))
        got = struct.unpack("<f", pr.read(addr, 4))[0]
        if abs(got - speed) > 1e-4:
            raise RuntimeError("speed write did not stick (read back %.3f)" % got)
    return started


if __name__ == "__main__":
    if sys.argv[1:] == ["--detect"]:  # check the in-game detection without the saved state file
        with _Proc(find_game_pid()) as pr:
            saved = _known_addr(pr)
            print("hooked:", _already_hooked(pr), "| saved address:", saved and "0x%X" % saved)
            a = find_wanted_speed(pr, verbose=True)
            print("found in game:", a and "0x%X = %.2fx" % (a, struct.unpack("<f", pr.read(a, 4))[0]),
                  "| matches saved:", (a == saved) if saved else "n/a")
    elif len(sys.argv) > 1:
        s = float(sys.argv[1])
        started = set_speed(s)
        print("%s; game speed set to %.2fx" % ("speedhack turned on" if started else "speedhack already on", s))
    else:
        pid = find_game_pid()
        print("speedhack on:", is_injected(pid), "| speed: %.2fx" % get_speed(pid))
