"""
probe_hp_chains.py — 同时读取多条 CE pointer chain，筛出稳定的 HP 指针。

只用标准库 + Win32 API，无第三方依赖。

用法（Dead Cells 运行中）：
    python probe_hp_chains.py           # 只在读数变化时打印一行
    python probe_hp_chains.py --dump    # 额外 dump entity 在 +0x100..+0x180 附近的 int32（用来找 maxLife）

测试流程：掉血 → 喝血瓶 → 进 boss 房 → 死亡开新局 → 重启游戏再运行一次。
每一步都读对的链才是稳定的。

CE 的 chain 语义：
    p = [module + base]
    对除最后一个以外的每个 offset：p = [p + off]
    最终地址 = p + 最后一个 offset
Dead Cells 是 32 位进程，所以指针按 4 字节读。
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time

PROCESS_NAME = "deadcells.exe"
MODULE_NAME = "libhl.dll"

# (label, base_offset, [offsets...])  —— 来自 CE pointer scan 截图
CANDIDATES = [
    # A 组：6 层，... → 1B0 → 1C → 12C
    ("A1 6C0/0/18/1B0/1C", 0x49184, [0x6C0, 0x0, 0x18, 0x1B0, 0x1C, 0x12C]),
    ("A2 6BC/4/18/1B0/1C", 0x49184, [0x6BC, 0x4, 0x18, 0x1B0, 0x1C, 0x12C]),
    ("A3 6B8/8/18/1B0/1C", 0x49184, [0x6B8, 0x8, 0x18, 0x1B0, 0x1C, 0x12C]),
    ("A4 4B0/0/58/1B0/1C", 0x49184, [0x4B0, 0x0, 0x58, 0x1B0, 0x1C, 0x12C]),
    ("A5 4AC/4/58/1B0/1C", 0x49184, [0x4AC, 0x4, 0x58, 0x1B0, 0x1C, 0x12C]),
    ("A6 4A8/8/58/1B0/1C", 0x49184, [0x4A8, 0x8, 0x58, 0x1B0, 0x1C, 0x12C]),
    # B 组：5 层，... → 68 → 12C（首选候选）
    ("B1 6C0/0/18/68",     0x49184, [0x6C0, 0x0, 0x18, 0x68, 0x12C]),
    ("B2 6BC/4/18/68",     0x49184, [0x6BC, 0x4, 0x18, 0x68, 0x12C]),
    ("B3 6B8/8/18/68",     0x49184, [0x6B8, 0x8, 0x18, 0x68, 0x12C]),
    ("B4 4B0/0/58/68",     0x49184, [0x4B0, 0x0, 0x58, 0x68, 0x12C]),
    ("B5 4AC/4/58/68",     0x49184, [0x4AC, 0x4, 0x58, 0x68, 0x12C]),
    ("B6 4A8/8/58/68",     0x49184, [0x4A8, 0x8, 0x58, 0x68, 0x12C]),
    # C 组：6 层，... → 114 → 80 → 12C
    ("C1 6B8/0/18/114/80", 0x49184, [0x6B8, 0x0, 0x18, 0x114, 0x80, 0x12C]),
    ("C2 6B4/4/18/114/80", 0x49184, [0x6B4, 0x4, 0x18, 0x114, 0x80, 0x12C]),
    ("C3 6B0/8/18/114/80", 0x49184, [0x6B0, 0x8, 0x18, 0x114, 0x80, 0x12C]),
    # Boss：GetHp.py 旧链的 offsets，只把 base 从 48184 改成 49184（进 boss 房后看）
    ("BOSS 6B8/0/18/11C",  0x49184, [0x6B8, 0x0, 0x18, 0x11C, 0x12C]),
]

HP_FIELD = 0x12C  # Entity 里 HP 字段的 offset（用于 --dump 时反推 entity 基址）

# ---------------- Win32 ----------------
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
LIST_MODULES_ALL = 0x03

k32.OpenProcess.restype = wt.HANDLE
k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
k32.ReadProcessMemory.restype = wt.BOOL
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k32.CloseHandle.argtypes = [wt.HANDLE]
psapi.EnumProcessModulesEx.restype = wt.BOOL
psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_void_p),
                                       wt.DWORD, ctypes.POINTER(wt.DWORD), wt.DWORD]
psapi.GetModuleBaseNameW.restype = wt.DWORD
psapi.GetModuleBaseNameW.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.LPWSTR, wt.DWORD]


TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = wt.HANDLE(-1).value


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("cntUsage", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wt.DWORD),
        ("cntThreads", wt.DWORD),
        ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wt.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
k32.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
k32.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]


def find_pid(name):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == INVALID_HANDLE_VALUE:
        return None
    try:
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            if e.szExeFile.lower() == name.lower():
                return e.th32ProcessID
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    return None


def module_base(h, name):
    arr = (ctypes.c_void_p * 1024)()
    needed = wt.DWORD()
    if not psapi.EnumProcessModulesEx(h, arr, ctypes.sizeof(arr), ctypes.byref(needed), LIST_MODULES_ALL):
        raise OSError(f"EnumProcessModulesEx failed: {ctypes.get_last_error()}")
    n = needed.value // ctypes.sizeof(ctypes.c_void_p)
    buf = ctypes.create_unicode_buffer(260)
    for i in range(min(n, 1024)):
        if arr[i] and psapi.GetModuleBaseNameW(h, arr[i], buf, 260):
            if buf.value.lower() == name.lower():
                return arr[i]
    return None


def read(h, addr, ctype):
    v = ctype()
    got = ctypes.c_size_t()
    ok = k32.ReadProcessMemory(h, ctypes.c_void_p(addr), ctypes.byref(v), ctypes.sizeof(v), ctypes.byref(got))
    return v.value if ok and got.value == ctypes.sizeof(v) else None


def read_ptr32(h, addr):
    return read(h, addr, ctypes.c_uint32)


def resolve(h, mod, base_off, offsets):
    p = read_ptr32(h, mod + base_off)
    for off in offsets[:-1]:
        if not p:
            return None
        p = read_ptr32(h, p + off)
    if not p:
        return None
    return p + offsets[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="dump entity +0x100..+0x180 int32 一次后继续监控")
    ap.add_argument("--interval", type=float, default=0.2)
    args = ap.parse_args()

    pid = find_pid(PROCESS_NAME)
    if pid is None:
        sys.exit(f"找不到 {PROCESS_NAME}，先开游戏。")
    h = k32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    if not h:
        sys.exit(f"OpenProcess 失败 ({ctypes.get_last_error()})，试试用管理员权限运行。")
    mod = module_base(h, MODULE_NAME)
    if mod is None:
        sys.exit(f"进程里找不到 {MODULE_NAME}")
    print(f"pid={pid}  {MODULE_NAME} base=0x{mod:08X}")

    for label, base_off, offs in CANDIDATES:
        addr = resolve(h, mod, base_off, offs)
        val = read(h, addr, ctypes.c_int32) if addr else None
        print(f"  {label:<22} addr={'0x%08X' % addr if addr else '--':<12} value={val}")

    if args.dump:
        # 满血时运行：在 HP 字段附近找一个和满血值相等的字段（maxLife）
        for want in ("B4", "BOSS"):
            for label, base_off, offs in CANDIDATES:
                if label.split()[0] != want:
                    continue
                addr = resolve(h, mod, base_off, offs)
                if not addr:
                    print(f"\n[dump] {label}: 解析失败（boss 链要在 boss 房里才有效）")
                    break
                ent = addr - HP_FIELD
                hp = read(h, addr, ctypes.c_int32)
                print(f"\n[dump] {label}  entity=0x{ent:08X}  HP={hp}")
                for o in range(0x100, 0x184, 4):
                    v = read(h, ent + o, ctypes.c_int32)
                    f = read(h, ent + o, ctypes.c_float)
                    mark = "  <-- HP" if o == HP_FIELD else ("  <-- 等于 HP" if v == hp else "")
                    print(f"  +0x{o:03X}: int={v!s:>12}  float={f if f is None else round(f, 3)!s:>14}{mark}")
                break

    print("\n监控中（只在变化时打印；Ctrl+C 退出）。列顺序：")
    print("  " + " | ".join(c[0].split()[0] for c in CANDIDATES))
    last = None
    try:
        while True:
            vals = []
            for _, base_off, offs in CANDIDATES:
                addr = resolve(h, mod, base_off, offs)
                v = read(h, addr, ctypes.c_int32) if addr else None
                vals.append("--" if v is None else str(v))
            if vals != last:
                print(time.strftime("%H:%M:%S ") + " | ".join(f"{v:>6}" for v in vals))
                last = vals
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        k32.CloseHandle(h)


if __name__ == "__main__":
    main()
