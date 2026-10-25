/*
 * dqn_speedhack: run a game faster/slower by scaling the clocks it reads
 * (the same idea as Cheat Engine's speedhack). Loaded into the game by
 * speedhack.py. The speed factor lives in a named shared-memory double
 * "Local\DQNSpeedhack_<pid>", so Python can change it at any time.
 *
 * Hooked (through import tables of every loaded module, re-scanned every
 * second for newly loaded modules, plus GetProcAddress lookups):
 *   QueryPerformanceCounter, GetTickCount, GetTickCount64, timeGetTime
 *
 * Build (any OS, see build.sh):
 *   zig cc -target x86-windows-gnu    -shared -O2 -o dqn_speedhack32.dll speedhack.c
 *   zig cc -target x86_64-windows-gnu -shared -O2 -o dqn_speedhack64.dll speedhack.c
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>

typedef BOOL(WINAPI *QPC_t)(LARGE_INTEGER *);
typedef ULONGLONG(WINAPI *GTC64_t)(void);
typedef DWORD(WINAPI *TGT_t)(void);
typedef FARPROC(WINAPI *GPA_t)(HMODULE, LPCSTR);

static QPC_t real_qpc;
static GTC64_t real_gtc64;
static TGT_t real_tgt;
static GPA_t real_gpa;
static HMODULE h_k32, h_kbase, h_winmm, h_self;
static volatile double *shared_speed;
static CRITICAL_SECTION lock;

/* Each clock is scaled independently and stays continuous when the speed changes:
 *   fake(t) = base_fake + (t - base_real) * speed                              */
typedef struct { int64_t base_real, base_fake; double speed; int init; } clock_state;
static clock_state c_qpc, c_gtc, c_gtc64, c_tgt;
static int64_t tgt_ext; static DWORD tgt_last; /* unwrap 32-bit timeGetTime */

static double cur_speed(void) {
    double s = shared_speed ? *shared_speed : 1.0;
    if (!(s >= 0.05 && s <= 20.0)) s = 1.0;
    return s;
}

static int64_t scale(clock_state *c, int64_t real) {
    double s = cur_speed();
    int64_t out;
    EnterCriticalSection(&lock);
    if (!c->init) { c->base_real = real; c->base_fake = real; c->speed = s; c->init = 1; }
    if (s != c->speed) {
        c->base_fake += (int64_t)((double)(real - c->base_real) * c->speed);
        c->base_real = real;
        c->speed = s;
    }
    out = c->base_fake + (int64_t)((double)(real - c->base_real) * c->speed);
    LeaveCriticalSection(&lock);
    return out;
}

static BOOL WINAPI hk_qpc(LARGE_INTEGER *p) {
    LARGE_INTEGER r;
    BOOL ok = real_qpc(&r);
    if (!ok || !p) return ok;
    p->QuadPart = scale(&c_qpc, r.QuadPart);
    return TRUE;
}
static DWORD WINAPI hk_gtc(void) { return (DWORD)scale(&c_gtc, (int64_t)real_gtc64()); }
static ULONGLONG WINAPI hk_gtc64(void) { return (ULONGLONG)scale(&c_gtc64, (int64_t)real_gtc64()); }
static DWORD WINAPI hk_tgt(void) {
    DWORD now = real_tgt();
    int64_t ext;
    EnterCriticalSection(&lock);
    if (tgt_ext == 0) tgt_ext = now;
    else tgt_ext += (DWORD)(now - tgt_last);
    tgt_last = now;
    ext = tgt_ext;
    LeaveCriticalSection(&lock);
    return (DWORD)scale(&c_tgt, ext);
}

static void *hook_for_name(const char *n) {
    if (!lstrcmpA(n, "QueryPerformanceCounter")) return (void *)hk_qpc;
    if (!lstrcmpA(n, "GetTickCount")) return (void *)hk_gtc;
    if (!lstrcmpA(n, "GetTickCount64")) return (void *)hk_gtc64;
    if (!lstrcmpA(n, "timeGetTime")) return (void *)hk_tgt;
    return NULL;
}

static FARPROC WINAPI hk_gpa(HMODULE m, LPCSTR name) {
    FARPROC f = real_gpa(m, name);
    if (f && ((ULONG_PTR)name >> 16) != 0 && m && (m == h_k32 || m == h_kbase || m == h_winmm)) {
        void *h = hook_for_name(name);
        if (h) return (FARPROC)h;
    }
    return f;
}

static int is_time_dll(const char *n) {
    char b[64];
    int i;
    for (i = 0; n[i] && i < 63; i++) b[i] = (n[i] >= 'A' && n[i] <= 'Z') ? n[i] + 32 : n[i];
    b[i] = 0;
    return !lstrcmpA(b, "kernel32.dll") || !lstrcmpA(b, "kernelbase.dll") || !lstrcmpA(b, "winmm.dll") ||
           (b[0] == 'a' && b[1] == 'p' && b[2] == 'i' && b[3] == '-');  /* api-ms-win-core-* sets */
}

static void patch_module(HMODULE mod) {
    BYTE *base = (BYTE *)mod;
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)base;
    IMAGE_NT_HEADERS *nt;
    IMAGE_DATA_DIRECTORY dir;
    IMAGE_IMPORT_DESCRIPTOR *imp;
    if (mod == h_self || dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    nt = (IMAGE_NT_HEADERS *)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;
    dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return;
    for (imp = (IMAGE_IMPORT_DESCRIPTOR *)(base + dir.VirtualAddress); imp->Name; imp++) {
        IMAGE_THUNK_DATA *ft, *ot;
        if (!is_time_dll((const char *)(base + imp->Name)) || !imp->OriginalFirstThunk) continue;
        ft = (IMAGE_THUNK_DATA *)(base + imp->FirstThunk);
        ot = (IMAGE_THUNK_DATA *)(base + imp->OriginalFirstThunk);
        for (; ft->u1.Function && ot->u1.AddressOfData; ft++, ot++) {
            void *repl;
            const char *name;
            if (IMAGE_SNAP_BY_ORDINAL(ot->u1.Ordinal)) continue;
            name = (const char *)((IMAGE_IMPORT_BY_NAME *)(base + ot->u1.AddressOfData))->Name;
            repl = hook_for_name(name);
            if (!repl && !lstrcmpA(name, "GetProcAddress")) repl = (void *)hk_gpa;
            if (repl && (void *)ft->u1.Function != repl) {
                DWORD old;
                if (VirtualProtect(&ft->u1.Function, sizeof(void *), PAGE_READWRITE, &old)) {
                    ft->u1.Function = (ULONG_PTR)repl;
                    VirtualProtect(&ft->u1.Function, sizeof(void *), old, &old);
                }
            }
        }
    }
}

static void patch_all(void) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
    MODULEENTRY32W me;
    if (snap == INVALID_HANDLE_VALUE) return;
    me.dwSize = sizeof(me);
    if (Module32FirstW(snap, &me)) {
        do { patch_module(me.hModule); } while (Module32NextW(snap, &me));
    }
    CloseHandle(snap);
}

static DWORD WINAPI init_thread(LPVOID unused) {
    WCHAR name[64] = L"Local\\DQNSpeedhack_";
    DWORD pid = GetCurrentProcessId(), d = 1000000000;
    int i = lstrlenW(name), started = 0;
    HANDLE map;
    (void)unused;
    while (d) {  /* append pid in decimal */
        int digit = (int)((pid / d) % 10);
        if (digit || started || d == 1) { name[i++] = (WCHAR)(L'0' + digit); started = 1; }
        d /= 10;
    }
    name[i] = 0;

    h_k32 = GetModuleHandleW(L"kernel32.dll");
    h_kbase = GetModuleHandleW(L"kernelbase.dll");
    h_winmm = LoadLibraryW(L"winmm.dll");
    real_qpc = (QPC_t)GetProcAddress(h_k32, "QueryPerformanceCounter");
    real_gtc64 = (GTC64_t)GetProcAddress(h_k32, "GetTickCount64");
    real_gpa = (GPA_t)GetProcAddress(h_k32, "GetProcAddress");
    real_tgt = h_winmm ? (TGT_t)GetProcAddress(h_winmm, "timeGetTime") : NULL;
    if (!real_qpc || !real_gtc64 || !real_gpa || !real_tgt) return 1;

    map = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(double), name);
    if (map) {
        int existed = GetLastError() == ERROR_ALREADY_EXISTS;
        shared_speed = (volatile double *)MapViewOfFile(map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(double));
        if (shared_speed && !existed) *shared_speed = 1.0;
    }
    for (;;) {  /* keep patching modules that get loaded later */
        patch_all();
        Sleep(1000);
    }
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        HANDLE t;
        h_self = (HMODULE)inst;
        DisableThreadLibraryCalls(inst);
        InitializeCriticalSection(&lock);
        t = CreateThread(NULL, 0, init_thread, NULL, 0, NULL);
        if (t) CloseHandle(t);
    }
    return TRUE;
}
