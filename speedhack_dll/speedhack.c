/*
 * dqn_speedhack: make a game run faster/slower by scaling the clocks it reads.
 *
 * Same method as Cheat Engine's speedhack (autorun/SpeedhackV3.lua, x86
 * targets): an inline jmp is written at the start of exactly three functions
 *     ntdll!RtlQueryPerformanceCounter   (QueryPerformanceCounter ends up here)
 *     kernel32!GetTickCount64
 *     kernel32!GetTickCount
 * Nothing else in the process is modified.
 *
 * Unlike CE, the replacements never call the original code, so no original
 * instructions have to be relocated:
 *   - real QPC  = ntdll!NtQueryPerformanceCounter (same time base, kernel side)
 *   - real tick = read from KUSER_SHARED_DATA, exactly like GetTickCount64 does
 *
 * The speed factor is a double in the named shared memory
 * "Local\DQNSpeedhack_<pid>" (written by speedhack.py at any time).
 *
 * Build: see build.sh (zig cc, 32- and 64-bit).
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>

typedef LONG(NTAPI *NtQPC_t)(LARGE_INTEGER *count, LARGE_INTEGER *freq);

static NtQPC_t real_ntqpc;
static volatile double *shared_speed;
static CRITICAL_SECTION lock;
static volatile LONG hooks_installed;

/* fake(t) = base_fake + (t - base_real) * speed, continuous when speed changes */
typedef struct { uint64_t base_real, base_fake; double speed; int init; } clock_state;
static clock_state c_qpc, c_tick;

static double cur_speed(void) {
    double s = shared_speed ? *shared_speed : 1.0;
    if (!(s >= 0.05 && s <= 20.0)) s = 1.0;
    return s;
}

static uint64_t scale(clock_state *c, uint64_t real) {
    double s = cur_speed();
    uint64_t out;
    EnterCriticalSection(&lock);
    if (!c->init) { c->base_real = real; c->base_fake = real; c->speed = s; c->init = 1; }
    if (s != c->speed) {
        c->base_fake += (uint64_t)((double)(real - c->base_real) * c->speed);
        c->base_real = real;
        c->speed = s;
    }
    out = c->base_fake + (uint64_t)((double)(real - c->base_real) * c->speed);
    LeaveCriticalSection(&lock);
    return out;
}

/* ---- real clocks (do not go through the hooked functions) */
static uint64_t real_qpc_value(void) {
    LARGE_INTEGER v;
    real_ntqpc(&v, NULL);
    return (uint64_t)v.QuadPart;
}

static uint64_t real_tick64(void) {
    /* KUSER_SHARED_DATA at 0x7FFE0000: TickCountMultiplier @0x004,
       TickCount (KSYSTEM_TIME) @0x320: LowPart, High1Time @0x324, High2Time @0x328 */
    volatile uint8_t *ud = (volatile uint8_t *)0x7FFE0000;
    uint32_t mult = *(volatile uint32_t *)(ud + 0x004);
    uint32_t low, hi1, hi2;
    for (;;) {
        hi1 = *(volatile int32_t *)(ud + 0x324);
        low = *(volatile uint32_t *)(ud + 0x320);
        hi2 = *(volatile int32_t *)(ud + 0x328);
        if (hi1 == hi2) break;
        YieldProcessor();
    }
    return (((uint64_t)low * mult) >> 24) + (((uint64_t)hi1 * mult) << 8);
}

/* ---- replacements (same signatures/calling convention as the originals) */
static BOOLEAN NTAPI new_RtlQueryPerformanceCounter(LARGE_INTEGER *count) {
    if (count) count->QuadPart = (LONGLONG)scale(&c_qpc, real_qpc_value());
    return TRUE;
}
static ULONGLONG WINAPI new_GetTickCount64(void) { return scale(&c_tick, real_tick64()); }
static DWORD WINAPI new_GetTickCount(void) { return (DWORD)scale(&c_tick, real_tick64()); }

/* ---- inline hook: overwrite the first bytes of target with a jump */
static int write_jmp(void *target, void *dest) {
    uint8_t code[16];
    size_t n;
    DWORD old;
#if defined(_WIN64)
    code[0] = 0x48; code[1] = 0xB8;                     /* mov rax, imm64 */
    *(uint64_t *)(code + 2) = (uint64_t)dest;
    code[10] = 0xFF; code[11] = 0xE0;                   /* jmp rax        */
    n = 12;
#else
    code[0] = 0xE9;                                      /* jmp rel32      */
    *(int32_t *)(code + 1) = (int32_t)((uint8_t *)dest - ((uint8_t *)target + 5));
    n = 5;
#endif
    if (!VirtualProtect(target, n, PAGE_EXECUTE_READWRITE, &old)) return 0;
    /* write everything except the first byte, then the first byte, so a thread
       entering the function never runs a half-written jump opcode */
    memcpy((uint8_t *)target + 1, code + 1, n - 1);
    *(volatile uint8_t *)target = code[0];
    VirtualProtect(target, n, old, &old);
    FlushInstructionCache(GetCurrentProcess(), target, n);
    return 1;
}

static void install_hooks(void) {
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    void *rqpc, *gtc64, *gtc;
    if (InterlockedExchange(&hooks_installed, 1)) return;
    real_ntqpc = (NtQPC_t)GetProcAddress(ntdll, "NtQueryPerformanceCounter");
    rqpc = (void *)GetProcAddress(ntdll, "RtlQueryPerformanceCounter");
    gtc64 = (void *)GetProcAddress(k32, "GetTickCount64");
    gtc = (void *)GetProcAddress(k32, "GetTickCount");
    if (!real_ntqpc) return;
    /* prime the clocks before any game thread can call the replacements */
    scale(&c_qpc, real_qpc_value());
    scale(&c_tick, real_tick64());
    if (rqpc) write_jmp(rqpc, (void *)new_RtlQueryPerformanceCounter);
    if (gtc64) write_jmp(gtc64, (void *)new_GetTickCount64);
    if (gtc && gtc != gtc64) write_jmp(gtc, (void *)new_GetTickCount);
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
    map = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(double), name);
    if (map) {
        int existed = GetLastError() == ERROR_ALREADY_EXISTS;
        shared_speed = (volatile double *)MapViewOfFile(map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(double));
        if (shared_speed && !existed) *shared_speed = 1.0;
    }
    install_hooks();
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        HANDLE t;
        DisableThreadLibraryCalls(inst);
        InitializeCriticalSection(&lock);
        /* hooks are installed outside the loader lock */
        t = CreateThread(NULL, 0, init_thread, NULL, 0, NULL);
        if (t) CloseHandle(t);
    }
    return TRUE;
}
