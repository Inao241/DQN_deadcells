"""Self-test: checks the whole pipeline WITHOUT the game (the game, screen
capture, keyboard and HP reading are replaced by a tiny fake boss fight).

    python selftest.py

Run it once before starting real experiments. It trains every config for a
few hundred steps on your GPU, evaluates, runs both analysis scripts, and
prints PASS / FAIL. Output goes to selftest_runs/ and selftest_analysis/
(delete them afterwards).
"""
import math
import os
import random
import subprocess
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOSS_MAX, PLAYER_MAX = 234930, 15482
ATTACKS = [1200, 2600, 4100]  # fake boss attacks, used to check the damage clustering


# ------------------------------------------------------------------ fake game
class FakeGame:
    def __init__(self):
        self.rng = random.Random(0)
        self.reset()

    def reset(self):
        self.php, self.bhp = PLAYER_MAX, BOSS_MAX
        self.kill_pending = False
        self.glitch_reads = 1 if self.rng.random() < 0.08 else 0  # boss HP misread right after reset

    def boss_hp(self):
        if self.glitch_reads > 0:
            return 1
        return self.bhp

    def step(self, move, action):
        if action == 0 and self.rng.random() < 0.5:
            self.bhp -= self.rng.randint(3000, 6000)
        if self.rng.random() < 0.08:
            if not (action in (1, 2) and self.rng.random() < 0.5):
                self.php -= self.rng.choice(ATTACKS)


GAME = FakeGame()


def install_fakes():
    lib = types.ModuleType("lib")

    GetScreen = types.ModuleType("lib.GetScreen")
    class _Screen:
        def grab(self):
            return torch.rand(4, 128, 128, device=DEVICE)
    GetScreen.GetScreen = _Screen

    Actions = types.ModuleType("lib.Actions")
    for n in ("Nothing", "Move_Left", "Move_Right", "Attack", "Jump", "Roll", "Shield"):
        setattr(Actions, n, lambda: None)

    GetHp = types.ModuleType("lib.GetHp")
    class _Hp:
        def get_self_hp(self):
            if GAME.kill_pending:
                GAME.kill_pending, GAME.php = False, 0
            return GAME.php
        def get_boss_hp(self):
            return GAME.boss_hp()
        def set_self_hp(self, v):
            GAME.php, GAME.kill_pending = v, True
        def set_boss_hp(self, v):
            GAME.bhp = v
    GetHp.Hp_getter = _Hp

    env = types.ModuleType("lib.env")
    class _Env:
        def _reset(self):
            GAME.reset()
        def step(self, move, action, pre_p, pre_b):
            GAME.step(move, action)
            p, b = GAME.php, GAME.boss_hp()
            if GAME.glitch_reads > 0:  # the misread lasts for exactly one step
                GAME.glitch_reads -= 1
            if p > 1 and b <= 1:
                return 100, True, p, b, 0, 0
            if p <= 1:
                return -10, True, p, b, 0, 0
            r = 3 * min((pre_b - b) / 2000, 1) - 5 * min((pre_p - p) / 1000, 1) - 0.01
            return max(min(r, 5), -5), False, p, b, pre_p - p, pre_b - b
    env.env = _Env

    for name, mod in (("GetScreen", GetScreen), ("Actions", Actions), ("GetHp", GetHp), ("env", env)):
        setattr(lib, name, mod)
        sys.modules["lib." + name] = mod
    sys.modules["lib"] = lib

    kb = types.ModuleType("keyboard")
    kb.is_pressed = lambda k: False
    sys.modules["keyboard"] = kb
    pg = types.ModuleType("pygetwindow")
    pg.getWindowsWithTitle = lambda t: [types.SimpleNamespace(size=None)]
    sys.modules["pygetwindow"] = pg
    sh = types.ModuleType("speedhack")
    sh.calls = []
    sh.set_speed = lambda v, pid=None: sh.calls.append(v)
    sys.modules["speedhack"] = sh
    time.sleep = lambda s: None  # the real reset sequence sleeps ~20 s


# ------------------------------------------------------------------ checks
FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    install_fakes()
    import ablation_common as C
    import evaluate
    import train_ablation
    from model import DQN_action, DQN_move
    from run_experiments import CONFIGS
    from analyze import read_csv

    print("device:", DEVICE, "| torch", torch.__version__)

    print("\n[1] old checkpoints still load into the (default dueling) model")
    for cls, n, f in ((DQN_action, 5, "checkpoints/best_action_model.pt"), (DQN_move, 3, "checkpoints/best_move_model.pt")):
        if os.path.exists(f):
            try:
                cls(n).load_state_dict(torch.load(f, map_location="cpu"))
                check(True, f)
            except Exception as e:
                check(False, "%s: %s" % (f, e))
        else:
            print("  skip  ", f, "not found")

    print("\n[2] Double DQN target uses s' (not s) and the plain-DQN target is a max")
    cfg = dict(dueling=1, double=1, lr=1e-4, gamma=0.9, batch_size=4, tau=0.5)
    L = C.Learner(cfg, DEVICE)
    bt = dict(state=torch.zeros(4, 4, 128, 128, device=DEVICE), new_state=torch.ones(4, 4, 128, 128, device=DEVICE),
              boss=torch.full((4, 1), .5, device=DEVICE), new_boss=torch.full((4, 1), .4, device=DEVICE),
              player=torch.full((4, 1), .5, device=DEVICE), new_player=torch.full((4, 1), .4, device=DEVICE),
              action=torch.zeros(4, dtype=torch.int64, device=DEVICE), move=torch.zeros(4, dtype=torch.int64, device=DEVICE),
              reward=torch.ones(4, device=DEVICE), done=torch.zeros(4, device=DEVICE))
    w = torch.ones(4, device=DEVICE)
    with torch.no_grad():
        a_next = L.action_net(bt["new_state"], bt["new_boss"], bt["new_player"]).argmax(1)
        exp_y = 1 + 0.9 * L.action_tgt(bt["new_state"], bt["new_boss"], bt["new_player"]).gather(1, a_next[:, None]).squeeze(1)
        q = L.action_net(bt["state"], bt["boss"], bt["player"])[:, 0]
    loss, td = L._action_loss(bt, w)
    check(np.allclose(td, (exp_y - q).abs().cpu().numpy(), atol=1e-4), "double target = r + γ Q_tgt(s', argmax_a Q(s', a))")
    L.cfg["double"] = 0
    with torch.no_grad():
        exp_y = 1 + 0.9 * L.action_tgt(bt["new_state"], bt["new_boss"], bt["new_player"]).max(1).values
    loss, td = L._action_loss(bt, w)
    check(np.allclose(td, (exp_y - q).abs().cpu().numpy(), atol=1e-4), "vanilla target = r + γ max_a Q_tgt(s', a)")
    bt["done"] = torch.ones(4, device=DEVICE)
    loss, td = L._action_loss(bt, w)
    check(np.allclose(td, (1 - q).abs().cpu().numpy(), atol=1e-4), "terminal transitions do not bootstrap")
    nd = C.Learner(dict(cfg, dueling=0), DEVICE)
    check(not hasattr(nd.action_net, "v") and hasattr(nd.action_net, "q"), "dueling=0 builds a plain Q head")

    print("\n[3] train every config for a few hundred steps")
    root = os.path.join(HERE, "selftest_runs")
    t_cfg = {}
    for name, (double, per, dueling) in CONFIGS.items():
        d = os.path.join(root, name, "seed0")
        t0 = time.time()
        code = train_ablation.main(
            ["--run-dir", d, "--double", str(double), "--per", str(per), "--dueling", str(dueling), "--seed", "0",
             "--max-steps", "400", "--batch-size", "32", "--capacity", "400", "--max-episode-steps", "120",
             "--damage-frame-prob", "0.1", "--no-resize-window", "--device", DEVICE, "--save-every", "2"],
            check_keys=lambda: False)
        t_cfg[name] = time.time() - t0
        rows = read_csv(os.path.join(d, "episodes.csv"))
        losses = [r["mean_loss_action"] for r in rows if isinstance(r.get("mean_loss_action"), float)]
        check(code == 0 and os.path.exists(os.path.join(d, "DONE")) and os.path.exists(os.path.join(d, "final.pt")),
              "%s: finished (%d episodes, %.0fs)" % (name, len(rows), t_cfg[name]))
        check(len(losses) > 0 and all(math.isfinite(x) for x in losses), "%s: losses finite" % name)
        check(os.path.getsize(os.path.join(d, "damage_events.csv")) > 200, "%s: hits logged" % name)
    outcomes = {r["outcome"] for n in CONFIGS for r in read_csv(os.path.join(root, n, "seed0", "episodes.csv"))}
    check("loss" in outcomes, "losses recorded (%s)" % sorted(outcomes))
    check("glitch" in outcomes, "glitched episodes are detected and skipped")

    code = train_ablation.main(["--run-dir", os.path.join(root, "dqn", "seed0"), "--no-resize-window"], check_keys=lambda: False)
    check(code == 0, "re-running a DONE run is skipped")
    partial = os.path.join(root, "partial", "seed0")
    os.makedirs(partial, exist_ok=True)
    open(os.path.join(partial, "episodes.csv"), "w").write("junk\n")
    train_ablation.main(["--run-dir", partial, "--max-steps", "50", "--no-resize-window", "--device", DEVICE,
                         "--capacity", "100", "--batch-size", "8"], check_keys=lambda: False)
    moved = [x for x in os.listdir(os.path.join(root, "partial")) if "_incomplete_" in x]
    check(len(moved) == 1, "an unfinished run folder is moved aside, not mixed")
    check(sys.modules["speedhack"].calls and set(sys.modules["speedhack"].calls) == {2.0},
          "training sets the game speed to 2x through speedhack.py")
    import json as _json
    check(_json.load(open(os.path.join(root, "dqn", "seed0", "config.json")))["game_speed"] == 2.0,
          "game speed recorded in config.json")

    print("\n[4] evaluate every config + random baseline")
    for name in CONFIGS:
        d = os.path.join(root, name, "seed0")
        code = evaluate.main(["--run-dir", d, "--episodes", "4", "--no-resize-window", "--device", DEVICE],
                             check_keys=lambda: False)
        check(code == 0 and len(read_csv(os.path.join(d, "eval.csv"))) == 4, "%s: 4 eval episodes" % name)
    d = os.path.join(root, "random", "seed0")
    code = evaluate.main(["--random", "--run-dir", d, "--episodes", "4", "--no-resize-window"], check_keys=lambda: False)
    check(code == 0 and os.path.exists(os.path.join(d, "EVAL_DONE")), "random baseline")
    ev_out = {r["outcome"] for n in CONFIGS for r in read_csv(os.path.join(root, n, "seed0", "eval.csv"))}
    check("win" in ev_out and "loss" in ev_out, "wins and losses both handled (%s)" % sorted(ev_out))
    frames = os.listdir(os.path.join(HERE, "selftest_runs", "double_per_dueling", "seed0", "damage_frames"))
    check(any(f.startswith("eval_") for f in frames), "hit frames saved during eval (%d files)" % len(frames))

    print("\n[5] analysis scripts")
    for script in ("analyze.py", "analyze_damage.py"):
        r = subprocess.run([sys.executable, script, "--runs-root", "selftest_runs", "--out", "selftest_analysis"],
                           capture_output=True, text=True)
        check(r.returncode == 0, script + ("" if r.returncode == 0 else "\n" + r.stderr[-1500:]))
    rep = open(os.path.join("selftest_analysis", "damage_report.md"), encoding="utf-8").read()
    found = [a for a in ATTACKS if ("~%d dmg" % a) in rep]
    check(len(found) == 3, "damage grouping recovers the 3 fake boss attacks %s" % ATTACKS)

    print("\n[6] speed on this machine (no game; real runs are slower because of key presses/screen capture)")
    L = C.Learner(dict(dueling=1, double=1, lr=1e-4, gamma=0.99, batch_size=64, tau=1e-3), DEVICE)
    buf = C.ReplayBuffer(500, True, 0.6, 0.01, DEVICE)
    s = torch.rand(4, 128, 128, device=DEVICE)
    for _ in range(500):
        buf.append(C.Experience(s, 0, 0, 0.0, False, s, .5, .5, .5, .5))
    t0 = time.perf_counter()
    for _ in range(20):
        L.act(s, .5, .5)
    t_act = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    for _ in range(10):
        L.optimize(buf, 0.5)
    t_opt = (time.perf_counter() - t0) / 10 * 1000
    print("  action selection %.1f ms, one training update (2 x batch 64) %.1f ms" % (t_act, t_opt))

    print("\n" + ("PASS - pipeline works. Delete selftest_runs/ and selftest_analysis/ when you like."
                  if not FAILS else "FAIL (%d):\n  - " % len(FAILS) + "\n  - ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
