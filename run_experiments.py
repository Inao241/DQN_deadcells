"""Run the whole ablation unattended: every config x seed, train then evaluate.

    python run_experiments.py --plan                      # just print the queue
    python run_experiments.py --max-steps 80000 --seeds 0 1 2
    python run_experiments.py --configs dqn double --seeds 0

Order is seed-major (all configs for seed 0, then seed 1, ...), so if you run
out of time you still have a complete comparison for the seeds that finished.
Finished runs (DONE + EVAL_DONE) are skipped, so just re-run after a crash.
The queue stops if a run is quit with q or hits a game error.
"""
import argparse
import os
import subprocess
import sys
import time

CONFIGS = {
    # name:                (double, per, dueling)
    "dqn":                 (0, 0, 0),
    "double":              (1, 0, 0),
    "double_per":          (1, 1, 0),
    "double_dueling":      (1, 0, 1),
    "double_per_dueling":  (1, 1, 1),
}
DEFAULT_CONFIGS = ["dqn", "double", "double_per", "double_per_dueling"]


def log(msg, root):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line)
    with open(os.path.join(root, "queue_log.txt"), "a") as f:
        f.write(line + "\n")


def run(cmd, root):
    log("RUN " + " ".join(cmd), root)
    t0 = time.time()
    code = subprocess.call(cmd)
    log("exit %d after %.1f min" % (code, (time.time() - t0) / 60), root)
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-steps", type=int, default=80000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--random-baseline", type=int, default=1, help="also evaluate a random agent once (1/0)")
    p.add_argument("--game-speed", type=float, default=2.0, help="0 = don't touch (e.g. using Cheat Engine)")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--plan", action="store_true", help="print the queue and exit")
    p.add_argument("--expect-player-hp", type=int, default=0,
                   help="player max HP every episode must start with, e.g. 15482 (0 = each run checks against its own first episode)")
    p.add_argument("--extra", default="", help='extra args passed to train_ablation.py, e.g. "--damage-frame-prob 0.05"')
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to this folder
    os.makedirs(args.runs_root, exist_ok=True)
    py = sys.executable

    queue = []
    if args.random_baseline:
        queue.append(("random", None, 0))
    for seed in args.seeds:
        for name in args.configs:
            queue.append((name, CONFIGS[name], seed))

    if args.plan:
        for name, flags, seed in queue:
            d = os.path.join(args.runs_root, name, "seed%d" % seed)
            done = os.path.exists(os.path.join(d, "EVAL_DONE"))
            print("%-22s seed %d  %s" % (name, seed, "done" if done else "todo"))
        return 0

    for name, flags, seed in queue:
        d = os.path.join(args.runs_root, name, "seed%d" % seed)
        if name == "random":
            if not os.path.exists(os.path.join(d, "EVAL_DONE")):
                code = run([py, "evaluate.py", "--random", "--run-dir", d, "--seed", str(seed),
                            "--episodes", str(args.eval_episodes), "--game-speed", str(args.game_speed),
                            "--expect-player-hp", str(args.expect_player_hp)], args.runs_root)
                if code != 0:
                    log("stopping queue", args.runs_root)
                    return code
            continue
        if not os.path.exists(os.path.join(d, "DONE")):
            double, per, dueling = flags
            cmd = [py, "train_ablation.py", "--run-dir", d, "--seed", str(seed),
                   "--double", str(double), "--per", str(per), "--dueling", str(dueling),
                   "--max-steps", str(args.max_steps), "--game-speed", str(args.game_speed),
                   "--expect-player-hp", str(args.expect_player_hp)] + args.extra.split()
            code = run(cmd, args.runs_root)
            if code != 0:
                log("stopping queue (fix the game, then re-run the same command)", args.runs_root)
                return code
        if not os.path.exists(os.path.join(d, "EVAL_DONE")):
            code = run([py, "evaluate.py", "--run-dir", d, "--episodes", str(args.eval_episodes),
                        "--game-speed", str(args.game_speed), "--expect-player-hp", str(args.expect_player_hp)],
                       args.runs_root)
            if code != 0:
                log("stopping queue", args.runs_root)
                return code
    log("ALL DONE - now run: python analyze.py && python analyze_damage.py", args.runs_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
