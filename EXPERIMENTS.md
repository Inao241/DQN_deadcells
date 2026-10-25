# Ablation experiments (paper)

New files (the original `train.py` / `run.py` are untouched):

| File | What it does |
|---|---|
| `ablation_common.py` | switchable Double DQN / PER / Dueling, logging, damage-event recording |
| `train_ablation.py` | trains ONE config x seed into its own folder |
| `evaluate.py` | greedy evaluation of a trained run; `--random` and `--human` baselines |
| `run_experiments.py` | runs the whole queue unattended (skip finished runs, safe to re-run) |
| `analyze.py` | results table (+ LaTeX), learning curves, bootstrap CIs, Glass's δ, latency, compute cost |
| `analyze_damage.py` | what the agent was doing when hit, which boss attack hit it, cause of death |
| `selftest.py` | checks the whole pipeline with a fake game, no Dead Cells needed |

`model.py`: added `dueling=True/False` (default True, old checkpoints still load).
Extra Python packages for the analysis: `matplotlib` (everything else is already used by the project).

## 0. Self-test (once, ~2 min)
    python selftest.py

## 1. Pilot: measure speed (~20 min)
Game open at 2x speed, standing in front of the boss door, then:

    python train_ablation.py --run-dir runs_pilot/speed --max-steps 3000

Look at `runs_pilot/speed/summary.json` (`wall_time_h`) and `episodes.csv` (`mean_step_ms`).
Hours per run ≈ wall_time_h x (80000 / 3000). Total ≈ that x 12 runs + evaluation.

## 2. Full queue
    python run_experiments.py --plan           # see what will run
    python run_experiments.py --max-steps 80000 --seeds 0 1 2

Order: random baseline, then all configs for seed 0, then seed 1, then seed 2.
Press `q` to stop (the queue stops too); re-run the same command to continue.
A half-finished run is moved to `*_incomplete_<time>` and restarted from scratch
(runs are never resumed mid-way, so every run is a clean experiment).
`runs/queue_log.txt` records everything.

Optional 2x2 Dueling ablation: add `--configs dqn double double_per double_dueling double_per_dueling`.

## 3. Human baseline (optional, ~30 min)
    python evaluate.py --human --run-dir runs/human/me --episodes 20

You play; the script only records HP and hits.

## 4. Analysis
    python analyze.py          # -> analysis/results.md, results_table.tex, fig_*.png
    python analyze_damage.py   # -> analysis/damage_report.md, damage_clusters/*.png

Damage groups ≈ boss attacks (each attack deals a fixed amount). Open
`analysis/damage_clusters/cluster_*.png`, rename the attacks in
`analysis/attack_labels.json`, and run `analyze_damage.py` again.

## What changed vs. the original training code (say this in the paper)
* **Double DQN fix.** The original target picked a' with `net(state)` (current
  state s) instead of `net(next_state)` (s'). Double DQN needs a' = argmax_a Q_online(s', a).
* **Move head target.** At s' the move head is now conditioned on the action the
  policy would take at s' (it was conditioned on the action taken at s).
* Plain DQN, uniform replay and a non-dueling head are now available as switches.
* PER: both heads use the same priority formula (the action head's TD error was
  multiplied by 0.7 before).
* The first frame of an episode is captured after the reset (it was captured before).
* A "win" in the first 3 steps (boss HP misread) is logged as `glitch`, not learned from,
  instead of stopping training.
* Seeds are fixed (network init + exploration); the game itself is not deterministic.
