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
| `speedhack.py` | runs the game at 2x with Cheat Engine 7.7's own speedhack, no clicking in CE (see below) |

`model.py`: added `dueling=True/False` (default True, old checkpoints still load).
Extra Python packages for the analysis: `matplotlib` (everything else is already used by the project).

## 0. Self-test (once, ~2 min)
    python selftest.py

## Game speed (Cheat Engine's own speedhack, started from Python)
Training/evaluation set the game to 2x automatically (`--game-speed 2`, recorded in
`config.json`); `train.py` / `run.py` do the same (`GAME_SPEED = 2.0` near the bottom). By hand:

    python speedhack.py 2     # 2x
    python speedhack.py 1     # normal
    python speedhack.py       # show status

Cheat Engine 7.7 has no speedhack DLL any more; its speedhack is `autorun/SpeedhackV3.lua`.
The first time per game launch, `speedhack.py` starts Cheat Engine minimized; the helper
`speedhack_dll/dqn_speedhack.lua` (copied into CE's `autorun/`, inactive for normal CE use) opens
the game, calls CE's `speedhack_setSpeed`, and reports where CE's `speedhack_wantedspeed` float
lives in the game. Then CE is closed and later speed changes are a write to that float, which is
what CE itself does. Takes a few seconds; the speed stays until you change it or restart the game.
Re-running training on the same game needs no restart: the address is saved in `%TEMP%\dqn_speedhack_<pid>.txt`, and if that is missing (or you sped the game up in a CE window) `speedhack.py` finds CE's `speedhack_wantedspeed` in the game itself (the global both CE hooks read) instead of hooking twice. `python speedhack.py --detect` shows what it finds. `--game-speed 0` = don't touch the speed.

## 1. Pilot: measure speed (~20 min)
Game open, standing in front of the boss door (no Cheat Engine needed), then:

    python train_ablation.py --run-dir runs_pilot/speed --max-steps 3000

Look at `runs_pilot/speed/summary.json` (`wall_time_h`) and `episodes.csv` (`mean_step_ms`).
Hours per run ≈ wall_time_h x (80000 / 3000). Total ≈ that x 12 runs + evaluation.

## 2. Full queue
    python run_experiments.py --plan           # see what will run
    python run_experiments.py --max-steps 80000 --seeds 0 1 2 --expect-player-hp 15482

Every run is guarded (`RunGuard` in `ablation_common.py`): HP sanity check after each reset, a 1 s idle
preflight after the first one, and an abort after 5 instant deaths in a row or 3 glitches in 10 episodes.
An abort saves the data, exits with code 3 and stops the queue; fix the game, re-run the same command.

Order: random baseline, then all configs for seed 0, then seed 1, then seed 2.
Press `q` to stop (the queue stops too); re-run the same command to continue.
A half-finished run is moved to `*_incomplete_<time>` and restarted from scratch
(runs are never resumed mid-way, so every run is a clean experiment).
`runs/queue_log.txt` records everything.

Optional 2x2 Dueling ablation: add `--configs dqn double double_per double_dueling double_per_dueling`.

## 3. Human baseline (optional, ~30 min)
    python evaluate.py --human --run-dir runs/human/me --episodes 20

You play (at normal 1x speed by default); the script only records HP and hits.

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
