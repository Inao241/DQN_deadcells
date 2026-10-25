"""Evaluate a trained run (or a random agent) with exploration off.

    python evaluate.py --run-dir runs/double_per_dueling/seed0 --episodes 20
    python evaluate.py --random --run-dir runs/random/seed0 --episodes 20 --seed 0
    python evaluate.py --human --run-dir runs/human/me --episodes 20   # you play, script only records

Writes into --run-dir:
    eval.csv                 one row per evaluation episode
    eval_damage_events.csv   every hit taken during evaluation
    damage_frames/eval_*.png frames before/after every hit (for the damage analysis)
    EVAL_DONE
"""
# Anaconda numpy (MKL) and PyTorch each ship libiomp5md.dll; without this the process aborts with "OMP: Error #15"
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import os
import random
import sys
import time

import torch

import ablation_common as C


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--checkpoint", default="final.pt", help="final.pt (default, unbiased) / best.pt / latest.pt")
    p.add_argument("--eps", type=float, default=0.0, help="exploration during eval (0 = greedy)")
    p.add_argument("--random", action="store_true", help="uniform random agent baseline")
    p.add_argument("--human", action="store_true",
                   help="human baseline: you play with the keyboard, the script only records HP/hits")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-episode-steps", type=int, default=5000)
    p.add_argument("--human-poll-s", type=float, default=0.1)
    p.add_argument("--no-frames", action="store_true", help="do not save damage frames")
    p.add_argument("--game-speed", type=float, default=None,
                   help="default: the speed the run was trained at (2.0 for --random), 1.0 for --human; 0 = don't touch")
    C.add_guard_args(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-resize-window", action="store_true")
    return p


class _HumanSession:
    """Read-only wrapper: no keys are sent, we just poll HP while you play."""

    def __init__(self, inner, poll_s):
        self.inner, self.poll_s = inner, poll_s
        self.Actions = inner.Actions

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def reset(self):
        input("\nPress Enter when you are in the boss room and ready to fight...")
        s = self.inner
        s.player_hp, s.boss_hp = s.hp.get_self_hp(), s.hp.get_boss_hp()
        s.record_max_hp()
        s.state = s.screen.grab()

    def step(self, move, action):
        s = self.inner
        prev_state, php, bhp = s.state, s.player_hp, s.boss_hp
        t0 = time.perf_counter()
        time.sleep(self.poll_s)
        new_php, new_bhp = s.hp.get_self_hp(), s.hp.get_boss_hp()
        t1 = time.perf_counter()
        new_state = s.screen.grab()
        t2 = time.perf_counter()
        win = new_php > 1 and new_bhp <= 1
        done = win or new_php <= 1
        if done and not win and new_bhp <= 1:
            new_bhp = bhp  # boss pointer is gone after death: keep the last real value
        s.state, s.player_hp, s.boss_hp = new_state, new_php, new_bhp
        return C.StepResult(0.0, done, win, prev_state, new_state, php, new_php, bhp, new_bhp,
                            max(0, php - new_php), t1 - t0, t2 - t1, s.boss_max_hp)

    def force_end_episode(self):
        pass


def main(argv=None, session_factory=None, check_keys=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.run_dir, exist_ok=True)
    C.seed_everything(args.seed)
    random.seed(args.seed)
    check_keys = check_keys or C.check_pause_or_quit

    learner = None
    agent_kind = "random" if args.random else ("human" if args.human else "model")
    if agent_kind == "model":
        with open(os.path.join(args.run_dir, "config.json")) as f:
            cfg = json.load(f)
        learner = C.Learner(cfg, args.device)
        learner.load(os.path.join(args.run_dir, args.checkpoint))
        learner.action_net.eval()
        learner.move_net.eval()
    if args.game_speed is None:
        args.game_speed = 1.0 if agent_kind == "human" else (cfg.get("game_speed", 2.0) if learner else 2.0)

    for name in ("eval.csv", "eval_damage_events.csv", "EVAL_DONE"):  # fresh eval each time
        path = os.path.join(args.run_dir, name)
        if os.path.exists(path):
            os.replace(path, path + ".old")

    ep_log = C.CsvLogger(os.path.join(args.run_dir, "eval.csv"), ["agent", "checkpoint"] + C.EPISODE_FIELDS)
    dmg = C.DamageRecorder(args.run_dir, "eval_damage_events.csv",
                           frame_prob=0.0 if args.no_frames else 1.0, prefix="eval_")
    session = (session_factory or (lambda: C.GameSession(resize_window=not args.no_resize_window, game_speed=args.game_speed)))()
    if agent_kind == "human":
        session = _HumanSession(session, args.human_poll_s)

    guard = C.RunGuard(args.expect_player_hp, 0 if agent_kind == "human" else args.preflight_s,
                       max_glitches=args.max_glitches)
    wins, global_step, code = 0, 0, 0
    try:
        for episode in range(args.episodes):
            session.reset()
            guard.after_reset(getattr(session, "inner", session))  # human: write through to the real session
            ep = C.EpisodeStats()
            dmg.new_episode()
            while True:
                if check_keys():
                    raise KeyboardInterrupt
                t0 = time.perf_counter()
                if agent_kind != "model" or random.random() < args.eps:
                    move, action = random.randint(0, C.N_MOVES - 1), random.randint(0, C.N_ACTIONS - 1)
                else:
                    move, action = learner.act(session.state, session.boss_frac, session.player_frac)
                if agent_kind == "human":
                    move, action = C.N_MOVES - 1, C.N_ACTIONS - 1  # unknown -> logged as stay/none
                t_select = time.perf_counter() - t0
                res = session.step(move, action)
                global_step += 1
                glitch = C.is_glitch_win(res, ep.steps)
                truncated = (not res.done) and (ep.steps + 1) >= args.max_episode_steps
                dmg.observe(episode, ep, global_step, move, action, res, res.done and not res.win)
                ep.add_step(move, action, res.reward, t_select, res.t_env, res.t_capture, 0.0, None)
                if res.done or truncated or glitch:
                    break
            session.Actions.Nothing()
            outcome = "glitch" if glitch else ("truncated" if truncated else ("win" if res.win else "loss"))
            wins += outcome == "win"
            row = ep.row(episode, global_step, outcome, session, args.eps if agent_kind == "model" else 1.0)
            row.update(agent=agent_kind, checkpoint=args.checkpoint if learner else "")
            ep_log.write(row)
            guard.check_episode(outcome, ep.steps)
            print("eval episode %d/%d: %s, boss HP left %.1f%%, hits %d" % (
                episode + 1, args.episodes, outcome, 100 * session.boss_frac, ep.n_hits))
            if glitch or truncated:
                session.force_end_episode()
        C.write_json(os.path.join(args.run_dir, "EVAL_DONE"),
                     dict(episodes=args.episodes, wins=wins, agent=agent_kind, game_speed=args.game_speed))
    except KeyboardInterrupt:
        print("\nevaluation stopped by user")
        code = 2
    except RuntimeError as e:
        print("\nGAME ERROR:", e)
        code = 3
    finally:
        ep_log.close()
        dmg.log.close()
    print("win rate: %d/%d" % (wins, args.episodes))
    return code


if __name__ == "__main__":
    sys.exit(main())
