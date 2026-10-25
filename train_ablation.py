"""Train one configuration of the ablation (one config x one seed).

Examples (run from this folder, Dead Cells open, standing in front of the boss door):
    python train_ablation.py --run-dir runs/double_per_dueling/seed0 --double 1 --per 1 --dueling 1 --seed 0
    python train_ablation.py --run-dir runs/dqn/seed1 --double 0 --per 0 --dueling 0 --seed 1

Outputs in --run-dir:
    config.json          all hyperparameters + flags
    episodes.csv         one row per episode (return, win, HP left, latency breakdown, action counts)
    damage_events.csv    one row per hit taken (damage, what the agent was doing, fatal or not)
    latest.pt / best.pt / final.pt   checkpoints
    tb/                  TensorBoard logs
    DONE                 written only when the step budget is reached

Keys while running: q = stop (exit code 2), p = pause/resume.
"""
# Anaconda numpy (MKL) and PyTorch each ship libiomp5md.dll; without this the process aborts with "OMP: Error #15"
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

import ablation_common as C

EXIT_OK, EXIT_USER_QUIT, EXIT_GAME_ERROR = 0, 2, 3


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--double", type=int, default=1, choices=[0, 1])
    p.add_argument("--per", type=int, default=1, choices=[0, 1])
    p.add_argument("--dueling", type=int, default=1, choices=[0, 1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=80000,
                   help="training budget in environment steps (same for every config); "
                        "training stops at the end of the episode in which it is reached")
    p.add_argument("--max-episode-steps", type=int, default=5000,
                   help="safety cap; an episode this long is truncated (HP reading probably broke)")
    # hyperparameters: defaults = original train.py
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=0.9)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay", type=float, default=10000)
    p.add_argument("--beta-start", type=float, default=0.4)
    p.add_argument("--beta-end", type=float, default=1.0)
    p.add_argument("--beta-decay", type=float, default=10000)
    p.add_argument("--tau", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-prob", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--capacity", type=int, default=10000)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--save-every", type=int, default=10, help="episodes between latest.pt saves")
    p.add_argument("--best-window", type=int, default=20)
    p.add_argument("--damage-frame-prob", type=float, default=0.0,
                   help="probability of saving frames for a non-fatal hit during training "
                        "(fatal hits are always saved when > 0). Evaluation saves all.")
    p.add_argument("--game-speed", type=float, default=2.0,
                   help="game speed set through speedhack.py (2.0 = what Cheat Engine was doing). "
                        "0 = don't touch the speed. The reset timings in lib/env.py assume 2x.")
    C.add_guard_args(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-resize-window", action="store_true")
    return p


def prepare_run_dir(run_dir):
    """Never mix two attempts in one folder: a half-finished folder is renamed
    to <name>_incomplete_<timestamp> and a fresh one is started."""
    if os.path.exists(os.path.join(run_dir, "DONE")):
        return False
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        moved = "%s_incomplete_%s" % (run_dir.rstrip("/\\"), time.strftime("%Y%m%d-%H%M%S"))
        os.rename(run_dir, moved)
        print("found unfinished run, moved it to", moved)
    os.makedirs(run_dir, exist_ok=True)
    return True


def main(argv=None, session_factory=None, check_keys=None):
    args = build_parser().parse_args(argv)
    cfg = {k.replace("-", "_"): v for k, v in vars(args).items()}
    run_dir = args.run_dir
    if not prepare_run_dir(run_dir):
        print(run_dir, "already DONE, skipping")
        return EXIT_OK

    C.seed_everything(args.seed)
    C.write_json(os.path.join(run_dir, "config.json"), cfg)
    device = args.device

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    except Exception:  # tensorboard not installed (or broken): everything important is in the CSVs anyway
        class _NoWriter:
            def add_scalar(self, *a, **k):
                pass

            def close(self):
                pass
        writer = _NoWriter()
        print("(tensorboard not installed - skipping TensorBoard logs, CSV logs are unaffected)")

    learner = C.Learner(cfg, device)
    buffer = C.ReplayBuffer(args.capacity, bool(args.per), args.alpha, args.min_prob, device)
    ep_log = C.CsvLogger(os.path.join(run_dir, "episodes.csv"), C.EPISODE_FIELDS)
    dmg = C.DamageRecorder(run_dir, "damage_events.csv", frame_prob=args.damage_frame_prob, prefix="train_")
    check_keys = check_keys or C.check_pause_or_quit

    session = (session_factory or (lambda: C.GameSession(resize_window=not args.no_resize_window, game_speed=args.game_speed)))()

    global_step, episode = 0, 0
    best_mean, recent_returns = None, []
    exit_code, stop_reason = EXIT_OK, None
    t_run = time.time()
    print("config:", {k: cfg[k] for k in ("double", "per", "dueling", "seed", "max_steps", "game_speed")})

    guard = C.RunGuard(args.expect_player_hp, args.preflight_s, max_glitches=args.max_glitches)

    try:
        session.reset()
        guard.after_reset(session)
        ep = C.EpisodeStats()
        dmg.new_episode()
        while True:
            if check_keys():
                stop_reason, exit_code = "user_quit", EXIT_USER_QUIT
                break

            eps = C.epsilon_at(global_step, cfg)
            t0 = time.perf_counter()
            if random.random() < eps:
                move, action = random.randint(0, C.N_MOVES - 1), random.randint(0, C.N_ACTIONS - 1)
            else:
                move, action = learner.act(session.state, session.boss_frac, session.player_frac)
            t_select = time.perf_counter() - t0

            prev_boss_frac, prev_player_frac = session.boss_frac, session.player_frac
            res = session.step(move, action)
            global_step += 1

            # A "win" in the first 3 steps, or straight from a high boss HP, means the boss
            # HP read failed (original code stopped training here). Don't learn from it.
            glitch = C.is_glitch_win(res, ep.steps)
            truncated = (not res.done) and (ep.steps + 1) >= args.max_episode_steps

            if not glitch:
                buffer.append(C.Experience(
                    res.prev_state, move, action, res.reward, res.done, res.new_state,
                    prev_boss_frac, session.boss_frac, prev_player_frac, session.player_frac))

            info, t_train = None, 0.0
            if global_step % args.train_every == 0:
                t1 = time.perf_counter()
                info = learner.optimize(buffer, C.beta_at(global_step, cfg))
                t_train = time.perf_counter() - t1
                if info and learner.updates % 100 == 0:
                    for k, v in info.items():
                        writer.add_scalar("train/" + k, v, global_step)

            fatal = res.done and not res.win
            dmg.observe(episode, ep, global_step, move, action, res, fatal)
            ep.add_step(move, action, res.reward, t_select, res.t_env, res.t_capture, t_train, info)
            print("step %d ep %d r %.2f move %s act %s php %d bhp %d    " % (
                global_step, episode, res.reward, C.MOVE_NAMES[move], C.ACTION_NAMES[action],
                session.player_hp, session.boss_hp), end="\r")

            if not (res.done or truncated or glitch):
                continue

            # ---------------- episode end
            session.Actions.Nothing()
            outcome = "glitch" if glitch else ("truncated" if truncated else ("win" if res.win else "loss"))
            ep_log.write(ep.row(episode, global_step, outcome, session, eps))
            print("\nepisode %d %s return %.2f steps %d boss_hp %.1f%% eps %.3f buffer %d total_steps %d" % (
                episode, outcome, ep.ret, ep.steps, 100 * session.boss_frac, eps, len(buffer), global_step))
            writer.add_scalar("episode/return", ep.ret, global_step)
            writer.add_scalar("episode/win", int(res.win), global_step)
            writer.add_scalar("episode/boss_hp_frac_end", session.boss_frac, global_step)
            guard.check_episode(outcome, ep.steps)

            if outcome in ("win", "loss"):
                recent_returns.append(ep.ret)
                mean_r = float(np.mean(recent_returns[-args.best_window:]))
                if len(recent_returns) >= args.best_window and (best_mean is None or mean_r > best_mean):
                    best_mean = mean_r
                    learner.save(os.path.join(run_dir, "best.pt"),
                                 dict(episode=episode, global_step=global_step, mean_return=mean_r))
            episode += 1
            if episode % args.save_every == 0:
                learner.save(os.path.join(run_dir, "latest.pt"), dict(episode=episode, global_step=global_step))

            if glitch or truncated:
                session.force_end_episode()
            if global_step >= args.max_steps:
                stop_reason = "budget"
                break
            session.reset()
            guard.after_reset(session)
            ep = C.EpisodeStats()
            dmg.new_episode()

    except RuntimeError as e:  # game / HP reading problem: stop cleanly, keep data
        print("\nGAME ERROR:", e)
        stop_reason, exit_code = "game_error: %s" % e, EXIT_GAME_ERROR
    finally:
        try:
            session.Actions.Nothing()
        except Exception:
            pass
        learner.save(os.path.join(run_dir, "final.pt" if stop_reason == "budget" else "latest.pt"),
                     dict(episode=episode, global_step=global_step))
        ep_log.close()
        dmg.log.close()
        writer.close()

    summary = dict(stop_reason=stop_reason, episodes=episode, global_step=global_step,
                   wall_time_h=round((time.time() - t_run) / 3600, 3), best_mean_return=best_mean)
    C.write_json(os.path.join(run_dir, "summary.json"), summary)
    if stop_reason == "budget":
        C.write_json(os.path.join(run_dir, "DONE"), summary)
    print("\n", summary)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
