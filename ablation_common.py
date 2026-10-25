"""Shared pieces for the ablation experiments (train_ablation.py / evaluate.py).

Everything game-specific is still done by the original lib/ modules
(GetScreen, Actions, env, GetHp); this file only adds:
  - switchable Double DQN / PER / Dueling
  - seeding
  - per-episode CSV logging with latency breakdown
  - per-hit damage-event logging (+ optional frame snapshots)
"""
# Anaconda numpy (MKL) and PyTorch each ship libiomp5md.dll; without this the process aborts with "OMP: Error #15"
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import collections
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DQN_action, DQN_move

BOSS_MAX_HP = 234930
PLAYER_MAX_HP = 15482
N_MOVES = 3
N_ACTIONS = 5
# Index order must match lib/Actions.py
ACTION_NAMES = ["attack", "shield", "roll", "jump", "none"]
MOVE_NAMES = ["left", "right", "stay"]
DEFENSIVE_ACTIONS = {1, 2}  # shield, roll


# --------------------------------------------------------------------------- utils
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_at(step, cfg):
    return cfg["eps_end"] + (cfg["eps_start"] - cfg["eps_end"]) * math.exp(-1.0 * step / cfg["eps_decay"])


def beta_at(step, cfg):
    frac = step / cfg["beta_decay"]
    return min(frac * cfg["beta_end"] + (1 - frac) * cfg["beta_start"], cfg["beta_end"])


class CsvLogger:
    """Append rows to a CSV; header is written once. Flushes every row so a
    crash never loses data."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


# --------------------------------------------------------------------------- replay
Experience = collections.namedtuple(
    "Experience",
    ["state", "move", "action", "reward", "done", "new_state",
     "boss", "new_boss", "player", "new_player"],
)


class ReplayBuffer:
    """Uniform or proportional prioritized replay (Schaul et al. 2016).

    Like the original code, the move head and the action head keep separate
    priorities and are sampled separately."""

    def __init__(self, capacity, prioritized, alpha, min_prob, device):
        self.buffer = collections.deque(maxlen=capacity)
        self.prio = {"move": collections.deque(maxlen=capacity),
                     "action": collections.deque(maxlen=capacity)}
        self.prioritized = prioritized
        self.alpha = alpha
        self.min_prob = min_prob
        self.device = device
        self.max_td = {"move": 1.0, "action": 1.0}

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)
        # new transitions get the current max priority so they are seen at least once
        self.prio["move"].append(self.max_td["move"])
        self.prio["action"].append(self.max_td["action"])

    def sample(self, batch_size, beta, head):
        n = len(self.buffer)
        if self.prioritized:
            td = np.asarray(self.prio[head], dtype=np.float64)
            p = (td + self.min_prob) ** self.alpha
            probs = p / p.sum()
            idx = np.random.choice(n, batch_size, replace=True, p=probs)
            w = (n * probs[idx]) ** (-beta)
            w = w / w.max()
        else:
            idx = np.random.randint(0, n, size=batch_size)
            w = np.ones(batch_size)
        exps = [self.buffer[i] for i in idx]
        d = self.device
        batch = dict(
            state=torch.stack([e.state for e in exps]).to(d),
            move=torch.tensor([e.move for e in exps], dtype=torch.int64, device=d),
            action=torch.tensor([e.action for e in exps], dtype=torch.int64, device=d),
            reward=torch.tensor([e.reward for e in exps], dtype=torch.float32, device=d),
            done=torch.tensor([e.done for e in exps], dtype=torch.float32, device=d),
            new_state=torch.stack([e.new_state for e in exps]).to(d),
            boss=torch.tensor([[e.boss] for e in exps], dtype=torch.float32, device=d),
            new_boss=torch.tensor([[e.new_boss] for e in exps], dtype=torch.float32, device=d),
            player=torch.tensor([[e.player] for e in exps], dtype=torch.float32, device=d),
            new_player=torch.tensor([[e.new_player] for e in exps], dtype=torch.float32, device=d),
        )
        return batch, idx, torch.tensor(w, dtype=torch.float32, device=d)

    def update_priorities(self, head, idx, td):
        if not self.prioritized:
            return
        q = self.prio[head]
        for i, t in zip(idx, td):
            q[i] = float(t)
        self.max_td[head] = max(self.max_td[head], float(np.max(td)))


# --------------------------------------------------------------------------- learner
class Learner:
    """Two-headed (factored) agent from the original code: the action net picks
    one of 5 actions, the move net picks one of 3 moves conditioned on that
    action. Double / Dueling / PER are switchable."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        du = bool(cfg["dueling"])
        self.action_net = DQN_action(N_ACTIONS, dueling=du).to(device)
        self.action_tgt = DQN_action(N_ACTIONS, dueling=du).to(device)
        self.move_net = DQN_move(N_MOVES, dueling=du).to(device)
        self.move_tgt = DQN_move(N_MOVES, dueling=du).to(device)
        self.action_tgt.load_state_dict(self.action_net.state_dict())
        self.move_tgt.load_state_dict(self.move_net.state_dict())
        self.action_opt = torch.optim.Adam(self.action_net.parameters(), lr=cfg["lr"], amsgrad=True)
        self.move_opt = torch.optim.Adam(self.move_net.parameters(), lr=cfg["lr"], amsgrad=True)
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.updates = 0

    def _onehot(self, a):
        return F.one_hot(a, N_ACTIONS).float()

    @torch.no_grad()
    def act(self, state, boss_frac, player_frac):
        d = self.device
        s = state.unsqueeze(0).to(d)
        b = torch.tensor([[boss_frac]], dtype=torch.float32, device=d)
        p = torch.tensor([[player_frac]], dtype=torch.float32, device=d)
        action = int(self.action_net(s, b, p).argmax(1)[0])
        oh = self._onehot(torch.tensor([action], device=d))
        move = int(self.move_net(s, oh).argmax(1)[0])
        return move, action

    @torch.no_grad()
    def _next_action(self, bt, use_online):
        net = self.action_net if use_online else self.action_tgt
        return net(bt["new_state"], bt["new_boss"], bt["new_player"]).argmax(1)

    def _action_loss(self, bt, w):
        q = self.action_net(bt["state"], bt["boss"], bt["player"]).gather(
            1, bt["action"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg["double"]:
                # Double DQN: online net picks a', target net evaluates it (on s')
                a_next = self._next_action(bt, use_online=True)
                q_next = self.action_tgt(bt["new_state"], bt["new_boss"], bt["new_player"]).gather(
                    1, a_next.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.action_tgt(bt["new_state"], bt["new_boss"], bt["new_player"]).max(1).values
            y = bt["reward"] + self.cfg["gamma"] * (1.0 - bt["done"]) * q_next
        loss = (self.criterion(q, y) * w).sum()
        return loss, (y - q).abs().detach().cpu().numpy()

    def _move_loss(self, bt, w):
        q = self.move_net(bt["state"], self._onehot(bt["action"])).gather(
            1, bt["move"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # The move head at s' is conditioned on the action the policy would
            # take at s' (not on the action taken at s).
            a_next = self._next_action(bt, use_online=bool(self.cfg["double"]))
            oh_next = self._onehot(a_next)
            if self.cfg["double"]:
                m_next = self.move_net(bt["new_state"], oh_next).argmax(1)
                q_next = self.move_tgt(bt["new_state"], oh_next).gather(1, m_next.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.move_tgt(bt["new_state"], oh_next).max(1).values
            y = bt["reward"] + self.cfg["gamma"] * (1.0 - bt["done"]) * q_next
        loss = (self.criterion(q, y) * w).sum()
        return loss, (y - q).abs().detach().cpu().numpy()

    def optimize(self, buffer, beta):
        bs = self.cfg["batch_size"]
        if len(buffer) < bs:
            return None
        bm, im, wm = buffer.sample(bs, beta, "move")
        ba, ia, wa = buffer.sample(bs, beta, "action")
        loss_m, td_m = self._move_loss(bm, wm)
        loss_a, td_a = self._action_loss(ba, wa)

        self.move_opt.zero_grad()
        loss_m.backward()
        nn.utils.clip_grad_value_(self.move_net.parameters(), 5)
        self.move_opt.step()

        self.action_opt.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_value_(self.action_net.parameters(), 5)
        self.action_opt.step()

        buffer.update_priorities("move", im, td_m)
        buffer.update_priorities("action", ia, td_a)
        self._soft_update()
        self.updates += 1
        return dict(loss_move=loss_m.item(), loss_action=loss_a.item(),
                    td_move=float(td_m.mean()), td_action=float(td_a.mean()))

    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg["tau"]
        for tgt, net in ((self.move_tgt, self.move_net), (self.action_tgt, self.action_net)):
            for tp, p in zip(tgt.parameters(), net.parameters()):
                tp.mul_(1 - tau).add_(p, alpha=tau)

    def save(self, path, extra=None):
        torch.save(dict(
            action_net=self.action_net.state_dict(), action_tgt=self.action_tgt.state_dict(),
            move_net=self.move_net.state_dict(), move_tgt=self.move_tgt.state_dict(),
            action_opt=self.action_opt.state_dict(), move_opt=self.move_opt.state_dict(),
            updates=self.updates, cfg=self.cfg, extra=extra or {}), path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.action_net.load_state_dict(ck["action_net"])
        self.move_net.load_state_dict(ck["move_net"])
        self.action_tgt.load_state_dict(ck.get("action_tgt", ck["action_net"]))
        self.move_tgt.load_state_dict(ck.get("move_tgt", ck["move_net"]))
        return ck


# --------------------------------------------------------------------------- game
StepResult = collections.namedtuple(
    "StepResult",
    ["reward", "done", "win", "prev_state", "new_state", "prev_player_hp", "new_player_hp",
     "prev_boss_hp", "new_boss_hp", "damage_taken", "t_env", "t_capture"],
)


class GameSession:
    """Thin wrapper around the original env/screen/HP code that adds timing
    and damage bookkeeping. Game logic and reward are unchanged."""

    def __init__(self, resize_window=True, game_speed=None):
        from lib import GetScreen, Actions, env, GetHp  # imported here so the self-test can mock them
        self.Actions = Actions
        if game_speed:  # None/0 = leave the game speed alone (e.g. if you still use Cheat Engine)
            import speedhack
            speedhack.set_speed(game_speed)
            print("game speed: %.2fx" % game_speed)
        if resize_window:
            import pygetwindow
            # exact title: getWindowsWithTitle() matches substrings and can pick
            # the "...\Dead Cells\deadcells.exe" console window instead
            wins = [w for w in pygetwindow.getWindowsWithTitle("Dead Cells") if w.title.strip().lower() == "dead cells"]
            if not wins:
                raise RuntimeError("Dead Cells window not found - is the game running?")
            wins[0].size = (960, 540)
            time.sleep(0.5)
        self.screen = GetScreen.GetScreen()
        self.env = env.env()
        self.hp = GetHp.Hp_getter()
        self.state = None
        self.player_hp = None
        self.boss_hp = None

    def reset(self, max_recover=3):
        self.env._reset()
        tries = 0
        # same recovery as the original train.py: if we are not in the boss room, walk back in
        while not self.hp.get_boss_hp():
            tries += 1
            if tries > max_recover:
                raise RuntimeError("Could not enter the boss room after %d recovery attempts" % max_recover)
            time.sleep(1)
            self.Actions.Move_Right()
            time.sleep(8)
            self.Actions.Nothing()
            self.Actions.Move_Left()
            time.sleep(5.5)
            self.Actions.Nothing()
            self.env._reset()
        self.player_hp = self.hp.get_self_hp()
        self.boss_hp = self.hp.get_boss_hp()
        self.state = self.screen.grab()

    @property
    def boss_frac(self):
        return self.boss_hp / BOSS_MAX_HP

    @property
    def player_frac(self):
        return self.player_hp / PLAYER_MAX_HP

    def step(self, move, action):
        prev_state, prev_php, prev_bhp = self.state, self.player_hp, self.boss_hp
        t0 = time.perf_counter()
        reward, done, new_php, new_bhp, _, _ = self.env.step(move, action, prev_php, prev_bhp)
        t1 = time.perf_counter()
        new_state = self.screen.grab()
        t2 = time.perf_counter()
        win = bool(done and new_php > 1)
        # if the game stores negative HP we keep the true overkill damage;
        # if it floors at 0, a fatal hit only shows the HP that was left
        damage = max(0, prev_php - new_php)
        self.state, self.player_hp, self.boss_hp = new_state, new_php, new_bhp
        return StepResult(reward, bool(done), win, prev_state, new_state, prev_php, new_php,
                          prev_bhp, new_bhp, damage, t1 - t0, t2 - t1)

    def force_end_episode(self, timeout=40.0):
        """Used for truncated / glitched episodes: drop the player to 1 HP and
        wait for the boss to land a hit, so the normal death->reset sequence works."""
        self.Actions.Nothing()
        self.hp.set_self_hp(1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.hp.get_self_hp() < 1:
                time.sleep(1.0)
                return True
            time.sleep(0.2)
        raise RuntimeError("force_end_episode: player did not die within %.0fs" % timeout)


# --------------------------------------------------------------------------- logging
EPISODE_FIELDS = (
    ["episode", "global_step", "steps", "outcome", "win", "return",
     "boss_hp_end", "boss_hp_frac_end", "player_hp_end", "player_hp_frac_end",
     "epsilon", "duration_s", "n_hits", "damage_taken",
     "mean_select_ms", "mean_env_ms", "mean_capture_ms", "mean_train_ms",
     "mean_step_ms", "p95_step_ms", "train_updates", "mean_loss_move", "mean_loss_action"]
    + ["act_" + a for a in ACTION_NAMES] + ["move_" + m for m in MOVE_NAMES]
)

DAMAGE_FIELDS = [
    "episode", "step_in_episode", "t_in_episode_s", "global_step", "damage", "fatal",
    "player_hp_before", "player_hp_after", "boss_hp_frac",
    "action", "move", "prev_actions", "defended_recent", "frames",
]


class EpisodeStats:
    def __init__(self):
        self.t_start = time.time()
        self.steps = 0
        self.ret = 0.0
        self.n_hits = 0
        self.damage = 0
        self.act_counts = [0] * N_ACTIONS
        self.move_counts = [0] * N_MOVES
        self.t = collections.defaultdict(float)
        self.step_ms = []
        self.losses_m = []
        self.losses_a = []
        self.updates = 0

    def add_step(self, move, action, reward, t_select, t_env, t_capture, t_train, train_info):
        self.steps += 1
        self.ret += reward
        self.act_counts[action] += 1
        self.move_counts[move] += 1
        self.t["select"] += t_select
        self.t["env"] += t_env
        self.t["capture"] += t_capture
        self.t["train"] += t_train
        self.step_ms.append(1000 * (t_select + t_env + t_capture + t_train))
        if train_info:
            self.updates += 1
            self.losses_m.append(train_info["loss_move"])
            self.losses_a.append(train_info["loss_action"])

    def row(self, episode, global_step, outcome, session, epsilon):
        n = max(self.steps, 1)
        r = dict(
            episode=episode, global_step=global_step, steps=self.steps, outcome=outcome,
            win=int(outcome == "win"), **{"return": round(self.ret, 4)},
            boss_hp_end=session.boss_hp, boss_hp_frac_end=round(max(session.boss_hp, 0) / BOSS_MAX_HP, 5),
            player_hp_end=session.player_hp,
            player_hp_frac_end=round(max(session.player_hp, 0) / PLAYER_MAX_HP, 5),
            epsilon=round(epsilon, 5), duration_s=round(time.time() - self.t_start, 2),
            n_hits=self.n_hits, damage_taken=self.damage,
            mean_select_ms=round(1000 * self.t["select"] / n, 3),
            mean_env_ms=round(1000 * self.t["env"] / n, 3),
            mean_capture_ms=round(1000 * self.t["capture"] / n, 3),
            mean_train_ms=round(1000 * self.t["train"] / n, 3),
            mean_step_ms=round(float(np.mean(self.step_ms)) if self.step_ms else 0.0, 3),
            p95_step_ms=round(float(np.percentile(self.step_ms, 95)) if self.step_ms else 0.0, 3),
            train_updates=self.updates,
            mean_loss_move=round(float(np.mean(self.losses_m)), 6) if self.losses_m else "",
            mean_loss_action=round(float(np.mean(self.losses_a)), 6) if self.losses_a else "",
        )
        for i, a in enumerate(ACTION_NAMES):
            r["act_" + a] = self.act_counts[i]
        for i, m in enumerate(MOVE_NAMES):
            r["move_" + m] = self.move_counts[i]
        return r


def _to_uint8_frames(state):
    a = state.detach().float().cpu().numpy()  # (4,128,128) in [0,1]
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def save_frame_strip(prev_state, new_state, path):
    """Save the 4 frames the agent saw before the hit and the 4 frames after,
    side by side, as one PNG (8 x 128px wide)."""
    import cv2
    before = _to_uint8_frames(prev_state)
    after = _to_uint8_frames(new_state)
    sep = np.full((before.shape[1], 6), 255, np.uint8)
    row = np.hstack(list(before) + [sep, sep] + list(after))
    cv2.imwrite(path, row)


class DamageRecorder:
    """Logs every hit the player takes: when, how much, what the agent was doing."""

    def __init__(self, run_dir, csv_name, frame_prob=0.0, history=4, prefix=""):
        self.prefix = prefix
        self.log = CsvLogger(os.path.join(run_dir, csv_name), DAMAGE_FIELDS)
        self.frame_dir = os.path.join(run_dir, "damage_frames")
        self.frame_prob = frame_prob
        self.history = history
        self.recent = collections.deque(maxlen=history)
        if frame_prob > 0:
            os.makedirs(self.frame_dir, exist_ok=True)

    def new_episode(self):
        self.recent.clear()

    def observe(self, episode, ep_stats, global_step, move, action, res, fatal):
        prev_actions = list(self.recent)
        self.recent.append(action)
        if res.damage_taken <= 0:
            return
        ep_stats.n_hits += 1
        ep_stats.damage += res.damage_taken
        frames = ""
        if self.frame_prob > 0 and (fatal or random.random() < self.frame_prob):
            frames = "%sep%05d_s%05d.png" % (self.prefix, episode, ep_stats.steps)
            save_frame_strip(res.prev_state, res.new_state, os.path.join(self.frame_dir, frames))
        window = prev_actions[-2:] + [action]  # current step + 2 before
        self.log.write(dict(
            episode=episode, step_in_episode=ep_stats.steps,
            t_in_episode_s=round(time.time() - ep_stats.t_start, 3), global_step=global_step,
            damage=res.damage_taken, fatal=int(fatal),
            player_hp_before=res.prev_player_hp, player_hp_after=res.new_player_hp,
            boss_hp_frac=round(max(res.prev_boss_hp, 0) / BOSS_MAX_HP, 5),
            action=ACTION_NAMES[action], move=MOVE_NAMES[move],
            prev_actions="|".join(ACTION_NAMES[a] for a in prev_actions),
            defended_recent=int(any(a in DEFENSIVE_ACTIONS for a in window)),
            frames=frames,
        ))


def check_pause_or_quit():
    """Same keys as the original code: q = quit, p = pause/resume.
    Returns True if the user asked to quit."""
    import keyboard
    if keyboard.is_pressed("q"):
        return True
    if keyboard.is_pressed("p"):
        print("\npaused - press p again to resume (remember to pause the game too)")
        time.sleep(5)
        while not keyboard.is_pressed("p"):
            time.sleep(0.05)
        print("resumed")
        time.sleep(1)
    return False


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
