"""Why does the agent get hit, and what kills it?

    python analyze_damage.py                     # uses evaluation hits (default)
    python analyze_damage.py --source train --late 0.3   # last 30% of training instead

Two views of every hit:
  1. What the AGENT was doing (exact): its action at the hit step and whether
     it shielded/rolled in the last 3 steps (= tried to defend but mistimed).
  2. Which BOSS ATTACK it was (proxy): each boss attack deals a fixed amount of
     damage, so hits are grouped by damage amount. Check the groups with the
     frame sheets in analysis/damage_clusters/ and give them names in
     analysis/attack_labels.json, then re-run.

Fatal hits: if the game floors HP at 0, a fatal hit only shows the HP that was
left, so it is matched to an attack only when exactly one attack fits.

Outputs (analysis/): damage_report.md, fig_damage_hist.png,
fig_hit_agent_action.png, fig_hit_boss_attack.png, damage_clusters/*.png,
attack_labels.json
"""
import argparse
import collections
import glob
import json
import os
import random
import string

import numpy as np

from analyze import ORDER, discover, pretty, read_csv

ACTIONS = ["attack", "shield", "roll", "jump", "none"]


# ------------------------------------------------------------------ clustering
def cluster_damage(values, abs_gap, rel_gap, min_share):
    """1-D grouping: sort the distinct damage values and cut wherever the gap to
    the next value is larger than max(abs_gap, rel_gap * value)."""
    v = np.sort(np.asarray(values, float))
    if len(v) == 0:
        return []
    groups, cur = [], [v[0]]
    for x in v[1:]:
        if x - cur[-1] > max(abs_gap, rel_gap * cur[-1]):
            groups.append(cur)
            cur = [x]
        else:
            cur.append(x)
    groups.append(cur)
    out = []
    for g in groups:
        g = np.asarray(g)
        out.append(dict(lo=float(g.min()), hi=float(g.max()), median=float(np.median(g)),
                        n=len(g), rare=len(g) < min_share * len(v)))
    for i, c in enumerate(out):
        c["id"] = string.ascii_uppercase[i] if i < 26 else "Z%d" % i
    return out


def assign(dmg, clusters, tol):
    for c in clusters:
        if c["lo"] - tol <= dmg <= c["hi"] + tol:
            return c["id"]
    return None


def assign_fatal(observed, clusters, fatal_floor_suspected):
    """A fatal hit shows min(true damage, HP left). If its value matches a
    cluster exactly, use it; otherwise it can be any attack at least that big."""
    exact = assign(observed, clusters, tol=max(30, 0.02 * observed))
    if exact and not fatal_floor_suspected:
        return exact
    cands = [c["id"] for c in clusters if c["hi"] >= observed * 0.98 and not c["rare"]]
    if exact and exact not in cands:
        cands.append(exact)
    if len(cands) == 1:
        return cands[0]
    return "ambiguous(%s)" % "/".join(cands) if cands else "unknown"


def load_labels(path, clusters):
    labels = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            labels = json.load(f)
    named = {}
    for c in clusters:
        hit = [l for l in labels if l["min"] <= c["median"] <= l["max"]]
        if hit:
            named[c["id"]] = hit[0]["name"]
        else:
            name = "attack %s (~%d dmg)" % (c["id"], c["median"])
            named[c["id"]] = name
            labels.append(dict(min=c["lo"], max=c["hi"], name=name,
                               note="rename me after looking at damage_clusters/cluster_%s.png" % c["id"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    return named


# ------------------------------------------------------------------ helpers
def pct_table(title, rows_by_cfg, cats, cat_names=None, note=""):
    cat_names = cat_names or cats
    lines = ["", "## " + title, ""]
    if note:
        lines += [note, ""]
    lines += ["| Agent | n | " + " | ".join(cat_names) + " |", "|---|---|" + "---|" * len(cats)]
    for cfg, items in rows_by_cfg:
        if not items:
            continue
        cnt = collections.Counter(items)
        n = len(items)
        lines.append("| %s | %d | " % (pretty(cfg), n) + " | ".join("%.1f" % (100 * cnt[c] / n) for c in cats) + " |")
    return lines


def stacked_bar(rows_by_cfg, cats, cat_names, path, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_by_cfg = [(c, it) for c, it in rows_by_cfg if it]
    if not rows_by_cfg:
        return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bottom = np.zeros(len(rows_by_cfg))
    cmap = plt.get_cmap("tab10")
    for j, cat in enumerate(cats):
        vals = np.array([100 * collections.Counter(it)[cat] / len(it) for _, it in rows_by_cfg])
        ax.bar(range(len(rows_by_cfg)), vals, bottom=bottom, label=cat_names[j], color=cmap(j % 10))
        bottom += vals
    ax.set_xticks(range(len(rows_by_cfg)))
    ax.set_xticklabels([pretty(c) for c, _ in rows_by_cfg], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sheet(examples, path, suptitle, named):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 1.75 * len(examples)), squeeze=False)
    for ax, e in zip(axes[:, 0], examples):
        ax.imshow(mpimg.imread(e["frame_path"]), cmap="gray")
        ax.set_title("%s | %s %s ep %d | damage %d | agent: %s (before: %s)%s" % (
            named.get(e["cluster"], e["cluster"]), pretty(e["cfg"]), e["seed"], e["episode"], e["damage"],
            e["action"], e["prev_actions"] or "-", " | FATAL" if e["fatal"] else ""), fontsize=8, loc="left")
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def contact_sheets(events, clusters, named, out_dir, per_cluster, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    made = 0
    for c in clusters:
        ex = [e for e in events if e["cluster"] == c["id"] and e["frame_path"]]
        if ex:
            ex = rng.sample(ex, min(per_cluster, len(ex)))
            _sheet(ex, os.path.join(out_dir, "cluster_%s.png" % c["id"]),
                   "Group %s (%s): %d hits, damage %d-%d.  Left 4 frames = before the hit, right 4 = after." % (
                       c["id"], named[c["id"]], c["n"], c["lo"], c["hi"]), named)
            made += 1
    fatal = [e for e in events if e["fatal"] and e["frame_path"]]
    if fatal:
        fatal = rng.sample(fatal, min(2 * per_cluster, len(fatal)))
        _sheet(fatal, os.path.join(out_dir, "fatal_blows.png"),
               "Fatal blows (use these to resolve 'ambiguous' causes of death)", named)
        made += 1
    return made


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="analysis")
    p.add_argument("--source", choices=["eval", "train"], default="eval")
    p.add_argument("--late", type=float, default=0.3, help="with --source train: use only the last fraction of training")
    p.add_argument("--abs-gap", type=float, default=150, help="min damage gap that separates two attacks")
    p.add_argument("--rel-gap", type=float, default=0.08, help="relative gap that separates two attacks")
    p.add_argument("--min-share", type=float, default=0.01, help="clusters smaller than this share are marked rare")
    p.add_argument("--examples", type=int, default=8, help="example hits per cluster sheet")
    p.add_argument("--fatal-floor", type=int, default=1, choices=[0, 1],
                   help="1 = assume the game floors HP at 0 (fatal damage is truncated). "
                        "Set 0 if player_hp_after is negative on deaths.")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fname = "eval_damage_events.csv" if args.source == "eval" else "damage_events.csv"

    events = []
    for cfg, dirs in discover(args.runs_root):
        for d in dirs:
            rows = read_csv(os.path.join(d, fname))
            if args.source == "train" and rows:
                cut = (1 - args.late) * max(r["global_step"] for r in rows)
                rows = [r for r in rows if r["global_step"] >= cut]
            for r in rows:
                fp = os.path.join(d, "damage_frames", r["frames"]) if r.get("frames") else ""
                events.append(dict(cfg=cfg, seed=os.path.basename(d), episode=int(r["episode"]),
                                   damage=float(r["damage"]), fatal=int(r["fatal"]),
                                   hp_after=float(r["player_hp_after"]), boss_frac=float(r["boss_hp_frac"]),
                                   action=r["action"], prev_actions=r.get("prev_actions", "") or "",
                                   defended=int(r["defended_recent"]),
                                   frame_path=fp if fp and os.path.exists(fp) else ""))
    if not events:
        print("no damage events found (%s) under %s" % (fname, args.runs_root))
        return

    neg_on_death = any(e["fatal"] and e["hp_after"] < 0 for e in events)
    floor = bool(args.fatal_floor) and not neg_on_death
    nonfatal = [e["damage"] for e in events if not e["fatal"]]
    clusters = cluster_damage(nonfatal, args.abs_gap, args.rel_gap, args.min_share)
    named = load_labels(os.path.join(args.out, "attack_labels.json"), clusters)
    for e in events:
        if e["fatal"]:
            e["cluster"] = assign_fatal(e["damage"], clusters, floor)
        else:
            e["cluster"] = assign(e["damage"], clusters, tol=0) or "unknown"

    cfgs = [c for c in ORDER if any(e["cfg"] == c for e in events)] + \
        sorted({e["cfg"] for e in events} - set(ORDER))
    by_cfg = lambda f, filt=lambda e: True: [(c, [f(e) for e in events if e["cfg"] == c and filt(e)]) for c in cfgs]
    cl_ids = [c["id"] for c in clusters]
    cl_names = [named[c] for c in cl_ids]

    L = ["# Damage analysis", "",
         "Source: **%s** hits (%d hits from %d agents). Fatal hit damage %s." % (
             args.source, len(events), len(cfgs),
             "is truncated by the HP floor, so fatal hits are matched to an attack only when exactly one fits"
             if floor else "is exact (HP goes negative)"), ""]

    # attack groups
    L += ["## Damage groups (≈ boss attacks)", "",
          "| Group | Name | Damage range | Hits (non-fatal) | Rare |", "|---|---|---|---|---|"]
    for c in clusters:
        L.append("| %s | %s | %d–%d | %d | %s |" % (c["id"], named[c["id"]], c["lo"], c["hi"], c["n"],
                                                  "yes" if c["rare"] else ""))
    L += ["", "Check each group with `damage_clusters/cluster_<id>.png` and rename it in `attack_labels.json`. "
          "Two attacks with the same damage will share a group; one group with a very wide range may be two attacks."]

    # hits per episode
    L += ["", "## Hits per episode", "", "| Agent | Hits / episode | Mean damage / hit |", "|---|---|---|"]
    for cfg in cfgs:
        ev = [e for e in events if e["cfg"] == cfg]
        eps = len({(e["seed"], e["episode"]) for e in ev})
        L.append("| %s | %.2f | %.0f |" % (pretty(cfg), len(ev) / max(eps, 1), np.mean([e["damage"] for e in ev])))

    agent_cfgs = [(c, it) for c, it in by_cfg(lambda e: e["action"]) if c != "human"]
    L += pct_table("1. What the agent was doing when hit (% of hits)", agent_cfgs, ACTIONS,
                   note="Exact. Hits while `attack` = trading damage; hits while `shield`/`roll` = defended but mistimed.")
    L += pct_table("1b. Did the agent try to defend in the last 3 steps? (% of hits)",
                   [(c, it) for c, it in by_cfg(lambda e: "tried (shield/roll)" if e["defended"] else "no defense")
                    if c != "human"], ["tried (shield/roll)", "no defense"])
    L += pct_table("2. Which boss attack hit the agent (% of non-fatal hits)",
                   by_cfg(lambda e: e["cluster"], lambda e: not e["fatal"]), cl_ids, cl_names)

    fatal_cats = cl_ids + sorted({e["cluster"] for e in events if e["fatal"]} - set(cl_ids))
    L += pct_table("3. Cause of death: the fatal blow (% of deaths)",
                   by_cfg(lambda e: e["cluster"], lambda e: e["fatal"]), fatal_cats,
                   [named.get(c, c) for c in fatal_cats],
                   note="`ambiguous(X/Y)` = the HP left was smaller than several attacks; check the FATAL frames in the sheets.")

    bins = [(0.66, 1.01, "boss HP 66–100%"), (0.33, 0.66, "boss HP 33–66%"), (-1, 0.33, "boss HP 0–33%")]
    phase = lambda e: next(n for lo, hi, n in bins if lo <= e["boss_frac"] < hi)
    L += pct_table("4. When in the fight the hits happen (% of hits)", by_cfg(phase),
                   [b[2] for b in bins], note="Many bosses change behaviour at low HP.")

    # cross table pooled over trained agents
    L += ["", "## 5. Boss attack × agent action (trained agents pooled, non-fatal hits, counts)", "",
          "| Attack | " + " | ".join(ACTIONS) + " | tried to defend (%) |", "|---|" + "---|" * (len(ACTIONS) + 1)]
    pool = [e for e in events if e["cfg"] not in ("random", "human") and not e["fatal"]]
    for cid in cl_ids:
        ev = [e for e in pool if e["cluster"] == cid]
        if not ev:
            continue
        cnt = collections.Counter(e["action"] for e in ev)
        L.append("| %s | " % named[cid] + " | ".join(str(cnt[a]) for a in ACTIONS) +
                 " | %.0f |" % (100 * np.mean([e["defended"] for e in ev])))

    with open(os.path.join(args.out, "damage_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(nonfatal, bins=min(80, max(10, len(set(nonfatal)))), color="#4C72B0")
    for c in clusters:
        ax.axvspan(c["lo"] - 5, c["hi"] + 5, alpha=0.15, color="orange")
        ax.text(c["median"], ax.get_ylim()[1] * 0.92, c["id"], ha="center", fontsize=9)
    ax.set_xlabel("Damage per hit (non-fatal hits)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_damage_hist.png"), dpi=200)
    plt.close(fig)
    stacked_bar(agent_cfgs, ACTIONS, ACTIONS, os.path.join(args.out, "fig_hit_agent_action.png"),
                "% of hits (agent action at hit)")
    stacked_bar(by_cfg(lambda e: e["cluster"], lambda e: not e["fatal"]), cl_ids, cl_names,
                os.path.join(args.out, "fig_hit_boss_attack.png"), "% of non-fatal hits")
    made = contact_sheets(events, clusters, named, os.path.join(args.out, "damage_clusters"), args.examples)

    print("\n".join(L))
    print("\n%d frame sheets in %s" % (made, os.path.abspath(os.path.join(args.out, "damage_clusters"))))
    print("name the attacks in", os.path.abspath(os.path.join(args.out, "attack_labels.json")), "and re-run")


if __name__ == "__main__":
    main()
