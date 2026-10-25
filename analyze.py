"""Turn runs/ into paper-ready tables and figures.

    python analyze.py                 # reads runs/, writes analysis/
    python analyze.py --window 20     # rolling window for learning curves

Outputs (analysis/):
    results.md            main results table + significance + latency + compute cost
    results_table.csv     same numbers, machine-readable
    results_table.tex     LaTeX table to paste into the paper
    fig_learning_win.png / fig_learning_bosshp.png / fig_learning_return.png
    fig_eval_summary.png
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

ORDER = ["random", "human", "dqn", "double", "double_per", "double_dueling", "double_per_dueling"]
PRETTY = {
    "random": "Random", "human": "Human", "dqn": "DQN", "double": "Double DQN",
    "double_per": "Double + PER", "double_dueling": "Double + Dueling",
    "double_per_dueling": "Double + PER + Dueling",
}
VALID = ("win", "loss")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def discover(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "*"))):
        if not os.path.isdir(d) or "_incomplete_" in d:
            continue
        cfg = os.path.basename(os.path.dirname(d))
        runs.setdefault(cfg, []).append(d)
    names = [n for n in ORDER if n in runs] + sorted(n for n in runs if n not in ORDER)
    return [(n, runs[n]) for n in names]


def pretty(n):
    return PRETTY.get(n, n)


# ------------------------------------------------------------------ statistics
def hier_bootstrap(per_seed_values, n_boot=5000, rng=None):
    """Resample seeds, then episodes within each seed. Returns array of means."""
    rng = rng or np.random.default_rng(0)
    seeds = [np.asarray(v, float) for v in per_seed_values if len(v)]
    if not seeds:
        return np.array([np.nan])
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(seeds), len(seeds))
        out[b] = np.mean([rng.choice(seeds[i], len(seeds[i])).mean() for i in pick])
    return out


def ci(arr):
    return np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)


def fmt_pm(vals, scale=1.0, digits=1):
    vals = np.asarray(vals, float) * scale
    if len(vals) == 0:
        return "-"
    if len(vals) == 1:
        return "%.*f" % (digits, vals[0])
    return "%.*f ± %.*f" % (digits, vals.mean(), digits, vals.std(ddof=1))


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="analysis")
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = discover(args.runs_root)
    if not runs:
        print("no runs found under", args.runs_root)
        return

    table, boot = [], {}
    rand_bhp = None
    for name, dirs in runs:
        ev_seeds = [[r for r in read_csv(os.path.join(d, "eval.csv")) if r.get("outcome") in VALID] for d in dirs]
        ev_seeds = [s for s in ev_seeds if s]
        if not ev_seeds:
            continue
        win = [[r["win"] for r in s] for s in ev_seeds]
        bhp = [[r["boss_hp_frac_end"] for r in s] for s in ev_seeds]
        php = [[r["player_hp_frac_end"] for r in s] for s in ev_seeds]
        ret = [[r["return"] for r in s] for s in ev_seeds]
        dur = [[r["duration_s"] for r in s] for s in ev_seeds]
        hits = [[r["n_hits"] for r in s] for s in ev_seeds]
        excluded = sum(1 for d in dirs for r in read_csv(os.path.join(d, "eval.csv")) if r.get("outcome") not in VALID)
        b_win, b_bhp = hier_bootstrap(win), hier_bootstrap(bhp)
        boot[name] = dict(win=b_win, bhp=b_bhp)
        if name == "random":
            rand_bhp = np.concatenate([np.asarray(v) for v in bhp])
        summ = [json.load(open(os.path.join(d, "summary.json"))) for d in dirs
                if os.path.exists(os.path.join(d, "summary.json"))]
        table.append(dict(
            name=name, n_seeds=len(ev_seeds), n_eps=sum(len(s) for s in ev_seeds), excluded=excluded,
            win_seed=[np.mean(v) for v in win], bhp_seed=[np.mean(v) for v in bhp],
            php_seed=[np.mean(v) for v in php], ret_seed=[np.mean(v) for v in ret],
            dur_seed=[np.mean(v) for v in dur], hits_seed=[np.mean(v) for v in hits],
            win_ci=ci(b_win), bhp_ci=ci(b_bhp),
            train_h=[s["wall_time_h"] for s in summ], train_eps=[s["episodes"] for s in summ],
            train_steps=[s["global_step"] for s in summ],
        ))

    # Glass's delta vs random on boss HP left (positive = leaves the boss with less HP than random)
    for t in table:
        t["glass"] = np.nan
        if rand_bhp is not None and len(rand_bhp) > 1 and rand_bhp.std(ddof=1) > 0 and t["name"] != "random":
            t["glass"] = (rand_bhp.mean() - np.mean(t["bhp_seed"])) / rand_bhp.std(ddof=1)

    # ---------------- main table
    lines = ["# Results", "",
             "Evaluation with exploration off (greedy policy, `final.pt`). Mean ± std across seeds; "
             "[95% CI] from a hierarchical bootstrap (resample seeds, then episodes). "
             "Glitched/truncated episodes are excluded and counted separately.", "",
             "| Agent | Seeds | Episodes | Win rate (%) | 95% CI | Boss HP left (%) | 95% CI | Player HP left (%) | Hits / episode | Glass's δ | Excluded |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for t in table:
        lines.append("| %s | %d | %d | %s | [%.0f, %.0f] | %s | [%.1f, %.1f] | %s | %s | %s | %d |" % (
            pretty(t["name"]), t["n_seeds"], t["n_eps"], fmt_pm(t["win_seed"], 100, 1),
            100 * t["win_ci"][0], 100 * t["win_ci"][1], fmt_pm(t["bhp_seed"], 100, 1),
            100 * t["bhp_ci"][0], 100 * t["bhp_ci"][1], fmt_pm(t["php_seed"], 100, 1),
            fmt_pm(t["hits_seed"], 1, 2), "-" if np.isnan(t["glass"]) else "%.2f" % t["glass"], t["excluded"]))
        csv_rows.append(dict(agent=t["name"], seeds=t["n_seeds"], episodes=t["n_eps"],
                             win_mean=np.mean(t["win_seed"]), win_std=np.std(t["win_seed"], ddof=1) if t["n_seeds"] > 1 else 0,
                             win_ci_lo=t["win_ci"][0], win_ci_hi=t["win_ci"][1],
                             boss_hp_left_mean=np.mean(t["bhp_seed"]),
                             boss_hp_left_std=np.std(t["bhp_seed"], ddof=1) if t["n_seeds"] > 1 else 0,
                             player_hp_left_mean=np.mean(t["php_seed"]), hits_per_ep=np.mean(t["hits_seed"]),
                             glass_delta=t["glass"]))

    # ---------------- pairwise: each config vs the previous one in the ladder
    lines += ["", "## Does each component help? (hierarchical bootstrap)", "",
              "P(B > A) is the fraction of bootstrap samples where B's win rate is higher. "
              "With 3 seeds treat this as descriptive, not as a hypothesis test.", "",
              "| A | B | Δ win rate (pp) | 95% CI | P(B > A) | Δ boss HP left (pp) | 95% CI |",
              "|---|---|---|---|---|---|---|"]
    trained = [t["name"] for t in table if t["name"] not in ("random", "human")]
    pairs = list(zip(trained[:-1], trained[1:]))
    if "double" in trained and "double_dueling" in trained:
        pairs.append(("double", "double_dueling"))
    if "double_per" in trained and "double_per_dueling" in trained:
        pairs.append(("double_per", "double_per_dueling"))
    for a, b in dict.fromkeys(pairs):
        dw = boot[b]["win"] - boot[a]["win"]
        db = boot[b]["bhp"] - boot[a]["bhp"]
        lo, hi = ci(dw)
        blo, bhi = ci(db)
        lines.append("| %s | %s | %+.1f | [%+.1f, %+.1f] | %.2f | %+.1f | [%+.1f, %+.1f] |" % (
            pretty(a), pretty(b), 100 * dw.mean(), 100 * lo, 100 * hi, (dw > 0).mean(),
            100 * db.mean(), 100 * blo, 100 * bhi))

    # ---------------- latency + compute cost (from training logs)
    lines += ["", "## Control latency during training (ms per step, step-weighted)", "",
              "| Agent | select | env (keys + HP read) | screen capture | train update | total | p95 total | steps/s |",
              "|---|---|---|---|---|---|---|---|"]
    for name, dirs in runs:
        rows = [r for d in dirs for r in read_csv(os.path.join(d, "episodes.csv"))]
        if not rows:
            continue
        w = np.array([r["steps"] for r in rows], float)
        def wm(k):
            return np.sum(w * np.array([r[k] for r in rows], float)) / w.sum()
        tot = wm("mean_step_ms")
        lines.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
            pretty(name), wm("mean_select_ms"), wm("mean_env_ms"), wm("mean_capture_ms"),
            wm("mean_train_ms"), tot, np.median([r["p95_step_ms"] for r in rows]), 1000 / tot if tot else 0))

    lines += ["", "## Compute cost", "", "| Agent | Train wall-clock per run (h) | Episodes per run | Env steps per run |",
              "|---|---|---|---|"]
    for t in table:
        if t["train_h"]:
            lines.append("| %s | %s | %s | %s |" % (pretty(t["name"]), fmt_pm(t["train_h"], 1, 2),
                                                     fmt_pm(t["train_eps"], 1, 0), fmt_pm(t["train_steps"], 1, 0)))

    # ---------------- action usage in eval
    acts = ["attack", "shield", "roll", "jump", "none"]
    lines += ["", "## Action usage during evaluation (% of steps)", "",
              "| Agent | " + " | ".join(acts) + " |", "|---|" + "---|" * len(acts)]
    for name, dirs in runs:
        if name == "human":
            continue
        rows = [r for d in dirs for r in read_csv(os.path.join(d, "eval.csv"))]
        if not rows:
            continue
        tot = np.array([sum(r["act_" + a] for r in rows) for a in acts], float)
        if tot.sum():
            lines.append("| %s | " % pretty(name) + " | ".join("%.1f" % v for v in 100 * tot / tot.sum()) + " |")

    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "results_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    write_latex(table, os.path.join(args.out, "results_table.tex"))
    plot_curves(runs, args)
    plot_eval(table, args)
    print("\n".join(lines))
    print("\nwrote", os.path.abspath(args.out))


def write_latex(table, path):
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r"Agent & Win rate (\%) & Boss HP left (\%) & Player HP left (\%) & Hits / ep. \\", r"\midrule"]
    for t in table:
        def pm(v, s, d):
            v = np.asarray(v) * s
            return ("%.*f" % (d, v.mean())) if len(v) < 2 else (r"%.*f $\pm$ %.*f" % (d, v.mean(), d, v.std(ddof=1)))
        out.append(r"%s & %s & %s & %s & %s \\" % (pretty(t["name"]), pm(t["win_seed"], 100, 1),
                                                    pm(t["bhp_seed"], 100, 1), pm(t["php_seed"], 100, 1),
                                                    pm(t["hits_seed"], 1, 2)))
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Greedy evaluation over %s episodes per seed. Mean $\pm$ std across seeds.}" % (
                "/".join(sorted({str(int(t['n_eps'] / max(t['n_seeds'], 1))) for t in table}))),
            r"\label{tab:main}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def rolling(x, w):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def plot_curves(runs, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = [("win", "Win rate (rolling %d episodes)" % args.window, "fig_learning_win.png", 100),
             ("boss_hp_frac_end", "Boss HP left at episode end (%)", "fig_learning_bosshp.png", 100),
             ("return", "Episode return", "fig_learning_return.png", 1)]
    for key, ylabel, fname, scale in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        drew = False
        for name, dirs in runs:
            curves = []
            for d in dirs:
                rows = [r for r in read_csv(os.path.join(d, "episodes.csv")) if r.get("outcome") in VALID]
                if len(rows) < 2:
                    continue
                curves.append((np.array([r["global_step"] for r in rows]),
                               rolling([r[key] for r in rows], args.window) * scale))
            if not curves:
                continue
            xmax = min(c[0][-1] for c in curves)
            grid = np.linspace(0, xmax, 200)
            ys = np.array([np.interp(grid, x, y, left=np.nan) for x, y in curves])
            mean = np.nanmean(ys, 0)
            ax.plot(grid, mean, label="%s (n=%d)" % (pretty(name), len(curves)))
            if len(curves) > 1:
                sd = np.nanstd(ys, 0, ddof=1)
                ax.fill_between(grid, mean - sd, mean + sd, alpha=0.2)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, fname), dpi=200)
        plt.close(fig)


def plot_eval(table, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    names = [pretty(t["name"]) for t in table]
    for ax, key, label in ((axes[0], "win_seed", "Win rate (%)"), (axes[1], "bhp_seed", "Boss HP left (%)")):
        means = [100 * np.mean(t[key]) for t in table]
        sds = [100 * np.std(t[key], ddof=1) if len(t[key]) > 1 else 0 for t in table]
        ax.bar(range(len(names)), means, yerr=sds, capsize=3, color="#4C72B0")
        for i, t in enumerate(table):  # show individual seeds as dots
            ax.scatter([i] * len(t[key]), 100 * np.asarray(t[key]), color="k", s=10, zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_eval_summary.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
