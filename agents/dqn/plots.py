import argparse
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, no display needed
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FIG_DIR = "figures"
TRAIN_LOG = "training_run.txt"
TB_DIR = "runs"
VARIANT_TAGS = ["vanilla_dqn", "double_dqn", "dueling_dqn", "per_dqn", "nstep_dqn"]
BASELINES = [
    ("dqn", "models/dqn_breakout/best_model.zip"),
    ("ppo", "models/ppo_breakout/best_model.zip"),
    ("a2c", "models/a2c_breakout/best_model.zip"),
]
EVAL_LINE = re.compile(
    r"\[(\w+)\] step (\d+): eval mean_peak=([\d.]+) median=(\d+) max=(\d+)"
)

def parse_training_log(path):
    series = OrderedDict()
    if not os.path.exists(path):
        return series
    with open(path) as f:
        for line in f:
            m = EVAL_LINE.search(line)
            if not m:
                continue
            tag, step, mean_peak = m.group(1), int(m.group(2)), float(m.group(3))
            steps, vals = series.setdefault(tag, ([], []))
            steps.append(step)
            vals.append(mean_peak)
    return series


def tb_scalar(run_dir, tag="eval/mean_peak"):
    if not os.path.isdir(run_dir):
        return [], []
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    by_step = {s.step: s.value for s in ea.Scalars(tag)}  # last value per step wins
    steps = sorted(by_step)
    return steps, [by_step[s] for s in steps]


def collect_variant_peaks(episodes, seed):
    from agents.dqn.compare import VARIANTS, load_model, run_episodes

    peaks_by_variant = OrderedDict()
    for name, path in VARIANTS:
        try:
            net, cfg = load_model(path)
        except FileNotFoundError:
            print(f"  {name}: no checkpoint at {path}, skipping")
            continue
        peaks, _ = run_episodes(net, cfg["n_stack"], episodes, seed)
        peaks_by_variant[name] = peaks
        print(f"  {name}: {episodes} episodes done")
    return peaks_by_variant


def collect_baseline_peaks(episodes, seed, n_stack=4):
    from agents.evaluate import ALGOS, run_episodes
    from agents.train import make_stacked_env

    peaks_by_algo = OrderedDict()
    for name, path in BASELINES:
        if not os.path.exists(path):
            print(f"  {name}: no checkpoint at {path}, skipping")
            continue
        venv = make_stacked_env(1, n_stack, seed, {"max_steps": 100_000})
        model = ALGOS[name].load(path)
        peaks, _, _ = run_episodes(model, venv, episodes)
        peaks_by_algo[name] = peaks
        print(f"  {name}: {episodes} episodes done")
    return peaks_by_algo


def plot_variant_evolution(series, out):
    plt.figure(figsize=(9, 5))
    for tag in VARIANT_TAGS:
        if tag in series:
            steps, vals = series[tag]
            plt.plot(steps, vals, marker=".", ms=4, label=tag.replace("_dqn", ""))
    plt.yscale("log")  # PER dwarfs the rest on a linear axis
    plt.xlabel("training step")
    plt.ylabel("eval mean peak score (log scale)")
    plt.title("DQN variants — game-score evolution (greedy eval, 20 episodes)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_reward_ab(series, out):
    plt.figure(figsize=(9, 5))
    if "per_dqn" in series:
        s, v = series["per_dqn"]
        plt.plot(s, v, marker=".", ms=4, label="PER v1 (alignment reward)")
    s2, v2 = tb_scalar(os.path.join(TB_DIR, "per_dqn_v4"))
    if s2:
        plt.plot(s2, v2, marker=".", ms=4, label="PER v2 (scarcity + aim reward)")
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("eval mean peak score (log scale)")
    plt.title("PER reward A/B — game-score evolution (v1 alignment vs v2 scarcity + aim)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_boxplots(peaks_by_name, out, title="Greedy eval — peak-score distribution by variant"):
    names = list(peaks_by_name)
    arrs = [peaks_by_name[n] for n in names]
    plt.figure(figsize=(8, 5))
    plt.boxplot(arrs, showmeans=True)
    plt.xticks(range(1, len(names) + 1), names)  # version-safe label setting
    plt.yscale("log")
    plt.ylabel("per-episode peak score (log scale)")
    plt.title(title)
    plt.grid(True, axis="y", which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build report figures")
    parser.add_argument("--episodes", type=int, default=30,
                        help="greedy episodes per variant for the boxplots")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    series = parse_training_log(TRAIN_LOG)
    if not series:
        print(f"warning: no eval lines parsed from {TRAIN_LOG}")

    plot_variant_evolution(series, os.path.join(FIG_DIR, "variants_evolution.png"))
    print(f"wrote {FIG_DIR}/variants_evolution.png")

    plot_reward_ab(series, os.path.join(FIG_DIR, "reward_function.png"))
    print(f"wrote {FIG_DIR}/reward_function.png")

    print(f"running greedy eval for variant boxplots ({args.episodes} eps/variant)...")
    peaks = collect_variant_peaks(args.episodes, args.seed)
    if peaks:
        plot_boxplots(peaks, os.path.join(FIG_DIR, "boxplots.png"))
        print(f"wrote {FIG_DIR}/boxplots.png")
    else:
        print(" skipped: no variant checkpoints found")

    print(f"running greedy eval for baseline boxplots ({args.episodes} eps/algo)...")
    baseline_peaks = collect_baseline_peaks(args.episodes, args.seed)
    if baseline_peaks:
        plot_boxplots(baseline_peaks, os.path.join(FIG_DIR, "baselines_boxplots.png"),
                      title="Greedy eval — peak-score distribution by baseline algorithm")
        print(f"wrote {FIG_DIR}/baselines_boxplots.png")
    else:
        print(" skipped: no baseline checkpoints found")


if __name__ == "__main__":
    main()
