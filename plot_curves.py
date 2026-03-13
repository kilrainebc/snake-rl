"""
plot_curves.py: Plot learning curves from training metrics.

Reads the training_metrics.json file produced by train.py and generates a
two-panel figure:
    Left panel:  Score (foods eaten) over training timesteps.
    Right panel: Episode reward over training timesteps.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def rolling_average(data, window=50):
    """
    Compute a rolling (moving) average over a 1D array.

    Uses numpy's cumsum trick for efficiency. Pads the beginning with NaN
    so the output length matches the input length (the first `window-1`
    values don't have enough data for a full window).

    Args:
        data: 1D array-like of values.
        window: Number of points in the rolling window.

    Returns:
        Numpy array of rolling averages (same length as input).
    """
    data = np.array(data, dtype=float)
    if len(data) < window:
        # Not enough data for even one full window — just return cumulative mean
        return np.cumsum(data) / np.arange(1, len(data) + 1)

    # Cumulative sum trick for O(n) rolling average
    cumsum = np.cumsum(data)
    result = np.empty_like(data)
    # First (window-1) entries: use expanding window
    result[:window] = cumsum[:window] / np.arange(1, window + 1)
    # Remaining entries: full rolling window
    result[window:] = (cumsum[window:] - cumsum[:-window]) / window

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves from training metrics."
    )
    parser.add_argument(
        "--metrics-path", type=str, default="output/training_metrics.json",
        help="Path to training_metrics.json (default: output/training_metrics.json)."
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Path to save the plot (default: same directory as metrics file)."
    )
    parser.add_argument(
        "--window", type=int, default=50,
        help="Rolling average window size (default: 50 episodes)."
    )
    args = parser.parse_args()

    # Load metrics
    if not os.path.exists(args.metrics_path):
        print(f"Error: Metrics file not found at {args.metrics_path}")
        print("Train a model first with: python train.py")
        return

    with open(args.metrics_path, "r") as f:
        metrics = json.load(f)

    timesteps = np.array(metrics["timesteps"])
    scores = np.array(metrics["scores"])
    rewards = np.array(metrics["episode_rewards"])

    print(f"Loaded {len(scores)} episodes from {args.metrics_path}")

    # Compute rolling averages
    score_avg = rolling_average(scores, window=args.window)
    reward_avg = rolling_average(rewards, window=args.window)

    # Create the two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: Score over timesteps ---
    ax1.scatter(
        timesteps, scores,
        alpha=0.15, s=8, color="#74b9ff", label="Per-episode", zorder=1
    )
    ax1.plot(
        timesteps, score_avg,
        color="#0984e3", linewidth=2,
        label=f"Rolling avg (n={args.window})", zorder=2
    )
    ax1.set_xlabel("Training Timesteps", fontsize=12)
    ax1.set_ylabel("Score (Foods Eaten)", fontsize=12)
    ax1.set_title("Score over Training", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # --- Right panel: Episode reward over timesteps ---
    ax2.scatter(
        timesteps, rewards,
        alpha=0.15, s=8, color="#fdcb6e", label="Per-episode", zorder=1
    )
    ax2.plot(
        timesteps, reward_avg,
        color="#e17055", linewidth=2,
        label=f"Rolling avg (n={args.window})", zorder=2
    )
    ax2.set_xlabel("Training Timesteps", fontsize=12)
    ax2.set_ylabel("Episode Reward", fontsize=12)
    ax2.set_title("Reward over Training", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    # Save the figure
    if args.output_path is None:
        output_dir = os.path.dirname(args.metrics_path) or "."
        output_path = os.path.join(output_dir, "learning_curves.png")
    else:
        output_path = args.output_path

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    # Also try to display if running interactively
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
