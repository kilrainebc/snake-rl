"""
train.py — PPO training script for the Snake RL agent.

This script trains a PPO agent using stable-baselines3 with a CNN policy on
our custom Snake environment. It logs per-episode metrics to a JSON file for
later analysis.

Design Decisions:
    Why PPO over DQN?
        PPO (Proximal Policy Optimization) is an on-policy actor-critic method.
        While DQN works well for discrete action spaces, PPO tends to be more
        stable and sample-efficient for environments with shaped rewards. PPO's
        clipped objective prevents destructively large policy updates, which is
        important here because our reward signal has multiple components (food,
        death, distance, step penalty) that can cause volatile gradients.
        PPO also parallelizes naturally across multiple environments.

    Why CnnPolicy with a custom SmallCNN?
        Our observation is a 3-channel spatial grid, which is essentially a tiny
        image. CNNs excel at extracting spatial features — in this case, learning
        patterns like "food is above-right" or "body blocks the left path."
        A flat MLP would lose the 2D spatial relationships between cells.

        The default SB3 NatureCNN is designed for 84x84 Atari frames and requires
        at least 36x36 input. Our 10x10 grid is much smaller, so we define a
        custom SmallCNN with 3x3 kernels and no pooling — preserving spatial
        resolution on our already-small grid.

    Hyperparameter Rationale:
        n_steps=2048: Number of steps per environment before each update. 2048
            gives enough trajectory data for stable gradient estimates while keeping
            updates frequent enough for fast learning.
        batch_size=256: Mini-batch size for SGD updates. 256 balances between noisy
            (small batch) and slow (large batch) updates.
        n_epochs=4: Number of passes over the collected data per update. More epochs
            squeeze more learning from each batch, but too many cause overfitting
            to the current batch.
        learning_rate=3e-4: Adam optimizer LR. The "safe default" from the PPO paper.
        gamma=0.99: Discount factor. Close to 1.0 because long-term planning matters
            (eating food now affects survival later).
        gae_lambda=0.95: GAE smoothing parameter. 0.95 balances bias vs variance
            in advantage estimation.
        clip_range=0.2: PPO clipping parameter. Prevents the policy ratio from
            deviating too far from 1.0, stabilizing training.
        ent_coef=0.01: Entropy bonus. Encourages exploration by penalizing overly
            deterministic policies. 0.01 is a mild nudge — enough to prevent
            premature convergence without making the agent too random.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from snake_env import SnakeEnv


class SmallCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for small grid observations.

    The default SB3 NatureCNN is designed for 84x84 frames and requires
    at least 36x36 input. Our 10x10 grid is much smaller, so we use a custom
    architecture with smaller kernels and fewer layers.

    Architecture:
        Conv2d(3 -> 32, kernel=3, padding=1) -> ReLU   # 10x10 -> 10x10
        Conv2d(32 -> 64, kernel=3, padding=1) -> ReLU  # 10x10 -> 10x10
        Conv2d(64 -> 64, kernel=3, padding=1) -> ReLU  # 10x10 -> 10x10
        Flatten -> Linear(64*10*10 -> 256) -> ReLU

    Why this architecture:
        - 3x3 kernels with padding=1 preserve spatial dimensions, so we don't
          lose resolution on our already-small grid.
        - 3 conv layers give enough depth to learn spatial patterns (e.g.,
          "food is 3 cells north and body blocks the direct path").
        - 256-dim output is a standard feature size for RL policy heads.
        - No pooling layers - with only 10x10 input, pooling would destroy
          too much spatial information.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]  # 3 channels
        grid_h = observation_space.shape[1]      # 10
        grid_w = observation_space.shape[2]      # 10

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size after the conv layers
        flat_size = 64 * grid_h * grid_w  # 64 * 10 * 10 = 6400

        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


class MetricsCallback(BaseCallback):
    """
    Custom callback that logs per-episode metrics during training.

    Tracks score (foods eaten), episode length, total reward, and the
    timestep at which each episode ended. Saves everything to a JSON file
    so we can plot learning curves later.
    """

    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.metrics = {
            "scores": [],
            "episode_lengths": [],
            "episode_rewards": [],
            "timesteps": [],
        }
        # Track the current episode's cumulative reward
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        """Called after every environment step during training."""
        # Accumulate reward for the current episode
        # self.locals["rewards"] is a numpy array (one entry per env in the VecEnv)
        self.current_episode_reward += self.locals["rewards"][0]

        # Check if the episode just ended
        # "dones" is True when the episode terminates
        if self.locals["dones"][0]:
            # Extract score from the info dict.
            # In DummyVecEnv, when an episode ends, the info dict from the
            # terminal step() call is preserved in locals["infos"] before
            # the env auto-resets. So our custom info keys (score, snake_length,
            # total_steps) are directly available here.
            infos = self.locals.get("infos", [{}])
            info = infos[0]
            score = info.get("score", 0)
            episode_length = info.get("total_steps", 0)

            # Convert numpy types to Python natives for JSON serialization
            self.metrics["scores"].append(int(score))
            self.metrics["episode_lengths"].append(int(episode_length))
            self.metrics["episode_rewards"].append(
                round(float(self.current_episode_reward), 2)
            )
            self.metrics["timesteps"].append(int(self.num_timesteps))

            if self.verbose >= 1 and len(self.metrics["scores"]) % 100 == 0:
                recent_scores = self.metrics["scores"][-100:]
                avg_score = np.mean(recent_scores)
                print(
                    f"  Episode {len(self.metrics['scores']):>5d} | "
                    f"Avg score (last 100): {avg_score:.1f} | "
                    f"Timestep: {self.num_timesteps:>7d}"
                )

            # Reset episode reward tracker
            self.current_episode_reward = 0.0

        return True  # Returning False would stop training

    def _on_training_end(self):
        """Save metrics to JSON when training finishes."""
        with open(self.save_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {self.save_path}")


def make_env(grid_size):
    """Factory function that creates a Snake environment (required by DummyVecEnv)."""
    def _init():
        return SnakeEnv(grid_size=grid_size)
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent to play Snake."
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1000000)."
    )
    parser.add_argument(
        "--grid-size", type=int, default=10,
        help="Side length of the square grid (default: 10)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory to save model and metrics (default: output/)."
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Paths for saved artifacts
    model_path = os.path.join(args.output_dir, "snake_ppo")
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    config_path = os.path.join(args.output_dir, "config.json")

    print("=" * 60)
    print("Snake RL — PPO Training")
    print("=" * 60)
    print(f"  Grid size:    {args.grid_size}x{args.grid_size}")
    print(f"  Timesteps:    {args.timesteps:,}")
    print(f"  Output dir:   {args.output_dir}/")
    print("=" * 60)

    # Wrap environment in DummyVecEnv (required by stable-baselines3)
    # DummyVecEnv runs envs sequentially in the same process — simple and
    # sufficient for a single env. Use SubprocVecEnv for parallel envs.
    env = DummyVecEnv([make_env(args.grid_size)])

    # Define hyperparameters (see module docstring for rationale)
    hyperparams = {
        "n_steps": 2048,        # Steps per env before policy update
        "batch_size": 256,      # Mini-batch size for SGD
        "n_epochs": 4,          # Epochs per policy update
        "learning_rate": 3e-4,  # Adam learning rate (PPO paper default)
        "gamma": 0.99,          # Discount factor (high = long-term planning)
        "gae_lambda": 0.95,     # GAE advantage smoothing
        "clip_range": 0.2,      # PPO clipping (prevents large policy jumps)
        "ent_coef": 0.01,       # Entropy bonus (mild exploration pressure)
    }

    # Custom policy kwargs: use our SmallCNN instead of the default NatureCNN.
    # NatureCNN requires 36x36+ input, but our grid is only 10x10.
    # normalize_images=False because our observations are already in [0, 1] —
    # the default normalization divides by 255 (expecting uint8 pixel values),
    # which would squash our gradient values to near-zero.
    policy_kwargs = {
        "features_extractor_class": SmallCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }

    # Initialize PPO with CnnPolicy using our custom feature extractor.
    # CnnPolicy builds separate MLP heads for policy and value on top of
    # the shared CNN feature extractor.
    model = PPO(
        "CnnPolicy",
        env,
        seed=42,  # Fixed seed for reproducibility
        verbose=0,  # Suppress SB3's built-in logging (we use our own callback)
        policy_kwargs=policy_kwargs,
        **hyperparams,
    )

    # Save training configuration for reproducibility
    config = {
        "grid_size": args.grid_size,
        "total_timesteps": args.timesteps,
        "algorithm": "PPO",
        "policy": "CnnPolicy",
        "features_extractor": "SmallCNN (3x Conv2d 3x3 + Linear 256)",
        "normalize_images": False,
        "seed": 42,
        "hyperparameters": hyperparams,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Set up the metrics callback
    callback = MetricsCallback(save_path=metrics_path, verbose=1)

    # Train the agent
    print("\nTraining started...")
    start_time = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,  # tqdm progress bar
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Print summary statistics
    scores = callback.metrics["scores"]
    rewards = callback.metrics["episode_rewards"]

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Total episodes:     {len(scores):,}")
    print(f"  Best score:         {max(scores) if scores else 0}")
    print(f"  Last 100 avg score: {np.mean(scores[-100:]):.1f}" if scores else "  N/A")
    print(f"  Last 100 avg reward:{np.mean(rewards[-100:]):.1f}" if rewards else "  N/A")
    print(f"  Best reward:        {max(rewards):.1f}" if rewards else "  N/A")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
