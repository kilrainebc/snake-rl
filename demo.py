"""
demo.py: Pygame visualization for watching the Snake agent play.

Supports two modes:
    --random: Plays with a random agent (baseline for comparison).
    Default:  Loads a trained PPO model and plays with the learned policy.
"""

import argparse
import os
import time
import numpy as np
from snake_env import SnakeEnv

def run_random_episode(env):
    """
    Play one episode with random actions.

    Returns:
        score: Number of foods eaten.
        total_reward: Cumulative reward for the episode.
        steps: Total steps taken.
    """
    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return info["score"], total_reward, info["total_steps"]


def run_trained_episode(env, model):
    """
    Play one episode with the trained PPO model.

    Args:
        env: The Snake environment instance.
        model: A loaded stable-baselines3 PPO model.

    Returns:
        score: Number of foods eaten.
        total_reward: Cumulative reward for the episode.
        steps: Total steps taken.
    """
    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        # model.predict returns (action, hidden_states)
        # deterministic=True disables stochastic sampling for cleaner demos
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return info["score"], total_reward, info["total_steps"]


def main():
    parser = argparse.ArgumentParser(
        description="Watch the Agent play SNAKE"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Use random actions instead of a trained model."
    )
    parser.add_argument(
        "--model-path", type=str, default="output/snake_ppo.zip",
        help="Path to the trained model file (default: output/snake_ppo.zip)."
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to play (default: 3)."
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second for visualization (default: 10)."
    )
    parser.add_argument(
        "--grid-size", type=int, default=10,
        help="Grid size (default: 10). Must match training grid size."
    )
    args = parser.parse_args()

    # Load model if not in random mode
    model = None
    if not args.random:
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found at {args.model_path}")
            print("Train a model first with: `python3 train.py`")
            print("Or use --random for a random agent demo.")
            return

        # Import here to avoid requiring sb3 for untrained demos
        from stable_baselines3 import PPO
        model = PPO.load(args.model_path)
        print(f"Loaded model from {args.model_path}")

    # Create environment with Pygame rendering
    env = SnakeEnv(grid_size=args.grid_size, render_mode="human")
    env.metadata["render_fps"] = args.fps

    mode_name = "Random Agent" if args.random else "Trained PPO Agent"
    print(f"\n{'=' * 50}")
    print(f"  Snake Demo — {mode_name}")
    print(f"  Episodes: {args.episodes} | FPS: {args.fps}")
    print(f"{'=' * 50}\n")

    scores = []
    rewards = []
    lengths = []

    for episode in range(1, args.episodes + 1):
        if args.random:
            score, total_reward, steps = run_random_episode(env)
        else:
            score, total_reward, steps = run_trained_episode(env, model)

        scores.append(score)
        rewards.append(total_reward)
        lengths.append(steps)

        print(
            f"  Episode {episode}: "
            f"Score={score}, Reward={total_reward:.1f}, Steps={steps}"
        )

        # Brief pause between episodes
        if episode < args.episodes:
            time.sleep(1.5)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  Summary ({args.episodes} episodes)")
    print(f"{'=' * 50}")
    print(f"  Avg Score:  {np.mean(scores):.1f}")
    print(f"  Best Score: {max(scores)}")
    print(f"  Avg Reward: {np.mean(rewards):.1f}")
    print(f"  Avg Length: {np.mean(lengths):.0f}")
    print(f"{'=' * 50}")

    env.close()

if __name__ == "__main__":
    main()
