# Snake RL — Reinforcement Learning Agent for Snake

A complete reinforcement learning project that trains a PPO agent to play Snake using a custom Gymnasium environment. Built as a technical interview work sample.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the agent (~5 min on CPU)
python train.py

# 3. Plot learning curves
python plot_curves.py

# 4. Watch the trained agent play
python demo.py
```

To see the random baseline for comparison:
```bash
python demo.py --random
```

## Project Structure

```
snake-rl/
├── snake_env.py        # Custom Gymnasium environment
├── train.py            # PPO training script with metrics logging
├── demo.py             # Pygame visualization (random vs trained agent)
├── plot_curves.py      # Learning curve plotting
├── requirements.txt    # Minimal deps
├── README.md           # This file
└── output/             # Created during training
    ├── snake_ppo.zip           # Trained model weights
    ├── config.json             # Training hyperparameters
    ├── training_metrics.json   # Per-episode metrics
    └── learning_curves.png     # Score/reward plots
```

## Interview Demo Flow

Suggested order for walking through this project in an interview:

1. **Show the random agent** — `python demo.py --random` — establishes the baseline (score ~0-1)
2. **Show the learning curve** — open `output/learning_curves.png` — demonstrates the agent learned over time
3. **Show the trained agent** — `python demo.py` — contrast with the random baseline
4. **Code walkthrough** — walk through `snake_env.py` (state design, reward shaping), then `train.py` (PPO setup, callback)
5. **Discussion** — "What I'd improve with more time" (see below)

## Design Decisions

### State Representation: Why a 3-Channel Grid?

| Approach | Pros | Cons |
|----------|------|------|
| **Raw pixels** | Closest to Atari benchmarks | Huge input, slow training, wasteful encoding |
| **Flat feature vector** | Small input, fast training | Loses spatial relationships; requires hand-engineering features |
| **3-channel grid (chosen)** | Preserves spatial structure; compact (3x10x10 = 300 floats); CNN-friendly | Requires designing channels thoughtfully |

The three channels encode:
- **Channel 0 — Body gradient**: Head = 1.0, fading to ~0 at the tail. This encodes direction and body ordering — the CNN can infer which way the snake is moving from the gradient.
- **Channel 1 — Head position**: Binary mask. Redundant with channel 0's max value, but makes head extraction trivial for the CNN.
- **Channel 2 — Food position**: Binary mask. Isolated so the CNN can learn food-seeking without body interference.

### Reward Shaping

| Signal | Value | Rationale |
|--------|-------|-----------|
| Eat food | +10 | Primary objective — large positive reinforcement |
| Death | -10 | Strong deterrent for walls and self-collision |
| Move closer to food | +0.5 | Dense gradient toward food — critical for early learning |
| Move farther from food | -0.5 | Penalizes wandering away |
| Per-step penalty | -0.01 | Tiny cost to discourage stalling without overwhelming other signals |

The distance shaping magnitude (+0.5/-0.5) was tuned carefully. At +1/-1, the cumulative approach reward on a 10x10 grid (~18) rivals the +10 food reward, incentivizing oscillation near food. At +0.1/-0.1, signal is too weak for fast learning. At +0.5, max cumulative approach is ~9 — strong enough for fast learning, but the +10 food reward clearly dominates. The hunger timeout (100 + length*10 steps) provides an additional backstop against stalling.

### Custom SmallCNN Feature Extractor

The default SB3 NatureCNN is designed for 84x84 Atari frames (requires 36x36+ input). Our 10x10 grid is much smaller, so we use a custom 3-layer CNN:

```
Conv2d(3 -> 32, 3x3, pad=1) -> ReLU    # Preserves 10x10
Conv2d(32 -> 64, 3x3, pad=1) -> ReLU   # Preserves 10x10
Conv2d(64 -> 64, 3x3, pad=1) -> ReLU   # Preserves 10x10
Flatten -> Linear(6400 -> 256) -> ReLU
```

Key decisions: 3x3 kernels with padding=1 preserve spatial dimensions (no pooling — can't afford to lose resolution at 10x10). Three conv layers provide enough depth for spatial reasoning without overfitting.

### Why PPO over DQN?

- **Stability**: PPO's clipped objective prevents destructively large policy updates, which matters with our multi-component reward signal.
- **On-policy**: PPO learns from fresh experience, avoiding DQN's replay buffer staleness issues.
- **Actor-critic**: The value function baseline reduces variance in gradient estimates.
- **Simplicity**: Fewer hyperparameters to tune than DQN (no replay buffer size, target network update frequency, epsilon schedule).

### Hyperparameter Choices

| Parameter | Value | Why |
|-----------|-------|-----|
| `n_steps` | 2048 | Enough trajectory data per update for stable gradients |
| `batch_size` | 256 | Balances noise (too small) vs speed (too large) |
| `n_epochs` | 4 | Multiple passes per batch without overfitting |
| `learning_rate` | 3e-4 | PPO paper default; works well across many tasks |
| `gamma` | 0.99 | High discount — long-term planning matters in Snake |
| `gae_lambda` | 0.95 | Standard bias-variance tradeoff for advantage estimation |
| `clip_range` | 0.2 | Standard PPO clipping; prevents policy collapse |
| `ent_coef` | 0.01 | Mild exploration bonus; prevents premature convergence |

## CLI Options

### train.py
```bash
python train.py --timesteps 1000000 --grid-size 10 --output-dir output
```

### demo.py
```bash
python demo.py --random              # Random agent baseline
python demo.py --episodes 5 --fps 15 # Trained agent, 5 episodes, faster
python demo.py --model-path path/to/model.zip
```

### plot_curves.py
```bash
python plot_curves.py --metrics-path output/training_metrics.json --window 100
```

## Things I'd Improve with More Time

- **Curriculum learning**: Start with a smaller grid (e.g., 5x5) where food is easier to find, then gradually increase grid size. This accelerates early learning.
- **Hyperparameter sweep**: Use Optuna or similar to systematically search over learning rate, entropy coefficient, and n_steps. The current values are reasonable defaults but not optimized for this specific environment.
- **Separate eval harness**: Run periodic evaluation episodes (with deterministic policy) during training, separate from the training episodes. This gives cleaner learning curves without the noise of exploration.
- **Multi-environment training**: Use `SubprocVecEnv` with 8-16 parallel environments to collect experience faster and improve sample diversity.
- **Self-play / curriculum on snake length**: Gradually increase the initial snake length to force the agent to learn navigation in tighter spaces.
- **Frame stacking**: Stack the last 2-4 observations to give the agent explicit temporal context (current observation only provides implicit direction via the gradient).
