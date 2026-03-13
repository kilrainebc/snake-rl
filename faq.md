# FAQ — Snake RL

## RL Fundamentals

### What is reinforcement learning?
A type of machine learning where an agent learns by interacting with an environment. It takes actions, receives rewards or penalties, and adjusts its strategy to maximize cumulative reward over time. Unlike supervised learning, there are no labeled examples or human interventions; the agent discovers what works through trial and error.

### What is PPO?
Proximal Policy Optimization. It's a policy gradient algorithm that directly optimizes the agent's decision-making strategy. The key innovation is "clipping" — each update to the strategy is bounded so the agent can't change too drastically in one step. This prevents the catastrophic forgetting that plagues simpler policy gradient methods. It was developed by OpenAI in 2017 and has become one of the most widely used RL algorithms.

### Why did I choose PPO Algorithm?
After discussing with my rubber duck, Claude, we selected PPO.  It gave me Three reasons: (1) **Stability** — PPO's clipped updates handle our multi-component reward signal (food + death + distance + step penalty) without the volatile training other models (e.g. DQN) can exhibit. (2) **Simplicity** — Other models have more hyperparameters (replay buffer size, target network update frequency, epsilon schedule). (3) **On-policy** — PPO learns from fresh experience rather than a replay buffer that may contain stale data.

### What about other algorithms — A3C, SAC, DDPG?
A3C (Asynchronous Advantage Actor-Critic) is an older actor-critic method largely superseded by PPO. SAC (Soft Actor-Critic) is designed for continuous action spaces — our Snake has discrete actions (up/down/left/right), so SAC would be overkill. DDPG is also continuous-only. PPO is the standard choice for discrete action spaces with shaped rewards.

### What is a policy? What is a value function?
The **policy** is the agent's strategy — a function that maps observations to actions. The **value function** estimates how good a given state is — "if I'm in this position, how much total reward do I expect from here?".  Basically the 'EV'/'Expected Value' at the current state.  PPO uses both: the policy decides what to do, and the value function helps estimate whether an action was better or worse than average (the "advantage").

### What does the discount factor (gamma = 0.99) do?
It controls how much the agent values future rewards vs immediate rewards. Gamma=0.99 means a reward 100 steps in the future is worth about 37% of an immediate reward (0.99^100 ~ 0.37). We use 0.99 because Snake requires long-term planning — eating food now affects where you can go later, and the decisions you make early determine whether you survive.

### What is reward shaping?
Adding intermediate reward signals beyond the "natural" rewards (food/death) to guide the agent toward good behavior faster. Without distance-based shaping, the agent gets almost no feedback until it randomly finds food — which on a 10x10 grid could take hundreds of random steps. The +0.5/-0.5 distance reward creates a smooth gradient pointing toward the food, dramatically accelerating early learning - but needed quite a bit of tuning and testing to 'goldilocks' that size.

### Isn't reward shaping cheating? You're telling it how to play.
Not exactly. We're providing a gradient (get closer to food = good), not a strategy. The agent still has to learn *how* to navigate around its own body, avoid walls, and balance short-term food-seeking with long-term survival. The shaping just prevents the "needle in a haystack" problem of finding food by pure chance. It's analogous to giving a student a textbook vs making them derive all of physics from first principles.

---

## Environment & State Design

### Why a 3-channel grid instead of raw pixels?
Raw pixels would work (that's apparently how Atari benchmarks are set up), but they're wasteful for a simple grid game POC. A 84x84x3 Atari frame has ~21,000 values, mostly background color. Our 3x10x10 grid has just 300 values, all of which are meaningful. This makes training faster and the representation more learnable.

### Why a 3-channel grid instead of a flat feature vector?
A flat vector (e.g., [head_x, head_y, food_x, food_y, ...]) loses spatial relationships. The CNN can learn patterns like "food is 3 cells to the upper-left and body blocks the direct path" — that spatial reasoning is lost when you flatten to a 1D vector. The grid representation is also more generalizable — it naturally handles any snake length or food position.

### Why the body gradient instead of a binary mask?
A binary mask says "there's body here" but we also need to know 'which end is the head?'/'where am I going?'.  The gradient (head=1.0 fading to tail≈0) encodes direction and ordering easily. The CNN can see that the highest value is the head and values decrease toward the tail, inferring the direction of motion.

### How does the agent know which direction the snake is moving?
From the body gradient in channel 0. If the brightest cell is at the top and values fade downward, the snake is moving up. This is more efficient than frame stacking (which would require storing the last 2-4 frames and using more memory/compute), which is similar to what was needed in the PokemonRL project.

### What are the '4 actions'?
Up, down, left, right. We also block 180-degree turns (pressing the opposite direction is ignored).  This is how the Agent plays the game.

### Why 10x10 grid?
It's small enough to train quickly on CPU, but large enough that the agent needs real strategy. A 5x5 grid would be really easy, and a 20x20 grid would require much longer training. 10x10 is a very much a sweet spot.  I happen to have a homelab with quite a bit of GPU compute available, but I wanted this reproduceable for other folks.

---
## Architecture & Training

### Why a CNN instead of an MLP (fully connected network)?
Our observation is a 2D spatial grid — essentially an 'image'. CNNs excel at learning spatial features: "food is to the upper-right" or "body blocks the left path." An MLP treats every input independently and loses the 2D spatial relationships between cells.

### Why a custom CNN? Why not use the built-in one?
stable-baselines3's default CNN (NatureCNN) is designed for 84x84 Atari frames. It uses 8x8 and 4x4 kernels with stride, which requires at least 36x36 input. Our 10x10 grid is too small — those large kernels would destroy all spatial information. Our SmallCNN uses 3x3 kernels with padding=1 to preserve the full 10x10 resolution.

### Why no pooling layers?
Pooling reduces spatial resolution (e.g., max pooling 2x2 cuts dimensions in half). On a 10x10 grid, one pooling operation would drop us to 5x5, losing half our spatial detail. On an 84x84 Atari frame you can afford multiple pooling layers. We can't.

### How long does training take?
About 25ish minutes on CPU for 1 million timesteps (~47,000 episodes). GPU / Parallelism would make this quicker, but for this POC this is satisfactory.

### What is stable-baselines3?
A Python library that provides reliable implementations of RL algorithms (PPO, DQN, SAC, etc.) built on PyTorch. It handles the training loop, gradient computation, and logging — so we focus on the environment and architecture design rather than reimplementing PPO from scratch (Which is very much outside of my wheelhouse)

### What is Gymnasium?
The standard Python API for RL environments, maintained by the Farama Foundation (successor to OpenAI Gym). It defines the `reset()` and `step()` interface that all RL environments use. This standardization means our Snake environment works with any Gymnasium-compatible algorithm.

### How do you know it's actually learning, not memorizing?
Every episode starts with food in a random position (seeded via the environment's PRNG). The snake always starts in the center, but food placement is random every time. The agent can't memorize specific food positions — it has to learn a general strategy. The learning curves show steady improvement over thousands of unique episodes, not memorization of a few patterns.

### What are the key hyperparameters?
- **n_steps=2048**: Steps collected before each policy update
- **batch_size=256**: Mini-batch size for gradient updates
- **learning_rate=3e-4**: The "safe default" from the PPO paper
- **gamma=0.99**: Discount factor (high = long-term planning)
- **ent_coef=0.01**: Entropy bonus to encourage exploration
- **clip_range=0.2**: Prevents large policy jumps

These are standard PPO defaults. I didn't do hyperparameter tuning for this project — with more time, I could have used something like Optuna for systematic search.

---

## Results & Evaluation

### How good is the agent? What's the max possible score?
The theoretical max is 97 (filling the entire 10x10 grid minus the starting length of 3). Our agent averages about ~1.4 and peaks at ~7. There's a lot of room for improvement — the "things I'd improve" section covers what would help most.

### Why doesn't it score higher?
Several factors: (1) 1M training steps sounds like a lot, but is actually very modest — more training would help but would require more time / compute / some composite of both. (2) No curriculum learning — the agent struggles early because food is far away on a 10x10 grid. (3) No hyperparameter tuning — default values work but aren't 'optimized'. (4) Single environment — parallel envs would provide more diverse experience. The foundation is solid; performance would improve significantly with standard techniques.

### Is 1.4 average score good for 1M steps?
For this setup, yes. Consider that: the random baseline scores 0.1, the agent improved ~14x, reward went from -9.9 to +6.7, and the learning curves show clear upward trend. My research shows that RL environments typically require 10-100M steps to reach strong performance, but even at only 1M this is pretty cool.

### How would you improve performance with more time?
In priority order: (1) **Curriculum learning** — start on a 5x5 grid, gradually increase. (2) **Hyperparameter sweep** with Optuna (or similar). (3) **Multi-environment training** — 8-16 parallel envs via SubprocVecEnv. (4) **Longer training** — 10M+ steps.

### How do you evaluate an RL agent properly?
Ideally, run separate evaluation episodes with `deterministic=True` (no exploration noise) periodically during training. Our current metrics come from training episodes (which include exploration), so they're noisier. A proper eval harness would give cleaner curves. For the demo, we use deterministic evaluation.

---

## Software Engineering

### How is the code structured?
Six core Python files, ~1500 lines total:
- **snake_env.py** (~445 lines): Custom Gymnasium environment — game logic, state representation, rendering
- **train.py** (~320 lines): PPO setup, custom CNN, metrics callback, training loop
- **demo.py** (~150 lines): Pygame visualization, random vs trained comparison
- **plot_curves.py** (~140 lines): Learning curve plotting from metrics JSON
- **record_gif.py** (~230 lines): used in creating gif artifacts for presentations where live demo isn't possible / needs a backup
- **generate_visuals.py** (~170 lines): Used for creating other visuals for presentations.

### Why Python and not something faster?
Python is the standard for ML/RL. PyTorch, stable-baselines3, and Gymnasium are all popular and well supported Python libraries. The bottleneck is the neural network forward/backward pass (handled by PyTorch in C++/CUDA), not the environment step logic. For production scale, you might rewrite the environment in something computationally faster, but it's unnecessary here for a POC.

### How would this scale to a larger grid (e.g., 20x20)?
The environment generalizes directly — just change `--grid-size 20`. The SmallCNN architecture also generalizes (it computes `flat_size = 64 * grid_h * grid_w` dynamically). However, a larger grid needs (1) more training steps (larger state space), (2) possibly deeper CNN, and (3) curriculum learning would become essential.

### How would you deploy this?
Depends on the use case.  For a web demo: export the trained model, wrap it in a FastAPI endpoint, and build a simple JavaScript Snake frontend that calls the API for each move. The model is small (~20MB) so it's very deployable.  

Other considerations include: 

Containerization: 
— multi-stage docker build. Stage 1: training image (heavy, with PyTorch + CUDA). Stage 2: inference-only image (slim, just model + FastAPI)
- docker-compose — training service + API service + optional Pygame VNC for visual demo.  Useful for local dev


Orchestration:
- K8s — Deployment + Service for the inference API, Job/CronJob for retraining (OR)
- ECS — AWS-native. Task definition for inference, scheduled task for training
- Horizontal scaling — inference is stateless, scales trivially

CI/CD:
- GitHub Actions — train on push, upload model artifact, run eval, gate on minimum score
- Model registry — version models with DVC or MLflow, not just git

Monitoring:
- Prometheus/OTEL — inference latency, action distribution drift, avg episode score on eval runs
- Alerting — if avg eval score drops below threshold, something broke


### What testing would you add?
- **Environment unit tests**: verify step mechanics (collision detection, food placement, reward values, hunger timeout)
- **Observation shape tests**: ensure 3x10x10 output, values in [0,1]
- **Reward regression tests**: verify known scenarios produce expected rewards
- **Integration test**: short training run (1000 steps) to catch API/shape mismatches
- **Rendering test**: headless smoke test of the Pygame renderer

---

## Non-Technical Questions

### What was the hardest part?
Reward shaping calibration. The distance reward magnitude (+0.5/-0.5) required iteration — too high and the agent learns to oscillate near food without eating it because that's more 'valuable'.  Set it too low, and learning is painfully slow because the AI doesn't get strong enough feedback.  

Also, the oroboros / tail-chase collision bug was subtle and hard to diagnose — the agent's scores were mysteriously low, and even after finding a bug in collision detection, the root cause wasn't obvious.  

### How long did this take to build?
The core implementation (environment + training + demo + plotting) took a few hours. The debugging and tuning phase — finding the tail-chase bug, calibrating reward magnitudes, running multiple training experiments — took additional time. That's typical for RL projects: the code is straightforward, but getting the reward signal and environment tuning right is where the real work is.  

### Why Snake?

I was really inspired by Peter Whidden's PokemonRL project - for personal reasons as well as technological ones.   But we've got limited time, and I wanted something simpler to start with.  It also took me back to my youth playing on my hand-me-down blackberry, where I'd play snake in between texting on BBM.

It ended up being perfect.  Most folks know the rules, it has some depth (agent needs spatial reasoning, needs to plan ahead, etc).  It's visually compelling because you can see the agent's behavior in real time.   It is simple enough to not require frame-packing / nuanced novelty filters / etc (Which PokemonRL _did_ require).   

### Could this approach work for other games?
Absolutely. The PPO + custom environment pattern generalizes to any game with clear state/action/reward definitions. Pac-Man, Tetris, Minesweeper, simple platformers — same overall approach. The key engineering work is always the state representation and reward design, which is game-specific.
