"""
record_gif.py: Record Snake gameplay as an animated GIF.

Creates GIF demos of random and trained agents for use as backup
presentation assets (in case Pygame won't run on the presentation machine).
"""

import argparse
import os
import numpy as np
from PIL import Image
from snake_env import SnakeEnv


# Colors (RGB)
BG_COLOR = (26, 26, 46)
GRID_COLOR = (40, 40, 60)
BODY_COLOR = (0, 184, 148)
HEAD_COLOR = (85, 239, 196)
FOOD_COLOR = (255, 107, 107)
SCORE_BG = (15, 15, 30)
WHITE = (255, 255, 255)


def render_frame_to_image(env, cell_size=40):
    """Render the current game state to a PIL Image."""
    grid_size = env.grid_size
    window_w = grid_size * cell_size
    window_h = grid_size * cell_size + 40  # extra for score bar

    img = Image.new("RGB", (window_w, window_h), BG_COLOR)
    pixels = img.load()

    # Draw grid lines
    for i in range(grid_size + 1):
        for x in range(window_w):
            y = i * cell_size
            if 0 <= y < window_h:
                pixels[x, y] = GRID_COLOR
        for y in range(grid_size * cell_size):
            x = i * cell_size
            if 0 <= x < window_w:
                pixels[x, y] = GRID_COLOR

    # Draw food
    if env.food is not None:
        fr, fc = env.food
        for y in range(fr * cell_size + 3, (fr + 1) * cell_size - 3):
            for x in range(fc * cell_size + 3, (fc + 1) * cell_size - 3):
                pixels[x, y] = FOOD_COLOR

    # Draw snake body
    for segment in env.snake[1:]:
        sr, sc = segment
        for y in range(sr * cell_size + 2, (sr + 1) * cell_size - 2):
            for x in range(sc * cell_size + 2, (sc + 1) * cell_size - 2):
                pixels[x, y] = BODY_COLOR

    # Draw snake head
    if env.snake:
        hr, hc = env.snake[0]
        for y in range(hr * cell_size + 2, (hr + 1) * cell_size - 2):
            for x in range(hc * cell_size + 2, (hc + 1) * cell_size - 2):
                pixels[x, y] = HEAD_COLOR

    # Score bar background
    for y in range(grid_size * cell_size, window_h):
        for x in range(window_w):
            pixels[x, y] = SCORE_BG

    return img


def record_episode(env, model=None, max_steps=500):
    """Record one episode as a list of PIL Images."""
    obs, info = env.reset()
    frames = [render_frame_to_image(env)]
    done = False
    steps = 0

    while not done and steps < max_steps:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(render_frame_to_image(env))
        done = terminated or truncated
        steps += 1

    return frames, info


def add_label(img, text, position="top"):
    """Add a simple text label to the image using basic pixel drawing."""
    # We'll use PIL's ImageDraw for text
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    if position == "top":
        # Draw label bar at top
        draw.rectangle([(0, 0), (img.width, 30)], fill=(15, 15, 30))
        draw.text((10, 5), text, fill=WHITE, font=font)
    return img


def main():
    parser = argparse.ArgumentParser(description="Record Snake gameplay as GIF.")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save GIFs.")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Episodes per GIF.")
    parser.add_argument("--fps", type=int, default=8,
                        help="Frames per second in the GIF.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = SnakeEnv(grid_size=10)
    frame_duration = int(1000 / args.fps)

    # --- Record random agent ---
    print("Recording random agent...")
    random_frames = []
    for ep in range(args.episodes):
        frames, info = record_episode(env)
        # Add label to each frame
        labeled = []
        for f in frames:
            # Add top bar with label
            from PIL import Image as PILImage
            new_img = PILImage.new("RGB", (f.width, f.height + 30), (15, 15, 30))
            new_img.paste(f, (0, 30))
            add_label(new_img, f"Random Agent  |  Episode {ep+1}  |  Score: {info['score']}")
            labeled.append(new_img)
        random_frames.extend(labeled)
        # Add pause frames between episodes
        if ep < args.episodes - 1:
            random_frames.extend([labeled[-1]] * (args.fps))  # 1 second pause

    random_path = os.path.join(args.output_dir, "demo_random.gif")
    random_frames[0].save(
        random_path,
        save_all=True,
        append_images=random_frames[1:],
        duration=frame_duration,
        loop=0,
    )
    print(f"  Saved {random_path} ({len(random_frames)} frames)")

    # --- Record trained agent ---
    print("Recording trained agent...")
    model_path = os.path.join(args.output_dir, "snake_ppo.zip")
    if not os.path.exists(model_path):
        print(f"  Model not found at {model_path}, skipping trained agent GIF.")
        return

    from stable_baselines3 import PPO
    model = PPO.load(model_path)

    trained_frames = []
    for ep in range(args.episodes):
        frames, info = record_episode(env, model=model)
        labeled = []
        for f in frames:
            from PIL import Image as PILImage
            new_img = PILImage.new("RGB", (f.width, f.height + 30), (15, 15, 30))
            new_img.paste(f, (0, 30))
            add_label(new_img, f"Trained Agent  |  Episode {ep+1}  |  Score: {info['score']}")
            labeled.append(new_img)
        trained_frames.extend(labeled)
        if ep < args.episodes - 1:
            trained_frames.extend([labeled[-1]] * (args.fps))

    trained_path = os.path.join(args.output_dir, "demo_trained.gif")
    trained_frames[0].save(
        trained_path,
        save_all=True,
        append_images=trained_frames[1:],
        duration=frame_duration,
        loop=0,
    )
    print(f"  Saved {trained_path} ({len(trained_frames)} frames)")

    # --- Combined side-by-side comparison ---
    # Take best random + best trained episode for a compact comparison
    print("Recording comparison GIF...")
    env_r = SnakeEnv(grid_size=10)
    env_t = SnakeEnv(grid_size=10)

    r_frames, r_info = record_episode(env_r, model=None)
    t_frames, t_info = record_episode(env_t, model=model)

    # Pad shorter sequence
    max_len = max(len(r_frames), len(t_frames))
    while len(r_frames) < max_len:
        r_frames.append(r_frames[-1])
    while len(t_frames) < max_len:
        t_frames.append(t_frames[-1])

    comparison_frames = []
    for i in range(max_len):
        rf = r_frames[i]
        tf = t_frames[i]
        # Side by side with gap
        gap = 20
        combined = Image.new("RGB", (rf.width + tf.width + gap, rf.height + 30), (15, 15, 30))
        combined.paste(rf, (0, 30))
        combined.paste(tf, (rf.width + gap, 30))
        add_label(combined, f"Random (score: {r_info['score']})                    Trained (score: {t_info['score']})")
        comparison_frames.append(combined)

    comparison_path = os.path.join(args.output_dir, "demo_comparison.gif")
    comparison_frames[0].save(
        comparison_path,
        save_all=True,
        append_images=comparison_frames[1:],
        duration=frame_duration,
        loop=0,
    )
    print(f"  Saved {comparison_path} ({len(comparison_frames)} frames)")

    env.close()
    print("\nDone! GIFs saved to output/")


if __name__ == "__main__":
    main()
