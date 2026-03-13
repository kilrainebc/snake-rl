"""
generate_visuals.py: Generate presentation visuals from the Snake environment.

Creates:
  1. 3-channel observation heatmap (the most important visual in the deck)
  2. Game screenshot (for slide backgrounds)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from snake_env import SnakeEnv

def generate_observation_heatmap(output_path="output/observation_channels.png"):
    """
    Generate a 3-panel heatmap showing the 3 observation channels.

    This is the key visual for explaining state representation.
    """
    # Manually construct a snake state with a longer body for a clear gradient visual
    env = SnakeEnv(grid_size=10)
    env.reset()

    # Override snake to be longer with an interesting L-shaped path
    env.snake = [
        (3, 6),  # Head (brightest)
        (3, 5),
        (3, 4),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),  # Tail (dimmest)
    ]
    env.food = (1, 8)
    env.score = 5
    obs = env._get_observation()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor('#1a1a2e')

    channel_info = [
        ("Channel 0: Body Gradient", "Greens", "Head = 1.0, fading to tail"),
        ("Channel 1: Head Position", "Greens", "Binary mask (1.0 at head)"),
        ("Channel 2: Food Position", "Reds", "Binary mask (1.0 at food)"),
    ]

    for i, (title, cmap, subtitle) in enumerate(channel_info):
        ax = axes[i]
        data = obs[i]

        # Custom colormap: dark background for zeros
        if cmap == "Greens":
            colors_list = ['#1a1a2e', '#00b894', '#55efc4']
        else:
            colors_list = ['#1a1a2e', '#e17055', '#ff6b6b']

        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors_list, N=256)

        im = ax.imshow(data, cmap=custom_cmap, vmin=0, vmax=1, interpolation='nearest')

        # Draw grid lines
        for x in range(11):
            ax.axhline(x - 0.5, color='#2d2d4e', linewidth=0.5)
            ax.axvline(x - 0.5, color='#2d2d4e', linewidth=0.5)

        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(subtitle, color='#aaaaaa', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#1a1a2e')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.set_ylabel('Value', color='white', fontsize=9)
        plt.setp(cbar.ax.get_yticklabels(), color='white', fontsize=8)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Observation heatmap saved to {output_path}")


def generate_game_screenshot(output_path="output/game_screenshot.png"):
    """
    Generate a clean game-state image (no Pygame needed) for slide backgrounds.
    """
    from PIL import Image

    env = SnakeEnv(grid_size=10)
    env.reset()

    # Same L-shaped snake as the heatmap for visual consistency
    env.snake = [
        (3, 6),  # Head
        (3, 5),
        (3, 4),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),  # Tail
    ]
    env.food = (1, 8)
    env.score = 5

    cell_size = 60
    grid_size = 10
    w = grid_size * cell_size
    h = grid_size * cell_size

    img = Image.new("RGB", (w, h), (26, 26, 46))
    pixels = img.load()

    # Grid lines
    for i in range(grid_size + 1):
        for x in range(w):
            y = i * cell_size
            if 0 <= y < h:
                pixels[x, y] = (40, 40, 60)
        for y in range(h):
            x = i * cell_size
            if 0 <= x < w:
                pixels[x, y] = (40, 40, 60)

    # Food
    if env.food:
        fr, fc = env.food
        for y in range(fr * cell_size + 4, (fr + 1) * cell_size - 4):
            for x in range(fc * cell_size + 4, (fc + 1) * cell_size - 4):
                pixels[x, y] = (255, 107, 107)

    # Body
    for seg in env.snake[1:]:
        sr, sc = seg
        for y in range(sr * cell_size + 3, (sr + 1) * cell_size - 3):
            for x in range(sc * cell_size + 3, (sc + 1) * cell_size - 3):
                pixels[x, y] = (0, 184, 148)

    # Head
    if env.snake:
        hr, hc = env.snake[0]
        for y in range(hr * cell_size + 3, (hr + 1) * cell_size - 3):
            for x in range(hc * cell_size + 3, (hc + 1) * cell_size - 3):
                pixels[x, y] = (85, 239, 196)

    img.save(output_path)
    print(f"Game screenshot saved to {output_path}")

    # Also save a dimmed version for slide backgrounds
    dimmed = img.copy()
    dim_pixels = dimmed.load()
    for y in range(h):
        for x in range(w):
            r, g, b = dim_pixels[x, y]
            dim_pixels[x, y] = (r // 4, g // 4, b // 4)

    dimmed_path = output_path.replace(".png", "_dimmed.png")
    dimmed.save(dimmed_path)
    print(f"Dimmed background saved to {dimmed_path}")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    generate_observation_heatmap()
    generate_game_screenshot()
    print("\nDone! Visuals saved to output/")
