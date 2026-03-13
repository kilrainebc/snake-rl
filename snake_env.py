"""
snake_env.py — Custom Gymnasium environment for Snake.

Design Decisions:
    State Representation (3-channel 10x10 float32 grid):

        We use a spatial grid rather than a flat feature vector because the Snake
        game is inherently spatial — the relationship between the head, body segments,
        and food is best captured by preserving 2D structure. This lets us use a CNN
        policy, which excels at learning spatial patterns like "food is to the upper-left"
        or "my tail blocks the path ahead."

        Channel 0 — Snake body with gradient values (head=1.0, tail fades toward 0):
            A binary body mask loses directional information. By encoding a gradient
            from head (1.0) to tail (approaching 0), the CNN can infer which direction
            the snake is moving and which end is which. This is critical for learning
            to avoid self-collision — the agent needs to know where the head is relative
            to the body.

        Channel 1 — Head position (binary):
            Redundant with the brightest cell in channel 0, but makes the head position
            trivially easy for the CNN to extract. Reducing the burden on the network
            to learn "find the max value" speeds up training.

        Channel 2 — Food position (binary):
            Isolated in its own channel so the CNN can learn food-seeking behavior
            without interference from body information.

    Reward Shaping:
        +10 for eating food: Large positive signal for the primary objective.
        -10 for death: Strong negative signal to avoid walls and self-collision.
        +0.5/-0.5 for moving closer/farther from food: Provides a dense gradient
            toward the food. Without this, the agent gets almost no signal until it
            randomly stumbles onto food, making early training very slow. Tuned to
            balance signal strength vs incentive alignment: at +1/-1, cumulative
            approach rewards (~18 on 10x10) rival the +10 food reward and incentivize
            oscillation; at +0.1, learning is too slow. At +0.5, max cumulative
            approach (~9) is strong but the food reward clearly dominates.
        -0.01 per step: Tiny penalty to discourage stalling and encourage efficiency.
            Small enough not to overwhelm the distance-based shaping.

    Hunger Limit:
        100 + (length * 10) steps without eating triggers death. This prevents the
        agent from learning to circle endlessly (which the distance reward alone
        might encourage if the agent oscillates near the food). The limit scales
        with length because longer snakes need more room to maneuver.

    180-Degree Turn Blocking:
        Pressing the opposite of the current direction is ignored (the snake continues
        straight). 
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Pygame is imported in _render_frame() to avoid requiring it
# when the environment is used without rendering (e.g., during training).
pygame = None

# Action constants for readability
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Maps action index to (row_delta, col_delta)
DIRECTION_VECTORS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# Pairs of opposite actions — used to block 180-degree turns
OPPOSITE_ACTIONS = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

class SnakeEnv(gym.Env):
    """
    A custom Gymnasium environment implementing the Snake game.

    The snake moves on a grid, eats food to grow, and dies if it hits a wall/
    itself, or runs out of steps without eating (hunger timeout).

    Args:
        grid_size: Side length of the square grid (default 10).
        render_mode: None for no rendering, "human" for Pygame window.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        # 3 channels: body gradient, head position, food position
        # Each channel is a grid_size x grid_size float32 array
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, grid_size, grid_size),
            dtype=np.float32,
        )

        # 4 discrete actions: UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)

        # Pygame rendering state (initialized lazily)
        self.window = None
        self.clock = None
        self.cell_size = 40  # Pixels per grid cell

        # Game state — initialized in reset()
        self.snake = []
        self.direction = RIGHT
        self.food = None
        self.score = 0
        self.steps_since_food = 0
        self.total_steps = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.

        The snake starts at length 3 in the center of the grid, facing right.
        Food is placed randomly on an empty cell.

        Returns:
            observation: The initial 3-channel grid observation.
            info: Dict with initial game state metadata.
        """
        super().reset(seed=seed)

        # Place snake in center, facing right, length 3
        # snake[0] is the head, snake[-1] is the tail
        center_row = self.grid_size // 2
        center_col = self.grid_size // 2
        self.snake = [
            (center_row, center_col),      # Head
            (center_row, center_col - 1),  # Body
            (center_row, center_col - 2),  # Tail
        ]
        self.direction = RIGHT

        # Place food on a random empty cell
        self.food = self._place_food()

        # Reset counters
        self.score = 0
        self.steps_since_food = 0
        self.total_steps = 0

        observation = self._get_observation()
        info = {"score": self.score, "snake_length": len(self.snake)}

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Integer in {0, 1, 2, 3} corresponding to UP/DOWN/LEFT/RIGHT.

        Returns:
            observation: Updated 3-channel grid.
            reward: Float reward for this step.
            terminated: True if the snake died.
            truncated: Always False (we handle timeout via terminated).
            info: Dict with game state metadata.
        """
        # Convert numpy arrays/ints to Python int (SB3 predict() returns numpy)
        action = int(action)

        # Block 180-degree turns: if the action is opposite to current direction,
        # ignore it and continue in the current direction
        if action != OPPOSITE_ACTIONS.get(self.direction, None):
            self.direction = action

        # Compute new head position
        dr, dc = DIRECTION_VECTORS[self.direction]
        head_row, head_col = self.snake[0]
        new_head = (head_row + dr, head_col + dc)

        # Check for death conditions
        terminated = False
        reward = -0.01  # Small per-step penalty to discourage stalling

        # Wall collision: head moves outside the grid
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
        ):
            terminated = True
            reward = -10.0
            # Return current observation (before the invalid move)
            observation = self._get_observation()
            info = self._get_info()
            if self.render_mode == "human":
                self._render_frame()
            return observation, reward, terminated, False, info

        # Self collision: head moves into a body segment.
        # We exclude the tail (snake[-1]) from the check because the tail will
        # vacate its cell this step (unless food is eaten). If food IS eaten,
        # the head lands on the food cell — which is guaranteed to be empty
        # (food is never placed on the snake). 
        if new_head in self.snake[:-1]:
            terminated = True
            reward = -10.0
            observation = self._get_observation()
            info = self._get_info()
            if self.render_mode == "human":
                self._render_frame()
            return observation, reward, terminated, False, info

        # Calculate "Manhattan" distance before and after the move (for reward shaping)
        old_distance = self._manhattan_distance(self.snake[0], self.food)
        new_distance = self._manhattan_distance(new_head, self.food)

        # Move the snake: insert new head at the front
        self.snake.insert(0, new_head)

        # Check if food was eaten
        if new_head == self.food:
            # Food eaten: don't remove tail (snake grows), place new food
            self.score += 1
            self.steps_since_food = 0
            self.food = self._place_food()
            reward += 10.0  # Large positive reward for eating
        else:
            # No food eaten: remove tail (snake moves without growing)
            self.snake.pop()
            self.steps_since_food += 1

            # Distance-based reward shaping:
            # +0.5 for getting closer to food, -0.5 for moving farther away.
            # Tuned to balance signal strength vs incentive alignment:
            # - Too small (0.1): weak gradient, very slow early learning.
            # - Too large (1.0): cumulative approach reward (~18 on 10x10)
            #   rivals the +10 food reward, incentivizing oscillation.
            # - At 0.5: max cumulative approach is ~9, strong enough for fast
            #   learning but the +10 food reward clearly dominates.
            if new_distance < old_distance:
                reward += 0.5
            elif new_distance > old_distance:
                reward -= 0.5

        self.total_steps += 1

        # Hunger timeout: die if too many steps without eating
        hunger_limit = 100 + len(self.snake) * 10
        if self.steps_since_food >= hunger_limit:
            terminated = True
            reward = -10.0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _get_observation(self):
        """
        Build the 3-channel observation grid.

        Channel 0: Snake body with gradient (head=1.0, fading to ~0 at tail).
        Channel 1: Head position (1.0 at head cell, 0.0 elsewhere).
        Channel 2: Food position (1.0 at food cell, 0.0 elsewhere).
        """
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Body gradient
        # The head gets value 1.0, and each subsequent segment gets a linearly
        # decreasing value. This encodes direction and length information.
        num_segments = len(self.snake)
        for i, (row, col) in enumerate(self.snake):
            # i=0 is head (value=1.0), i=num_segments-1 is tail (approaches 0)
            if num_segments > 1:
                obs[0, row, col] = 1.0 - (i / num_segments)
            else:
                obs[0, row, col] = 1.0

        # Channel 1: Head position (binary)
        head_row, head_col = self.snake[0]
        obs[1, head_row, head_col] = 1.0

        # Channel 2: Food position (binary)
        if self.food is not None:
            food_row, food_col = self.food
            obs[2, food_row, food_col] = 1.0

        return obs

    def _get_info(self):
        """Return metadata about the current game state."""
        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps_since_food": self.steps_since_food,
            "total_steps": self.total_steps,
        }

    def _place_food(self):
        """
        Place food on a random empty cell (not occupied by the snake).

        Uses the PRNG from gymnasium's reset(seed=...) for reproducibility.
        """
        # Collect all empty cells
        snake_set = set(self.snake)
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in snake_set
        ]

        if not empty_cells:
            # Snake fills the entire grid — game is won (extremely rare)
            return None

        # Use numpy's random generator (seeded via gymnasium) for reproducibility
        idx = self.np_random.integers(0, len(empty_cells))
        return empty_cells[idx]

    @staticmethod
    def _manhattan_distance(pos_a, pos_b):
        """Compute "Manhattan" distance between two grid positions.
        Further reading: https://en.wikipedia.org/wiki/Taxicab_geometry 
        """
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])

    def render(self):
        """Render the current frame (called automatically if render_mode='human')."""
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        """
        Draw the game state using Pygame.

        Colors:
            Background: dark gray (#1a1a2e) — easy on the eyes
            Snake body: green (#00b894)
            Snake head: bright green (#55efc4) — distinguishes head from body
            Food: red (#ff6b6b)
            Score text: white
        """
        global pygame
        if pygame is None:
            import pygame as _pygame
            pygame = _pygame

        if self.window is None:
            pygame.init()
            pygame.display.init()
            window_size = self.grid_size * self.cell_size
            self.window = pygame.display.set_mode((window_size, window_size + 40))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 24)

        # Process Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        window_size = self.grid_size * self.cell_size

        # Dark background
        self.window.fill((26, 26, 46))

        # Draw grid lines (subtle)
        for i in range(self.grid_size + 1):
            # Horizontal lines
            pygame.draw.line(
                self.window, (40, 40, 60),
                (0, i * self.cell_size), (window_size, i * self.cell_size)
            )
            # Vertical lines
            pygame.draw.line(
                self.window, (40, 40, 60),
                (i * self.cell_size, 0), (i * self.cell_size, window_size)
            )

        # Draw food
        if self.food is not None:
            food_rect = pygame.Rect(
                self.food[1] * self.cell_size + 2,
                self.food[0] * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4,
            )
            pygame.draw.rect(self.window, (255, 107, 107), food_rect)

        # Draw snake body (skip head, draw it separately)
        for segment in self.snake[1:]:
            seg_rect = pygame.Rect(
                segment[1] * self.cell_size + 1,
                segment[0] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2,
            )
            pygame.draw.rect(self.window, (0, 184, 148), seg_rect)

        # Draw snake head (brighter green)
        if self.snake:
            head = self.snake[0]
            head_rect = pygame.Rect(
                head[1] * self.cell_size + 1,
                head[0] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2,
            )
            pygame.draw.rect(self.window, (85, 239, 196), head_rect)

        # Draw score bar at the bottom
        score_bar = pygame.Rect(0, window_size, window_size, 40)
        pygame.draw.rect(self.window, (15, 15, 30), score_bar)
        score_text = self.font.render(
            f"Score: {self.score}  Length: {len(self.snake)}", True, (255, 255, 255)
        )
        self.window.blit(score_text, (10, window_size + 8))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Clean up Pygame resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
