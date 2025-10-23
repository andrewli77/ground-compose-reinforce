import gymnasium as gym
from gymnasium import spaces
import numpy as np, torch
from PIL import Image, ImageDraw
import pygame

class GridWorldEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.grid_size = 8
        self.num_colours = 3  # Blue, Green, Red
        self.num_shapes = 2 # Circle, triangle

        # Action space: 0 = up, 1 = down, 2 = left, 3 = right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 8x8x2
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, self.num_colours + self.num_shapes + 1),
            dtype=np.float32
        )
        
        self.time_limit = 100

        self.grid_colours = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid_shapes = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.num_steps = 0

        self.all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

        self.metadata.setdefault("render_modes", ["rgb_array"])
        self.render_mode = 'rgb_array'

    def reset(self, *, seed=None, options=None):
        """Resets the environment to its initial state."""
        # Initialize grid with colours
        self.grid_colours = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid_shapes = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place 2 blue, 2 green, and 2 red cells randomly. Make sure there is one circle and one triangle for each colour
        # Initialize agent position randomly
        self.agent_pos = self.get_placements({1: 2, 2: 2, 3: 2})

        self.num_steps = 0

        return self._get_observation(), self._get_classifier_progress()

    def get_placements(self, colour_counts):
        """Places squares of specific values randomly on the grid without replacement."""
        self.key_squares = []
        
        np.random.shuffle(self.all_cells)
        
        index = 0
        for colour, count in colour_counts.items():
            for i in range(count):
                x, y = self.all_cells[index]
                self.grid_colours[x, y] = colour
                self.grid_shapes[x, y] = i+1
                self.key_squares.append((x,y))
                index += 1

        return self.all_cells[index]

    def _get_observation(self):
        """Returns the current observation as an 8x8x2 array."""
        state = np.zeros((self.grid_size, self.grid_size, self.num_colours + self.num_shapes + 1), dtype=np.float32)
        
        # First channel: agent position
        state[self.agent_pos[0], self.agent_pos[1], 0] = 1.

        # Next `self.num_colours` channels: one-hot encoding of grid colours
        # The following `self.num_shapes` channels: one-hot encoding of grid shapes
        for key_square in self.key_squares:
            assert(self.grid_colours[key_square] != 0 and self.grid_shapes[key_square] != 0)
            state[key_square[0], key_square[1], self.grid_colours[key_square]] = 1.
            state[key_square[0], key_square[1], self.num_colours + self.grid_shapes[key_square]] = 1.

        return state

    def step(self, action):
        """Performs a single step in the environment."""
        x, y = self.agent_pos

        # Update agent position based on action
        if action == 0 and x > 0:         # Up
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # Down
            x += 1
        elif action == 2 and y > 0:       # Left
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Right
            y += 1

        self.agent_pos = (x, y)
        self.num_steps += 1

        # Observation, reward, terminated, truncated, info
        return self._get_observation(), 0.0, False, self.num_steps == self.time_limit, self._get_classifier_progress()

    # Labels: [r,g,b,c,t]
    # Progresses: [r,g,b,c,t,!r,!g,!b,!c,!t] + [rc, rt, gc, gt, bc, bt, r&!t, g&!t, b&!t]
    def _get_classifier_progress(self):
        dists = np.array([0]*6) # [rc, rt, gc, gt, bc, bt]
        for pos in self.key_squares:
            d = abs(self.agent_pos[0] - pos[0]) + abs(self.agent_pos[1] - pos[1])
            idx = (self.grid_colours[pos] - 1) * 2 + self.grid_shapes[pos]-1
            dists[idx] = d


        labels = np.array([
            dists[0] == 0 or dists[1] == 0,
            dists[2] == 0 or dists[3] == 0,
            dists[4] == 0 or dists[5] == 0,
            dists[0] == 0 or dists[2] == 0 or dists[4] == 0,
            dists[1] == 0 or dists[3] == 0 or dists[5] == 0,
        ], dtype=np.uint8)

        dists_all = np.array([
            min(dists[0], dists[1]),
            min(dists[2], dists[3]),
            min(dists[4], dists[5]),
            min(dists[0], dists[2], dists[4]),
            min(dists[1], dists[3], dists[5]),
            labels[0],
            labels[1],
            labels[2],
            labels[3],
            labels[4],
            dists[0],
            dists[1],
            dists[2],
            dists[3],
            dists[4],
            dists[5],
            dists[0],
            dists[2],
            dists[4],
        ], dtype=np.uint8)

        return {"labels": labels, "dists": dists_all}

    def render(self, mode='rgb_array'):
        return np.array(self.get_image())

    def get_image(self):
        # Constants
        grid_size = 8
        square_size = 50  # Size of each square in the grid (pixels)
        shape_size = 30  # Size of the shape inside the square (pixels)

        # Define colors
        color_map = {0: (255, 255, 255),  # white for empty
                     1: (255, 0, 0),      # red
                     2: (0, 255, 0),      # green
                     3: (0, 0, 255)}      # blue

        shape_map = {0: None,  # No shape
                     1: 'circle',  # Circle
                     2: 'triangle'}  # Triangle


        # Create a white background image
        img = Image.new('RGB', (grid_size * square_size + 100, grid_size * square_size + 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Function to draw a shape
        def draw_shape(x, y, shape_type, color):
            center_x, center_y = (50 + x * square_size + square_size // 2, 50 + y * square_size + square_size // 2)
            
            if shape_type == 'circle':
                draw.ellipse([center_x - shape_size // 2, center_y - shape_size // 2,
                              center_x + shape_size // 2, center_y + shape_size // 2], 
                             fill=color)
            elif shape_type == 'triangle':
                triangle_points = [
                    (center_x, center_y - shape_size // 2),  # top
                    (center_x - shape_size // 2, center_y + shape_size // 2),  # bottom left
                    (center_x + shape_size // 2, center_y + shape_size // 2)   # bottom right
                ]
                draw.polygon(triangle_points, fill=color)

        # Draw the grid and shapes
        for y in range(grid_size):
            for x in range(grid_size):
                draw.rectangle([50 + x * square_size, 50 + y * square_size, 
                                50 + (x + 1) * square_size, 50 + (y + 1) * square_size], 
                               fill=(255, 255, 255))

        # Highlight the agent's position with an orange border
        agent_y, agent_x = self.agent_pos
        draw.rectangle([50 + agent_x * square_size, 50 + agent_y * square_size, 
                        50 + (agent_x + 1) * square_size, 50 + (agent_y + 1) * square_size],
                       fill=(255, 165, 0))  # Orange color with thicker border

        for y in range(grid_size):
            for x in range(grid_size):
                # Get shape and color for the current square
                color_idx = self.grid_colours[y, x]
                shape_idx = self.grid_shapes[y, x]
                
                color = color_map[color_idx]
                shape_type = shape_map[shape_idx]
                
                # Draw the shape if it is not empty
                if shape_type is not None:
                    draw_shape(x, y, shape_type, color)
    

        # Draw grid lines (light grey) after everything else to avoid overwriting
        for y in range(grid_size + 1):
            # Horizontal grid lines
            draw.line([(50, 50 + y * square_size), (50 + grid_size * square_size, 50 + y * square_size)], fill=(0, 0, 0), width=2)
        for x in range(grid_size + 1):
            # Vertical grid lines
            draw.line([(50 + x * square_size, 50), (50 + x * square_size, 50 + grid_size * square_size)], fill=(0, 0, 0), width=2)

        return img

class InteractiveGUI:
    def __init__(self, env, foundation_model, PRINT_GROUND_TRUTH_LABELS=False, PRINT_PREDICTED_LABELS=False, PRINT_PREDICTED_PROGRESS=False, PRINT_PREDICTED_RM_INFO=False):
        self.env = env
        self.foundation_model = foundation_model

        self.PRINT_GROUND_TRUTH_LABELS = PRINT_GROUND_TRUTH_LABELS
        self.PRINT_PREDICTED_LABELS = PRINT_PREDICTED_LABELS
        self.PRINT_PREDICTED_PROGRESS = PRINT_PREDICTED_PROGRESS
        self.PRINT_PREDICTED_RM_INFO = PRINT_PREDICTED_RM_INFO

        # Initialize pygame
        pygame.init()

        # Get the environment image size
        self.square_size = 50  # Assuming square size is 50 for rendering
        self.img_width = self.env.unwrapped.grid_colours.shape[1] * self.square_size + 100
        self.img_height = self.env.unwrapped.grid_colours.shape[0] * self.square_size + 100
        
        # Set up the pygame display
        self.screen = pygame.display.set_mode((self.img_width, self.img_height))
        pygame.display.set_caption('Gridworld Environment')

    def update_screen(self):
        # Get the image from the environment
        img = self.env.get_image()

        # Convert the PIL image to a Pygame surface
        img = np.array(img)
        img = np.transpose(img, (1, 0, 2))  # Convert from (height, width, channels) to (width, height, channels)
        pygame_img = pygame.surfarray.make_surface(img)

        # Update the screen with the new image
        self.screen.blit(pygame_img, (0, 0))
        pygame.display.flip()


    def prettyprint(self, x):
        output = ""
        for each in x:
            output += "%.2f "%(each)
        return output

    def run(self):
        # Main loop to handle events and keyboard input
        obs, info = self.env.reset()
        print("RM state:", self.env.current_agent_u_id, "RM potential:", self.env.current_potential)

        classifications, progresses = self.foundation_model.query(torch.tensor(obs['observation']).unsqueeze(0), info=info)
        self.env.initialize_potential(classifications[0], progresses[0])

        running = True
        while running:
            self.update_screen()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Handle key presses for actions
                    if event.key == pygame.K_w:
                        action = 0
                    elif event.key == pygame.K_a:
                        action = 2
                    elif event.key == pygame.K_s:
                        action = 1
                    elif event.key == pygame.K_d:
                        action = 3
                    else:
                        raise ValueError("Key not valid.")

                    obs, _, terminated, truncated, info = self.env.step(action)
                    #print("RM state:", self.env.current_agent_u_id, "RM potential:", self.env.current_potential)

                    classifications, progresses = self.foundation_model.query(torch.tensor(obs['observation']).unsqueeze(0), info=info)
                    rm_reward, potential_reward, rm_done = self.env.sync(classifications[0], progresses[0])

                    if self.PRINT_GROUND_TRUTH_LABELS:
                        print("Ground truth labels:", info['labels'])
                    if self.PRINT_PREDICTED_LABELS:
                        print("Predicted labels:", self.prettyprint(classifications))
                    if self.PRINT_PREDICTED_PROGRESS:
                        print("Predicted progress:", self.prettyprint(progresses))
                    if self.PRINT_PREDICTED_RM_INFO:
                        print("Reward machine state:", obs['rm_state'], "RM potential:", torch.tensor(self.env.current_potential).item())

                    if terminated or truncated or rm_done:
                        running = False

        # Quit pygame when done
        pygame.quit()

import numpy as np
import torch
import pygame

class ActionGUI:
    """
    Minimal programmatic GUI for your rollout loop.

    API expected by caller:
      - gui = ActionGUI(env, foundation_model, PRINT_*, ...)
      - o, info = gui.reset()
      - o, done = gui.step(action_numpy)

    Notes:
      • No keyboard handling.
      • Adds a small delay after each step for visibility.
    """
    def __init__(
        self,
        env,
        foundation_model,
        PRINT_GROUND_TRUTH_LABELS=False,
        PRINT_PREDICTED_LABELS=False,
        PRINT_PREDICTED_PROGRESS=False,
        PRINT_PREDICTED_RM_INFO=False,
        step_delay_ms=300,  # <-- slow down playback (adjust as desired)
    ):
        self.env = env
        self.foundation_model = foundation_model

        self.PRINT_GROUND_TRUTH_LABELS = PRINT_GROUND_TRUTH_LABELS
        self.PRINT_PREDICTED_LABELS = PRINT_PREDICTED_LABELS
        self.PRINT_PREDICTED_PROGRESS = PRINT_PREDICTED_PROGRESS
        self.PRINT_PREDICTED_RM_INFO = PRINT_PREDICTED_RM_INFO

        self.step_delay_ms = int(step_delay_ms)

        # Init pygame + window sized to the env image
        pygame.init()
        # Determine H,W from unwrapped grid (works through wrappers)
        H, W = self.env.unwrapped.grid_colours.shape
        self.square_size = 50
        self.img_width = W * self.square_size + 100
        self.img_height = H * self.square_size + 100
        self.screen = pygame.display.set_mode((self.img_width, self.img_height))
        pygame.display.set_caption("Gridworld Viewer")

    # ---------- internal helpers ----------

    def _update_screen(self):
        """Render current env frame to the pygame window."""
        img = self.env.get_image()           # PIL Image
        arr = np.array(img)                  # H x W x 3
        arr = np.transpose(arr, (1, 0, 2))   # to W x H x 3 for pygame
        surf = pygame.surfarray.make_surface(arr)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def _obs_tensor(self, obs):
        """Return the raw observation array for the VLM/foundation model."""
        if isinstance(obs, dict) and "observation" in obs:
            data = obs["observation"]
        else:
            data = obs
        return torch.as_tensor(data).unsqueeze(0)

    def _pretty(self, x):
        try:
            return " ".join(f"{float(v):.2f}" for v in x)
        except Exception:
            return str(x)

    # ---------- public API used by your script ----------

    def reset(self):
        """
        Resets the env, initializes RM via foundation_model if available,
        draws the first frame, and returns (obs, info) untouched for your pipeline.
        """
        obs, info = self.env.reset()

        # Optional RM bootstrap via foundation model
        if self.foundation_model is not None:
            x = self._obs_tensor(obs)
            classifications, progresses = self.foundation_model.query(x, info=info)

            if hasattr(self.env, "initialize_potential"):
                self.env.initialize_potential(classifications[0], progresses[0])

            if self.PRINT_PREDICTED_LABELS:
                print("Predicted labels:", self._pretty(classifications))
            if self.PRINT_PREDICTED_PROGRESS:
                print("Predicted progress:", self._pretty(progresses))

        if self.PRINT_GROUND_TRUTH_LABELS and isinstance(info, dict) and "labels" in info:
            print("Ground truth labels:", info["labels"])
        if self.PRINT_PREDICTED_RM_INFO and hasattr(self.env, "current_potential"):
            try:
                print("RM potential:", torch.tensor(self.env.current_potential).item())
            except Exception:
                print("RM potential:", self.env.current_potential)

        # Initial draw
        self._update_screen()
        # Small delay so the first frame is visible
        if self.step_delay_ms > 0:
            pygame.time.delay(self.step_delay_ms)

        return obs, info

    def step(self, action):
        """
        Steps the env with a numpy/int action, updates the screen,
        optionally syncs RM with foundation_model, and returns (obs, done).
        """
        # Allow array/tensor and scalar
        if isinstance(action, (np.ndarray, torch.Tensor)):
            action = int(np.array(action).item())
        else:
            action = int(action)

        # Keep window responsive (handle close events quietly)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # We still proceed; caller controls loop termination via done
                pass

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Optional RM sync per step
        rm_done = False
        if self.foundation_model is not None:
            x = self._obs_tensor(obs)
            classifications, progresses = self.foundation_model.query(x, info=info)

            if self.PRINT_PREDICTED_LABELS:
                print("Predicted labels:", self._pretty(classifications))
            if self.PRINT_PREDICTED_PROGRESS:
                print("Predicted progress:", self._pretty(progresses))

            if hasattr(self.env, "sync"):
                _, _, rm_done = self.env.sync(classifications[0], progresses[0])

        if self.PRINT_GROUND_TRUTH_LABELS and isinstance(info, dict) and "labels" in info:
            print("Ground truth labels:", info["labels"])
        if self.PRINT_PREDICTED_RM_INFO and hasattr(self.env, "current_potential"):
            try:
                print("RM potential:", torch.tensor(self.env.current_potential).item())
            except Exception:
                print("RM potential:", self.env.current_potential)

        # Draw and slow down to make it easy to watch
        self._update_screen()
        if self.step_delay_ms > 0:
            pygame.time.delay(self.step_delay_ms)

        done = bool(terminated) or bool(truncated) or bool(rm_done)
        return obs, done
