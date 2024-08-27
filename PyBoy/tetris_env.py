import imageio
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from io import BytesIO

actions = ['', 'a', 'left', 'right', 'down']
matrix_shape = (18, 10)  # Update this to the actual shape returned by pyboy.game_area()
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class TetrisEnv(gym.Env):
    def __init__(self, rom_path, save_state_path=None, debug=False):
        super().__init__()
        self.pyboy = PyBoy(rom_path)
        self.pyboy.set_emulation_speed(0)
        self.rotation_count = 0
        self.left_move_count = 0
        self.right_move_count = 0
        self.max_allowed_rotations = 4 
        self.previous_total_lines = 0

        self.tetris = self.pyboy.game_wrapper
        self.tetris.game_area_mapping(self.tetris.mapping_compressed, 0)
        self.tetris.start_game(timer_div=0x00)
        self.pyboy.tick()
        self.pyboy.screen.image.save("Tetris1.png")
        self.save_state_buffer = BytesIO()

        assert self.pyboy.cartridge_title == "TETRIS", "Loaded ROM is not Tetris"

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space

        self.debug = debug
        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.pyboy.screen.image.save("Tetris1.png")
        self._initialize_save_state()
        self.reset()

        # Initialize variable to track the previous tetromino
        self.previous_tetromino = self.tetris.next_tetromino()

    def _initialize_save_state(self):
        self.pyboy.save_state(self.save_state_buffer)
        self.save_state_buffer.seek(0)

    def reset(self, seed=None):
        self.seed = seed
        
        try:
            self.save_state_buffer.seek(0)
            self.pyboy.load_state(self.save_state_buffer)
        except Exception as e:
            print(f"Error loading save state: {e}")

        self._fitness = 0
        self._previous_fitness = 0
        self.previous_gap_count = self.count_gaps(self.pyboy.game_area())

        try:
            observation = self.pyboy.game_area()
            if len(observation.shape) == 2:
                if observation.shape != matrix_shape:
                    raise ValueError(f"Observation shape mismatch on reset: expected {matrix_shape}, got {observation.shape}")
            else:
                raise ValueError(f"Unexpected observation shape: {observation.shape}")
        except Exception as e:
            print(f"Error in getting game area: {e}")
            observation = np.zeros(matrix_shape, dtype=np.uint8)

        self.previous_tetromino = self.tetris.next_tetromino()

        info = {}
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        # Execute the action if it's not a no-op
        if action != 0:
            self.pyboy.button(actions[action])
        
        self.pyboy.tick(10)  # Progress the game

        piece_dropped = self._check_piece_dropped()  # Check if a piece was placed

        reward = 0  # Initialize reward to 0

        if piece_dropped:
            # Piece has been placed; calculate rewards here
            self._calculate_fitness()  # Update fitness based on the current game state
            reward += self._fitness - self._previous_fitness  # Reward based on score difference
            self._previous_fitness = self._fitness

            current_game_area = self.tetris.game_area()
            initial_gaps = self.count_gaps(self.tetris.game_area())
            initial_fill = self.calculate_line_fill(self.tetris.game_area())
            current_gaps = self.count_gaps(current_game_area)

            # Calculate rewards/penalties only after the piece is placed
            reward += self.penalize_gaps(initial_gaps, current_gaps)
            # reward += self.penalize_height(current_game_area)
            reward += self.reward_for_lines_cleared()

            self.reset_move_counts()  # Reset rotation and movement counters
        else:
            # No piece placed; potential to penalize rotations or excessive moves
            reward += self.penalize_rotation(action)
            reward += self.penalize_indecision(action)

        done = self.tetris.game_over()  # Check if the game is over
        observation = self.pyboy.game_area()  # Get the current state

        if observation.shape != matrix_shape:
            raise ValueError(f"Observation shape mismatch: expected {matrix_shape}, got {observation.shape}")

        info = {}
        truncated = False
        
        return observation, reward, done, truncated, info

    def reward_for_low_height(self, game_area):
        # Calculate the maximum height in each column
        max_height = np.max(np.where(game_area != 0, np.arange(game_area.shape[0])[:, None], 0), axis=0)
        
        # Calculate the average height of the stack
        average_height = np.mean(max_height)
        # print(average_height)
        # Define a threshold height below which we reward the agent
        reward_threshold = 10
        
        # Reward strategy: higher reward for lower average height
        if average_height < reward_threshold:
            height_reward = (reward_threshold - average_height) * 2  # Reward scales with how low the average height is
        else:
            height_reward = -(average_height - reward_threshold)  # Penalty if average height exceeds the threshold
        
        # Additional penalty if any column exceeds a dangerous height
        if np.any(max_height > 15):
            height_reward -= 10.0
        
        return height_reward

    def penalize_rotation(self, action):
        penalty = 0.0
        if action == 1:
            self.rotation_count += 1
        if self.rotation_count > self.max_allowed_rotations:
            penalty -= 3.0
            #print("penalize_rotation")
        return penalty

    def penalize_indecision(self, action):
        penalty = 0.0
        if action == 2:
            self.left_move_count += 1
        elif action == 3:
            self.right_move_count += 1
        if self.left_move_count > 2 and self.right_move_count > 2:
            penalty -= 3.0
            #print("penalize_indecision")
        return penalty

    def reset_move_counts(self):
        self.rotation_count = 0
        self.left_move_count = 0
        self.right_move_count = 0

    def penalize_gaps(self, initial_gaps, current_gaps):
        if current_gaps < initial_gaps:
            return 5.0
        elif current_gaps == initial_gaps:
            return 1.0
        else:
            return -10.0
            #print("penalize_indecision")

    # def penalize_height(self, game_area):
    #     """Apply a penalty if the stack goes above a certain height."""
    #     # Calculate the maximum height in each column
    #     max_height = np.max(np.where(game_area != 0, np.arange(game_area.shape[0])[:, None], 0), axis=0)
    #     # Calculate the penalty for heights above 12
    #     height_penalty = np.sum(np.maximum(max_height - 10, 0))
    #     # Check if any column has a height greater than 15
    #     if np.any(max_height > 15):  # Use np.any() to check if any column exceeds height 15
    #         height_penalty += 10.0
    #     return -height_penalty
    
    def calculate_line_fill(self, game_area):
        """Calculate the line fill percentage."""
        total_cells = game_area.size
        filled_cells = np.sum(game_area != 0)
        return filled_cells / total_cells

    def reward_for_lines_cleared(self):
        # Get the current total number of lines cleared
        current_total_lines = self.tetris.lines

         # Calculate the difference in lines cleared since the last step
        lines_cleared = current_total_lines - self.previous_total_lines

        # Reward calculation based on the number of lines cleared
        reward = 0.0
        if lines_cleared > 0:
            if lines_cleared == 1:
                 reward += 10.0
            elif lines_cleared == 2:
                reward += 30.0
            elif lines_cleared == 3:
                reward += 60.0
            elif lines_cleared >= 4:
                 reward += 100.0
    
    #   Update the previous total lines cleared
        self.previous_total_lines = current_total_lines

        return reward

    def _check_piece_dropped(self):
        current_tetromino = self.tetris.next_tetromino()
        piece_dropped = current_tetromino != self.previous_tetromino
        return piece_dropped

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        self._fitness = self.tetris.score

    def count_gaps(self, grid):
        gap_count = 0
        for x in range(grid.shape[1]):
            hit_block = False
            for y in range(grid.shape[0]):
                if grid[y][x] != 0:
                    hit_block = True
                elif hit_block and grid[y][x] == 0:
                    gap_count += 1
        return gap_count

    def render(self, mode='human'):
        if mode == 'human':
            pass  # Add human rendering logic if needed
        elif mode == 'rgb_array':
            screen_image = self.pyboy.screen.image
            if screen_image is not None:
                return np.array(screen_image)
            else:
                print("Warning: Screen image is None.")
                return np.zeros((18, 10, 3), dtype=np.uint8)

    def close(self):
        self.pyboy.send_input(WindowEvent.SCREEN_RECORDING_TOGGLE)
        self.pyboy.screen.image.save("Tetris2.png")
        self.pyboy.stop()

    def _get_info(self):
        return {
            'score': self.tetris.score,
            'level': self.tetris.level,
            'lines': self.tetris.lines
        }

def capture_frames(env, model):
    frames = []
    obs = env.reset()
    done = False
    while not done:
        frames.append(env.render(mode='rgb_array'))
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            print("Game over!")
            break

    return frames
