import imageio
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent

actions = ['', 'a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']
matrix_shape = (18, 10)  # Update this to the actual shape returned by pyboy.game_area()
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class TetrisEnv(gym.Env):
    def __init__(self, rom_path, debug=False):
        super().__init__()
        self.pyboy = PyBoy(rom_path)
        self.pyboy.set_emulation_speed(0)
        self.tetris = self.pyboy.game_wrapper
        
        assert self.pyboy.cartridge_title == "TETRIS", "Loaded ROM is not Tetris"

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space
        
        self.debug = debug
        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.reset()
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    
        if action != 0:
            self.pyboy.button(actions[action])
        
        self.pyboy.tick(1)
        
        done = self.tetris.game_over()
        
        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness
        
        observation = self.pyboy.game_area()
        if observation.shape != matrix_shape:
            raise ValueError(f"Observation shape mismatch: expected {matrix_shape}, got {observation.shape}")

        info = {}
        truncated = False
        
        print(f"Step complete: reward={reward}, done={done}")
        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        self._fitness = self.tetris.score
    
    def reset(self, seed=None):
        self.seed = seed
        print("Setting Wrapper...")
        self.tetris.game_area_mapping(self.tetris.mapping_compressed, 0)
        print("Starting game...")
        self.tetris.start_game(timer_div=0x00)
        self._fitness = 0
        self._previous_fitness = 0

        print("Game started, getting observation...")
        observation = self.pyboy.game_area()
        if observation.shape != matrix_shape:
            raise ValueError(f"Observation shape mismatch on reset: expected {matrix_shape}, got {observation.shape}")

        info = {}
        print("Reset complete.")
        return observation, info

    def render(self, mode='human'):
        if mode == 'human':
            # Implement visualization if necessary
            pass
        elif mode == 'rgb_array':
            return self.pyboy.screen.image
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def close(self):
        # Save the final screen and stop recording if enabled
        self.pyboy.send_input(WindowEvent.SCREEN_RECORDING_TOGGLE)
        self.pyboy.screen.image.save("Tetris2.png")
        self.pyboy.stop()

    def _get_info(self):
        return {
            'score': self.tetris.score,
            'level': self.tetris.level,
            'lines': self.tetris.lines
        }

def capture_frames(env, model, num_frames=1000):
    frames = []  # Store frames for GIF
    obs = env.reset()
    for _ in range(num_frames):
        frames.append(env.render(mode='rgb_array'))  # Capture the screen
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            obs = env.reset()
    return frames
        
