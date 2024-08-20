import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from tetris import tetris # Assuming tetris.py is in the same directory and correctly structured.

actions = ['', 'a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']

matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class GenericPyBoyEnv(gym.Env):
    def __init__(self, tetris_rom, debug=False):
        super().__init__()
        self.pyboy = PyBoy(tetris_rom)
        self.tetris = tetris(self.pyboy)  # Instantiate the Tetris game wrapper from tetris.py
        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space

        self.tetris.start_game()  # Start the Tetris game

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == 0:
            pass
        else:
            self.pyboy.button(actions[action])

        self.pyboy.tick(1)

        done = self.tetris.game_over  # Check if the game is over

        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness

        observation = self.tetris.game_area()  # Get the current game area
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        self._fitness = self.tetris.score  # Use the score from the Tetris game

    def reset(self, **kwargs):
        self.tetris.reset_game()  # Reset the Tetris game
        self._fitness = 0
        self._previous_fitness = 0

        observation = self.tetris.game_area()  # Get the initial game area
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()
