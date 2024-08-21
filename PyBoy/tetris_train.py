import torch
import pandas as pd
import time
import imageio
import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from tetris_env import TetrisEnv  # Assuming this is a custom environment you've defined
from pyboy import PyBoy

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:  # Adjust frequency as needed
            last_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(last_info['r'])
            self.episode_lengths.append(last_info['l'])
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame({
            'Episode Rewards': self.episode_rewards,
            'Episode Lengths': self.episode_lengths
        })
        df.to_csv('training_progress.csv', index=False)

class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:  # Adjust this to control how often updates are printed
            elapsed_time = time.time() - self.start_time
            print(f"Step: {self.num_timesteps}, Elapsed Time: {elapsed_time:.2f} seconds")
        return True

class GIFCallback(BaseCallback):
    def __init__(self, save_freq, gif_path_prefix, env, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.gif_path_prefix = gif_path_prefix
        self.env = env
        self.frames = []
        self.save_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            frame = self.env.render(mode='rgb_array')
            self.frames.append(frame)

            if self.num_timesteps % (self.save_freq * 5) == 0:  # Save every 5 times save_freq
                self._save_gif()

        return True

    def _on_training_end(self) -> None:
        if self.frames:
            self._save_gif()

    def _save_gif(self):
        gif_path = f"{self.gif_path_prefix}_{self.save_count}.gif"
        imageio.mimsave(gif_path, self.frames, fps=10)
        self.frames = []
        self.save_count += 1

def make_env(rom_path):
    env = TetrisEnv(rom_path)
    return env

def env_quck_test():
    env = TetrisEnv(rom_path="Tetris.gb")
    print("Env created!")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step result: observation shape={obs.shape}, reward={reward}, done={done}")
    # Check the environment directly

def test_pyboy_start_game():
    pyboy = PyBoy("Tetris.gb")
    pyboy.set_emulation_speed(1)  # Use a non-zero speed for testing
    tetris = pyboy.game_wrapper
    tetris.start_game(timer_div=0x00)  # Test this in isolation
    print("Game started.")

def create_env(rom_path='Tetris.gb', num_envs=1):
    print(f"Initializing PyBoy with ROM: {rom_path}")

    env = DummyVecEnv([lambda: make_env(rom_path) for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    print("Checking env...")
    check_env(env, warn=None)

    print(f"Environment created with {num_envs} vectorized environments.")
    return env

def train_model(env):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=2, 
        device=device, 
        policy_kwargs=dict(normalize_images=False)
    )
    print(f"PPO model initialized.")

    gif_callback = GIFCallback(
        save_freq=1000,  # Adjust this to control how often GIFs are captured
        gif_path_prefix="progress",
        env=env
    )

    return model

def model_learn(model):
    progress_callback = ProgressCallback(verbose=1)

    print("Starting model training...")
    model.learn(total_timesteps=10000, callback=progress_callback)
    print("Model training completed.")

    return model
