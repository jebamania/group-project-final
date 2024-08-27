import torch
import torch.nn as nn
import torch.optim as optim
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
        if len(self.model.ep_info_buffer) > 0:
            # Get the latest episode information
            last_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(last_info['r'])
            self.episode_lengths.append(last_info['l'])

            # Log rewards every 100000 timesteps
            if self.num_timesteps % 100000 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                print(f"Mean Reward (last 100 episodes): {mean_reward}")

                # Log the mean reward to TensorBoard
                self.logger.record('eval/mean_reward', mean_reward)

        return True

    def _on_training_end(self) -> None:
        # Save rewards to a file for later analysis
        df = pd.DataFrame({
            'Episode Rewards': self.episode_rewards,
            'Episode Lengths': self.episode_lengths
        })
        df.to_csv('training_progress.csv', index=False)

class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        # This method is called at each step
        return True
    
    def _on_rollout_end(self) -> None:
        # This method is called at the end of each rollout
        episode_reward = self.locals.get('reward', 0)
        self.episode_rewards.append(episode_reward)
        
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            print(f"Mean reward so far: {mean_reward:.2f}")
        
        return super(RewardTrackingCallback, self)._on_rollout_end()

class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Print elapsed time and total timesteps every 1000 calls
        if self.n_calls % 100000 == 0:  # Adjust this to control how often updates are printed
            elapsed_time = time.time() - self.start_time
            print(f"Step: {self.num_timesteps}, Elapsed Time: {elapsed_time:.2f} seconds")
        return True

def make_env(rom_path):
    env = TetrisEnv(rom_path)
    return env

def env_quck_test():
    env = TetrisEnv(rom_path="Tetris.gb")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step result: observation shape={obs.shape}, reward={reward}, done={done}")
    # Check the environment directly

def create_env(rom_path='Tetris.gb', num_envs=1):
    print(f"Initializing PyBoy with ROM: {rom_path}")

    env = DummyVecEnv([lambda: make_env(rom_path) for _ in range(num_envs)])
    ##env = VecNormalize(env, norm_obs=True, norm_reward=True)

    print(f"Environment created with {num_envs} vectorized environments.")
    return env

def train_model(env):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Specify the TensorBoard log directory
    tensorboard_log_dir = "./ppo_tetris_tensorboard/"

    # Initialize the PPO model with TensorBoard logging enabled
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=tensorboard_log_dir, 
        device=device,
        learning_rate=1e-3,  # Decreased learning rate for stability
        n_steps=512,  # Increased steps per update
        batch_size=32,  # Increased batch size
        gamma=0.99,  # Increased gamma for long-term rewards
        gae_lambda=0.98,  # Increased GAE lambda
        ent_coef=.01,  # Increased entropy coefficient for more exploration
        n_epochs=10,
        clip_range=0.1,  # Reduced clipping range for more stable updates
        max_grad_norm=0.5  # Gradient clipping to prevent exploding gradients
    )
    print(f"PPO model initialized.")

    # Set up CustomCallback for mean_reward and mean_ep_length reporting
    custom_callback = CustomCallback(verbose=1)
    return model, custom_callback

def model_learn(model, custom_callback, timesteps=100000):
    progress_callback = ProgressCallback(verbose=1)

    print("Starting model training...")
    model.learn(
        total_timesteps=timesteps,
        callback=[progress_callback, custom_callback],  # Ensure callbacks are passed
        progress_bar=True
    )
    print("Model training completed.")

    return model

def play_and_record(model, env, num_frames=1000, gif_filename='model_play.gif'):
    # Optionally load a saved state
    # env.load_state('saved_state.pkl')  # Assuming you have a method for this

    obs, info = env.reset()  # Extract the observation only
    frames = []

    done = False  # Initialize done to False to start the game loop
    while not done:  # Continue until the game is over
        frames.append(env.render(mode='rgb_array'))  # Capture the screen
        action, _ = model.predict(obs)  # Predict the next action based on the observation
        obs, reward, done, truncated, info = env.step(action)  # Unpack all values returned by step()

    # Save the frames as a GIF
    imageio.mimsave(gif_filename, frames, fps=30)
    print(f"Game play GIF saved as {gif_filename}")
