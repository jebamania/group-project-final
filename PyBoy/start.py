import imageio
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pyboy import PyBoy
from PyBoy.tetris_env import GenericPyBoyEnv  # Ensure this import matches the file and class names

def main():
    # Initialize PyBoy with the Tetris ROM
    pyboy = PyBoy('Tetris.gb')
    env = GenericPyBoyEnv(pyboy)

    # Wrap your environment to be vectorized
    env = make_vec_env(lambda: env, n_envs=1)

    # Initialize the PPO model with MlpPolicy
    model = PPO("CnnPolicy", env, verbose=1)

    # Record the first attempt
    obs = env.reset()
    frames = []  # Store frames for GIF
    for _ in range(1000):
        frames.append(env.render(mode='rgb_array'))  # Capture the screen
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break

    # Save the first attempt GIF
    imageio.mimsave('first_attempt.gif', frames, fps=30)

    # Train the model
    model.learn(total_timesteps=10000)

    # Record the last attempt
    obs = env.reset()
    frames = []  # Store frames for GIF
    for _ in range(1000):
        frames.append(env.render(mode='rgb_array'))  # Capture the screen
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break

    # Save the last attempt GIF
    imageio.mimsave('last_attempt.gif', frames, fps=30)

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
