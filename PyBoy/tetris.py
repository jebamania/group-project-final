import imageio
from pyboy import PyBoy
from PyBoy.tetris_env import GenericPyBoyEnv

# Initialize PyBoy with the Tetris ROM
rom_path = 'Tetris.gb'
env = GenericPyBoyEnv(rom_path)

# Wrap your environment to be vectorized
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env(lambda: GenericPyBoyEnv(rom_path), n_envs=1)

# Initialize the PPO model
from stable_baselines3 import PPO
model = PPO("CnnPolicy", env, verbose=1)

# Function to capture frames for GIF
def capture_frames(env, num_frames=1000):
    frames = []  # Store frames for GIF
    obs = env.reset()
    for _ in range(num_frames):
        frames.append(env.render(mode='rgb_array'))  # Capture the screen
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            obs = env.reset()
    return frames

# Record the first attempt
frames = capture_frames(env)
# Save the first attempt GIF
imageio.mimsave('first_attempt.gif', frames, fps=30)

# Train the model
model.learn(total_timesteps=10000)

# Record the last attempt
frames = capture_frames(env)
# Save the last attempt GIF
imageio.mimsave('last_attempt.gif', frames, fps=30)

# Close the environment
env.close()
