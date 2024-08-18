from ale_py import ALEInterface, roms

ale = ALEInterface()
ale.loadROM(roms.Tetris)
ale.reset_game()

reward = ale.act(0)  # noop
screen_obs = ale.getScreenRGB()

import gymnasium as gym
import ale_py
import time

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

# Create the Tetris environment
env = gym.make('ALE/Tetris-v5', render_mode="human")  # remove render_mode in training

# Optionally set the target FPS
env.metadata['render_fps'] = 30
target_fps = env.metadata['render_fps']
frame_duration = 1.0 / target_fps

obs = env.reset()
done = False

while not done:
    start_time = time.time()

    # Use a random action for demonstration purposes
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment
    env.render()

    # Ensure the loop runs at the target FPS
    elapsed_time = time.time() - start_time
    sleep_time = max(0, frame_duration - elapsed_time)
    time.sleep(sleep_time)

env.close()
