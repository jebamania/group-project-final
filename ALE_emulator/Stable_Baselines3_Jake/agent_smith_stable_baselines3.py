# Additional Requirements
#!pip install pyqlearning
#!pip install stable-baselines3
#!pip install opencv-python

from ale_py import ALEInterface, roms
from stable_baselines3.common.evaluation import evaluate_policy


ale = ALEInterface()
ale.loadROM(roms.Tetris)
ale.reset_game()

import gymnasium as gym
import ale_py
import time

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

# Initialize parameters
#obs_type="rgb"
#obs_type="grayscale"
obs_type="ram"

ale_model = 'Tetris-v5'
if obs_type == "ram":
    ale_model = 'Tetris-ram-v5'

new_model = True
agentName = 'dqn_tetris'

policy = 'MlpPolicy'
if obs_type == "rgb":
    policy = 'CnnPolicy' # only works with obs_type="rgb"

total_timesteps = 1000

## Create the Tetris environment

env = gym.make(f'ALE/{ale_model}'
    , obs_type = obs_type
    , frameskip = 4
    # , n_steps = 4
    # , noop_max = 30
) #, render_mode="human" # remove render_mode in training

episodes = 5

####### Create or Load Model

from agent_smith import DQNAgentSmith
agentSmith = DQNAgentSmith(env.get_state_size())
obs = env.reset()
done = False

for episode in range(episodes):
    idx = 0
    while not done:
        idx += 1
        start_time = time.time()

        # Use a random action for demonstration purposes
        action = env.action_space.sample()
        # next_states = env.get_next_states()
        # best_state = agentSmith.best_state(next_states.values())

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            done = True

        # Add a small delay to slow down the game for human observation
        time.sleep(0.001)


env.close()