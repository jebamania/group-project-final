# Additional Requirements
#!pip install pyqlearning
#!pip install stable-baselines3
#!pip install opencv-python

from ale_py import ALEInterface, roms
from stable_baselines3.common.evaluation import evaluate_policy


ale = ALEInterface()
ale.loadROM(roms.Tetris)
ale.reset_game()

#action_space = [1,2,3,4]  # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4; DOWN
reward = ale.act(0)  # noop

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

total_timesteps = 440000000

## Create the Tetris environment

env = gym.make(f'ALE/{ale_model}'
    , obs_type = obs_type
    , frameskip = 4
    # , n_steps = 4
    # , noop_max = 30
) #, render_mode="human" # remove render_mode in training

episodes = 5

####### Create or Load Model

from stable_baselines3 import DQN

modelName = f"{agentName}_{policy}_{total_timesteps}_obs_{obs_type}"

if new_model:
    model = DQN(
        policy
        , env
        , buffer_size=40000
        , verbose=1
        , learning_rate=1e-3
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
else:
    model = DQN.load(modelName)

mean_reward, std_reward = evaluate_policy(
    model
    , model.get_env()
    , n_eval_episodes=10
    , deterministic=False
)

obs, info = env.reset()
for episode in range(episodes):
    idx = 0
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=False)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Episode: {episode}, idx: {idx}")
        print(f"Action: {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        # print reward only if it is not 0
        if reward != 0:
            print(f"Reward: {reward}")
            total_reward += reward
        
        #print a line break
        print("")

        if terminated or truncated:

            #obs.save(f"{modelName}_obs_{obs_type}_finishedGame.jpeg")
            env.env.ale.saveScreenPNG(f"Images/{modelName}_obs_{obs_type}_episode_{episode}.png")

            #print total reward for the episode
            print(f"FINAL EPISODE: {episode}, Total Reward: {total_reward}")

            obs, info = env.reset()
            done = True

        idx += 1

print(f"Evaluation Complete: mean_reward: {mean_reward}, std_reward: {std_reward}")

if new_model:
    model.save(modelName)
    del model # remove to demonstrate saving and loading
