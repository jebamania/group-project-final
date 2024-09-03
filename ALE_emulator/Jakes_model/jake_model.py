# Additional Requirements
#!pip install pyqlearning
#!pip install stable-baselines3
#!pip install opencv-python

from ale_py import ALEInterface, roms
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN

ale = ALEInterface()
ale.loadROM(roms.Tetris)
ale.reset_game()

import gymnasium as gym
import ale_py
import time
import math

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

# Initialize parameters
#obs_type="rgb"
#obs_type="grayscale"
obs_type="ram"

ale_model = 'Tetris-v5'
if obs_type == "ram":
    ale_model = 'Tetris-ram-v5'

new_model = False
agentName = 'dqn_tetris'

policy = 'MlpPolicy'
if obs_type == "rgb":
    policy = 'CnnPolicy' # only works with obs_type="rgb"

total_timesteps = 1000000

#######

def run_test_games(episodes, modelName, obs_type, env):
    for episode in range(episodes):
        idx = 0
        done = False
        total_reward = 0
        action = 0

        while not done:

            obs, reward, terminated, truncated, info = env.step(action)

            if reward != 0:
                total_reward += reward

            if terminated or truncated:

                env.env.ale.saveScreenPNG(f"Images/{modelName}_obs_{obs_type}_episode_{episode}.png")

                #print total reward for the episode
                print(f"FINAL EPISODE: {episode}, Total Reward: {total_reward}")

                obs, info = env.reset()
                done = True

            action, _states = model.predict(obs, deterministic=False)

            idx += 1

def load_model(new_model, modelName, total_timesteps, env):
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
        , env
        , n_eval_episodes=10
        , deterministic=False
    )

    print(f"Evaluation Complete: mean_reward: {mean_reward}, std_reward: {std_reward}")

    return model

def get_env(obs_type, ale_model):
    env = gym.make(f'ALE/{ale_model}'
        , obs_type = obs_type
        , frameskip = 4
        # , n_steps = 4
        # , noop_max = 30
    ) #, render_mode="human" # remove render_mode in training

    env = CustomRewardWrapper(env)

    return env

# class RelativePosition(ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

#     def observation(self, obs):
#         return obs["target"] - obs["agent"]

# Create custom reward and penalty functions
class CustomRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        
        # Number of steps for a single piece to fall
        self.fall_steps = 15
        self.step_count = 0

        # Counters for specific actions taken by the agent
        self.fire_count = 0
        self.left_right_count = 0

        # Thresholds for applying penalties and rewards
        self.left_right_threshold = 100
        self.fire_penalty_threshold = 500

        # Penalties and rewards definitions
        self.fire_penalty = -0.01  # Penalty for excessive "FIRE" actions
        self.termination_penalty = -0.1  # Penalty for game over
        self.movement_reward = 0.25  # Reward for moving left or right

        # Additional bonuses
        self.survival_bonus = 0.005  # Bonus for surviving longer

    def reset(self, **kwargs):
        # Reset counters when a new episode starts
        self.fire_count = 0
        self.left_right_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1

        # Execute the action and observe the outcome
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Thresholds for applying penalties and rewards. x l/r per fall steps
        self.left_right_threshold = math.ceil(self.step_count * (3/self.fall_steps))

        # reduce the reward as you get closer to the threshold
        self.movement_reward = 1 - (self.left_right_count / self.left_right_threshold)

        # Update thresholds based on the number of steps taken. x spins per fall steps
        self.fire_penalty_threshold = math.ceil(self.step_count * (3/self.fall_steps))

        # Increment fire count and apply penalty if threshold is exceeded
        if action == 1:
            self.fire_count += 1
            if self.fire_count > self.fire_penalty_threshold:
                reward += self.fire_penalty  # Penalize for excessive fire actions

        # Increment left/right count and reward movement within threshold
        if action == 2 or action == 3:
            self.left_right_count += 1
            if self.left_right_count <= self.left_right_threshold:
                reward += self.movement_reward # Reward for moving left or right

        # Apply bonuses for gameplay strategies
        reward += self.survival_bonus  # Bonus for each step the game continues

        # Apply termination penalty if the game ends
        if terminated:
            reward += self.termination_penalty

        return obs, reward, terminated, truncated, info

####### Create the Tetris environment

env = get_env(obs_type, ale_model)

episodes = 5

####### Create or Load Model

modelName = f"{agentName}_{policy}_{total_timesteps}_obs_{obs_type}"

model = load_model(new_model, modelName, total_timesteps, env)

obs, info = env.reset()

if new_model:
    model.save(modelName)
    del model # remove to demonstrate saving and loading
else:
    run_test_games(episodes, modelName, obs_type, env)

