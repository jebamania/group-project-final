# Additional Requirements
#!pip install pyqlearning
#!pip install stable-baselines3

from ale_py import ALEInterface, roms
from agent_smith import DQNAgentSmith

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
env = gym.make('ALE/Tetris-v5') #, render_mode="human" # remove render_mode in training

# agentSmith = DQNAgentSmith(env.get_state_size())
episodes = 5

####### use stable_baselines3

from stable_baselines3 import DQN

new_model = True

# try to load saved model
# try:
agentName = 'dqn_tetris'
#policy = 'MlpPolicy'
#agentName = 'dqn_cartpole'
policy = 'CnnPolicy'
total_timesteps = 50000
modelName = f"{agentName}_{policy}_{total_timesteps}"

if new_model:
    model = DQN(policy, env, buffer_size=40000, verbose=1)
    # about an hour total_timesteps=4076
    model.learn(total_timesteps=total_timesteps, log_interval=total_timesteps/10)
else:
    model = DQN.load(modelName)

obs, info = env.reset()
for episode in range(episodes):
    idx = 0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=False)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Episode: {episode}, idx: {idx}")
        print(f"Action: {action}")

        # print reward only if it is not 0
        if reward != 0:
            print(f"Reward: {reward}")

        obs, reward, terminated, truncated, info = env.step(action)
        
        #print a line break
        print("")

        if terminated or truncated:

            #print total reward for the episode
            print(f"FINAL EPISODE: {episode}, Total Reward: {reward}, obs: {obs}, env: {env}")

            obs, info = env.reset()
            done = True

        idx += 1

if new_model:
    model.save(modelName)
    del model # remove to demonstrate saving and loading



# end stable_baselines3

''' custom agent attempt
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
'''