import gymnasium as gym
import ale_py  # Import necessary packages
from stable_baselines3 import DQN  # Assuming you used DQN for training
import imageio
from IPython.display import display, Image
from gymnasium.wrappers import RecordVideo, TransformReward

# Create and register the environment
gym.register_envs(ale_py)
env = gym.make('ALE/Tetris-v5', render_mode="rgb_array")  # Set render_mode to "human" for visualization


video_folder = "/Users/dmoyes/Desktop/video_output"  # Change path as needed
env = RecordVideo(env, video_folder, name_prefix="Tetris")

# Load the trained model (adjust the path to where your model is saved)
model_path = "./2.zip" 
model = DQN.load(model_path, env=env)

# Run the model in the environment
obs, info = env.reset()
episode_over = False

while not episode_over:
    # Use the model to predict the action
    action, _ = model.predict(obs)
    # Take a step in the environment using the predicted action
    obs, reward, terminated, truncated, info = env.step(action)
    # Render the environment to visualize the game
    env.render()
    # Print info
    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)
    # Check if the episode is over
    episode_over = terminated or truncated

env.close()  # Close the environment after the episode ends