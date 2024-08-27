# group-project-final - AI Videogame

## Team
1. Jacob Van Skyhawk
2. Ian Taylor
3. David Moyes
4. Jake Backues

## Goal
Build a Reinforcement Learning AI model that could beat us (but really just David) at Tetris in 2 weeks.

## Data Set
We use the data generated by the model playing the game in the emulated environment. For most of the examples in this repo, images represent the state of the environment of the game.

## High-level plan
1. Emulate and run the game locally or on Google Collab
2. Figure out how to have a model play the running game
3. Build model(s) (Divide and conquer)
4. Train model(s)

## ALE - Emulator

The files in this folder all build off of the ALE emulator. We use different Models and Policies which you can review in the individual files within.

## Results
We were able to implement emulators and game wrappers, then inject premade RL model "Agents" into the games to take actions. We had trouble implementing good feedback (rewards and punishments) policies to help the agents score points. Unfortunately, our models never were able to score any points. We are still happy with the results as we learned a lot through this project. None of us had learned about or implemented Reinforcement Learning models before.

## Resources

1. [Deep Q Learning](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47)
2. [Gym - Tetris](https://gymnasium.farama.org/environments/atari/tetris/#actions)
3. [Q Learning Example](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning)
4. [ALE Emulator - M. G. Bellemare, Y. Naddaf, J. Veness and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013.](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
5. [Loonride - AI learns to play Tetris](https://www.youtube.com/watch?v=pXTfgw9A08w&t=103s)
6. [Stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
7. [Q-Learning Agent](https://github.com/nuno-faria/tetris-ai/blob/master/dqn_agent.py)