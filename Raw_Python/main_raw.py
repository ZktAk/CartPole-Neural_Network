"""
Created by: GitHub user ZktAk
Date: 7/11/2024
Name: Neural Network Cart Pole

This program solves OpenAI's Gym environment, Cart Pole, using a neural network programmed in raw python

This code is not original. It was copied from https://github.com/abhavk/dqn_from_scratch/blob/master/cartpole.py
That code was broken by updates to OpenAI's Gym library and refactored here to restore functionality and to add
ZktAk's own touch and preferences.

One such preferential choice was to separate the main program loop and neural network classes into distinct files
"""


import gymnasium  # This is the first change. Old way was to import gym instead.
import numpy as np
from collections import deque
from agent_raw import RLAgent


if __name__ == "__main__":

	env = gymnasium.make('CartPole-v1')

	# Global variables
	MAX_TIMESTEPS = 1000  # max length of each episode

	AVERAGE_REWARD_TO_SOLVE = 195  # Achieving this average score after at least...
	NUM_EPS_TO_SOLVE = 100  # ...this many episodes will qualify as "solving" the Cart Pole problem

	scores_last_timesteps = deque([], NUM_EPS_TO_SOLVE)  # this is to help calculate the average score

	# Neural Network variables
	batch_size = 32
	GAMMA = 0.95
	EPSILON_DECAY = 0.997
	hidden_layer_size = 128
	num_hidden_layers = 2

	agent = RLAgent(env, num_hidden_layers, hidden_layer_size, GAMMA, EPSILON_DECAY)

	# The main program loop
	for i_episode in range(10000):

		observation = env.reset()[0]
		# env.reset() returns (array([ 0.01418237,  0.03814205,  0.00987436, -0.02633131], dtype=float32), {})
		# which is the Observation Space and an empty dictionary. The dictionary is utilized in other environments
		# but is not used in Cart Pole, hence why it is empty in this case. In the past, the Cart Pole environment
		# intelligently omitted the dictionary, but this feature was removed at some point in favor of obfuscation.

		# That is why we must now use env.reset()[0] instead to retrieve the desired Observation Space.


		# Check if Cart Pole has been "solved"
		if i_episode >= NUM_EPS_TO_SOLVE:
			if i_episode >= NUM_EPS_TO_SOLVE and np.mean(scores_last_timesteps) >= AVERAGE_REWARD_TO_SOLVE:
				print("solved after {} episodes".format(i_episode))
				break

		# Iterating through time steps within an episode
		for t in range(MAX_TIMESTEPS):

			if i_episode == 75 and t == 0:
				env = gymnasium.make('CartPole-v1', render_mode="human")
				# render_mode="human" tells gym to visually display the environment. However, displaying the environment
				# slows down the processing dramatically, and therefore we don't want this enabled from the start.
				# That is why we do not specify a render mode initially (it defaults to no display).
				# After approximately 100 episodes (depending on the random seed) is when this particular model
				# starts to perform well. At that time we can re-make the environment and set render_mode to "human".
				# There may be a better way of updating the render mode without re-making the environment, but that method
				# is unknown to me.

				observation = env.reset()[0]
				# Again, notice env.reset()[0]. Explained above.


			action = agent.select_action(observation)

			prev_obs = observation

			observation, reward, done, info, _ = env.step(action)
			# Simular to the empty dictionary produced above by env.reset(), env.step() also returns an empty
			# dictionary at the last index. The reason here is the same reason as described above. Here we
			# assign it to _ to indicate that we do not care about _'s contents.

			# Keep a store of the agent's experiences
			agent.remember(done, action, observation, prev_obs)
			agent.experience_replay(batch_size)

			if done:  # If the pole has tipped over, end this episode
				scores_last_timesteps.append(t + 1)
				print(f"Episode {i_episode} ended after {t + 1} timesteps | average score: {np.mean(scores_last_timesteps)}")
				break
