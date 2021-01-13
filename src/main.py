import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
from generative_replay import GenerativeReplay
from datetime import datetime

r_before = 0

# Evaluate the policy with a new env
def eval_policy(policy, env_name, seed, eval_episodes, replay, replay_component):
	global r_before
	eval_env = gym.make(env_name)
	eval_env.seed(42)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			old = state
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward


	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")

	return avg_reward


if __name__ == "__main__":
	
	# Hyper parameters

	# General
	USE_GENERATIVE = True
	NO_REPLAY = False
	ENV = "InvertedPendulum-v2"
	START_TIMESTEPS = 15e3
	END = START_TIMESTEPS + 50e3
	EVAL_FREQ = 5e3
	MAX_TIMESTEPS = 2e5
	SEED = 13
	FILE_NAME = ENV + "_" + list(str(datetime.now()).split())[-1]

	# TD3 parameters
	EXPL_NOISE = 0.1
	BATCH_SIZE = 256
	DISCOUNT = 0.99
	TAU = 0.005
	POLICY_NOISE = 0.2
	NOISE_CLIP = 0.5
	POLICY_FREQ = 2
	

	print(f"Start new process with {ENV} and file name {FILE_NAME}")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(ENV)

	# Set seeds
	env.seed(SEED)
	torch.manual_seed(SEED)
	np.random.seed(SEED)
	
	# Some env dimentions
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Build TD3
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": DISCOUNT,
		"tau": TAU,
		"policy_noise": POLICY_NOISE * max_action,
		"noise_clip": NOISE_CLIP * max_action,
		"policy_freq": POLICY_FREQ
	}

	policy = TD3.TD3(**kwargs)

	# Make the replay component
	replay_component = None
	if USE_GENERATIVE:
		replay_component = GenerativeReplay()
	elif NO_REPLAY:
		replay_component = utils.ReplayBuffer(state_dim, action_dim, BATCH_SIZE)
	else:
		replay_component = utils.ReplayBuffer(state_dim, action_dim)
	

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, ENV, SEED, 10, replay_component, None)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0


	for t in range(int(MAX_TIMESTEPS)):


		
		episode_timesteps += 1

		if t >= END:
			raise ValueError

		# Select action randomly or according to policy based on the start timesteps
		if t < START_TIMESTEPS:
			action = env.action_space.sample()
		else:
			replay_component.training = True
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay component
		# If the VAE reaches buffer max, it will train itself blocking this for a while
		replay_component.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= START_TIMESTEPS:
			policy.train(replay_component, BATCH_SIZE)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True

			print(f"Total timesteps: {t},  Episode {episode_num} done, lasted {episode_timesteps} timesteps, total reward is {episode_reward}")
			if t >= START_TIMESTEPS:
				evaluations.append(episode_reward)
				np.save(f"./results/incoming{FILE_NAME}", evaluations)

			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		# if (t + 1) % EVAL_FREQ == 0:
		# 	print(f"Total timesteps: {t}")
		# 	if t >= START_TIMESTEPS:
		# 		evaluations.append(eval_policy(policy, ENV, SEED,20, replay_component, replay_component))
		# 	else:
		# 		evaluations.append(eval_policy(policy, ENV, SEED,20, replay_component, None))
		# 	np.save(f"./results/td3/{FILE_NAME}", evaluations)
