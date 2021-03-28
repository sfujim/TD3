import numpy as np
import torch
import gym
import os, datetime

import utils
import DDPG

env_name = 'BipedalWalker-v3' # 'Pendulum-v0' 
seed = 0

start_timesteps = 1e3 #25e3
eval_freq = 5e3
max_timesteps = 5 * 1e6
mean_reward_greater_than = 230.0

expl_noise = 0.1
batch_size = 128 # 256
discount = 0.99
tau = 1e-3 # 0.005

policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2

lr_actor = 3e-4 # 1e-3
lr_critic = 3e-4 # 1e-3
critic_weight_decay = 1e-3 # 1e-2



if __name__ == "__main__":
  
  env = gym.make(env_name)

  # Set seeds
  env.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0] 
  max_action = float(env.action_space.high[0])

  # import pdb; pdb.set_trace()

  policy = DDPG.DDPG(state_dim, action_dim, max_action, discount=discount,
                     tau=tau, lr_actor=lr_actor, lr_critic=lr_critic,
                     critic_weight_decay=critic_weight_decay)



  file_name = "p"
  print("---------------------------------------")
  print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
  print("---------------------------------------")

  policy_file = file_name
  policy.load(f"./models/{policy_file}")

  state, done = env.reset(), False
  episode_reward = 0
  
  for t in range(int(max_timesteps)):
    action = policy.select_action(np.array(state))
    state, reward, done, _ = env.step(action)

    env.render()
    episode_reward += reward
    
    if done:
      print('Episode reward: %3.f' % episode_reward)
      episode_reward = 0.0
      state, done = env.reset(), False
