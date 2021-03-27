import numpy as np
import torch
import gym
import os

import utils
import DDPG

env_name = 'Pendulum-v0' # 'BipedalWalker-v3'
seed = 0
start_timesteps = 1e3 #25e3
eval_freq = 5e3
max_timesteps = 1e6
expl_noise = 0.1
batch_size = 256
discount = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
save_model = False
load_model = ""


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
  eval_env = gym.make(env_name)
  eval_env.seed(seed + 100)

  avg_reward = 0.
  for _ in range(eval_episodes):
    state, done = eval_env.reset(), False
    while not done:
      action = policy.select_action(np.array(state))
      state, reward, done, _ = eval_env.step(action)
      avg_reward += reward

  avg_reward /= eval_episodes

  print("---------------------------------------")
  print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
  print("---------------------------------------")
  return avg_reward


if __name__ == "__main__":
  if not os.path.exists("./results"):
    os.makedirs("./results")

  if save_model and not os.path.exists("./models"):
    os.makedirs("./models")

  env = gym.make(env_name)

  # Set seeds
  env.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0] 
  max_action = float(env.action_space.high[0])

  policy = DDPG.DDPG(state_dim, action_dim, max_action, discount=discount, tau=tau)


  file_name = f"{policy}_{env}_{seed}"
  print("---------------------------------------")
  print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
  print("---------------------------------------")

  if load_model != "":
    policy_file = file_name if load_model == "default" else load_model
    policy.load(f"./models/{policy_file}")

  replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
  
  # Evaluate untrained policy
  evaluations = [eval_policy(policy, env_name, seed)]

  state, done = env.reset(), False
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0

  for t in range(int(max_timesteps)):
    
    episode_timesteps += 1

    # Select action randomly or according to policy
    if t < start_timesteps:
      action = env.action_space.sample()
    else:
      action = (
        policy.select_action(np.array(state))
        + np.random.normal(0, max_action * expl_noise, size=action_dim)
      ).clip(-max_action, max_action)

    # Perform action
    next_state, reward, done, _ = env.step(action) 
    done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= start_timesteps:
      policy.train(replay_buffer, batch_size)

    if done: 
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
      # Reset environment
      state, done = env.reset(), False
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 

    # Evaluate episode
    if (t + 1) % eval_freq == 0:
      evaluations.append(eval_policy(policy, env_name, seed))
      np.save(f"./results/{file_name}", evaluations)
      if save_model: policy.save(f"./models/{file_name}")
