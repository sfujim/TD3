import numpy as np
import torch
import gym
import argparse
import os
import utils
import TD3
import OurDDPG
import DDPG
import pickle
from random import random
from tensorboardX import SummaryWriter
from network import Net
from utils import update_backward_model,update_forward_model
from utils import unapply_norm,apply_norm
from tqdm import trange

# Runs policy for X episodes and returns average reward
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=2, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--fwd_model_update_freq", default=1e3, type=float)  # How often (time steps) we update the forward model
    parser.add_argument("--bwd_model_update_freq", default=1e3, type=float)  # How often (time steps) we update the backward model
    parser.add_argument("--imagination_depth", default=1, type=int)  # How often (time steps) we update the forward model
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--load_model", default="None")  # Load a pretrained model
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print(f"Settings: {file_name}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    writer = SummaryWriter(log_dir='results/TrainingLogs/'+str(int(random()*100000)))


    env = gym.make(args.env_name)
    print('-- CREATED ENVIRONMENT -- ')
    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])


    forward_dynamics_model = Net(n_feature=state_dim+action_dim,
                                 n_hidden=500,
                                 n_output=state_dim+1 # reward is the + 1
                                 ).cuda()

    backward_dynamics_model = Net(n_feature=state_dim + action_dim,
                                 n_hidden=500,
                                 n_output=state_dim+1 # reward is the + 1
                                 ).cuda()

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy

    if args.load_model is "None":
        print('-- LOADING RANDOM POLICY --')
        if args.policy_name == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            policy = TD3.TD3(**kwargs)
        elif args.policy_name == "OurDDPG":
            policy = OurDDPG.DDPG(**kwargs)
        elif args.policy_name == "DDPG":
            policy = DDPG.DDPG(**kwargs)
    else:
        print('-- LOADING PRETRAINED POLICY --')
        policy = pickle.load(open(args.load_model, "rb", -1))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    # introduce 2 new replay buffers to store synthetic transitions
    fwd_model_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e7))
    bwd_model_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e7))


    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env_name, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    Ts = [] # list of trajectories
    T = [[]] # buffer for storing a single trajectory

    first_update = False

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)

        # T[0] because currently using only 1 thread for environment
        T[0].append((state, action, reward))  # append the state, action and reward into trajectory

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        # update the forward and backward models here
        if (not first_update and t >= 1000) or (t >= args.fwd_model_update_freq and t % args.fwd_model_update_freq == 0):
            fwd_norm = update_forward_model(forward_dynamics_model, Ts)
            first_update = True # done

        # if t >= args.bwd_model_update_freq and t % args.bwd_model_update_freq == 0:
        #     bwd_norm = update_backward_model(backward_dynamics_model, Ts)

        # add imagined trajectories from fwd model into fwd_model_replay_buffer
        if first_update: # model has been updated atleast once
            forward_dynamics_model.eval()

            for _ in range(100): # collect 100 times the data
                t_s, t_a, t_ns, t_r, t_nd = replay_buffer.sample(args.batch_size)
                t_s = t_s.cpu().numpy()
                t_a = t_a.cpu().numpy()

                for model_depth in range(args.imagination_depth):
                    # add noise to actions and predict
                    t_a =  (t_a + np.random.normal(0, max_action*args.expl_noise,
                                                   (t_a.shape[0], t_a.shape[1]))).clip(-max_action, max_action)

                    fwd_input = np.hstack((t_s, t_a))
                    fwd_input = apply_norm(fwd_input, fwd_norm[0]) # normalize the data before feeding in

                    fwd_input = torch.tensor(fwd_input).float().cuda()
                    fwd_output = forward_dynamics_model.forward(fwd_input)
                    fwd_output = fwd_output.detach().cpu().numpy()

                    fwd_output = unapply_norm(fwd_output, fwd_norm[1]) # unnormalize the output data

                    t_ns = fwd_output[:, :-1] + t_s # predicted next state = predicted delta next state + current state
                    t_r = fwd_output[:, -1] # predicted reward

                    for k in range(t_s.shape[0]):
                        fwd_model_replay_buffer.add(t_s[k], t_a[k], t_ns[k], t_r[k], False) # store predicted transition in buffer

                    if args.imagination_depth > 1:
                        # get ready for next transition
                        t_s = t_ns
                        print('Aquiring samples of next actions and states to query ~forward model~')
                        for k in trange(t_s.shape[0]):
                            t_a[k] = (policy.select_action(np.array(t_s[k]))
                                   + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        # Real world data
        if t >= args.batch_size:
            policy.train(replay_buffer, args.batch_size)

        # Imagined data
        if t >= args.batch_size and t >= args.fwd_model_update_freq:
            policy.train(fwd_model_replay_buffer, args.batch_size*100)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            writer.add_scalars('reward',
                           {'episode_reward' : episode_reward},
                           t)

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            Ts.extend(T)
            T = [[]]  #

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env_name, args.seed))
            np.save("./results/%s" % (file_name), evaluations)

    # save the model
    print('-- SAVING THE MODEL --')
    if args.save_models:
        with open("./results/" + policy.__class__.__name__ + ".pkl", "wb") as file_:
            pickle.dump(policy, file_, -1)
    print('-- MODEL SAVED --')

    writer.close()
