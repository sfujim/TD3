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
from utils import unapply_norm, apply_norm, set_seed
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
    parser.add_argument("--start_timesteps", default=1e3,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--fwd_model_update_freq", default=5e3, type=float)  # How often (time steps) we update the forward model
    parser.add_argument("--bwd_model_update_freq", default=5e3, type=float)  # How often (time steps) we update the backward model
    parser.add_argument("--imagination_depth", default=1, type=int)  # How deep to propagate the fwd model
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
    parser.add_argument("--model_based", default="None")  # What model should we use choose from forward / backward / None
    parser.add_argument("--model_iters", default=100, type=int)  # Frequency of delayed policy updates

    args = parser.parse_args()

    file_name = "%s_%s_%s_%s" % (args.policy_name, args.env_name, str(args.model_based), str(args.seed))
    print("---------------------------------------")
    print(f"Settings: {file_name}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    experiment_directory_name = \
    str(args.policy_name) + \
    '_env_' + str(args.env_name) + \
    '_seed_'+str(args.seed) + \
    '_model_based_' + str(args.model_based) + \
    '_max_timesteps_' + str(args.max_timesteps) + \
    '_model_iters_' + str(args.model_iters)

    writer = SummaryWriter(log_dir='results/TrainingLogs/' + experiment_directory_name)


    env = gym.make(args.env_name)
    print('-- CREATED ENVIRONMENT -- ')

    set_seed(env, seed=args.seed)   # set seeds


    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.model_based == "forward":
        forward_dynamics_model = Net(n_feature=state_dim+action_dim,
                                     n_hidden=64,
                                     n_output=state_dim+1 # reward is the + 1
                                     ).cuda()

    elif args.model_based == "backward":
        backward_dynamics_model = Net(n_feature=state_dim + action_dim,
                                     n_hidden=64,
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

    if args.load_model == "None":
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

    ########### SETTING REPLAY BUFFERS HERE ###########
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    # introduce 2 new replay buffers to store synthetic transitions
    if args.model_based == "forward":
        fwd_model_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e7))

    if args.model_based == "backward":
        bwd_model_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e7))


    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env_name, args.seed)]

    state, done = env.reset(), False
    episode_reward, episode_timesteps, episode_num = 0, 0, 0

    if args.model_based is not "None":
        print(' ~~ MODEL BASED METHOD IN USE ~~')
        Ts = [] # list of trajectories
        T = [[]] # buffer for storing a single trajectory
        first_update = False
    else:
        print('~~ MODEL FREE METHOD ~~')

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
        if args.model_based is not "None":
            T[0].append((state, action, reward))  # append the state, action and reward into trajectory

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        # update the forward and backward models here
        if args.model_based == "forward":
            if (not first_update and t >= 1000) or (t >= args.fwd_model_update_freq and t % args.fwd_model_update_freq == 0):
                fwd_norm = update_forward_model(forward_dynamics_model, Ts, checkpoint_name=file_name)
                first_update = True # done

        if args.model_based == "backward":
            if (not first_update and t >= 1000) or (t >= args.bwd_model_update_freq and t % args.bwd_model_update_freq == 0):
                bwd_norm = update_backward_model(backward_dynamics_model, Ts, checkpoint_name=file_name)
                first_update = True # done

        if args.model_based == "forward":
            # add imagined trajectories from fwd model into fwd_model_replay_buffer
            if first_update: # model has been updated atleast once
                forward_dynamics_model.eval() # model has a dropout layer !

                for _ in range(args.model_iters): # collect 100 times the data
                    t_s, t_a, t_ns, t_r, t_nd = replay_buffer.sample(args.batch_size)
                    t_s = t_s.cpu().numpy()
                    t_a = t_a.cpu().numpy()

                    for model_depth in range(args.imagination_depth):
                        # add noise to actions and predict
                        t_a =  (t_a + np.random.normal(0, max_action*args.expl_noise/10,
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
                            fwd_model_replay_buffer.add(t_s[k], t_a[k], t_ns[k], t_r[k], False) # store predicted forward transition in buffer

                        if args.imagination_depth > 1:
                            # get ready for next transition
                            t_s = t_ns
                            print('Aquiring samples of next actions and states to query ~forward model~')
                            for k in trange(t_s.shape[0]):
                                t_a[k] = (policy.select_action(np.array(t_s[k]))
                                       + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        if args.model_based == "backward":
            # add imagined trajectories from fwd model into fwd_model_replay_buffer
            if first_update: # model has been updated atleast once
                backward_dynamics_model.eval() # model has a dropout layer !

                for _ in range(args.model_iters): # collect 100 times the data
                    t_s, t_a, t_ns, t_r, t_nd = replay_buffer.sample(args.batch_size)
                    t_s = t_s.cpu().numpy()
                    t_a = t_a.cpu().numpy()

                    for model_depth in range(args.imagination_depth):
                        # add noise to actions and predict
                        t_a =  (t_a + np.random.normal(0, max_action*args.expl_noise/10,
                                                       (t_a.shape[0], t_a.shape[1]))).clip(-max_action, max_action)

                        bwd_input = np.hstack((t_s, t_a)) # using the next state in backward dynamics
                        bwd_input = apply_norm(bwd_input, bwd_norm[0]) # normalize the data before feeding in

                        bwd_input = torch.tensor(bwd_input).float().cuda()
                        bwd_output = backward_dynamics_model.forward(bwd_input)
                        bwd_output = bwd_output.detach().cpu().numpy()

                        bwd_output = unapply_norm(bwd_output, bwd_norm[1]) # unnormalize the output data

                        t_ps = bwd_output[:, :-1] + t_s # predicted previous state = predicted delta previous state + next state
                        t_r = bwd_output[:, -1] # predicted reward

                        for k in range(t_s.shape[0]):
                            bwd_model_replay_buffer.add(t_ps[k], t_a[k], t_s[k], t_r[k], False) # store predicted backward transition in buffer

                        if args.imagination_depth > 1:
                            # get ready for next transition
                            t_s = t_ps
                            print('Aquiring samples of next actions and states to query ~backward model~')
                            for k in trange(t_s.shape[0]):
                                t_a[k] = (policy.select_action(np.array(t_s[k]))
                                       + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        # Real world data
        if args.model_based == "None":
            if t >= args.batch_size:
                policy.train(replay_buffer, args.batch_size)

        # Imagined data (forward)
        elif args.model_based == "forward":
            if t >= args.batch_size and t >= args.fwd_model_update_freq:
                for _ in range(10):
                    policy.train(fwd_model_replay_buffer, args.batch_size*10)

        # Imagined data (backward)
        elif args.model_based == "backward":
            if t >= args.batch_size and t >= args.bwd_model_update_freq:
                for _ in range(10):
                    policy.train(bwd_model_replay_buffer, args.batch_size*10)

        else:
            print('Something wrong.')

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
            if args.model_based is not "None":
                Ts.extend(T)
                T = [[]]  #

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env_name, args.seed))
            np.save("./results/%s" % (file_name), evaluations)

    # save the model
    print('-- SAVING THE MODEL --')
    if args.save_models:
        with open("./results/TrainingLogs/" + experiment_directory_name +"/" + "model.pkl", "wb") as file_:
            pickle.dump(policy, file_, -1)
    print('-- MODEL SAVED --')

    writer.close()
