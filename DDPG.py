import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


def hidden_init(layer):
  fan_in = layer.weight.data.size()[0]
  lim = 1. / np.sqrt(fan_in)
  return (-lim, lim)

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()

    # self.seed = torch.manual_seed(88)
    self.fc1 = nn.Linear(state_dim, 400)
    self.fc2 = nn.Linear(400, 300)
    self.fc3 = nn.Linear(300, action_dim)
  
  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    return F.torch.tanh(self.fc3(x))


class Critic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()

    # self.seed = torch.manual_seed(88)
    self.l1 = nn.Linear(state_dim + action_dim, 400)
    self.l2 = nn.Linear(400, 300)
    self.l3 = nn.Linear(300, 1)


  def forward(self, state, action):
    x = torch.cat([state, action], dim=1)
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    return self.l3(x)


class DDPG(object):
  def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001,
               lr_actor=1e-3, lr_critic=1e-3, critic_weight_decay=0.0):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=critic_weight_decay)

    self.discount = discount
    self.tau = tau


  def select_action(self, state):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    # state = state.unsqueeze(0)
    return self.actor(state).cpu().data.numpy().flatten()


  def train(self, replay_buffer, batch_size=64):
    # Sample replay buffer 
    state, action, next_state, reward, done = replay_buffer.sample(batch_size)

    # Compute the target Q value
    target_Q = self.critic_target(next_state, self.actor_target(next_state))
    target_Q = reward + ((1.0 - done) * self.discount * target_Q).detach()

    # Get current Q estimate
    current_Q = self.critic(state, action)

    # Compute critic loss
    critic_loss = F.mse_loss(current_Q, target_Q)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Compute actor loss
    actor_loss = -self.critic(state, self.actor(state)).mean()
    
    # Optimize the actor 
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Update the frozen target models
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


  def save(self, filename):
    torch.save(self.critic.state_dict(), filename + "_critic")
    torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    torch.save(self.actor.state_dict(), filename + "_actor")
    torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


  def load(self, filename):
    self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
    self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
    self.critic_target = copy.deepcopy(self.critic)

    self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
    self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
    self.actor_target = copy.deepcopy(self.actor)
    
