import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import utils


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)


def var(tensor, volatile=False):
	if torch.cuda.is_available():
		return Variable(tensor, volatile=volatile).cuda()
	else:
		return Variable(tensor, volatile=volatile)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * F.tanh(self.l3(x)) 
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, x, u):
		x1 = F.relu(self.l1(torch.cat([x, u], 1)))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(torch.cat([x, u], 1)))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)

		return x1, x2


class TD3(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action)
		self.actor_target = Actor(state_dim, action_dim, max_action)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())		

		if torch.cuda.is_available():
			self.actor = self.actor.cuda()
			self.actor_target = self.actor_target.cuda()
			self.critic = self.critic.cuda()
			self.critic_target = self.critic_target.cuda()

		self.criterion = nn.MSELoss()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action


	def select_action(self, state):
		state = var(torch.FloatTensor(state.reshape(-1, self.state_dim)), volatile=True)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = var(torch.FloatTensor(x))
			action = var(torch.FloatTensor(u))
			next_state = var(torch.FloatTensor(y), volatile=True)
			done = var(torch.FloatTensor(1 - d))
			reward = var(torch.FloatTensor(r))

			# Select action according to policy and add clipped noise 
			noise = np.clip(np.random.normal(0, policy_noise, size=(batch_size, self.action_dim)), -noise_clip, noise_clip)
			next_action = self.actor_target(next_state) + var(torch.FloatTensor(noise))
			next_action = next_action.clamp(-self.max_action, self.max_action)

			# Q target = reward + discount * min(Qi(next_state, pi(next_state)))
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(torch.cat([target_Q1, target_Q2], 1), 1)[0].view(-1, 1)
			target_Q.volatile = False 
			target_Q = reward + (done * discount * target_Q)

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = self.criterion(current_Q1, target_Q) + self.criterion(current_Q2, target_Q) 

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if it % policy_freq == 0:

				# Compute actor loss
				Q1, Q2 = self.critic(state, self.actor(state)) 
				actor_loss = -Q1.mean()
				
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
