import numpy as np
import torch
from tqdm import trange
from tensorboardX import SummaryWriter
from random import random
from torch.utils import data
import torch.nn as nn


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

def apply_norm(dataset, norm):
    """Normalizes data given a (mean, std) tuple"""
    return (dataset - norm[0]) / norm[1]

def unapply_norm(dataset, norm):
    """Inverse operation of _apply_norm"""
    return dataset*norm[1] + norm[0]

def train_model_es(model,
                   x_list,
                   y_list,
                   optimizer,
                   criterion=torch.nn.MSELoss(),
                   max_epochs=100,
                   v_size=0.2,
                   patience=10,
				   num_cores=30,
				   checkpoint_name='checkpoint.pt'):
    """Trains a model from given labels with early stopping

    :param model: the model to be trained
    :type model: torch.nn.module
    :param x_list: list of features (same size as y_list)
    :type x_list: list
    :param y_list: list of labels (same size as x_list)
    :type y_list: list
    :param optimizer: optimizer
    :type optimizer: optimizer
    :param criterion: criterion
    :type criterion: criterion
    :param max_epochs: maximum number of epochs
    :type max_epochs: int
    :param v_size: size of validation set [0,1)
    :type v_size: float
    :param patience: number of epochs to wait before early stopping
    :type patience: int
    """
    # for visualizing plots on tensorboard
    # writer = SummaryWriter(log_dir='data/TrainingLogs/'+str(int(random()*100000)))

    valid_size = int(np.floor(len(x_list) * v_size))
    train_size = len(x_list) - valid_size

    # Convert to nd-array
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    # Convert to torch tensor
    x_list = torch.tensor(x_list).float()
    y_list = torch.tensor(y_list).float()

    # Split into training and validation sets
    train_data, valid_data = data.random_split(
        data.TensorDataset(x_list, y_list),
        [train_size, valid_size])

    train_loader = data.DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=1024,
        num_workers=0)

    valid_loader = data.DataLoader(
        dataset=valid_data,
        shuffle=True,
        batch_size=1024,
        num_workers=0)

    # train the model
    epoch = 0
    best_epoch = 0
    best_loss = None
    while (epoch < max_epochs and
           epoch < best_epoch+patience):
        epoch += 1
        train_losses = []
        valid_losses = []

        # Train for one epoch
        model.train() # prep for training
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)
            print('\repoch: '+str(epoch)+ ' loss: '+str(loss.item()), end='')
            # Zero the gradients
            optimizer.zero_grad()
            # perform a backward pass (backpropagation)
            loss.backward()
            # Update the parameters
            optimizer.step()
            train_losses.append(loss.item())

        # Validate model
        model.eval() # prep for evaluation
        for x_batch, y_batch in valid_loader:
            x_batch = x_batch.to('cuda')
            y_batch = y_batch.to('cuda')
            y_pred = model.forward(x_batch)
            loss = criterion(y_pred, y_batch)
            valid_losses.append(loss.item())

        # writer.add_scalars('data/err_vs_epoch',
        #                    {'train_loss' : float(np.average(train_losses))},
        #                    epoch)
        # writer.add_scalars('data/err_vs_epoch',
        #                    {'valid_loss' : float(np.average(valid_losses))},
        #                    epoch)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        print('\ntraining loss:', train_loss)
        print('validation loss:', valid_loss)
        if best_loss is None or valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_name)
        print('current best:', best_loss, '(epoch', best_epoch, ')')
    model.load_state_dict(torch.load(checkpoint_name))
    # writer.close()

def update_forward_model(model, Ts, checkpoint_name='xyz'):
	"""
	function to update the forward dynamics model
	:param model: ANN defining the forward dynamics model
	:param Ts: Transitions stored in a list
	:return: norm of input and output for the current trained model
	"""
	print('--- UPDATING THE FORWARD DYNAMICS MODEL --- ')
	print('           --- PARSING DATA ---             ')

	X_list = []
	Y_list = []

	for T in Ts:
		for i in range(len(T)-1):
			X_list.append(np.hstack((T[i][0], T[i][1]))) # current state, current action
			Y_list.append(np.hstack((T[i+1][0]-T[i][0], T[i][2]))) # delta next state, current reward

	X_norm = (np.mean(X_list, axis=0), np.std(X_list, axis=0))
	Y_norm = (np.mean(Y_list, axis=0), np.std(Y_list, axis=0))

	X_list_normalized = apply_norm(X_list ,X_norm)
	Y_list_normalized = apply_norm(Y_list, Y_norm)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	print('--- TRAINING THE FORWARD DYNAMICS MODEL --- ')

	train_model_es(model,
				   X_list_normalized,
				   Y_list_normalized,
				   optimizer,
				   criterion,
				   checkpoint_name=checkpoint_name+'_fwd.pt')

	print('            --- FORWARD MODEL TRAINING DONE ---           ')

	return X_norm, Y_norm

def update_backward_model(model, Ts, checkpoint_name='xyz'):
	"""
	function to update the backward dynamics model
	:param model: ANN defining the forward dynamics model
	:param Ts: Transitions stored in a list
	:return: norm of input and output for the current trained model
	"""
	print('--- UPDATING THE BACKWARD DYNAMICS MODEL --- ')
	print('           --- PARSING DATA ---             ')

	X_list = []
	Y_list = []

	for T in Ts:
		for i in range(len(T)-1):
			X_list.append(np.hstack((T[i+1][0], T[i][1]))) # next state, current action
			Y_list.append(np.hstack((T[i][0]-T[i+1][0], T[i][2]))) # delta current state, current reward

	X_norm = (np.mean(X_list, axis=0), np.std(X_list, axis=0))
	Y_norm = (np.mean(Y_list, axis=0), np.std(Y_list, axis=0))

	X_list_normalized = apply_norm(X_list,X_norm)
	Y_list_normalized = apply_norm(Y_list,Y_norm)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	print('--- TRAINING THE BACKWARD DYNAMICS MODEL --- ')

	train_model_es(model,
				   X_list_normalized,
				   Y_list_normalized,
				   optimizer,
				   criterion,
				   checkpoint_name=checkpoint_name+'_bwd.pt')

	print('            --- TRAINING DONE ---           ')

	return X_norm, Y_norm


def set_seed(env, seed):
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)