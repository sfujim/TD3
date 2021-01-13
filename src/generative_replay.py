import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gym
from tqdm import tqdm
import random
from torch.autograd import Variable

# Hyper parameters
BATCH_SIZE = 64
EPOCHS = 20
INPUT_SIZE = 11
LAYER_SIZE = 10
LATENT_SIZE = 3 
LEARNING_RATE = 0.001
BUFFER_SIZE = BATCH_SIZE
TRAIN_TO_TEST = 0.9
C = 0.01

# Env sizes
state_dim = 4
action_dim = 1
action_low = -3.0
action_high = 3.0
state_low = -15
state_high = 15

# Make CPU if cuda doesnt work
device = "cuda"  


class GenerativeReplay:
    def __init__(self):
        self.model = VAE().to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.buffer = [None for x in range(int(BUFFER_SIZE))]
        self.training = False
        self.first = True
        self.i = 0
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=240, profile=None)


    # Add new experiences as they come
    def add(self, state, action, next_state, reward, done):
        experience = [s for s in state]
        experience.append(action)
        experience.extend([s for s in next_state])
        experience.extend([reward, done])
        experience = self.normalize(experience)

        self.buffer[self.i] = experience
        self.i += 1

        if self.i >= BUFFER_SIZE:
            self.i = 0
            self.train()
            self.test()



    def loss_function(self, recon_x, x, mu, sigma):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return BCE + C*KLD


    # Train the model with what we have in the buffer and some generated data
    def train(self):
        self.model.train()
        train_data = self.buffer[:int(len(self.buffer)*TRAIN_TO_TEST)]
        for w in range(EPOCHS):
            train_loss = 0

            for i in range(0, len(train_data), BATCH_SIZE):
                self.model.zero_grad()

                batch = torch.FloatTensor(train_data[i:i+BATCH_SIZE]).to(device)
                recons, mu, sigma = self.model(batch)

                loss = self.loss_function(recons, batch, mu, sigma)
                loss.backward()
                train_loss += loss.item()

                self.opt.step()

            # print(f"Trained the VAE with loss :{(train_loss/len(train_data))*100}")

        

    # Test the model for stats
    def test(self):
        self.model.eval()
        test_data = self.buffer[int(len(self.buffer)*TRAIN_TO_TEST):]

        test_loss = 0
        for i in range(0, len(test_data), BATCH_SIZE):
            batch =  torch.FloatTensor(test_data[i:i+BATCH_SIZE]).to(device)
            recons, mu, sigma = self.model(batch)
            loss = self.loss_function(recons, batch, mu, sigma)
            test_loss += loss.item()

        print(f"Tested the VAE with loss {(test_loss/len(test_data))*100}") 


    # Given an experience from the env, make an array that can fit the model
    def normalize(self, experience):
        # [s0, s1, s2, s3, a, s0, s1, s2, s3, r, d]
        return np.concatenate((
            self.normalize_state(experience[:state_dim]),
            np.array(self.normalize_action(experience[state_dim])), # TODO: maybe turn into array, if we dont get it as such
            self.normalize_state(experience[state_dim+1:state_dim+1+state_dim]),
            self.normalize_reward(experience[-2]),
            np.array([experience[-1]])
        ))

    def normalize_action(self, x):
        return (x-action_low)/(action_high-action_low)

    def normalize_state(self, s):
        res = []
        for i in s:
            res.append((i-state_low)/(state_high-state_low))
        return res

    def normalize_reward(self, r):
        return np.array([r/20.0])   


    # Descale the output of the network to real values
    def descale(self, x):
        # State
        ((x[:, 0].mul_(state_high-state_low)).add_(state_low))
        ((x[:, 1].mul_(state_high-state_low)).add_(state_low))
        ((x[:, 2].mul_(state_high-state_low)).add_(state_low))
        ((x[:, 3].mul_(state_high-state_low)).add_(state_low))

        # Action
        ((x[:, 4].mul_(action_high-action_low)).add_(action_low))
        
        # State
        ((x[:, 5].mul_(state_high-state_low)).add_(state_low))
        ((x[:, 6].mul_(state_high-state_low)).add_(state_low))
        ((x[:, 7].mul_(state_high-state_low)).add_(state_low))
        ((x[:, 8].mul_(state_high-state_low)).add_(state_low))
        
        # Reward
        ((x[:, 9].mul_(20.0)).round_())
        
        # Done
        (x[:, 10].round_())
        
        return x


    # Print some samples in the evaluestion for error checking
    def eval(self, state, action, nstate, reward, done):
        # torch.set_printoptions(precision=3, sci_mode=False, linewidth=240, profile="short")
        pass

        # print(self.sample(1))

        # print(action)


    # Sample a give amount of new experiences from the model
    # The output should be in GPU
    def sample(self, amount):

        with torch.no_grad():

            sample_batch = self.get_latent_samples(amount)
            
            outputs = self.descale(self.model.decode(sample_batch)).to("cpu")


            return (
                torch.FloatTensor(outputs[:, 0:4]).to(device),
                torch.FloatTensor(outputs[:, 4]).unsqueeze(1).to(device),
                torch.FloatTensor(outputs[:, 5:9]).to(device),
                torch.FloatTensor(outputs[:, -2]).unsqueeze(1).to(device),
                torch.FloatTensor(outputs[:, -1]).unsqueeze(1).to(device)
            )

            
    # Random some from latents
    def get_latent_samples(self, amount):

        res = torch.rand(amount, LATENT_SIZE)
        ((res.mul_(10)).sub_(5))
        # TODO change this scaling based on the latent space created dynamically
        res = res.to(device)
        return res


# Variational Autoencoder that mostly I built
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.l1 = nn.Linear(INPUT_SIZE, LAYER_SIZE)
        self.l2a = nn.Linear(LAYER_SIZE, LATENT_SIZE)
        self.l2b = nn.Linear(LAYER_SIZE, LATENT_SIZE)
        
        # Decoder
        self.l3 = nn.Linear(LATENT_SIZE, LAYER_SIZE)
        self.l4 = nn.Linear(LAYER_SIZE, INPUT_SIZE)

        self.latents = []
        self.l_size = 1e3

    
    def encode(self, x):
        out = F.relu(self.l1(x))
        return self.l2a(out), self.l2b(out)
    
    # Taken from https://github.com/pytorch/examples/blob/master/vae/main.py
    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)

        res = mu + eps*std
        res = torch.clamp(res, min=-5.0, max=5.0)
        return res
    
    def decode(self, x):
        out = F.relu(self.l3(x))
        return torch.sigmoid(self.l4(out))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma


