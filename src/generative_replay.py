import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gym
from tqdm import tqdm

BATCH_SIZE = 64
EPOCHS = 50
INPUT_SIZE = 11
LAYER_SIZE = 10
LATENT_SIZE = 3 
LEARNING_RATE = 0.001
BUFFER_SIZE = 1024

# Env sizes
state_dim = 4
action_dim = 1
action_low = -3.0
action_high = 3.0
state_low = -15
state_high = 15


# Make CPU if cuda doesnt work
device = "cuda"  


class GenerativeReplay():
    def __init__(self):
        # Model
        self.model = VAE().to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.buffer = []      

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) >= BUFFER_SIZE:
            self.train()
            self.test()
            self.buffer = []

    def sample():
        pass

    def train():
        pass

    def test():
        pass

    def normalize(experience):
        # [s0, s1, s2, s3, a, s0, s1, s2, s3, r, d]
        res = []

    def normalize_action(x):
        return (x-action_low)/(action_high-action_low)
    def normalize_state(s):
        res = []
        for i in s:
            res.append((i-state_low)/(state_high-state_low))
        return res
    def normalize_reward(r):
        return np.array([r/20.0])   



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

    
    def encode(self, x):
        out = F.relu(self.l1(x))
        return self.l2a(out), self.l2b(out)
    
    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, x):
        out = F.relu(self.l3(x))
        return torch.sigmoid(self.l4(out))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma

    def loss_function(recon_x, x, mu, sigma):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return BCE + KLD
