import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

BATCH_SIZE = 64
EPOCHS = 50
INPUT_SIZE = 11
LAYER_SIZE = 10
LATENT_SIZE = 3   


class GenerativeReplay():
    def __init__(self):
        pass

    def add():
        pass

    def sample():
        pass

    def train():
        pass

    def test():
        pass





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


