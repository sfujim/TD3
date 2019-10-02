import torch
import torch.nn as nn

class Net(torch.nn.Module):
    """network that defines the forward dynamics model"""
    def __init__(self, n_feature, n_hidden, n_output, activations=nn.ReLU, action_activation=False):
        super(Net, self).__init__()

        self.fc_in = nn.Linear(n_feature, n_hidden)
        self.fc_h1 = nn.Linear(n_hidden, n_hidden)
        self.fc_h2 = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(0.5)

        self.fc_out = nn.Linear(n_hidden, n_output)
        self.last_activation = action_activation
        self.activations = activations

    # pylint: disable=arguments-differ
    def forward(self, x):
        out = self.activations()(self.fc_in(x))
        out = self.activations()(self.fc_h1(out))
        out = self.dropout(out)
        out = self.activations()(self.fc_h2(out))
        out = self.fc_out(out)

        if self.last_activation:
            return nn.Tanh()(out) # using a tanh activation for actions ?

        return out