from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_hidden(layer):
    input_size = layer.weight.data.size()[0]
    lim = np.sqrt(1. / input_size)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, input_size, output_size, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        init_hidden_num = 128
        self.num_layers = 2

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, init_hidden_num)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(num_features=init_hidden_num)])
        for i in range(self.num_layers):
            self.hidden_layers.append(nn.Linear(init_hidden_num*(2**i), init_hidden_num*(2**(i+1))))
            self.bn_layers.append(nn.BatchNorm1d(num_features=init_hidden_num*(2**(i+1))))
        self.final_layer = nn.Linear(init_hidden_num*(2**self.num_layers), output_size)
        self.init_params()

    def init_params(self):
        for i in range(len(self.hidden_layers)):
            nn.init.uniform_(self.hidden_layers[i].weight.data,*init_hidden(self.hidden_layers[i]))
        nn.init.uniform_(self.final_layer.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x = state
        for i in range(len(self.hidden_layers)):
            x = torch.relu(self.hidden_layers[i](x))
            x = self.bn_layers[i](x)
        x = torch.tanh(self.final_layer(x))
        return x

class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        init_hidden_num = 128
        self.num_layers = 2
        input_size = state_size + action_size
        output_size = 1

        #norm layer for state
        self.state_bn_layer = nn.BatchNorm1d(num_features=state_size)
        self.state_linear_layer = nn.Linear(state_size, state_size)

        #hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, init_hidden_num)])
        for i in range(self.num_layers):
            self.hidden_layers.append(nn.Linear(init_hidden_num*(2**i), init_hidden_num*(2**(i+1))))
        self.final_layer = nn.Linear(init_hidden_num*(2**self.num_layers), output_size)

        self.init_params()

    def init_params(self):
        nn.init.uniform_(self.state_linear_layer.weight.data,*init_hidden(self.state_linear_layer))
        for i in range(len(self.hidden_layers)):
            nn.init.uniform_(self.hidden_layers[i].weight.data,*init_hidden(self.hidden_layers[i]))
        nn.init.uniform_(self.final_layer.weight.data, -3e-3, 3e-3)

    def norm_state_layer(self,x):
        x = torch.relu(self.state_linear_layer(x))
        x = self.state_bn_layer(x)
        return x

    def forward(self, state, action):
        norm_state = self.norm_state_layer(state)
        x = torch.cat((norm_state, action), dim=1)
        for i in range(len(self.hidden_layers)):
            x = torch.relu(self.hidden_layers[i](x))
        x = self.final_layer(x)
        return x