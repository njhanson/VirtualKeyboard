# imports
import snntorch as snn
from snntorch import utils

import torch
import torch.nn as nn

def createSNN(dim_inputs, hidden_layer, num_outputs=2, betas=[0.9, 0.9], thresholds=[1, 1]):
    """
    Function wrapper that initiates a fully connected 3 layer SNN model
    
    Inputs:
    - dim_inputs: dimension size of the input features (flattened)
    - hidden_layer: number of neurons in the hidden layer of the model
    - num_outputs: number of output neurons/classes in the model. Default = 2
    - betas: decay constant for each LIF neuron layer as an array. Default = [0.9, 0.9]
    - thresholds: array of membrane potential thresholds for the neurons in each layer to produce a spike, Default = [1, 1]

    Returns: 
    - A fully connected 3 layer spiking neural network with the specified parameters
    """
    return fcSNN(dim_inputs=dim_inputs, hidden_layer=hidden_layer, num_outputs=num_outputs, betas=betas, thresholds=thresholds)


class fcSNN(nn.Module):
    def __init__(self, dim_inputs, hidden_layer, num_outputs, betas, thresholds):
        super().__init__()

        # initializes lif and linear layers for the SNN
        self.fc1 = nn.Linear(dim_inputs, hidden_layer)
        self.lif1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], init_hidden=True)
        self.fc2 = nn.Linear(hidden_layer, num_outputs)
        self.lif2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], init_hidden=True, output=True)

        # initializes fully connected layer weights and biases in a small normal distribution.
        # weights are skewed positive to encourage positive membrane potentials and to produce output spikes
        nn.init.normal_(self.fc1.weight, mean=0.015, std=0.007)
        nn.init.normal_(self.fc2.weight, mean=0.01, std=0.02)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=0.001)
        nn.init.normal_(self.fc2.bias, mean=0.0, std=0.01)
    
    def forward(self, x, batch_first=False):
        """
        does a forward pass of the model on data x
        
        Inputs:
        - x: data to be passed into model for a forward pass. In the form of (time steps x batch x feature dimension) or
                       (batch x time steps x feature dimension)
        - batch_first: Is True if batch is the first dimension. If not specified, assumes batch_first is False

        Returns: 
        - tensor containing the raw output spike data of shape (time x batch x num_outputs)
        - tensor containing the membrane potential of the output neurons of shape (time x batch x num_outputs)
        """
        # transposes x to the form of (time x batch x flattened feature dimension) if not already in that form
        if(batch_first):
            x = x.transpose(0, 1)
        x = torch.flatten(x, start_dim=2)

        utils.reset(self)

        # record final layer
        spk2_rec = []
        mem2_rec = []

        # through the time steps of the data
        for step in range(x.size(0)): # number of time steps in x
            cur1 = self.fc1(x[step])
            spk1 = self.lif1(cur1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)