# Quantum
import pennylane as qml

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Numpy, Pandas
import numpy as np
import pandas as pd

# Data processing
from utils import utils

class RNNQE(nn.Module):
    def __init__(self, n_feature, num_layers):
        '''
            Args:
                n_feature(int) : # of feature
                rnn_args(int) : # of layer iteration
        '''
        super(RNNQE, self).__init__()

        self.util = utils(n_qu = n_feature)
        self.n_qu = n_feature
        self.quantum_layer = self.util.fidelity

        self.rnn = nn.RNN(input_size=n_feature, hidden_size=(2 * n_feature - 1), num_layers=num_layers, batch_first=True)

    def forward_input_RNN(self, inputs):
        """
            TODO :
                주어진 sequenced feature(input)에 대한 angle을 rnn으로 생성
            Args :
                inputs(torch.tensor) : feature 값을 담은 tensor, shape : (batch_size, seq_len, feature_size)
            Return :
                result(torch.tensor) : 주어진 sequenced feature에 대해 생성된 angle, shape : (batch_size, 2 * feature_size - 1)
        """
        output, _ = self.rnn(inputs) # _ is hidden
        output = output.transpose(0, 1)
        result = 2 * torch.pi * F.relu(output[-1])
        # result = 2 * torch.pi * output[-1]
        
        return result
    
    def forward_RNN(self, inputs):
        # Shape (batch_size, 2, len_sequence, feature) -> (2, batch_size, len_sequence, feature)로 transpose
        inputs = inputs.transpose(0, 1)
        input1 = inputs[0]
        input2 = inputs[1]
        output1 = self.forward_input_RNN(input1)
        output2 = self.forward_input_RNN(input2)

        output = self.quantum_layer(output1, output2, self.n_qu)[ : , 0]
        output = output.unsqueeze(1)
        return output

    def forward(self, inputs):
        return self.forward_RNN(inputs)