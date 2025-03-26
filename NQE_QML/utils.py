# Quantum
import pennylane as qml

# numpy
import numpy as np

# PyTorch
import torch
import torch.nn.functional as F


class utils:

    def __init__(self, n_qu):
        self.dev = qml.device('default.qubit', wires = n_qu)
        self.n_qu = n_qu       

    def embedding(self, params, n_qu):
        '''
        embedding layer
        '''
        n = n_qu
        for i in range(n):
            qml.Hadamard(i)
            qml.RZ(2.0 * params[ : , i], i)
        
        for i in range(n - 1):
            qml.IsingZZ(2.0 * params[ : , n + i] , [i, i + 1])

    def fidelity(self, vec1, vec2, n_qu):
        @qml.qnode(self.dev, interface = "torch")
        def inner_fidelity(vec1, vec2, n_qu):
            '''
                Args:
                    vec1 : list, (2n - 1)개의 element로 이루어진 vector
                    vec2 : list, (2n - 1)개의 element로 이루어진 vector
            '''
            self.embedding(vec1, n_qu) # Phi(x1) circuit 적용
            qml.adjoint(self.embedding)(vec2, n_qu) # Phi^t(x2) 적용
            return qml.probs()
        return inner_fidelity(vec1, vec2, n_qu)