# Quantum
import pennylane as qml

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Numpy, Pandas
import numpy as np
import pandas as pd

# Data processing
from sklearn.preprocessing import MinMaxScaler

# Numpy
import numpy as np

# Plot
import matplotlib.pyplot as plt

# User-Defined Classes
from NQE_class import RNNQE
from trainer import trainer
from data_prepare import data_prepare

class qml_model(nn.Module):
    def __init__(self, num_qubit, num_layer, nqe_model=None, data_reuploading=True):
        """
            nun_qubit(int) : 사용할 qubit 개수
            num_layer(int) : QML 모델이 반복할 총 layer의 횟수 (quantum layer 개수)
            nqe_model(RNNQE) : 사용할 NQE 객체
            data_reuploading(bool) : data reuploading 여부
        """
        super(qml_model, self).__init__()

        ## Instance Initialize ##
        self.num_qubit = num_qubit
        self.num_layer = num_layer
        self.data_reuploading = data_reuploading
        self.nqe = nqe_model

        ## Parameter Generation ##
        self.required_parameters = 3 * self.num_qubit * num_layer
        self.theta = nn.Parameter(
            torch.rand(self.required_parameters) * 2 * torch.pi, requires_grad=True
        )  # parameter 초기화 시 범위(0 ~ 2 * pi?) 적용

        ## Quantum Device Initialize ##
        self.device = qml.device("default.qubit", wires=num_qubit)
    
    def get_angles(self, sequenced_input):
        """
            TODO :
                sequenced input을 받아 Angle로 변환하여, self의 instance로 저장 및 return
            Args :
                sequenced_input(torch.tensor) : input(feature)의 sequence, shape : (batch, seq_len, feature_size)
        """
        if not self.nqe:
            last_sequence = sequenced_input[:, -1, :]  # Shape: (64, n)
    
            # 원래 데이터 복사 (64, n)
            first_n = last_sequence.clone()
            
            # phi(x, y) 계산 (64, n-1)
            phi_values = (torch.pi - last_sequence[:, :-1]) * (torch.pi - last_sequence[:, 1:])
            
            # 최종 데이터 결합 (64, 2*n-1)
            processed_data = torch.cat([first_n, phi_values], dim=1)
            return processed_data
        nqe_result = self.nqe.forward_input_RNN(sequenced_input)
        self.upload_angle = nqe_result
        return self.upload_angle
    
    def data_upload(self, angles):
        """
            TODO :
                sequenced input에 대해 변환된 Angle을 받아,, ZZ feature map으로 Embedding하는 회로에 적용
            Args :
                angles(torch.tensor) : sequenced_input에 대해 생성된 angles, shape : (batch, 2 * feature size - 1)
        """
        n = self.num_qubit

        ## Single Qubit Gates ##
        for i in range(n):
            qml.Hadamard(wires=i)
            qml.RZ(2.0 * angles[ : , i], i)
        
        ## Two Qubit Gates ##
        for i in range(n - 1):
            qml.IsingZZ(2.0 * angles[ : , n + i] , [i, i + 1])

    def unit_layer(self, theta, fully_entangle=False):
        """
            TODO : 
                RX RY RZ CNOT으로 이루어진 단위 layer
            Args :
                theta(list or tensor) : 1개의 layer에 대한 (3 * num_qubit)개의 parameter set
                fully_entangle(bool) : True -> last qubit과 first qubit에까지 CNOT을 적용
        """
        for i in range(self.num_qubit):
            qml.RX(theta[3 * i], wires=i)
            qml.RY(theta[3 * i + 1], wires=i)
            qml.RZ(theta[3 * i + 2], wires=i)

        for i in range(self.num_qubit - 1):
            qml.CNOT(wires=[i, i + 1])
        if fully_entangle:
            qml.CNOT(wires=[self.num_qubit - 1, 0])
    
    def outer_forward(self, sequenced_input):
        """
            TODO
                1. sequenced_input에 대한 NQE 결과, 즉 angle을 저장
                2. num_layer만큼 layer를 반복하여, data reuploading과 ansatz 적용(ansatz에 self.theta 사용)
                3. Measure (qml.expval)
            Args
                sequenced_input(tensor) : sequenced_input(torch.tensor) : input(feature)의 sequence, shape : (batch, seq_len, feature_size)
        """
        ## Get NQE Result ##
        nqe_result = self.get_angles(sequenced_input=sequenced_input)

        ## Layer Iteration ##
        for i in range(self.num_layer):
            ## Data Reuploading ##
            self.data_upload(nqe_result)
            qml.Barrier()

            ## Ansatz ##
            self.unit_layer(theta=self.theta[i * (3 * self.num_qubit) : (i + 1) * (3 * self.num_qubit)])
            qml.Barrier()
        return qml.expval(qml.PauliZ(0))

    def forward(self, sequenced_input, chk=False):
        """
            TODO :

            Args :
                sequenced_input(torch.tensor) : features with sequence, shape : (batch, seq_len, feature_size)
        """
        @qml.qnode(self.device)
        def inner_forward(input):
            result = self.outer_forward(sequenced_input=input)
            return result
        
        if chk:
            qml.draw_mpl(inner_forward, style="pennylane")(sequenced_input)
            return
        
        output = inner_forward(input=sequenced_input)
        output = output.reshape(-1, 1) # ??
        return output.float()  # -1 ~ 1까지만의 value 가질 수 있음