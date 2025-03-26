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

# Numpy
import numpy as np


class data_prepare:
    def __init__(self, feature_csv="../data/02_PCA/pca_train_data_Perth.csv", label_csv="../data/02_PCA/pca_label_data_Perth.csv"):
        self.feature_df = pd.read_csv(feature_csv)
        self.label_df = pd.read_csv(label_csv)

    def make_dataset(self, len_sequence=8):
        '''
            TODO :
                self의 instance인 feature_df와 label_df를 활용하여, 
                
        '''
        feature_list = self.feature_df.values
        label_list = self.label_df.values

        if len_sequence == 1:
            feature_tensor = torch.tensor(feature_list, dtype=torch.float32)
            label_tensor = torch.tensor(label_list, dtype=torch.float32)
            return TensorDataset(feature_tensor, label_tensor)

        num_of_data = len(feature_list)

        sequenced_feature_list = []
        sequenced_label_list = []

        for i in range(num_of_data - len_sequence + 1):
            feature_sequence = feature_list[i : i + len_sequence]
            corresponding_label = label_list[i + len_sequence - 1]

            sequenced_feature_list.append(feature_sequence)
            sequenced_label_list.append(corresponding_label)

        feature_tensor = torch.tensor(sequenced_feature_list, dtype=torch.float32)
        label_tensor = torch.tensor(sequenced_label_list, dtype=torch.float32)

        # print('size of feature_tensor :', feature_tensor.size())
        # print('size of label_tensor :', label_tensor.size())

        dataset = TensorDataset(feature_tensor, label_tensor) # Torch의 TensorDataset 함수를 활용해서 학습에 사용할 Dataset 형식으로 변환
        return dataset
    
    def make_nqe_dataset(self, len_sequence=8):
        feature_list = self.feature_df.values
        label_list = self.label_df.values

        num_of_data = len(feature_list)

        sequenced_feature_list = []
        sequenced_label_list = []

        for i in range(num_of_data - len_sequence + 1):
            feature_sequence = feature_list[i : i + len_sequence]
            corresponding_label = label_list[i + len_sequence - 1]

            sequenced_feature_list.append(feature_sequence)
            sequenced_label_list.append(corresponding_label)
        

        nqe_feature_list = []
        nqe_label_list = []
        sequenced_feature_size = len(sequenced_feature_list)

        for i in range(sequenced_feature_size):
            idx1, idx2 = i, np.random.randint(0, sequenced_feature_size)
            nqe_feature = [sequenced_feature_list[idx1], sequenced_feature_list[idx2]]
            nqe_label = [sequenced_label_list[idx1], sequenced_label_list[idx2]]
            nqe_feature_list.append(nqe_feature)
            nqe_label_list.append(nqe_label)
        
        nqe_feature_tensor = torch.tensor(nqe_feature_list, dtype=torch.float32)
        nqe_label_tensor = torch.tensor(nqe_label_list, dtype=torch.float32)

        nqe_dataset = TensorDataset(nqe_feature_tensor, nqe_label_tensor)

        return nqe_dataset
    
    def train_test_dataloader(self, for_nqe, train_ratio = 0.8, batch_size=64, len_sequence=8, chk=False):
        # Train/Test set 분할 (train_ratio = 0.8 -> 80% / 20% 비율)
        if for_nqe == True:
            dataset = self.make_nqe_dataset(len_sequence=len_sequence)
        else:
            dataset = self.make_dataset(len_sequence=len_sequence)
        
        num_of_data = len(dataset)
        train_size = int(train_ratio * num_of_data)
        test_size = num_of_data - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        if chk:
            feature_batch, label_batch = next(iter(train_loader))
            # print("one of feature batch :", feature_batch)
            print("shape of batch feature :", feature_batch.shape)  # Expected:
            # print("one of label batch :", label_batch)
            print("shape of batch label :", label_batch.shape)  # Expected:

        return train_loader, test_loader