import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, clear_output


class trainer:
    def __init__(
        self, model, train_loader, test_loader, criterion=nn.MSELoss(), lr=0.001
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.train_loss_list = []
        self.test_loss_list = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def plot_list(self, train_loss=[], test_loss=[], title=""):
        # 텐서 리스트를 numpy 배열로 변환
        loss_values = [e.item() for e in train_loss]
        if test_loss:
            test_loss_values = [e.item() for e in test_loss]
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label="Train Loss")
        if test_loss:
            plt.plot(test_loss_values, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def train(self, epochs=10, chk=False):
        self.train_loss_list = []
        batch_size = self.train_loader.batch_size
        print('batch_size :', batch_size)
        for i in tqdm(range(epochs)):
            count = 0
            cost = 0
            for idx, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                cost += loss
                loss.backward()
                self.optimizer.step()
                count += 1
            self.train_loss_list.append(cost / count)

            count_test = 0
            cost_test = 0
            for idx, (x, y) in enumerate(self.test_loader):
                pred_test = self.model(x)
                loss_test = self.criterion(pred_test, y)
                cost_test += loss_test
                count_test += 1
            self.test_loss_list.append(cost_test / count_test)

        if chk:
            self.plot_list(self.train_loss_list, self.test_loss_list, title="Train Loss")

    def test(self, chk=False):
        self.test_loss_list = []
        for idx, (x, y) in enumerate(self.test_loader):
            pred = self.model(x)
            loss = self.criterion(pred, y)
            self.test_loss_list.append(loss.mean().detach())
        if chk:
            self.plot_list(self.test_loss_list, title="Test Loss")
        return self.test_loss_list
