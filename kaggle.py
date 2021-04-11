import sys
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from mnist import MNIST
import random
from math import sqrt
from torch.utils.data import Subset
import numpy as np
import pandas as pd
print("Python version: {}".format(sys.version))
print("Pytorch version: {}".format(torch.__version__))

n_epochs = 5
batch_size_train = 500
batch_size_test = 100
learning_rate = 0.001
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20, 1)  # 6*6 from image dimension
        self.fc2 = nn.Linear(1, 4096)
        self.fc3 = nn.Linear(4096, 1)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # x = x.view(-1, self.num_flat_features(x))
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def train(epoch,train_loader):
    net.train()
    #for batch_idx, (data, target) in enumerate(train_loader):
    for index, row in train_loader.iterrows():
        print(row)
        target = torch.tensor(row.pop("label")).float()
        print(row)
        data = torch.tensor(row.values).float()
        
        #optimizer.zero_grad()
        output = net(data)
        loss = loss_fn(output, target)
        net.zero_grad()
        loss.backward()
        #optimizer.step()
        with torch.no_grad():
            for para in net.parameters():
                para -= learning_rate*para.grad

if __name__ == "__main__":
    net = Net()
    loss_fn = nn.CrossEntropyLoss()
    train_loader = pd.read_csv("train.csv")
    train_loader = train_loader.drop(["id","name", "slug","path","description","published","modified","ratings-given","link-tags","prev-games","num-authors","links"], axis=1)
    train_loader.loc[(train_loader.category == 'jam'),'category']=1
    train_loader.loc[(train_loader.category == 'compo'),'category']=0
    for epoch in range(n_epochs):
        train(epoch,train_loader)
    print("finish")