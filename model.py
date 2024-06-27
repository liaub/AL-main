import torch
import torch.nn as nn
import torch.nn.functional as F

def get_net(name,Data_Vector_Length):

    return Net(Data_Vector_Length)

class Net(nn.Module):
    def __init__(self,Data_Vector_Length=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(Data_Vector_Length, 250)
        self.fc2 = nn.Linear(250, 250)
        # self.fc3 = nn.Linear(250, 250)
        self.fc4 = nn.Linear(250, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

    def get_embedding_dim(self):
        return 50