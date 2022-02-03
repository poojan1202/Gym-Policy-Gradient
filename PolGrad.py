import random

import torch
import torch.nn as nn
import torch.nn.functional as functions

class Actor_Network(nn.Module):
    """
    Gaussian Policy
    Outputs means and std deviation of a gaussian policy
    """
    def __init__(self,input_layer):
        super(Actor_Network,self).__init__()
        self.linear1 = nn.Linear(input_layer,16)
        self.linear2 = nn.Linear(16,8)
        self.linear3 = nn.Linear(8,2)

    def forward(self,x):
        x = functions.relu(self.linear1(x))
        x = functions.relu(self.linear2(x))
        x = functions.relu(self.linear3(x))
        return x

class Critic_Network(nn.Module):
    """
    Outputs Value Function
    """
    def __init__(self,input_layer):
        super(Critic_Network,self).__init__()
        self.linear1 = nn.Linear(input_layer,16)
        self.linear2 = nn.Linear(16,8)
        self.linear3 = nn.Linear(8,1)

    def forward(self,x):
        x = functions.relu(self.linear1(x))
        x = functions.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ReplayMemory():
    def __init__(self,size):
        self.size = size
        self.memory = [[],[],[],[],[]]
    def store(self,data):
        if len(self.memory[0])==self.size:
            for idx in range(5):
                self.memory[idx].pop(0)
        for idx,part in enumerate(data):
            self.memory[idx].append(part)
    def sample(self,batch_size):
        rows = random.sample(range(0,len(self.memory[0])),batch_size)
        experiences = [[],[],[],[],[]]
        for row in rows:
            for col in range(5):
                experiences[col].append(self.memory[col][row])
        return experiences

    def __len__(self):
        return len(self.memory[0])


#class Agent():


