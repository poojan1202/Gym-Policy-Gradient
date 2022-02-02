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

#class Agent():


