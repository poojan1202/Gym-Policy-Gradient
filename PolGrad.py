import gym
import torch
import torch.nn as nn
import torch.nn.functional as functions
import random
import matplotlib.pyplot as plt
import numpy as np

class Actor_Network(nn.Module):
    """
    Gaussian Policy
    Outputs means and std deviation of a gaussian policy
    """
    def __init__(self, input_layer, output_layer):
        super(Actor_Network,self).__init__()
        self.linear1 = nn.Linear(input_layer,16)
        self.linear2 = nn.Linear(16,40)
        self.linear3 = nn.Linear(40,8)
        self.linear4 = nn.Linear(8,output_layer)

    def forward(self,x):
        x = functions.relu(self.linear1(x))
        x = functions.relu(self.linear2(x))
        x = functions.relu(self.linear3(x))
        x = functions.relu(self.linear4(x))
        return x

class Critic_Network(nn.Module):
    """
    Outputs Value Function
    """
    def __init__(self,input_layer):
        super(Critic_Network,self).__init__()
        self.linear1 = nn.Linear(input_layer,16)
        self.linear2 = nn.Linear(16,40)
        self.linear3 = nn.Linear(40,8)
        self.linear4 = nn.Linear(8,1)
        

    def forward(self,x):
        x = functions.relu(self.linear1(x))
        x = functions.relu(self.linear2(x))
        x = functions.relu(self.linear3(x))
        x = self.linear4(x)
        return x

#class Agent():

def trainer(AN, CN, nE):
	optimA = torch.optim.Adam(actor.parameters(), lr = 0.01)
	optimC = torch.optim.Adam(critic.parameters(), lr = 0.01)
	E = 0
	while(E>nE):
		S = env.reset()

