import gym
import torch
import torch.nn as nn
import torch.nn.functional as functions
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.normal import Normal

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
       #x = self.linear4(x)
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



actor = Actor_Network(2,2)

print(actor(torch.FloatTensor([5,6])), actor(torch.FloatTensor([5,6])).detach().numpy().squeeze())


env = gym.make('Pendulum-v0')
actor_model = Actor_Network(3,2)
critic_model = Critic_Network(3)

state = env.reset()
print('state',state)
done=False
while not done:
    with torch.no_grad():
        dist_param = actor_model(torch.from_numpy(state).float())
        dist = Normal(dist_param[0],dist_param[1])
        #action = dist.sample().item()
        action = dist.sample().numpy().reshape(1)
        #print(action)
    new_state, reward, done, info = env.step(action)
    state = new_state



