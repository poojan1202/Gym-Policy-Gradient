import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.normal import Normal
from torch.autograd import Variable

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
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
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
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

#class Agent():



actor = Actor_Network(2,2)

#print(actor(torch.FloatTensor([5,6])), actor(torch.FloatTensor([5,6])).detach().numpy().squeeze())


env = gym.make('Pendulum-v0')
actor_model = Actor_Network(3,2)
critic_model = Critic_Network(3)

gamma = 0.9
learning_rate = 0.005
critic_optimizer = torch.optim.Adam(critic_model.parameters(),lr=learning_rate)
actor_optimizer = torch.optim.Adam(actor_model.parameters(),lr=learning_rate)

for ep in range(500):
    state = env.reset()
    print('state', state)
    done = False
    total_rew = 0
    while not done:
        #if ep==0:
        #    env.render()
        #if (ep+1)%50==0:
        #    env.render()
        env.render()
        with torch.no_grad():
            dist_param = actor_model(torch.from_numpy(state).float())
            if dist_param[1] == 0:
                dist_param[1] = 0.0001
            dist = Normal(dist_param[0], dist_param[1])
            # action = dist.sample().item()
            action = dist.sample().numpy().reshape(1)
            # print(state)
            # print(dist_param)
        new_state, rew, done, info = env.step(action)

        total_rew += rew

        state_val = critic_model(torch.from_numpy(state).float())
        target_state_val = rew + gamma*critic_model(torch.from_numpy(new_state).float())

        state_val = torch.FloatTensor(state_val)
        with torch.no_grad():
            target_state_val = torch.FloatTensor(target_state_val)

        critic_loss = F.mse_loss(state_val,target_state_val)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        with torch.no_grad():
            Vs = critic_model(torch.FloatTensor(state))
            Vns = critic_model(torch.FloatTensor(new_state))
        adv_val = torch.from_numpy(np.array(rew)).float() + gamma*Vns - Vs

        actor_loss = -dist.log_prob(torch.from_numpy(action).float())*adv_val
        actor_loss = Variable(actor_loss,requires_grad=True)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()




        state = new_state
    print('total_rew',total_rew)







