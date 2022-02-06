import gym
import torch
import torch.nn as nn
import torch.nn.functional as functions
import random
import matplotlib.pyplot as plt
import numpy as np

class AN(nn.Module):
  def __init__(self, nAIP, nAOP):
    super(AN,self).__init__()
    self.linear1 = (nn.Linear(nAIP, 8))
    self.linear2 = (nn.Linear(8, 16))
    self.linear3 = (nn.Linear(16, 16))
    self.linear4 = (nn.Linear(16, nAOP))
    #self.relu = functions.relu()
    self.relu = nn.ReLU()
  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.relu(x)
    x = self.linear3(x)
    x = self.relu(x)
    x = self.linear4(x)
    return x

class CN(nn.Module):
  def __init__(self, nCIP):
    super(CN,self).__init__()
    self.linear1 = (nn.Linear(nCIP, 8))
    self.linear2 = (nn.Linear(8, 16))
    self.linear3 = (nn.Linear(16, 16))
    self.linear4 = (nn.Linear(16, 1))
    #self.relu = functions.relu()
    self.relu = nn.ReLU()
  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.relu(x)
    x = self.linear3(x)
    x = self.relu(x)
    x = self.linear4(x)
    return x

def discounted_rewards(h_rev, gamma):
  val = 0
  nRev = []
  for r in h_rev[::-1]:
    val = r + val*gamma
    nRev.append(val)
  return (nRev[::-1])

def trainer(nE, lr = 0.01, gamma = 0.9):
  optimA = torch.optim.Adam(actor.parameters(), lr)
  optimC = torch.optim.Adam(critic.parameters(), lr)
  E = 0
  lengths = []
  while (E < nE):
    S = env.reset()
    env.render()
    Erev = 0
    h_S = []
    h_ln_p = []
    h_act = []
    h_rev = []
    h_ANS = []
    h_NS = []
    length = 0
    done = False
    #Running and Recording an episode
    while not done:
      S = torch.FloatTensor(S)
      h_S.append(S)
      dist = actor(S)
      h_ANS.append(dist)
      #For these two environments
      dist = torch.distributions.Normal(dist[0], abs(dist[1]))
      action = dist.sample()
      h_act.append(action)
      #print(action.item(), type(action.item()))
      nS, R, done, _ = env.step([action.item()])

      h_NS.append(torch.FloatTensor(nS))
      h_rev.append(R)
      h_ln_p.append(dist.log_prob(action))

      Erev += R
      length += 1
    
    h_rev = [torch.FloatTensor([r]) for r in discounted_rewards(h_rev,gamma)]
    crit = [critic(h_S[i]) for i in range(len(h_S))]
    advantage = [(h_rev[i] - crit[i]) for i in range(len(h_rev))]
    p_loss = [(-1*h_ln_p[i]*advantage[i]) for i in range(len(h_rev))]
    c_loss = [Loss(crit[i], h_rev[i]) for i in range(len(h_rev))]
    optimA.zero_grad()
    optimC.zero_grad()
    #for i in range(len(p_loss)):
    #  p_loss[i].backward()
    #  optimA.step()
    #  print(c_loss[i])
    #  c_loss[i].backward()
    #  optimC.step()
    p_sum = sum(p_loss)
    c_sum = sum(c_loss)
    p_sum.backward()
    optimA.step()
    c_sum.backward()
    optimC.step()
    E = E + 1
    lengths.append(length)

def test(nT):
  scores = []
  for _ in range(nT):
    S = env.reset()
    score = 0
    done = False
    while not done:
      S = torch.FloatTorch(S)
      dist = actor(S)
      dist = torch.distributions.Normal(dist[0],abs(dist[1]))
      action = dist.sample()
      nS, R, done, _ = env.step(action.item())
      score = R + score
      S = nS
    scores.append(score)
    print("Score - ", score)
  print("Avg Score - ", scores.sum()/len(scores))

if __name__ == "__main__":
	env = gym.make('Pendulum-v1')
	env.reset()
	Loss = nn.MSELoss()
	actor = AN(3,2)
	critic = CN(3)
	trainer(500)
	test(10)