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

"""
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
"""

def Q_trainer(nE, lr = 0.01, gamma = 0.9):
  optimA = torch.optim.Adam(actor.parameters(), lr)
  optimC = torch.optim.Adam(critic.parameters(), lr)
  E = 0
  lengths = []
  rewards = []
  pl = []
  cl = []
  while (E < nE):
    S = env.reset()
    #env.render()
    Erev = 0
    done = False
    p_loss = 0
    length = 0
    while not done:
      S = torch.FloatTensor(S)
      pol_s1 = actor(S)
      dist1 = torch.distributions.Normal(pol_s1[0], abs(pol_s1[1]))
      action1 = dist1.sample()

      nS, R, done, _ = env.step([action1.item()])

      tnS = torch.FloatTensor(nS)
      pol_s2 = actor(tnS)
      dist2 = torch.distributions.Normal(pol_s2[0], abs(pol_s2[1]))
      action2 = dist2.sample()
      
      action1 = torch.flatten(torch.reshape(action1, (1,1)))
      with torch.no_grad():
        val1 = critic(torch.cat((S, action1)))
      ln_p = dist1.log_prob(action1)
      
      optimA.zero_grad()
      p_loss = (ln_p*val1)
      p_loss.backward()
      optimA.step()

      action2 = torch.flatten(torch.reshape(action2, (1,1)))
      with torch.no_grad():
        val2 = critic(torch.cat((tnS, action2)))
      val1 = critic(torch.cat((S, action1)))
      targe = R + (gamma*val2) - (val1)
      
      optimC.zero_grad()
      c_loss = targe*val1
      c_loss.backward(retain_graph = True)
      optimC.step() 

      S = nS

      Erev += R
      length += 1

      cl.append(c_loss[0].item())
      pl.append(p_loss[0].item())
    E = E+1
    rewards.append(Erev)
    lengths.append(length)
    print(E)
  show = 1
  if show:
    print("Reward vs Epsiode")
    plot = plt.plot([i for i in range(len(rewards))],rewards)
    plt.show()
    print("Critic Loss vs Step")
    plot = plt.plot([i for i in range(len(cl))],cl)
    plt.show()
    print("Actor Loss vs Step")
    plot = plt.plot([i for i in range(len(pl))],pl)
    plt.show()
    #print("Length vs Episode")
    #plot = plt.plot([i for i in range(len(lengths))],lengths)
    #plt.show()

def test(nT):
  scores = []
  for _ in range(nT):
    S = env.reset()
    score = 0
    done = False
    while not done:
      env.render()
      S = torch.FloatTensor(S)
      dist = actor(S)
      dist = torch.distributions.Normal(dist[0],abs(dist[1]))
      action = dist.sample()
      #print(action)
      nS, R, done, _ = env.step([action.item()])
      score = R + score
      S = nS
    scores.append(score)
    print("Score\t - ", score)
  print("Avg Score\t - ", sum(scores)/len(scores))

if __name__ == "__main__":
  env = gym.make('Pendulum-v1')
  env.reset()
  Loss = nn.MSELoss()
  actor = AN(3,2)
  critic = CN(4)
  Q_trainer(1500)
  test(10)