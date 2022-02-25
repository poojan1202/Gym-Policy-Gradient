import gym
import random
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG

def test(nT):
	scores = []
	for _ in range(nT):
    	S = env.reset()
    	score = 0
    	done = False
    	while not done:
			env.render()
			action, _states = model.predict(S)
			S, R, done, _ = env.step([action.item()])
			score = R + score
		scores.append(score)
		print("Score\t - ", score)
	print("Avg Score\t - ", sum(scores)/len(scores))


if __name__ == "__main__":
	env = gym.make("Pendulum-v1")

	nA = (env.action_space.shape[-1])
	print(nA)

	model = DDPG("MlpPolicy", 
					env, 
					learning_rate = 0.01, 
					buffer_size = 1000, 
					learning_start = 10, 
					batch_size = 128,
					gamma = 0.9,
					verbose = 1)

	model.learn(total_timesteps = 10000, log_interval = 10)
	test(10)