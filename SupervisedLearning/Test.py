import gym
import numpy as np
from threading import Thread

env = gym.make('CartPole-v0')

env.reset()

def gameThread (Thread):










while True:

    done = False

    while not done:

        env.render()

        action = int(input("Input"))
        obs, rew, done, info = env.step(action)
