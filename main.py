import gym
import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import AdamW
from IPython import display
from matplotlib import pyplot as plt
from tqdm import tqdm

def main():
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    env = PreprocessEnv(env)
    state = env.reset()
    action = torch.tensor(0)
    next_state, reward, done, _ = env.step(action)
    print(f"Sample state: {state}")
    print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    120
    def reset(self):
        state = self.env.reset()[0]
        
        return torch.from_numpy(state).unsqueeze(dim=0).float()
    
    def step(self, action):
        action = action.item()
        next_state, reward, done, _, info= self.env.step(action)

        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)

        return next_state, reward, done, info


def test_env(env: gym.Env) -> None:
    env.reset()
    done = False
    img = plt.imshow(env.render())
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        img.set_data(env.render())
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)

main()