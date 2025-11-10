import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import gymnasium as gym
import ale_py
import ale
import os
import cv2
from tqdm import tqdm

from envs import *
from network import DQN
from replaymemory import ReplayMemory
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(4, 4)
env = wrap_deepmind("BreakoutNoFrameskip-v4",render_mode='human')

def load_checkpoint(path):
    policy_net.load_state_dict(torch.load(path))

load_checkpoint('weights.pth')
policy_net.to(device)
policy_net.eval() 

episode_rewards = []

for episode in range(5):
    dead = False
    total_reward = 0
    step_count = 0
    
    obs = env.reset()
    if isinstance(obs, tuple):  # gymnasium returns (obs, info)
        obs= obs[0]
    
    for _ in range(np.random.randint(1, 30)):
        obs, _, _, info = env.step(1)
    
    while not dead:
        current_life = info['lives']
        
        with torch.no_grad():
            q_values = policy_net(torch.Tensor(obs).unsqueeze(0).to(device))
            action = torch.argmax(q_values, dim=1).item()
        
        # 执行动作
        next_obs, reward, dead,  info = env.step(action)
        
        
        total_reward += reward
        step_count += 1
        obs = next_obs
        
        env.render()
        
        if info['lives'] < current_life:
            for _ in range(np.random.randint(1, 30)):
                obs, _, _, info = env.step(1)
            
        if dead:
            break
    
    episode_rewards.append(total_reward)
    print(f"回合 {episode + 1}: 总奖励 = {total_reward}, 步数 = {step_count}")

env.close()