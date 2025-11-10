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

# 超参数设置
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
NUM_STEPS = 50000000
M_SIZE = 100000
POLICY_UPDATE = 4
EVALUATE_FREQ = 200000
LEARNING_RATE = 2.5e-4
SAVE_PATH = 'checkpoint.pth'
SAVE_FREQ = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = wrap_deepmind("BreakoutNoFrameskip-v4")

n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
step_done = 0

# 网络设置
policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# 优化器
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# 经验回放缓存区
replay_memory = ReplayMemory(M_SIZE, BATCH_SIZE, device)

def save_checkpoint(path):
    checkpoint = {
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'memory': replay_memory.memory,
        'step_done': step_done
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path):
    checkpoint = torch.load(path)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    replay_memory.memory = checkpoint['memory']
    step_done = checkpoint['step_done']
    print(f"Checkpoint loaded from {path}, starting from step {step_done}")
    return step_done

def optimize_model(train):
    if not train:
        return

    state_batch, action_batch, reward_batch, n_state_batch, done_batch = replay_memory.sample()

    q_values = policy_net(state_batch).gather(1, action_batch)

    next_q_values = target_net(n_state_batch).max(1)[0].detach()

    expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))

    loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()


def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            q_values = policy_net(state)
            action = q_values.max(1)[1].item()
    else:
        action = random.randrange(n_actions)
    return action


done = True
episode_len = 0
reward_buffer = []
loss_buffer = []

if os.path.exists(SAVE_PATH):
    step_done = load_checkpoint(SAVE_PATH)
else:
    print("预热经验回放缓存区...")
    obs = env.reset()
    for _ in range(np.random.randint(1, 30)):
        obs, _, _, _, info = env.step(1)
    progress = tqdm(range(M_SIZE), total=M_SIZE, ncols=50, leave=False, unit='b')
    for _ in progress:
        if done:
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        replay_memory.push(state, action, reward, new_state, done)
        state = new_state

smooth_rewards = []
done = True
obs = env.reset()
for _ in range(np.random.randint(1, 30)):
        obs, _, _, _, info = env.step(1)

print("开始训练...")
progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
for step in progressive:
    if done:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        for _ in range(np.random.randint(1, 30)):
            obs, _, _, _, info = env.step(1)
        episode_len = 0
        done = False


    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step_done / EPS_DECAY)
    action = select_action(state, epsilon)
    new_state, reward, done, info = env.step(action)
    replay_memory.push(state, action, reward, new_state, done)
    episode_len += 1
    reward_buffer.append(reward)
    state = new_state

    optimize_model(True)
    
    step_done += 1
    if step_done % SAVE_FREQ == 0:
        save_checkpoint(SAVE_PATH)
    
    if step_done % EVALUATE_FREQ == 0:
        smooth_rewards.append(np.sum(reward_buffer))
        reward_buffer = []

    if step_done % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if step_done % 10000 == 0:
        plt.figure(figsize=(12,5))
        plt.plot(smooth_rewards)
        plt.title('Smooth Rewards over Time')
        plt.xlabel('Episodes')
        plt.ylabel('Smooth Rewards')
        plt.savefig('smooth_rewards.png')

env.close()
print("训练结束！")

