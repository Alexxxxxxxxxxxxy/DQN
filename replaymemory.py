import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity, batch_size, device):
        """
        初始化ReplayMemory类
        :param capacity: 内存的最大容量
        :param batch_size: 每次采样的批量大小
        :param device: 模型所在的设备，'cpu' 或 'cuda'
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=capacity)  # 使用deque来存储经验，最大容量为capacity

    def push(self, state, action, reward, next_state, done):
        """
        将一条经验（state, action, reward, next_state, done）推入ReplayMemory
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_state: 下一个状态
        :param done: 是否终止标志
        """
        # 将经验压入内存
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """
        从ReplayMemory中随机采样一个batch
        :return: 一批样本，包括状态、动作、奖励、下一状态和是否终止标志
        """
        # 如果内存中样本数小于批量大小，则不采样
        if len(self.memory) < self.batch_size:
            return None

        # 随机采样一个batch
        batch = random.sample(self.memory, self.batch_size)

        # 将batch中的数据分成几个部分
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将这些部分转换为torch张量，并移到指定设备（GPU 或 CPU）
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).reshape(self.batch_size, -1).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        返回ReplayMemory中当前存储的经验数量
        """
        return len(self.memory)