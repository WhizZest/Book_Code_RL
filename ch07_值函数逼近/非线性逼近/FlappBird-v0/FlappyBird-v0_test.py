import time
import flappy_bird_gym
import os
import torch
import torch.nn as nn

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # 可训练参数
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 非参数化噪声
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # 初始化可训练参数
        bound = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))

    def reset_noise(self):
        # 采样噪声
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

class Dueling_NoisyDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Dueling_NoisyDQN, self).__init__()
        self.fc = nn.Sequential(
            NoisyLinear(input_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 128),
            nn.ReLU(),
        )
        self.V = NoisyLinear(128, 1)
        self.A = NoisyLinear(128, output_dim)

    def forward(self, x):
        x = self.fc(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
    
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

import os
import time
import pygame
import torch
import flappy_bird_gym
import matplotlib.pyplot as plt
import numpy as np
import collections
from collections import deque

# 初始化模型路径和环境
script_dir = os.path.dirname(os.path.abspath(__file__))
filePath = os.path.join(script_dir, 'models/fb_v0_no_score_2024-12-14_13-44-36.pth')

if os.path.exists(filePath):
    env = flappy_bird_gym.make("FlappyBird-v0")
    number_of_states = 12
    input_dim = env.observation_space.shape[0] * number_of_states
    output_dim = env.action_space.n
    q_net = Dueling_NoisyDQN(input_dim, output_dim)
    q_net.load_state_dict(torch.load(filePath, weights_only=True))
    q_net.eval()
    print("模型已加载")
    obs = env.reset()
    state_queue = deque([obs.copy() for _ in range(number_of_states)], maxlen=number_of_states)  # 初始化队列，初始状态填充队列
    state = np.concatenate(state_queue) # 将队列内容展平
    bExit = False
    current_score = 0
    min_steps_between_flaps = 999999
    max_steps_between_flaps = 0
    steps_between_flaps_list = []
    steps_between_flaps = 0
    previous_x = -float('inf')
    v_list = []
    points = []
    # 距离前一个管道的步数
    steps_to_previous_pipe = 10
    while bExit == False:
        points.append(obs)
        #obs = tuple(obs)
        # Next action:
        # (feed the observation to your agent here)
        action = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()

        # Processing:
        obs, reward, done, info = env.step(action)
        state_queue.append(obs)
        state = np.concatenate(state_queue)  # 将队列内容展平
        '''if previous_x > obs[0]:
            v = previous_x - obs[0]
            v_list.append(v)'''
        if steps_to_previous_pipe < 10:
            steps_to_previous_pipe += 1
        previous_x = obs[0]
        steps_between_flaps += 1
        if info['score'] > current_score:
            current_score = info['score']
            steps_to_previous_pipe = 0
            if steps_between_flaps < min_steps_between_flaps:
                min_steps_between_flaps = steps_between_flaps
            if steps_between_flaps > max_steps_between_flaps:
                max_steps_between_flaps = steps_between_flaps
            steps_between_flaps_list.append(steps_between_flaps)
            steps_between_flaps = 0
        # Rendering the game:
        # (remove this two lines during training)
        env.render()
        time.sleep(1 / 30)  # FPS

        # 处理 pygame 事件队列，防止窗口卡死
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                #exit()
                bExit = True
                done = True
        
        # Checking if the player is still alive
        if done:
            print(f"score: {info['score']}")
            print(f"steps_to_previous_pipe: {steps_to_previous_pipe}")
            print("Game Over")
            current_score = 0
            steps_between_flaps = 0
            steps_to_previous_pipe = 10
            obs = env.reset()

    env.close()
    print(f"min_steps_between_flaps: {min_steps_between_flaps}")
    print(f"max_steps_between_flaps: {max_steps_between_flaps}")
    # 打印steps_between_flaps_list的均值
    print(f"steps_between_flaps_list_mean: {np.mean(steps_between_flaps_list)}")
    # 绘制 steps_between_flaps_list 的图像
    plt.plot(steps_between_flaps_list)
    plt.xlabel('Step')
    plt.ylabel('v')
    plt.title('v over Time')
    plt.show()

    # 绘制 points 的散点图
    plt.scatter(*zip(*points))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points Scatter Plot')
    plt.show()
else:
    print("未找到模型文件")
