import os
import time
import flappy_bird_gym
import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
import pygame
import torch.nn.functional as F

def preprocess_image(frame, method='binarize'):
    # 预处理图像：背景黑化(颜色值为200的像素点设为黑色)、灰度化、裁剪、缩放、二值化
    frame[frame == 200] = 0
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_cropped = frame_gray[:, :420]  # 裁剪掉地面部分
    frame_resize = cv2.resize(frame_cropped, (84, 84))
    if method == 'binarize':
        processed_frame = cv2.threshold(frame_resize, 1, 255, cv2.THRESH_BINARY)[1]
    else:
        # 归一化到 [0, 1]
        processed_frame = frame_resize / 255.0
    return processed_frame

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

class Dueling_Noisy_DistributionalDQN(nn.Module):
    def __init__(self, input_shape, output_dim, num_atoms=51):
        super(Dueling_Noisy_DistributionalDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # (C, H, W) -> Conv,输出：32@20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 输出：64@9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 输出：64@7x7
            nn.ReLU(),
        )

        # 计算卷积后的特征图的尺寸
        conv_output_size = self._get_conv_output_size(input_shape)

        # Noisy 全连接层
        self.fc = nn.Sequential(
            NoisyLinear(conv_output_size, 512),
            nn.ReLU(),
        )

        # Dueling 分支
        self.V = NoisyLinear(512, num_atoms)            # 状态价值分支
        self.A = NoisyLinear(512, output_dim * num_atoms)   # 优势值分支
        self.num_atoms = num_atoms
        self.output_dim = output_dim

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)  # 临时张量用于计算
        x = self.conv(x)
        return int(torch.flatten(x, 1).size(1))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # 展平为向量
        x = self.fc(x)

        V = self.V(x).view(-1, 1, self.num_atoms)
        A = self.A(x).view(-1, self.output_dim, self.num_atoms)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        Q_prob = F.softmax(Q, dim=2) # 将 Q 值转换为概率分布
        return Q_prob

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# 创建 FlappyBird-rgb-v0 环境
env = flappy_bird_gym.make("FlappyBird-rgb-v0")

# 重置环境
obs = env.reset()
number_of_states = 4
skip_frames = 2
atoms = 51
v_min = -0.02
v_max = 2.8
supports = torch.linspace(v_min, v_max, atoms)
preprocessHeight = 84
preprocessWidth = 84
input_shape = (number_of_states, preprocessHeight, preprocessWidth)
output_dim = env.action_space.n
processed_frame = preprocess_image(obs)
state_queue = deque([processed_frame.copy() for _ in range(number_of_states)], maxlen=number_of_states)  # 初始化队列，初始状态填充队列
state = np.array(state_queue)
# 初始化模型路径和环境
script_dir = os.path.dirname(os.path.abspath(__file__))
filePath = os.path.join(script_dir, 'flappy_bird_v0_model_best_score.pth')
q_net = Dueling_Noisy_DistributionalDQN(input_shape, output_dim, atoms)
q_net.load_state_dict(torch.load(filePath, weights_only=True))
q_net.eval()

bExit = False
step_count = 0
while bExit == False:
    # 这里可以将观察值传递给你的智能体来决定动作
    #action = env.action_space.sample()  # 随机动作
    if step_count % skip_frames == 0:
        dist = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        action = (dist * supports).sum(dim=2).argmax().item()  # 使用 Q 网络选择动作
    else:
        action = 0

    # 执行动作
    obs, reward, done, info = env.step(action)
    step_count += 1
    if step_count % skip_frames == 0:
        processed_frame = preprocess_image(obs)
        state_queue.append(processed_frame)
        state = np.array(state_queue)

    # 渲染游戏画面
    env.render()
    time.sleep(1 / 30)  # 控制帧率

    # 处理 pygame 事件队列，防止窗口卡死
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            #exit()
            bExit = True
            #done = True

    # 检查游戏是否结束
    if done:
        print(f"score: {info['score']}")
        print("Game Over")
        obs = env.reset()
        #break

env.close()
