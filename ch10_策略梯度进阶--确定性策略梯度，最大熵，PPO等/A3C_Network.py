import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, action_dim)
        self.sigma = nn.Linear(128, action_dim)

    def forward(self, x):
        x0 = self.fc1(x)
        x1 = F.relu(x0)
        x2 = F.relu(self.fc2(x1))
        mu = torch.tanh(self.mu(x2)) * 2  # Pendulum动作范围是[-2, 2]
        sigma = F.softplus(self.sigma(x1)).clamp(min=0.00001)  # 保证标准差为正
        return mu, sigma

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        #x2 = F.relu(self.fc2(x1))
        value = self.value(x1)
        return value