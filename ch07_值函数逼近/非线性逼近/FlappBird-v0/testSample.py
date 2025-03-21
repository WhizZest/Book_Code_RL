import time
import flappy_bird_gym
import pygame
import matplotlib.pyplot as plt
import random
env = flappy_bird_gym.make("FlappyBird-v0")

# 初始化action列表
actions = []

obs = env.reset()
points = []
# 采样的概率
p = 0.07
skip_frames = 1
min_y = float('inf')
max_y = float('-inf')
for t in range(30000):
    points.append(obs)
    # Next action:
    # (feed the observation to your agent here)
    #action = env.action_space.sample()  # for a random action
    if t % skip_frames == 0:
        # 根据概率决定采样
        if random.random() < p * skip_frames:
            action = 1
        else:
            action = 0

    # Processing:
    obs, reward, done, info = env.step(action)
    if obs[1] > 0.5:
        done = True
    y = abs(obs[1])
    if y < min_y:
        min_y = y
    if y > max_y:
        max_y = y
    
    # 将action添加到列表中
    actions.append(action)
    
    # Checking if the player is still alive
    if done:
        obs = env.reset()

    # 处理 pygame 事件队列，防止窗口卡死
    '''for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            done = True'''

# 绘制action的采样分布
plt.hist(actions, bins=env.action_space.n, edgecolor='black')
plt.title('Action Distribution')
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.show()

# 绘制 points 的散点图
plt.scatter(*zip(*points))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points Scatter Plot')
plt.show()
env.close()

print("min_y:", min_y)
print("max_y:", max_y)