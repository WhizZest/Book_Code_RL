import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.multiprocessing as mp
from collections import deque
import matplotlib.pyplot as plt
import keyboard
import os
import time
import random
from A3C_Network import Actor, Critic

# 超参数
GAMMA = 0.95
LR_ACTOR = 0.0001
LR_CRITIC = 0.008
LR_ALPHA = 0.0002
NUM_EPISODES = 50000
MAX_STEPS = 200
UPDATE_EVERY = mp.Value('i', 10)  # 每隔多少步更新一次网络
NUM_WORKERS = 8  # 并行工作进程数
ACTION_LOW = -2.0  # Pendulum动作下限
ACTION_HIGH = 2.0  # Pendulum动作上限
TAU = 0.01           # 目标网络软更新系数
EPSILON = 0.2   # 新旧策略概率比的裁剪系数
bExit = mp.Value('b', False)  # 使用共享变量
bTest = mp.Value('b', False)  # 使用共享变量
currentDir = os.path.dirname(os.path.abspath(__file__))

# 定义SA3C+PPO Agent：动态最大熵+A3C+TD(n)+PPO
class SA3C_PPOAgent:
    def __init__(self, state_dim, action_dim, global_actor, global_critic, global_critic_target, actor_optimizer, critic_optimizer, global_episode, device, log_alpha, alpha_optimizer):
        self.local_actor = Actor(state_dim, action_dim).to(device)
        self.local_critic = Critic(state_dim).to(device)
        self.local_critic_target = Critic(state_dim).to(device)
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.global_critic_target = global_critic_target
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.global_episode = global_episode
        self.device = device
        self.log_alpha = log_alpha
        self.local_log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device, dtype=torch.float32)
        self.alpha_optimizer = alpha_optimizer
        self.alpha = torch.exp(self.local_log_alpha)
        self.target_entropy = -action_dim

        # 从全局网络加载初始参数
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())
        self.local_critic_target.load_state_dict(self.global_critic_target.state_dict())
        self.local_log_alpha.data.copy_(self.log_alpha.data)

    def saveModel(self, actor_path, critic_path):
        torch.save(self.local_actor.state_dict(), actor_path)
        torch.save(self.local_critic.state_dict(), critic_path)
        # 打印学习率
        print(f"Save model, Actor learning rate: {self.actor_optimizer.param_groups[0]['lr']}")
    
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self.local_actor(state)
        dist = torch.distributions.Normal(mean, std)
        # 重采样，直到动作在范围内
        while True:
            action = dist.sample()
            if -2.0 <= action <= 2.0:
                break
        # 修正动作概率
        cdf_min = dist.cdf(torch.tensor(ACTION_LOW, dtype=torch.float32))
        cdf_max = dist.cdf(torch.tensor(ACTION_HIGH, dtype=torch.float32))
        normalization_constant = (cdf_max - cdf_min).clamp(min=1e-6)  # 避免数值问题
        log_prob = (dist.log_prob(action) - torch.log(normalization_constant)).sum()
        # 计算熵
        '''normal_entropy = dist.entropy().sum()
        truncated_entropy = normal_entropy / normalization_constant + torch.log(normalization_constant).sum()'''
        return action.numpy(), log_prob
    
    def select_greedy_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self.local_actor(state)
        action = mean
        dist = torch.distributions.Normal(mean, std)
        # 修正动作概率
        cdf_min = dist.cdf(torch.tensor(ACTION_LOW, dtype=torch.float32))
        cdf_max = dist.cdf(torch.tensor(ACTION_HIGH, dtype=torch.float32))
        normalization_constant = (cdf_max - cdf_min).clamp(min=1e-6)  # 避免数值问题
        log_prob = (dist.log_prob(action) - torch.log(normalization_constant)).sum()
        # 计算熵
        normal_entropy = dist.entropy().sum()
        truncated_entropy = normal_entropy / normalization_constant + torch.log(normalization_constant).sum()
        return action.detach().numpy(), log_prob, truncated_entropy
    
    def calc_td_error(self, states, final_state, log_probs, rewards, dones):
        values = self.local_critic(states)
        final_value = self.local_critic_target(final_state).detach()
        # 计算累积回报
        R = torch.tensor(0, dtype=torch.float32).to(self.device) if dones[-1] else final_value
        returns = []
        #mean_log_probs = []
        for i in reversed(range(len(rewards))):
            r = rewards[i] - self.alpha * log_probs[i]
            R = r + GAMMA * R
            returns.insert(0, R)
            #mean_log_probs.insert(0, log_probs[i: -1].mean())
        #mean_log_probs = torch.stack(mean_log_probs).to(self.device)  # 形状: [batch_size, 1]
        td_errors = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1) - values  # 形状: [batch_size, 1]
        return td_errors
        
    def train(self, states, actions, log_probs, rewards, dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        log_probs = torch.stack(log_probs).unsqueeze(-1).to(self.device)  # 直接堆叠张量
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(self.device)  # 形状: [batch_size, 1]
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(-1).to(self.device)  # 形状: [batch_size, 1]

        td_errors = self.calc_td_error(states[:len(actions)], states[-1], log_probs, rewards, dones)  # 形状: [batch_size, 1]
        # 计算Critic的损失
        critic_loss = td_errors.pow(2).mean()
        # 更新Critic网络
        self.local_critic.zero_grad()  # 清空局部网络的梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        for local_param, global_param in zip(self.local_critic.parameters(), self.global_critic.parameters()):
            global_param._grad = local_param.grad
        self.critic_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.global_critic.parameters(), self.global_critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        #td_errors = self.calc_td_error(states[:len(actions)], states[-1], log_probs, rewards, dones)  # 形状: [batch_size, 1]
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        means, stds = self.local_actor(states[:len(actions)])
        dists = torch.distributions.Normal(means, stds)
        cdf_min = dists.cdf(torch.tensor(ACTION_LOW, dtype=torch.float32))
        cdf_max = dists.cdf(torch.tensor(ACTION_HIGH, dtype=torch.float32))
        normalization_constant = (cdf_max - cdf_min).clamp(min=1e-6)  # 避免数值问题
        new_log_probs = (dists.log_prob(actions) - torch.log(normalization_constant))
        ratio = torch.exp(new_log_probs - log_probs.detach())
        surr1 = ratio * td_errors.detach()
        surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * td_errors.detach()
        # 计算Actor的损失（带熵正则化）
        actor_loss = -torch.min(surr1, surr2).mean()
        #actor_loss = -(log_probs * td_errors.detach()).mean()
        # 更新Actor网络
        self.local_actor.zero_grad()  # 清空局部网络的梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for local_param, global_param in zip(self.local_actor.parameters(), self.global_actor.parameters()):
            global_param._grad = local_param.grad
        self.actor_optimizer.step()

        # 更新熵正则化系数
        alpha_loss = -(self.local_log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.local_log_alpha.grad = None # 清空本地log_alpha的梯度
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        # 将本地梯度传递给全局 log_alpha
        if self.local_log_alpha.grad is not None:
            self.log_alpha.grad = self.local_log_alpha.grad.clone()
        else:
            print("local_log_alpha.grad is None")
        self.alpha_optimizer.step()
        self.local_log_alpha.data.copy_(self.log_alpha.data)
        self.alpha = self.local_log_alpha.exp()
        #print(f"alpha: {self.alpha.item()}, at episode: {self.global_episode.value}")

        # 更新本地网络
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())
        self.local_critic_target.load_state_dict(self.global_critic_target.state_dict())

# 训练函数
def worker(global_actor, global_critic, global_critic_target, actor_optimizer, critic_optimizer, global_episode, reward_list, device, bTest, bExit, UPDATE_EVERY, log_alpha, alpha_optimizer):
    env = gym.make('Pendulum-v1')
    test_env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SA3C_PPOAgent(state_dim, action_dim, global_actor, global_critic, global_critic_target, actor_optimizer, critic_optimizer, global_episode, device, log_alpha, alpha_optimizer)

    while global_episode.value < NUM_EPISODES and bExit.value == False:
        state = env.reset()
        state = state[0]
        episode_reward = 0
        states, actions, log_probs, rewards, dones = [], [], [], [], []

        for step in range(MAX_STEPS):
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)  # 归一化奖励
            dones.append(0)

            state = next_state
            episode_reward += reward

            if (step + 1) % UPDATE_EVERY.value == 0 or terminated or truncated:
                states.append(next_state)
                agent.train(states, actions, log_probs, rewards, dones)
                states, actions, log_probs, rewards, dones = [], [], [], [], []
                if bTest.value:
                    success = True
                    for i in range(30):
                        test_state = test_env.reset()
                        test_state = test_state[0]
                        test_episode_reward = 0
                        for step in range(MAX_STEPS):
                            test_action, test_log_prob, test_entropy = agent.select_greedy_action(test_state)
                            test_next_state, test_reward, test_terminated, test_truncated, _ = test_env.step(test_action)
                            
                            test_state = test_next_state
                            test_episode_reward += test_reward
                            if test_terminated or test_truncated:
                                break
                        if test_episode_reward < -600 or test_reward < -0.0025:
                            success = False
                            break
                    if success:
                        # 保存模型：actor和critic
                        actor_save_path = os.path.join(currentDir, test_env.spec.id + "_SA3C_PPO-actor.pth")
                        critic_save_path = os.path.join(currentDir, test_env.spec.id + "_SA3C_PPO-critic.pth")
                        agent.saveModel(actor_save_path, critic_save_path)
                        print(f"Model saved at episode: {global_episode.value}, UPDATE_EVERY: {UPDATE_EVERY.value}, alpha: {log_alpha.exp().item()}")
                        # 按下Esc键退出
                        keyboard.press('esc')
                        bExit.value = True
                        print("已达到训练目标，提前退出训练")

            if terminated or truncated:
                with global_episode.get_lock():
                    global_episode.value += 1
                # 将回合总奖励和最后一步的奖励放入列表
                reward_list.append((global_episode.value, episode_reward, reward))
                break
            # 按下Esc键退出
            if keyboard.is_pressed('esc'):
                print("Worker exits")
                bExit.value = True
            if bExit.value:
                break

# 动态绘图函数
def monitor(reward_list, actor_optimizer, critic_optimizer, save_path):
    global UPDATE_EVERY
    global bTest
    global bExit
    episode_rewards = []  # 使用 list 代替 deque
    final_step_rewards = []  # 使用 list 代替 deque
    success_count = 0
    success_count2 = 0
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    line1, = ax1.plot([], [], label="Total Reward")
    line2, = ax2.plot([], [], label="Final Step Reward", color="orange")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Final Step Reward")
    ax2.legend()
    ax2.grid(True)

    while bExit.value == False:
        updated = False
        # 一次性取出所有数据
        if len(reward_list) > 0:
            all_rewards = list(reward_list)  # 复制所有数据
            reward_list[:] = []  # 清空列表

            # 使用 zip 一次性提取 total_reward 和 final_reward
            episodes, total_rewards, final_rewards = zip(*all_rewards)
            episode_rewards.extend(total_rewards)
            final_step_rewards.extend(final_rewards)

            updated = True

            # 处理成功条件
            for episode, total_reward, final_reward in all_rewards:
                if total_reward > -600:
                    success_count += 1
                else:
                    success_count = 0
                if final_reward > -0.02:
                    success_count2 += 1
                else:
                    success_count2 = 0
                    UPDATE_EVERY.value = 10
                    actor_optimizer.param_groups[0]['lr'] = LR_ACTOR
                if success_count >= 30 and success_count2 >= 30 and success_count2 % 10 == 0:
                    UPDATE_EVERY.value = min(UPDATE_EVERY.value + 10, MAX_STEPS)  # 每成功10次，更新间隔加10
                    # 降低学习率
                    '''current_actor_lr = actor_optimizer.param_groups[0]['lr']
                    current_critic_lr = critic_optimizer.param_groups[0]['lr']
                    actor_optimizer.param_groups[0]['lr'] = max(current_actor_lr * 0.8, LR_ACTOR * 0.001)
                    print(f"Actor learning rate: {actor_optimizer.param_groups[0]['lr']:.8f}, Critic learning rate: {critic_optimizer.param_groups[0]['lr']:.8f}, Update every: {UPDATE_EVERY}, at episode {episode}")'''
                if success_count >= 10 and success_count2 >= 30:
                    if bTest.value == False:
                        print("Test mode enabled at episode: ", episode)
                    bTest.value = True
                else:
                    if bTest.value == True:
                        print("Test mode disabled at episode: ", episode)
                    bTest.value = False

        if updated:
            # 更新总奖励图像
            line1.set_data(range(len(episode_rewards)), episode_rewards)
            ax1.relim()
            ax1.autoscale_view()

            # 更新最后一步奖励图像
            line2.set_data(range(len(final_step_rewards)), final_step_rewards)
            ax2.relim()
            ax2.autoscale_view()

            plt.pause(0.1)

        # 训练结束后保存图像
        if len(episode_rewards) >= NUM_EPISODES or bExit.value == True:
            print("Plotting process exits")
            break

    if save_path:
        plt.savefig(save_path)  # 保存图像
    plt.ioff()  # 关闭交互模式
    plt.show()

# 主函数
if __name__ == "__main__":
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    global_actor = Actor(state_dim, action_dim).to(device)
    global_critic = Critic(state_dim).to(device)
    global_critic_target = Critic(state_dim).to(device)
    global_critic_target.load_state_dict(global_critic.state_dict())
    global_actor.share_memory()
    global_critic.share_memory()
    global_critic_target.share_memory()
    log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device, dtype=torch.float32)
    log_alpha.share_memory_()

    # 分别为Actor和Critic创建优化器
    actor_optimizer = optim.Adam(global_actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(global_critic.parameters(), lr=LR_CRITIC)
    alpha_optimizer = optim.Adam([log_alpha], lr=LR_ALPHA)

    global_episode = mp.Value('i', 0)
    manager = mp.Manager()
    reward_list = manager.list()

    # 启动绘图进程
    '''plot_process = mp.Process(target=plot_rewards, args=(reward_list, "training_plot.png"))
    plot_process.start()'''

    workers = [
        mp.Process(
            target=worker,
            args=(global_actor, global_critic, global_critic_target, actor_optimizer, critic_optimizer, global_episode, reward_list, device, bTest, bExit, UPDATE_EVERY, log_alpha, alpha_optimizer)
        )
        for _ in range(NUM_WORKERS)
    ]

    for w in workers:
        w.start()
    
    # 获取当前py文件夹(不是项目文件夹)，将图片和模型保存在当前文件夹下
    save_path = os.path.join(currentDir, "SA3C_PPO_training_plot.png")
    '''actor_save_path = os.path.join(currentDir, env.spec.id + "_SA3C_PPO-actor1.pth")
    critic_save_path = os.path.join(currentDir, env.spec.id + "_SA3C_PPO-critic1.pth")'''
    # 在主线程中执行绘图函数
    monitor(reward_list, actor_optimizer, critic_optimizer, save_path)
    
    #for w in workers:
        #w.join()

    #plot_process.terminate()
    
    # 保存模型：游戏名+模型类型
    '''torch.save(global_actor.state_dict(), actor_save_path)
    torch.save(global_critic.state_dict(), critic_save_path)
    print("Model saved to: ", actor_save_path, critic_save_path)'''