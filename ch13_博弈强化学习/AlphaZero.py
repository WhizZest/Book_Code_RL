import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import keyboard
import torch.multiprocessing as mp
import threading
import time
import queue
import os
import sys
import pickle
import configparser
import shutil
from scipy.stats import multivariate_normal
import torch.nn.functional as F

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mcts_device = "cpu"

# 游戏环境配置
BOARD_SIZE = 15  # 使用BOARD_SIZExBOARD_SIZE棋盘加速训练
WIN_STREAK = 5
Max_step = BOARD_SIZE * BOARD_SIZE

# 训练参数，其中多进程参数可以在运行过程中调整，而不必中止程序
MCTS_simulations = 800 # 每次选择动作时进行的蒙特卡洛树搜索模拟次数
MCTS_simulations_takeback = 800 # 每次回退时进行的蒙特卡洛树搜索模拟次数（多进程参数）
MCTS_parant_root_reserve_num = 12 # 父节点保留数量，如果数值太大，可能会导致内存溢出，应根据自己的内存大小进行调整，还与棋盘大小有关，如果是9x9，可以适当调大，如果是15x15，可以适当调小（多进程参数）
takeback_max_count = 3 # 在某一步中的最大回退次数（多进程参数）
batch_size = 4096  # 每次训练的批量大小
train_frequency = 2048  # 每隔多少步进行一次训练
num_games_per_iter = 5  # 每个迭代中进行的游戏数量
isEvaluate = False # 是否进行评估，评估比较耗时，如果是15x15的棋盘，建议关闭评估
evaluate_games_num = 20  # 每次评估的游戏数量
num_epochs = 10  # 训练的轮数
learning_rate = 0.001  # 学习率
buffer_size = 500000  # 经验回放缓冲区大小
temperature = 1.0 # 温度参数
temperature_end = 0.01 # 温度参数的最小值（多进程参数）
temperature_decay_start = 30 # 温度参数开始衰减的时间步数（多进程参数）
total_iterations = 1000 # 迭代次数
dirichlet_alpha = 0.3 # 控制噪声集中程度（值越小噪声越稀疏）（多进程参数）
dirichlet_epsilon=0.25  # 原策略与噪声的混合比例（多进程参数）
c_puct = 5 # 控制探索与利用的平衡（多进程参数）
stop_training = False # 是否停止训练
epsilon_first = 0.2 # 第一步的探索概率（多进程参数）

class GomokuEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.first_player = 1
        self.current_player = self.first_player
        self.done = False
        self.winner = 0
        self.win_paths = []
        self.action_history = []

    def copy_env(self):
        new_env = GomokuEnv()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.win_paths = self.win_paths.copy()
        new_env.action_history = self.action_history.copy()
        return new_env
    
    def get_valid_moves(self):
        return (self.board == 0).astype(int)

    def step(self, action):
        row, col = action
        try:
            if self.board[row][col] != 0:
                return None  # 非法动作
        except IndexError:
            print("IndexError: ", row, col)
            sys.exit(1)  # 退出程序，返回状态码1
        
        self.board[row][col] = self.current_player
        
        if self.check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0
            reward = 0
        else:
            self.done = False
            self.winner = 0
            reward = 0
        self.action_history.append(action)
        self.current_player = -self.current_player
        return reward

    def check_win(self, row, col):
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        player = self.current_player
        
        for dx, dy in directions:
            count = 1
            self.win_paths = [(row, col)]
            for d in [-1, 1]:
                x, y = row + d*dx, col + d*dy
                while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    if self.board[x][y] == player:
                        count += 1
                        self.win_paths.append((x, y))
                        x += d*dx
                        y += d*dy
                    else:
                        break
            if count >= WIN_STREAK:
                return True
        self.win_paths = []
        return False

    def reset(self):
        self.board.fill(0)
        self.current_player = self.first_player
        self.done = False
        self.winner = 0
        self.win_paths = []
        self.action_history = []

# 神经网络模型
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1) # 输入通道数为4，输出通道数为32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 策略头
        self.policy_conv = nn.Conv2d(128, 2, 1)
        self.policy_fc = nn.Linear(2*BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE)
        
        # 价值头
        self.value_conv = nn.Conv2d(128, 1, 1)
        self.value_fc1 = nn.Linear(BOARD_SIZE*BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 输入形状: (batch_size, channels, BOARD_SIZE, BOARD_SIZE)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # 策略输出
        p = torch.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        policy = torch.softmax(p, dim=1)
        
        # 价值输出
        v = torch.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value

def generate_2d_gaussian_distribution(board_size, sigma_x, sigma_y, rho=0.0):
    """
    生成一个二维正态分布的概率矩阵，用于棋盘上第一步落子
    :param board_size: 棋盘大小 (N×N)
    :param sigma_x: x 方向的标准差
    :param sigma_y: y 方向的标准差
    :param rho: x, y 方向的相关性（默认为 0，表示独立）
    :return: 归一化的二维概率分布 (N×N numpy 数组)
    """
    # 计算棋盘中心点
    center = (board_size - 1) / 2  # 例如 15x15 的棋盘，中心是 (7, 7)
    mu = np.array([center, center])

    # 生成网格坐标
    x, y = np.meshgrid(np.arange(board_size), np.arange(board_size))
    pos = np.dstack((x, y))

    # 定义协方差矩阵
    cov_matrix = np.array([
        [sigma_x**2, rho * sigma_x * sigma_y], 
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])

    # 计算二维正态分布概率密度函数 (PDF)
    pdf = multivariate_normal(mean=mu, cov=cov_matrix).pdf(pos)

    # 归一化，使得概率总和为 1
    pdf /= pdf.sum()

    return pdf

# 蒙特卡洛树搜索
class MCTSNode:
    def __init__(self, state, player, parent=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.result = None # 添加结果属性, -1表示失败，1表示胜利，0表示平局, None表示不确定
        self.layer = 0 if parent is None else parent.layer + 1 # 添加层号
        self.simulation_env = False
        self.takeback_count = 0
        self.haveNoise = False
    
    def select_child(self, c_puct):
        if self.state is None:
            if self.parent is None:
                raise ValueError("Error: self.parent is None")
            self.state = self.parent.state.copy()
            self.state[self.action[0], self.action[1]] = self.parent.player
        total_visits = sum(child.visit_count for child in self.children)
        
        best_score = -np.inf
        best_child = None
        
        random.shuffle(self.children) # 随机打乱子节点顺序
        result = 0
        draw_list = [] # 存放平局的子节点列表
        best_child = None
        env_copy = None
        for child in self.children:
            if not child.simulation_env:
                child.simulation_env = True
                if env_copy is None:
                    env_copy = GomokuEnv()
                    env_copy.board = self.state.copy()
                    env_copy.current_player = self.player
                env_copy.step(child.action)
                if env_copy.done:
                    child.result = 1 if env_copy.winner == self.player else 0 if env_copy.winner == 0 else -1
                    if child.result == 1:
                        self.result = -1
                    return child
                env_copy.board[child.action[0], child.action[1]] = 0
                env_copy.current_player = self.player
            if child.result == 1:
                self.result = -1
                return child
            elif child.result is not None:
                result += child.result
                if child.result == 0:
                    draw_list.append(child)
                elif child.result == -1:
                    continue
            q = child.total_value / child.visit_count if child.visit_count else 0
            u = c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        if result == -len(self.children): # 所有子节点都是失败
            self.result = 1
            if self.parent is not None:
                self.parent.result = -1
        elif len(draw_list) == len(self.children): # 所有子节点都是平局
            self.result = 0
        if best_child is None:
            best_child = random.choice(self.children)
        return best_child

class MCTS_Pure:
    def __init__(self):
        self.c_puct = c_puct  # 探索系数
        self.root = None # 树重用

    def search(self, env, simulations):
        if self.root is None:
            self.root = MCTSNode(env.board.copy(), env.current_player)
        else:
            # 节点迁移
            if self.root.player != env.current_player: # 说明不是自我对弈，需要迁移到下一层
                # 迁移成功标志
                bFound = False
                new_action = env.action_history[-1]
                for child in self.root.children:
                    if new_action == child.action and child.player == env.current_player:
                        self.root = child
                        self.root.parent = None
                        #print("root visit_count: ", self.root.visit_count)
                        bFound = True
                        break
                if not bFound:
                    print("Error: No matching child node found for the current board state.")
                    self.root = MCTSNode(env.board.copy(), env.current_player)
            elif (self.root.state != env.board).any():
                print("Error: The current board state does not match the root node's state.")
                self.root = MCTSNode(env.board.copy(), env.current_player)

        env_copy = env.copy_env()
        if self.root.state is None:
            self.root.state = env_copy.board.copy()
        for _ in range(simulations):
            node = self.root
            
            # 选择
            while node.children:
                node = node.select_child(self.c_puct)
                env_copy.step(node.action)
            
            # 扩展
            if not env_copy.done:
                valid_moves = env_copy.get_valid_moves()
                policy = valid_moves / valid_moves.sum()
                
                for move in np.argwhere(valid_moves):
                    child = MCTSNode(None, -env_copy.current_player, parent=node)
                    child.prior = policy[move[0], move[1]]
                    child.action = (move[0], move[1])
                    node.children.append(child)
                
                value = 0
            else:
                if env_copy.winner == node.parent.player:
                    value = 1000
                    node.result = 1
                elif env_copy.winner == -node.parent.player:
                    value = -1000
                    node.result = -1
                else:
                    value = 0
                    node.result = 0
            
            # 回溯更新
            env_copy.done = False
            env_copy.win_paths.clear()
            while node is not None:
                node.visit_count += 1
                if node.result is None:
                    node.total_value += np.clip(value, -1, 1)
                else:
                    node.total_value += value
                if node is self.root:
                    node = None
                else:
                    env_copy.board[node.action] = 0
                    env_copy.current_player = -env_copy.current_player
                    node = node.parent
                value = -value
        
        # 修改后的概率计算部分
        visit_counts = np.array([child.visit_count for child in self.root.children])
        actions = [child.action for child in self.root.children]
        
        # 评估时选择访问次数最多的动作
        selected_idx = np.argmax(visit_counts)
        selected_action = actions[selected_idx]
        value_pred = self.root.children[selected_idx].total_value / self.root.children[selected_idx].visit_count if self.root.children[selected_idx].visit_count > 0 else 0
        
        self.root = self.root.children[selected_idx] if self.root.children else None
        if self.root.state is None: # 如果子节点的状态为空，则从环境中复制状态
            env_copy.step(self.root.action)
            self.root.state = env_copy.board.copy()
        return selected_action, value_pred, self.root.result

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class MCTS:
    def __init__(self, model):
        self.model = AlphaZeroNet().to(mcts_device)
        self.model.load_state_dict(model.state_dict())
        self.c_puct = c_puct  # 探索系数
        self.temperature = temperature  # 添加温度参数
        self.dirichlet_alpha = dirichlet_alpha  # Dirichlet分布参数α
        self.dirichlet_epsilon = dirichlet_epsilon  # 噪声混合比例
        self.root = None # 树重用
        self.parant_root_reserve_num = MCTS_parant_root_reserve_num # 父节点保留数量，如果数值太大，可能会导致内存溢出
        self.stop = False
        self.is_reserve_parent_root_when_eval = False # 评估时是否保留父节点

    def _prepare_dirichlet_noise(self, node):
        """生成与合法动作对应的Dirichlet噪声"""
        valid_mask = (node.state == 0).flatten()
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            return None
        
        # 生成Dirichlet分布噪声
        dirichlet_noise = np.random.dirichlet([self.dirichlet_alpha]*valid_count)
        
        # 映射到完整动作空间
        full_noise = np.zeros(BOARD_SIZE*BOARD_SIZE, dtype=np.float32)
        full_noise[valid_mask] = dirichlet_noise
        
        return full_noise
    
    def search(self, env, simulations, training=True, takeback=False, justThink=False):
        if self.root is None:
            self.root = MCTSNode(env.board.copy(), env.current_player)
        else:
            # 节点迁移
            bFound = False # 迁移成功标志
            if takeback: # 悔棋回退
                # 循环找父节点
                root = self.root
                takeback_max_count_temp = takeback_max_count if env.current_player == 1 else takeback_max_count + 2 # 后手的回退次数比先手多，因为后手胜率低
                while root.parent is not None:
                    root = root.parent
                    if root.player == env.current_player and (root.state == env.board).all(): # 找到匹配的父节点
                        bFound = True
                        self.root = root
                        self.root.takeback_count += 1
                        if self.root.takeback_count > takeback_max_count_temp: # 如果该节点回退次数超过阈值，则认为该节点的所有动作都会输
                            for child in self.root.children:
                                child.result = -1
                        break
                '''if i >= self.parant_root_reserve_num * 0.6:
                    self.parant_root_reserve_num *= 2 # 如果回退的父节点数超过一定阈值，则扩大父节点保留数量，防止父节点不够用
                    print(f"parant_root_reserve_num: {self.parant_root_reserve_num}")'''
                if not bFound: # 没找到匹配的父节点
                    #print(f"Error: No matching parent node found for the current board state. takeback: {takeback}, prelayer: {self.root.layer}, curlayer: {curLayer}")
                    #self.root = MCTSNode(env.board.copy(), env.current_player)
                    print(f"Info: No matching parent node found for the current board state. root.layer: {self.root.layer}, env.player:{env.current_player}")
                    return None, None, None, -1
                else:
                    # 剪枝，删除必输分支的子节点，节省内存和模拟次数
                    i = 0
                    for child in self.root.children:
                        if child.result == -1:
                            child.children = []
                            #child.visit_count = 1
                            i += 1
                    # 如果所有分支的子节点都被清空，则说明所有动作必输，前一个玩家必赢
                    if i == len(self.root.children):
                        #print(f"All children nodes of the root node have been pruned. root.takeback_count > takeback_max_count: {self.root.takeback_count > takeback_max_count}")
                        self.root.result = 1
                        if self.root.parent is not None:
                            self.root.parent.result = -1
                        return None, None, -1, -1
            elif self.root.player != env.current_player: # 说明不是自我对弈，需要迁移到下一层
                new_action = env.action_history[-1]
                for child in self.root.children:
                    if new_action == child.action and child.player == env.current_player:
                        self.root = child
                        if not self.is_reserve_parent_root_when_eval:
                            self.root.parent = None # 与他人对弈时，没有悔棋，所以不需要保留父节点
                        #print("root visit_count: ", self.root.visit_count)
                        bFound = True
                        break
                if not bFound:
                    print(f"Error: No matching child node found for the current board state. takeback: {takeback}")
                    self.root = MCTSNode(env.board.copy(), env.current_player)
            elif (self.root.state != env.board).any():
                print(f"Error: The current board state does not match the root node's state. takeback: {takeback}")
                self.root = MCTSNode(env.board.copy(), env.current_player)
        
        if self.root.state is None:
            self.root.state = env.board.copy()
        # 仅在训练模式且为根节点时准备噪声
        if training and not self.root.haveNoise:
            self.root.haveNoise = True
            noise = self._prepare_dirichlet_noise(self.root)
            if self.root.children:
                for child in self.root.children:
                    child.prior = child.prior * (1 - self.dirichlet_alpha) + noise[child.action[0]*BOARD_SIZE + child.action[1]] * self.dirichlet_alpha
        else:
            noise = None

        env_copy = env.copy_env()
        for _ in range(simulations):
            node = self.root
            
            # 选择
            while node.children:
                node = node.select_child(self.c_puct)
                env_copy.step(node.action)
            
            # 扩展
            if not env_copy.done and node.result is None:
                valid_moves = env_copy.get_valid_moves()
                with torch.no_grad():
                    state_tensor = self.preprocess_state(env_copy.board, env_copy.current_player, env_copy.current_player == env_copy.first_player, device=mcts_device)
                    policy, value = self.model(state_tensor)
                    value = -value.item() # 转为前一个玩家的动作价值（胜率）
                
                policy = policy.squeeze().cpu().numpy() * valid_moves.flatten()
                policy /= policy.sum()
                # 仅在根节点且训练模式时混合噪声
                if node is self.root and training and noise is not None and not node.haveNoise:
                    node.haveNoise = True
                    policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
                
                for move in np.argwhere(valid_moves):
                    child = MCTSNode(None, -env_copy.current_player, parent=node)
                    child.prior = policy[move[0]*BOARD_SIZE + move[1]]
                    child.action = (move[0], move[1])
                    node.children.append(child)
            else:
                if env_copy.done:
                    if env_copy.winner == node.parent.player:
                        value = 1000
                        node.result = 1
                    elif env_copy.winner == -node.parent.player:
                        value = -1000
                        node.result = -1
                    else:
                        value = 0
                        node.result = 0
                else:
                    value = node.result * 1000
            
            # 回溯更新
            env_copy.done = False
            env_copy.winner = 0
            env_copy.win_paths.clear()
            while node is not None:
                node.visit_count += 1
                if node.result is None:
                    node.total_value += np.clip(value, -1, 1)
                else:
                    node.total_value += value
                if node is self.root:
                    node = None
                else:
                    env_copy.board[node.action] = 0
                    env_copy.current_player = -env_copy.current_player
                    node = node.parent
                value = -value
            if justThink and self.stop:
                self.stop = False
                print("stop thinking")
                return None, None, None, None
        
        # 概率计算部分
        if takeback:
            visit_counts_takeback = np.array([child.visit_count if child.result is None or child.result != -1 else 1 for child in self.root.children])
        visit_counts = np.array([child.visit_count for child in self.root.children])
        actions = [child.action for child in self.root.children]
        if len(actions) == 0:
            print(f"No valid moves available, root.result: {self.root.result}, env.player:{env.current_player}")
            return None, None, -self.root.result, -self.root.result
        # 应用温度参数到概率分布
        probs = self._apply_temperature(visit_counts, tau=self.temperature)
        
        # 根据模式选择动作
        if training:
            # 训练时按概率分布采样
            if self.root.layer == 0 and not justThink and np.random.rand() < epsilon_first: # 在第一步时以一定概率随机选择动作
                probs_gaussian = generate_2d_gaussian_distribution(board_size=BOARD_SIZE, sigma_x=2.0, sigma_y=2.0)
                probs_gaussian = [probs_gaussian[action] for action in actions]
                selected_idx = np.random.choice(len(probs_gaussian), p=probs_gaussian)
                '''probs_high_temp = self._apply_temperature(visit_counts, tau=temperature_first)
                selected_idx = np.random.choice(len(probs_high_temp), p=probs_high_temp)'''
                print(f"processID: {os.getpid()}, Randomly selected action: {actions[selected_idx]} in first layer, epsilon_first: {epsilon_first}")
            else:
                selected_idx = np.random.choice(len(probs), p=probs)
            selected_action = actions[selected_idx]
        else:
            # 评估时选择访问次数最多的动作
            if takeback:
                selected_idx = np.argmax(visit_counts_takeback)
            else:
                selected_idx = np.argmax(visit_counts)
            selected_action = actions[selected_idx]
        
        # 构建完整概率图（用于训练数据）
        action_probs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for action, prob in zip(actions, probs):
            action_probs[action] = prob
        
        if justThink:
            #print("Think done")
            return None, action_probs.flatten(), None, None
        '''value_pred_list = [child.total_value / child.visit_count if child.visit_count > 0 else 0 for child in self.root.children]
        for i, child in enumerate(self.root.children):
            u = self.c_puct * child.prior * np.sqrt(self.root.visit_count) / (1 + child.visit_count)
            print(f"{i}, action: {child.action}, value_pred: {value_pred_list[i]}, u: {u}, q + u: {value_pred_list[i] + u}, visit_count: {child.visit_count}, prior: {child.prior}, result: {child.result}")
        print("selected_idx: ", selected_idx)'''
        
        value_pred = self.root.children[selected_idx].total_value / self.root.children[selected_idx].visit_count if self.root.children[selected_idx].visit_count > 0 else 0

        # 为了节省内存，只保留有限个父节点
        root = self.root
        i = 0
        while root.parent is not None:
            root = root.parent
            i += 1
            if i > self.parant_root_reserve_num or root.layer < 4:
                root.parent = None
                break

        # 根节点迁移到选择的子节点
        self.root = self.root.children[selected_idx]
        if self.root.state is None: # 如果子节点的状态为空，则从环境中复制状态
            env_copy.step(self.root.action)
            self.root.state = env_copy.board.copy()
        if takeback and self.root.result is not None and self.root.result == -1 and len(self.root.children) == 0:
            selected_action = None # 回退时，如果选择的是失败的动作，则不执行该动作，视作认输，节约计算资源
            if self.root.parent.result != 1:
                print(f"takeback, selected_action: {selected_action}, value_pred: {value_pred}, result: {self.root.result}, parent.result: {self.root.parent.result}")
        return selected_action, action_probs.flatten(), value_pred, self.root.result
    
    def _apply_temperature(self, visit_counts, tau):
        """数值稳定的温度参数应用方法"""
        visit_counts = np.array(visit_counts, dtype=np.float64)  # 使用高精度计算
        
        # 处理极端情况
        if tau < 1e-8:
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
            return probs.astype(np.float32)  # 最终返回float32
        
        # 添加平滑因子防止零值
        visit_counts += 1e-8
        
        # 数值稳定的指数计算
        scaled = softmax(np.log(visit_counts) / tau)# 使用softmax函数
        
        return scaled.astype(np.float32)  # 转换为float32节省内存

    @staticmethod
    def preprocess_state(board, player, is_first_player, device):
        """
        将棋盘状态转换为模型输入
        :param board: 棋盘状态
        :param player: 当前玩家
        :param is_first_player: 是否为先手
        """
        plane1 = (board == player).astype(float) # 当前玩家棋子
        plane2 = (board == -player).astype(float)  # 对手棋子
        plane3 = (board == 0).astype(float) # 空位
        plane4 = np.full_like(board, is_first_player, dtype=float)  # 是否为先手
        return torch.FloatTensor(np.stack([plane1, plane2, plane3, plane4])).unsqueeze(0).to(device)  # 添加设备转移

def augment_data(state, policy):
    """生成所有对称变换后的数据"""
    aug_states = []
    aug_policies = []
    
    # 原始数据
    aug_states.append(state)
    aug_policies.append(policy)
    
    # 旋转增强
    for k in range(1,4):
        rotated_state = np.rot90(state, k=k)
        '''if np.all(rotated_state == state): # 避免重复添加
            continue'''
        rotated_policy = np.rot90(policy.reshape(BOARD_SIZE, BOARD_SIZE), k=k).flatten()
        aug_states.append(rotated_state)
        aug_policies.append(rotated_policy)
    
    # 镜像增强
    mirrors = [
        lambda x: np.fliplr(x),   # 水平镜像
        lambda x: np.flipud(x),   # 垂直镜像
        lambda x: x.T,            # 主对角线镜像
        lambda x: np.fliplr(x.T)  # 副对角线镜像
    ]
    for mirror in mirrors:
        mirrored_state = mirror(state)
        '''if np.all(mirrored_state == state): # 避免重复添加
            continue'''
        mirrored_policy = mirror(policy.reshape(BOARD_SIZE, BOARD_SIZE)).flatten()
        aug_states.append(mirrored_state)
        aug_policies.append(mirrored_policy)
    
    return aug_states, aug_policies

def play_single_eval_gamedata(global_model, bExit, eval_game_data, result_queue):
    """ 运行评估产生的游戏数据 """
    eval_game_actions = eval_game_data[0]
    temperature_decay = (temperature - temperature_end) / (Max_step - temperature_decay_start)  # 计算温度衰减率
    game_data = []
    steps = 0
    env = GomokuEnv()
    mcts = MCTS(model=global_model)    
    steps_TakeBack = eval_game_data[1] # 回退时记录当前步数
    game_data_TackBack_index = steps_TakeBack * 8 # 回退时记录当前游戏数据索引
    temperature_history = [] # 记录温度变化
    action_history_TackBack = [] # 回退时记录动作历史
    current_player_TakeBack = 0
    result = None
    takeback_count = 0
    while True:
        while not env.done:
            if bExit.value:
                return
            mcts.is_reserve_parent_root_when_eval = True
            if steps < len(eval_game_actions):
                _, action_probs, _, _ = mcts.search(env, training=True, simulations=MCTS_simulations, justThink=True)
                action = eval_game_actions[steps]
                if steps_TakeBack == steps:
                    action_history_TackBack = env.action_history.copy()
                    state_TakeBack = env.board.copy()
                    current_player_TakeBack = env.current_player
                    takeback_count += 1
            else:
                action, action_probs, value_pred, result = mcts.search(env, training=(steps_TakeBack != steps), simulations=MCTS_simulations if steps_TakeBack != steps else MCTS_simulations_takeback, takeback=(steps_TakeBack == steps))
                if steps_TakeBack == steps:
                    steps_TakeBack = -1
                if result is not None and result != 0 and steps_TakeBack < 0 and (len(env.action_history)-2) >= 0 and value_pred is not None: # 如果预测到会输，则标记回退点
                    if result == -1:
                        takeback_deltaStep = -2
                    else:
                        takeback_deltaStep = -1
                    steps_TakeBack = steps + takeback_deltaStep
                    game_data_TackBack_index = len(game_data) + 8 * takeback_deltaStep # 2步，每步数据增强所以有8个数据
                    state_TakeBack = env.board.copy()
                    current_player_TakeBack = env.current_player if result == -1 else -env.current_player
                    for move in env.action_history[takeback_deltaStep:]: # 回退时将最后两步棋子置为0
                        state_TakeBack[move[0], move[1]] = 0
                    action_history_TackBack = env.action_history[:takeback_deltaStep]
                    takeback_count += 1
                if action is None:
                    if result is not None:
                        if result == -1:
                            env.winner = -env.current_player
                        elif result == 1:
                            env.winner = env.current_player
                        else:
                            env.winner = 0
                    break

            state = env.board.copy()
            states_aug, policies_aug = augment_data(state, action_probs)
            
            for s, p in zip(states_aug, policies_aug):
                game_data.append((s, env.current_player, p))

            env.step(action)
            steps += 1
            temperature_history.append(mcts.temperature)
            if steps > temperature_decay_start:
                mcts.temperature -= temperature_decay
                mcts.temperature = max(mcts.temperature, temperature_end)

        eval_game_actions = []
        winner = env.winner
        if winner == 0:
            print(f"Draw, processID: {os.getpid()}, steps: {steps}, steps_TakeBack: {steps_TakeBack}, game_data_TackBack_index: {game_data_TackBack_index}") # 如果平局，则打印信息，15x15棋盘不容易出现平局，回退次数越多，越容易出现平局，9x9棋盘更容易出现平局，根据是否出现平局，可以粗略估计模型的棋力
            steps_TakeBack = -1
            game_data_TackBack_index = 0
        elif steps_TakeBack >= 0 and winner == current_player_TakeBack:
            #game_data_TackBack_index = 0
            print(f"takeback error, processID: {os.getpid()}, steps: {steps}, steps_TakeBack: {steps_TakeBack}, current_player_TakeBack: {current_player_TakeBack}")
            #steps_TakeBack = -1
        # 将每个样本单独放入队列
        for s, player, p in game_data[game_data_TackBack_index:]: # 退点之前的数据还不确定胜负，不放入队列
            state_tensor = MCTS.preprocess_state(s, player, player == env.first_player, device="cpu")
            policy_target = torch.FloatTensor(p)
            value_target = torch.FloatTensor([1 if winner == player else 0 if winner == 0 else -1])
            result_queue.put( (state_tensor, policy_target, value_target) )
        if steps_TakeBack >= 0:
            env.reset()
            env.action_history = action_history_TackBack
            env.board = state_TakeBack
            env.current_player = current_player_TakeBack
            game_data = game_data[:game_data_TackBack_index] # 删除回退点之后的数据
            game_data_TackBack_index = 0
            steps = steps_TakeBack
            temperature_history = temperature_history[:steps]
            mcts.temperature = temperature_history[-1]
        else:
            break
    print(f"Game over, processID: {os.getpid()}, steps: {steps}, steps_TakeBack: {steps_TakeBack}, winner: {winner}, current_player_TakeBack: {current_player_TakeBack}, first action: {env.action_history[0]}, takeback_count: {takeback_count}")

def play_single_game(global_model, bExit, result_queue):
    """ 运行一局自我对弈 """
    temperature_decay = (temperature - temperature_end) / (Max_step - temperature_decay_start)  # 计算温度衰减率
    game_data = []
    steps = 0
    env = GomokuEnv()
    mcts = MCTS(model=global_model)
    game_data_TackBack_index = 0 # 回退时记录当前游戏数据索引
    actions_TackBack = {} # 回退时记录动作 key:步数，value:动作列表
    steps_TakeBack = -1 # 回退时记录当前步数
    temperature_history = [] # 记录温度变化
    action_history_TackBack = [] # 回退时记录动作历史
    current_player_TakeBack = 0
    takeback_error_flag = False
    takeback_count = 0
    while True:
        while not env.done:
            if bExit.value:
                return

            action, action_probs, value_pred, result = mcts.search(env, training=(steps_TakeBack != steps), simulations=MCTS_simulations if steps_TakeBack != steps else MCTS_simulations_takeback, takeback=(steps_TakeBack == steps))
            if steps_TakeBack == steps:
                steps_TakeBack = -1
            if result is not None and result != 0 and steps_TakeBack < 0 and (len(env.action_history)-2) >= 0 and value_pred is not None: # 如果预测到会输，则标记回退点
                if result == -1:
                    takeback_deltaStep = -2
                else:
                    takeback_deltaStep = -1
                steps_TakeBack = steps + takeback_deltaStep
                action_temp = env.action_history[takeback_deltaStep]
                action_list = actions_TackBack.get(steps_TakeBack, [])
                #if len(action_list) < takeback_max_count: # 限制回退次数
                game_data_TackBack_index = len(game_data) + 8 * takeback_deltaStep # 2步，每步数据增强所以有8个数据
                state_TakeBack = env.board.copy()
                current_player_TakeBack = env.current_player if result == -1 else -env.current_player
                action_list.append(action_temp)
                actions_TackBack[steps_TakeBack] = action_list
                for move in env.action_history[takeback_deltaStep:]: # 回退时将最后两步棋子置为0
                    state_TakeBack[move[0], move[1]] = 0
                action_history_TackBack = env.action_history[:takeback_deltaStep]
                takeback_count += 1
                '''else:
                    steps_TakeBack = -1
                    game_data_TackBack_index = 0'''
            if action is None:
                if result is not None:
                    if result == -1:
                        env.winner = -env.current_player
                    elif result == 1:
                        env.winner = env.current_player
                    else:
                        env.winner = 0
                break
                
            state = env.board.copy()
            states_aug, policies_aug = augment_data(state, action_probs)

            for s, p in zip(states_aug, policies_aug):
                game_data.append((s, env.current_player, p))

            env.step(action)
            steps += 1
            temperature_history.append(mcts.temperature)
            if steps > temperature_decay_start:
                mcts.temperature -= temperature_decay
                mcts.temperature = max(mcts.temperature, temperature_end)

        winner = env.winner
        if takeback_error_flag:
            print(f"takeback_error_flag, processID: {os.getpid()}, steps: {steps}, steps_TakeBack: {steps_TakeBack}, winner: {winner}, current_player_TakeBack: {current_player_TakeBack}")
        takeback_error_flag = False
        if winner == 0:
            print(f"Draw, processID: {os.getpid()}, steps: {steps}, steps_TakeBack: {steps_TakeBack}, game_data_TackBack_index: {game_data_TackBack_index}") # 如果平局，则打印信息，15x15棋盘不容易出现平局，回退次数越多，越容易出现平局，9x9棋盘更容易出现平局，根据是否出现平局，可以粗略估计模型的棋力
            steps_TakeBack = -1
            game_data_TackBack_index = 0
        elif steps_TakeBack >= 0 and winner == current_player_TakeBack:
            #game_data_TackBack_index = 0
            print(f"takeback error, processID: {os.getpid()}, steps: {steps}, steps_TakeBack: {steps_TakeBack}, current_player_TakeBack: {current_player_TakeBack}")
            #steps_TakeBack = -1
            takeback_error_flag = True
        # 将每个样本单独放入队列
        for s, player, p in game_data[game_data_TackBack_index:]: # 退点之前的数据还不确定胜负，不放入队列
            state_tensor = MCTS.preprocess_state(s, player, player == env.first_player, device="cpu")
            policy_target = torch.FloatTensor(p)
            value_target = torch.FloatTensor([1 if winner == player else 0 if winner == 0 else -1])
            result_queue.put( (state_tensor, policy_target, value_target) )
        if steps_TakeBack >= 0:
            print_str = None
            env.reset()
            env.action_history = action_history_TackBack
            env.board = state_TakeBack
            env.current_player = current_player_TakeBack
            game_data = game_data[:game_data_TackBack_index] # 删除回退点之后的数据
            game_data_TackBack_index = 0
            steps = steps_TakeBack
            temperature_history = temperature_history[:steps]
            mcts.temperature = temperature_history[-1]
        else:
            break
    print(f"Game over, processID: {os.getpid()}, steps: {steps}, steps_TakeBack: {steps_TakeBack}, winner: {winner}, current_player_TakeBack: {current_player_TakeBack}, first action: {env.action_history[0]}, takeback_count: {takeback_count}")

def evaluate_single_game(global_model, bExit, result_queue, best_model=None, current_model_player=None):
    """ 运行一局评估对局 """
    env = GomokuEnv()
    mcts = MCTS(model=global_model)
    if best_model is not None:
        mcts_best = MCTS(model=best_model)
    else:
        mcts_pure = MCTS_Pure()
    
    if current_model_player is None:
        # 随机待评估模型的玩家
        current_model_player = random.choice([1, -1])

    while not env.done:
        if bExit.value:
            print(f"Evaluation process {mp.current_process().pid} exiting due to ESC...")
            return 0

        if env.current_player == current_model_player:
            action, _, value_pred, result = mcts.search(env, training=False, simulations=MCTS_simulations)
        else:
            '''valid_moves = np.argwhere(env.get_valid_moves())
            action = tuple(valid_moves[np.random.choice(len(valid_moves))])'''
            if best_model is not None:
                action, _, value_pred, result = mcts_best.search(env, training=False, simulations=MCTS_simulations)
            else:
                action, value_pred, result = mcts_pure.search(env, simulations=MCTS_simulations)

        env.step(action)

    result_queue.put(1 if env.winner == current_model_player else 0 if env.winner == 0 else -1)

# 训练流程
class AlphaZeroTrainer:
    def __init__(self, modelFileName=None, cache_file='cache.pkl', config_file='config.ini', isEvaluate=False):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(self.script_dir, "model")
        self.isEvaluate = isEvaluate
        os.makedirs(self.save_path, exist_ok=True)
        self.model = AlphaZeroNet().to(device)  # 初始化时转移到设备
        if modelFileName is not None:
            filePath = os.path.join(self.script_dir, modelFileName)
            self.model.load_state_dict(torch.load(filePath, map_location=device, weights_only=True))
            print("加载模型成功")
        self.global_model = AlphaZeroNet().to(mcts_device)  # 初始化时转移到设备
        self.global_model.load_state_dict(self.model.state_dict())
        self.global_model.share_memory()  # 共享模型参数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.best_model = None
        self.buffer = deque(maxlen=buffer_size)
        self.batch_data_count = 0
        self.cache_file = os.path.join(self.script_dir, cache_file)
        self.cache_file_temp = os.path.join(self.script_dir, 'cache_temp.pkl')
        self.config_file = os.path.join(self.script_dir, config_file)
        self.win_history = []
        self.lose_history = []
        self.draw_history = []
        self._write_param_to_config()
    
    def _write_param_to_config(self):
        config = configparser.ConfigParser()
        config['TRAINING'] = {'learning_rate': str(learning_rate)}
        config['TRAINING']['batch_size'] = str(batch_size)
        config['TRAINING']['num_epochs'] = str(num_epochs)
        config['TRAINING']['train_frequency'] = str(train_frequency)
        config['TRAINING']['MCTS_simulations'] = str(MCTS_simulations)
        config['TRAINING']['num_games_per_iter'] = str(num_games_per_iter)
        config['TRAINING']['stop_training'] = str(int(stop_training))
        with open(self.config_file, 'w') as configfile:
            config.write(configfile)
    
    def _read_param_from_config(self):
        global batch_size
        global num_epochs
        global train_frequency
        global MCTS_simulations
        global num_games_per_iter
        global stop_training

        config = configparser.ConfigParser()
        config.read(self.config_file)
        lr = float(config['TRAINING']['learning_rate'])
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        batch_size = int(config['TRAINING']['batch_size'])
        num_epochs = int(config['TRAINING']['num_epochs'])
        train_frequency = int(config['TRAINING']['train_frequency'])
        MCTS_simulations = int(config['TRAINING']['MCTS_simulations'])
        num_games_per_iter = int(config['TRAINING']['num_games_per_iter'])
        stop_training = bool(int(config['TRAINING']['stop_training']))

    def self_play_eval_gamedata(self, num_games=4, bExit=None):
        """ 从评估数据中并行执行自我对弈 """
        # 加载评估数据列表，根据num_games进行切分，分批次调用play_batch_eval_gamedata
        eval_gamedatas = self.load_eval_cache()
        if len(eval_gamedatas) == 0:
            return
        num_games = min(mp.cpu_count(), num_games)
        result_count = 0
        for i in range(0, len(eval_gamedatas), num_games):
            endIndex = min(i + num_games, len(eval_gamedatas))
            result_count += self.play_batch_eval_gamedata(eval_gamedatas[i:endIndex], bExit)
            if bExit.value:
                break
        print(f"self_play_eval_gamedata result_count:{result_count}")

    def play_batch_eval_gamedata(self, eval_gamedatas, bExit=None):
        """ 从评估数据中批量并行执行自我对弈 """
        num_workers = len(eval_gamedatas)
        result_queue = mp.Queue()
        processes = []
        result_count = 0
        for i in range(num_workers):
            p = mp.Process(target=play_single_eval_gamedata, 
                args=(self.global_model, bExit, eval_gamedatas[i], result_queue))
            p.start()
            processes.append(p)
        
        # 在子进程运行期间持续处理队列
        while any(p.is_alive() for p in processes):
            try:
                # 非阻塞获取数据
                while not result_queue.empty():
                    result = result_queue.get(block=False)
                    self.buffer.append(result)
                    self.batch_data_count += 1
                    if self.batch_data_count >= train_frequency and len(self.buffer) >= batch_size:
                        self.batch_data_count = 0
                        epochs = len(self.buffer) // batch_size
                        epochs = np.clip(epochs, 1, num_epochs)
                        self.train(batch_size=batch_size, epochs=epochs)
                    result_count += 1
            except Exception as e:
                if not isinstance(e, queue.Empty):
                    print(f"Error1 processing queue: {e}")
            time.sleep(0.1)  # 避免CPU过载

        # 确保所有进程结束
        for p in processes:
            p.join()

        # 处理剩余数据
        while not result_queue.empty():
            try:
                result = result_queue.get(block=False)
                self.buffer.append(result)
                self.batch_data_count += 1
                if self.batch_data_count >= train_frequency and len(self.buffer) >= batch_size:
                    self.batch_data_count = 0
                    epochs = len(self.buffer) // batch_size
                    epochs = np.clip(epochs, 1, num_epochs)
                    self.train(batch_size=batch_size, epochs=epochs)
                result_count += 1
            except Exception as e:
                if not isinstance(e, queue.Empty):
                    print(f"Error2 processing queue: {e}")
                break
        
        self.global_model.load_state_dict(self.model.state_dict())  # 更新全局模型
        print(f"Collected {result_count} samples, avereage steps: {result_count / num_workers / 8}")
        return result_count
    
    def self_play(self, num_games=100, bExit=None):
        """ 并行执行 num_games 场自我对弈 """
        num_workers = min(mp.cpu_count(), num_games)
        result_queue = mp.Queue()
        processes = []
        result_count = 0
        for _ in range(num_workers):
            p = mp.Process(target=play_single_game, 
                args=(self.global_model, bExit, result_queue))
            p.start()
            processes.append(p)

        # 在子进程运行期间持续处理队列
        while any(p.is_alive() for p in processes):
            try:
                # 非阻塞获取数据
                while not result_queue.empty():
                    result = result_queue.get(block=False)
                    self.buffer.append(result)
                    self.batch_data_count += 1
                    if self.batch_data_count >= train_frequency and len(self.buffer) >= batch_size:
                        self.batch_data_count = 0
                        epochs = len(self.buffer) // batch_size
                        epochs = np.clip(epochs, 1, num_epochs)
                        self.train(batch_size=batch_size, epochs=epochs)
                    result_count += 1
            except Exception as e:
                if not isinstance(e, queue.Empty):
                    print(f"Error1 processing queue: {e}")
            time.sleep(0.1)  # 避免CPU过载

        # 确保所有进程结束
        for p in processes:
            p.join()

        # 处理剩余数据
        while not result_queue.empty():
            try:
                result = result_queue.get(block=False)
                self.buffer.append(result)
                self.batch_data_count += 1
                if self.batch_data_count >= train_frequency and len(self.buffer) >= batch_size:
                    self.batch_data_count = 0
                    epochs = len(self.buffer) // batch_size
                    epochs = np.clip(num_epochs, 1, num_epochs)
                    self.train(batch_size=batch_size, epochs=epochs)
                result_count += 1
            except Exception as e:
                if not isinstance(e, queue.Empty):
                    print(f"Error2 processing queue: {e}")
                break

        self.global_model.load_state_dict(self.model.state_dict())  # 更新全局模型
        print(f"Collected {result_count} samples, avereage steps: {result_count / num_games / 8}")
    
    def train(self, batch_size=32, epochs=10):
        if len(self.buffer) < batch_size:
            return
        
        for _ in range(epochs):
            batch = random.sample(self.buffer, batch_size)
            states, policy_targets, value_targets = zip(*batch)
            # 将数据转移到设备
            states = torch.cat([s.to(device) for s in states])
            policy_targets = torch.stack(policy_targets).to(device)
            value_targets = torch.cat(value_targets).to(device)

            self.optimizer.zero_grad()
            policy_pred, value_pred = self.model(states)
            
            # 添加维度验证
            assert policy_pred.shape == policy_targets.shape, \
                f"Pred shape {policy_pred.shape} != Target shape {policy_targets.shape}"
            
            policy_loss = -torch.mean(torch.sum(policy_targets * torch.log(policy_pred + 1e-10), dim=1))
            value_loss = torch.mean((value_pred.squeeze() - value_targets)**2)
            loss = policy_loss + value_loss
            
            loss.backward()
            self.optimizer.step()
        entropy = -torch.mean(torch.sum(policy_pred * torch.log(policy_pred + 1e-10), dim=1))
        print(f"Training completed, loss: {loss.item():.4f}, entropy: {entropy.item():.4f}")

    def evaluate(self, num_games=20, bExit=None):
        """ 并行评估 """
        if bExit.value:
            print(f"Evaluation process exiting due to ESC...")
            return 0, 0, 0, 0
        #self.global_model.load_state_dict(self.model.state_dict())
        num_workers = min(mp.cpu_count(), num_games)

        # 创建一个队列来存储子进程的返回值
        result_queue = mp.Queue()

        # 创建并启动子进程
        start_t = time.time()
        processes = []
        current_model_player = 1
        for _ in range(num_workers):
            p = mp.Process(target=evaluate_single_game, args=(self.global_model, bExit, result_queue, self.best_model, current_model_player))
            current_model_player = - current_model_player # 切换当前模型玩家的先后手
            p.start()
            processes.append(p)

        # 等待所有子进程完成
        for p in processes:
            p.join()
        
        # 从队列中获取结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        win_count = results.count(1) # 统计胜场数
        lose_count = results.count(-1) # 统计败场数
        draw_count = results.count(0) # 统计平局场数

        return win_count, lose_count, draw_count, time.time() - start_t
    
    def update_plot(self):
        """动态更新胜率曲线"""
        self.line1.set_xdata(np.arange(len(self.win_history)))
        self.line1.set_ydata(self.win_history)
        self.line2.set_xdata(np.arange(len(self.lose_history)))
        self.line2.set_ydata(self.lose_history)
        self.line3.set_xdata(np.arange(len(self.draw_history)))
        self.line3.set_ydata(self.draw_history)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.05)  # 短暂暂停让图表更新

    def run(self, iterations=10, starting_iteration=0):
        bExit = mp.Value('b', False)  # 使用共享变量
        bStopProcess = mp.Value('b', False)  # 使用共享变量
        if self.isEvaluate:
            # 初始化动态绘图
            plt.ion()  # 开启交互模式
            self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
            self.ax.set_title('Training Progress')
            self.ax.set_xlabel('Evaluation Number')
            self.ax.set_ylabel('Win Rate')
            self.ax.grid(True)
            self.line1, = self.ax.plot([], [], 'g-', label='Win count')
            self.line2, = self.ax.plot([], [], 'r-', label='Lose count')
            self.line3, = self.ax.plot([], [], 'b-', label='Draw count')
            self.ax.legend()

        # 启动键盘监听线程
        listener = threading.Thread(target=self._esc_listener, args=(bExit, bStopProcess,))
        listener.start()

        for i in range(starting_iteration, iterations):
            if bExit.value:
                print("[INFO] ESC detected. Stopping training loop...")
                break
            print(f"Iteration {i}/{iterations}")
            # 每5次迭代评估一次
            if i % 5 == 0:
                if self.isEvaluate:
                    checkpoint = False
                    win_count, lose_count, draw_count, cost_time = self.evaluate(bExit=bExit, num_games=evaluate_games_num)
                    if self.best_model is None:
                        if win_count == evaluate_games_num:
                            self.best_model = AlphaZeroNet().to(mcts_device)
                            self.best_model.load_state_dict(self.model.state_dict())
                            self.best_model.share_memory()
                            checkpoint = True
                            print(f"Best model updated at iteration {i}")
                    elif self.best_model is not None and win_count > lose_count:
                        self.best_model.load_state_dict(self.model.state_dict())
                        checkpoint = True
                        print(f"Best model updated at iteration {i}")
                    self.win_history.append(win_count)
                    self.lose_history.append(lose_count)
                    self.draw_history.append(draw_count)
                    print(f"Iteration {i}, win_count: {win_count}, lose_count: {lose_count}, draw_count: {draw_count}, Cost Time: {cost_time:.2f}")
                    self.update_plot()
                else:
                    checkpoint = i>0
            
                if checkpoint:
                    # 保存模型检查点
                    # 检查文件夹是否存在，不存在则创建
                    os.makedirs(self.save_path, exist_ok=True)
                    filePath = os.path.join(self.save_path, f"az_model_{i}.pth")
                    torch.save(self.model.state_dict(), filePath)
                    self.save_cache(self.cache_file_temp)
                    if self.isEvaluate:
                        plt.savefig(os.path.join(self.script_dir, f"az_plot_checkpoint.png"))

            self.self_play_eval_gamedata(num_games=num_games_per_iter, bExit=bExit)
            self.self_play(num_games=num_games_per_iter, bExit=bExit)
            #self.train(batch_size=batch_size, epochs=num_epochs)

            # 读取配置文件中的学习率
            self._read_param_from_config()

            if stop_training:
                print("停止训练")
                break
        
        bStopProcess.value = True  # 停止监听进程
        if self.best_model is not None:
            torch.save(self.best_model.state_dict(), os.path.join(self.save_path, "az_model_best.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "az_model_final.pth"))
        self.save_cache(self.cache_file)
        if bExit.value == False and listener is not None and listener.is_alive():
            listener.join() # 等待监听线程结束

        if self.isEvaluate:
            plt.ioff()  # 关闭交互模式
            plt.show()
    
    def _esc_listener(self, bExit, bStopProcess):
        """ 监听 ESC 按键，通知所有进程退出 """
        print("ESC 按键监听线程已启动")
        while True:
            if keyboard.is_pressed("esc"):
                print("[INFO] ESC detected. Stopping all processes...")
                bExit.value = True  # 设置退出标志
                break
            if bStopProcess.value:
                break
            time.sleep(0.1)  # 避免 CPU 过载
    
    def save_cache(self, cache_file):
        try:
            with open(cache_file, 'wb') as file:
                pickle.dump(self.buffer, file)
            print("缓存已保存到硬盘")
        except Exception as e:
            print(f"保存缓存时发生错误: {e}")

    def load_cache(self):
        try:
            with open(self.cache_file, 'rb') as file:
                buffer_temp = pickle.load(file)
            self.buffer = deque(buffer_temp, maxlen=buffer_size)
            print("缓存已从硬盘加载, buffer size:", len(self.buffer))
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                print("未找到缓存文件，将创建新的缓存")
            else:
                print(f"加载缓存时发生错误: {e}")
    
    def load_cache_list(self, cache_file_list):
        # 从缓存文件列表中加载缓存, 用于合并多个缓存文件
        buffer_temp = []
        for cache_file in cache_file_list:
            try:
                cache_file = os.path.join(self.script_dir, cache_file)
                with open(cache_file, 'rb') as file:
                    buffer_temp.extend(pickle.load(file))
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    print(f"未找到缓存文件: {cache_file}")
                else:
                    print(f"加载缓存时发生错误: {cache_file}: {e}")
                return
        self.buffer = deque(buffer_temp, maxlen=len(buffer_temp))
        print("缓存已从硬盘加载, buffer size:", len(self.buffer))
    
    def load_eval_cache(self):
        # 加载self.script_dir路径下eval_buffer文件夹中的所有pkl文件，每个文件都是一个包含若干评估数据的列表，把这些列表合并为一个列表eval_gamedatas
        eval_gamedatas = []
        file_name_list = []
        eval_buffer_dir = os.path.join(self.script_dir, "eval_buffer")
        if os.path.exists(eval_buffer_dir):
            for file_name in os.listdir(eval_buffer_dir):
                if file_name.endswith(".pkl"):
                    file_path = os.path.join(eval_buffer_dir, file_name)
                    with open(file_path, 'rb') as file:
                        eval_gamedatas.extend(pickle.load(file))
                    file_name_list.append(file_name)
            if len(eval_gamedatas) > 0:
                print("评估缓存已从硬盘加载, buffer size:", len(eval_gamedatas))
            # 将file_name_list中的文件转移到eval_buffer_old文件夹中
            eval_buffer_old_dir = os.path.join(self.script_dir, "eval_buffer_old")
            if not os.path.exists(eval_buffer_old_dir):
                os.makedirs(eval_buffer_old_dir)
            for file_name in file_name_list:
                file_path = os.path.join(eval_buffer_dir, file_name)
                new_file_path = os.path.join(eval_buffer_old_dir, file_name)
                shutil.move(file_path, new_file_path)
            return eval_gamedatas
        else:
            #print("未找到评估缓存文件夹")
            return []

if __name__ == "__main__":
    print(f"Using device: {device}, mcts_device: {mcts_device}, cpu_cores: {mp.cpu_count()}")
    trainer = AlphaZeroTrainer(modelFileName=None, isEvaluate=isEvaluate)
    trainer.load_cache()
    trainer.run(iterations=total_iterations)