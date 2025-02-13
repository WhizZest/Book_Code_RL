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

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mcts_device = "cpu"

# 游戏环境配置
BOARD_SIZE = 9  # 使用9x9棋盘加速训练
WIN_STREAK = 5
Max_step = BOARD_SIZE * BOARD_SIZE

# 训练参数
MCTS_simulations = 400 # 每次选择动作时进行的蒙特卡洛树搜索模拟次数
batch_size = 2048  # 每次训练的批量大小
num_games_per_iter = 5  # 每个迭代中进行的游戏数量
evaluate_games_num = 20  # 每次评估的游戏数量
num_epochs = 10  # 训练的轮数
learning_rate = 0.001  # 学习率
buffer_size = 30000  # 经验回放缓冲区大小
temperature = 1.0 # 温度参数
temperature_end = 0.01 # 温度参数的最小值
temperature_decay_start = 30
total_iterations = 500 # 迭代次数
dirichlet_alpha = 0.3 # 控制噪声集中程度（值越小噪声越稀疏）
dirichlet_epsilon=0.25  # 原策略与噪声的混合比例
c_puct = 5 # 控制探索与利用的平衡

bExit = mp.Value('b', False)  # 使用共享变量
bStopProcess = mp.Value('b', False)  # 使用共享变量

def generate_chebyshev_matrix(size):
    # 创建一个全零的方阵
    matrix = np.zeros((size, size))
    
    # 计算中心的位置
    center = size // 2
    
    # 遍历方阵的每个位置
    for i in range(size):
        for j in range(size):
            # 计算切比雪夫距离
            distance = max(abs(i - center), abs(j - center))
            # 将距离乘以0.1并赋值给方阵
            matrix[i, j] = 1 - distance * 0.1
    
    return matrix

chebyshev_matrix = generate_chebyshev_matrix(BOARD_SIZE)

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
                return None, -1  # 非法动作
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
            reward = 0
        else:
            reward = 0
        self.action_history.append(action)
        self.current_player = -self.current_player
        return self.board.copy(), reward

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
        self.search_action = None # 搜索动作：用于回退复盘时深度搜索最优动作
    
    def select_child(self, c_puct):
        total_visits = sum(child.visit_count for child in self.children)
        
        best_score = -np.inf
        best_child = None
        
        random.shuffle(self.children) # 随机打乱子节点顺序
        result = 0
        draw_list = [] # 存放平局的子节点列表
        best_child = None
        for child in self.children:
            env_copy = GomokuEnv()
            env_copy.board = self.state.copy()
            env_copy.current_player = self.player
            env_copy.step(child.action)
            if env_copy.done:
                child.result = 1 if env_copy.winner == self.player else 0 if env_copy.winner == 0 else -1
                self.result = -1
                return child
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
                for child in self.root.children:
                    if child.state.shape == env.board.shape and (child.state == env.board).all() and child.player == env.current_player:
                        self.root = child
                        self.root.parent = None
                        #print("root visit_count: ", self.root.visit_count)
                        bFound = True
                        break
                if not bFound:
                    # 回退三层
                    if self.root.parent is not None and self.root.parent.parent is not None and self.root.parent.parent.parent is not None and (self.root.parent.parent.parent.state == env.board).all():
                        self.root = self.root.parent.parent.parent
                        bFound = True
                        #print("root visit_count: ", self.root.visit_count)
                    else:
                        print("Error: No matching child node found for the current board state.")
                        self.root = MCTSNode(env.board.copy(), env.current_player)
            elif (self.root.state != env.board).any():
                print("Error: The current board state does not match the root node's state.")
                self.root = MCTSNode(env.board.copy(), env.current_player)

        action_probs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        valid_moves = env.get_valid_moves()
        if valid_moves.sum() <= (BOARD_SIZE*BOARD_SIZE - WIN_STREAK * 2 + 1):
            for move in np.argwhere(valid_moves):
                action = (move[0], move[1])
                env_copy = GomokuEnv()
                env_copy.board = env.board.copy()
                env_copy.current_player = env.current_player
                env_copy.step(action)
                if env_copy.done:
                    action_probs[move[0], move[1]] = 1.0
                    return action, 1, 1
        for i in range(simulations):
            node = self.root
            env_copy = GomokuEnv()
            env_copy.board = node.state.copy()
            env_copy.current_player = node.player
            
            # 选择
            while node.children:
                node = node.select_child(self.c_puct)
                env_copy.step(node.action)
            
            # 扩展
            if not env_copy.done:
                valid_moves = env_copy.get_valid_moves()
                policy = valid_moves / valid_moves.sum()
                
                for move in np.argwhere(valid_moves):
                    child_state = env_copy.board.copy()
                    child_state[move[0], move[1]] = env_copy.current_player
                    child = MCTSNode(child_state, -env_copy.current_player, parent=node)
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
            while node is not None:
                node.visit_count += 1
                if node.result is None:
                    node.total_value += np.clip(value, -1, 1)
                else:
                    node.total_value += value
                if node is self.root:
                    node = None
                else:
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
    
    def sync_models(self, model):
        """同步训练好的模型参数到MCTS中的模型"""
        self.model.load_state_dict(model.state_dict())
    
    def search(self, env, simulations, training=True):
        if self.root is None:
            self.root = MCTSNode(env.board.copy(), env.current_player)
        else:
            # 节点迁移
            if self.root.player != env.current_player: # 说明不是自我对弈，需要迁移到下一层
                # 迁移成功标志
                bFound = False
                for child in self.root.children:
                    if child.state.shape == env.board.shape and (child.state == env.board).all() and child.player == env.current_player:
                        self.root = child
                        self.root.parent = None
                        #print("root visit_count: ", self.root.visit_count)
                        bFound = True
                        break
                if not bFound:
                    # 回退三层
                    if self.root.parent is not None and self.root.parent.parent is not None and self.root.parent.parent.parent is not None and (self.root.parent.parent.parent.state == env.board).all():
                        self.root = self.root.parent.parent.parent
                        bFound = True
                        #print("root visit_count: ", self.root.visit_count)
                    else:
                        print("Error: No matching child node found for the current board state.")
                        self.root = MCTSNode(env.board.copy(), env.current_player)
            elif (self.root.state != env.board).any():
                print("Error: The current board state does not match the root node's state.")
                self.root = MCTSNode(env.board.copy(), env.current_player)
        # 仅在训练模式且为根节点时准备噪声
        if training:
            noise = self._prepare_dirichlet_noise(self.root)
            if self.root.children:
                for child in self.root.children:
                    child.prior = child.prior * (1 - self.dirichlet_alpha) + noise[child.action[0]*BOARD_SIZE + child.action[1]] * self.dirichlet_alpha
        else:
            noise = None

        # 模拟下一步是否会结束
        action_probs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        valid_moves = env.get_valid_moves()
        if valid_moves.sum() <= (BOARD_SIZE*BOARD_SIZE - WIN_STREAK * 2 + 1):
            for move in np.argwhere(valid_moves):
                action = (move[0], move[1])
                env_copy = GomokuEnv()
                env_copy.board = env.board.copy()
                env_copy.current_player = env.current_player
                env_copy.step(action)
                if env_copy.done:
                    action_probs[move[0], move[1]] = 1.0
                    return action, action_probs.flatten(), 1 + self.c_puct, 1
        for _ in range(simulations):
            node = self.root
            env_copy = GomokuEnv()
            env_copy.board = node.state.copy()
            env_copy.current_player = node.player
            
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
                if node is self.root and training and noise is not None: 
                    policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
                
                for move in np.argwhere(valid_moves):
                    action_temp = (move[0], move[1])
                    child_state = env_copy.board.copy()
                    child_state[move[0], move[1]] = env_copy.current_player
                    child = MCTSNode(child_state, -env_copy.current_player, parent=node)
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
            while node is not None:
                node.visit_count += 1
                if node.result is None:
                    node.total_value += np.clip(value, -1, 1)
                else:
                    node.total_value += value
                if node is self.root:
                    node = None
                else:
                    node = node.parent
                value = -value
        
        # 修改后的概率计算部分
        visit_counts = np.array([child.visit_count for child in self.root.children])
        actions = [child.action for child in self.root.children]
        if len(actions) == 0:
            print("No valid moves available")
            return None, None, None, None
        # 应用温度参数到概率分布
        if training:
            tau = self.temperature
        else:
            tau = 0.001  # 评估时接近贪婪
            
        probs = self._apply_temperature(visit_counts, tau)
        
        # 根据模式选择动作
        if training:
            # 训练时按概率分布采样
            selected_idx = np.random.choice(len(probs), p=probs)
            selected_action = actions[selected_idx]
        else:
            # 评估时选择访问次数最多的动作
            selected_idx = np.argmax(visit_counts)
            selected_action = actions[selected_idx]
        
        # 构建完整概率图（用于训练数据）
        action_probs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for action, prob in zip(actions, probs):
            action_probs[action] = prob
        
        '''value_pred_list = [child.total_value / child.visit_count if child.visit_count > 0 else 0 for child in self.root.children]
        for i, child in enumerate(self.root.children):
            u = self.c_puct * child.prior * np.sqrt(self.root.visit_count) / (1 + child.visit_count)
            print(f"{i}, action: {child.action}, value_pred: {value_pred_list[i]}, u: {u}, q + u: {value_pred_list[i] + u}, visit_count: {child.visit_count}, prior: {child.prior}, result: {child.result}")
        print("selected_idx: ", selected_idx)'''
        
        value_pred = self.root.children[selected_idx].total_value / self.root.children[selected_idx].visit_count if self.root.children[selected_idx].visit_count > 0 else 0

        # 根节点迁移到选择的子节点
        self.root = self.root.children[selected_idx]
        #self.root.parent = None
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

# 训练流程
class AlphaZeroTrainer:
    def __init__(self, modelFileName=None, cache_file='cache.pkl', config_file='config.ini'):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(self.script_dir, "model")
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
        self.buffer_temp = []
        self.batch_data_count = 0
        self.cache_file = os.path.join(self.script_dir, cache_file)
        self.config_file = os.path.join(self.script_dir, config_file)
        self.win_history = []
        self.lose_history = []
        self.draw_history = []
        self.temperature_decay = (temperature - temperature_end) / (Max_step - temperature_decay_start)  # 计算温度衰减率
        self._write_learning_rate_to_config(learning_rate)
    
    def _write_learning_rate_to_config(self, lr):
        config = configparser.ConfigParser()
        config['TRAINING'] = {'learning_rate': str(lr)}
        with open(self.config_file, 'w') as configfile:
            config.write(configfile)
    
    def _read_learning_rate_from_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        return float(config['TRAINING']['learning_rate'])
    
    def _play_single_game(self, global_model, bExit, result_queue):
        """ 运行一局自我对弈 """
        game_data = []
        steps = 0
        env = GomokuEnv()
        mcts = MCTS(model=global_model)
        game_data_TackBack_index = 0 # 回退时记录当前游戏数据索引
        actions_TackBack = {} # 回退时记录动作 key:步数，value:动作列表
        steps_TakeBack = -1 # 回退时记录当前步数
        temperature_history = [] # 记录温度变化
        action_history_TackBack = [] # 回退时记录动作历史
        while True:
            while not env.done:
                if bExit.value:
                    return

                action, action_probs, value_pred, result = mcts.search(env, training=(steps_TakeBack != steps), simulations=MCTS_simulations if steps_TakeBack != steps else 2000)
                if result is not None and result == -1 and game_data_TackBack_index == 0: # 如果预测到会输，则标记回退点
                    steps_TakeBack = steps - 2
                    action_temp = env.action_history[-2]
                    action_list = actions_TackBack.get(steps_TakeBack, [])
                    if action_temp not in action_list and len(action_list) < 1: # 避免重复添加
                        game_data_TackBack_index = len(game_data) - 8 * 2 # 2步，每步数据增强所以有8个数据
                        state_TakeBack = env.board.copy()
                        current_player_TakeBack = env.current_player
                        action_list.append(action_temp)
                        actions_TackBack[steps_TakeBack] = action_list
                        for move in env.action_history[-2:]: # 回退时将最后两步棋子置为0
                            state_TakeBack[move[0], move[1]] = 0
                        action_history_TackBack = env.action_history[:-2]
                    else:
                        steps_TakeBack = -1
                        game_data_TackBack_index = 0
                        if action_temp in action_list:
                            env.winner = winner
                            break

                    
                state = env.board.copy()
                states_aug, policies_aug = augment_data(state, action_probs)

                for s, p in zip(states_aug, policies_aug):
                    game_data.append((s, env.current_player, p))

                env.step(action)
                steps += 1
                temperature_history.append(mcts.temperature)
                if steps > temperature_decay_start:
                    mcts.temperature -= self.temperature_decay
                    mcts.temperature = max(mcts.temperature, temperature_end)

            winner = env.winner
            # 将每个样本单独放入队列
            for s, player, p in game_data[game_data_TackBack_index:]: # 退点之前的数据还不确定胜负，不放入队列
                state_tensor = MCTS.preprocess_state(s, player, player == env.first_player, device="cpu")
                policy_target = torch.FloatTensor(p)
                value_target = torch.FloatTensor([1 if winner == player else 0 if winner == 0 else -1])
                result_queue.put( (state_tensor, policy_target, value_target) )
            if game_data_TackBack_index > 0:
                env.reset()
                env.action_history = action_history_TackBack
                env.board = state_TakeBack
                env.current_player = current_player_TakeBack
                valid_moves = env.get_valid_moves()
                valid_action_num = np.sum(valid_moves)
                if valid_action_num == len(actions_TackBack.get(steps_TakeBack, [])): # 如果所有合法动作都已被回退，则退出
                    break
                game_data = game_data[:game_data_TackBack_index] # 删除回退点之后的数据
                game_data_TackBack_index = 0
                steps = steps_TakeBack
                temperature_history = temperature_history[:steps]
                mcts.temperature = temperature_history[-1]
            else:
                break

    def self_play(self, num_games=100, bExit=None):
        """ 并行执行 num_games 场自我对弈 """
        num_workers = min(mp.cpu_count(), num_games)
        result_queue = mp.Queue()
        processes = []
        result_count = 0
        for _ in range(num_workers):
            p = mp.Process(target=self._play_single_game, 
                args=(self.global_model, bExit, result_queue))
            p.start()
            processes.append(p)

        # 在子进程运行期间持续处理队列
        while any(p.is_alive() for p in processes):
            try:
                # 非阻塞获取数据
                while not result_queue.empty():
                    result = result_queue.get(block=False)
                    self.buffer_temp.append(result)
                    result_count += 1
            except Exception as e:
                if not isinstance(e, queue.Empty):
                    print(f"Error processing queue: {e}")
            time.sleep(0.1)  # 避免CPU过载

        # 确保所有进程结束
        for p in processes:
            p.join()

        # 处理剩余数据
        while not result_queue.empty():
            try:
                result = result_queue.get(block=False)
                self.buffer_temp.append(result)
                result_count += 1
            except queue.Empty:
                break

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
    
    def _evaluate_single_game(self, global_model, bExit, result_queue, best_model=None, current_model_player=None):
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
            p = mp.Process(target=self._evaluate_single_game, args=(self.global_model, bExit, result_queue, self.best_model, current_model_player))
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

    def run(self, iterations=10):
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
        self.model = self.model.to(mcts_device)
        listener.start()
        self.model = self.model.to(device)

        # 测试代码
        self.global_model.load_state_dict(self.model.state_dict())
        self.model = self.model.to(mcts_device)
        win_count, lose_count, draw_count, cost_time = self.evaluate(bExit=bExit)
        self.win_history.append(win_count)
        self.lose_history.append(lose_count)
        self.draw_history.append(draw_count)
        if win_count == evaluate_games_num:
            self.best_model = AlphaZeroNet().to(mcts_device)
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_model.share_memory()
        self.update_plot()
        self.model = self.model.to(device)
        print(f"Initial win_count: {win_count}, lose_count: {lose_count}, draw_count: {draw_count}, Cost Time: {cost_time:.2f}")
        # 测试代码结束      

        for i in range(iterations):
            if bExit.value:
                print("[INFO] ESC detected. Stopping training loop...")
                break
            print(f"Iteration {i+1}/{iterations}")
            self.model = self.model.to(mcts_device)
            self.self_play(num_games=num_games_per_iter, bExit=bExit)

            # 读取配置文件中的学习率
            lr = self._read_learning_rate_from_config()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.model = self.model.to(device)
            random.shuffle(self.buffer_temp)
            while len(self.buffer_temp) > 0:
                result = self.buffer_temp.pop(0)
                self.buffer.append(result)
                self.batch_data_count += 1
                if self.batch_data_count >= batch_size and len(self.buffer) >= batch_size:
                    self.batch_data_count = 0
                    epochs = len(self.buffer) // batch_size
                    epochs = np.clip(num_epochs, 1, num_epochs)
                    self.train(batch_size=batch_size, epochs=epochs)
            self.global_model.load_state_dict(self.model.state_dict())  # 更新全局模型
            
            # 每5次迭代评估一次
            if (i+1) % 5 == 0:
                checkpoint = False
                self.model = self.model.to(mcts_device)
                win_count, lose_count, draw_count, cost_time = self.evaluate(bExit=bExit, num_games=evaluate_games_num)
                if self.best_model is None:
                    if win_count == evaluate_games_num:
                        self.best_model = AlphaZeroNet().to(mcts_device)
                        self.best_model.load_state_dict(self.model.state_dict())
                        self.best_model.share_memory()
                        checkpoint = True
                        print(f"Best model updated at iteration {i+1}")
                elif self.best_model is not None and win_count > lose_count:
                    self.best_model.load_state_dict(self.model.state_dict())
                    checkpoint = True
                    print(f"Best model updated at iteration {i+1}")
                self.win_history.append(win_count)
                self.lose_history.append(lose_count)
                self.draw_history.append(draw_count)
                print(f"Iteration {i+1}, win_count: {win_count}, lose_count: {lose_count}, draw_count: {draw_count}, Cost Time: {cost_time:.2f}")
                self.update_plot()
            
                if checkpoint:
                    # 保存模型检查点
                    # 检查文件夹是否存在，不存在则创建
                    os.makedirs(self.save_path, exist_ok=True)
                    filePath = os.path.join(self.save_path, f"az_model_{i+1}.pth")
                    torch.save(self.model.state_dict(), filePath)
        
        bStopProcess.value = True  # 停止监听进程
        if self.best_model is not None:
            torch.save(self.best_model.state_dict(), os.path.join(self.save_path, "az_model_best.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "az_model_final.pth"))
        self.save_cache()
        listener.join() # 等待监听线程结束

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
    
    def save_cache(self):
        with open(self.cache_file, 'wb') as file:
            pickle.dump(self.buffer, file)
        print("缓存已保存到硬盘")

    def load_cache(self):
        try:
            with open(self.cache_file, 'rb') as file:
                self.buffer = pickle.load(file)
            print("缓存已从硬盘加载")
        except FileNotFoundError:
            print("未找到缓存文件，将创建新的缓存")

if __name__ == "__main__":
    print(f"Using device: {device}, mcts_device: {mcts_device}, cpu_cores: {mp.cpu_count()}")
    trainer = AlphaZeroTrainer(modelFileName=None)
    trainer.load_cache()
    trainer.run(iterations=total_iterations)