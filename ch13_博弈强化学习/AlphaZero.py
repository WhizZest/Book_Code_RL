import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import keyboard
import torch.multiprocessing as mp
import time
import queue
import os
import sys
import pickle

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mcts_device = "cpu"

# 游戏环境配置
BOARD_SIZE = 9  # 使用9x9棋盘加速训练
WIN_STREAK = 5
Max_step = BOARD_SIZE * BOARD_SIZE

# 训练参数
MCTS_simulations = 400 # 每次选择动作时进行的蒙特卡洛树搜索模拟次数
batch_size = 1024  # 每次训练的批量大小
num_games_per_iter = 5  # 每个迭代中进行的游戏数量
num_epochs = 10  # 训练的轮数
learning_rate = 0.001  # 学习率
buffer_size = 50000  # 经验回放缓冲区大小
temperature = 1.0 # 温度参数
temperature_end = 0.01 # 温度参数的最小值
temperature_decay_start = 30
total_iterations = 500 # 迭代次数
dirichlet_alpha = 0.3 # 控制噪声集中程度（值越小噪声越稀疏）
dirichlet_epsilon=0.25  # 原策略与噪声的混合比例
c_puct = 5 # 控制探索与利用的平衡

bExit = mp.Value('b', False)  # 使用共享变量
bStopProcess = mp.Value('b', False)  # 使用共享变量

class GomokuEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.first_player = 1
        self.current_player = self.first_player
        self.done = False
        self.winner = 0
        self.win_paths = []

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
    
    def select_child(self, c_puct):
        total_visits = sum(child.visit_count for child in self.children)
        
        best_score = -np.inf
        best_child = None
        
        random.shuffle(self.children) # 随机打乱子节点顺序
        for child in self.children:
            env_copy = GomokuEnv()
            env_copy.board = self.state.copy()
            env_copy.current_player = self.player
            env_copy.step(child.action)
            if env_copy.done:
                return child
            q = child.total_value / child.visit_count if child.visit_count else 0
            u = c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

class MCTS_Pure:
    def __init__(self, simulations=100):
        self.simulations = simulations
        self.c_puct = c_puct  # 探索系数

    def search(self, env):
        root = MCTSNode(env.board.copy(), env.current_player)

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
                    return action
        for i in range(self.simulations):
            node = root
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
                value = 1 if env_copy.winner == node.player else -1
            
            # 回溯更新
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                node = node.parent
                value = -value
        
        # 修改后的概率计算部分
        visit_counts = np.array([child.visit_count for child in root.children])
        actions = [child.action for child in root.children]
        
        # 评估时选择访问次数最多的动作
        selected_idx = np.argmax(visit_counts)
        selected_action = actions[selected_idx]
        return selected_action

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class MCTS:
    def __init__(self, model, simulations=100):
        self.model = AlphaZeroNet().to(mcts_device)
        self.model.load_state_dict(model.state_dict())
        self.simulations = simulations
        self.c_puct = c_puct  # 探索系数
        self.temperature = temperature  # 添加温度参数
        self.dirichlet_alpha = dirichlet_alpha  # Dirichlet分布参数α
        self.dirichlet_epsilon = dirichlet_epsilon  # 噪声混合比例

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

    def search(self, env, training=True):
        if env.done:
            valid_moves = env.get_valid_moves()
            sum_value = valid_moves.sum()
            if sum_value == 0:
                return None, None
            else:
                action_probs = valid_moves / sum_value
            return None, action_probs.flatten()
        
        root = MCTSNode(env.board.copy(), env.current_player)
        
        # 仅在训练模式且为根节点时准备噪声
        if training and root.parent is None:
            noise = self._prepare_dirichlet_noise(root)
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
                    return action, action_probs.flatten()
        for _ in range(self.simulations):
            node = root
            env_copy = GomokuEnv()
            env_copy.board = node.state.copy()
            env_copy.current_player = node.player
            
            # 选择
            while node.children:
                node = node.select_child(self.c_puct)
                env_copy.step(node.action)
            
            # 扩展
            if not env_copy.done:
                with torch.no_grad():
                    state_tensor = self.preprocess_state(env_copy.board, env_copy.current_player, env_copy.current_player == env_copy.first_player, device=mcts_device)
                    policy, value = self.model(state_tensor)
                
                valid_moves = env_copy.get_valid_moves().flatten()
                policy = policy.squeeze().cpu().numpy() * valid_moves
                policy /= policy.sum()
                # 仅在根节点且训练模式时混合噪声
                if node is root and training and noise is not None: 
                    policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
                
                for move in np.argwhere(valid_moves.reshape(BOARD_SIZE, BOARD_SIZE)):
                    child_state = env_copy.board.copy()
                    child_state[move[0], move[1]] = env_copy.current_player
                    child = MCTSNode(child_state, -env_copy.current_player, parent=node)
                    child.prior = policy[move[0]*BOARD_SIZE + move[1]]
                    child.action = (move[0], move[1])
                    node.children.append(child)
                
                value = value.item()
            else:
                value = 1 if env_copy.winner == node.player else -1
            
            # 回溯更新
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                node = node.parent
                value = -value
        
        # 修改后的概率计算部分
        visit_counts = np.array([child.visit_count for child in root.children])
        actions = [child.action for child in root.children]
        
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
        
        return selected_action, action_probs.flatten()
    
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
    def __init__(self, modelFileName=None, cache_file='cache.pkl'):
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
        self.cache_file = os.path.join(self.script_dir, cache_file)
        self.win_rates = []  # 记录胜率历史
        self.temperature_decay = (temperature - temperature_end) / (Max_step - temperature_decay_start)  # 计算温度衰减率
    
    def _play_single_game(self, global_model, bExit, result_queue):
        """ 运行一局自我对弈 """
        env = GomokuEnv()
        game_data = []
        steps = 0
        mcts = MCTS(model=global_model, simulations=MCTS_simulations)

        while not env.done:
            if bExit.value:
                return []

            action, action_probs = mcts.search(env, training=True)
            state = env.board.copy()
            states_aug, policies_aug = augment_data(state, action_probs)

            for s, p in zip(states_aug, policies_aug):
                game_data.append((s, env.current_player, p))

            env.step(action)
            steps += 1

            if steps > temperature_decay_start:
                mcts.temperature -= self.temperature_decay
                mcts.temperature = max(mcts.temperature, temperature_end)

        winner = env.winner
        # 将每个样本单独放入队列
        for s, player, p in game_data:
            state_tensor = MCTS.preprocess_state(s, player, player == env.first_player, device="cpu")
            policy_target = torch.FloatTensor(p)
            value_target = torch.FloatTensor([1 if winner == player else -1])
            result_queue.put( (state_tensor, policy_target, value_target) )

    def self_play(self, num_games=100, bExit=None):
        """ 并行执行 num_games 场自我对弈 """
        num_workers = min(mp.cpu_count(), num_games)
        result_queue = mp.Queue()
        processes = []
        result_count = 0
        for _ in range(num_workers):
            p = mp.Process(target=self._play_single_game, args=(self.global_model, bExit, result_queue))
            p.start()
            processes.append(p)

        # 在子进程运行期间持续处理队列
        while any(p.is_alive() for p in processes):
            try:
                # 非阻塞获取数据
                while not result_queue.empty():
                    result = result_queue.get(block=False)
                    self.buffer.append(result)
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
                self.buffer.append(result)
                result_count += 1
            except queue.Empty:
                break

        print(f"Collected {result_count} samples, avereage steps: {result_count / num_games / 8}")
    
    def train(self, batch_size=32, epochs=10):
        if len(self.buffer) < batch_size:
            return
        
        batch = random.sample(self.buffer, batch_size)
        states, policy_targets, value_targets = zip(*batch)
        #states, policy_targets, value_targets = zip(*self.buffer)
        # 将数据转移到设备
        states = torch.cat([s.to(device) for s in states])
        policy_targets = torch.stack(policy_targets).to(device)
        value_targets = torch.cat(value_targets).to(device)
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            policy_pred, value_pred = self.model(states)
            
            # 添加维度验证
            assert policy_pred.shape == policy_targets.shape, \
                f"Pred shape {policy_pred.shape} != Target shape {policy_targets.shape}"
            
            policy_loss = -torch.sum(policy_targets * torch.log(policy_pred + 1e-10))
            value_loss = torch.mean((value_pred.squeeze() - value_targets)**2)
            loss = policy_loss + value_loss
            
            loss.backward()
            self.optimizer.step()
        
        self.global_model.load_state_dict(self.model.state_dict())  # 更新全局模型
        print(f"Training completed, loss: {loss.item()}")
    
    def _evaluate_single_game(self, global_model, bExit, result_queue, best_model=None, current_model_player=None):
        """ 运行一局评估对局 """
        env = GomokuEnv()
        mcts = MCTS(model=global_model, simulations=MCTS_simulations)
        if best_model is not None:
            mcts_best = MCTS(model=best_model, simulations=MCTS_simulations)
        else:
            mcts_pure = MCTS_Pure(simulations=MCTS_simulations)
        
        if current_model_player is None:
            # 随机待评估模型的玩家
            current_model_player = random.choice([1, -1])

        while not env.done:
            if bExit.value:
                print(f"Evaluation process {mp.current_process().pid} exiting due to ESC...")
                return 0

            if env.current_player == current_model_player:
                action, _ = mcts.search(env, training=False)
            else:
                '''valid_moves = np.argwhere(env.get_valid_moves())
                action = tuple(valid_moves[np.random.choice(len(valid_moves))])'''
                if best_model is not None:
                    action, _ = mcts_best.search(env, training=False)
                else:
                    action = mcts_pure.search(env)

            env.step(action)

        result_queue.put(1 if env.winner == current_model_player else 0)

    def evaluate(self, num_games=20, bExit=None):
        """ 并行评估 """
        if bExit.value:
            print(f"Evaluation process exiting due to ESC...")
            return 0
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

        return sum(results) / num_games, time.time() - start_t
    
    def update_plot(self):
        """动态更新胜率曲线"""
        self.line.set_xdata(np.arange(len(self.win_rates)))
        self.line.set_ydata(self.win_rates)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.05)  # 短暂暂停让图表更新

    def run(self, iterations=10):
        # 初始化动态绘图
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.ax.set_title('Training Progress')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Win Rate')
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], 'b-', label='Win Rate')

        # 启动键盘监听进程
        listener = mp.Process(target=self._esc_listener, args=(bExit, bStopProcess,))
        self.model = self.model.to(mcts_device)
        listener.start()
        self.model = self.model.to(device)

        # 测试代码
        self.global_model.load_state_dict(self.model.state_dict())
        self.model = self.model.to(mcts_device)
        win_rate, cost_time = self.evaluate(bExit=bExit)
        self.win_rates.append(win_rate)
        if win_rate >= 0.99:
            self.best_model = AlphaZeroNet().to(mcts_device)
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_model.share_memory()
        self.update_plot()
        self.model = self.model.to(device)
        print(f"Initial Win Rate: {win_rate:.2f}, Cost Time: {cost_time:.2f}")
        # 测试代码结束      

        for i in range(iterations):
            if bExit.value:
                print("[INFO] ESC detected. Stopping training loop...")
                break
            print(f"Iteration {i+1}/{iterations}")
            self.model = self.model.to(mcts_device)
            self.self_play(num_games=num_games_per_iter, bExit=bExit)
            self.model = self.model.to(device)
            self.train(batch_size=batch_size, epochs=num_epochs)
            
            # 每5次迭代评估一次
            if (i+1) % 5 == 0:
                checkpoint = False
                self.model = self.model.to(mcts_device)
                win_rate, cost_time = self.evaluate(bExit=bExit)
                if self.best_model is None and win_rate >= 0.99:
                    self.best_model = AlphaZeroNet().to(mcts_device)
                    self.best_model.load_state_dict(self.model.state_dict())
                    self.best_model.share_memory()
                    checkpoint = True
                    print(f"Best model updated at iteration {i+1}")
                elif self.best_model is not None and win_rate >= 0.55:
                    self.best_model.load_state_dict(self.model.state_dict())
                    checkpoint = True
                    print(f"Best model updated at iteration {i+1}")
                #self.model = self.model.to(device)
                self.win_rates.append(win_rate)
                print(f"Iteration {i+1}, Win Rate: {win_rate:.2f}, Cost Time: {cost_time:.2f}")
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

        plt.ioff()  # 关闭交互模式
        plt.show()
    
    def _esc_listener(self, bExit, bStopProcess):
        """ 监听 ESC 按键，通知所有进程退出 """
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