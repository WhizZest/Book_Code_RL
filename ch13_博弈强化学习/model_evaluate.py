from AlphaZero import GomokuEnv, MCTS_Pure, MCTS, AlphaZeroNet, BOARD_SIZE, mcts_device, MCTS_simulations
import keyboard
import torch.multiprocessing as mp
import os
import time
import random
import torch
import threading
import numpy as np
import matplotlib.pyplot as plt

bExit = mp.Value('b', False)  # 使用共享变量
bStopProcess = mp.Value('b', False)  # 使用共享变量
evaluate_games_num = 20

def evaluate_single_game(play1, bExit, result_queue, play2=None, current_model_player=None):
    """ 运行一局评估对局 """
    env = GomokuEnv()
    mcts = MCTS(model=play1)
    if play2 is not None:
        mcts_best = MCTS(model=play2)
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
            if play2 is not None:
                action, _, value_pred, result = mcts_best.search(env, training=False, simulations=MCTS_simulations)
            else:
                action = mcts_pure.search(env, simulations=MCTS_simulations)

        env.step(action)

    result_queue.put(1 if env.winner == current_model_player else 0 if env.winner == 0 else -1)

class AlphaZeroEvaluate:
    def __init__(self, modelFileName1, modelFileName2):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(self.script_dir, "model")
        os.makedirs(self.save_path, exist_ok=True)
        self.model1 = AlphaZeroNet().to(mcts_device)  # 初始化时转移到设备
        filePath = os.path.join(self.script_dir, modelFileName1)
        self.model1.load_state_dict(torch.load(filePath, map_location=mcts_device, weights_only=True))
        self.model1.share_memory()  # 共享模型
        print(f"加载模型1成功:{modelFileName1}")
        self.model2 = AlphaZeroNet().to(mcts_device)  # 初始化时转移到设备
        filePath = os.path.join(self.script_dir, modelFileName2)
        self.model2.load_state_dict(torch.load(filePath, map_location=mcts_device, weights_only=True))
        self.model2.share_memory()  # 共享模型
        print(f"加载模型2成功:{modelFileName2}")
        self.model1.eval()
        self.model2.eval()
        self.win_history = 0
        self.lose_history = 0
        self.draw_history = 0

    def evaluate(self, num_games=20, bExit=None):
        """ 并行评估 """
        if bExit.value:
            print(f"Evaluation process exiting due to ESC...")
            return 0, 0, 0, 0
        num_workers = min(mp.cpu_count(), num_games)

        # 创建一个队列来存储子进程的返回值
        result_queue = mp.Queue()

        # 创建并启动子进程
        start_t = time.time()
        processes = []
        current_model_player = 1
        for _ in range(num_workers):
            p = mp.Process(target=evaluate_single_game, args=(self.model1, bExit, result_queue, self.model2, current_model_player))
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

    def run(self):
        # 启动键盘监听线程
        listener = threading.Thread(target=self._esc_listener, args=(bExit, bStopProcess,))
        listener.start()
        num_games = 10 # 每次评估的局数
        iterations = evaluate_games_num // num_games
        num_games_per_iteration = [num_games] * iterations
        if evaluate_games_num % num_games != 0:
            num_games_per_iteration.append(evaluate_games_num % num_games)
            iterations += 1

        plt.ion()  # 开启交互模式
        fig, ax = plt.subplots()
        for i in range(len(num_games_per_iteration)):
            # 测试代码
            win_count, lose_count, draw_count, cost_time = self.evaluate(bExit=bExit, num_games=num_games_per_iteration[i])
            self.win_history += win_count
            self.lose_history += lose_count
            self.draw_history += draw_count
            print(f"win_count: {self.win_history}, lose_count: {self.lose_history}, draw_count: {self.draw_history}, Cost Time: {cost_time:.2f}")
            # 测试代码结束
            # 清除当前图像
            ax.clear()

            # 绘制直方图
            categories = ["Win", "Lose", "Draw"]
            values = [self.win_history, self.lose_history, self.draw_history]
            ax.bar(categories, values, color=["green", "red", "blue"])

            # 设置标题和标签
            ax.set_title("Win/Lose/Draw History")
            ax.set_ylabel("Count")

            # 更新图像
            plt.pause(0.1)  # 暂停一小会，确保图像更新
        
        bStopProcess.value = True  # 停止监听进程
        listener.join()  # 等待监听进程结束
        plt.ioff()  # 关闭交互模式
        plt.show()  # 结束后显示最终图像

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

if __name__ == "__main__":
    print(f"Using mcts_device: {mcts_device}, cpu_cores: {mp.cpu_count()}")
    evaluator = AlphaZeroEvaluate(modelFileName1="model/az_model_55.pth", modelFileName2="model/az_model_296.pth")
    evaluator.run()