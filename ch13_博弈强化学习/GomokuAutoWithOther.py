import cv2
import numpy as np
import pyautogui
import time
from AlphaZero import GomokuEnv, MCTS_Pure, MCTS, AlphaZeroNet, BOARD_SIZE, device, mcts_device
import os
import torch
import matplotlib.pyplot as plt
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pickle

class GomokuAutoWithOtherGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gomoku Auto with Other")
        master.attributes("-topmost", True)  # 确保窗口始终在最前面
        master.geometry("500x650")
        
        self.player_dict = {1: 1}  # 玩家对应的棋子，key：1表示黑子，-1表示白子，value：1表示先手，-1表示后手
        self.player_dict[-1] = -self.player_dict[1]
        self.current_model_player = self.player_dict[-1]  # 当前模型玩家, key：1表示黑子，-1表示白子
        self.MCTS_simulations = 2000  # MCTS模拟次数
        self.think_thread = None  # 思考线程
        self.value_history = [] # 胜率记录

        self.buffer = []  # 缓冲区

        # 默认模型路径
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_model_path = os.path.join(self.script_dir, "model\\az_model_final.pth")

        self.create_widgets()

        # 绑定关闭事件
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def save_buffer(self):
        """ 保存缓冲区数据 """
        if len(self.buffer) > 0:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            filename = f"buffer_{timestamp}.pkl"
            filepath = os.path.join(self.script_dir, "eval_buffer", filename)
            folder = os.path.join(self.script_dir, "eval_buffer")
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(filepath, "wb") as f:
                pickle.dump(self.buffer, f)
            print(f"缓冲区数据已保存至 {filepath}，共 {len(self.buffer)} 条数据。")

    def on_closing(self):
        self.stop_game()
        self.save_buffer()
        self.master.destroy()
    
    def create_widgets(self):
        # 控制面板：开始、停止
        control_frame = ttk.Frame(self.master)
        control_frame.pack(pady=10)

        self.start_button = ttk.Button(control_frame, text="开始", command=self.start_game)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="停止", command=self.stop_game)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 参数设置面板：黑子or白子Radiobutton，是否先手Checkbutton
        settings_frame = ttk.Frame(self.master)
        settings_frame.pack(pady=10)
        self.piece_color_var = tk.IntVar(value=-1)  # 默认白子
        self.first_move_var = tk.BooleanVar(value=True)  # 默认先手

        self.black_piece_radio = ttk.Radiobutton(settings_frame, text="黑子", variable=self.piece_color_var, value=1)
        self.black_piece_radio.pack(side=tk.LEFT, padx=5)
        self.white_piece_radio = ttk.Radiobutton(settings_frame, text="白子", variable=self.piece_color_var, value=-1)
        self.white_piece_radio.pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(settings_frame, text="先手", variable=self.first_move_var).pack(side=tk.LEFT, padx=5)

        # MCTS模拟次数：
        mcts_frame = ttk.Frame(self.master)
        mcts_frame.pack(pady=10)
        # Radiobutton选项：400，800，1600，2000，2400，2800，自定义
        self.mcts_simulations_var = tk.IntVar(value=self.MCTS_simulations)
        ttk.Radiobutton(mcts_frame, text="400", variable=self.mcts_simulations_var, value=400).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mcts_frame, text="800", variable=self.mcts_simulations_var, value=800).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mcts_frame, text="1600", variable=self.mcts_simulations_var, value=1600).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mcts_frame, text="2000", variable=self.mcts_simulations_var, value=2000).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mcts_frame, text="2400", variable=self.mcts_simulations_var, value=2400).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mcts_frame, text="2800", variable=self.mcts_simulations_var, value=2800).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mcts_frame, text="自定义", variable=self.mcts_simulations_var, value=0).pack(side=tk.LEFT, padx=5)

        CustomMCTS_frame = ttk.Frame(self.master)
        CustomMCTS_frame.pack(pady=10)
        ttk.Label(CustomMCTS_frame, text="MCTS模拟次数:").pack(side=tk.LEFT)
        self.mcts_simulations_entry = ttk.Entry(CustomMCTS_frame)
        self.mcts_simulations_entry.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.mcts_simulations_entry.insert(0, str(self.MCTS_simulations))
        
        # 模型设置面板：选择模型文件
        model_frame = ttk.Frame(self.master)
        model_frame.pack(pady=10)

        ttk.Label(model_frame, text="模型路径:").pack(side=tk.LEFT)
        self.model_path_entry = ttk.Entry(model_frame, width=40)
        self.model_path_entry.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.model_path_entry.insert(0, self.default_model_path)
        ttk.Button(model_frame, text="浏览", command=self.browse_model).pack(side=tk.LEFT)

        # 中文显示问题
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        # 胜率变化曲线
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_title("胜率变化曲线")
        self.ax.set_xlabel("步数")
        self.ax.set_ylabel("胜率估计")
        self.line, = self.ax.plot([], [], label="胜率估计")
        #self.ax.legend()
        self.ax.grid(True)

        self.canvas_plot = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas_plot.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH)

    def update_chart(self):
        self.line.set_data(range(len(self.value_history)), self.value_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas_plot.draw()

    def browse_model(self):
        filepath = filedialog.askopenfilename(
            initialdir=os.path.join(self.script_dir, "model"),
            title="选择模型文件",
            filetypes=(("PyTorch模型", "*.pth"), ("所有文件", "*.*"))
        )
        if filepath:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, filepath)

    def detect_pieces(self, board_img, intersections, show_result=False, timeout=2000):
        """ 遍历交叉点附近局部区域的颜色值来识别棋子 """
        board = np.zeros((15, 15), dtype=int)
        
        radius = 10  # 局部区域的大小
        for (i, j), (x, y) in intersections.items():
            x, y = int(x), int(y)  # 确保索引是整数
            region = board_img[max(y-radius, 0):min(y+radius, board_img.shape[0]), 
                            max(x-radius, 0):min(x+radius, board_img.shape[1]), :]
            if region.size == 0:
                continue
            
            avg_color = np.mean(region, axis=(0, 1))  # 计算局部区域的平均颜色
            b, g, r = avg_color
            
            if r < 100 and g < 100 and b < 100:  # 接近黑色
                board[i, j] = self.player_dict[1]  # 黑子
                cv2.circle(board_img, (x, y), radius, (0, 0, 255), -1)
            elif r < 100 and g > 150 and b > 150:  # 接近白色
                board[i, j] = self.player_dict[-1]  # 白子
                cv2.circle(board_img, (x, y), radius, (255, 0, 0), -1)
        
        if show_result:
            cv2.imshow("Detected Pieces", board_img)
            cv2.waitKey(timeout)
            cv2.destroyAllWindows()
        return board
    
    def evaluate_single_game(self, ai_play, actionMapToCoords, current_model_player, MCTS_simulations):
        """ 运行一局评估对局 """
        env = GomokuEnv()
        mcts = MCTS(model=ai_play)
        think_thread = threading.Thread(target=mcts.search, args=(env, 20000, False, False, True))
        think_thread.start()
        wait_time = 0.5
        wait_count = 0
        self.value_history = []
        self.update_chart()
        steps_TakeBack = -1
        save_buffer_flag = False

        screen = capture_board(region=None)  # 截取全屏图像
        board_matrix_current = self.detect_pieces(screen.copy(), actionMapToCoords, show_result=False, timeout=0)
        if env.current_player == current_model_player and not np.array_equal(env.board, board_matrix_current):
            #raise ValueError("当前玩家为先手，但当前棋盘不为空，请检查先后手是否正确")
            print("当前玩家为先手，但当前棋盘不为空，请检查先后手是否正确，将自动切换玩家为后手")
            current_model_player = -current_model_player

        while not env.done:
            if self.stop:
                if think_thread is not None and think_thread.is_alive():
                    mcts.stop = True
                    think_thread.join()
                break
            if env.current_player == current_model_player:
                time_cost = time.time()
                mcts.stop = True
                think_thread.join()
                action, _, value_pred, result = mcts.search(env, training=False, simulations=MCTS_simulations)
                if result is not None and result == -1 and steps_TakeBack < 0 and len(env.action_history) >= 2:
                    steps_TakeBack = len(env.action_history) - 2
                self.value_history.append(np.clip(value_pred, -1, 1)) # 记录胜率
                if value_pred < -0.8:
                    save_buffer_flag = True
                x, y = actionMapToCoords[action]
                pyautogui.moveTo(x, y)
                pyautogui.click()
                time_cost = time.time() - time_cost
                self.update_chart()  # 更新图表
                print(f"回合{len(env.action_history)+1}：AI 选择动作 {action}，预测胜率 {np.clip(value_pred, -1, 1)}, 预测结果 {result}, 耗时 {time_cost:.2f} 秒")
            else:
                if think_thread is not None and not think_thread.is_alive():
                    think_thread = threading.Thread(target=mcts.search, args=(env, 20000, False, False, True))
                    think_thread.start()
                time.sleep(wait_time)
                action = None
                wait_count = 0
                while action is None:
                    screen = capture_board(region=None)  # 截取全屏图像
                    board_matrix_current = self.detect_pieces(screen.copy(), actionMapToCoords, show_result=False, timeout=0)
                    action = detect_board_change(board_matrix_current, env.board)
                    if action is None:
                        #wait_time += 0.1
                        time.sleep(wait_time)
                        wait_count += 1
                        if wait_count > 20:
                            # 保存screen图片，用当前时间命名，保存在error_images文件夹中，如果不存在则创建
                            now = datetime.now()
                            timestamp = now.strftime("%Y%m%d_%H%M%S")
                            if not os.path.exists("error_images"):
                                os.makedirs("error_images")
                            cv2.imwrite(f"error_images/error_{timestamp}.png", screen)
                    if self.stop:
                        if think_thread is not None and think_thread.is_alive():
                            mcts.stop = True
                            think_thread.join()
                        break

            if action is not None:
                reward = env.step(action)
            else:
                reward = None
            if reward is None:
                print(f"当前玩家：{env.current_player}，无效动作 {action}")
                if result is not None:
                    env.winner = current_model_player if result == 1 else -current_model_player if result == -1 else 0
                break
        if len(env.action_history) > 0 and (env.winner == -current_model_player or save_buffer_flag):
            self.buffer.append((env.action_history, steps_TakeBack))
            print(f"当前局面存入缓冲区")
        return 1 if env.winner == current_model_player else 0 if env.winner == 0 else -1

    def getPieceColorByVision(self):
        # 读取给定图片
        template = cv2.imread('template.png', 0)

        # 获取屏幕截图
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # 使用模板匹配方法
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

        # 设定匹配阈值
        threshold = 0.8

        # 找到匹配的位置
        loc = np.where(result >= threshold)
        if len(loc) == 2:
            if loc[1][0] < 1300:
                return -1 # 白棋
            else:
                return 1 # 黑棋
        else:
            return None
        
    def stop_game(self):
        self.stop = True
        time.sleep(0.5)
    def start_game(self):
        self.stop_game()
        piece_color = self.getPieceColorByVision()
        if piece_color is None:
            piece_color = self.piece_color_var.get()
        else:
            if piece_color == 1:
                self.piece_color_var.set("黑棋")
                #self.black_piece_radio.select()
            else:
                self.piece_color_var.set("白棋")
                #self.white_piece_radio.select()
            # 禁用单选按钮
            self.black_piece_radio.state(['disabled'])
            self.white_piece_radio.state(['disabled'])
        if self.first_move_var.get():
            self.player_dict[piece_color] = 1
        else:
            self.player_dict[piece_color] = -1
        self.player_dict[-piece_color] = -self.player_dict[piece_color]
        self.current_model_player = self.player_dict[piece_color]
        if int(self.mcts_simulations_var.get()) <= 0:
            self.MCTS_simulations = int(self.mcts_simulations_entry.get())
        else:
            self.MCTS_simulations = int(self.mcts_simulations_var.get())
        self.stop = False
        
        def game_thread():
            screen = capture_board(region=None)  # 截取全屏图像
            # 设置棋盘左上角和右下角的坐标，以及棋盘大小
            actionMapToCoords = detect_grid_intersections(screen.copy(), top_left=(1076, 510), bottom_right=(1566,1000), board_size=BOARD_SIZE, show_result=False, timeout=0)
            if actionMapToCoords is not None:
                #self.detect_pieces(screen.copy(), actionMapToCoords, show_result=True, timeout=0)
                # 获取本地文件夹路径，与"model"文件夹拼接,得到模型文件路径,与当前模型文件名拼接，得到完整模型文件路径
                #local_folder = os.path.dirname(os.path.abspath(__file__))
                #model_file_path = os.path.join(local_folder, "model", "az_model_260.pth")
                ai_play = AlphaZeroNet().to(mcts_device)
                ai_play.load_state_dict(torch.load(self.model_path_entry.get(), map_location=mcts_device, weights_only=True))
                ai_play.eval()
                result = self.evaluate_single_game(ai_play, actionMapToCoords, self.current_model_player, self.MCTS_simulations)
                print(f"评估结果: {result}")
            else:
                print("未能检测到棋盘线")
            # 启用单选按钮
            self.black_piece_radio.state(['!disabled'])
            self.white_piece_radio.state(['!disabled'])
        
        self.game_thread = threading.Thread(target=game_thread)
        self.game_thread.start()
def select_region():
    """ 让用户手动框选棋盘区域 """
    screen = pyautogui.screenshot()
    frame = np.array(screen)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    region = cv2.selectROI("Select Region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region")
    return region

def capture_board(region):
    """ 截取指定区域的棋盘图像 """
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 读取为彩色图像
    return frame

def preprocess_image(image):
    """ 进行边缘检测 """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def merge_lines(lines, min_dist=10):
    """ 合并相近的线条 """
    lines = sorted(lines)
    merged = []
    for line in lines:
        if not merged or abs(merged[-1] - line) > min_dist:
            merged.append(line)
    return merged

def detect_grid_intersections(board_img, top_left, bottom_right, board_size=15, show_result=False, timeout=2000):
    """ 计算棋盘交叉点坐标，并可视化 """
    intersections = {}
    
    x_step = (bottom_right[0] - top_left[0]) / (board_size - 1)
    y_step = (bottom_right[1] - top_left[1]) / (board_size - 1)
    
    for i in range(board_size):
        for j in range(board_size):
            x = int(top_left[0] + j * x_step)
            y = int(top_left[1] + i * y_step)
            intersections[(i, j)] = (x, y)
            cv2.circle(board_img, (x, y), 3, (0, 255, 0), -1)
    if show_result:
        cv2.imshow("Detected Grid", board_img)
        cv2.waitKey(timeout)
        cv2.destroyAllWindows()
    
    return intersections

def detect_board_change(board_matrix_current, board_matrix_previous):
    """ 检测board_matrix_current相对于board_matrix_previous的变化位置 """
    change_list = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board_matrix_previous[i, j] != board_matrix_current[i, j]:
                change_list.append((i, j))
    '''if len(change_list) != 1:
        print(f"检测到{len(change_list)}个变化位置: {change_list}")'''
        #raise ValueError(f"检测到{len(change_list)}个变化位置: {change_list}")
    return change_list[0] if len(change_list) == 1 else None

if __name__ == "__main__":
    root = tk.Tk()
    gui = GomokuAutoWithOtherGUI(root)
    #gui.start_game()
    root.mainloop()