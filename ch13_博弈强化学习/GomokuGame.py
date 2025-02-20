import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import torch
import os
import numpy as np
from AlphaZero import GomokuEnv, MCTS_Pure, MCTS, AlphaZeroNet, BOARD_SIZE, device, mcts_device
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GomokuGUI:
    def __init__(self, master):
        self.master = master
        master.title("五子棋对弈测试")
        master.geometry("1200x700")
        self.board_size = 600
        
        # 游戏状态变量
        self.game_env = None
        self.ai_players = {}
        self.current_player = 1
        self.running = False
        self.ai_thread = None
        self.ai_thread_thinking = None
        self.thinking_player = None
        
        # 默认模型路径
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_model_path = os.path.join(self.script_dir, "model\\az_model_final.pth")

        # 中文显示问题
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        # 胜率图表区域
        self.value_history = {1: [], -1: []}  # 存储双方胜率数据

        self.board_layout = False
        
        # 创建界面组件
        self.create_widgets()

        self.reset_game()

        # 绑定关闭事件
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        self.reset_game()
        self.master.destroy()
        
    def create_widgets(self):
        # 控制面板
        control_frame = ttk.LabelFrame(self.master, text="游戏设置")
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        # 模式选择
        self.mode_var = tk.StringVar(value="human_vs_ai")
        mode_frame = ttk.LabelFrame(control_frame, text="对战模式")
        mode_frame.pack(padx=5, pady=5, fill=tk.X)
        
        ttk.Radiobutton(mode_frame, text="人机对弈", variable=self.mode_var, 
                       value="human_vs_ai", command=self.update_ui).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="机机对弈", variable=self.mode_var,
                       value="ai_vs_ai", command=self.update_ui).pack(anchor=tk.W)
        
        # 控制按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="开始", command=self.start_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="重置", command=self.reset_game).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="布局", command=self.board_layout_f).pack(side=tk.LEFT, padx=5)
        #ttk.Button(btn_frame, text="退出", command=self.master.quit).pack(side=tk.LEFT)
        
        # 人机设置
        self.human_frame = ttk.LabelFrame(control_frame, text="人机设置")
        self.human_frame.pack(padx=5, pady=5, fill=tk.X)
        
        self.human_first_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.human_frame, text="人类先手", variable=self.human_first_var).pack(anchor=tk.W)
        
        # 人机设置->AI类型选择
        self.ai_type_var = tk.StringVar(value="pure_mcts")
        ttk.Label(self.human_frame, text="AI类型:").pack(anchor=tk.W)
        ttk.Radiobutton(self.human_frame, text="纯MCTS", variable=self.ai_type_var, 
                       value="pure_mcts").pack(anchor=tk.W)
        ttk.Radiobutton(self.human_frame, text="MCTS+网络", variable=self.ai_type_var,
                       value="mcts_net").pack(anchor=tk.W)
        
        # 人机设置->MCTS模拟次数
        ttk.Label(self.human_frame, text="MCTS模拟次数:").pack(anchor=tk.W)
        self.mcts_sim_entry = ttk.Entry(self.human_frame)
        self.mcts_sim_entry.insert(0, "400")
        self.mcts_sim_entry.pack(anchor=tk.W)

        # 人机设置->模型路径选择
        model_frame = ttk.Frame(self.human_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="模型路径:").pack(side=tk.LEFT)
        self.model_path_entry = ttk.Entry(model_frame)
        self.model_path_entry.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.model_path_entry.insert(0, self.default_model_path)
        ttk.Button(model_frame, text="浏览", command=self.browse_model).pack(side=tk.LEFT)

        # 机机设置
        self.ai_ai_frame = ttk.LabelFrame(control_frame, text="AI设置")
        
        # AI1设置
        self.ai1_frame = ttk.Frame(self.ai_ai_frame)
        self.ai1_type_var = tk.StringVar(value="pure_mcts")
        ttk.Label(self.ai1_frame, text="AI1类型:").pack(anchor=tk.W)
        ttk.Radiobutton(self.ai1_frame, text="纯MCTS", variable=self.ai1_type_var, 
                       value="pure_mcts").pack(anchor=tk.W)
        ttk.Radiobutton(self.ai1_frame, text="MCTS+网络", variable=self.ai1_type_var,
                       value="mcts_net").pack(anchor=tk.W)
        self.ai1_sim_var = tk.StringVar(value="400")
        ttk.Label(self.ai1_frame, text="模拟次数:").pack(anchor=tk.W)
        ttk.Entry(self.ai1_frame, textvariable=self.ai1_sim_var, width=10).pack(anchor=tk.W)
        
        # AI2设置
        self.ai2_frame = ttk.Frame(self.ai_ai_frame)
        self.ai2_type_var = tk.StringVar(value="pure_mcts")
        ttk.Label(self.ai2_frame, text="AI2类型:").pack(anchor=tk.W)
        ttk.Radiobutton(self.ai2_frame, text="纯MCTS", variable=self.ai2_type_var, 
                       value="pure_mcts").pack(anchor=tk.W)
        ttk.Radiobutton(self.ai2_frame, text="MCTS+网络", variable=self.ai2_type_var,
                       value="mcts_net").pack(anchor=tk.W)
        self.ai2_sim_var = tk.StringVar(value="400")
        ttk.Label(self.ai2_frame, text="模拟次数:").pack(anchor=tk.W)
        ttk.Entry(self.ai2_frame, textvariable=self.ai2_sim_var, width=10).pack(anchor=tk.W)

        # 机机设置->模型2路径选择
        model2_frame = ttk.Frame(self.ai_ai_frame)
        model2_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        ttk.Label(model2_frame, text="模型2路径:").pack(side=tk.LEFT)
        self.model2_path_entry = ttk.Entry(model2_frame)
        self.model2_path_entry.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.model2_path_entry.insert(0, self.default_model_path)
        ttk.Button(model2_frame, text="浏览", command=self.browse_model2).pack(side=tk.LEFT)

        # 机机设置->模型1路径选择
        model1_frame = ttk.Frame(self.ai_ai_frame)
        model1_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        ttk.Label(model1_frame, text="模型1路径:").pack(side=tk.LEFT)
        self.model1_path_entry = ttk.Entry(model1_frame)
        self.model1_path_entry.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.model1_path_entry.insert(0, self.default_model_path)
        ttk.Button(model1_frame, text="浏览", command=self.browse_model1).pack(side=tk.LEFT)

        self.ai1_frame.pack(side=tk.LEFT, padx=10)
        self.ai2_frame.pack(side=tk.LEFT, padx=10)
        
        # 棋盘绘制区域
        self.canvas = tk.Canvas(self.master, width=self.board_size, height=self.board_size, bg="#CDBA96")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        # 胜率变化曲线
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_title("胜率变化曲线")
        self.ax.set_xlabel("步数")
        self.ax.set_ylabel("胜率估计")
        self.line1, = self.ax.plot([], [], 'r-', label="AI1（黑棋）")
        self.line2, = self.ax.plot([], [], 'b-', label="AI2（白棋）")
        self.ax.legend()
        
        self.canvas_plot = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas_plot.get_tk_widget().pack(side=tk.RIGHT, padx=10, pady=10)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        
        self.update_ui()
    
    def update_ui(self):
        # 根据模式显示对应设置
        if self.mode_var.get() == "human_vs_ai":
            self.human_frame.pack()
            self.ai_ai_frame.pack_forget()
        else:
            self.human_frame.pack_forget()
            self.ai_ai_frame.pack()
    
    def browse_model(self):
        filepath = filedialog.askopenfilename(
            initialdir=os.path.join(self.script_dir, "model"),
            title="选择模型文件",
            filetypes=(("PyTorch模型", "*.pth"), ("所有文件", "*.*"))
        )
        if filepath:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, filepath)
    
    def browse_model1(self):
        filepath = filedialog.askopenfilename(
            initialdir=os.path.join(self.script_dir, "model"),
            title="选择模型文件",
            filetypes=(("PyTorch模型", "*.pth"), ("所有文件", "*.*"))
        )
        if filepath:
            self.model1_path_entry.delete(0, tk.END)
            self.model1_path_entry.insert(0, filepath)

    def browse_model2(self):
        filepath = filedialog.askopenfilename(
            initialdir=os.path.join(self.script_dir, "model"),
            title="选择模型文件",
            filetypes=(("PyTorch模型", "*.pth"), ("所有文件", "*.*"))
        )
        if filepath:
            self.model2_path_entry.delete(0, tk.END)
            self.model2_path_entry.insert(0, filepath)
    
    def load_model(self, path):
        model = AlphaZeroNet().to(mcts_device)
        try:
            model.load_state_dict(torch.load(path, map_location=mcts_device, weights_only=True))
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
            return None
    
    def board_layout_f(self):
        if self.running:
            return
            
        # 初始化游戏环境
        self.reset_game()

        self.board_layout = True

    def start_game(self):
        if self.running:
            return
        
        # 初始化游戏环境
        if self.board_layout:
            self.board_layout = False
        else:
            self.reset_game()
        self.running = True
        
        # 设置AI参数
        mode = self.mode_var.get()
        self.current_mode = mode
        try:
            if mode == "human_vs_ai":
                # 人类玩家设置
                human_first = self.human_first_var.get()
                ai_type = self.ai_type_var.get()
                
                # 初始化AI
                if ai_type == "pure_mcts":
                    self.ai_players[-1] = MCTS_Pure()
                    self.ai_types[-1] = "pure_mcts"
                else:
                    model = self.load_model(self.model_path_entry.get())
                    if not model:
                        return
                    self.ai_players[-1] = MCTS(model)
                    self.ai_types[-1] = "mcts_net"
                
                if self.game_env.current_player == self.game_env.first_player:
                    # 如果AI先手
                    if not human_first:
                        self.current_player = -1
                        self.ai_move()
                    else:
                        self.ai_think(-self.current_player)
                else:
                    if human_first:
                        self.current_player = -1
                        self.ai_move()
                    else:
                        self.ai_think(-self.current_player)
            else:
                self.current_player = self.game_env.current_player
                # 机机对战设置
                
                # 初始化AI1
                if self.ai1_type_var.get() == "pure_mcts":
                    self.ai_players[1] = MCTS_Pure()
                    self.ai_types[1] = "pure_mcts"
                else:
                    model = self.load_model(self.model1_path_entry.get())
                    if not model:
                        return
                    self.ai_players[1] = MCTS(model)
                    self.ai_types[1] = "mcts_net"
                
                # 初始化AI2
                if self.ai2_type_var.get() == "pure_mcts":
                    self.ai_players[-1] = MCTS_Pure()
                    self.ai_types[-1] = "pure_mcts"
                else:
                    model = self.load_model(self.model2_path_entry.get())
                    if not model:
                        return
                    self.ai_players[-1] = MCTS(model)
                    self.ai_types[-1] = "mcts_net"
                
                # 启动AI对战
                self.ai_move()
            
            self.draw_board()
            self.update_status()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字参数")
    
    def reset_game(self):
        if self.ai_thread_thinking is not None and self.ai_thread_thinking.is_alive():
            self.stop_ai_think()
        self.running = False
        self.game_env = GomokuEnv()
        self.board_layout = False
        self.current_player = 1
        self.ai_players = {}
        self.ai_types = {}
        self.draw_board()
        self.update_status()
        self.value_history = {1: [], -1: []}  # 初始化胜率数据
        self.update_chart()

    def update_chart(self):
        self.ax.clear()
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_title("胜率变化曲线")
        self.ax.set_xlabel("步数")
        self.ax.set_ylabel("胜率估计")
        
        x = range(len(self.value_history[1]))
        if x:
            self.ax.plot(x, self.value_history[1], 'r-', label="AI1（黑棋）")
            self.ax.legend()
        
        x = range(len(self.value_history[-1]))
        if x:
            self.ax.plot(x, self.value_history[-1], 'b-', label="AI2（白棋）")
            self.ax.legend()
        
        self.canvas_plot.draw()

    def ai_move(self):
        if not self.running:
            return
        
        def calculate_move():
            try:
                if self.ai_thread_thinking is not None and self.ai_thread_thinking.is_alive():
                    self.stop_ai_think()
                ai = self.ai_players[self.current_player]
                if self.current_mode == "ai_vs_ai":
                    if self.current_player == 1:
                        simulations = int(self.ai1_sim_var.get())
                    else:
                        simulations = int(self.ai2_sim_var.get())
                else:
                    simulations = int(self.mcts_sim_entry.get())
                if self.ai_types[self.current_player] == "pure_mcts":
                    action, value_pred, result = ai.search(self.game_env, simulations=simulations)
                else:
                    action, _, value_pred, result = ai.search(self.game_env, training=False, simulations=simulations)
                print(f"AI {self.current_player} 选择动作 {action}，预测胜率 {np.clip(value_pred, -1, 1)}, 预测结果 {result}")
                # 记录胜率数据
                self.value_history[self.current_player].append(float(np.clip(value_pred, -1, 1)))
                self.master.after(0, self.update_chart)  # 更新图表
                self.master.after(0, self.handle_action, action)
                if self.current_mode == "human_vs_ai" and self.ai_types[self.current_player] == "mcts_net":
                    self.master.after(1, self.ai_think, self.current_player)
            except Exception as e:
                print(f"AI {self.current_player} 计算错误: {str(e)}")
        
        self.ai_thread = threading.Thread(target=calculate_move)
        self.ai_thread.start()
    
    def ai_think(self, player):
        try:
                ai = self.ai_players[player]
                if self.current_mode == "ai_vs_ai":
                    if player == 1:
                        simulations = int(self.ai1_sim_var.get())
                    else:
                        simulations = int(self.ai2_sim_var.get())
                else:
                    simulations = int(self.mcts_sim_entry.get())
                if self.ai_types[player] == "pure_mcts":
                    #action, value_pred, result = ai.search(self.game_env, simulations=simulations)
                    return
                else:
                    self.ai_thread_thinking = threading.Thread(target=ai.search, args=(self.game_env, 20000, False, False, True))
                    print("ai_think")
                    self.ai_thread_thinking.start()
                    self.thinking_player = player
        except Exception as e:
            print(f"AI {player} ai_think错误: {str(e)}")

    def stop_ai_think(self):
        if self.thinking_player is None:
            return
        ai = self.ai_players[self.thinking_player]
        if self.current_mode == "ai_vs_ai":
            if self.thinking_player == 1:
                simulations = int(self.ai1_sim_var.get())
            else:
                simulations = int(self.ai2_sim_var.get())
        else:
            simulations = int(self.mcts_sim_entry.get())
        if self.ai_types[self.thinking_player] == "pure_mcts":
            #action, value_pred, result = ai.search(self.game_env, simulations=simulations)
            return
        else:
            ai.stop = True
            self.ai_thread_thinking.join()
        self.thinking_player = None
    def handle_action(self, action):
        if not self.running:
            return
        
        # 执行动作
        self.game_env.step(action)
        self.draw_board()
        
        if self.game_env.done:
            self.game_over()
        else:
            self.current_player *= -1
            self.update_status()
            
            # 继续机机对战
            if self.mode_var.get() == "ai_vs_ai":
                self.ai_move()
    
    def on_click(self, event):
        if (not self.running or self.mode_var.get() != "human_vs_ai") and not self.board_layout:
            return
        
        # 转换坐标到棋盘位置
        cell_size = self.board_size / BOARD_SIZE
        col = int(event.x // cell_size)
        row = int(event.y // cell_size)
        
        # 检查是否合法落子
        if self.game_env.board[row][col] == 0 and (self.current_player == 1 or self.board_layout):
            self.game_env.step((row, col))
            self.draw_board()
            
            if self.game_env.done:
                self.game_over()
            else:
                self.current_player *= -1
                self.update_status()
                if not self.board_layout:
                    self.ai_move()
                    self.update_status()
    
    def draw_board(self):
        self.canvas.delete("all")
        cell_size = self.board_size / BOARD_SIZE
        
        # 绘制棋盘线
        for i in range(BOARD_SIZE):
            self.canvas.create_line(0, i*cell_size, self.board_size, i*cell_size)
            self.canvas.create_line(i*cell_size, 0, i*cell_size, self.board_size)
        
        # 绘制棋子
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                player = self.game_env.board[row][col]
                if player != 0:
                    x = col * cell_size + cell_size/2
                    y = row * cell_size + cell_size/2
                    color = "black" if player == 1 else "white"
                    self.canvas.create_oval(x-15, y-15, x+15, y+15, 
                                          fill=color, outline="black")
        
        # 绘制胜利路径
        if self.game_env.done and self.game_env.win_paths:
            for (row, col) in self.game_env.win_paths:
                x = col * cell_size + cell_size/2
                y = row * cell_size + cell_size/2
                self.canvas.create_oval(x-18, y-18, x+18, y+18, 
                                      outline="red", width=3)
        elif self.game_env.action_history: # 标记最后一步
            row, col = self.game_env.action_history[-1]
            x = col * cell_size + cell_size/2
            y = row * cell_size + cell_size/2
            self.canvas.create_oval(x-18, y-18, x+18, y+18, 
                                  outline="blue", width=3)
        # 绘制行列序号
        offset = 1
        for i in range(BOARD_SIZE):
            # 列号（顶部）
            self.canvas.create_text(
                i * cell_size + cell_size/2,  # X坐标居中
                offset,  # 顶部留出空间
                text=str(i+1),
                font=("Arial", 10),
                fill="green",
                anchor=tk.N
            )
            # 行号（左侧）
            self.canvas.create_text(
                offset,  # 左侧留出空间
                i * cell_size + cell_size/2,  # Y坐标居中
                text=str(i+1),#chr(ord('A') + i),
                font=("Arial", 10),
                fill="green",
                anchor=tk.W
            )

    def update_status(self):
        if self.game_env.done:
            if self.game_env.winner == 0:
                text = f"游戏结束！平局！总步数: {len(self.game_env.action_history)}"
            else:
                text = f"游戏结束！胜利者：{'黑棋' if self.game_env.winner == 1 else '白棋'}！总步数: {len(self.game_env.action_history)}"
        else:
            player = "黑棋" if self.current_player == 1 else "白棋"
            mode = "（人类）" if (self.mode_var.get() == "human_vs_ai" and 
                               ((self.current_player == 1 and self.human_first_var.get()) or
                                (self.current_player == -1 and not self.human_first_var.get()))) else "（AI）"
            text = f"当前回合{len(self.game_env.action_history) + 1}：{player} {mode}"
        self.status_var.set(text)
    
    def game_over(self):
        self.running = False
        self.update_status()
        if self.game_env.winner == 0:
            messagebox.showinfo("游戏结束", "平局！")
        else:
            messagebox.showinfo("游戏结束", f"{'黑棋' if self.game_env.winner == 1 else '白棋'}获胜！")

if __name__ == "__main__":
    root = tk.Tk()
    gui = GomokuGUI(root)
    root.mainloop()