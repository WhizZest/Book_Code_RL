import cv2
import numpy as np
import pyautogui
import time
from AlphaZero import GomokuEnv, MCTS_Pure, MCTS, AlphaZeroNet, BOARD_SIZE, device, mcts_device
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

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

def detect_pieces(board_img, intersections, show_result=False, timeout=2000):
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
            board[i, j] = 1  # 黑子
            cv2.circle(board_img, (x, y), radius, (0, 0, 255), -1)
        elif r > 150 and g > 150 and b > 150:  # 接近白色
            board[i, j] = -1  # 白子
            cv2.circle(board_img, (x, y), radius, (255, 0, 0), -1)
    
    if show_result:
        cv2.imshow("Detected Pieces", board_img)
        cv2.waitKey(timeout)
        cv2.destroyAllWindows()
    return board

def evaluate_single_game(ai_play, actionMapToCoords, current_model_player, MCTS_simulations):
    """ 运行一局评估对局 """
    env = GomokuEnv()
    mcts = MCTS(model=ai_play)
    wait_time = 0.5
    result = None
    value_pred_list = []
    steps_TakeBack = -1
    buffer = []  # 缓冲区
    save_buffer_flag = False
    value_pred_min = 1
    value_pred_min_step = 0

    screen = capture_board(region=None)  # 截取全屏图像
    board_matrix_current = detect_pieces(screen.copy(), actionMapToCoords, show_result=False, timeout=0)
    if env.current_player == current_model_player and not np.array_equal(env.board, board_matrix_current):
        raise ValueError("当前玩家为先手，但当前棋盘不为空，请检查先后手是否正确")

    while not env.done:
        if env.current_player == current_model_player:
            action, _, value_pred, result = mcts.search(env, training=False, simulations=MCTS_simulations)
            if result is not None and result == -1 and steps_TakeBack < 0 and len(env.action_history) >= 2:
                steps_TakeBack = len(env.action_history) - 2
            value_pred_list.append(np.clip(value_pred, -1, 1))
            if value_pred < value_pred_min:
                value_pred_min = value_pred
                value_pred_min_step = len(env.action_history)
            x, y = actionMapToCoords[action]
            pyautogui.moveTo(x, y)
            pyautogui.click()
            print(f"回合{len(env.action_history)+1}：AI 选择动作 {action}，预测胜率 {np.clip(value_pred, -1, 1)}, 预测结果 {result}")
        else:
            time.sleep(wait_time)
            if result is not None and result == -1:
                board_img = capture_board((800, 356, 135, 124))
                # 平均颜色为白色
                if np.mean(board_img) > 210:
                    print(f"AI 认输")
                    env.winner = env.current_player
                    break
            action = None
            while action is None:
                screen = capture_board(region=None)  # 截取全屏图像
                board_matrix_current = detect_pieces(screen.copy(), actionMapToCoords, show_result=False, timeout=0)
                action = detect_board_change(board_matrix_current, env.board)
                if action is None:
                    #wait_time += 0.1
                    time.sleep(wait_time)

        reward = env.step(action)
        if reward is None:
            print(f"当前玩家：{env.current_player}，无效动作 {action}")
            return None
    if value_pred_min < -0.7:
        save_buffer_flag = True
        if steps_TakeBack < 0:
            env.action_history = env.action_history[:value_pred_min_step]
            print(f"预测胜率最低为{value_pred_min}，回退到第{value_pred_min_step+1}步")
    if len(env.action_history) > 0 and save_buffer_flag:
        buffer.append((env.action_history, steps_TakeBack))
    # 绘制value_pred_list图像
    plt.plot(value_pred_list)
    plt.xlabel('Step')
    plt.ylabel('Value Prediction')
    plt.title('Value Prediction Over Time')
    plt.show()
    return 1 if env.winner == current_model_player else 0 if env.winner == 0 else -1, buffer

def detect_board_change(board_matrix_current, board_matrix_previous):
    """ 检测board_matrix_current相对于board_matrix_previous的变化位置 """
    change_list = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board_matrix_previous[i, j] != board_matrix_current[i, j]:
                change_list.append((i, j))
    if len(change_list) > 1:
        raise ValueError(f"检测到{len(change_list)}个变化位置: {change_list}")
    return change_list[0] if len(change_list) == 1 else None

def save_buffer(buffer):
    """ 保存缓冲区数据 """
    if len(buffer) > 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"buffer_{timestamp}.pkl"
        filepath = os.path.join(script_dir, "eval_buffer", filename)
        folder = os.path.join(script_dir, "eval_buffer")
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(filepath, "wb") as f:
            pickle.dump(buffer, f)
        print(f"缓冲区数据已保存至 {filepath}，共 {len(buffer)} 条数据。")

if __name__ == "__main__":
    screen = capture_board(region=None)  # 截取全屏图像
    # 设置棋盘左上角和右下角的坐标，以及棋盘大小
    actionMapToCoords = detect_grid_intersections(screen.copy(), top_left=(723, 162), bottom_right=(1884,1326), board_size=BOARD_SIZE, show_result=True, timeout=0)
    if actionMapToCoords is not None:
        # 获取本地文件夹路径，与"model"文件夹拼接,得到模型文件路径,与当前模型文件名拼接，得到完整模型文件路径
        local_folder = os.path.dirname(os.path.abspath(__file__))
        model_file_path = os.path.join(local_folder, "model", "az_model_260.pth")
        ai_play = AlphaZeroNet().to(mcts_device)
        ai_play.load_state_dict(torch.load(model_file_path, map_location=mcts_device, weights_only=True))
        ai_play.eval()
        current_model_player = 1  # 当前模型玩家, 1表示黑子，-1表示白子
        MCTS_simulations = 200  # MCTS模拟次数
        result, buffer = evaluate_single_game(ai_play, actionMapToCoords, current_model_player, MCTS_simulations)
        save_buffer(buffer)
        print(f"评估结果: {result}")
    else:
        print("未能检测到棋盘线")
