import cv2
import numpy as np
import pyautogui
import time
from AlphaZero import GomokuEnv, MCTS_Pure, MCTS, AlphaZeroNet, BOARD_SIZE, device, mcts_device
import os
import torch
import matplotlib.pyplot as plt

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

def detect_grid_intersections(edges, board_img, region, board_size=15, show_result=False, timeout=2000):
    """ 识别棋盘的交叉点并计算每个格子的间距，同时可视化检测效果 """
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=100, maxLineGap=10)
    if lines is None:
        return None
    
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < abs(x1 - x2):
            horizontal_lines.append((y1 + y2) // 2)
        else:
            vertical_lines.append((x1 + x2) // 2)
    
    horizontal_lines = merge_lines(horizontal_lines, min_dist=10)
    vertical_lines = merge_lines(vertical_lines, min_dist=10)
    
    if len(horizontal_lines) < board_size or len(vertical_lines) < board_size:
        return None
    
    intersections = {}
    for i, y in enumerate(horizontal_lines):
        for j, x in enumerate(vertical_lines):
            new_x = int(x + region[0])
            new_y = int(y + region[1])
            intersections[(i, j)] = (new_x, new_y)
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

    screen = capture_board(region=None)  # 截取全屏图像
    board_matrix_current = detect_pieces(screen.copy(), actionMapToCoords, show_result=False, timeout=0)
    if env.current_player == current_model_player and not np.array_equal(env.board, board_matrix_current):
        raise ValueError("当前玩家为先手，但当前棋盘不为空，请检查先后手是否正确")

    while not env.done:
        if env.current_player == current_model_player:
            action, _, value_pred, result = mcts.search(env, training=False, simulations=MCTS_simulations)
            value_pred_list.append(np.clip(value_pred, -1, 1))
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
                    env.winner = -env.current_player
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
    # 绘制value_pred_list图像
    plt.plot(value_pred_list)
    plt.xlabel('Step')
    plt.ylabel('Value Prediction')
    plt.title('Value Prediction Over Time')
    plt.show()
    return 1 if env.winner == current_model_player else 0 if env.winner == 0 else -1

def detect_board_change(board_matrix_current, board_matrix_previous):
    """ 检测board_matrix_current相对于board_matrix_previous的变化位置 """
    change_list = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board_matrix_previous[i, j] != board_matrix_current[i, j]:
                change_list.append((i, j))
    if len(change_list) != 1:
        print(f"检测到{len(change_list)}个变化位置: {change_list}")
    return change_list[0] if len(change_list) == 1 else None

if __name__ == "__main__":
    screen = capture_board(region=None)  # 截取全屏图像
    region = select_region()  # 让用户手动选择棋盘区域
    print(f"选择的区域: {region}")
    board_img = capture_board(region)
    #print(f"截取的棋盘图像平均颜色: {np.mean(board_img)}")
    edges = preprocess_image(board_img)
    actionMapToCoords = detect_grid_intersections(edges, board_img.copy(), region, board_size=BOARD_SIZE, show_result=True, timeout=0)
    if actionMapToCoords is not None:
        # 获取本地文件夹路径，与"model"文件夹拼接,得到模型文件路径,与当前模型文件名拼接，得到完整模型文件路径
        local_folder = os.path.dirname(os.path.abspath(__file__))
        model_file_path = os.path.join(local_folder, "model", "az_model_260.pth")
        ai_play = AlphaZeroNet().to(mcts_device)
        ai_play.load_state_dict(torch.load(model_file_path, map_location=mcts_device, weights_only=True))
        ai_play.eval()
        current_model_player = 1  # 当前模型玩家, 1表示黑子，-1表示白子
        MCTS_simulations = 200  # MCTS模拟次数
        result = evaluate_single_game(ai_play, actionMapToCoords, current_model_player, MCTS_simulations)
        print(f"评估结果: {result}")
    else:
        print("未能检测到棋盘线")
