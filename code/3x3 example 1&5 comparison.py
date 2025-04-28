import numpy as np
from itertools import product
import random
import time

# 定义棋盘大小
K = 3

#0=空, 1=X（玩家）, 2=O（random player）

# 初始化状态：3x3的空棋盘（多维数组）
def initialize_board():
    return np.zeros((K, K), dtype=int)
# print(initialize_board())

# 检查是否终局，返回获胜状态（1=X赢, 2=O赢, 0=平局，-1=未结束）
def check_winner(board):
    lines = [  # 所有可能的获胜组合连接位置 board[i,j]意思是棋盘数组的第i行第j列
        # 一整行
        [board[0, 0], board[0, 1], board[0, 2]],
        [board[1, 0], board[1, 1], board[1, 2]],
        [board[2, 0], board[2, 1], board[2, 2]],
        # 一整列
        [board[0, 0], board[1, 0], board[2, 0]],
        [board[0, 1], board[1, 1], board[2, 1]],
        [board[0, 2], board[1, 2], board[2, 2]],
        # 对角线
        [board[0, 0], board[1, 1], board[2, 2]],
        [board[0, 2], board[1, 1], board[2, 0]],
    ]
    for line in lines:  # 检查所有组合，看是否形成获胜
        if len(set(line)) == 1 and line[0] == 1:
            return 1  # X赢
        if len(set(line)) == 1 and line[0] == 2:
           return 2  # O赢
    if all(cell != 0 for row in board for cell in row):
        return 0  # 平局
    return -1  # 未结束

# 生成所有可能状态（没有考虑对称性）
def generate_states():
    states = {}
    for cells in product([0, 1, 2], repeat=9):  # 生成所有可能的3x3棋盘状态
        board = np.array(cells).reshape(3, 3)  # 将长度为9的数组转化为3x3的二维数组（类棋盘）
        count_1 = np.sum(board == 1)  # 玩家1的棋子数
        count_2 = np.sum(board == 2)  # 玩家2的棋子数
        # 检查回合合法性（假设X先手）：X和O的数量差不超过1，且X先手即一定是X比O多1
        if count_1-count_2 <= 1 and count_1-count_2 >= 0: # 只存储合法状态对应的价值V(s)
            if check_winner(board) == 1: # X赢，V(s)=1
                states[tuple(cells)] = 1  
            elif check_winner(board) == 2: # O赢，V(s)=-1
                states[tuple(cells)] = -1
            else: # 平局或未结束（checkwinner=0或-1），V(s)=0
                states[tuple(cells)] = 0
    return states # 所有合法状态对应的价值V(s)。结构：{状态s：价值V(s)}
# print(generate_states())


def is_fixed_point_reached(old_values, new_values, tol=1e-6):
    # 检查是否达到收敛条件（状态价值V(s)的更新变化极小）
    for state in old_values.keys():
        if abs(old_values[state] - new_values[state]) > tol:
            return False
    return True

# VI
def synchronous_value_iteration(states, gamma=1.0, tol=1e-6):
    start_time = time.time()  # 记录开始时间
    iteration_count = 0  # 迭代次数
    operation_count = 0  # 操作（计算）次数
    while True:
        iteration_count += 1 # 迭代次数+1
        new_states = {} # 存储更新了价值V(s)的新{状态、价值}pair
        is_fixed_point = True  # 假设当前已收敛
        
        for state in states.keys(): # 遍历所有状态以更新价值V(s)
            board = np.array(state).reshape(K, K) # 将正在检查的这一个状态转化为3x3
            winner = check_winner(board) # 检查是否结束
            
            if winner != -1: # 如果已经结束，则直接赋予1/-1/0的价值，无需迭代得最优
                new_v = 1 if winner == 1 else (-1 if winner == 2 else 0)
            else: # 如果未结束，则需要迭代得最优
                # 判断当前玩家（X或O）
                if np.sum(board == 1) == np.sum(board == 2):
                    player = 1  # X的回合
                else:
                    player = 2  # O的回合
                
                if player == 1: # X的回合，取更新后价值最大的状态为下一步
                    max_v = -np.inf # 初始化"最优价值"为负无穷
                    for i, j in zip(*np.where(board == 0)): #遍历所有空格，即能落子形成下一步的位置
                        new_board = board.copy() # 临时棋盘，储存一个落子后的状态，用于计算即时价值进行比较
                        new_board[i, j] = 1 # 在某一个空位落子X
                        next_state = tuple(new_board.flatten()) # 将临时棋盘展平，对应states中的key
                        max_v = max(max_v, gamma * states.get(next_state)) # 更新"最优价值"。这里更新用的值是原始states
                        # 计算方式实际也是Bellman方程期望值，只是一定会选择最大值，所以只有对应最大项的概率为1，其余都为0
                    new_v = max_v # 结束所有落子可能循环之后，取了max即为当前状态的最佳值（a=argmaxV(s)代表对应的落子方式）
                else:  # O的回合，状态的更新为期望值
                    empty_cells = list(zip(*np.where(board == 0))) # 所有空格位置坐标
                    operation_count += len(empty_cells) #增加计算次数
                    total_v = 0
                    for i, j in empty_cells: # 遍历所有空格坐标（等可能落子）
                        new_board = board.copy() # 临时棋盘
                        new_board[i, j] = 2 # 在某一个空位落子O
                        next_state = tuple(new_board.flatten())
                        total_v += gamma * states.get(next_state) / len(empty_cells) # 概率*新状态价值，加权求和
                    new_v = total_v # 期望值
            
            new_states[state] = new_v # 更新newstates中该states的最优价值V(s)，而不是原始states，所以更新value步骤时用不到前面更新过的值
            
            if abs(states[state] - new_v) > tol: # 检查是否收敛
                is_fixed_point = False
        
        states = new_states # 把newstate更新为current state。注意，更新时全部用的上一步的原始状态，在for state循环外
        if is_fixed_point: # 结果收敛，结束迭代
            break
    
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time
    
    return states, iteration_count, operation_count, execution_time


# Random CyclicVI
def random_cyclic_value_iteration(states, gamma=1.0, max_iter=1000, tol=1e-6):
    start_time = time.time()  # 记录开始时间
    iteration_count = 0  # 迭代计数器
    operation_count = 0  # 操作计数器
    for _ in range(max_iter):
        iteration_count += 1  # 每次迭代计数
        is_fixed_point = True  # 假设当前已收敛
        
        # 随机打乱状态顺序
        shuffled_states = list(states.keys())
        random.shuffle(shuffled_states) ###如果不需要随机迭代，只是CyclicVI，直接删除这行即可###
        
        for state in shuffled_states:
            board = np.array(state).reshape(K, K)
            winner = check_winner(board)
            if winner != -1:  # 终局状态
                new_v = 1 if winner == 1 else (-1 if winner == 2 else 0)
            else:
                # 当前玩家（假设X的回合，随机玩家是O）
                if np.sum(board == 1) == np.sum(board == 2):
                    player = 1  # X的回合
                else:
                    player = 2  # O的回合
                
                if player == 1:  # X的回合：取最大值
                    max_v = -np.inf
                    for i, j in zip(*np.where(board == 0)):
                        new_board = board.copy()
                        new_board[i, j] = 1
                        next_state = tuple(new_board.flatten())
                        # Bellman方程：V(s) = max [R + γ * V(s')]
                        # 井字棋无中间奖励，R=0
                        max_v = max(max_v, gamma * states.get(next_state, 0))
                    new_v = max_v
                else:  # O的回合：随机玩家，取期望
                    empty_cells = list(zip(*np.where(board == 0)))
                    total_v = 0
                    for i, j in empty_cells:
                        new_board = board.copy()
                        new_board[i, j] = 2
                        next_state = tuple(new_board.flatten())
                        total_v += gamma * states.get(next_state, 0)
                    new_v = total_v / len(empty_cells)  # 期望值
                    operation_count += len(empty_cells)
            
            # 检查是否偏离固定点条件
            if abs(states[state] - new_v) > tol:
                is_fixed_point = False
                
            states[state] = new_v
        
        if is_fixed_point:  # 如果已收敛，退出循环
            break
            
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time
    
    return states, iteration_count, operation_count, execution_time



# 主流程
if __name__ == "__main__":
    # 生成状态空间
    print("Generate all states...")
    states = generate_states()
    print(f"Size of all states: {len(states)}")
    
    # 准备比较结果
    results = {
        'Approach': [],
        'Excute time (s)': [],
        'Iteration counts': [],
        'Operation counts': [],
    }
    
    # 运行同步值迭代
    print("\nRunning Synchronous VI...")
    sync_states = states.copy()
    sync_states, sync_iterations, sync_ops, sync_time = synchronous_value_iteration(sync_states)
    
    # 找出同步值迭代的最优第一步
    board = initialize_board()
    best_sync_move = None
    best_sync_value = -np.inf
    
    print("Value at each position for Sync-VI:")
    sync_values = {}
    for i, j in zip(*np.where(board == 0)):
        new_board = board.copy()
        new_board[i, j] = 1
        next_state = tuple(new_board.flatten())
        value = sync_states.get(next_state)
        sync_values[(i, j)] = value
        print(f"pos ({i}, {j}): value = {value}")
    
    unique_vals = set(sync_values.values())
    if len(unique_vals) == 1:
        print("All starting positions have the same value. No unique best first step.")
        best_sync_moves = None
    else:
        max_val = max(sync_values.values())
        best_sync_moves = [pos for pos, val in sync_values.items() if np.isclose(val, max_val, atol=1e-6)]
        print(f"Best first step(s) of Sync-VI: {best_sync_moves}, value: {max_val}")
    
    # 记录结果
    results['Approach'].append('Synchronous VI')
    results['Excute time (s)'].append(sync_time)
    results['Iteration counts'].append(sync_iterations)
    results['Operation counts'].append(sync_ops)

    
    
    # 运行随机循环值迭代
    print("\nRunning Random Cyclic VI...")
    random_states = states.copy()
    random_states, random_iterations, random_ops, random_time = random_cyclic_value_iteration(random_states)
    
    # 找出随机循环值迭代的最优第一步
    best_random_move = None
    best_random_value = -np.inf
    
    print("Value at each position for Random Cyclic VI:")
    random_values = {}
    for i, j in zip(*np.where(board == 0)):
        new_board = board.copy()
        new_board[i, j] = 1
        next_state = tuple(new_board.flatten())
        value = random_states.get(next_state)
        random_values[(i, j)] = value
        print(f"pos ({i}, {j}): value = {value}")
    
    unique_vals_rand = set(random_values.values())
    if len(unique_vals_rand) == 1:
        print("All starting positions have the same value. No unique best first step.")
        best_random_moves = None
    else:
        max_val_rand = max(random_values.values())
        best_random_moves = [pos for pos, val in random_values.items() if np.isclose(val, max_val_rand, atol=1e-6)]
        print(f"Best first step(s) of Random Cyclic VI: {best_random_moves}, value: {max_val_rand}")
    
    
    print(f"Excute time (s): {random_time:.4f}s, Iteration counts: {random_iterations}, Operation counts: {random_ops}")
    
    # 记录结果
    results['Approach'].append('Random Cyclic VI')
    results['Excute time (s)'].append(random_time)
    results['Iteration counts'].append(random_iterations)
    results['Operation counts'].append(random_ops)

    
    # 打印比较结果表格
    print("\nComparison:")
    print("-" * 80)
    print(f"{'Approach':<12} {'Excute time(s)':<10} {'Iteration counts':<10} {'Operation counts':<10} ")
    print("-" * 80)
    
    for i in range(len(results['Approach'])):
        algo = results['Approach'][i]
        time_used = results['Excute time (s)'][i]
        iterations = results['Iteration counts'][i]
        operations = results['Operation counts'][i]

        
        print(f"{algo:<20} {time_used:<15.4f} {iterations:<10} {operations:<15} ")
    
    print('\n')

    # 同步值迭代 vs 随机循环值迭代
    time_improvement2 = (results['Excute time (s)'][0] - results['Excute time (s)'][1]) / results['Excute time (s)'][0] * 100
    iter_improvement2 = (results['Iteration counts'][0] - results['Iteration counts'][1]) / results['Iteration counts'][0] * 100
    print(f"Compare to Sync-VI, Random Cyclic VI improve in excute time: {time_improvement2:.2f}%")
    print(f"Compare to Sync-VI, Random Cyclic VI improve in iteration counts: {iter_improvement2:.2f}%")


