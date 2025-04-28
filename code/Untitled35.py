# !/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from tqdm import tqdm

class MDPSolver:
    def __init__(self, states, actions, transition_probs, costs, gamma=0.9):
        """
        初始化MDP求解器
        
        参数:
            states: 状态数量
            actions: 每个状态可用的动作集合 {state: [actions]}
            transition_probs: 转移概率 {(state, action): [probs]}
            costs: 即时成本 {(state, action): cost}
            gamma: 折扣因子
        """
        self.m = states  # 状态数量
        self.actions = actions  # 每个状态的动作集合
        self.P = transition_probs  # 转移概率
        self.c = costs  # 即时成本
        self.gamma = gamma  # 折扣因子
        
        # 计算动作总数
        self.n = sum(len(actions[s]) for s in range(states))
        
    def value_iteration(self, max_iter=1000, tol=1e-6, verbose=False):
        """
        标准值迭代方法
        
        参数:
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否输出详细信息
        
        返回:
            values: 最优值函数
            policy: 最优策略
            errors: 每次迭代的误差
            iterations: 迭代次数
        """
        # 初始化值函数
        V = np.zeros(self.m)
        errors = []
        
        start_time = time.time()
        
        iter_range = range(max_iter)
        if verbose:
            iter_range = tqdm(iter_range, desc="Standard Value Iteration")
            
        for k in iter_range:
            V_next = np.zeros(self.m)
            
            # 对每个状态进行更新
            for s in range(self.m):
                # 计算每个动作的Q值
                q_values = []
                for a in self.actions[s]:
                    # 计算期望成本
                    expected_cost = self.c[(s, a)] + self.gamma * np.dot(self.P[(s, a)], V)
                    q_values.append(expected_cost)
                
                # 取最小Q值作为新的值函数
                V_next[s] = min(q_values)
            
            # 计算误差
            error = np.max(np.abs(V_next - V))
            errors.append(error)
            
            # 更新值函数
            V = V_next.copy()
            
            # 收敛检查
            if error < tol:
                break
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # 提取最优策略
        policy = {}
        for s in range(self.m):
            q_values = []
            for a in self.actions[s]:
                expected_cost = self.c[(s, a)] + self.gamma * np.dot(self.P[(s, a)], V)
                q_values.append((a, expected_cost))
            
            best_action, _ = min(q_values, key=lambda x: x[1])
            policy[s] = best_action
        
        return V, policy, errors, k+1, computation_time
    
    def cyclic_value_iteration(self, max_iter=1000, tol=1e-6, verbose=False):
        """
        循环值迭代方法
        
        参数:
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否输出详细信息
        
        返回:
            values: 最优值函数
            policy: 最优策略
            errors: 每次迭代的误差
            iterations: 迭代次数
        """
        # 初始化值函数
        V = np.zeros(self.m)
        errors = []
        
        start_time = time.time()
        
        iter_range = range(max_iter)
        if verbose:
            iter_range = tqdm(iter_range, desc="Cyclic Value Iteration")
            
        for k in iter_range:
            V_old = V.copy()
            
            # 循环地更新每个状态
            for s in range(self.m):
                # 计算每个动作的Q值
                q_values = []
                for a in self.actions[s]:
                    # 使用最新的V值计算期望成本
                    expected_cost = self.c[(s, a)] + self.gamma * np.dot(self.P[(s, a)], V)
                    q_values.append(expected_cost)
                
                # 立即更新当前状态的值函数
                V[s] = min(q_values)
            
            # 计算误差
            error = np.max(np.abs(V - V_old))
            errors.append(error)
            
            # 收敛检查
            if error < tol:
                break
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # 提取最优策略
        policy = {}
        for s in range(self.m):
            q_values = []
            for a in self.actions[s]:
                expected_cost = self.c[(s, a)] + self.gamma * np.dot(self.P[(s, a)], V)
                q_values.append((a, expected_cost))
            
            best_action, _ = min(q_values, key=lambda x: x[1])
            policy[s] = best_action
        
        return V, policy, errors, k+1, computation_time

# 创建稠密转移矩阵的MDP
def create_dense_mdp(m, actions_per_state, gamma=0.9, seed=42):
    """
    创建具有稠密转移矩阵的MDP
    """
    np.random.seed(seed)
    
    # 每个状态的动作集合
    actions = {s: list(range(actions_per_state)) for s in range(m)}
    
    # 稠密转移概率矩阵 - 每个动作可以转移到所有状态
    transition_probs = {}
    for s in range(m):
        for a in actions[s]:
            # 生成随机概率
            probs = np.random.random(m)
            # 归一化使总和为1
            probs = probs / np.sum(probs)
            transition_probs[(s, a)] = probs
    
    # 即时成本
    costs = {}
    for s in range(m):
        for a in actions[s]:
            costs[(s, a)] = np.random.uniform(0, 10)
    
    return MDPSolver(m, actions, transition_probs, costs, gamma)

# 创建稀疏转移矩阵的MDP
def create_sparse_mdp(m, actions_per_state, sparsity=0.8, gamma=0.9, seed=42):
    """
    创建具有稀疏转移矩阵的MDP
    
    参数:
        sparsity: 稀疏度，表示转移概率矩阵中为0的元素比例
    """
    np.random.seed(seed)
    
    # 每个状态的动作集合
    actions = {s: list(range(actions_per_state)) for s in range(m)}
    
    # 稀疏转移概率矩阵
    transition_probs = {}
    for s in range(m):
        for a in actions[s]:
            # 生成稀疏概率向量
            probs = np.zeros(m)
            
            # 随机选择一些非零元素位置，确保至少有一个非零元素
            non_zero_count = max(1, int(m * (1 - sparsity)))
            non_zero_indices = np.random.choice(m, size=non_zero_count, replace=False)
            
            # 为非零元素赋随机值
            probs[non_zero_indices] = np.random.random(non_zero_count)
            
            # 归一化使总和为1
            probs = probs / np.sum(probs)
            transition_probs[(s, a)] = probs
    
    # 即时成本
    costs = {}
    for s in range(m):
        for a in actions[s]:
            costs[(s, a)] = np.random.uniform(0, 10)
    
    return MDPSolver(m, actions, transition_probs, costs, gamma)

# 运行实验比较稠密和稀疏MDP的值迭代性能
def run_experiments():
    # 参数设置
    m = 20  # 状态数量
    actions_per_state = 4  # 每个状态的动作数量
    gamma = 0.9  # 折扣因子
    
    # 创建不同的MDP
    dense_mdp = create_dense_mdp(m, actions_per_state, gamma)
    sparse_mdp = create_sparse_mdp(m, actions_per_state, sparsity=0.8, gamma=gamma)
    
    # 运行标准值迭代
    print("Running Standard Value Iteration...")
    dense_values, dense_policy, dense_errors, dense_iters, dense_time = dense_mdp.value_iteration(verbose=True)
    sparse_values, sparse_policy, sparse_errors, sparse_iters, sparse_time = sparse_mdp.value_iteration(verbose=True)
    
    # 运行循环值迭代
    print("\nRunning Cyclic Value Iteration...")
    dense_cyclic_values, dense_cyclic_policy, dense_cyclic_errors, dense_cyclic_iters, dense_cyclic_time = dense_mdp.cyclic_value_iteration(verbose=True)
    sparse_cyclic_values, sparse_cyclic_policy, sparse_cyclic_errors, sparse_cyclic_iters, sparse_cyclic_time = sparse_mdp.cyclic_value_iteration(verbose=True)
    
    # 绘制收敛曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.semilogy(dense_errors, label='Standard VI')
    plt.semilogy(dense_cyclic_errors, label='Cyclic VI')
    plt.title("Convergence Performance for Dense MDP")
    plt.xlabel("Iterations")
    plt.ylabel("Error (log scale)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(sparse_errors, label='Standard VI')
    plt.semilogy(sparse_cyclic_errors, label='Cyclic VI')
    plt.title("Convergence Performance for Sparse MDP")
    plt.xlabel("Iterations")
    plt.ylabel("Error (log scale)")
    plt.legend()
    plt.grid(True)
    
    # 计算迭代次数和计算时间
    methods = ['Standard VI', 'Cyclic VI']
    dense_iters_data = [dense_iters, dense_cyclic_iters]
    sparse_iters_data = [sparse_iters, sparse_cyclic_iters]
    dense_times = [dense_time, dense_cyclic_time]
    sparse_times = [sparse_time, sparse_cyclic_time]
    
    # 显示迭代次数比较
    plt.subplot(2, 2, 3)
    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width/2, dense_iters_data, width, label='Dense MDP')
    plt.bar(x + width/2, sparse_iters_data, width, label='Sparse MDP')
    plt.xlabel('Solution Method')
    plt.ylabel('Number of Iterations')
    plt.title('Iteration Count Comparison')
    plt.xticks(x, methods)
    plt.legend()
    
    # 显示计算时间比较
    plt.subplot(2, 2, 4)
    plt.bar(x - width/2, dense_times, width, label='Dense MDP')
    plt.bar(x + width/2, sparse_times, width, label='Sparse MDP')
    plt.xlabel('Solution Method')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time Comparison')
    plt.xticks(x, methods)
    plt.legend()
    
    plt.tight_layout()
    plt.show()  # 在Jupyter中直接显示
    
    # 创建转移矩阵可视化
    plt.figure(figsize=(12, 5))
    
    # 选择第一个状态第一个动作的转移概率来可视化
    dense_trans = dense_mdp.P[(0, 0)]
    sparse_trans = sparse_mdp.P[(0, 0)]
    
    plt.subplot(1, 2, 1)
    sns.heatmap(dense_trans.reshape(1, -1), cmap='viridis', annot=False)
    plt.title("Dense Transition Matrix Example (State 0, Action 0)")
    plt.xlabel("Target State")
    plt.yticks([])
    
    plt.subplot(1, 2, 2)
    sns.heatmap(sparse_trans.reshape(1, -1), cmap='viridis', annot=False)
    plt.title("Sparse Transition Matrix Example (State 0, Action 0)")
    plt.xlabel("Target State")
    plt.yticks([])
    
    plt.tight_layout()
    plt.show()  # 在Jupyter中直接显示

    # 输出结果汇总
    print("\nResults Summary:")
    print("-" * 50)
    print(f"Dense MDP - Standard VI: {dense_iters} iterations, {dense_time:.4f} seconds")
    print(f"Dense MDP - Cyclic VI: {dense_cyclic_iters} iterations, {dense_cyclic_time:.4f} seconds")
    print(f"Sparse MDP - Standard VI: {sparse_iters} iterations, {sparse_time:.4f} seconds")
    print(f"Sparse MDP - Cyclic VI: {sparse_cyclic_iters} iterations, {sparse_cyclic_time:.4f} seconds")
    
    # 计算最终值函数误差
    dense_diff = np.max(np.abs(dense_values - dense_cyclic_values))
    sparse_diff = np.max(np.abs(sparse_values - sparse_cyclic_values))
    
    print(f"\nDense MDP: Final value function difference between Standard VI and Cyclic VI: {dense_diff:.6f}")
    print(f"Sparse MDP: Final value function difference between Standard VI and Cyclic VI: {sparse_diff:.6f}")
    
    # 进行更大规模的测试来分析计算复杂性
    print("\nAnalyzing computational complexity for different MDP sizes...")
    
    # 不同状态数量
    state_sizes = [10, 20, 50, 100]
    dense_std_times = []
    dense_cyclic_times = []
    sparse_std_times = []
    sparse_cyclic_times = []
    
    for size in state_sizes:
        print(f"Testing MDP with {size} states...")
        
        # 创建MDP
        dense = create_dense_mdp(size, actions_per_state, gamma)
        sparse = create_sparse_mdp(size, actions_per_state, sparsity=0.8, gamma=gamma)
        
        # 运行算法并记录时间
        _, _, _, _, d_std_time = dense.value_iteration()
        _, _, _, _, d_cyc_time = dense.cyclic_value_iteration()
        _, _, _, _, s_std_time = sparse.value_iteration()
        _, _, _, _, s_cyc_time = sparse.cyclic_value_iteration()
        
        dense_std_times.append(d_std_time)
        dense_cyclic_times.append(d_cyc_time)
        sparse_std_times.append(s_std_time)
        sparse_cyclic_times.append(s_cyc_time)
    
    # 绘制计算复杂性图
    plt.figure(figsize=(10, 6))
    plt.plot(state_sizes, dense_std_times, 'o-', label='Dense MDP - Standard VI')
    plt.plot(state_sizes, dense_cyclic_times, 's-', label='Dense MDP - Cyclic VI')
    plt.plot(state_sizes, sparse_std_times, '^-', label='Sparse MDP - Standard VI')
    plt.plot(state_sizes, sparse_cyclic_times, 'd-', label='Sparse MDP - Cyclic VI')
    plt.xlabel('Number of States')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time vs. MDP Size')
    plt.legend()
    plt.grid(True)
    plt.show()  # 在Jupyter中直接显示
    
    # 验证收敛速率
    print("\nValidating convergence rate of Value Iteration...")
    
    # 创建新的MDP以进行收敛速率分析
    test_mdp = create_dense_mdp(10, 3, gamma=0.9)
    
    # 运行大量迭代以确保收敛
    optimal_values, _, _, _, _ = test_mdp.value_iteration(max_iter=10000, tol=1e-10)
    
    # 从零向量开始运行固定次数的迭代，并计算每次迭代与最优解的无穷范数距离
    V = np.zeros(10)
    gamma = test_mdp.gamma
    convergence_rates = []
    errors = []
    
    for _ in range(20):  # 运行20次迭代
        V_next = np.zeros(10)
        
        for s in range(10):
            q_values = []
            for a in test_mdp.actions[s]:
                expected_cost = test_mdp.c[(s, a)] + gamma * np.dot(test_mdp.P[(s, a)], V)
                q_values.append(expected_cost)
            
            V_next[s] = min(q_values)
        
        # 计算误差
        error = np.max(np.abs(V_next - optimal_values))
        errors.append(error)
        
        # 更新值函数
        V = V_next.copy()
        
        # 如果有上一次误差，计算收敛速率
        if len(errors) > 1:
            rate = errors[-1] / errors[-2] if errors[-2] > 0 else 0
            convergence_rates.append(rate)
    
    # 绘制收敛速率与理论上界gamma的比较
    plt.figure(figsize=(8, 5))
    iterations = range(1, len(convergence_rates) + 1)
    plt.plot(iterations, convergence_rates, 'o-', label='Actual Convergence Rate')
    plt.axhline(y=gamma, color='r', linestyle='--', label=f'Theoretical Bound γ = {gamma}')
    plt.xlabel('Iteration')
    plt.ylabel('Convergence Rate ||y^(k+1) - y*|| / ||y^k - y*||')
    plt.title('Convergence Rate Analysis of Value Iteration')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.show()  # 在Jupyter中直接显示
    
    print(f"Average actual convergence rate: {np.mean(convergence_rates):.4f}")
    print(f"Theoretical convergence bound (γ): {gamma}")

# Jupyter Notebook魔术命令，确保图表内联显示
# get_ipython().run_line_magic('matplotlib', 'inline')

# 在Jupyter中运行此函数将直接显示所有结果
run_experiments()


# In[7]:


import numpy as np
import time

class TicTacToeMDP:
    def __init__(self, board_size=3, win_count=3):
        """初始化井字棋MDP"""
        self.board_size = board_size
        self.win_count = win_count
        
        # 定义状态值
        self.EMPTY = 0
        self.PLAYER = 1  # 优化玩家
        self.OPPONENT = 2  # 随机玩家
        
        # 游戏结果状态
        self.IN_PROGRESS = 0
        self.PLAYER_WIN = 1
        self.OPPONENT_WIN = 2
        self.DRAW = 3
        
        # 奖励和折扣设置
        self.reward_win = 1.0
        self.reward_loss = -1.0
        self.reward_draw = 0.0
        self.reward_step = -0.01  # 步数惩罚，鼓励快速获胜
        self.gamma = 0.9  # 折扣因子
        
        # 状态与价值存储
        self.states = {}  # 状态到索引的映射
        self.values = {}  # 状态到价值的映射
        self.policy = {}  # 状态到最优动作的映射
        
        # 生成所有可能的状态
        self._generate_states()
        
    def _generate_states(self):
        """生成并存储所有可能的游戏状态"""
        start_time = time.time()
        
        # 初始空棋盘
        initial_board = tuple(self.EMPTY for _ in range(self.board_size * self.board_size))
        
        # 使用BFS生成所有状态
        queue = [(initial_board, self.PLAYER)]
        visited = set()
        
        while queue:
            current_board, current_player = queue.pop(0)
            board_tuple = tuple(current_board)
            
            # 状态标识
            key = (board_tuple, current_player)
            if key in visited:
                continue
                
            visited.add(key)
            
            # 检查游戏结果
            result = self._check_game_result(current_board)
            
            if result == self.IN_PROGRESS:
                # 对每个空格子生成下一个状态
                for i in range(len(current_board)):
                    if current_board[i] == self.EMPTY:
                        new_board = list(current_board)
                        new_board[i] = current_player
                        next_player = self.OPPONENT if current_player == self.PLAYER else self.PLAYER
                        queue.append((tuple(new_board), next_player))
            
            # 存储状态和初始价值
            if key not in self.states:
                self.states[key] = len(self.states)
                
                if result == self.PLAYER_WIN:
                    self.values[key] = self.reward_win
                elif result == self.OPPONENT_WIN:
                    self.values[key] = self.reward_loss
                elif result == self.DRAW:
                    self.values[key] = self.reward_draw
                else:
                    self.values[key] = 0.0
        
        end_time = time.time()
        print(f"Generated {len(self.states)} states in {end_time - start_time:.2f} seconds")
        
    def _check_game_result(self, board):
        """检查游戏结果"""
        # 转为二维数组便于检查
        board_2d = np.array(board).reshape(self.board_size, self.board_size)
        
        # 检查玩家获胜
        if self._check_win(board_2d, self.PLAYER):
            return self.PLAYER_WIN
            
        # 检查对手获胜
        if self._check_win(board_2d, self.OPPONENT):
            return self.OPPONENT_WIN
            
        # 检查平局
        if self.EMPTY not in board:
            return self.DRAW
            
        return self.IN_PROGRESS
    
    def _check_win(self, board_2d, player):
        """检查指定玩家是否获胜"""
        # 检查行
        for i in range(self.board_size):
            for j in range(self.board_size - self.win_count + 1):
                if all(board_2d[i, j+k] == player for k in range(self.win_count)):
                    return True
        
        # 检查列
        for i in range(self.board_size - self.win_count + 1):
            for j in range(self.board_size):
                if all(board_2d[i+k, j] == player for k in range(self.win_count)):
                    return True
        
        # 检查主对角线
        for i in range(self.board_size - self.win_count + 1):
            for j in range(self.board_size - self.win_count + 1):
                if all(board_2d[i+k, j+k] == player for k in range(self.win_count)):
                    return True
        
        # 检查次对角线
        for i in range(self.board_size - self.win_count + 1):
            for j in range(self.win_count - 1, self.board_size):
                if all(board_2d[i+k, j-k] == player for k in range(self.win_count)):
                    return True
        
        return False
    
    def get_valid_actions(self, board):
        """获取当前棋盘上的所有有效动作（空格子）"""
        return [i for i, cell in enumerate(board) if cell == self.EMPTY]
    
    def next_state_distribution(self, state, action):
        """计算采取动作后的下一个状态分布"""
        board, player = state
        
        # 玩家采取动作
        new_board = list(board)
        new_board[action] = self.PLAYER
        new_board = tuple(new_board)
        
        # 检查游戏是否结束
        result = self._check_game_result(new_board)
        if result != self.IN_PROGRESS:
            return [((new_board, self.OPPONENT), 1.0)]
        
        # 随机玩家的回合
        empty_cells = self.get_valid_actions(new_board)
        num_empty = len(empty_cells)
        
        next_states = []
        for opponent_action in empty_cells:
            opponent_board = list(new_board)
            opponent_board[opponent_action] = self.OPPONENT
            opponent_board = tuple(opponent_board)
            
            # 每个动作的概率相等
            probability = 1.0 / num_empty
            next_states.append(((opponent_board, self.PLAYER), probability))
        
        return next_states
    
    def cyclic_value_iteration(self, max_iterations=1000, tolerance=1e-6):
        """执行循环价值迭代算法"""
        start_time = time.time()
        
        # 构建玩家回合状态列表
        player_states = [(state, idx) for state, idx in self.states.items() 
                         if state[1] == self.PLAYER and 
                         self._check_game_result(state[0]) == self.IN_PROGRESS]
        
        for iteration in range(max_iterations):
            max_delta = 0
            
            # 按顺序更新每个状态
            for state, _ in player_states:
                board, player = state
                old_value = self.values[state]
                
                # 计算所有动作的值
                action_values = {}
                valid_actions = self.get_valid_actions(board)
                
                for action in valid_actions:
                    next_state_dist = self.next_state_distribution(state, action)
                    
                    # 计算期望值
                    expected_value = 0
                    for next_state, prob in next_state_dist:
                        next_result = self._check_game_result(next_state[0])
                        
                        if next_result == self.PLAYER_WIN:
                            expected_value += prob * self.reward_win
                        elif next_result == self.OPPONENT_WIN:
                            expected_value += prob * self.reward_loss
                        elif next_result == self.DRAW:
                            expected_value += prob * self.reward_draw
                        else:
                            # 使用最新的值
                            expected_value += prob * (self.reward_step + self.gamma * self.values[next_state])
                    
                    action_values[action] = expected_value
                
                # 选择值最大的动作
                if action_values:
                    best_action = max(action_values, key=action_values.get)
                    self.policy[state] = best_action
                    self.values[state] = action_values[best_action]
                
                # 更新最大变化量
                max_delta = max(max_delta, abs(self.values[state] - old_value))
            
            # 打印进度
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: max delta = {max_delta}")
            
            # 检查收敛
            if max_delta < tolerance:
                print(f"CyclicVI converged after {iteration} iterations")
                break
        
        end_time = time.time()
        print(f"CyclicVI completed in {end_time - start_time:.2f} seconds")
    
    def analyze_first_moves(self):
        """分析所有可能的第一步动作的价值"""
        # 初始空棋盘状态
        initial_board = tuple(self.EMPTY for _ in range(self.board_size * self.board_size))
        initial_state = (initial_board, self.PLAYER)
        
        # 计算所有第一步动作的价值
        action_values = {}
        valid_actions = self.get_valid_actions(initial_board)
        
        for action in valid_actions:
            next_state_dist = self.next_state_distribution(initial_state, action)
            
            # 计算期望值
            expected_value = 0
            for next_state, prob in next_state_dist:
                next_result = self._check_game_result(next_state[0])
                
                if next_result == self.PLAYER_WIN:
                    expected_value += prob * self.reward_win
                elif next_result == self.OPPONENT_WIN:
                    expected_value += prob * self.reward_loss
                elif next_result == self.DRAW:
                    expected_value += prob * self.reward_draw
                else:
                    expected_value += prob * (self.reward_step + self.gamma * self.values[next_state])
            
            row, col = action // self.board_size, action % self.board_size
            action_values[(row, col)] = expected_value
        
        # 按价值排序
        sorted_actions = sorted(action_values.items(), key=lambda x: x[1], reverse=True)
        
        print("\nAll first moves ranked by value:")
        for (row, col), value in sorted_actions:
            print(f"Position ({row}, {col}) - Value: {value:.6f}")
        
        # 返回价值矩阵
        value_matrix = np.zeros((self.board_size, self.board_size))
        for (row, col), value in action_values.items():
            value_matrix[row, col] = value
        
        return value_matrix, sorted_actions[0]
    
    def print_value_matrix(self, value_matrix):
        """打印棋盘上各位置的价值"""
        print("\nValue matrix of board positions:")
        for row in range(self.board_size):
            row_str = " ".join([f"{value_matrix[row, col]:+.6f}" for col in range(self.board_size)])
            print(row_str)
    
    def analyze_optimal_strategy(self, best_move):
        """分析最优策略"""
        (best_row, best_col), best_value = best_move
        
        print(f"\nOptimal first move: Position ({best_row}, {best_col}) with value {best_value:.6f}")
        
        if best_value > 0.9:
            print("Analysis: First player with optimal strategy can almost always win")
        elif best_value > 0.5:
            print("Analysis: First player with optimal strategy has high winning probability")
        elif best_value > 0:
            print("Analysis: First player with optimal strategy has some advantage")
        elif best_value == 0:
            print("Analysis: Game likely ends in a draw with optimal play")
        else:
            print("Analysis: First player may be at disadvantage even with optimal strategy")
        
        # 检查对称性
        initial_board = tuple(self.EMPTY for _ in range(self.board_size * self.board_size))
        initial_state = (initial_board, self.PLAYER)
        symmetry_actions = self.check_value_symmetry(initial_state)
        
        if symmetry_actions:
            print("\nSymmetric optimal first moves with identical values:")
            for action, value in symmetry_actions:
                row, col = action // self.board_size, action % self.board_size
                print(f"Position ({row}, {col}) - Value: {value:.6f}")
    
    def check_value_symmetry(self, state):
        """检查最优动作是否有多个（由于对称性）"""
        board, player = state
        
        action_values = {}
        valid_actions = self.get_valid_actions(board)
        
        for action in valid_actions:
            next_state_dist = self.next_state_distribution(state, action)
            
            expected_value = 0
            for next_state, prob in next_state_dist:
                next_result = self._check_game_result(next_state[0])
                
                if next_result == self.PLAYER_WIN:
                    expected_value += prob * self.reward_win
                elif next_result == self.OPPONENT_WIN:
                    expected_value += prob * self.reward_loss
                elif next_result == self.DRAW:
                    expected_value += prob * self.reward_draw
                else:
                    expected_value += prob * (self.reward_step + self.gamma * self.values[next_state])
            
            action_values[action] = expected_value
        
        # 找出最大值
        if not action_values:
            return []
            
        max_value = max(action_values.values())
        
        # 找出所有达到最大值的动作
        symmetric_actions = [(action, value) for action, value in action_values.items() 
                            if abs(value - max_value) < 1e-6]
        
        return symmetric_actions if len(symmetric_actions) > 1 else []

def analyze_tic_tac_toe(board_size=3, win_count=3):
    """分析井字棋游戏"""
    print(f"Analyzing {board_size}x{board_size} Tic-Tac-Toe (win with {win_count} in a row)...")
    mdp = TicTacToeMDP(board_size=board_size, win_count=win_count)
    
    # 执行循环价值迭代
    print("\nRunning Cyclic Value Iteration...")
    mdp.cyclic_value_iteration()
    
    # 分析所有可能的第一步动作
    value_matrix, best_move = mdp.analyze_first_moves()
    mdp.print_value_matrix(value_matrix)
    
    # 分析最优策略
    mdp.analyze_optimal_strategy(best_move)
    
    return mdp

# 运行分析
if __name__ == "__main__":
    # 分析3x3井字棋
    mdp_3x3 = analyze_tic_tac_toe(board_size=3, win_count=3)
    
    # 取消下面的注释以分析4x4井字棋
    # mdp_4x4 = analyze_tic_tac_toe(board_size=4, win_count=4)


# In[8]:


import numpy as np

class TicTacToe:
    def __init__(self):
        # 0: empty, 1: player X, -1: player O
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player X starts
        self.game_over = False
        self.winner = None
        
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.get_state()
        
    def get_state(self):
        # Convert board to a string representation
        return str(self.board.flatten().tolist())
        
    def get_valid_actions(self):
        if self.game_over:
            return []
        # Return indices of empty cells
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def is_terminal(self):
        # Check rows
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                return True, np.sum(self.board[i, :]) // 3
                
        # Check columns
        for i in range(3):
            if abs(np.sum(self.board[:, i])) == 3:
                return True, np.sum(self.board[:, i]) // 3
                
        # Check diagonals
        if abs(np.sum(np.diag(self.board))) == 3:
            return True, np.sum(np.diag(self.board)) // 3
            
        if abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            return True, np.sum(np.diag(np.fliplr(self.board))) // 3
            
        # Check if board is full (draw)
        if len(self.get_valid_actions()) == 0:
            return True, 0
            
        return False, None
    
    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True
            
        i, j = action
        if self.board[i, j] != 0:
            return self.get_state(), 0, False  # Invalid move
            
        self.board[i, j] = self.current_player
        self.game_over, self.winner = self.is_terminal()
        
        reward = 0
        if self.game_over:
            if self.winner == 1:  # Player X wins
                reward = 1
            elif self.winner == -1:  # Player O wins
                reward = -1
                
        self.current_player = -self.current_player  # Switch player
        return self.get_state(), reward, self.game_over


def cyclic_vi_for_tictactoe():
    env = TicTacToe()
    
    # Initialize value function for all possible states
    value_function = {}
    all_states = generate_all_states()
    
    # Initialize values for all states
    for state in all_states:
        value_function[state] = 0
    
    # Set terminal state values
    for state in all_states:
        board = np.array(eval(state)).reshape(3, 3)
        env.board = board
        is_terminal, winner = env.is_terminal()
        if is_terminal:
            if winner == 1:  # X wins
                value_function[state] = 1
            elif winner == -1:  # O wins
                value_function[state] = -1
            else:  # Draw
                value_function[state] = 0
    
    gamma = 1.0  # Discount factor (1.0 for deterministic games)
    max_iterations = 100
    convergence_threshold = 1e-6
    
    # Cyclic Value Iteration
    for iteration in range(max_iterations):
        max_delta = 0
        
        # Cyclic update through all states
        for state in all_states:
            if state in value_function and is_terminal_state(state):
                continue
                
            board = np.array(eval(state)).reshape(3, 3)
            env.board = board
            
            # Determine current player
            num_x = np.sum(board == 1)
            num_o = np.sum(board == -1)
            if num_x > num_o:
                env.current_player = -1  # O's turn
            else:
                env.current_player = 1   # X's turn
            
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                continue
                
            current_player = env.current_player
            
            if current_player == 1:  # Max player (X)
                max_value = float('-inf')
                for action in valid_actions:
                    env.board = board.copy()
                    env.current_player = current_player
                    next_state, reward, done = env.step(action)
                    max_value = max(max_value, reward + gamma * value_function.get(next_state, 0))
                new_value = max_value
            else:  # Min player (O)
                min_value = float('inf')
                for action in valid_actions:
                    env.board = board.copy()
                    env.current_player = current_player
                    next_state, reward, done = env.step(action)
                    min_value = min(min_value, reward + gamma * value_function.get(next_state, 0))
                new_value = min_value
            
            # Calculate delta and update immediately (cyclic update)
            old_value = value_function[state]
            value_function[state] = new_value
            max_delta = max(max_delta, abs(old_value - new_value))
        
        # Check for convergence
        if max_delta < convergence_threshold:
            print(f"Converged after {iteration+1} iterations")
            break
    
    return value_function


def generate_all_states():
    """Generate all possible valid tic-tac-toe states."""
    def is_valid_board(board):
        # Check if the board is valid (X and O counts make sense)
        num_x = np.sum(board == 1)
        num_o = np.sum(board == -1)
        return num_x == num_o or num_x == num_o + 1
    
    states = []
    # Generate all possible board configurations
    for config in range(3**9):
        board = np.zeros(9, dtype=int)
        temp_config = config
        for i in range(9):
            board[i] = (temp_config % 3) - 1  # Convert to -1, 0, 1
            temp_config //= 3
        
        if is_valid_board(board):
            states.append(str(board.tolist()))
    
    return states


def is_terminal_state(state):
    """Check if a state is terminal."""
    board = np.array(eval(state)).reshape(3, 3)
    
    # Check rows, columns, and diagonals
    for i in range(3):
        if abs(np.sum(board[i, :])) == 3:
            return True
        if abs(np.sum(board[:, i])) == 3:
            return True
    
    if abs(np.sum(np.diag(board))) == 3:
        return True
    if abs(np.sum(np.diag(np.fliplr(board)))) == 3:
        return True
    
    # Check if board is full
    if np.sum(board == 0) == 0:
        return True
    
    return False


def get_optimal_policy(value_function):
    """Extract optimal policy from value function."""
    env = TicTacToe()
    policy = {}
    
    for state, value in value_function.items():
        if is_terminal_state(state):
            continue
            
        board = np.array(eval(state)).reshape(3, 3)
        env.board = board
        
        # Determine current player
        num_x = np.sum(board == 1)
        num_o = np.sum(board == -1)
        if num_x > num_o:
            env.current_player = -1  # O's turn
        else:
            env.current_player = 1   # X's turn
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            continue
            
        current_player = env.current_player
        best_action = None
        
        if current_player == 1:  # Max player (X)
            best_value = float('-inf')
            for action in valid_actions:
                env.board = board.copy()
                env.current_player = current_player
                next_state, reward, done = env.step(action)
                action_value = reward + value_function.get(next_state, 0)
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
        else:  # Min player (O)
            best_value = float('inf')
            for action in valid_actions:
                env.board = board.copy()
                env.current_player = current_player
                next_state, reward, done = env.step(action)
                action_value = reward + value_function.get(next_state, 0)
                if action_value < best_value:
                    best_value = action_value
                    best_action = action
        
        policy[state] = best_action
    
    return policy


def print_state_values(value_function):
    """Print some interesting state values."""
    # Initial state
    initial_state = str([0, 0, 0, 0, 0, 0, 0, 0, 0])
    print(f"Value of initial state: {value_function.get(initial_state, 'unknown')}")
    
    # Some example states
    examples = [
        # X in center
        str([0, 0, 0, 0, 1, 0, 0, 0, 0]),
        # X in corner, O in center
        str([1, 0, 0, 0, -1, 0, 0, 0, 0]),
        # X in corner, O in opposite corner
        str([1, 0, 0, 0, 0, 0, 0, 0, -1])
    ]
    
    for i, example in enumerate(examples):
        print(f"Example {i+1} value: {value_function.get(example, 'unknown')}")


# Main execution
if __name__ == "__main__":
    # Run the Cyclic VI algorithm
    value_function = cyclic_vi_for_tictactoe()
    
    # Print some state values
    print_state_values(value_function)
    
    # Get optimal policy
    policy = get_optimal_policy(value_function)
    
    # Print optimal first move for X
    initial_state = str([0, 0, 0, 0, 0, 0, 0, 0, 0])
    best_first_move = policy.get(initial_state)
    print(f"Optimal first move for X: {best_first_move}")


# In[1]:


import numpy as np
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        # 初始状态为空棋盘，由玩家X开始
        self.initial_state = ('.........',  'X')  # 9个点表示棋盘，最后一位表示下一个玩家
        self.states = {}  # 保存所有可能的状态
        self.values = {}  # 保存每个状态的价值
        self.policy = {}  # 保存每个状态的最优动作
        self.gamma = 1.0  # 折扣因子，井字棋中通常使用1

    def get_actions(self, state):
        """返回在当前状态下可行的动作列表"""
        board, player = state
        actions = []
        for i in range(9):
            if board[i] == '.':
                actions.append(i)
        return actions

    def get_next_state(self, state, action):
        """执行动作后的下一个状态"""
        board, player = state
        new_board = board[:action] + player + board[action+1:]
        next_player = 'O' if player == 'X' else 'X'
        return (new_board, next_player)

    def is_terminal(self, state):
        """判断是否为终止状态（有人赢或平局）"""
        board, _ = state
        # 检查行、列、对角线是否有人赢
        win_positions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 行
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 列
            (0, 4, 8), (2, 4, 6)              # 对角线
        ]
        
        for pos in win_positions:
            if board[pos[0]] != '.' and board[pos[0]] == board[pos[1]] == board[pos[2]]:
                return True
        
        # 检查是否平局（棋盘已满）
        if '.' not in board:
            return True
            
        return False

    def get_reward(self, state):
        """获取终止状态的奖励"""
        board, player = state
        # 检查是否有人赢
        win_positions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 行
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 列
            (0, 4, 8), (2, 4, 6)              # 对角线
        ]
        
        for pos in win_positions:
            if board[pos[0]] != '.' and board[pos[0]] == board[pos[1]] == board[pos[2]]:
                winner = board[pos[0]]
                # 对于当前玩家，如果赢了得1，输了得-1
                return 1 if winner == player else -1
        
        # 平局
        return 0

    def generate_all_states(self):
        """生成并保存所有可能的状态"""
        self.generate_states_recursive(self.initial_state)
        print(f"Total states: {len(self.states)}")
        
    def generate_states_recursive(self, state):
        """递归生成所有可能的状态"""
        if state in self.states:
            return
            
        self.states[state] = True
        
        if self.is_terminal(state):
            # 终止状态的价值就是奖励
            self.values[state] = self.get_reward(state)
            return
            
        # 初始价值为0
        self.values[state] = 0
        
        for action in self.get_actions(state):
            next_state = self.get_next_state(state, action)
            self.generate_states_recursive(next_state)

    def cyclic_value_iteration(self, max_iterations=100, epsilon=1e-6):
        """应用CyclicVI方法求解MDP"""
        # 初始化所有状态的价值
        for state in self.states:
            if not self.is_terminal(state):
                self.values[state] = 0
                
        # 迭代计算
        for k in range(max_iterations):
            max_diff = 0
            
            # 按照某种顺序遍历所有状态（这里简单使用字典顺序）
            for state in sorted(self.states.keys()):
                if self.is_terminal(state):
                    continue
                    
                old_value = self.values[state]
                
                board, player = state
                best_value = float('-inf')
                best_action = None
                
                # 更新当前状态的价值（采用极小极大策略）
                for action in self.get_actions(state):
                    next_state = self.get_next_state(state, action)
                    # 注意这里使用的是对手视角下的负价值
                    action_value = -self.values[next_state]
                    
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                
                # 更新价值和策略
                self.values[state] = best_value
                self.policy[state] = best_action
                
                # 计算价值变化最大值
                max_diff = max(max_diff, abs(old_value - best_value))
            
            print(f"Iteration {k+1}, Max diff: {max_diff}")
            
            # 收敛判断
            if max_diff < epsilon:
                print(f"Converged after {k+1} iterations")
                break
    
    def display_policy(self):
        """显示最优策略"""
        state = self.initial_state
        print("初始棋盘状态:")
        self.print_board(state)
        
        while not self.is_terminal(state) and state in self.policy:
            board, player = state
            action = self.policy[state]
            print(f"玩家 {player} 选择位置 {action // 3},{action % 3}")
            state = self.get_next_state(state, action)
            self.print_board(state)
            
        board, player = state
        if self.is_terminal(state):
            reward = self.get_reward(state)
            if reward > 0:
                print(f"玩家 {player} 赢了!")
            elif reward < 0:
                print(f"玩家 {player} 输了!")
            else:
                print("平局!")
    
    def print_board(self, state):
        """打印棋盘"""
        board, player = state
        print(f"当前玩家: {player}")
        for i in range(3):
            print(board[i*3:i*3+3].replace('.', ' '))
        print()
        
    def display_state_values(self, num_states=10):
        """显示部分状态的价值"""
        print("\n部分状态的价值:")
        count = 0
        for state, value in sorted(self.values.items(), key=lambda x: abs(x[1]), reverse=True):
            if count >= num_states:
                break
            board, player = state
            print(f"状态: {board}, 玩家: {player}, 价值: {value}")
            count += 1

def main():
    # 创建井字棋游戏
    game = TicTacToe()
    
    # 生成所有可能的状态
    game.generate_all_states()
    
    # 应用CyclicVI算法
    game.cyclic_value_iteration()
    
    # 显示一些状态的价值
    game.display_state_values()
    
    # 显示从初始状态开始的最优策略
    game.display_policy()

if __name__ == "__main__":
    main()


# In[ ]:




