import random
import time
import matplotlib.pyplot as plt
import numpy as np

def synchronous_value_iteration(states, transitions, gamma, tol=1e-6, max_iter=1000):
    start_time = time.time()
    iteration_count = 0
    y_k = states.copy() # 初始值向量
    errors = []

    for _ in range(max_iter):
        iteration_count += 1
        y_k_plus_1 = {}  # 下一轮的值向量y(k+1)
        max_diff = 0 # 追踪||y^(k+1) - y^k||_∞

        for i in y_k: # 遍历所有状态，分别更新一个状态i的新的最佳状态值、
            # 对于每个状态i，寻找min_{j∈A_i} {c_j + γ·p_j^T·y^k}最佳动作j和对应的状态值
            action_values = [] # 存储所有可能动作j的值：c_j + γ·p_j^T·y^k
            for j in transitions[i]: # j∈A_i，遍历状态i的所有可能动作
                value = 0 # 用于更新进行动作j、到不同下个状态的期望状态值
                # 计算c_j + γ·p_j^T·y^k
                for prob, next_state, reward in transitions[i][j]: # prob=p_j[next_state]进行动作j到不同下个状态的概率, reward=c_j
                    value += prob * (reward + gamma * y_k[next_state]) 
                    # 根据公式更新进行动作j的期望状态值。注意，这里用于更新的y_k统一还是上一次迭代的结果
                action_values.append(value)
            
            new_value = max(action_values) if action_values else 0 # 这里我们解决的是奖励最大化问题，所以取max而不是min
            y_k_plus_1[i] = new_value # 把i状态的新的最佳状态值更新到y^(k+1)（新的状态值向量，不用于此次迭代）
            
            # 更新||y^(k+1) - y^k||_∞
            diff = abs(new_value - y_k[i])
            max_diff = max(max_diff, diff)
        
        #记录此次迭代后的“error”
        errors.append(max_diff)
        # 统一更新y^k为y^(k+1)
        y_k = y_k_plus_1.copy()
        # 检查收敛：||y^(k+1) - y^k||_∞ < ε
        if max_diff < tol:
            break

    execution_time = time.time() - start_time
    return y_k, errors, iteration_count, execution_time

def random_cyclic_value_iteration(states, transitions, gamma, tol=1e-6, max_iter=1000):
    start_time = time.time()
    iteration_count = 0
    y_k = states.copy()
    errors = []

    for _ in range(max_iter):
        iteration_count += 1
        max_diff = 0
        state_keys = list(y_k.keys())
        random.shuffle(state_keys) #生成随机状态排列
        # 遍历打乱后的状态顺序，迭代同上
        for i in state_keys:
            old_value = y_k[i]
            action_values = []
            for j in transitions[i]:
                value = 0
                for prob, next_state, reward in transitions[i][j]:
                    value += prob * (reward + gamma * y_k[next_state])
                action_values.append(value)
            y_k[i] = max(action_values) if action_values else 0
            diff = abs(y_k[i] - old_value)
            max_diff = max(max_diff, diff)
        
        errors.append(max_diff)
        if max_diff < tol:
            break

    execution_time = time.time() - start_time
    return y_k, errors, iteration_count, execution_time

def generate_random_mdp(num_states, num_actions, sparsity='sparse'):
    transitions = {}
    for i in range(num_states): #对于每一个状态i
        transitions[i] = {}
        for a in range(num_actions): #每一个状态i相应的每一个动作j
            action = []
            if sparsity == 'sparse': 
                next_states = random.sample(range(num_states), k=int(0.01*num_states))
                #稀疏矩阵，所有状态中的百分之1可能作为下一状态
            else:
                next_states = random.sample(range(num_states), k=int(0.7*num_states))
                #稠密矩阵，所有状态中的百分之70都可能作为下一状态
            for ns in next_states:
                prob = random.random() # 为每个next state随机生成一个转移概率值
                reward = random.uniform(-1, 1) # 为每个next state随机生成一个奖励值
                action.append((prob, ns, reward))
            # 归一化使所有转移概率和为1
            total = sum([x[0] for x in action]) 
            action = [(p/total, ns, r) for (p, ns, r) in action]
            # 保存状态i下动作a的所有转移（元组：概率，下一状态，奖励）
            transitions[i][a] = action
    return transitions

#用于比较随机生成的统一例子下两个算法的性能
num_states = 200
num_actions = 10
gamma = 0.9
tol = 1e-6
max_iter = 1000
init_states = {i: 0.0 for i in range(num_states)}

sparse_mdp = generate_random_mdp(num_states, num_actions, 'sparse')
dense_mdp = generate_random_mdp(num_states, num_actions, 'dense')

sync_sparse, err_sync_sparse, iter_sync_sparse, time_sync_sparse = synchronous_value_iteration(init_states, sparse_mdp, gamma, tol, max_iter)
rand_sparse, err_rand_sparse, iter_rand_sparse, time_rand_sparse = random_cyclic_value_iteration(init_states, sparse_mdp, gamma, tol, max_iter)
sync_dense, err_sync_dense, iter_sync_dense, time_sync_dense = synchronous_value_iteration(init_states, dense_mdp, gamma, tol, max_iter)
rand_dense, err_rand_dense, iter_rand_dense, time_rand_dense = random_cyclic_value_iteration(init_states, dense_mdp, gamma, tol, max_iter)



# 生成不同state size的随机例子，比较不同算法、不同矩阵密度随状态增加的表现性能
state_sizes = [20, 50, 100, 200]

results = {
    "sync_sparse_time": [],
    "sync_dense_time": [],
    "random_sparse_time": [],
    "random_dense_time": [],
    "sync_sparse_iters": [],
    "sync_dense_iters": [],
    "random_sparse_iters": [],
    "random_dense_iters": [],
}

# 运行实验
for num_states in state_sizes:
    num_actions = 10
    init_states = {i: 0.0 for i in range(num_states)}

    for sparsity in ['sparse', 'dense']:
        mdp = generate_random_mdp(num_states, num_actions, sparsity=sparsity)

        y1, _, iters1, time1 = synchronous_value_iteration(init_states, mdp, gamma, tol, max_iter)
        y2, _, iters2, time2 = random_cyclic_value_iteration(init_states, mdp, gamma, tol, max_iter)

        if sparsity == 'dense':
           results["sync_dense_time"].append(time1)
           results["random_dense_time"].append(time2)
           results["sync_dense_iters"].append(iters1)
           results["random_dense_iters"].append(iters2)
        else:
           results["sync_sparse_time"].append(time1)
           results["random_sparse_time"].append(time2)
           results["sync_sparse_iters"].append(iters1)
           results["random_sparse_iters"].append(iters2)


print("Iteration Counts and Execution Times:")
print(f"Sparse Matrix - Sync:   {iter_sync_sparse} iterations, {time_sync_sparse:.4f} seconds")
print(f"Sparse Matrix - Random: {iter_rand_sparse} iterations, {time_rand_sparse:.4f} seconds")
print(f"Dense Matrix - Sync:    {iter_sync_dense} iterations, {time_sync_dense:.4f} seconds")
print(f"Dense Matrix - Random:  {iter_rand_dense} iterations, {time_rand_dense:.4f} seconds")

# 创建一个 3x2 的子图布局（3行2列），用于显示六个图
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

# 绘制稀疏 MDP 的收敛曲线（第一部分）
axs[0, 0].plot(err_sync_sparse, label='Sync')
axs[0, 0].plot(err_rand_sparse, label='Random')
axs[0, 0].set_yscale('log')
axs[0, 0].set_title('Sparse MDP: Convergence Curve')
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Max Error')
axs[0, 0].legend()

# 绘制稠密 MDP 的收敛曲线（第一部分）
axs[0, 1].plot(err_sync_dense, label='Sync')
axs[0, 1].plot(err_rand_dense, label='Random')
axs[0, 1].set_yscale('log')
axs[0, 1].set_title('Dense MDP: Convergence Curve')
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Max Error')
axs[0, 1].legend()

# 绘制迭代次数对比图（第二部分）
bar_width = 0.35
x = np.arange(2)
axs[1, 0].bar(x - bar_width / 2, [iter_sync_sparse, iter_sync_dense], bar_width, label='Sync')
axs[1, 0].bar(x + bar_width / 2, [iter_rand_sparse, iter_rand_dense], bar_width, label='Random')
axs[1, 0].set_title('Iterations Comparison')
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(['Sparse', 'Dense'])
axs[1, 0].set_ylabel('Iterations')
axs[1, 0].legend()

# 绘制执行时间对比图（第二部分）
axs[1, 1].bar(x - bar_width / 2, [time_sync_sparse, time_sync_dense], bar_width, label='Sync')
axs[1, 1].bar(x + bar_width / 2, [time_rand_sparse, time_rand_dense], bar_width, label='Random')
axs[1, 1].set_title('Execution Time Comparison')
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(['Sparse', 'Dense'])
axs[1, 1].set_ylabel('Time (s)')
axs[1, 1].legend()

# 绘制时间与输入大小的关系图（第三部分）
axs[2, 0].plot(state_sizes, results["sync_sparse_time"], label="Sync Sparse", marker='o')
axs[2, 0].plot(state_sizes, results["random_sparse_time"], label="Random Sparse", marker='o')
axs[2, 0].plot(state_sizes, results["sync_dense_time"], label="Sync Dense", marker='o')
axs[2, 0].plot(state_sizes, results["random_dense_time"], label="Random Dense", marker='o')
axs[2, 0].set_xlabel("Number of States")
axs[2, 0].set_ylabel("Execution Time (s)")
axs[2, 0].set_title("Execution Time vs Input Size")
axs[2, 0].legend()
axs[2, 0].grid(True)

# 绘制迭代次数与输入大小的关系图（第三部分）
axs[2, 1].plot(state_sizes, results["sync_sparse_iters"], label="Sync Sparse", marker='o')
axs[2, 1].plot(state_sizes, results["random_sparse_iters"], label="Random Sparse", marker='o')
axs[2, 1].plot(state_sizes, results["sync_dense_iters"], label="Sync Dense", marker='o')
axs[2, 1].plot(state_sizes, results["random_dense_iters"], label="Random Dense", marker='o')
axs[2, 1].set_xlabel("Number of States")
axs[2, 1].set_ylabel("Iterations to Converge")
axs[2, 1].set_title("Iterations vs Input Size")
axs[2, 1].legend()
axs[2, 1].grid(True)

# 确保布局紧凑
plt.tight_layout()

# 显示所有六个子图
plt.show()


