import numpy as np
from copy import deepcopy
from utils import calculate_completion_metrics
from utils import calculate_interruption_ratio
from utils import calculate_security_level
import csv

# 设置NumPy的打印选项
np.set_printoptions(suppress=True, precision=6)
# CPU占用率
MIN_USAGE = 0.1
MAX_USAGE = 0.8

LAMBDA_MEAN = 4
# 任务大小
MIN_SIZE = 0.5  # MB*1024*8
MAX_SIZE = 9.0  # MB*1024*8 #bits
# 任务所需周期数
MAX_CYCLE = 0.397
MIN_CYCLE = 0.197  # gigacycles/Mbits
# 端节点的计算能力
MIN_RES = 20.5  # GHz*10**9 #cycles per second
MAX_RES = 28.5  # GHz*10**9 #cycles per second
# 边缘设备的计算能力
MIN_CAPABILITY_E = 31.8  # GHz*10**9 #cycles per second
MAX_CAPABILITY_E = 41.8  # GHz*10**9 #cycles per second
# 系统带宽
W_BANDWIDTH = 40
# 任务重要性
MIN_IMP = 0.1
MAX_IMP = 1
# 边缘设备的数量
NUM_EDGE = 2
# 通信信道数量
# K_CHANNEL = 10
# 定义端设备的功率范围
MAX_POWER = 24  # 10**(24/10) # 24 dBm converting 24 dB to watt(j/s)
MIN_POWER = 1  # 10**(1/10) # converting 1 dBm to watt(j/s)
# 增益
MAX_GAIN = 14  # dB no units actually but conver to linear if it is dB but not dBm
MIN_GAIN = 5  # no units actually but conver to linear if it is dB

Time_Slot = 0.15  # 一个时隙是4秒
DDL = 0.2 # 任务的最大截止时长

LOCAL_MEMORY = 300
EDGE_MEMORY = 600
MR = 0.4
TR = 0.6
DL = 0.25


class MecEnv(object):
    def __init__(self, n_agents, env_seed=None):
        if env_seed is not None:
            np.random.seed(env_seed)
        self._seed = env_seed
        self.episode_limit = 100
        self.t = 0  # 时隙
        self.n_agents = n_agents
        self.W_BANDWIDTH = W_BANDWIDTH
        self.S_power = np.zeros(self.n_agents)
        self.tasks = [[] for _ in range(n_agents)]
        self.running_task_queue = [[] for _ in range(n_agents)]
        self.waiting_task_queue = [[] for _ in range(n_agents)]
        self.S_res = np.zeros(self.n_agents)
        self.total_CPU = np.zeros(self.n_agents)
        self.lambda_mean = LAMBDA_MEAN
        self.S_gain = np.zeros(n_agents)
        # 构建running_task_queue队列和waiting_task_queue队列的一个深拷贝
        # # 这两个拷贝队列用于计算任务在两种情况（本地，边缘）下的预计完成时间，并作为奖励的一部分
        self.running_task_queue_copy = [[] for _ in range(n_agents)]
        self.waiting_task_queue_copy = [[] for _ in range(n_agents)]

        # 定义内存相关的属性
        self.S_memory = np.zeros(self.n_agents)  # 每个智能体的内存容量
        self.EDGE_S_memory = np.zeros(NUM_EDGE)  # 边缘节点的内存容量
        # 计算任务调度处理过程中违反内存规定的次数
        self.time_exceeded_memory = np.zeros(self.n_agents)
        self.EDGE_time_exceeded_memory = np.zeros(NUM_EDGE)
        # 用来模拟上传排队的过程
        self.waiting_off = [[] for _ in range(NUM_EDGE)]
        self.finish_time = []
        # 该列表用于存储一个时隙内，所有已经完成的2类型的任务，并在一个时隙结束后清空
        self.completed_c_tasks = []
        self.c_levels = np.zeros(n_agents)  # 定义每个智能体的指标，初始化为0
        self.distributed = np.zeros(n_agents)  # 定义每个智能体在分布式环境下的指标
        self.relative_distributed = np.zeros(n_agents)
        self.Interruption_frequency = np.zeros((n_agents, 3))  # 存储每个设备最近三个时隙的中断次数
        self.task_array = np.full(self.n_agents,False)
        self.task_num = 0
        self.scheduling_comparison = []
        self.current_step = 0
        self.on_time_ratios = []
        self.total_execution_times = []
        self.interruption_ratios = []
        self.total_c_level = []

        ###这一部分可以放到reset函数中
        for n in range(self.n_agents):
            # 每个智能体的计算能力，cycles/s
            self.S_res[n] = np.random.uniform(MIN_RES, MAX_RES)
            # 每个智能体生成随机功率
            self.S_power[n] = np.random.uniform(MIN_POWER, MAX_POWER)
            # 每个智能体的增益
            self.S_gain[n] = np.random.uniform(MIN_GAIN, MAX_GAIN)

        # 定义边缘节点的个数
        self.NUM_EDGE = NUM_EDGE
        # 动作（0-n）
        self.n_actions = NUM_EDGE + 1
        # 为每个边缘节点定义其执行队列
        self.EDGE_running_task_queue = [[] for _ in range(self.NUM_EDGE)]
        # 为每个边缘节点定义其等待队列
        self.EDGE_waiting_task_queue = [[] for _ in range(self.NUM_EDGE)]
        # 边缘节点的计算能力
        self.EDGE_S_res = np.zeros(self.NUM_EDGE)
        # 边缘节点的总CPU容量
        self.EDGE_total_CPU = np.zeros(self.NUM_EDGE)
        ###这一部分可以放到reset函数中
        for n in range(self.NUM_EDGE):
            self.EDGE_S_res[n] = np.random.uniform(MIN_CAPABILITY_E, MAX_CAPABILITY_E)


    def _generate_tasks(self, agent_index):
        """
        该函数单独生成了一个任务
        返回值为该任务的信息，以5元组的形式表示
        """
        size = np.random.uniform(MIN_SIZE, MAX_SIZE)
        # size = np.random.uniform(50,50)
        task_type_probability = np.random.uniform(0.5, 0.6)
        task_type = 1 if np.random.rand() < task_type_probability else 2
        S_cycle = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
        task_CPU_usage = np.random.uniform(MIN_USAGE, MAX_USAGE)
        task_DDL = DDL
        # 任务类型为2的任务重要性略低
        if task_type == 1:
            task_importance = np.random.uniform(MIN_IMP, MAX_IMP)
        else:
            task_importance = np.random.uniform(MIN_IMP, MAX_IMP - 0.3)
        time_interval = np.random.exponential(1 / self.lambda_mean)
        task = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance, agent_index, task_DDL])
        # print("生成任务信息：")
        # print(
        #     f"类型={task_type}, 大小={size}, "
        #     f"CPU请求={task_CPU_usage}, 任务所需周期={S_cycle}, 重要性={task_importance}"
        #     f"任务来自的智能体编号：{int(task[5])}")
        # print('----------------------------------------------------------------------------------------------')
        if task[0] == 1:
            # print(f"由于任务{task}的类型为：{int(task[0])},因此该任务直接在本地执行")
            self.process_tasks_for_agent(agent_index, task, is_simulate=False)
        return task, time_interval

    def get_avail_agent_actions(self, agent_id):  # 非法动作屏蔽
        # 如果智能体等待队列中没有任务，则无法执行卸载动作
        if not self.tasks[agent_id]:
            # 只有本地处理（动作0）是可用的
            avail_actions = [1] + [0] * (self.n_actions - 1)
        else:
            # 所有动作都是可用的
            avail_actions = [1] * self.n_actions

        return avail_actions

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_obs_agent(self, agent_id):
        # Device information 6
        device_info = np.array([
            self.S_res[agent_id],
            self.S_power[agent_id],
            self.S_gain[agent_id],
            self.total_CPU[agent_id],
            (self.S_memory[agent_id])/LOCAL_MEMORY,
            self.c_levels[agent_id]
        ])

        # MEC server information
        mec_info = np.concatenate([self.EDGE_S_res.flatten(),
                                   self.EDGE_total_CPU.flatten(), (self.EDGE_S_memory/EDGE_MEMORY).flatten()])

        if self.tasks[agent_id]:
            task_info = self.tasks[agent_id][0]
        else:
            task_info = np.zeros(6)
        agent_obs = np.concatenate([
            device_info,
            mec_info,
            task_info[1:5]
        ])
        return agent_obs

    def get_obs(self):
        obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return obs

    def get_state(self):
        obs = self.get_obs()
        return np.concatenate(obs)

    def calculate_task_statistics(self, agent_index, alpha=1.0, beta=1.0, lambda_param=0.5, seita=0.5):
        # 初始化统计变量
        count = 0
        time_sum = 0
        time_DDL = 0
        task_priority = 0
        interruption_current_slot = 0

        # 遍历 completed_c_tasks，检查任务是否来自指定的端设备
        for task in self.completed_c_tasks:
            if task[1] == agent_index:  # 检查任务是否来自指定的端设备
                count += 1
                time_sum += task[0]  # 累加该任务在CPU中的时间
                time_DDL += task[2]  # 累加DDL
                task_priority += task[3]  # 获取任务的重要性
        # print(f"该时隙智能体{agent_index+1}中已完成的任务的个数为：{count}")

        # 如果有任务来自该设备，计算平均时间
        if count > 0:
            # avg_time = time_sum / count
            avg_time_DDL = time_DDL / count
            avg_task_priority = task_priority / count
        else:
            avg_time = 0
            interruption_current_slot = 1  # 如果没有任务，标记为中断

        # interruption_current_slot = np.sum(~self.task_array)

        # 确保 self.task_num 不为零，防止除以零错误
        if self.task_num > 0:
            if count > 0:
                task_completion_ratio = ((avg_time_DDL) ** alpha) / \
                                        (1 + beta * (1 - avg_task_priority))
                self.c_levels[agent_index] = task_completion_ratio
            else:
                if self.task_array[agent_index] == False:
                    self.c_levels[agent_index] = -0.2
                else:
                    self.c_levels[agent_index] = -0.05

    def calculate_distributed(self, alpha=0.5, epsilon=1e-6, k=0.5):
        for i in range(self.n_agents):
            sum_sec = 0
            for j in range(self.n_agents):
                if j != i:
                    sum_sec += self.c_levels[j] - k * abs(self.c_levels[j] - self.c_levels[i])
            self.distributed[i] = alpha * (self.c_levels[i] + epsilon) \
                                  + (1 - alpha) * sum_sec / (self.n_agents - 1)

        return self.distributed


    def approximate_time(self, device_index, running_task_queue,
                         waiting_task_queue, total_CPU, new_task_info, is_local=True):
        # 创建运行队列和等待队列的拷贝，避免修改原始队列
        running_task_queue_copy = deepcopy(running_task_queue[:])
        waiting_task_queue_copy = deepcopy(waiting_task_queue[:])
        total_CPU_copy = total_CPU
        # if is_local == False:
            # print(f"实际的边缘运行队列：{self.EDGE_running_task_queue}")
            # print(f"实际的边缘等待队列：{self.EDGE_waiting_task_queue}")
            # print(f"复制的边缘队列为：running_copy{running_task_queue_copy},waiting_copy_{waiting_task_queue_copy}")
        # else:
            # print(f"实际的本地运行队列：{self.running_task_queue}")
            # print(f"实际的本地等待队列：{self.waiting_task_queue}")
            # print(f"复制的本地队列为：running_copy{running_task_queue_copy},waiting_copy_{waiting_task_queue_copy}")
        # print(f"开始计算预计完成时间")
        # print(f"原始队列为：running_{running_task_queue},waiting_{waiting_task_queue}")

        # print(f"原始内存：{total_CPU},复制内存：{total_CPU_copy}")
        # total_memory_copy = memory
        # 依次遍历运行队列中的任务，如果找到了当前的任务，那么直接计算其执行时间
        for task in running_task_queue_copy:
            if (task[0] == new_task_info).all():
                if is_local:
                    # print(f"查找结束")
                    # print(f"原始队列为：running_{running_task_queue},waiting_{waiting_task_queue}")
                    # print(f"复制队列为：running_copy{running_task_queue_copy},waiting_copy_{waiting_task_queue_copy}")
                    # print(f"原始内存：{total_CPU},复制内存：{total_CPU_copy}")
                    return self.calculate_local_time_processing(device_index, new_task_info)
                else:
                    # print(f"查找结束")
                    # print(f"原始队列为：running_{running_task_queue},waiting_{waiting_task_queue}")
                    # print(f"复制队列为：running_copy{running_task_queue_copy},waiting_copy_{waiting_task_queue_copy}")
                    # print(f"原始内存：{total_CPU},复制内存：{total_CPU_copy}")
                    return self.calculate_MEC_time_processing(device_index, new_task_info)
        # 设置当前时间为0
        current_time = 0
        # 如果新任务在等待队列中或在运行队列中
        while running_task_queue_copy:
            # 2.1 从运行队列中取出剩余执行时间最小的任务
            if running_task_queue_copy:
                min_remaining_time = min(task[1] for task in running_task_queue_copy)
            else:
                min_remaining_time = 0
            # 更新当前时间
            current_time += min_remaining_time
            # 2.2 更新所有正在运行的任务的剩余执行时间
            for task in running_task_queue_copy:
                task[1] -= min_remaining_time
            # 2.3 移除完成的任务，并更新CPU占用率
            completed_tasks = [task for task in running_task_queue_copy if task[1] <= 0]
            running_task_queue_copy = [task for task in running_task_queue_copy if task[1] > 0]
            for task in completed_tasks:
                total_CPU_copy -= task[0][2]
            # 2.4 尝试将等待队列中的任务加入到运行队列
            while waiting_task_queue_copy and total_CPU_copy + waiting_task_queue_copy[0][2] <= 1 - 1e-6:
                next_task = waiting_task_queue_copy.pop(0)
                if is_local:
                    processing_time = self.calculate_local_time_processing(device_index, next_task)
                else:
                    processing_time = self.calculate_MEC_time_processing(device_index, next_task)
                running_task_queue_copy.append([next_task, processing_time])
                total_CPU_copy += next_task[2]
            # 2.5 检查新任务是否已经在运行队列中
            if ((task[0] == new_task_info).all() for task in running_task_queue_copy):
                # 任务已在运行队列中，返回其预计完成时间
                if is_local:
                    estimated_time = current_time + self.calculate_local_time_processing(device_index, new_task_info)
                else:
                    estimated_time = current_time + self.calculate_MEC_time_processing(device_index, new_task_info)
                # print(f"查找结束")
                # print(f"原始队列为：running_{running_task_queue},waiting_{waiting_task_queue}")
                # print(f"复制队列为：running_copy{running_task_queue_copy},waiting_copy_{waiting_task_queue_copy}")
                # print(f"原始内存：{total_CPU},复制内存：{total_CPU_copy}")
                return estimated_time
            # 如果新任务不在运行队列中，继续循环

        # 如果运行和等待队列都为空，且未找到目标任务，返回惩罚值
        return 1  # 返回一个较大的值，表示任务丢失的惩

    def reset(self):
        self.t = 0
        for n in range(self.n_agents):
            self.S_res[n] = np.random.uniform(MIN_RES, MAX_RES)
            self.S_power[n] = np.random.uniform(MIN_POWER, MAX_POWER)
            self.S_gain[n] = np.random.uniform(MIN_GAIN, MAX_GAIN)

        for n in range(self.NUM_EDGE):
            self.EDGE_S_res[n] = np.random.uniform(MIN_CAPABILITY_E, MAX_CAPABILITY_E)

        self.running_task_queue = [[] for _ in range(self.n_agents)]
        self.waiting_task_queue = [[] for _ in range(self.n_agents)]
        self.total_CPU = np.zeros(self.n_agents)
        self.EDGE_running_task_queue = [[] for _ in range(self.NUM_EDGE)]
        self.EDGE_waiting_task_queue = [[] for _ in range(self.NUM_EDGE)]
        self.EDGE_total_CPU = np.zeros(self.NUM_EDGE)
        self.S_memory = np.zeros(self.n_agents)  # 每个智能体的内存容量
        self.EDGE_S_memory = np.zeros(NUM_EDGE)  # 边缘节点的内存容量
        # 用来模拟上传排队的过程
        self.waiting_off = [[] for _ in range(NUM_EDGE)]
        self.finish_time = []
        # 该列表用于存储一个时隙内，所有已经完成的2类型的任务，并在一个时隙结束后清空
        self.completed_c_tasks = []
        self.c_levels = np.zeros(self.n_agents)  # 定义每个智能体的指标，初始化为0
        self.distributed = np.zeros(self.n_agents)  # 定义每个智能体在分布式环境下的指标
        self.relative_distributed = np.zeros(self.n_agents)
        self.Interruption_frequency = np.zeros((self.n_agents, 3))  # 存储每个设备最近三个时隙的中断次数
        self.task_array = np.full(self.n_agents, False)
        self.task_num = 0

        for agent_index in range(self.n_agents):  # 重新生成任务
            # 任务生成概率设置为0.8
            # if np.random.random() < 0.8:  # 生成任务的概率
            if np.random.random() < 1:  # 测试用
                task_info, _ = self._generate_tasks(agent_index)
                if task_info[0] == 2:
                    self.tasks[agent_index].append(task_info)

        # print(f"task队列：{self.tasks}")


        for agent_index in range(self.n_agents):
            if not self.tasks[agent_index]:
                size = np.random.uniform(MIN_SIZE, MAX_SIZE)
                S_cycle = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
                task_CPU_usage = np.random.uniform(MIN_USAGE, MAX_USAGE)
                task_type = 2  # 强制将任务类型设为2
                task_importance = np.random.uniform(MIN_IMP, MAX_IMP - 0.3)
                task_DDL = DDL
                task_info = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance, agent_index,task_DDL])
                self.tasks[agent_index].append(task_info)

        # print(f"task队列：{self.tasks}")

        return self.get_obs(), self.get_state()

    def calculate_upload_time(self, n, task_info):

        task_size = task_info[1]
        DataRate = self.W_BANDWIDTH * 10 ** 6 * np.log(
            1 + self.S_power[n] * 10 ** (self.S_gain[n] / 10)) / np.log(2)  # Shannon-Hartley 定理
        # DataRate = self.W_BANDWIDTH * 10 ** 6 * np.log(
        #     1 + 1 * 10 ** (self.S_gain[n] / 10)) / np.log(2)  # Shannon-Hartley 定理
        # DataRate /= K_CHANNEL  # 除以信道数量
        Time_off = task_size * 8 * 1024 / DataRate

        return Time_off

    def calculate_local_time_processing(self, agent_index, task):  # 本地计算计算
        # task = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance])
        cycles_per_unit_time = self.S_res[agent_index] * 10 ** 9 * task[2]
        time = task[3] * task[1] * 10 ** 9 / cycles_per_unit_time
        # print(f"以智能体{agent_index + 1}的处理能力，在CPU中处理该任务的时间为: {time}")  # 打印处理时间
        return time

    # 边缘计算时间
    def calculate_MEC_time_processing(self, edge_index, task):
        cycles_per_unit_time = self.EDGE_S_res[edge_index] * 10 ** 9 * task[2]
        time = task[3] * task[1] * 10 ** 9 / cycles_per_unit_time
        return time

    def reset_agent(self, agent_index, time_interval):  # 智能体状态变化
        """
            模拟指定智能体在给定的时间间隔内，运行队列和等待队列的变化。
            """
        time = time_interval
        # 如果运行队列为空，直接返回当前状态
        if not self.running_task_queue:
            return self.running_task_queue[
                agent_index], self.waiting_task_queue, self.total_CPU, time_interval  # 这边与下面的返回值多了个time_interval

        # 对当前智能体的运行队列按剩余时间进行排序
        # self.running_task_queue[agent_index].sort(key=lambda x: x[1])  # 按剩余时间排序
        # 遍历当前智能体的运行队列
        while self.running_task_queue[agent_index]:
            self.running_task_queue[agent_index].sort(key=lambda x: x[1])
            current_task = self.running_task_queue[agent_index][0]  # 获取当前任务
            remaining_time = current_task[1] - time_interval  # 计算时延 - 时隙
            # current_task[1] = remaining_time
            # 如果剩余时间小于等于0，表示当前任务已完成
            if remaining_time <= 0:
                time_spent = current_task[1]
                current_task[0][6]-=time_spent
                self.running_task_queue[agent_index].pop(0)
                # task_info = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance, agent_index,task_DDL])
                if current_task[0][0] == 2:
                    # completed_c_tasks列表中保存的依次为：任务在CPU中执行所需时间、智能体编号、当前任务的剩余截止时间、任务重要性
                    self.completed_c_tasks.append([current_task[1], current_task[0][5],
                                                   current_task[0][6],current_task[0][4]])
                    index = int(current_task[0][5])  # 对应的端设备编号
                    self.task_array[index] = True
                self.total_CPU[agent_index] -= current_task[0][2]
                self.S_memory[agent_index] -= current_task[0][1]
                time_interval -= time_spent
                # 更新运行队列中其他任务的剩余时间
                for i in range(len(self.running_task_queue[agent_index])):
                    self.running_task_queue[agent_index][i][1] -= time_spent
                    self.running_task_queue[agent_index][i][0][6]-=time_spent
                for i in range(len(self.waiting_task_queue[agent_index])):
                    self.waiting_task_queue[agent_index][i][6] -= time_spent
                # 将等待队列中的任务加入运行队列
                # task = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance])
                while self.waiting_task_queue[agent_index]:
                    next_task = self.waiting_task_queue[agent_index][0]  # 获取等待队列中的下一个任务
                    # 检查当前CPU容量是否足够
                    if self.total_CPU[agent_index] + next_task[2] <= 1:
                        self.running_task_queue[agent_index].append([
                            next_task,
                            self.calculate_local_time_processing(agent_index, next_task)  # 计算下一个任务的处理时间
                        ])
                        # print(f"将等待队列中的任务{next_task}放入运行队列中")
                        self.total_CPU[agent_index] += next_task[2]  # 更新CPU占用
                        self.waiting_task_queue[agent_index].pop(0)  # 从等待队列中移除任务
                    else:
                        break
            else:
                # 如果当前任务未完成，更新其他任务的剩余时间
                for i in range(len(self.running_task_queue[agent_index])):
                    self.running_task_queue[agent_index][i][1] -= time_interval  # 剩余时间缩写
                    self.running_task_queue[agent_index][i][0][6] -= time_interval
                for i in range(len(self.waiting_task_queue[agent_index])):
                    self.waiting_task_queue[agent_index][i][6] -= time_interval

                for task in self.running_task_queue[agent_index]:
                    if task[0][0] == 2:
                        index = int(task[0][5])
                        if not self.task_array[index]:
                            self.task_array[index] = True
                #     self.completed_c_tasks.append([task[1], index])
                break
        # print(f"经过时间{time}后，端设备{agent_index + 1}的变化如下：")
        # for task in self.running_task_queue[agent_index]:
        #     print(f"任务{task[0]}的剩余时间更新为: {task[1]}")
        # print(f"端设备{agent_index + 1}的CPU占用率: {self.total_CPU[agent_index]}")
        # print(f"端设备{agent_index + 1}的当前内存: {self.S_memory[agent_index]}")
        # print(f"端设备{agent_index + 1}的运行任务队列: {self.running_task_queue[agent_index]}")
        # print(f"端设备{agent_index + 1}的等待任务队列: {self.waiting_task_queue[agent_index]}")

        return self.running_task_queue[agent_index], self.waiting_task_queue[agent_index], self.total_CPU[agent_index]

    def reset_mec(self, edge_index, time_interval):  # 智能体状态变化
        """
            模拟指定智能体在给定的时间间隔内，运行队列和等待队列的变化。
            """
        # 如果运行队列为空，直接返回当前状态
        time = time_interval
        # print(f"当前所需快进的时间为：{time}")
        if not self.EDGE_running_task_queue:
            return self.EDGE_running_task_queue[
                edge_index], self.EDGE_waiting_task_queue, self.EDGE_total_CPU, time_interval  # 这边与下面的返回值多了个time_interval

        # 对当前智能体的运行队列按剩余时间进行排序
        # self.EDGE_running_task_queue[edge_index].sort(key=lambda x: x[1])  # 按剩余时间排序
        while self.EDGE_running_task_queue[edge_index]:
            self.EDGE_running_task_queue[edge_index].sort(key=lambda x: x[1])
            current_task = self.EDGE_running_task_queue[edge_index][0]  # 获取当前任务
            remaining_time = current_task[1] - time_interval  # 计算时延 - 时隙
            # 如果剩余时间小于等于0，表示当前任务已完成
            if remaining_time <= 0:
                time_spent = current_task[1]
                current_task[0][6] -= time_spent
                # task = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance,agent_index])
                self.EDGE_running_task_queue[edge_index].pop(0)
                if current_task[0][0] == 2:
                    # 将类型2的任务的执行时间以及产生任务的端设备编号加入到列表中
                    self.completed_c_tasks.append([current_task[1], current_task[0][5],
                                                   current_task[0][6],current_task[0][4]])
                    index = int(current_task[0][5])# 对应的端设备编号
                    self.task_array[index] = True
                self.EDGE_total_CPU[edge_index] -= current_task[0][2]
                self.EDGE_S_memory[edge_index] -= current_task[0][1]
                time_interval -= time_spent
                # 更新运行队列中其他任务的剩余时间以及运行队列中其他任务的DDL
                for i in range(len(self.EDGE_running_task_queue[edge_index])):
                    self.EDGE_running_task_queue[edge_index][i][1] -= time_spent
                    self.EDGE_running_task_queue[edge_index][i][0][6] -= time_spent
                # 更新等待队列中所有任务的DDL
                for i in range(len(self.EDGE_waiting_task_queue[edge_index])):
                    self.EDGE_waiting_task_queue[edge_index][i][6] -= time_spent
                # 将等待队列中的任务加入运行队列
                # task = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance])
                while self.EDGE_waiting_task_queue[edge_index]:
                    next_task = self.EDGE_waiting_task_queue[edge_index][0]  # 获取等待队列中的下一个任务
                    # 检查当前CPU容量是否足够
                    if self.EDGE_total_CPU[edge_index] + next_task[2] <= 1:
                        self.EDGE_running_task_queue[edge_index].append([
                            next_task,
                            self.calculate_MEC_time_processing(edge_index, next_task)  # 计算下一个任务的处理时间
                        ])
                        # print(f"将等待队列中的任务{next_task}放入运行队列中")
                        self.EDGE_total_CPU[edge_index] += next_task[2]  # 更新CPU占用
                        self.EDGE_waiting_task_queue[edge_index].pop(0)  # 从等待队列中移除任务
                    else:
                        break
            else:
                # 如果当前任务未完成，更新其他任务的剩余时间
                # 在一个时隙全部完成的情况下，该行代码不起作用
                for i in range(len(self.EDGE_running_task_queue[edge_index])):
                    self.EDGE_running_task_queue[edge_index][i][1] -= time_interval  # 剩余时间缩写
                    self.EDGE_running_task_queue[edge_index][i][0][6] -= time_interval
                for i in range(len(self.EDGE_waiting_task_queue[edge_index])):
                    self.EDGE_waiting_task_queue[edge_index][i][6] -= time_interval

                for task in self.EDGE_running_task_queue[edge_index]:
                    if task[0][0] == 2:
                        index = int(task[0][5])
                        if not self.task_array[index]:
                            self.task_array[index] = True
                    # self.completed_c_tasks.append([task[1], index])

                break
        # print(f"经过时间{time}后，边缘设备{edge_index + 1}的变化如下：")
        # for task in self.EDGE_running_task_queue[edge_index]:
        #     print(f"任务{task[0]}的剩余时间更新为: {task[1]}")
        # print(f"边缘设备{edge_index + 1}的CPU占用率: {self.EDGE_total_CPU[edge_index]}")
        # print(f"边缘设备{edge_index + 1}的当前内存: {self.EDGE_S_memory[edge_index]}")
        # print(f"边缘设备{edge_index + 1}的运行任务队列: {self.EDGE_running_task_queue[edge_index]}")
        # print(f"边缘设备{edge_index + 1}的等待任务队列: {self.EDGE_waiting_task_queue[edge_index]}")

        return self.EDGE_running_task_queue[edge_index], self.EDGE_waiting_task_queue[edge_index], self.EDGE_total_CPU[
            edge_index]

    def process_tasks_for_agent(self, agent_index, task_info, is_simulate=False):
        local_delay = Time_Slot  # 如果任务没有在运行队列里计算，那么处理时延默认为在等待队列里等待一个时隙
        """
        模拟智能体在当前时刻处理新任务，决定任务进入运行队列还是等待队列。
        """
        # 如果当前任务是真实要在智能体中执行，那么is_simulate为False，这时队列会发生变化
        # 但是无论是否是真实在智能体中执行，都会返回运行队列与等待队列，用于后续计算预测时间
        if is_simulate:
            # print(f"如果任务在本地执行，其设备的状态变化如下：")
            running_task_queue = deepcopy(self.running_task_queue[agent_index])
            waiting_task_queue = deepcopy(self.waiting_task_queue[agent_index])
            total_CPU = self.total_CPU[agent_index]
            total_memory = self.S_memory[agent_index]
        else:
            running_task_queue = self.running_task_queue[agent_index]
            waiting_task_queue = self.waiting_task_queue[agent_index]
            total_CPU = self.total_CPU[agent_index]
            total_memory = self.S_memory[agent_index]

        if self.S_memory[agent_index] + task_info[1] < LOCAL_MEMORY:
            total_memory += task_info[1]
            # 如果运行队列和等待队列都为空
            if not waiting_task_queue and not running_task_queue:
                running_task_queue.append([
                    task_info,
                    self.calculate_local_time_processing(agent_index, task_info)
                ])
                local_delay = self.calculate_local_time_processing(agent_index, task_info)
                # print(f"将任务{task_info}直接插入运行队列中")
                total_CPU += task_info[2]
            else:
                # 当等待队列为空且CPU占用率允许时，直接添加到运行队列
                if not waiting_task_queue and total_CPU + task_info[2] <= 1:
                    running_task_queue.append([
                        task_info,
                        self.calculate_local_time_processing(agent_index, task_info)
                    ])
                    local_delay = self.calculate_local_time_processing(agent_index, task_info)
                    # print(f"将任务{task_info}直接插入运行队列中")
                    total_CPU += task_info[2]
                elif waiting_task_queue and total_CPU + task_info[2] <= 1:
                    waiting_task_queue.append(task_info)
                    waiting_task_queue.sort(key=lambda x: -x[4])  # 按重要性排序
                    # 检查排序后的第一个任务是否是新任务
                    if (waiting_task_queue[0] == task_info).all():
                        # 新任务优先级最高，从等待队列中移除并放入运行队列
                        waiting_task_queue.pop(0)
                        running_task_queue.append([
                            task_info,
                            self.calculate_local_time_processing(agent_index, task_info)
                        ])
                        local_delay = self.calculate_local_time_processing(agent_index, task_info)
                        # print(f"将任务{task_info}插入边缘服务器{edge_index + 1}的运行队列中")
                        total_CPU += task_info[2]
                    else:
                        # 新任务没有最高优先级，留在等待队列中
                        # print(f"将任务{task_info}插入边缘服务器{edge_index + 1}的等待队列中")
                        pass
                else:
                    waiting_task_queue.append(task_info)
                    waiting_task_queue.sort(key=lambda x: -x[4])  # 按重要性排序
                    # print(f"将任务{task_info}插入等待队列中")
        else:
            self.time_exceeded_memory[agent_index] += 1

        if not is_simulate:
            self.running_task_queue[agent_index] = running_task_queue
            self.waiting_task_queue[agent_index] = waiting_task_queue
            self.total_CPU[agent_index] = total_CPU
            self.S_memory[agent_index] = total_memory
        return running_task_queue, waiting_task_queue, total_CPU, total_memory, local_delay

    def process_tasks_for_mec(self, edge_index, task_info, is_simulate=False):
        edge_delay = Time_Slot  # 一个时隙
        """
        模拟边缘服务器在当前时刻处理新任务，决定任务进入运行队列还是等待队列。
        """
        # 如果当前任务是真实要在边缘服务器中执行，那么is_simulate为False，这时队列会发生变化
        # 但是无论是否是真实在边缘服务器中执行，都会返回运行队列与等待队列，用于后续计算预测时间
        if is_simulate:
            running_task_queue = deepcopy(self.EDGE_running_task_queue[edge_index])
            waiting_task_queue = deepcopy(self.EDGE_waiting_task_queue[edge_index])
            total_CPU = self.EDGE_total_CPU[edge_index]
            total_memory = self.EDGE_S_memory[edge_index]
        else:
            running_task_queue = self.EDGE_running_task_queue[edge_index]
            waiting_task_queue = self.EDGE_waiting_task_queue[edge_index]
            total_CPU = self.EDGE_total_CPU[edge_index]
            total_memory = self.EDGE_S_memory[edge_index]

        # 首先看内存是否满足条件
        if self.EDGE_S_memory[edge_index] + task_info[1] < EDGE_MEMORY:
            total_memory += task_info[1]
            # 如果运行队列和等待队列都为空
            if not waiting_task_queue and not running_task_queue:
                running_task_queue.append([
                    task_info,
                    self.calculate_MEC_time_processing(edge_index, task_info)
                ])
                edge_delay = self.calculate_MEC_time_processing(edge_index, task_info)
                # print(f"将任务{task_info}直接插入边缘服务器{edge_index + 1}的运行队列中")
                total_CPU += task_info[2]
            else:
                # 当等待队列为空且CPU占用率允许时，直接添加到运行队列
                if not waiting_task_queue and total_CPU + task_info[2] <= 1:
                    running_task_queue.append([
                        task_info,
                        self.calculate_MEC_time_processing(edge_index, task_info)
                    ])
                    edge_delay = self.calculate_MEC_time_processing(edge_index, task_info)
                    # print(f"将任务{task_info}直接插入边缘服务器{edge_index + 1}的运行队列中")
                    total_CPU += task_info[2]
                # 如果出现等待队列不为空但是可以新来的任务可以放到CPU中的情况
                # 因为要考虑等待队列中优先级高的任务，那么就让这个任务先进入等待队列，并按照优先级进行一次排序
                # 如果排序后，该新任务的优先级最高，那么该任务占用CPU，否则该任务需等待前置任务完成CPU使用之后，再占用CPU
                elif waiting_task_queue and total_CPU + task_info[2] <= 1:
                    waiting_task_queue.append(task_info)
                    waiting_task_queue.sort(key=lambda x: -x[4])  # 按重要性排序
                    # 检查排序后的第一个任务是否是新任务
                    if (waiting_task_queue[0] == task_info).all():
                        # 新任务优先级最高，从等待队列中移除并放入运行队列
                        waiting_task_queue.pop(0)
                        running_task_queue.append([
                            task_info,
                            self.calculate_MEC_time_processing(edge_index, task_info)
                        ])
                        edge_delay = self.calculate_MEC_time_processing(edge_index, task_info)
                        # print(f"将任务{task_info}插入边缘服务器{edge_index + 1}的运行队列中")
                        total_CPU += task_info[2]
                    else:
                        # 新任务没有最高优先级，留在等待队列中
                        # print(f"将任务{task_info}插入边缘服务器{edge_index + 1}的等待队列中")
                        pass
                else:
                    waiting_task_queue.append(task_info)
                    waiting_task_queue.sort(key=lambda x: -x[4])  # 按重要性排序
                    # print(f"将任务{task_info}插入边缘服务器{edge_index + 1}的等待队列中")

        self.EDGE_time_exceeded_memory[edge_index] += 1

        if not is_simulate:
            self.EDGE_running_task_queue[edge_index] = running_task_queue
            self.EDGE_waiting_task_queue[edge_index] = waiting_task_queue
            self.EDGE_total_CPU[edge_index] = total_CPU
            self.EDGE_S_memory[edge_index] = total_memory
        # print("----------------------------------------------------------------------------------------------")
        # print(f"智能体选择将任务{task_info}在边缘服务器{edge_index + 1}处理，此时服务器变化如下：")
        # print(f"边缘服务器{edge_index + 1}的CPU占用率: {total_CPU}")
        # print(f"边缘节点{edge_index + 1}的当前内存: {total_memory}")
        # print(f"边缘服务器{edge_index + 1}的运行任务队列: {running_task_queue}")
        # print(f"边缘服务器{edge_index + 1}的等待任务队列: {waiting_task_queue}")
        # print("----------------------------------------------------------------------------------------------")
        # if is_simulate:
        #     print("边缘模拟状态：")
        #     print(f"running_{running_task_queue},waiting_{waiting_task_queue}")
        # else:
        #     print("边缘非模拟状态：")
        #     print(f"running_{running_task_queue},waiting_{waiting_task_queue}")
        return running_task_queue, waiting_task_queue, total_CPU, total_memory, edge_delay

    def process_offloading_tasks(self, edge_index, time_slot):
        # 获取当前边缘设备的任务队列
        waiting_off = self.waiting_off[edge_index]
        # print(f"在该时隙内，上传到{edge_index+1}设备的任务有")
        # print(waiting_off)
        # 按照上传时间从小到大排序任务队列
        waiting_off.sort(key=lambda x: x[1])
        if waiting_off:  # 如果队列不为空，说明有端设备卸载任务到该边缘节点
            first_task = waiting_off.pop(0)  # 获取队首任务
            task_info, time_off, time_process = first_task
            task_info[6] -= time_off # 截止时间减去上传时间，以更新一次DDL
            time_elapse = 0  # 记录设备上已经消耗的时间

            # 设备快进到第一个任务的上传时间，此时，该任务到达边缘设备
            self.reset_mec(edge_index, time_off)
            time_elapse += time_off

            # 使用process_tasks_for_mec函数来处理该任务，决定该任务是放到边缘设备的CPU中还是等待队列中
            running_task_queue, waiting_task_queue, total_CPU, memory, edge_delay = self.process_tasks_for_mec(
                edge_index, task_info, is_simulate=False)
            # running_task_queue1, waiting_task_queue1, total_CPU1, memory1, edge_delay1 = self.process_tasks_for_agent(
            #     int(task_info[5]), task_info, is_simulate=True)
            # print(f"验证所有的本地运行队列是否发生变化，所有的本地运行队列为{self.running_task_queue}")
            # 估计任务完成时间
            # print("估计在边缘执行的预计完成时间")
            device_task_delay = self.approximate_time(edge_index, running_task_queue, waiting_task_queue, total_CPU,
                                                      task_info, is_local=False)
            # print("估计在本地执行的预计完成时间")
            # if_loacl_task_delay = self.approximate_time(int(task_info[5]), running_task_queue1, waiting_task_queue1,
            #                                             total_CPU1,
            #                                             task_info, is_local=True)
            # print(f"如果该任务在本地执行，预计完成时间为：{if_loacl_task_delay}")
            # time_comparison = (if_loacl_task_delay - device_task_delay) / if_loacl_task_delay
            # self.scheduling_comparison.append(time_comparison)
            self.finish_time.append(device_task_delay+time_off)
            # print(f"边缘设备{edge_index + 1}中，任务{task_info}的估计完成时间为："
            #       f"估计完成时间{device_task_delay}+上传时间{time_off} = {device_task_delay + time_off}")
            for i in range(len(waiting_off)):
                # waiting_off队列中包含的是一个时隙内上传到某个边缘节点的任务集合
                # 假设这些任务在一个时隙内全部都可以完成上传（但不一定都能执行完成）
                # 而这个for循环遍历了所有waiting_off队列的任务，也就是这一个时隙内的全部新生成的任务
                # 所以说每个任务都可以通过approximate_time函数计算出预计完成时间。
                # 这些完成时间保存在finish_time列表里面
                # 因此，每次对finish_time列表的处理一定是处理了这一个决策时隙内的任务
                next_task = waiting_off.pop(0)
                _, next_time_off, _ = next_task
                # 计算该任务与上一个任务的时间间隔
                interval_n = next_time_off - time_off
                time_off = next_time_off

                # 快进设备状态到下一个任务的到达时间
                self.reset_mec(edge_index, interval_n)
                time_elapse += interval_n
                # 使用process_tasks_for_mec函数来处理下一个任务
                task_info = next_task[0]
                task_info[6] -= next_time_off # 截止日期减去上传时间，更新一次DDL
                running_task_queue, waiting_task_queue, total_CPU, memory, edge_delay = self.process_tasks_for_mec(
                    edge_index, task_info, is_simulate=False)
                # running_task_queue1, waiting_task_queue1, total_CPU1, memory1, edge_delay1 = self.process_tasks_for_agent(
                #     int(task_info[5]), task_info, is_simulate=True)
                # 估算完成时间
                # task = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance,agent_index])
                device_task_delay = self.approximate_time(edge_index, running_task_queue, waiting_task_queue, total_CPU,
                                                          task_info, is_local=False)
                # if_loacl_task_delay = self.approximate_time(int(task_info[5]), running_task_queue1, waiting_task_queue1,
                #                                             total_CPU1,
                #                                             task_info, is_local=True)
                # print(f"如果该任务在本地执行，预计完成时间为：{if_loacl_task_delay}")
                # time_comparison = (if_loacl_task_delay - device_task_delay) / if_loacl_task_delay
                self.finish_time.append(device_task_delay + time_elapse)
                # print(f"边缘设备{edge_index + 1}中，任务{task_info}的估计完成时间为："
                #       f"估计完成时间{device_task_delay}+上传时间{time_off} = {device_task_delay + time_off}")
                # self.scheduling_comparison.append(time_comparison)
                # print(f"边缘设备{edge_index + 1}中，任务{task_info}的估计完成时间为：{device_task_delay + time_elapse}")
            # 所有任务处理完成后，计算剩余的时隙长度
            time_last = time_slot - time_elapse
            # print(f"time_elapse = {time_elapse}")
            if time_last > 0:
                # print("-----------------------------------------------------------------------------------------")
                # print(f"time_last = {time_last}")
                self.reset_mec(edge_index, time_last)
        else:
            # 如果队列为空，则直接快进设备状态为整个时隙的时间
            self.reset_mec(edge_index, time_slot)

    # def save_to_csv(self):
    #     avg_on_time_ratio = np.mean(self.on_time_ratios)
    #     avg_execution_time = np.mean(self.total_execution_times)
    #     avg_interruption_ratio = np.mean(self.interruption_ratios)
    #     avg_security_level = np.mean(self.total_c_level)
    #     with open('task_statistics.csv', mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([avg_on_time_ratio, avg_execution_time, avg_interruption_ratio,avg_security_level])
    def process_step(self, completed_c_tasks, DDL, task_array, c_level):
        # 生成任务按时完成比例和总执行时间
        # self.current_step += 1
        on_time_ratio, total_execution_time = calculate_completion_metrics(completed_c_tasks, DDL)
        # self.on_time_ratios.append(on_time_ratio)
        # self.total_execution_times.append(total_execution_time)
        # 生成中断比例
        interruption_ratio = calculate_interruption_ratio(task_array)
        # self.interruption_ratios.append(interruption_ratio)
        # 计算任务的安全性
        total_task_security = calculate_security_level(c_level)
        # self.total_c_level.append(total_task_security)
        # 每100步计算平均值并保存到CSV文件
        # if self.current_step >= self.episode_limit:
        #     self.save_to_csv()
        #     self.reset_statistics()
        return on_time_ratio, total_execution_time, interruption_ratio, total_task_security
    # def reset_statistics(self):
    #     self.current_step = 0
    #     self.on_time_ratios = []
    #     self.total_execution_times = []
    #     self.interruption_ratios = []
    #     self.total_c_level = []






    def step(self, actions):
        """
        依次处理每个智能体的任务。
        """
        self.t += 1
        if self.t == self.episode_limit:  # 每轮交互长度
            terminated = True
        else:
            terminated = False
        info = []  # 自定义后续需要打印的信息
        # 动作：0 代表任务本地计算，1-n代表任务被卸载到第几个mec服务器计算
        actions_int = [int(a) for a in actions]
        actions_int = np.array(actions_int)
        time_reward = 0
        memory_reward = 0
        total_reward = 0

        num_agents_with_tasks = sum(1 for tasks in self.tasks if len(tasks) > 0)
        tasks_executed_edge = np.zeros(self.NUM_EDGE)  # 存储选择在边缘节点执行的任务信息，如果该边缘节点此时没有接受任务，则为0
        # print(f"action:{actions_int}")
        for agent_index, action in enumerate(actions_int):  # 计算奖励,与智能体状态更新
            self.task_num += 1
            if self.tasks[agent_index]:  # 获取任务
                task_info = self.tasks[agent_index][0]
                # action = np.random.randint(1, self.NUM_EDGE + 1)
                # action = 0
                if action == 0:  # 本地处理
                    running_task_queue, waiting_task_queue, total_CPU, memory, local_delay = self.process_tasks_for_agent(
                        agent_index, task_info, is_simulate=False)

                    task_delay = self.approximate_time(agent_index, running_task_queue, waiting_task_queue
                                                       , total_CPU, task_info, is_local=True)
                    self.finish_time.append(task_delay)
                    # print(f"端设备{agent_index+1}中，任务{task_info}的估计完成时间为：{task_delay}")
                    # print("----------------------------------------------------------------------------------------------")
                else:  # 边缘处理
                    edge_index = action - 1
                    # 计算该任务的上传时间
                    Time_off = self.calculate_upload_time(agent_index, task_info)
                    time_process = self.calculate_MEC_time_processing(edge_index, task_info)
                    task = [task_info, Time_off, time_process]
                    self.waiting_off[edge_index].append(task)
            else:
                continue

        # print(f"时隙开始前，传输队列为：{self.waiting_off}")
        for i in range(len(self.waiting_off)):
            self.process_offloading_tasks(i, Time_Slot)
        for i in range(self.n_agents):
            self.reset_agent(i, Time_Slot)
        # 计算每个节点的指标
        for i in range(self.n_agents):
            self.calculate_task_statistics(i)
        agent_time_reward = np.array(self.finish_time)

        # print(f"各个智能体获得的实时性奖励：{agent_time_reward}")
        # print(f"各个智能体获得的平均实时性奖励：{np.mean(agent_time_reward)}")
        # print(f"各个智能体获得的总实时性奖励：{np.sum(agent_time_reward)}")
        # print(f"该时隙的任务中断情况：{self.task_array}")
        # print(f"一个时隙后，传输队列为：{self.waiting_off}")
        if agent_time_reward.size == 0:
            time_reward = 0
        else:
            time_reward = -1*np.sum(agent_time_reward)
        self.calculate_distributed()
        #
        # print(f"每个智能体的指标为：{self.c_levels}")
        # print(f"智能体的平均指标奖励为：{np.mean(self.c_levels)}")
        # # print(f"各个智能体的总指标奖励为：{np.sum(self.c_levels)}")
        # print(f"task_num:{self.task_num}")
        # non_empty_count = sum(1 for sublist in self.tasks if sublist)
        # print(f"self.tasks:{non_empty_count}")
        # print(f"self.completed_c_tasks:{self.completed_c_tasks}")
        #
        # print(f"每个智能体的分布式指标为：{self.distributed}")
        # print(f"智能体的平均分布式指标奖励为：{np.mean(self.distributed)}")

        creward = np.sum(self.distributed)
        # print(f"reward:{creward}")

        # total_reward = TR * time_reward + MR * creward
        total_reward = creward

        on_time_ratio, total_execution_time, interruption_ratio, total_task_security = \
            self.process_step(self.completed_c_tasks, DDL, self.task_array, self.c_levels)

        info.append(on_time_ratio)
        info.append(total_execution_time)
        info.append(interruption_ratio)
        info.append(total_task_security)

        self.finish_time.clear()
        self.completed_c_tasks.clear()
        # self.time_exceeded_memory = np.zeros(self.n_agents)
        # self.EDGE_time_exceeded_memory = np.zeros(NUM_EDGE)
        self.task_array = np.full(self.n_agents, False)
        self.task_num = 0

        self.tasks = [[] for _ in range(self.n_agents)]  # 任务已经通过决策进入等待队列或者运行队列
        for n in range(self.n_agents):
            self.S_res[n] = np.random.uniform(MIN_RES, MAX_RES)
            self.S_power[n] = np.random.uniform(MIN_POWER, MAX_POWER)
            self.S_gain[n] = np.random.uniform(MIN_GAIN, MAX_GAIN)
        for agent_index in range(self.n_agents):  # 重新生成任务
            # 任务生成概率设置为0.8
            # if np.random.random() < 0.8:  # 生成任务的概率
            if np.random.random() < 1:  # 测试用
                task_info, _ = self._generate_tasks(agent_index)
                if task_info[0] == 2:
                    self.tasks[agent_index].append(task_info)
        # print(f"task队列：{self.tasks}")

        for agent_index in range(self.n_agents):
            if not self.tasks[agent_index]:
                size = np.random.uniform(MIN_SIZE, MAX_SIZE)
                S_cycle = np.random.uniform(MIN_CYCLE, MAX_CYCLE)
                task_CPU_usage = np.random.uniform(MIN_USAGE, MAX_USAGE)
                task_type = 2  # 强制将任务类型设为2
                task_importance = np.random.uniform(MIN_IMP, MAX_IMP - 0.3)
                task_DDL = DDL
                task_info = np.array([task_type, size, task_CPU_usage, S_cycle, task_importance, agent_index,task_DDL])
                self.tasks[agent_index].append(task_info)

        # print(f"task队列：{self.tasks}")

        return total_reward, terminated, info

    def get_env_info(self):
        return {
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs_agent(0)),
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# def test_mec_env():
#     n_agents = 16
#     env = MecEnv(n_agents, env_seed=42)
#     env.reset()
#     # actions = [1,2,1,0,1,2,1,2,1,0,1,2,1,2,1,2]
#     actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     actions = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#     actions = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2]
#     actions = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0]
#     # actions = [1,1,1,1]
#     # actions = np.random.choice(actions)
#     # actions = [actions]
#
#     for step in range(100):  # 进行10个时间步的测试
#         print(f"\nStep {step + 1} ==========================")
#         env.step(actions)
#
#
# # 运行测试函数
# test_mec_env()