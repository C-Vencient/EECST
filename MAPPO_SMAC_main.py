import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_smac import MAPPO_SMAC
from env import MecEnv
import csv
import os


class Runner_MAPPO_SMAC:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = MecEnv(n_agents=16, env_seed=42)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = MAPPO_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir=r'D:\MAPPO_mec_with_no_introduction\MAPPO_mec\runs\MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        # self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps,_,_,_,_ = self.run_episode_smac(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0# 初始化评估奖励总和为0
        evaluate_on_time_ratio=0
        evaluate_total_execution_time=0
        evaluate_interruption_ratio=0
        evaluate_total_task_security=0
        # 循环指定次数进行评估
        for _ in range(self.args.evaluate_times):
            # 运行一个评估episode并获取奖励
            episode_reward, _ ,on_time_ratio,total_execution_time,\
                interruption_ratio,total_task_security = self.run_episode_smac(evaluate=True) # 可根据需要返回更多指标
            # 累加每个episode的奖励
            evaluate_reward += episode_reward
            evaluate_on_time_ratio+=on_time_ratio
            evaluate_total_execution_time+=total_execution_time
            evaluate_interruption_ratio+=interruption_ratio
            evaluate_total_task_security+=total_task_security

        # 计算平均评估奖励
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        evaluate_on_time_ratio=evaluate_on_time_ratio/self.args.evaluate_times
        evaluate_total_execution_time = evaluate_total_execution_time/self.args.evaluate_times
        evaluate_interruption_ratio = evaluate_interruption_ratio/self.args.evaluate_times
        evaluate_total_task_security = evaluate_total_task_security/self.args.evaluate_times
        # 打印当前总步数和平均评估奖励
        print("total_steps:{} \t  \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        # 将平均评估奖励记录到TensorBoard的step_rewards标签下
        # self.writer.add_scalar('step_rewards', evaluate_reward, global_step=self.total_steps) # 指标存储在tensorboard中
        # self.writer.add_scalar('step_on_time_ratio', evaluate_on_time_ratio, global_step=self.total_steps)
        # self.writer.add_scalar('step_total_execution_time', evaluate_total_execution_time, global_step=self.total_steps)
        # self.writer.add_scalar('step_interruption_ratio', evaluate_interruption_ratio, global_step=self.total_steps)
        # self.writer.add_scalar('step_total_task_security', evaluate_total_task_security, global_step=self.total_steps)
        # 定义文件名
        csv_filename = 'result.csv'
        # 检查文件是否存在
        file_exists = os.path.isfile(csv_filename)
        # 打开 CSV 文件并写入数据
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 如果文件不存在，则写入表头
            if not file_exists:
                writer.writerow(['total_steps', 'evaluate_reward', 'evaluate_on_time_ratio',
                                 'evaluate_total_execution_time', 'evaluate_interruption_ratio',
                                 'evaluate_total_task_security'])

            # 写入当前的评估数据
            writer.writerow([self.total_steps, evaluate_reward, evaluate_on_time_ratio,
                             evaluate_total_execution_time, evaluate_interruption_ratio,
                             evaluate_total_task_security])

    def run_episode_smac(self, evaluate=False):
        # 该函数的作用就是获得单个智能体的观测，获得全部智能体的状态
        # 每个智能体通过观测，获得动作
        # 通过观测，状态获得状态价值
        # 通过动作与环境进行一次交互，获得奖励
        # 如果不是评估模式，则对奖励进行处理，交互次数episode_step增加，收集轨迹放入经验回放缓冲区
        # 如果不是评估模式，则获取新的观测和状态
        episode_reward = 0
        on_time_ratio = 0
        total_execution_time = 0
        interruption_ratio = 0
        total_task_security = 0
        self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            # 打印avail_a_n的维度
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            v_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
            r, done, info = self.env.step(a_n)  # Take a step
            episode_reward += r
            on_time_ratio += info[0]# 一次交互的正点率
            total_execution_time += info[1]# 一次交互的总时延
            interruption_ratio += info[2]# 一次交互的中断率
            total_task_security += info[3]# 一次交互的所有节点总安全性
            # 这些值一共累加了100次（episode_limit=100）

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                elif args.use_reward_scaling:
                    r = self.reward_scaling(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw)

            if done:
                break

        on_time_ratio = on_time_ratio/self.args.episode_limit
        total_execution_time = total_execution_time/self.args.episode_limit
        interruption_ratio = interruption_ratio/self.args.episode_limit
        total_task_security = total_task_security/self.args.episode_limit

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)
        # 返回的指标是100次的累加和
        return episode_reward, episode_step + 1,on_time_ratio,\
            total_execution_time,interruption_ratio,total_task_security


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e4), help="Save frequency")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_agent_specific", type=float, default=False, help="Whether to use agent specific global state.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    runner = Runner_MAPPO_SMAC(args, env_name="mec", number=1, seed=0)
    runner.run()
