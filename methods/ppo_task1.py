import sys,os
import gym
import torch
import datetime
from methods.ppo2 import PPO
from envs.env1 import RoadState

curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加路径到系统路径
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class PPOConfig:
    def __init__(self, weight_tag) -> None:
        self.algo = "PPO"  # 算法名称
        self.env_name = 'RoadState' # 环境名称
        self.continuous = False  # 环境是否为连续动作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 1000 # 训练的回合数
        self.test_eps = 20  # 测试的回合数
        self.batch_size = 200
        self.gamma = 0.95
        self.n_epochs = 15
        self.actor_lr = 0.00005
        self.critic_lr = 0.00005
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 128
        self.update_fre = 10  # frequency of agent update
        self.weight_tag = weight_tag
        self.algo = "PPO"  # 算法名称
        self.env_name = 'RoadState' # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.save = True # 是否保存图片
        

class TrainAndTestPPO:
    def __init__(self, cfg):
        # 创建环境和智能体
        self.cfg = cfg
        self.env = RoadState(weight_tag=self.cfg.weight_tag)
        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n  # 动作维度
        self.agent = PPO(n_states, n_actions, cfg)

    def train(self):
        print('Start training!')
        print(f'Env:{self.cfg.env_name}, Algo:{self.cfg.algo}, Device:{self.cfg.device}')
        rewards = []  # 记录所有回合的奖励
        ma_rewards = []  # 记录所有回合的滑动平均奖励
        ma_finished_rate = []
        time_delays = []
        ma_time_delay = []
        costs = []
        ma_costs = []
        bat_lifes = []
        ma_bat_life = []
        rsu1_offloading_rate = []
        rsu2_offloading_rate = []
        rsu3_offloading_rate = []
        cloud_offloading_rate = []
        vehicle_offloading_rate = []
        rsu_total_cap = []
        rsu1_cap = []
        rsu2_cap = []
        rsu3_cap = []
        steps = 0
        for i_ep in range(self.cfg.train_eps):
            ep_reward = 0  # 记录一回合内的奖励
            ep_cost = 0
            ep_time_delay = 0
            ep_bat_life = 0
            finished = 0
            ep_step = 0
            state = self.env.reset()
            offloading_position = [0, 0, 0, 0, 0]
            while True:
                # 与环境互动
                action, prob, val = self.agent.choose_action(state)  # 选择动作
                if action == 0:
                    offloading_position[0] += 1
                elif action < 2:
                    offloading_position[1] += 1
                elif action < 3:
                    offloading_position[2] += 1
                elif action < 4:
                    offloading_position[3] += 1
                elif action < 35:
                    offloading_position[4] += 1
                state_, reward, done = self.env.step(action)
                steps += 1
                ep_reward += reward
                time_delay = state[50]
                cost = state[44] + state[45] + state[46]
                bat_life = state[47] + state[48] + state[49]
                ep_bat_life += bat_life
                ep_cost += cost
                ep_time_delay += time_delay
                self.agent.memory.push(state, action, prob, val, reward, done)
                if steps % self.cfg.update_fre == 0:
                    self.agent.update()
                state = state_
                # if offloading_position[1] + offloading_position[2] + offloading_position[3] == 0:
                #     # 对于除数为0的情况，可以根据需要进行处理
                #     ep_cost = 0
                # else:
                #     ep_cost = ep_cost / (offloading_position[1] + offloading_position[2] + offloading_position[3])
                if reward > -20:
                    finished += 1
                if done:
                    break
            rewards.append(ep_reward)
            bat_lifes.append(ep_bat_life)
            costs.append(ep_cost)
            time_delays.append(ep_time_delay)
            # 求滑动平均奖励
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)
            if ma_costs:
                ma_costs.append(0.9 * ma_costs[-1] + 0.1 * ep_cost)
            else:
                ma_costs.append(ep_cost)
            if ma_bat_life:
                ma_bat_life.append(0.9 * ma_bat_life[-1] + 0.1 * ep_bat_life)
            else:
                ma_bat_life.append(ep_bat_life)
            if ma_time_delay:
                ma_time_delay.append(0.9 * ma_time_delay[-1] + 0.1 * ep_time_delay)
            else:
                ma_time_delay.append(ep_time_delay)
            cloud_offloading_rate.append(offloading_position[0] / 1000)
            rsu1_offloading_rate.append(offloading_position[1] / 1000)
            rsu2_offloading_rate.append(offloading_position[2] / 1000)
            rsu3_offloading_rate.append(offloading_position[3] / 1000)
            vehicle_offloading_rate.append(offloading_position[4] / 1000)
            rsu1_cap.append(state[41])
            rsu2_cap.append(state[42])
            rsu3_cap.append(state[43])
            rsu_total_cap.append(rsu1_cap[-1] + rsu2_cap[-1] + rsu3_cap[-1])
            finish_rate = finished / 1000
            ma_finished_rate.append(
                0.9 * ma_finished_rate[-1] + 0.1 * finish_rate) if ma_finished_rate else ma_finished_rate.append(
                finish_rate)
            if (i_ep + 1) % 1 == 0:
                print(
                    f'Episode：{i_ep + 1}/{self.cfg.train_eps}, Reward:{ep_reward:.2f},time_delay:{ep_time_delay:.10f}, cap1:{state[41]:.12f},cap2:{state[42]:.12f},cap1:{state[43]:.12f},cost:{ep_cost:.10f}, total_cap: {rsu1_cap[-1] + rsu2_cap[-1] + rsu3_cap[-1]:.10f}life1:{ep_bat_life}'
                    f' Cloud Offloading Rate：{cloud_offloading_rate[i_ep]:.5f},'
                    f' RSU1 Offloading Rate：{rsu1_offloading_rate[i_ep]:.5f},'
                    f' RSU2 Offloading Rate：{rsu2_offloading_rate[i_ep]:.5f},'
                    f' RSU3 Offloading Rate：{rsu3_offloading_rate[i_ep]:.5f},'
                    f' Vehicle Offloading Rate：{vehicle_offloading_rate[i_ep]:.5f}'
                    f' Finished:{finish_rate:.3f}')

        print('Finish training!')
        self.env.close()
        return rewards, ma_rewards, ma_time_delay, ma_finished_rate, rsu1_cap, rsu2_cap, rsu3_cap, ma_costs, ma_bat_life, vehicle_offloading_rate, rsu1_offloading_rate, rsu2_offloading_rate, rsu3_offloading_rate, rsu_total_cap
