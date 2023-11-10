import os
import sys
import torch
import datetime
from utils import make_dir
from utils import plot_time_delay, plot_costs1, plot_rewards_algo_version
from utils import save_rewards, save_delays, save_costs, save_bat_life
from envs.config1 import VehicularEnvConfig

from methods.max_reward_algo import MINConfig, TrainAndTestMIN

from methods.ppo_task1 import PPOConfig, TrainAndTestPPO
import traceback

#  --------------------------------基础准备--------------------------------  #
current_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
traceback.print_exc()
parent_path = os.path.dirname(current_path)  # current_path的父路径
sys.path.append(parent_path)  # 将父路径添加到系统路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
current_time = datetime.datetime.now().strftime("""%Y%m%d-%H%M%S""")  # 获取当前时间


#  --------------------------------画图准备--------------------------------  #
class PlotConfig:
    """ 画图超参数 """

    def __init__(self) -> None:
        self.device = device
        self.config = VehicularEnvConfig()
        self.info = "different algo"
        self.change = list(self.config.w.values())  # 读取权重
        self.train_eps = 1000
        # 结果保存主路径
        self.main_path = parent_path + "/results/" + "/algo/" + current_time
        self.result_path = self.main_path + '/results/'  # 结果保存路径
        self.model_path = self.main_path + '/models/'  # 模型保存路径
        self.result_path = self.main_path + '/results/'
        self.plot_path = self.result_path + '/plots/'
        self.reward_path = self.result_path + '/rewards/'
        self.cost_path = self.result_path + '/costs/'
        self.delay_path = self.result_path + '/delays/'
        self.bat_life_path = self.result_path + '/bat_life/'

        self.save = True  # 是否保存图片



def traffic_comparison():
    """ 不同算法下的对比 """
    cfg_PPO = PPOConfig(weight_tag="weight_1") #将config里面的weight_1改成0.3
    cfg_min = MINConfig(weight_tag="weight_1")

    plot_cfg = PlotConfig()  # 画图参数

    ppo_mind = TrainAndTestPPO(cfg_PPO)
    min_mind = TrainAndTestMIN(cfg_min)


    # -----------------------------------训练过程----------------------------------- #
    rewards_2, ma_rewards_2, time_delays_2,  finished, rsu_cap1_2, rsu_cap2_2, rsu_cap3_2, costs2, bat_lifes_2, vehicle_offloading_rate_2, rsu1_offloading_rate_2, rsu2_offloading_rate_2, rsu3_offloading_rate_2, rsu_total_cap_2 = min_mind.train()
    rewards_3, ma_rewards_3, time_delays_3,  finished, rsu_cap1_3, rsu_cap2_3, rsu_cap3_3, costs3, bat_lifes_3, vehicle_offloading_rate_3, rsu1_offloading_rate_3, rsu2_offloading_rate_3, rsu3_offloading_rate_3, rsu_total_cap_3 = ppo_mind.train()

    # 创建保存结果和模型路径的文件夹
    make_dir( plot_cfg.model_path + '/PPO/', plot_cfg.model_path + '/min/')
    make_dir(plot_cfg.reward_path, plot_cfg.plot_path, plot_cfg.delay_path, plot_cfg.cost_path,
             plot_cfg.bat_life_path,)
    make_dir(plot_cfg.result_path + '/costs/')

    save_rewards(ma_rewards_2, ma_rewards_3, tag="train", path=plot_cfg.reward_path)
    save_costs(costs2, costs3,  tag="train", path=plot_cfg.cost_path)
    save_delays(time_delays_2, time_delays_3,  tag="train", path=plot_cfg.delay_path)
    save_bat_life(bat_lifes_2, bat_lifes_3, tag="train", path=plot_cfg.bat_life_path)




    # 画图
    plot_rewards_algo_version(ma_rewards_2, rewards_3, cfg=plot_cfg, tag="train")
    plot_costs1(costs2, costs3, cfg=plot_cfg, tag="train")
    plot_time_delay(time_delays_2, time_delays_3, cfg=plot_cfg, tag="train")



    



if __name__ == "__main__":
    traffic_comparison()


