import os
import sys
import torch
import datetime
from utils import make_dir
from utils import plot_rewards1, plot_time_delay1, plot_costs, plot_offloading_rate, plot_finish_rate1, plot_total_cap2
from utils import save_rewards, save_delays, save_costs, save_rsu_total_cap, save_finished_ratio, save_offloading_rate
from envs.config1 import VehicularEnvConfig
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
        self.info = "w"
        self.change = list(self.config.w.values())  # 读取权重
        self.train_eps = 1000
        # 结果保存主路径
        self.main_path = parent_path + "/results/" + "/weight/" + current_time
        self.model_path = self.main_path + '/models/'
        # 结果保存路径
        self.result_path = self.main_path + '/results/'
        self.plot_path = self.result_path + '/plots/'
        self.reward_path = self.result_path + '/rewards/'
        self.cost_path = self.result_path + '/costs/'
        self.delay_path = self.result_path + '/delays/'
        self.finished_ratio_path = self.result_path + '/finished_ratios/'
        self.offloading_rate_path = self.result_path + '/offloading_rate/'
        self.bat_life_path = self.result_path + '/bat_life/'
        self.total_cap_path = self.result_path + '/total_cap/'

        self.save = True  # 是否保存图片


# ---------------------不同车辆数量下的reward对比--------------------- #
def traffic_comparison():
    """ 不同权重下的对比 """
    cfg_1 = PPOConfig(weight_tag="weight_1")
    cfg_2 = PPOConfig(weight_tag="weight_2")
    cfg_3 = PPOConfig(weight_tag="weight_3")
    plot_cfg = PlotConfig()  # 画图参数
    ppo_mind_1 = TrainAndTestPPO(cfg_1)
    ppo_mind_2 = TrainAndTestPPO(cfg_2)
    ppo_mind_3 = TrainAndTestPPO(cfg_3)


    # -----------------------------------训练过程----------------------------------- #
    rewards_1, ma_rewards_1,  ma_time_delay_1,  finished_1, rsu_cap1_1, rsu_cap2_1, rsu_cap3_1, ma_costs1, ma_bat_life_1, vehicle_offloading_rate_1, rsu1_offloading_rate_1, rsu2_offloading_rate_1, rsu3_offloading_rate_1, rsu_total_cap_1 = ppo_mind_1.train()
    rewards_2, ma_rewards_2,  ma_time_delay_2,  finished_2, rsu_cap1_2, rsu_cap2_2, rsu_cap3_2, ma_costs2, ma_bat_life_2, vehicle_offloading_rate_2, rsu1_offloading_rate_2, rsu2_offloading_rate_2, rsu3_offloading_rate_2, rsu_total_cap_2 = ppo_mind_2.train()
    rewards_3, ma_rewards_3,  ma_time_delay_3,  finished_3, rsu_cap1_3, rsu_cap2_3, rsu_cap3_3, ma_costs3, ma_bat_life_3, vehicle_offloading_rate_3, rsu1_offloading_rate_3, rsu2_offloading_rate_3, rsu3_offloading_rate_3, rsu_total_cap_3 = ppo_mind_3.train()

    # 创建保存结果和模型路径的文件夹
    make_dir(plot_cfg.model_path + '/traffic_1/', plot_cfg.model_path + '/traffic_2/',
             plot_cfg.model_path + '/traffic_3/')
    make_dir(plot_cfg.reward_path, plot_cfg.plot_path, plot_cfg.delay_path, plot_cfg.cost_path,
             plot_cfg.finished_ratio_path, plot_cfg.offloading_rate_path, plot_cfg.bat_life_path, plot_cfg.total_cap_path)
    make_dir(plot_cfg.result_path + '/costs/')

    # 保存模型
    ppo_mind_1.agent.save(plot_cfg.model_path + '/traffic_1/')
    ppo_mind_2.agent.save(plot_cfg.model_path + '/traffic_2/')
    ppo_mind_3.agent.save(plot_cfg.model_path + '/traffic_3/')

    # 保存结果
    save_rewards(ma_rewards_1, ma_rewards_2, ma_rewards_3, tag="train", path=plot_cfg.reward_path)
    save_costs(ma_costs1, ma_costs2, ma_costs3, tag="train", path=plot_cfg.cost_path)
    save_delays(ma_time_delay_1, ma_time_delay_2, ma_time_delay_3, tag="train", path=plot_cfg.delay_path)
    save_finished_ratio(finished_1, finished_2, finished_3, tag="train", path=plot_cfg.finished_ratio_path)
    save_offloading_rate(vehicle_offloading_rate_1, vehicle_offloading_rate_2, vehicle_offloading_rate_3, tag="train", path=plot_cfg.offloading_rate_path)
    save_rsu_total_cap(rsu_total_cap_1, rsu_total_cap_2, rsu_total_cap_3, tag="train", path=plot_cfg.total_cap_path)

    # 画图
    plot_rewards1(ma_rewards_1, ma_rewards_2, ma_rewards_3, cfg=plot_cfg, tag="train")
    plot_costs(ma_costs1, ma_costs2, ma_costs3, cfg=plot_cfg, tag="train")
    plot_time_delay1(ma_time_delay_1, ma_time_delay_2, ma_time_delay_3, cfg=plot_cfg, tag="train")
    plot_offloading_rate(vehicle_offloading_rate_1, vehicle_offloading_rate_2, vehicle_offloading_rate_3, cfg=plot_cfg, tag="train")
    plot_total_cap2(rsu_total_cap_1, rsu_total_cap_2, rsu_total_cap_3, cfg=plot_cfg, tag="train")
    plot_finish_rate1(finished_1, finished_2, finished_3, cfg=plot_cfg, tag="train")


if __name__ == "__main__":
    traffic_comparison()


