#!/usr/bin/env python
# coding=utf-8
"""
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2022-07-13 22:15:46
Description:
Environment:
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.font_manager import FontProperties  # 导入字体模块


def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

#  --------------------------------保存结果--------------------------------  #


def save_rewards(*rewards, tag, path):
    for i, reward in zip(range(len(rewards)), rewards):
        np.save(path + '{}_ma_rewards_{}.npy'.format(tag, i), reward)
    print('Result saved!')


def save_costs(*costs, tag, path):
    for i, reward in zip(range(len(costs)), costs):
        np.save(path + '{}_ma_costs_{}.npy'.format(tag, i), costs)
    print('Result saved!')


def save_delays(*delays, tag, path):
    for i, delay in zip(range(len(delays)), delays):
        np.save(path + '{}_ma_delays_{}.npy'.format(tag, i), delay)
    print('Delays saved!')


def save_finished_ratio(*finished_ratios, tag, path):
    for i, finished_ratio in zip(range(len(finished_ratios)), finished_ratios):
        np.save(path + '{}_ma_finished_ratios_{}.npy'.format(tag, i), finished_ratio)
    print('Completion ratios saved!')


def save_rsu_total_cap(*rsu_total_cap, tag, path):
    for i, su_total_cap in zip(range(len(rsu_total_cap)), rsu_total_cap):
        np.save(path + '{}_rsu_total_cap_{}.npy'.format(tag, i), rsu_total_cap)
    print('Result saved!')


def save_offloading_rate(*offloading_rate, tag, path):
    for i, offloading_rate in zip(range(len(offloading_rate)), offloading_rate):
        np.save(path + '{}_rsu_offloading_rate_{}.npy'.format(tag, i), offloading_rate)
    print('Result saved!')


def save_bat_life(*bat_life, tag, path):
    for i, bat_life in zip(range(len(bat_life)), bat_life):
        np.save(path + '{}_rsu_bat_life_{}.npy'.format(tag, i), bat_life)
    print('Result saved!')

# def plot_rewards(*rewards, cfg, tag="train"):
#     plt.figure() # 创建一个图形实例，方便同时多画几个图
#     plt.rcParams['xtick.direction'] = 'in'
#     plt.rcParams['ytick.direction'] = 'in'
#     plt.xlabel('episodes', fontsize=12)
#     plt.ylabel('average rewards', fontsize=12)
#     plt.grid(True, linestyle=":", alpha=0.5)
#     if cfg.info == "flow":
#     for i, reward in zip(range(len(rewards)), rewards):
#     plt.plot(reward, label='{}'.format(cfg.change[i]))
#     elif cfg.info == "algo":
#     plt.plot(rewards[0], label='LQ')
#     plt.plot(rewards[1], label='DQN')
#     plt.plot(rewards[2], label='Our proposal')
#     else:
#     for i, reward in zip(range(len(rewards)), rewards):
#     plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
#     plt.legend(edgecolor="black", fontsize=12)
#     plt.tight_layout()
#     if cfg.save:
#     plt.savefig(cfg.plot_path + "{}_rewards_curve.pdf".format(tag))
#     plt.show()


def plot_rewards1(*rewards, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=14)
    plt.ylabel('average reward',fontsize=14)
    plt.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    plt.grid(True, linestyle=":", alpha=0.5)
    for i, reward in zip(range(len(rewards)), rewards):
        plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}ma_rewards_curve.pdf".format(tag))
    plt.show()


def plot_finish_rate1(*finish_rate, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=12)
    plt.ylabel('finished ratio',fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    for i, reward in zip(range(len(finish_rate)), finish_rate):
        plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}finish_rate_curve.pdf".format(tag))
    plt.show()


def plot_costs(*costs, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=14)
    plt.ylabel('average battery degradation cost',fontsize=14)
    plt.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    plt.grid(True, linestyle=":", alpha=0.5)
    for i, reward in zip(range(len(costs)), costs):
        plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_costs_curve.pdf".format(tag))
    plt.show()


def plot_time_delay1(*time_delay, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=14)
    plt.ylabel('average task delay',fontsize=14)
    plt.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    plt.grid(True, linestyle="--", alpha=0.5)
    for i, reward in zip(range(len(time_delay)), time_delay):
        plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_time_delay_curve.pdf".format(tag))
    plt.show()


def plot_time_delay(min_time_delay, ppo_time_delay, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=12)
    plt.ylabel('average task delay',fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.plot(ppo_time_delay, label='ppo')
    plt.plot(min_time_delay, label='min_time_delay')
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_time_delay_curve.pdf".format(tag))
    plt.show()


def plot_bat_life(*bat_life, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=12)
    plt.ylabel('average bat life(*1e+9)',fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    for i, reward in zip(range(len(bat_life)), bat_life):
        plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_bat_life_curve.pdf".format(tag))
    plt.show()


def plot_total_cap2(*rsu_total_cap, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=12)
    plt.ylabel('battery total cap(1e-5+3.59999e1)',fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    for i, reward in zip(range(len(rsu_total_cap)), rsu_total_cap):
        plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_rsu_total_cap_curve.pdf".format(tag))
    plt.show()


# def plot_bat_cap1(rsu_cap1_3, rsu_cap2_3, rsu_cap3_3, cfg, tag="train"):
#     sns.set()
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.xlabel('epsiodes')
#     plt.plot(rsu_cap1_3, label='rsu_cap1_3'.format(cfg.info))
#     plt.plot(rsu_cap2_3, label='rsu_cap2_3'.format(cfg.info))
#     plt.plot(rsu_cap3_3, label='rsu_cap3_3'.format(cfg.info))
#     plt.legend()
#     if cfg.save:
#         plt.savefig(cfg.result_path + "{}_w3_rsu_cap_curve".format(tag))
#     plt.show()


# def plot_bat_cap2(rsu_cap1_1, rsu_cap2_1, rsu_cap3_1, cfg, tag="train"):
#     sns.set()
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.xlabel('epsiodes')
#     plt.plot(rsu_cap1_1, label='rsu_cap1_1'.format(cfg.info))
#     plt.plot(rsu_cap2_1, label='rsu_cap2_1'.format(cfg.info))
#     plt.plot(rsu_cap3_1, label='rsu_cap3_1'.format(cfg.info))
#     plt.legend()
#     if cfg.save:
#         plt.savefig(cfg.result_path + "{}_w1_rsu_cap_curve".format(tag))
#     plt.show()

def chinese_font():
    """ 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体 """
    try:
        font = FontProperties(
            fname='/System/Library/Fonts/STHeiti Light.ttc', size=15)  # fname系统字体路径，此处是mac的
    except:
        font = None
    return font


def plot_finish_rate(*ma_finish_rate, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("finish rate curve on {} of {} for {}".format(cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(ma_finish_rate, label='finish ratio')
    plt.legend()
    plt.tight_layout()
    if cfg.save_fig:
        plt.savefig(cfg.result_path + "{}_finish_rate_curve.eps".format(tag), format='eps', dpi=1000)
    plt.show()


def plot_offloading_rate(*offloading_rate_v,  cfg, tag='train'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=12)
    plt.ylabel('vehicle offloading ratio',fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    for i, reward in zip(range(len(offloading_rate_v)), offloading_rate_v):
        plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_offloading_rate_v_curve.pdf".format(tag))
    plt.show()



def plot_rewards_cn(rewards, ma_rewards, cfg, tag='train'):
    """ 中文画图 """
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(cfg.env_name, cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if cfg.save:
        plt.savefig(cfg.result_path + f"{tag}_rewards_curve_cn")
    # plt.show()


# def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
#     sns.set()
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.xlabel('epsiodes')
#     plt.plot(rewards, label='rewards'.format(cfg.info))
#     plt.plot(ma_rewards, label='ma rewards'.format(cfg.info))
#     plt.legend()
#     plt.tight_layout()
#     if cfg.save:
#         plt.savefig(cfg.result_path + "{}_rewards_curve".format(tag))
#     plt.show()


def plot_rewards_algo_version(min_rewards, ppo_rewards, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=12)
    plt.ylabel('average reward',fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.plot(ppo_rewards, label='ppo')
    plt.plot(min_rewards, label='max_reward')
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_rewards_curve.pdf".format(tag))
    plt.show()


def plot_costs1(min_costs, ppo_costs, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=12)
    plt.ylabel('average battery degradation cost',fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.plot(ppo_costs, label='ppo')
    plt.plot(min_costs, label='min_costs')
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_costs_curve.pdf".format(tag))
    plt.show()


def plot_bat_life1(min_bat_life, ppo_bat_life, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes',fontsize=12)
    plt.ylabel('average bat life(*1e+9)',fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.plot(ppo_bat_life, label='ppo')
    plt.plot(min_bat_life, label='min_bat_life')
    plt.legend(edgecolor="none",fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.result_path + "{}_bat_life_curve.pdf".format(tag))
    plt.show()


# def plot_offloading_rate1(max_offloading_rate, ddqn_offloading_rate, cfg, tag="train"):
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.rcParams['xtick.direction'] = 'in'
#     plt.rcParams['ytick.direction'] = 'in'
#     plt.xlabel('episodes')
#     plt.ylabel('vehicle_offloading_ratio')
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.plot(max_offloading_rate, label='max_cap')
#     plt.plot(ddqn_offloading_rate, label='DDQN')
#     plt.legend(edgecolor="none")
#     plt.tight_layout()
#     if cfg.save:
#         plt.savefig(cfg.result_path + "{}_vehicle_offloading_rate_curve.eps".format(tag), format='eps', dpi=1000)
#     plt.show()


# def plot_total_cap1(max_total_cap, ddqn_total_cap, cfg, tag="train"):
#     plt.figure()  # 创建一个图形实例，方便同时多画几个图
#     plt.rcParams['xtick.direction'] = 'in'
#     plt.rcParams['ytick.direction'] = 'in'
#     plt.xlabel('episodes',fontsize=12)
#     plt.ylabel('total_cap',fontsize=12)
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.plot(max_total_cap, label='max_cap')
#     plt.plot(ddqn_total_cap, label='DDQN')
#     plt.legend(edgecolor="none")
#     plt.tight_layout()
#     if cfg.save:
#         plt.savefig(cfg.result_path + "{}_total_cap_curve.eps".format(tag), format='eps', dpi=1000)
#     plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


def save_results_1(dic, tag='train', path='./results'):
    """ 保存奖励 """
    for key, value in dic.items():
        np.save(path + '{}_{}.npy'.format(tag, key), value)
    print('Results saved！')


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """ 保存奖励 """
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Result saved!')


def make_dir(*paths):
    """ 创建文件夹 """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    """ 删除目录下所有空文件夹 """
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def save_args(args):
    # save parameters
    argsDict = args.__dict__
    with open(args.result_path + 'params.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    print("Parameters saved!")
