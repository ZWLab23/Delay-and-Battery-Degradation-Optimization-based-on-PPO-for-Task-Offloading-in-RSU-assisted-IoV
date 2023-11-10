import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical


class MIN(object):
    # def __init__(self):
    #     self.total_time1 = 0
    #     self.total_time2 = 0
    #     self.total_time3 = 0
    #     self.total_time0 = 0
    #     self.total_time4 = []

    def __init__(self, n_states, n_actions, cfg):
        self.device = cfg.device
        self.n_states = n_states
        self.n_actions = n_actions


    def choose_action(self, state):
        action_space = self.n_actions
        self.total_time1 = 0
        self.total_time2 = 0
        self.total_time3 = 0
        self.total_time4 = 0
        self.total_time5 = []
        self.dod1 = 0
        self.dod2 = 0
        self.dod3 = 0
        self.reward = 0
        self.reward1 = 0
        self.reward2 = 0
        self.reward3 = 0
        self.reward4 = 0
        self.reward5 = 0
        self.reward6 = []
        self.bat_de_cost1 = 0
        self.bat_de_cost2 = 0
        self.bat_de_cost3 = 0
        self.w = 0.3


        for action in range(action_space):
            if action == 0:  # 卸载到Cloud
                upload_time = state[0] * 8
                wait_time = 0
                compute_time = 0
                download_time = state[2] * 6
                self.total_time1 = upload_time + wait_time + compute_time + download_time
                if -self.total_time1 < -state[3]:
                    self.reward1 = -21
                else:
                    self.reward1 = - ((1 - self.w) * self.total_time1)
                self.bat_de_cost1 = 0
                self.bat_de_cost2 = 0
                self.bat_de_cost3 = 0
            elif action < 2:  # 卸载到RSU
                upload_time = 0
                wait_time = state[action + 6] / state[5]
                compute_time = state[0] / (state[1] * state[5])
                download_time = 1 / 5 * state[2]
                self.total_time2 = upload_time + wait_time + compute_time + download_time
                energy_w_consum = 0
                energy_c_consum = 4 * compute_time / 3600
                energy_d_consum = 3 * download_time / 3600
                energy_o_consum = state[6]
                total_energy_consum = energy_w_consum + energy_d_consum + energy_c_consum + energy_o_consum
                self.dod1 = total_energy_consum / state[41]
                self.bat_life1 = 4980 * np.power(self.dod1, -1.98) * np.exp(-0.016 * self.dod1)
                self.bat_de_cost1 = 200 / (
                        2 * self.bat_life1 * 0.95 * 0.95)
                self.rsu_ba_cap1 = state[41] - 12 / self.bat_life1
                self.bat_life1 = self.bat_life1 / 100000000
                if -self.total_time2 < -state[3]:
                    self.reward2 = -21
                else:
                    self.reward2 = - ((1 - self.w) * self.total_time2 + self.w * 10000000 * self.bat_de_cost1)

            elif action < 3:  # 卸载到RSU
                upload_time = 0
                wait_time = state[action + 6] / state[5]
                compute_time = state[0] / (
                        state[1] * state[5])
                download_time = 1 / 5 * state[2]
                self.total_time3 = upload_time + wait_time + compute_time + download_time
                energy_w_consum = 0
                energy_c_consum = 4 * compute_time / 3600
                energy_d_consum = 3 * download_time / 3600
                energy_o_consum = state[6]
                total_energy_consum = energy_w_consum + energy_d_consum + energy_c_consum + energy_o_consum
                self.dod2 = total_energy_consum / state[42]
                self.bat_life2 = 4980 * np.power(self.dod2, -1.98) * np.exp(-0.016 * self.dod2)
                self.bat_de_cost2 = 200 / (
                        2 * self.bat_life2 * 0.95 * 0.95)
                self.rsu_ba_cap2 = state[42] - 12 / self.bat_life2
                self.bat_life2 = self.bat_life2 / 100000000
                if -self.total_time3 < -state[3]:
                    self.reward3 = -21
                else:
                    self.reward3 = - ((1 - self.w) * self.total_time3 + self.w * 10000000 * self.bat_de_cost2)

            elif action < 4:  # 卸载到RSU
                upload_time = 0
                wait_time = state[action + 6] / state[5]
                compute_time = state[0] / (
                        state[1] * state[5])
                download_time = 1 / 5 * state[2]
                self.total_time4 = upload_time + wait_time + compute_time + download_time
                energy_w_consum = 0
                energy_c_consum = 4 * compute_time / 3600
                energy_d_consum = 3 * download_time / 3600
                energy_o_consum = state[6]
                total_energy_consum = energy_w_consum + energy_d_consum + energy_c_consum + energy_o_consum
                self.dod3 = total_energy_consum / state[43]
                self.bat_life3 = 4980 * np.power(self.dod3, -1.98) * np.exp(-0.016 * self.dod3)
                self.bat_de_cost3 = 200 / (
                        2 * self.bat_life3 * 0.95 * 0.95)
                self.rsu_ba_cap3 = state[43] - 12 / self.bat_life3
                self.bat_life3 = self.bat_life3 / 100000000
                if -self.total_time4 < -state[3]:
                    self.reward4 = -21
                else:
                    self.reward4 = - ((1 - self.w) * self.total_time4 + self.w * 10000000 * self.bat_de_cost3)

            elif 4 <= action < 35:  # 卸载到Vehicle
                upload_time = 1 / 5 * state[0]
                wait_time = state[action + 6] / state[4]
                compute_time = state[0] / (
                        state[1] * state[4])
                download_time = 2 * 1 / 5 * state[2]
                self.total_time = upload_time + wait_time + compute_time + download_time
                if -self.total_time < -state[3]:
                    self.reward5 = -21
                else:
                    self.reward5 = - ((1 - self.w) * self.total_time)
                self.bat_de_cost1 = 0
                self.bat_de_cost2 = 0
                self.bat_de_cost3 = 0
                self.reward6.append(self.reward5)
                self.total_time5.append(self.total_time)
            else:
                self.reward = -21

            if self.dod1 > 0.8:
                self.reward2 = -21
            if self.dod2 > 0.8:
                self.reward3 = -21
            if self.dod3 > 0.8:
                self.reward4 = -21
        index = max(enumerate([self.reward1, self.reward2, self.reward3, self.reward4] + self.reward6), key=lambda x: x[1])[0]
        action = index
        return action
