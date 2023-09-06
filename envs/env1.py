
from typing import Optional, Union, List, Tuple

import gym
from gym import spaces
from gym.core import RenderFrame, ObsType
from envs.dataStruct import VehicleList, RSUList, TimeSlot, Function
from envs import config1
import numpy as np
import copy
import random



class RoadState(gym.Env):
    """updating"""

    def __init__(
            self,
            weight_tag: str,
            env_config: Optional[config1.VehicularEnvConfig] = None,
            time_slot: Optional[TimeSlot] = None,
            vehicle_list: Optional[VehicleList] = None,
            rsu_list: Optional[RSUList] = None,
            function: Optional[Function] = None
    ):
        self.config = env_config or config1.VehicularEnvConfig()
        self.w = self.config.w.get(weight_tag)
        self.timeslot = time_slot or TimeSlot(start=self.config.start_time, end=self.config.end_time)

        if function is None:
            self._function: Function = Function(
                input_size=self.config.start_input_size,
                difficulty=self.config.start_difficulty,
                output_size=self.config.start_output_size,
                delay_threshold=self.config.start_delay_threshold,
                vehicle_compute_speed=self.config.min_vehicle_compute_speed,
                rsu_compute_speed=self.config.min_rsu_compute_speed,
                energy_o_consume=self.config.min_rsu_other_energy
            )
        else:
            self._function = function

        if vehicle_list is None:
            self._vehicle_list: VehicleList = VehicleList(
                vehicle_number=self.config.start_vehicle_number,
                road_range=self.config.road_range,
                min_vehicle_speed=self.config.min_vehicle_speed,
                max_vehicle_speed=self.config.max_vehicle_speed,
                max_vehicle_compute_speed=self.config.max_vehicle_compute_speed,
                min_vehicle_compute_speed=self.config.min_vehicle_compute_speed,
                max_task_number=self.config.max_vehicle_task_number,
                min_task_number=self.config.min_vehicle_task_number,
                max_task_data_size=self.config.max_task_data_size,
                min_task_data_size=self.config.min_task_data_size,
                seed=self.config.vehicle_seed
            )
        else:
            self._vehicle_list = vehicle_list

        if rsu_list is None:
            self._rsu_list: RSUList = RSUList(
                rsu_number=self.config.rsu_number,
                max_task_number=self.config.max_rsu_task_number,
                min_task_number=self.config.min_rsu_task_number,
                max_task_data_size=self.config.max_task_data_size,
                min_task_data_size=self.config.min_task_data_size,
                max_rsu_compute_speed=self.config.max_rsu_compute_speed,
                min_rsu_compute_speed=self.config.min_rsu_compute_speed,
                rsu_ba_cap=self.config.rsu_ba_cap,
                rsu_cop_pow=self.config.rsu_cop_pow,
                rsu_tr_pow=self.config.rsu_tr_pow,
                min_rsu_other_energy=self.config.min_rsu_other_energy,
                max_rsu_other_energy=self.config.max_rsu_other_energy,
                rsu_ba_charge=self.config.rsu_ba_charge,
                rsu_ba_discharge=self.config.rsu_ba_discharge,
                rsu_cost=self.config.rsu_cost,
                seed=self.config.rsu_seed
            )
        else:
            self._rsu_list = rsu_list

        self.action_space = spaces.Discrete(self.config.action_size)
        self.observation_space = spaces.Box(low=self.config.low, high=self.config.high, dtype=np.float32)

        self.state = None
        self.seed = self.timeslot.get_now()
        self.reward = 0
        self.bat_de_cost1 = 0
        self.bat_de_cost2 = 0
        self.bat_de_cost3 = 0
        self.rsu_ba_cap1 = 12
        self.rsu_ba_cap2 = 12
        self.rsu_ba_cap3 = 12
        self.dod1 = 0
        self.dod2 = 0
        self.dod3 = 0
        self.bat_life1 = 0
        self.bat_life2 = 0
        self.bat_life3 = 0
        self.bat_de_cost1 = 0
        self.bat_de_cost2 = 0
        self.bat_de_cost3 = 0
        self.total_time = 0
        self.total_time1 = 0
        self.total_time2 = 0
        self.total_time3 = 0
        self.total_time4 = 0
        self.total_time5 = 0


    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:

        self.timeslot.reset()  # 重置时间
        self.rsu_ba_cap1 = self.config.rsu_ba_cap
        self.rsu_ba_cap2 = self.config.rsu_ba_cap
        self.rsu_ba_cap3 = self.config.rsu_ba_cap
        self._function = Function(
            input_size=self.config.start_input_size,
            difficulty=self.config.start_difficulty,
            output_size=self.config.start_output_size,
            delay_threshold=self.config.start_delay_threshold,
            vehicle_compute_speed=self.config.min_vehicle_compute_speed,
            rsu_compute_speed=self.config.min_rsu_compute_speed,
            energy_o_consume=self.config.min_rsu_other_energy
        )
        self._rsu_list = RSUList(
            rsu_number=self.config.rsu_number,
            max_task_number=self.config.max_rsu_task_number,
            min_task_number=self.config.min_rsu_task_number,
            max_task_data_size=self.config.max_task_data_size,
            min_task_data_size=self.config.min_task_data_size,
            max_rsu_compute_speed=self.config.max_rsu_compute_speed,
            min_rsu_compute_speed=self.config.min_rsu_compute_speed,
            rsu_ba_cap=self.config.rsu_ba_cap,
            rsu_cop_pow=self.config.rsu_cop_pow,
            rsu_tr_pow=self.config.rsu_tr_pow,
            min_rsu_other_energy=self.config.min_rsu_other_energy,
            max_rsu_other_energy=self.config.max_rsu_other_energy,
            rsu_ba_charge=self.config.rsu_ba_charge,
            rsu_ba_discharge=self.config.rsu_ba_discharge,
            rsu_cost=self.config.rsu_cost,
            seed=self.config.vehicle_seed
        )
        self._vehicle_list = VehicleList(
            vehicle_number=self.config.start_vehicle_number,
            road_range=self.config.road_range,
            min_vehicle_speed=self.config.min_vehicle_speed,
            max_vehicle_speed=self.config.max_vehicle_speed,
            max_vehicle_compute_speed=self.config.max_vehicle_compute_speed,
            min_vehicle_compute_speed=self.config.min_vehicle_compute_speed,
            max_task_number=self.config.max_vehicle_task_number,
            min_task_number=self.config.min_vehicle_task_number,
            max_task_data_size=self.config.max_task_data_size,
            min_task_data_size=self.config.min_task_data_size,
            seed=self.config.vehicle_seed
        )
        obs_1 = [
            self._function.get_input_size(),
            self._function.get_difficulty(),
            self._function.get_output_size(),
            self._function.get_delay_threshold(),
            self._function.get_vehicle_compute_speed(),
            self._function.get_rsu_compute_speed(),
            self._function.get_energy_o_consume()
        ]
        obs_2 = []
        for rsu in self._rsu_list.get_rsu_list():
            obs_2.append(rsu.get_sum_tasks())
        obs_3 = []
        for vehicle in self._vehicle_list.get_vehicle_list():
            obs_3.append(vehicle.get_sum_tasks())
        obs_4 = []
        for _ in range(self.config.action_size - self._vehicle_list.get_vehicle_number()
                       - self._rsu_list.get_rsu_number() - 1):
            obs_4.append(self.config.threshold_data_size)
        obs_5 = [
            float(self.rsu_ba_cap1),
            float(self.rsu_ba_cap2),
            float(self.rsu_ba_cap3),
            float(self.bat_de_cost1),
            float(self.bat_de_cost2),
            float(self.bat_de_cost3),
            float(self.bat_life1),
            float(self.bat_life2),
            float(self.bat_life3),
            float(self.total_time)
        ]
        self.state = obs_1 + obs_2 + obs_3 + obs_4 + obs_5

        return np.array(self.state)

    def _get_vehicle_number(self) -> int:
        return self._vehicle_list.get_vehicle_number()

    def _get_rsu_number(self) -> int:
        return self._rsu_list.get_rsu_number()

    def _function_generation(self):
        self.seed = self.timeslot.get_now()
        np.random.seed(self.seed)
        input_data = int(np.random.uniform(self.config.min_input_size, self.config.max_input_size))
        np.random.seed(self.seed)
        difficulty = float(np.random.randint(8, 10) / 10)
        np.random.seed(self.seed)
        output_data = float(np.random.uniform(self.config.min_output_size, self.config.max_output_size))
        delay_threshold = np.random.uniform(self.config.min_delay, self.config.max_delay)
        np.random.seed(self.seed)
        vehicle_compute_speed = float(np.random.uniform(self.config.min_vehicle_compute_speed, self.config.max_vehicle_compute_speed))
        np.random.seed(self.seed)
        rsu_compute_speed = np.random.uniform(self.config.min_rsu_compute_speed, self.config.max_rsu_compute_speed)
        np.random.seed(self.seed)
        energy_o = np.random.uniform(self.config.min_rsu_other_energy, self.config.max_rsu_other_energy)
        self._function = Function(input_data, difficulty, output_data, delay_threshold, vehicle_compute_speed, rsu_compute_speed, energy_o)
        return self._function

    def _function_allocation(self, action) -> None:
        """放置任务"""
        random_factor = random.uniform(1, 1.1)
        if action == 0:
            pass
        elif action < 4:
            if action == 1:
                self._rsu_list.get_rsu_list()[0].get_task_list().add_task_list(random_factor * self._function.get_input_size())
            if action == 2:
                self._rsu_list.get_rsu_list()[1].get_task_list().add_task_list(
                    random_factor * self._function.get_input_size())
            if action == 3:
                self._rsu_list.get_rsu_list()[2].get_task_list().add_task_list(
                    random_factor * self._function.get_input_size())
        elif action < (4 + self._vehicle_list.get_vehicle_number()):
            self._vehicle_list.get_vehicle_list()[action - 4].get_task_list().add_task_list(
                self._function.get_input_size())
        else:
            pass

    def _update_road(self, action) -> object:
        """更新道路状态"""
        # 增加时隙
        self.timeslot.add_time()
        now = self.timeslot.get_now()

        # 放置任务
        self._function_allocation(action)

        # 更新任务队列
        for vehicle in self._vehicle_list.get_vehicle_list():
            vehicle.decrease_stay_time()
            process_ability = copy.deepcopy(self.state[4])
            vehicle.get_task_list().delete_data_list(process_ability)
            vehicle.get_task_list().add_by_slot(1)

        for rsu in self._rsu_list.get_rsu_list():
            process_ability = copy.deepcopy(self.state[5])
            rsu.get_task_list().delete_data_list(process_ability)
            rsu.get_task_list().add_by_slot(1)
        #print(process_ability)
        # 判断是否要删除车辆
        self._vehicle_list.delete_out_vehicle()

        # 将满足要求停留车辆放到车辆数量队列里面
        if self.config.vehicle_number[now] == 0:
            pass
        elif self._vehicle_list.get_vehicle_number() + self.config.vehicle_number[now] > 31:
            pass
        else:
            self._vehicle_list.add_stay_vehicle(self.config.vehicle_number[now])

        return self._vehicle_list, self._rsu_list

    def get_reward(self, action):
        if action == 0:  # 卸载到Cloud
            upload_time = self.state[0] * self.config.theta_rc
            wait_time = 0
            compute_time = 0
            download_time = self.state[2] * self.config.theta_vc
            self.total_time1 = upload_time + wait_time + compute_time + download_time
            if -self.total_time1 < -self.state[3]:
                self.reward = -21
            else:
                self.reward = - ((1 - self.w) * self.total_time1)
            self.bat_de_cost1 = 0
            self.bat_de_cost2 = 0
            self.bat_de_cost3 = 0

        elif action < 2:  # 卸载到RSU
            upload_time = 0
            wait_time = self.state[action + 6] / self.state[5]
            compute_time = self.state[0] / (self.state[1] * self.state[5])
            download_time = self.config.theta_rv * self.state[2]
            self.total_time2 = upload_time + wait_time + compute_time + download_time
            energy_w_consum = 0
            energy_c_consum = self.config.rsu_cop_pow * compute_time/3600
            energy_d_consum = self.config.rsu_tr_pow * download_time/3600
            energy_o_consum = self.state[6]
            total_energy_consum = energy_w_consum + energy_d_consum + energy_c_consum +energy_o_consum
            self.dod1 = total_energy_consum / self.state[41]
            self.bat_life1 = 4980 * np.power(self.dod1, -1.98) * np.exp(-0.016 * self.dod1)
            self.bat_de_cost1 = self.config.rsu_cost / (2 * self.bat_life1 * self.config.rsu_ba_charge * self.config.rsu_ba_discharge)
            self.rsu_ba_cap1 = self.state[41] - self.config.rsu_ba_cap / self.bat_life1
            self.bat_life1 = self.bat_life1 / 100000000
            if -self.total_time2 < -self.state[3]:
                self.reward = -21
            else:
                self.reward = - ((1 - self.w) * self.total_time2 + self.w * 10000000 * self.bat_de_cost1)

        elif action < 3:  # 卸载到RSU
            upload_time = 0
            wait_time = self.state[action + 6] / self.state[5]
            compute_time = self.state[0] / (
                    self.state[1] * self.state[5])
            download_time = self.config.theta_rv * self.state[2]
            self.total_time3 = upload_time + wait_time + compute_time + download_time
            energy_w_consum = 0
            energy_c_consum = self.config.rsu_cop_pow * compute_time/3600
            energy_d_consum = self.config.rsu_tr_pow * download_time/3600
            energy_o_consum = self.state[6]
            total_energy_consum = energy_w_consum + energy_d_consum + energy_c_consum + energy_o_consum
            self.dod2 = total_energy_consum / self.state[42]
            self.bat_life2 = 4980 * np.power(self.dod2, -1.98) * np.exp(-0.016 * self.dod2)
            self.bat_de_cost2 = self.config.rsu_cost / (2 * self.bat_life2 * self.config.rsu_ba_charge * self.config.rsu_ba_discharge)
            self.rsu_ba_cap2 = self.state[42] - self.config.rsu_ba_cap / self.bat_life2
            self.bat_life2 = self.bat_life2 / 100000000
            if -self.total_time3 < -self.state[3]:
                self.reward = -21
            else:
                self.reward = - ((1 - self.w) * self.total_time3 + self.w * 10000000 * self.bat_de_cost2)

        elif action < 4:  # 卸载到RSU
            upload_time = 0
            wait_time = self.state[action + 6] / self.state[5]
            compute_time = self.state[0] / (
                    self.state[1] * self.state[5])
            download_time = self.config.theta_rv * self.state[2]
            self.total_time4 = upload_time + wait_time + compute_time + download_time
            energy_w_consum = 0
            energy_c_consum = self.config.rsu_cop_pow * compute_time/3600
            energy_d_consum = self.config.rsu_tr_pow * download_time/3600
            energy_o_consum = self.state[6]
            total_energy_consum = energy_w_consum + energy_d_consum + energy_c_consum + energy_o_consum
            self.dod3 = total_energy_consum / self.state[43]
            self.bat_life3 = 4980 * np.power(self.dod3, -1.98) * np.exp(-0.016 * self.dod3)
            self.bat_de_cost3 = self.config.rsu_cost / (2 * self.bat_life3 * self.config.rsu_ba_charge * self.config.rsu_ba_discharge)
            self.rsu_ba_cap3 = self.state[43] - self.config.rsu_ba_cap / self.bat_life3
            self.bat_life3 = self.bat_life3 / 100000000
            if -self.total_time4 < -self.state[3]:
                self.reward = -21
            else:
                self.reward = - ((1 - self.w) * self.total_time4 + self.w * 10000000 * self.bat_de_cost3)

        elif 4 <= action < (self._vehicle_list.get_vehicle_number() + 4):  # 卸载到Vehicle
            upload_time = self.config.theta_rv * self.state[0]
            wait_time = self.state[action + 6] / self.state[4]
            compute_time = self.state[0] / (
                        self.state[1] * self.state[4])
            download_time = 2 * self.config.theta_rv * self.state[2]
            self.total_time5 = upload_time + wait_time + compute_time + download_time
            if -self.total_time5 < -self.state[3]:
                self.reward = -21

            else:
                self.reward = - ((1 - self.w) * self.total_time5)
            self.bat_de_cost1 = 0
            self.bat_de_cost2 = 0
            self.bat_de_cost3 = 0
        else:
            self.reward = -21

        if self.dod1 > 0.8:
            self.reward = -21
        if self.dod2 > 0.8:
            self.reward = -21
        if self.dod3 > 0.8:
            self.reward = -21
        if action == 0:
            self.total_time = self.total_time1
        if action == 1:
            self.total_time = self.total_time2
        if action == 2:
            self.total_time = self.total_time3
        if action == 3:
            self.total_time = self.total_time4
        if 4 <= action < (self._vehicle_list.get_vehicle_number() + 4):
            self.total_time = self.total_time5
        #print(self.state[44], self.state[45], self.state[46], self.total_time, self.reward, action)
        #if action >= 4 and action < len(self._vehicle_list.get_vehicle_list()) + 4:
            #print(self._vehicle_list.get_vehicle_list()[action - 4].get_vehicle_compute_speed(), self.state[action + 3], self.total_time)
        #else:
            #print("Invalid action:", action)
        #if 0 < action < 4:
            #print(self._rsu_list.get_rsu_list()[action-1].get_rsu_other_energy())
        #else:
            #print("Invalid action:", action)
        return float(self.reward)

    def step(self, action):
        # 奖励值更新
        reward = self.get_reward(action)
        # 电池状态值更新
        bat_de_cost1 = self.bat_de_cost1
        bat_de_cost2 = self.bat_de_cost2
        bat_de_cost3 = self.bat_de_cost3
        rsu_ba_cap1 = self.rsu_ba_cap1
        rsu_ba_cap2 = self.rsu_ba_cap2
        rsu_ba_cap3 = self.rsu_ba_cap3
        dod1 = self.dod1
        dod2 = self.dod2
        dod3 = self.dod3
        # done更新
        done = self.timeslot.is_end()
        # 状态更新
        function = self._function_generation()
        self._vehicle_list, self._rsu_list = self._update_road(action)
        obs_p1 = [function.get_input_size(), function.get_difficulty(), function.get_output_size(),
                  function.get_delay_threshold(), function.get_vehicle_compute_speed(), function.get_rsu_compute_speed(),
                  function.get_energy_o_consume()]
        obs_p3 = []
        for vehicle in self._vehicle_list.get_vehicle_list():
            obs_p3.append(vehicle.get_sum_tasks())
        obs_p2 = []
        for rsu in self._rsu_list.get_rsu_list():
            obs_p2.append(rsu.get_sum_tasks())
        obs_p4 = []
        if self.config.action_size - self._vehicle_list.get_vehicle_number() - self._rsu_list.get_rsu_number() - 1 > 0:
            for i in range(self.config.action_size - self._vehicle_list.get_vehicle_number()
                           - self._rsu_list.get_rsu_number() - 1):
                obs_p4.append(self.config.threshold_data_size)
        obs_p5 = [
            float(self.rsu_ba_cap1),
            float(self.rsu_ba_cap2),
            float(self.rsu_ba_cap3),
            float(self.bat_de_cost1),
            float(self.bat_de_cost2),
            float(self.bat_de_cost3),
            float(self.bat_life1),
            float(self.bat_life2),
            float(self.bat_life3),
            float(self.total_time)
        ]
        self.state = obs_p1 + obs_p2 + obs_p3 + obs_p4 + obs_p5
        state = np.array(self.state, dtype=np.float32)
        return state, reward,  done

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def close(self):
        pass
