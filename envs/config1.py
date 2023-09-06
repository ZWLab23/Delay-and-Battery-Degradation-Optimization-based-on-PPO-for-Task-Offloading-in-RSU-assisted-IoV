import dataclasses
import numpy as np

@dataclasses.dataclass
class VehicularEnvConfig:
    """ 道路场景参数设置 """

    def __init__(self):
        # 道路信息
        self.road_range: int = 500  # 500m
        self.env_seed: int = 1

        # 时间信息
        self.start_time: int = 0
        self.end_time: int = 999

        # 车辆信息
        self.min_vehicle_speed: float = 25
        self.max_vehicle_speed: float = 50
        self.max_vehicle_compute_speed: float = 5
        self.min_vehicle_compute_speed: float = 1
        self.vehicle_seed: int = 1
        self.start_vehicle_number = 5
        self.vehicle_number = np.random.randint(1, 9, 4001)

        # RSU信息
        self.rsu_number: int = 3
        self.rsu_seed: int = 1
        self.max_rsu_compute_speed: float = 8
        self.min_rsu_compute_speed: float = 5

        """RSU电池相关"""
        self.rsu_ba_cap = 12
        self.rsu_cop_pow = 4
        self.rsu_tr_pow = 3
        self.min_rsu_other_energy = 200 / 3600
        self.max_rsu_other_energy = 210 / 3600
        self.rsu_ba_charge = 0.95
        self.rsu_ba_discharge = 0.95
        self.rsu_cost = 200
        # 通讯信息
        self.theta_rv: float = 1 / 5  # 3.5s/MB
        self.theta_rc: float = 8  # 6.2s/MB
        self.theta_vc: float = 6 # 15s/MB

        # 任务信息
        self.min_delay: int = 5
        self.max_delay: int = 20
        self.start_delay_threshold: int = 10
        self.start_input_size: int = 5
        self.start_output_size: float = 0.5
        self.min_input_size: float = 5
        self.max_input_size: float = 8
        self.min_output_size: float = 0.5
        self.max_output_size: float = 2
        self.start_difficulty: float = 0.9
        self.average_input_data_size: float = 5  # 5MB
        self.average_output_data_size: float = 0.5  # 512KB
        self.min_rsu_task_number: int = 1
        self.max_rsu_task_number: int = 2
        self.min_vehicle_task_number: int = 1
        self.max_vehicle_task_number: int = 2
        self.min_task_data_size: float = 3   # 5 MB
        self.max_task_data_size: float = 6  # 10 MB
        self.threshold_data_size: float = 100   # 限制参数
        """环境相关"""
        self.high = np.array([np.finfo(np.float32).max for _ in range(51)])
        self.low = np.array([0 for _ in range(51)])
        self.action_size = 35
        # 模型信息
        self.w: dict = {"weight_1": 0.2, "weight_2": 0.5, "weight_3": 0.8}
