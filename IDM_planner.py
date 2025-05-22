import pandas as pd
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
import sys
import threading as td
import json
import cv2
import matplotlib.patches as mpatches
import argparse
import imageio.v2 as imageio
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Polygon as pg
from PIL import Image
from nuscenes.map_expansion.map_api import NuScenesMap
from scipy.integrate import RK45

# 設定 IDM 模型的參數
# origin: v0=30, T=1.5, a=1.0, b=2.0, s0=2.0
class IDM:
    def __init__(self, v0=5, T=1.5, a=5.0, b=2.0, s0=2.0, delta=4.0):
        self.v0 = v0  # Desired velocity (m/s)
        self.T = T    # Safe time headway (s)
        self.a = a    # Maximum acceleration (m/s^2)
        self.b = b    # Comfortable deceleration (m/s^2)
        self.s0 = s0  # Minimum distance (m)
        # Boundary time - the integration won’t continue beyond it.
        self.t_bound = 0.1 # like 1 / FPS
        self.delta = delta

    def acceleration(self, ego_v, lead_v, s):
        """Calculates the acceleration of the ego vehicle."""
        delta_v = ego_v - lead_v
        s_star = self.s0 + ego_v * self.T + (ego_v * delta_v) / (2 * np.sqrt(self.a * self.b))
        # print("delta_v:", delta_v, s_star)
        print("return:", (ego_v / self.v0) ** 4, (s_star / s) ** 2)
        return self.a * (1 - (ego_v / self.v0) ** 4 - (s_star / s) ** 2)
    
    def acceleration_with_penalty(self, ego_v, lead_v, s, current_pos, target_trajectory, penalty_weight=1.0):
        delta_v = ego_v - lead_v
        s_star = self.s0 + ego_v * self.T + (ego_v * delta_v) / (2 * np.sqrt(self.a * self.b))
        
        # 計算到目標軌跡的偏差
        deviation = np.min(np.linalg.norm(target_trajectory - current_pos, axis=1))
        
        # 加速度計算，包含懲罰項
        return self.a * (1 - (ego_v / self.v0) ** 4 - (s_star / s) ** 2) - penalty_weight * deviation
    def compute_target_speed_idm(self, ego_v, lead_v, distance):
        """Compute the target speed using IDM with RK45 integration."""
        def idm_equations(t, x):
            ego_pos, ego_speed = x
            speed_diff = ego_speed - lead_v
            s_star = self.s0 + ego_speed * self.T + ego_speed * speed_diff / (2. * np.sqrt(self.a * self.b))
            s = max(0.1, distance - ego_pos)  # Ensure distance is positive
            dvdt = self.a * (1. - (ego_speed / self.v0) ** self.delta - (s_star / s) ** 2)
            return [ego_speed, dvdt]

        y0 = [0., ego_v]  # Initial position and velocity
        rk45 = RK45(fun=idm_equations, t0=0., y0=y0, t_bound=self.t_bound)

        while rk45.status == "running":
            rk45.step()

        target_speed = rk45.y[1]  # Final speed
        return np.clip(target_speed, 0, np.inf)

        
def simulate_idm_trajectory(df, ego_id='ego', lead_id=None, delta_t=0.1):
    
    # 建立 IDM 模型實例
    idm = IDM()
    
    # 將 DataFrame 中 TRACK_ID 為 "ego" 的車輛挑選出來
    ego_df = df[df['TRACK_ID'] == ego_id].copy()
    non_ego_df = df[df['TRACK_ID'] != ego_id]
    lead_df = df[df['TRACK_ID'] == lead_id].copy() if lead_id else pd.DataFrame()

    # 初始化 "ego" 車的數據
    ego_pos = ego_df[['X', 'Y']].values
    ego_v = ego_df['V'].values
    timestamps = ego_df['TIMESTAMP'].values
    # lead_car = None
    
    # 更新 "ego" 車的軌跡
    new_positions = [ego_pos[0]]
    new_velocities = [ego_v[0]]
    for i in range(1, len(ego_df)):
        # 找到 "ego" 車當前時間點的前車 (最接近的前一台車)
        current_time = timestamps[i]
        # lead_car = non_ego_df[non_ego_df['TIMESTAMP'] == current_time]
        lead_car = lead_df[lead_df['TIMESTAMP'] == current_time]

        
        if not lead_car.empty:
            lead_x, lead_y = lead_car.iloc[0][['X', 'Y']]
            lead_v = lead_car.iloc[0]['V']
            distance = np.linalg.norm(new_positions[-1] - np.array([lead_x, lead_y]))

            # 用 IDM 計算 "ego" 車的加速度
            # acc = idm.acceleration(new_velocities[-1], lead_v, distance)

            target_speed = idm.compute_target_speed_idm(new_velocities[-1], lead_v, distance)

        else:
            print("No attacker!!")
            # 若無前車，ego 車只根據最大加速度向目標速度加速
            # acc = idm.a * (1 - (new_velocities[-1] / idm.v0) ** 4)

            target_speed = min(idm.v0, new_velocities[-1] + idm.a * delta_t)


        # 更新速度和位置
        # new_velocity = max(0, new_velocities[-1] + acc * delta_t)
        new_velocity = max(0, target_speed)
        new_position = new_positions[-1] + new_velocity * delta_t * (
            ego_pos[i] - ego_pos[i - 1]) / np.linalg.norm(ego_pos[i] - ego_pos[i - 1])
        # print("pos:", new_position)
        new_positions.append(new_position)
        new_velocities.append(new_velocity)
    
    # 更新 "ego" 的新軌跡到 DataFrame
    ego_df['X'] = [pos[0] for pos in new_positions]
    ego_df['Y'] = [pos[1] for pos in new_positions]
    ego_df['V'] = new_velocities
    
    # 將更新後的 "ego" 車的數據與其他車輛數據合併
    updated_df = pd.concat([non_ego_df, ego_df]).sort_values(by=['TIMESTAMP', 'TRACK_ID']).reset_index(drop=True)
    return updated_df

if __name__ == "__main__":
    dt = 0.1
    # folder = './output_csv_7_24/'
    # folder = './output_csv_moving_foward_interpolation_7_24/'
    # IDM_folder = './idm_traj/'
    
    # folder = './output_csv_moving_foward_interpolation_7_24/'
    # IDM_folder = './idm_traj_new_v/'

    folder = 'nuscenes_csv_result_on_STRIVE_forward/'
    IDM_folder = './idm_traj_on_STRIVE_new_v/'
    
    if not os.path.isdir(IDM_folder):
        os.mkdir(IDM_folder)
    for scenario_name in sorted(os.listdir(folder)):
        split_name = scenario_name.split('_')
        # if int(split_name[6].split('-')[1]) != 93:
        #    continue
        lead_id = split_name[9]
        traj_df = pd.read_csv(os.path.join(folder + scenario_name))
        idm_traj_df = simulate_idm_trajectory(traj_df, 'ego', lead_id, dt)
        idm_traj_df.to_csv(IDM_folder + scenario_name + '_idm.csv', index=False)
