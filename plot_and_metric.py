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
from tqdm import tqdm
import csv
import glob
# from nuscenes.utils.geometry_utils import Box
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="descartes.patch")


VEH_COLL_THRESH = 0.02

def angle_vectors(v1, v2):
    """ Returns angle between two vectors.  """
    # 若是車輛靜止不動 ego_vec為[0 0]
    if v1[0] < 0.1 and v1[1] < 0.1:
        v1_u = [1.0, 0.1]
    else:
        v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # 因為np.arccos給的範圍只有0~pi (180度)，但若cos值為-0.5，不論是120度或是240度，都會回傳120度，因此考慮三四象限的情況，強制轉180度到一二象限(因車輛和行人面積的對稱性，故可這麼做)
    if v1_u[1] < 0:
        v1_u = v1_u * (-1)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if math.isnan(angle):
        return 0.0
    else:
        return angle

def main(data_source, mode, past_len, interpolation_frame, sav_folder, topo_folder, map_path):
    vehicle_length = 4.7
    vehicle_width = 2
    interpolation_frame = 5
    folder = data_source
    scenario_types = ['LC', 'HO', 'RE', 'JC', 'LTAP']
    result_df = pd.DataFrame(columns=["Scenario_Name"] + scenario_types)
    # for scenario_name in sorted(os.listdir(folder)):
    for scenario_name in sorted(os.listdir(folder), key=lambda x: int(x.split('scene-')[1].split('_')[0])):
        scenario_data = {"Scenario_Name": scenario_name.split('_further')[0]}

        town_name = scenario_name.split('_')[4]
        split_name = scenario_name.split('_')
        type_name = split_name[5]        
        # if int(split_name[6].split('-')[1]) > 50:
        #    continue
        if int(split_name[6].split('-')[1]) != 292: #93
           continue
        print(scenario_name)
        dir_name = split_name[5] + '_' + split_name[6] + '_' + split_name[7] + '_' + split_name[8] + '_' + split_name[9]
        sav_path = sav_folder + dir_name +  '/'
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        else:
            continue
        nusc_map = NuScenesMap(dataroot=map_path, map_name=town_name)
        traj_df = pd.read_csv(os.path.join(
            data_source + scenario_name))
        ############################
        if mode == 'Initial nuscenes':
            traj_df['YAW'] = np.degrees(traj_df['YAW'])
        traj_df.loc[traj_df['X'] == 0, 'X'] = None
        traj_df.loc[traj_df['Y'] == 0, 'Y'] = None
        traj_df[['X', 'Y']] = traj_df.groupby('TRACK_ID')[['X', 'Y']].apply(lambda group: group.ffill().bfill())
        ############################
        ego_list = []
        risky_vehicle_list = []
        angle_list = []
        flag = 0            
        vehicle_list = []
        fill_dict = {}
        collision_flag = 0
        right_attacker_flag = 0
        real_yaw_distance = -999
        record_yaw_distance = -999
        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
            vehicle_list.append(remain_df)
        d = dict()
        d['scenario_id'] = scenario_name
        split_name = scenario_name.split('_')
        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
        
        lane_feature = np.load(topo_folder + initial_name + '.npy', allow_pickle=True)
        for n in range(len(vehicle_list)):
            vl = vehicle_list[n].to_numpy()
            now_id = vl[0][0]
            data_length = vl.shape[0]
            if now_id == "ego":
                forever_present_x = vl[-1][3]
                forever_present_y = vl[-1][4]
        attacker_id = scenario_name.split('_')[9].split('.')[0]
        
        for track_id, remain_df in traj_df.groupby('TRACK_ID'):
            fill_dict[track_id] = []
            if str(track_id) == 'ego':
                ego_list.append(remain_df.reset_index())
        scenario_length = len(vehicle_list[0])

        for t in range(1, scenario_length + 1):
            
            fig, ax = nusc_map.render_layers(["drivable_area"])
            print(initial_name, t)
            black_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='darkgray', edgecolor='black', label='Agents')
            black_legend = ax.legend(handles=[black_patch], loc='upper left', bbox_to_anchor=(0.02, 0.98))
            for text in black_legend.get_texts():
                text.set_fontsize(20)
            ax.add_artist(black_legend)
            ego_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='red', edgecolor='black', label='Ego')
            ego_legend = ax.legend(handles=[ego_patch], loc='upper left', bbox_to_anchor=(0.02, 0.95))
            for text in ego_legend.get_texts():
                text.set_fontsize(20)
            ax.add_artist(ego_legend)
            att_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='blue',edgecolor='black', label='Attacker')
            att_legend = ax.legend(handles=[att_patch], loc='upper left', bbox_to_anchor=(0.02, 0.92))
            for text in att_legend.get_texts():
                text.set_fontsize(20)
            ego_x = ego_list[0].loc[t - 1, 'X']
            # ego_x_next = ego_list[0].loc[t, 'X']
            ego_y = ego_list[0].loc[t - 1, 'Y']
            # ego_y_next = ego_list[0].loc[t, 'Y']
            # ego_vec = [ego_y_next - ego_y,ego_x_next - ego_x]
            # ego_angle = np.rad2deg(angle_vectors(ego_vec, [1, 0])) * np.pi / 180
            real_ego_angle = ego_list[0].loc[t - 1, 'YAW'] + 360.0 if ego_list[0].loc[t - 1, 'YAW'] < 0 else ego_list[0].loc[t - 1, 'YAW']
            real_ego_angle = (real_ego_angle + 90.0) * np.pi / 180
            #### real_ego_angle
            # ego_rec = [ego_x_next, ego_y_next, vehicle_width, vehicle_length, ego_angle]
            ego_rec = [ego_x, ego_y, vehicle_width, vehicle_length, real_ego_angle]
            
            x_1 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_2 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_3 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            x_4 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            y_1 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_2 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_3 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            y_4 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            ego_coords = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3], [x_1, y_1]])
            ego_polygon = Polygon(ego_coords)
            # ego_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
            # ego_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
            if t <= past_len * interpolation_frame:
                ego_color = 'lightcoral'
            else:
                ego_color = 'red'

            ########### trajectory ###########
            # (x_1, y_1) ---- (x_2, y_2)
            #     |                |
            #     |                |
            # (x_3, y_3) ---- (x_4, y_4)
            # gradient = np.linspace(0, 1, 100)
            # colors_gradient = plt.cm.viridis(gradient)
            # plt.fill_between([ego_x_next], [ego_y, ego_y_next], color=colors_gradient, label='Fill Between Gradient')
            # plt.fill_between([ego_y_next], [ego_x, ego_x_next], color=colors_gradient, label='Fill Between Gradient')
            ########### trajectory ###########
            #             
            ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
            ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color=ego_color, alpha=1)
            fill_dict['ego'].append(np.array([[x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3]]))
            
            for n in range(len(vehicle_list)):
                vl = vehicle_list[n].to_numpy()
                # vl : frame, id, x, y
                # => id, frame, v, x, y, yaw(arc)
                now_id = vl[0][0]
                if str(now_id) == 'ego':
                    continue
                real_pred_x = vl[t - 1][3]
                real_pred_y = vl[t - 1][4]
                real_other_angle = (vl[t - 1][5] + 90.0)
                real_other_angle = real_other_angle * np.pi / 180
                # other_angle = vl[past_len][4]
                # ego_angle = ego_list[0][4][int(filename_t) + past_len]
                #print(ego_x, ego_y, real_pred_x, real_pred_y)
                # ego_rec = [real_pred_x_next, real_pred_y_next, vehicle_width, vehicle_length, other_angle]
                ego_rec = [real_pred_x, real_pred_y, vehicle_width, vehicle_length, real_other_angle]
                x_1 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_2 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_3 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                x_4 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                y_1 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_2 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_3 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                y_4 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                other_coords = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3], [x_1, y_1]])
                other_polygon = Polygon(other_coords)                
                ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                fill_dict[now_id].append(np.array([[x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3]]))
                if now_id == attacker_id:
                    # print(ego_x, ego_y, real_pred_x, real_pred_y, attacker_id)
                    if t <= past_len * interpolation_frame:
                        attacker_color = 'cyan'
                    else:
                        attacker_color = 'blue'
                    ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color=attacker_color, alpha=0.5)
                else:
                    ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='darkgray', alpha=0.5)
                
                cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
                #print(t, now_id, ego_polygon.intersection(other_polygon).area, "GT:", attacker_id_gt)
                
                if cur_iou > VEH_COLL_THRESH:
                    print(attacker_id, "COLLIDE!", now_id)
                    ax.plot([real_pred_x, ego_x], [
                            real_pred_y, ego_y], '-.', color='red', markersize=1)
                    collision_flag = 1
                    if str(now_id) == str(attacker_id):
                        # plt.close()
                        # fig,ax = plt.subplots()
                        # ax.add_patch(ego_pg)
                        # ax.add_patch(other_pg)
                 
                        right_attacker_flag = 1
                        # real_yaw_distance = angle_vectors(other_vec, ego_vec) * 180 / np.pi
                        record_yaw_distance = real_ego_angle - real_other_angle
                        #print(record_yaw_distance)
            
            # Plot Past Trajectory
            if t > 1:
                for n in range(len(vehicle_list)):
                    vl = vehicle_list[n].to_numpy()
                    now_id = vl[0][0]
                    if str(now_id) == attacker_id:
                        if t <= past_len * interpolation_frame:
                            fill_color = 'cyan'
                        else:
                            fill_color = 'blue'
                    elif now_id == 'ego':
                        if t <= past_len * interpolation_frame:
                            fill_color = 'lightcoral'
                        else:
                            fill_color = 'red'
                    else:
                        fill_color = 'darkgray'
                    for fill_t in range(t):
                        alpha_value = 0 if 1 - (t - fill_t - 1) * 0.2 <= 0 else 1 - (t - fill_t - 1) * 0.2
                        # print(alpha_value, fill_dict[now_id][fill_t])
                        ax.fill(fill_dict[now_id][fill_t][0], fill_dict[now_id][fill_t][1], '-',  color=fill_color, alpha=alpha_value)
                        # ax.fill_between(fill_dict[now_id][fill_t][0], fill_dict[now_id][fill_t][1], '-',  color=fill_color, alpha=alpha_value)
                        # if concat
                        if t > 0 and (fill_t + 1) != t:
                            # print(fill_dict)
                            # exit()
                            front_point_left_x = fill_dict[now_id][fill_t][0][0]
                            front_point_left_y = fill_dict[now_id][fill_t][1][0]
                            front_point_right_x = fill_dict[now_id][fill_t][0][1]
                            front_point_right_y = fill_dict[now_id][fill_t][1][1]
                            back_point_left_x_next_t = fill_dict[now_id][fill_t + 1][0][2]
                            back_point_left_y_next_t = fill_dict[now_id][fill_t + 1][1][2]
                            back_point_right_x_next_t = fill_dict[now_id][fill_t + 1][0][3]
                            back_point_right_y_next_t = fill_dict[now_id][fill_t + 1][1][3]
                            # concat_x = [front_point_left_x, front_point_right_x, back_point_right_x_next_t, back_point_left_x_next_t]
                            # concat_y = [front_point_left_y, front_point_right_y, back_point_right_y_next_t, back_point_left_y_next_t]
                            concat_x = [front_point_left_x, front_point_right_x, back_point_left_x_next_t, back_point_right_x_next_t]
                            concat_y = [front_point_left_y, front_point_right_y, back_point_left_y_next_t, back_point_right_y_next_t]
                            # print(now_id, concat_x, concat_y)
                            not_detected_vehicle = abs(front_point_left_x) < 10 and abs(front_point_left_y) < 10 and abs(front_point_right_x) < 10 and abs(front_point_right_y)  < 10
                            # print(not abs(front_point_left_x) < 10 and abs(front_point_left_y) < 10 and abs(front_point_right_x) < 10 and abs(front_point_right_y)  < 10)
                            if not not_detected_vehicle:
                                ax.fill(concat_x, concat_y, '-',  color=fill_color, alpha=alpha_value)

                    
                
            ax.set_xlim(forever_present_x - 50,
                    forever_present_x + 50)
            ax.set_ylim(forever_present_y - 50,
                    forever_present_y + 50)
            # ax.set_clip_box([[forever_present_x - 50, forever_present_y - 50], [forever_present_x + 50, forever_present_y]])
            
            scenario_data[type_name] = "No"
            
            if right_attacker_flag:
                right_attacker_flag_str = "Collide with attacker!"
                scenario_data[type_name] = "Collision"
            else:
                right_attacker_flag_str = "Safe!"
                
            

            
            title = right_attacker_flag_str

            
            ax.set_title(title, fontsize=20)
            print(title)
            # sav_path = sav_folder + dir_name + '_' + right_attacker_flag_str +  '/'
            # if not os.path.exists(sav_path):
            #                     os.makedirs(sav_path)
            fig.savefig(sav_path + str(t) + '.png')
            # plt.close(fig)
            plt.cla()
            
            #     if collision_flag:
            #         break
            # if collision_flag:
            #     break
        
        result_df = result_df.append(scenario_data, ignore_index=True)
            
        images = []
        for filename in sorted(os.listdir(sav_path)):
            #images.append(imageio.imread(filename))
            if filename.split('.')[-1] == 'gif':
                continue
            front_num = filename.split('.')[0]
            if int(front_num) >= 10:
                continue
            images.append(Image.open(sav_path + filename))
        for filename in sorted(os.listdir(sav_path)):
            #images.append(imageio.imread(filename))
            if filename.split('.')[-1] == 'gif':
                continue
            front_num = filename.split('.')[0]
            if int(front_num) < 10:
                continue
            images.append(Image.open(sav_path + filename))
        images[0].save(
            sav_path + scenario_name + '.gif', 
            save_all=True, 
            append_images=images[1:], 
            optimize=True,
            loop=0,
            duration=100,
        )
    result_df.to_csv("scenario_summary.csv", index=False)


def only_metric(data_source, past_len, interpolation_frame, sav_folder, topo_folder, map_path):
    vehicle_length = 4.7
    vehicle_width = 2
    interpolation_frame = 5
    folder = data_source
    # all_scenario_type_dict = {'HO': 0, 'LTAP': 0, 'RE': 0, 'JC': 0, 'LC': 0}
    # col_scenario_type_dict = {'HO': 0, 'LTAP': 0, 'RE': 0, 'JC': 0, 'LC': 0}
    all_scenario_type_dict = {'LC': 0, 'HO': 0, 'RE': 0, 'JC': 0, 'LTAP': 0}
    col_scenario_type_dict = {'LC': 0, 'HO': 0, 'RE': 0, 'JC': 0, 'LTAP': 0}
    each_scenario_col_dict = {}
    for scenario_name in tqdm(sorted(os.listdir(folder))):
        town_name = scenario_name.split('_')[4]
        split_name = scenario_name.split('_')
        print(scenario_name)
        print("all:", all_scenario_type_dict)
        print("col:", col_scenario_type_dict)
        dir_name = split_name[5] + '_' + split_name[6] + '_' + split_name[7] + '_' + split_name[8] + '_' + split_name[9]
        sav_path = sav_folder + dir_name +  '/'
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)

        traj_df = pd.read_csv(os.path.join(
            data_source + scenario_name))
        
        scenario_type = scenario_name.split('_')[5]
        all_scenario_type_dict[scenario_type] += 1

        ############################
        invalid_track_ids = traj_df.groupby('TRACK_ID').filter(lambda group: (group['X'] == 0).all() and (group['Y'] == 0).all())['TRACK_ID'].unique()
        traj_df = traj_df[~traj_df['TRACK_ID'].isin(invalid_track_ids)]

        traj_df.loc[traj_df['X'] == 0, 'X'] = None
        traj_df.loc[traj_df['Y'] == 0, 'Y'] = None
        traj_df[['X', 'Y']] = traj_df.groupby('TRACK_ID')[['X', 'Y']].apply(lambda group: group.ffill().bfill())
        ############################

        ego_list = []
        risky_vehicle_list = []
        angle_list = []
        flag = 0            
        vehicle_list = []
        fill_dict = {}
        collision_flag = 0
        right_attacker_flag = 0
        real_yaw_distance = -999
        record_yaw_distance = -999
        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
            vehicle_list.append(remain_df)
            
        d = dict()
        d['scenario_id'] = scenario_name
        split_name = scenario_name.split('_')
        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
        
        attacker_id = scenario_name.split('_')[9].split('.')[0]
        
        for track_id, remain_df in traj_df.groupby('TRACK_ID'):
            fill_dict[track_id] = []
            if str(track_id) == 'ego':
                ego_list.append(remain_df.reset_index())
        scenario_length = len(vehicle_list[0])

        for t in range(1, scenario_length + 1):
            ego_x = ego_list[0].loc[t - 1, 'X']
            # ego_x_next = ego_list[0].loc[t, 'X']
            ego_y = ego_list[0].loc[t - 1, 'Y']
            # ego_y_next = ego_list[0].loc[t, 'Y']
            # ego_vec = [ego_y_next - ego_y,ego_x_next - ego_x]
            # ego_angle = np.rad2deg(angle_vectors(ego_vec, [1, 0])) * np.pi / 180
            real_ego_angle = ego_list[0].loc[t - 1, 'YAW'] + 360.0 if ego_list[0].loc[t - 1, 'YAW'] < 0 else ego_list[0].loc[t - 1, 'YAW']
            real_ego_angle = (real_ego_angle + 90.0) * np.pi / 180
            #### real_ego_angle
            # ego_rec = [ego_x_next, ego_y_next, vehicle_width, vehicle_length, ego_angle]
            ego_rec = [ego_x, ego_y, vehicle_width, vehicle_length, real_ego_angle]
            
            x_1 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_2 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_3 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            x_4 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            y_1 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_2 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_3 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            y_4 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            ego_coords = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3], [x_1, y_1]])
            ego_polygon = Polygon(ego_coords)
            for n in range(len(vehicle_list)):
                vl = vehicle_list[n].to_numpy()
                # vl : frame, id, x, y
                # => id, frame, v, x, y, yaw(arc)
                now_id = vl[0][0]
                if str(now_id) == 'ego':
                    continue
                real_pred_x = vl[t - 1][3]
                # real_pred_x_next = vl[t][3]
                real_pred_y = vl[t - 1][4]
                # real_pred_y_next = vl[t][4]
                # other_vec = [real_pred_y_next - real_pred_y,real_pred_x_next - real_pred_x]
                # other_angle = np.rad2deg(angle_vectors(other_vec, [1, 0])) * np.pi / 180
                real_other_angle = (vl[t - 1][5] + 90.0)
                # 90 or 180 => tend to wrong
                # check_angle = real_other_angle + 360.0 if real_other_angle < 0 else real_other_angle
                # if abs(abs(check_angle) - 90.0) or abs(abs(check_angle) - 270.0) < 10:
                #     real_other_angle = 90
                # elif abs(abs(check_angle) - 180.0) or abs(abs(check_angle) - 360.0) < 10:
                #     real_other_angle = 0                
                real_other_angle = real_other_angle * np.pi / 180
                # other_angle = vl[past_len][4]
                # ego_angle = ego_list[0][4][int(filename_t) + past_len]
                #print(ego_x, ego_y, real_pred_x, real_pred_y)
                # ego_rec = [real_pred_x_next, real_pred_y_next, vehicle_width, vehicle_length, other_angle]
                ego_rec = [real_pred_x, real_pred_y, vehicle_width, vehicle_length, real_other_angle]
                x_1 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_2 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_3 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                x_4 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                y_1 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_2 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_3 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                y_4 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                other_coords = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3], [x_1, y_1]])
                print("other_coords:", other_coords)
                other_polygon = Polygon(other_coords)               
                cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
                
                if cur_iou > VEH_COLL_THRESH:
                    collision_flag = 1
                    if str(now_id) == str(attacker_id):
                        right_attacker_flag = 1
                        col_scenario_type_dict[scenario_type] += 1
                if collision_flag:
                    break
            if collision_flag:
                break
                
    print("all:", all_scenario_type_dict)
    for variant_key in all_scenario_type_dict:
        col_scenario_type_dict[variant_key] /= all_scenario_type_dict[variant_key]
    col_df = pd.DataFrame(col_scenario_type_dict, index=['collision rate'])
    col_df.to_csv('collision_rate_on_idm.csv', mode='a', header=False)
    print("col:", col_scenario_type_dict)

def cal_cr_and_similarity(traj_df, attacker_id_gt):
    vehicle_width, vehicle_length = 2, 4.7
    
    collision_moment = None
    collision_flag = 0
    right_attacker_flag = 0
    real_yaw_distance = -999
    record_yaw_distance = -999
    vehicle_list = []
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        vehicle_list.append(remain_df)
    ego_list = []
    attacker_list = []
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        if str(track_id) == 'ego':
            ego_list.append(remain_df.reset_index())
        elif str(track_id) == attacker_id_gt:
            attacker_list.append(remain_df.reset_index())
    # print(traj_df)

    scenario_length = len(vehicle_list[0])
    for t in range(1, scenario_length+1):
        ego_x = ego_list[0].loc[t - 1, 'X']
        # ego_x_next = ego_list[0].loc[t, 'X']
        ego_y = ego_list[0].loc[t - 1, 'Y']
        # ego_y_next = ego_list[0].loc[t, 'Y']
        # ego_vec = [ego_y_next - ego_y,
        #                     ego_x_next - ego_x]
        # ego_angle = np.rad2deg(angle_vectors(ego_vec, [1, 0])) * np.pi / 180
        # real_ego_angle = ego_list[0].loc[t - 1, 'YAW']
        real_ego_angle = ego_list[0].loc[t - 1, 'YAW'] + 360.0 if ego_list[0].loc[t - 1, 'YAW'] < 0 else ego_list[0].loc[t - 1, 'YAW']
        real_ego_angle = (real_ego_angle + 90.0) * np.pi / 180
        ego_rec = [ego_x, ego_y, vehicle_width, vehicle_length, real_ego_angle]
        x_1 = float(np.cos(
            ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
        x_2 = float(np.cos(
            ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
        x_3 = float(np.cos(
            ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
        x_4 = float(np.cos(
            ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
        y_1 = float(np.sin(
            ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
        y_2 = float(np.sin(
            ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
        y_3 = float(np.sin(
            ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
        y_4 = float(np.sin(
            ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
        ego_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
        ego_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
        
        plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='pink', alpha=0.5)
        # plt.plot([x_1, x_2, x_4, x_3, x_1], [
        #                             y_1, y_2, y_4, y_3, y_1], '-',  color='lime', markersize=3)
        attacker_x = attacker_list[0].loc[t-1, 'X']
        attacker_y = attacker_list[0].loc[t-1, 'Y']
        # attacker_x_next = attacker_list[0].loc[t, 'X']
        # attacker_y_next = attacker_list[0].loc[t, 'Y']
        real_attacker_angle = attacker_list[0].loc[t - 1, 'YAW'] + 360.0 if attacker_list[0].loc[t - 1, 'YAW'] < 0 else attacker_list[0].loc[t - 1, 'YAW']
        real_attacker_angle = (real_attacker_angle + 90.0) * np.pi / 180
        # ego_rec = [attacker_x_next, attacker_y_next, self.vehicle_width, self.vehicle_length, real_attacker_angle]
        ego_rec = [attacker_x, attacker_y, vehicle_width, vehicle_length, real_attacker_angle]
        x_1 = float(np.cos(
            ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
        x_2 = float(np.cos(
            ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
        x_3 = float(np.cos(
            ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
        x_4 = float(np.cos(
            ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
        y_1 = float(np.sin(
            ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
        y_2 = float(np.sin(
            ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
        y_3 = float(np.sin(
            ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
        y_4 = float(np.sin(
            ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
        attacker_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
        for n in range(len(vehicle_list)):
            vl = vehicle_list[n].to_numpy()
            # vl : frame, id, x, y
            # => id, frame, v, x, y, yaw(arc)
            now_id = vl[0][0]
            if str(now_id) == 'ego':
                continue
            real_pred_x = vl[t - 1][3]
            # real_pred_x_next = vl[t][3]
            real_pred_y = vl[t - 1][4]
            # real_pred_y_next = vl[t][4]
            # other_vec = [real_pred_y_next - real_pred_y,
                                    # real_pred_x_next - real_pred_x]
            # other_angle = np.rad2deg(angle_vectors(other_vec, [1, 0])) * np.pi / 180
            real_other_angle = (vl[t - 1][5] + 90.0) * np.pi / 180
            # other_angle = vl[past_len][4]
            # ego_angle = ego_list[0][4][int(filename_t) + past_len]
            #print(ego_x, ego_y, real_pred_x, real_pred_y)
            ego_rec = [real_pred_x, real_pred_y, vehicle_width, vehicle_length, real_other_angle]
            # ego_rec = [real_pred_x_next, real_pred_y_next, self.vehicle_width
            #                             , self.vehicle_length, other_angle]
            
            x_1 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_2 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_3 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            x_4 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            y_1 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_2 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_3 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            y_4 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            other_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
            other_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
            # plt.plot([x_1, x_2, x_4, x_3, x_1], [
            #                         y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
            if str(now_id) != str(attacker_id_gt):
                attacker_other_iou = attacker_polygon.intersection(other_polygon).area / attacker_polygon.union(other_polygon).area
                
                if attacker_other_iou > VEH_COLL_THRESH:
                    print("attacker early hit")
                    #print(t, now_id, "other:", attacker_id_gt, real_pred_x, real_pred_y, real_other_angle, "GT attacker:", ego_x, ego_y, real_ego_angle)

                    collision_flag = 2
                    # print(now_id, attacker_other_iou, real_pred_x_next, real_pred_y_next)
            cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
            # print(t, now_id, ego_polygon.intersection(other_polygon).area, cur_iou, "GT:", attacker_id_gt, "flag:", collision_flag)
            
            if cur_iou > VEH_COLL_THRESH:
                # print(attacker_id_gt, "COLLIDE!", now_id)
                collision_flag = 1
                if str(now_id) == str(attacker_id_gt):
                    # plt.close()
                    # fig,ax = plt.subplots()
                    # ax.add_patch(ego_pg)
                    # ax.add_patch(other_pg)
                    # ax.set_xlim([1821,1835])
                    # ax.set_ylim([2529,2544])
                    #plt.show()

                    #print("COLLIDE! GT!!!!!!!! ", cur_iou)
                    
                    right_attacker_flag = 1
                    collision_moment = t
                    # Must collide with GT attacker
                    
                    # real_yaw_distance = angle_vectors(other_vec, ego_vec) * 180 / np.pi
                    real_yaw_distance = None
                    record_yaw_distance = (real_ego_angle - real_other_angle) * 180 / np.pi
                    #print(record_yaw_distance)
                else:
                    plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='green', alpha=1)
                    #plt.plot([x_1, x_2, x_4, x_3, x_1], [
                    #                 y_1, y_2, y_4, y_3, y_1], '-',  color='violet', markersize=3)

                
            if collision_flag:
                break
        if collision_flag:
            break
    return collision_flag, real_yaw_distance, right_attacker_flag, record_yaw_distance, collision_moment

def only_metric_from_tnt_trainer(folder, mode, past_len, interpolation_frame, sav_folder, topo_folder, map_path):
    pred_collision_rate = 0
    attacker_right_collision_rate = 0
    real_yaw_dist_average = 0
    all_data_num = 0

    lane_change_all = 0
    junction_crossing_all = 0
    LTAP_all = 0
    opposite_direction_all = 0
    rear_end_all = 0
    
    col_lane_change_num = 0.0000001
    col_junction_crossing_num = 0.0000001
    col_LTAP_num = 0.0000001
    col_opposite_direction_num = 0.0000001
    col_rear_end_num = 0.0000001

    lane_change_yaw_distance = 0
    junction_crossing_yaw_distance = 0
    LTAP_yaw_distance = 0
    opposite_direction_yaw_distance = 0
    rear_end_yaw_distance = 0
    
    for scenario_name in tqdm(sorted(os.listdir(folder))):
        attacker_id = scenario_name.split('_')[9]
        gt_record_yaw_distance = float(scenario_name.split('_')[8])
        condition = scenario_name.split('_')[5]
        all_data_num += 1
        # split_name = scenario_name.split('_')
        print(scenario_name, attacker_id)
        # dir_name = split_name[5] + '_' + split_name[6] + '_' + split_name[7] + '_' + split_name[8] + '_' + split_name[9]
        # sav_path = sav_folder + dir_name +  '/'
        # if not os.path.exists(sav_path):
        #     os.makedirs(sav_path)
        if condition == 'LTAP':
            LTAP_all += 1
            ideal_yaw_offset = 90
        elif condition == 'JC':
            junction_crossing_all += 1
            ideal_yaw_offset = 90
        elif condition == 'HO':
            opposite_direction_all += 1
            ideal_yaw_offset = 180
        elif condition == 'RE':
            rear_end_all += 1
            ideal_yaw_offset = 0
        elif condition == 'LC':
            lane_change_all += 1
            ideal_yaw_offset = 20
        traj_df = pd.read_csv(os.path.join(
            folder + scenario_name))
        collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance, collision_moment = cal_cr_and_similarity(traj_df, attacker_id)
        if collision_flag:
            pred_collision_rate += 1
            
            while record_yaw_distance < 0:
                record_yaw_distance = (record_yaw_distance + 360.0)
            record_yaw_distance = abs(record_yaw_distance - 360.0) if record_yaw_distance > 180 else record_yaw_distance
            if attacker_right_flag:
                attacker_right_collision_rate += 1
                
                # yaw_distance = abs(ideal_yaw_offset - record_yaw_distance)
                yaw_distance = abs(gt_record_yaw_distance - record_yaw_distance)
                
                real_yaw_dist_average += yaw_distance                                
                if condition == 'LTAP':
                    LTAP_yaw_distance += yaw_distance
                    col_LTAP_num += 1
                elif condition == 'JC':
                    junction_crossing_yaw_distance += yaw_distance
                    col_junction_crossing_num += 1
                elif condition == 'HO':
                    opposite_direction_yaw_distance += yaw_distance
                    col_opposite_direction_num += 1
                elif condition == 'RE':
                    rear_end_yaw_distance += yaw_distance
                    col_rear_end_num += 1
                elif condition == 'LC':
                    lane_change_yaw_distance += yaw_distance
                    col_lane_change_num += 1
    type_name_list = ["LC", "HO", "RE", "JC", "LTAP", "All", "-"]
    all_right_col_num = col_LTAP_num + col_junction_crossing_num + col_lane_change_num + col_opposite_direction_num + col_rear_end_num
    type_cr_list = [str(round(col_lane_change_num / lane_change_all, 2)), str(round(col_opposite_direction_num / opposite_direction_all, 2)),
                str(round(col_rear_end_num / rear_end_all, 2)), str(round(col_junction_crossing_num / junction_crossing_all, 2)),
                    str(round(col_LTAP_num / LTAP_all, 2)), str(round((col_lane_change_num + col_opposite_direction_num + col_rear_end_num + col_junction_crossing_num + col_LTAP_num) / all_data_num, 2))]
    csv_file = './run/idm_metric_from_tnt_trainer_' + mode + '.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'a+') as f:
            writer = csv.writer(f)
            # writer.writerow([ 'Type', 'CR', 'Sim(5)', 'Sim(10)', 'Sim(20)', 'Sim(30)', 'Yaw dist']) # RCNN version
            writer.writerow([ 'Type', 'CR'])
            f.close()
    with open(csv_file, 'a+') as f:
        writer = csv.writer(f)
        # writer.writerow([save_folder.split('/')[-1], self.lr, self.weight_decay, ego_iou_50 / all_data_num, sum_fde / all_data_num, all_yaw_distance / all_data_num, 
        #                  attacker_iou_50 / all_data_num, sum_attacker_de / all_data_num, ideal_yaw_dist_average / all_data_num,
        #                  all_tp_cls_acc / all_data_num, cr, Average_Similarity])

        for type_index in range(len(type_name_list)):
            if type_index == len(type_name_list) - 1:
                writer.writerow(['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'])
            else:
                writer.writerow([type_name_list[type_index],
                                type_cr_list[type_index]])  # RCNN version
        f.close()


def plot_paper_figure(args, mode):
    
    # past => future (collision point) => forward (4 sec * 5 f)
    
    vehicle_length = 4.7
    vehicle_width = 2
    line_w = 5
    forward_seconds = 4
    interpolation_frame = args.interpolation_frame
    folder = args.data_path
    scenario_types = ['LC', 'HO', 'RE', 'JC', 'LTAP']
    result_df = pd.DataFrame(columns=["Scenario_Name"] + scenario_types)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    # for scenario_name in sorted(os.listdir(folder)):
    for scenario_name in sorted(os.listdir(folder), key=lambda x: int(x.split('scene-')[1].split('_')[0])):
        scenario_data = {"Scenario_Name": scenario_name.split('_further')[0]}
        if len(scenario_name.split('_')) > 1:
            town_name = scenario_name.split('_')[4]
            split_name = scenario_name.split('_')
            attacker_id = scenario_name.split('_')[9].split('.')[0]
            type_name = split_name[5]        
        # if int(split_name[6].split('-')[1]) != 292: #93
        #    continue

        if type_name != 'HO':
           continue

        if mode == 'Initial nuscenes':
            if int(scenario_name.split('-')[1]) != 93:
                continue    
            for file2 in os.listdir(args.mapping_town_path):
                if scenario_name in file2:
                    town_name = file2.split("_")[4]
                    attacker_id = None # file2.split('_')[9]
                    type_name = file2.split('_')[5]

        
        print(scenario_name)
        # dir_name = split_name[5] + '_' + split_name[6] + '_' + split_name[7] + '_' + split_name[8] + '_' + split_name[9]
        # sav_path = args.sav_folder + dir_name +  '/'
        # if not os.path.exists(sav_path):
        #     os.makedirs(sav_path)
        # else:
        #     continue
        nusc_map = NuScenesMap(dataroot=args.map_path, map_name=town_name)
        traj_df = pd.read_csv(os.path.join(
            folder + scenario_name))
        ############################
        if mode == 'Initial nuscenes':
            traj_df['YAW'] = np.degrees(traj_df['YAW'])
        traj_df.loc[traj_df['X'] == 0, 'X'] = None
        traj_df.loc[traj_df['Y'] == 0, 'Y'] = None
        traj_df[['X', 'Y']] = traj_df.groupby('TRACK_ID')[['X', 'Y']].apply(lambda group: group.ffill().bfill())

        if mode == 'Initial nuscenes':
            timestamps = traj_df["TIMESTAMP"].unique()
            track_ids = traj_df["TRACK_ID"].unique()

            # 創建時間戳與 TRACK_ID 的完整範圍
            full_index = pd.MultiIndex.from_product([timestamps, track_ids], names=["TIMESTAMP", "TRACK_ID"])

            # 重新索引數據以填補缺失值
            traj_df = traj_df.set_index(["TIMESTAMP", "TRACK_ID"])
            traj_df = traj_df.reindex(full_index)

            # 前向填充和後向填充
            traj_df = traj_df.groupby(level="TRACK_ID").apply(lambda group: group.ffill().bfill())

            # 重置索引，保存結果
            traj_df = traj_df.reset_index()

            columns = ["TRACK_ID", "TIMESTAMP", "V", "X", "Y", "YAW"]
            traj_df = traj_df[columns]

        ############################
        ego_list = []
        risky_vehicle_list = []
        angle_list = []
        flag = 0            
        vehicle_list = []
        fill_dict = {}
        collision_flag = 0
        right_attacker_flag = 0
        real_yaw_distance = -999
        record_yaw_distance = -999
        
        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
            vehicle_list.append(remain_df)
        d = dict()
        d['scenario_id'] = scenario_name
        
        if mode == 'Initial nuscenes':
            for n in range(len(vehicle_list)):
                vl = vehicle_list[n].to_numpy()
                now_id = vl[0][0]
                data_length = vl.shape[0]
                if now_id == "ego":
                    forever_present_x = vl[8][3]
                    forever_present_y = vl[8][4]
                    scenario_length = vl.shape[0]
            

        else:
            for n in range(len(vehicle_list)):
                vl = vehicle_list[n].to_numpy()
                now_id = vl[0][0]
                data_length = vl.shape[0]
                if now_id == "ego":
                    forever_present_x = vl[-1][3]
                    forever_present_y = vl[-1][4]
                    scenario_length = vl.shape[0]
        
        
        collide_x, collide_y = forever_present_x, forever_present_y 
        for track_id, remain_df in traj_df.groupby('TRACK_ID'):
            fill_dict[track_id] = []
            if str(track_id) == 'ego':
                ego_list.append(remain_df.reset_index())
        # scenario_length = len(vehicle_list[0])

        past_alphas = np.linspace(0.2, 0.8, args.past_len * interpolation_frame )
        future_alphas = np.linspace(0.1, 1, scenario_length - args.past_len * interpolation_frame - 1)
        # future_alphas = np.linspace(0.1, 0.5, scenario_length - args.past_len * interpolation_frame - forward_seconds * interpolation_frame)



        fig, ax = nusc_map.render_layers(["drivable_area"])
        black_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='darkgray', edgecolor='black', label='Agents')
        black_legend = ax.legend(handles=[black_patch], loc='upper left', bbox_to_anchor=(0.02, 0.98))
        for text in black_legend.get_texts():
            text.set_fontsize(20)
        ax.add_artist(black_legend)
        ego_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='red', edgecolor='black', label='Ego')
        ego_legend = ax.legend(handles=[ego_patch], loc='upper left', bbox_to_anchor=(0.02, 0.95))
        for text in ego_legend.get_texts():
            text.set_fontsize(20)
        ax.add_artist(ego_legend)
        att_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='blue',edgecolor='black', label='Attacker')
        att_legend = ax.legend(handles=[att_patch], loc='upper left', bbox_to_anchor=(0.02, 0.92))
        for text in att_legend.get_texts():
            text.set_fontsize(20)



        for t in range(1, scenario_length):
            
            ego_x = ego_list[0].loc[t - 1, 'X']
            ego_y = ego_list[0].loc[t - 1, 'Y']
            ego_x_next = ego_list[0].loc[t, 'X']
            ego_y_next = ego_list[0].loc[t, 'Y']
            real_ego_angle = ego_list[0].loc[t - 1, 'YAW'] + 360.0 if ego_list[0].loc[t - 1, 'YAW'] < 0 else ego_list[0].loc[t - 1, 'YAW']
            real_ego_angle = (real_ego_angle + 90.0) * np.pi / 180
            ego_rec = [ego_x_next, ego_y_next, vehicle_width, vehicle_length, real_ego_angle]
            x_1 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_2 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
            x_3 = float(np.cos(
                ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            x_4 = float(np.cos(
                ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
            y_1 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_2 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
            y_3 = float(np.sin(
                ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            y_4 = float(np.sin(
                ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
            ego_coords = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3], [x_1, y_1]])
            ego_polygon = Polygon(ego_coords)
            # print(t, args.past_len * interpolation_frame, scenario_length - 1)
            if t <= args.past_len * interpolation_frame:
                ego_color = 'lightcoral'
                ax.plot([ego_x, ego_x_next], [ego_y, ego_y_next], color=ego_color, alpha=past_alphas[t-1], linewidth=line_w)
            else:
                ego_color = 'red'
                ax.plot([ego_x, ego_x_next], [ego_y, ego_y_next], color=ego_color, alpha=future_alphas[t-args.past_len * interpolation_frame-1], linewidth=line_w)
            if t == args.past_len * interpolation_frame or t == scenario_length - 1:
                ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color=ego_color, alpha=1)
            ego_bbox = np.vstack(([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3]))
            fill_dict['ego'].append(np.array([ego_bbox]))
            
            for n in range(len(vehicle_list)):
                vl = vehicle_list[n].to_numpy()
                # vl : frame, id, x, y
                # => id, frame, v, x, y, yaw(arc)
                now_id = vl[0][0]
                if str(now_id) == 'ego':
                    continue
                real_pred_x = vl[t - 1][3]
                real_pred_x_next = vl[t][3]
                real_pred_y = vl[t - 1][4]
                real_pred_y_next = vl[t][4]
                real_other_angle = (vl[t - 1][5] + 90.0)
                real_other_angle = real_other_angle * np.pi / 180
                ego_rec = [real_pred_x_next, real_pred_y_next, vehicle_width, vehicle_length, real_other_angle]
                x_1 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_2 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_3 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                x_4 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                y_1 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_2 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_3 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                y_4 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                other_coords = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3], [x_1, y_1]])
                other_polygon = Polygon(other_coords)    

                if abs(real_pred_x_next - real_pred_x) > 100 or abs(real_pred_y_next - real_pred_y) > 100:
                    print("filter")
                    continue
                
                if now_id == attacker_id:
                    if t <= args.past_len * interpolation_frame:
                        attacker_color = 'cyan'
                        ax.plot([real_pred_x, real_pred_x_next], [real_pred_y, real_pred_y_next], color=attacker_color, alpha=past_alphas[t-1], linewidth=line_w)
                    else:
                        attacker_color = 'blue'
                        ax.plot([real_pred_x, real_pred_x_next], [real_pred_y, real_pred_y_next], color=attacker_color, alpha=future_alphas[t-args.past_len * interpolation_frame-1], linewidth=line_w)
                    if t == args.past_len * interpolation_frame or t == scenario_length - 1:
                        ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                        ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color=attacker_color, alpha=1)
                    right_attacker_bbox = np.vstack(([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3]))
                else:
                    if t <= args.past_len * interpolation_frame:
                        other_color = 'gray'
                        ax.plot([real_pred_x, real_pred_x_next], [real_pred_y, real_pred_y_next], color=other_color, alpha=past_alphas[t-1], linewidth=line_w)
                    else:
                        other_color = 'darkgray'
                        # ax.plot([real_pred_x, real_pred_x_next], [real_pred_y, real_pred_y_next], color=other_color, alpha=future_alphas[t-args.past_len * interpolation_frame-1], linewidth=line_w)
                    # if t == args.past_len * interpolation_frame or t == scenario_length - 1:
                    #     ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                    #     ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color=other_color, alpha=1)
                    if t == args.past_len * interpolation_frame:
                        ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                        ax.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color=other_color, alpha=1)

                # fill_dict[now_id].append(np.array([[x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3]]))
                cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
                #print(t, now_id, ego_polygon.intersection(other_polygon).area, "GT:", attacker_id_gt)
                if cur_iou > VEH_COLL_THRESH:
                    print(attacker_id, "COLLIDE!", now_id)
                    print(ego_color, other_color)
                    # ax.plot([real_pred_x, ego_x], [
                    #         real_pred_y, ego_y], '-.', color='red', markersize=2)
                    collision_flag = 1
                    if str(now_id) == str(attacker_id):
                        right_attacker_flag = 1
                        # real_yaw_distance = angle_vectors(other_vec, ego_vec) * 180 / np.pi
                        record_yaw_distance = real_ego_angle - real_other_angle
                        collide_x = real_pred_x
                        collide_y = real_pred_y
                        #print(record_yaw_distance)
                        ax.fill(ego_bbox[0], ego_bbox[1], '-',  color=ego_color, alpha=0.7)
                        ax.fill(right_attacker_bbox[0], right_attacker_bbox[1], '-',  color=attacker_color, alpha=0.7)
                if collision_flag:
                    break
            if collision_flag:
                break
        # ax.set_xlim(forever_present_x - 50,
        #         forever_present_x + 50)
        # ax.set_ylim(forever_present_y - 50,
        #         forever_present_y + 50)
        
        # if mode == 'Initial nuscenes':
        #     ax.set_xlim(collide_x - 200,
        #             collide_x + 200)
        #     ax.set_ylim(collide_y - 200,
        #             collide_y + 200)
        # else:
        ax.set_xlim(collide_x - 40,
                collide_x + 40)
        ax.set_ylim(collide_y - 40,
                collide_y + 40)
        # ax.set_clip_box([[forever_present_x - 50, forever_present_y - 50], [forever_present_x + 50, forever_present_y]])
        
        scenario_data[type_name] = "No"
        
        if right_attacker_flag:
            right_attacker_flag_str = "Collide with attacker!"
            scenario_data[type_name] = "Collision"
        else:
            right_attacker_flag_str = "Safe!"
        title = right_attacker_flag_str
        ax.set_title(title, fontsize=20)
        print(title)
        # sav_path = sav_folder + dir_name + '_' + right_attacker_flag_str +  '/'
        if not os.path.exists(args.figure_save_path):
                            os.makedirs(args.figure_save_path)
        fig.savefig(args.figure_save_path + str(scenario_name) + '.png')
        print(args.figure_save_path + str(scenario_name))
        # plt.show()
        exit()
        # plt.close(fig)
        plt.cla()
        result_df = result_df.append(scenario_data, ignore_index=True)
        #     if collision_flag:
        #         break
        # if collision_flag:
        #     break
    result_df.to_csv("scenario_summary.csv", index=False)

def compute_attacker_speed_from_positions(df, x_col="X", y_col="Y", t_col="TIMESTAMP"):
    df = df.sort_values(t_col).copy()
    df["computed_speed"] = 0.0

    # 逐筆計算速率
    timestamps = df[t_col].values
    xs = df[x_col].values
    ys = df[y_col].values

    for i in range(1, len(df)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        dt = (timestamps[i] - timestamps[i-1]) / 1e6  # microseconds to seconds

        if dt > 0:
            dist = math.sqrt(dx*dx + dy*dy)
            speed = dist / dt
            df.loc[df.index[i], "computed_speed"] = speed
        else:
            # 時間差 = 0，避免除以 0
            df.loc[df.index[i], "computed_speed"] = 0.0

    return df
                        
# def compare_attacker_speed(ours_folder, strive_folder, save_folder):
#     # ours_files = glob.glob(os.path.join(ours_folder))

#     results = []  # 用來儲存比較結果

#     for filename in tqdm(sorted(os.listdir(ours_folder))):
#         # filename = os.path.basename(ours_csv)  # e.g. data_hcis_v4.5_trainval_boston-seaport_HO_scene-0064_5-0-...
#         strive_csv = os.path.join(strive_folder, filename)

#         # print(filename, os.path.exists(strive_csv))

#         if not os.path.exists(strive_csv):
#             continue
        
#         scenario_type = filename.split('_')[5]
#         if scenario_type != 'HO':
#             continue
#         df_ours = pd.read_csv(os.path.join(ours_folder + '/' + filename))
#         df_strive = pd.read_csv(strive_csv)

        
#         attacker_id = filename.split('_')[9].split('.')[0]
#         df_ours_attacker = df_ours[df_ours["TRACK_ID"] == attacker_id].copy()
#         df_strive_attacker = df_strive[df_strive["TRACK_ID"] == attacker_id].copy()

#         # print("ours:", df_ours_attacker["X"], df_ours_attacker["Y"])
#         # print("strive:", df_strive_attacker["X"], df_ours_attacker["Y"])
        
#         # 依 TIMESTAMP 排序
#         # df_ours_attacker.sort_values("TIMESTAMP", inplace=True)
#         # df_strive_attacker.sort_values("TIMESTAMP", inplace=True)

#         df_ours_attacker = compute_attacker_speed_from_positions(df_ours_attacker)
#         df_strive_attacker = compute_attacker_speed_from_positions(df_strive_attacker)

#         # 計算平均速度
#         avg_speed_ours = df_ours_attacker["computed_speed"].mean()
#         avg_speed_strive = df_strive_attacker["computed_speed"].mean()

#         # 儲存結果
#         results.append({
#             "scene_file": filename,
#             "ours_avg_speed": avg_speed_ours,
#             "strive_avg_speed": avg_speed_strive,
#             "ours_rows": len(df_ours_attacker),
#             "strive_rows": len(df_strive_attacker)
#         })

#         # === 繪製速度隨時間的比較圖 ===
#         # 這裡假設 TIMESTAMP 單位是微秒, 轉成秒
#         # 也可直接用 RangeIndex(0, n) 當 x 軸
#         ours_t = (df_ours_attacker["TIMESTAMP"] - df_ours_attacker["TIMESTAMP"].min())/1e6
#         strive_t = (df_strive_attacker["TIMESTAMP"] - df_strive_attacker["TIMESTAMP"].min())/1e6

#         plt.figure(figsize=(8,6))
#         plt.plot(ours_t, df_ours_attacker["computed_speed"], label="Ours Attacker", color='red')
#         plt.plot(strive_t, df_strive_attacker["computed_speed"], label="STRIVE Attacker", color='blue')
#         plt.xlabel("Time (s)")
#         plt.ylabel("Speed (m/s)")  # 如果 V 單位確定是 m/s
#         plt.title(f"Attacker Speed Comparison - {filename}")
#         plt.legend()
#         plt.grid(True, linestyle="--", alpha=0.5)

#         # 可在這裡把圖存檔 或 plt.show()
#         plt.savefig(save_folder + filename + ".png")
#         # plt.show()
    
#     # === 印出彙總結果 ===
#     result_df = pd.DataFrame(results)
#     print("=== Comparison Results ===")
#     print(result_df)
#     result_df.to_csv("speed_comparison_HO.csv", index=False)

#     # 可另外算整體平均
#     if not result_df.empty:
#         overall_ours = result_df["ours_avg_speed"].mean()
#         overall_strive = result_df["strive_avg_speed"].mean()
#         print(f"\nOverall average speed across all scenes => Ours: {overall_ours:.3f}, STRIVE: {overall_strive:.3f}")

def compare_attacker_speed(ours_folder, strive_folder, save_folder):
    # ours_files = glob.glob(os.path.join(ours_folder))

    results = []  # 用來儲存比較結果
    target_type = 'JC'

    for filename in tqdm(sorted(os.listdir(ours_folder))):
        # filename = os.path.basename(ours_csv)  # e.g. data_hcis_v4.5_trainval_boston-seaport_HO_scene-0064_5-0-...
        strive_csv = os.path.join(strive_folder, filename)

        # print(filename, os.path.exists(strive_csv))

        if not os.path.exists(strive_csv):
            continue
        
        scenario_type = filename.split('_')[5]
        if scenario_type != target_type:
            continue
        attacker_id = filename.split('_')[9].split('.')[0]
        df_ours = pd.read_csv(os.path.join(ours_folder + '/' + filename))
        df_strive = pd.read_csv(strive_csv)

        collision_flag_ours, _, attacker_right_flag_ours, _, collision_moment_ours = cal_cr_and_similarity(df_ours, attacker_id)
        collision_flag_strive, _, attacker_right_flag_strive, _, collision_moment_strive = cal_cr_and_similarity(df_strive, attacker_id)

        avg_speed_ours = None
        avg_speed_strive = None
        collision_speed_ours = None
        collision_speed_strive = None
        if attacker_right_flag_ours:
            df_ours_attacker = df_ours[df_ours["TRACK_ID"] == attacker_id].copy()
            df_ours_attacker = compute_attacker_speed_from_positions(df_ours_attacker)
            avg_speed_ours = df_ours_attacker["computed_speed"].mean()
            collision_speed_ours = df_ours_attacker.iloc[collision_moment_ours - 1]["computed_speed"]
            # dx = df_ours_attacker.iloc[collision_moment_ours - 1]["X"] - df_ours_attacker.iloc[collision_moment_ours - 2]["X"]
            # dy = df_ours_attacker.iloc[collision_moment_ours - 1]["Y"] - df_ours_attacker.iloc[collision_moment_ours - 2]["Y"]
            # dt = 0.5
            # dist = math.sqrt(dx*dx + dy*dy)
            # speed = dist / dt
            # df.loc[df.index[i], "computed_speed"] = speed
            
        
        if attacker_right_flag_strive:
            df_strive_attacker = df_strive[df_ours["TRACK_ID"] == attacker_id].copy()
            df_strive_attacker = compute_attacker_speed_from_positions(df_strive_attacker)
            avg_speed_strive = df_strive_attacker["computed_speed"].mean()
            collision_speed_strive = df_strive_attacker.iloc[collision_moment_strive - 1]["computed_speed"]
            # print(df_strive_attacker.X, df_strive_attacker["computed_speed"])
        
        # 儲存結果
        results.append({
            "scene_file": filename,
            "ours_avg_speed": avg_speed_ours,
            "strive_avg_speed": avg_speed_strive,
            "ours_collision_speed": collision_speed_ours,
            "strive_collision_speed": collision_speed_strive,
            "collision_moment_ours": collision_moment_ours,
            "collision_moment_strive": collision_moment_strive
                            })

        # === 繪製速度隨時間的比較圖 ===
        # 這裡假設 TIMESTAMP 單位是微秒, 轉成秒
        # 也可直接用 RangeIndex(0, n) 當 x 軸
        if attacker_right_flag_ours and attacker_right_flag_strive:
            # print(filename)
            # exit()
            ours_t = (df_ours_attacker["TIMESTAMP"] - df_ours_attacker["TIMESTAMP"].min())/1e6
            strive_t = (df_strive_attacker["TIMESTAMP"] - df_strive_attacker["TIMESTAMP"].min())/1e6

            plt.figure(figsize=(8,6))
            plt.plot(ours_t, df_ours_attacker["computed_speed"], label="Ours Attacker", color='red')
            plt.plot(strive_t, df_strive_attacker["computed_speed"], label="STRIVE Attacker", color='blue')
            plt.xlabel("Time (s)")
            plt.ylabel("Speed (m/s)")  # 如果 V 單位確定是 m/s
            plt.title(f"Attacker Speed Comparison - {filename}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)

            # 可在這裡把圖存檔 或 plt.show()
            plt.savefig(save_folder + filename + ".png")
            # plt.show()
    
    # === 印出彙總結果 ===
    result_df = pd.DataFrame(results)
    print("=== Comparison Results ===")
    print(result_df)
    result_df.to_csv("speed_comparison_" + target_type + ".csv", index=False)

    # 可另外算整體平均
    if not result_df.empty:
        overall_ours = result_df["ours_avg_speed"].mean()
        overall_strive = result_df["strive_avg_speed"].mean()
        print(f"\nOverall average speed across all scenes => Ours: {overall_ours:.3f}, STRIVE: {overall_strive:.3f}")
        overall_ours_collision_speed = result_df["ours_collision_speed"].mean()
        overall_strive_collision_speed = result_df["strive_collision_speed"].mean()
        print(f"\nOverall Collision_speed across all scenes => Ours: {overall_ours_collision_speed:.3f}, STRIVE: {overall_strive_collision_speed:.3f}")
        overall_collision_moment_ours = result_df["collision_moment_ours"].mean()
        overall_collision_moment_strive = result_df["collision_moment_strive"].mean()
        print(f"\nOverall Collision moment across all scenes => Ours: {overall_collision_moment_ours:.3f}, STRIVE: {overall_collision_moment_strive:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    mode = 'STRIVE generation'

    if mode == 'Our generation':
        parser.add_argument('--data_path', type=str, default='output_csv_moving_foward_interpolation_7_24/')
        parser.add_argument('--data_path_2', type=str, default='../TNT_Nuscenes_ranking/output_csv_moving_foward_interpolation/')
        parser.add_argument('--save_path', type=str, default='animation/')
        parser.add_argument('--figure_save_path', type=str, default='paper_figure/')
        parser.add_argument('--interpolation_frame', type=int, default=5)
    elif mode == 'Our':
        # parser.add_argument('--data_path', type=str, default='idm_traj/') 
        # parser.add_argument('--save_path', type=str, default='animation_idm/')
        parser.add_argument('--data_path', type=str, default='idm_traj_new_v/') 
        parser.add_argument('--save_path', type=str, default='animation_idm_new_v/')
        parser.add_argument('--figure_save_path', type=str, default='paper_figure/')
        parser.add_argument('--interpolation_frame', type=int, default=5)
    elif mode == 'STRIVE':
        # parser.add_argument('--data_path', type=str, default='idm_traj_on_STRIVE/') 
        # parser.add_argument('--save_path', type=str, default='animation_idm_on_STRIVE/')
        parser.add_argument('--data_path', type=str, default='idm_traj_on_STRIVE_new_v/') 
        parser.add_argument('--save_path', type=str, default='animation_idm_on_STRIVE_new_v/')
        parser.add_argument('--interpolation_frame', type=int, default=5)
    elif mode == 'Our collision data':
        parser.add_argument('--data_path', type=str, default='yaw_distribution/4.5_4.6_4.7_after_collide_with_other_check/')
        parser.add_argument('--save_path', type=str, default='animation_raw_data_collision_data/')
        parser.add_argument('--figure_save_path', type=str, default='paper_figure_collision_data/')
        parser.add_argument('--interpolation_frame', type=int, default=1)
    elif mode == 'Initial nuscenes':
        parser.add_argument('--data_path', type=str, default='../init-trainval/') 
        parser.add_argument('--save_path', type=str, default='animation_init_nuscenes/')
        parser.add_argument('--mapping_town_path', type=str, default='output_csv_moving_foward_interpolation_7_24/')
        parser.add_argument('--figure_save_path', type=str, default='paper_figure/')
        parser.add_argument('--interpolation_frame', type=int, default=1)
    elif mode == 'STRIVE generation':
        parser.add_argument('--data_path', type=str, default='nuscenes_csv_result_on_STRIVE/')
        parser.add_argument('--save_path', type=str, default='animation_raw_data_' + mode + '/')
        parser.add_argument('--figure_save_path', type=str, default='paper_figure_' + mode + '/')
        parser.add_argument('--interpolation_frame', type=int, default=1)

    
    
    # parser.add_argument('--future_length', default='12',
    #                     type=int)
    parser.add_argument('--past_len', default='8',
                        type=int)
    parser.add_argument('--map_path', type=str, default='./NuScenes/')
    parser.add_argument('--topo_folder', type=str, default='nuscenes_data/initial_topology/')
    parser.add_argument('--plot', default=1, type=int)
    # parser.add_argument('--interpolation_frame', default=5, type=int)
    args = parser.parse_args()
    # inference
    # main(args.data_path, mode, args.past_len, args.interpolation_frame, args.save_path, args.topo_folder, args.map_path)
    # only_metric_from_tnt_trainer(args.data_path, mode, args.past_len, args.interpolation_frame, args.save_path, args.topo_folder, args.map_path)
    
    # plot_paper_figure(args, mode)
    compare_attacker_speed(
    ours_folder="output_csv_7_24", #"output_csv_foward_no_interp",
    strive_folder="nuscenes_csv_result_on_STRIVE",
    save_folder="compare_speed_plots/"
)
