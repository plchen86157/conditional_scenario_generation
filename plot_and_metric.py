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
# from nuscenes.utils.geometry_utils import Box

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

def main(data_source, past_len, if_plot, sav_folder, topo_folder, map_path):
    vehicle_length = 4.7
    vehicle_width = 2
    interpolation_frame = 5
    folder = data_source
    for scenario_name in sorted(os.listdir(folder)):
        town_name = scenario_name.split('_')[4]
        split_name = scenario_name.split('_')
        if scenario_name != 'data_hcis_v4.5_trainval_boston-seaport_LTAP_scene-0207_3-5-0.0-10_90.0_c930c758a6d94d41becc47266fd23988_foward.csv':
            continue
        # if split_name[6].split('-')[1] != '0660':
        #    continue
        print(scenario_name)
        dir_name = split_name[5] + '_' + split_name[6] + '_' + split_name[7] + '_' + split_name[8] + '_' + split_name[9]
        sav_path = sav_folder + dir_name +  '/'
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)

        traj_df = pd.read_csv(os.path.join(
            data_source + scenario_name))
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
            
            
            # fig, ax = plt.subplots()
            # ax.set_xlabel("x axis(m)")
            # ax.set_ylabel("y axis(m)")

        
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
        attacker_id = scenario_name.split('_')[9]
        
        for track_id, remain_df in traj_df.groupby('TRACK_ID'):
            fill_dict[track_id] = []
            if str(track_id) == 'ego':
                ego_list.append(remain_df.reset_index())
        scenario_length = len(vehicle_list[0])

        for t in range(1, scenario_length + 1):
            nusc_map = NuScenesMap(dataroot=map_path, map_name=town_name)
            fig, ax = nusc_map.render_layers(["drivable_area"])
            print(initial_name, t)
            # plt.figure(dpi=500)
            # plt.xlabel("x axis(m)")
            # plt.ylabel("y axis(m)")
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
            
            # ego, = ax.plot(
            #     [0, 0], label="detected collision", color='red', linestyle='--')
            # first_legend = ax.legend(
            #     handles=[ego], loc='lower left', bbox_to_anchor=(0.6, 0.3))
            # ax.add_artist(first_legend)
            # black_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
            #                                     edgecolor='black', label='agents')
            # black_legend = ax.legend(
            #     handles=[black_patch], loc='lower left', bbox_to_anchor=(0.6, 0.2))
            # ax.add_artist(black_legend)
            # darkgreen_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
            #                                         edgecolor='lime', label='ego')
            # darkgreen_legend = ax.legend(
            #     handles=[darkgreen_patch], loc='lower left', bbox_to_anchor=(0.6, 0.1))
            # ax.add_artist(darkgreen_legend)
            # purple_patch = mpatches.Rectangle([0, 0], 0, 0, facecolor='white',
            #                                     edgecolor='violet', label='risk')
            # ax.legend(
            #     handles=[purple_patch], loc='lower left', bbox_to_anchor=(0.6, 0))
        
            
            # ego, = ax.plot([0, 0], [0, 0], '--o',
            #                 color='blue', markersize=1)
            # agent, = ax.plot([0, 0], [0, 0], '--o',
            #                 color='red', markersize=1)
            # vehicle, = ax.plot([0, 0], [0, 0], '--o',
            #                     color='green', markersize=1)
            # ax.legend([ego, agent, vehicle], [
            #         "ego", "attacker", "vehicle"])

            # for features in lane_feature:
            #     xs, ys = np.vstack((features[0][:, :2], features[0][-1, 3:5]))[
            #         :, 0], np.vstack((features[0][:, :2], features[0][-1, 3:5]))[:, 1]
            #     plt.plot(xs, ys, '-.', color='lightgray')
            #     x_s, y_s = np.vstack((features[1][:, :2], features[1][-1, 3:5]))[
            #         :, 0], np.vstack((features[1][:, :2], features[1][-1, 3:5]))[:, 1]
            #     plt.plot(x_s, y_s, '-.', color='lightgray')
            
            # ego_fill_bewteen_np = np.zeros((len(ego_list[0]),2))
            
            
            
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

            # ego_y_list = [841, 842, 843]
            # left_x_list = [1702, 1703, 1704]
            # right_x_list = [1706, 1707, 1708]
            # plt.fill_betweenx(ego_y, left_x, right_x, color='gray', alpha=0.5)
            # plt.fill_between(ego_y_list, left_x_list, right_x_list, color='blue', alpha=0.5)
            
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
                other_polygon = Polygon(other_coords)
                # other_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
                # other_pg = pg([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]], facecolor = 'k')
                
                ax.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                fill_dict[now_id].append(np.array([[x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3]]))

                if now_id == attacker_id:
                    print(ego_x, ego_y, real_pred_x, real_pred_y, attacker_id)
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
            # ax.set_xlim(forever_present_x - 100,
            #         forever_present_x + 100)
            # ax.set_ylim(forever_present_y - 100,
            #         forever_present_y + 100)
            # ax.set_clip_box([[forever_present_x - 50, forever_present_y - 50], [forever_present_x + 50, forever_present_y]])
            
            if right_attacker_flag:
                right_attacker_flag_str = "Collide with attacker!"
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


def metric(args):
    #all_cnt = {"junction_crossing": 0, "LTAP": 0, "TCD_violation": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    all_cnt = {"junction_crossing": 0, "LTAP": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    col_cnt = {"junction_crossing": 0, "LTAP": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    inside_cnt = {"junction_crossing": 0, "LTAP": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    distance_cnt = {"junction_crossing": 0, "LTAP": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    folder = args.data_path
    all_data_num = 0
    col_data_num = 0
    yaw_offset_degree = 30
    for scenario_name in sorted(os.listdir(folder)):
        for variant in sorted(os.listdir(folder + scenario_name + '/')):
            #print(scenario_name, variant)
            all_cnt[variant] += 1
            all_data_num += 1
            txt_file = folder + scenario_name + '/' + variant + '/collsion_description/' + scenario_name + '.txt'
            if variant == 'lane_change' or variant == 'TCD_violation':
                ideal_yaw_distance = 15
            elif variant == 'junction_crossing' or variant == 'LTAP':
                ideal_yaw_distance = 90
            elif variant == 'opposite_direction':
                ideal_yaw_distance = 180
            elif variant == 'rear_end':
                ideal_yaw_distance = 0
            if os.path.isfile(txt_file):
                col_data_num += 1
                col_cnt[variant] += 1
                traj_df = pd.read_csv(txt_file, sep='\t', header=None)
                ego_end_yaw = traj_df[0].values[0]
                tp_end_yaw = traj_df[1].values[0]
                real_yaw_distance = abs(abs(ego_end_yaw) - abs(tp_end_yaw))
                distance = abs(abs(real_yaw_distance) - abs(ideal_yaw_distance))
                distance_cnt[variant] += distance
                if distance <= yaw_offset_degree:
                    print(scenario_name, variant)
                    inside_cnt[variant] += 1
    for variant_key in all_cnt:
        distance_cnt[variant_key] /= col_cnt[variant_key]
    
    print("all:", all_cnt)
    print("collide:", col_cnt)
    print("inside similarity:", inside_cnt)
    print("distance similarity:", distance_cnt)
    all_df = pd.DataFrame(all_cnt, index=['all'])
    col_df = pd.DataFrame(col_cnt, index=['collision_scenarios'])
    #cr_df = pd.DataFrame(col_cnt, index=['collision_rate'])
    cr_df = col_df.copy()
    for variant_key in cr_df:
        cr_df[variant_key] = col_cnt[variant_key] / all_cnt[variant_key]
    inside_df = pd.DataFrame(inside_cnt, index=['similarity(inside)'])
    ir_df = inside_df.copy()
    for variant_key in ir_df:
        if int(col_cnt[variant_key]) == 0:
            ir_df[variant_key] = 0
        else:
            ir_df[variant_key] = inside_cnt[variant_key] / col_cnt[variant_key]
    print(ir_df)
    distance_df = pd.DataFrame(distance_cnt, index=['similarity(distance)'])
    result = pd.concat([all_df, col_df, cr_df, inside_df, ir_df, distance_df]).T
    result.columns = ['all', 'collision_scenarios', 'collision_rate', 'similarity(inside)', 'similarity(inside rate)', 'similarity(distance)']
    print(result)
    result.to_csv('social-gan_metric.csv')

if __name__ == "__main__":
    # x = np.array([1, 2, 3, 4, 5])  # 例如，车辆轨迹的 x 坐标
    # y = np.array([2, 3, 4, 5, 6])
    # car_length = 4  # 车辆长度
    # car_width = 2  # 车辆宽度

    # # 计算车辆轨迹左右两侧的坐标
    # left_x = x - 0.5 * car_length
    # right_x = x + 0.5 * car_length
    # bottom_y = y - 0.5 * car_width
    # top_y = y + 0.5 * car_width

    # # 创建图形并绘制轨迹
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, y, color='blue', linestyle='-', marker='o', markersize=5, label='Trajectory')

    # print(y, left_x, right_x)
    # # 使用 fill_between 方法创建渐变区域
    # plt.fill_betweenx(y, left_x, right_x, color='gray', alpha=0.5)

    # # 设置图形标题和标签
    # plt.title('Car Trajectory with Gradient Area')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()

    # # 显示图形
    # plt.grid(True)
    # plt.show()
    # exit()



    # nusc_map = NuScenesMap(dataroot='./NuScenes/', map_name='boston-seaport')
    # xmin, ymin, xmax, ymax = 0, 0, 2000, 2000
    # # clip_box = Box([xmin, ymin, 0], [xmax - xmin, ymax - ymin, 100], axis_aligned=True)
    # # map_patch = nusc_map.render_map_patch(clip_box=clip_box)
    # sample_drivable_area = nusc_map.drivable_area[0]
    # # fig, ax = nusc_map.render_record('drivable_area', sample_drivable_area['token'], other_layers=[])
    # fig, ax = nusc_map.render_layers(["drivable_area"])
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # plt.show()
    # exit()


    parser = argparse.ArgumentParser()
    # parser.add_argument('--future_length', default='12',
    #                     type=int)
    parser.add_argument('--past_length', default='8',
                        type=int)
    parser.add_argument('--map_path', type=str, default='./NuScenes/')
    parser.add_argument('--data_path', type=str, default='output_csv_moving_foward_interpolation/')
    parser.add_argument('--save_path', type=str, default='animation/')
    parser.add_argument('--topo_folder', type=str, default='nuscenes_data/initial_topology/')
    parser.add_argument('--plot', default=1, type=int)
    args = parser.parse_args()
    main(args.data_path, args.past_length, args.plot, args.save_path, args.topo_folder, args.map_path)
    #metric(args)
