import pandas as pd
import numpy as np
import os
from collections import defaultdict
import shutil
import sys
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Polygon as pg
import math
from nuscenes.map_expansion.map_api import NuScenesMap

VEH_COLL_THRESH = 0.02

def replace_less_than_001(value):
    #print("modify", type(value), value)
    return 0.0 if abs(float(value)) < 0.001 else value

def is_intersection(x, y, town_name):
    nusc_map = NuScenesMap(dataroot='./NuScenes/', map_name=town_name)
    rstk = nusc_map.record_on_point(x, y, "road_segment")
    if rstk == "":
        return False
    rs = nusc_map.get("road_segment", rstk)
    return rs["is_intersection"]

def data_proc():
    if_quintic_no_collision_filter = True
    data_path = '../csvs_4.5/' #'../csvs/' #'../csvs_4.5/'
    output_path = 'nuscenes_data/'
    train_split = 0.7
    val_split = 0.1
    filter_num = 0
    nearest_target_to_ego_collision_dist_too_far_dist = 2
    filter_num_nearest_target_to_ego_collision_dist_too_far = 0
    max_tar_candts_dist_too_far_dist = 150
    filter_num_max_tar_candts_dist_too_far_dist = 0
    f_8_vehicles = 0
    scenario_num = 0
    filter_quintic_no_collision = 0
    JC_yaw_distribution = []
    LTAP_yaw_distribution = []
    LC_yaw_distribution = []
    HO_yaw_distribution = []
    RE_yaw_distribution = []
    all_yaw_distribution = []

    JC_yaw_last_f_distribution = []
    LTAP_yaw_last_f_distribution = []
    LC_yaw_last_f_distribution = []
    HO_yaw_last_f_distribution = []
    RE_yaw_last_f_distribution = []
    all_yaw_last_f_distribution = []

    JC_ideal_yaw = 90
    LTAP_ideal_yaw = 90
    LC_ideal_yaw = 20
    HO_ideal_yaw = 180
    RE_ideal_yaw = 0

    yaw_threshold = 10

    new_LTAP_intersection_filter = 0
    new_JC_intersection_filter = 0
    intersection_filtered = 0
    stop_ego_from_origin = 0
    not_in_5 = 0

    collsion_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    now_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    # 4.5: 'JC': 10895, 'LTAP': 1610, 'LC': 2949, 'HO': 3788, 'RE': 4204
    # ratio_cnt = {"JC": 0.15, "LTAP": 1.0, "LC": 1.0, "HO": 0.5, "RE": 0.4}
    ratio_cnt = {"JC": 1.0, "LTAP": 1.0, "LC": 1.0, "HO": 1.0, "RE": 1.0}
    ttc_list = []
    for scenario in sorted(os.listdir(data_path)):
        type_name = scenario.split('_')[5]
        collsion_cnt[type_name] += 1
        variant_id = scenario.split('_')[7]#scenario.split('.')[1].split('_')[5]
        ttc = int(variant_id.split('-')[-1]) - 8 + 1
        ttc_list.append(ttc)
    # print(collsion_cnt)
    
    collsion_cnt_filtered = collsion_cnt 
    ttc_range_len = max(ttc_list) - min(ttc_list)
    print("4.5 min ttc:", min(ttc_list), " ttc_range_length:", ttc_range_len)
    # for scenario in tqdm(sorted(os.listdir(data_path))):
    for scenario in tqdm(os.listdir(data_path)):
        # print(scenario.split('.')[1])
        type_name = scenario.split('_')[5]
        # if type_name != 'LC':
        #     continue
        if scenario != 'data_hcis_v4.5_trainval_boston-seaport_HO_scene-0906_5-0-16_a23c577400794ca9a1e4846873c2d23f.csv':
            continue
        # print(scenario, scenario_num)
        
        scenario_type = scenario.split('_')[5]
        variant_id = scenario.split('.')[1].split('_')[5]
        now_ttc = int(variant_id.split('-')[-1]) - 8 + 1
        normalize_ttc = (now_ttc - min(ttc_list)) / ttc_range_len

        
        

        if now_cnt[scenario_type] > collsion_cnt_filtered[scenario_type] * ratio_cnt[scenario_type]:
            continue

        df = pd.read_csv(data_path + scenario)

        ######################
        # obs_horizon = 8
        # for track_id, remain_df in df.groupby('TRACK_ID'):
        #     if track_id == 'ego':
        #         ego_traj = np.concatenate((
        #             remain_df.X.to_numpy().reshape(-1, 1),
        #             remain_df.Y.to_numpy().reshape(-1, 1)), 1)
        #         orig = ego_traj[obs_horizon - 1]
        # pre = (orig - ego_traj[obs_horizon-4]) / 2.0
        # theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
        # rot = np.asarray([
        #     [np.cos(theta), -np.sin(theta)],
        #     [np.sin(theta), np.cos(theta)]], np.float32)
        # topology_id = scenario.split('_')[3] + '_' + scenario.split('_')[4] + '_' + scenario.split('_')[6]
        # topology = np.load("./nuscenes_data/initial_topology/" + topology_id + ".npy", allow_pickle=True)
        # ctr_line_candts_list = []
        # if len(topology) == 0:
        #     continue
        # t_sum = 0
        # for i, _ in enumerate(topology):
        #     t_sum += topology[:, 2][i][:, :2].shape[0]
        #     t = np.matmul(rot, (topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T).T            
        #     if len(t) <= 1:
        #         tmp = (np.matmul(rot, (topology[:, 2][i][:, 3:5] - orig.reshape(-1, 2)).T).T)[0]
        #         t = np.vstack((t[0], tmp))
        #     ctr_line_candts_list.append(t)
        # ctr_line_candts = np.array(ctr_line_candts_list, dtype=object)
        
        # candidates = []
        # for line_id, line in enumerate(ctr_line_candts):
        #     for i in range(len(line) - 1):
        #         # print(i, "line:", line[i])
        #         # if np.any(np.isnan(line[i])) or np.any(np.isnan(line[i+1])):
        #         #     continue
        #         [x_diff, y_diff] = line[i+1] - line[i]
        #         if x_diff == 0.0 and y_diff == 0.0:
        #             continue
        #         candidates.append(line[i])
        
        # if len(candidates) == 0:
        #     continue
        
        # agt_traj_fut = ego_traj[obs_horizon:].copy().astype(np.float32)
        # agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T

        # candidates = np.unique(np.asarray(candidates), axis=0)
        # max_tar_candts_dist = np.max(candidates)
        # if max_tar_candts_dist > max_tar_candts_dist_too_far_dist:
        #     filter_num_max_tar_candts_dist_too_far_dist += 1
        #     collsion_cnt_filtered[type_name] -= 1
        #     continue

        # displacement = candidates - agt_traj_fut[-1]
        # dist = np.sqrt(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))
        # nearest_target_to_ego_collision_dist = min(dist)
        # if nearest_target_to_ego_collision_dist > nearest_target_to_ego_collision_dist_too_far_dist:
        #     filter_num_nearest_target_to_ego_collision_dist_too_far += 1
        #     collsion_cnt_filtered[type_name] -= 1
        #     continue

        
        #print(scene_name)
        # if scene_name != 'scene-0163':
        #     continue
        # if variant_name != '5-0-25':
        #     continue
        with open('/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/filter_agent.json') as f:
            data = json.load(f)
        attacker_id = scenario.split('_')[-1].split('.')[0]
        scene_id = scenario.split('_')[6]
        variant_id = scenario.split('_')[7]
        sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
        #df = pd.read_csv(data_path + 'all_traj/' + scenario, index_col=0)
        
        # for track_id, remain_df in df.groupby('TRACK_ID'):
        #     if track_id == attacker_id:
        #         t = np.array((remain_df.TIMESTAMP.values, remain_df.X.values)).T
        #         print(t)
        #print(df)
        df.iloc[:,5] *= 180
        df.iloc[:,5] /= np.pi
        df = (df.pivot_table(columns='TIMESTAMP', index=['TRACK_ID'], fill_value=0)
                .stack('TIMESTAMP')
                .sort_index(level=['TRACK_ID','TIMESTAMP'])
                .reset_index())
        if sce_temp in data:
            #print("filter:", list(data[sce_temp]), df)
            for idx in range(len(list(data[sce_temp]))):
                #print(list(data[sce_temp])[idx])
                parked_idx = df[df["TRACK_ID"] == list(data[sce_temp])[idx]].index
                df = df.drop(parked_idx).reset_index(drop=True)
        objs = df.groupby(['TRACK_ID']).groups
        keys = list(objs.keys())
        if attacker_id not in keys:
            filter_num += 1
            collsion_cnt_filtered[type_name] -= 1
            # print("filter")
            continue
        vehicle_list = []
        vehicle_pd = pd.DataFrame()
        for track_id, remain_df in df.groupby('TRACK_ID'):
            x_8 = remain_df.X.values[8]
            y_8 = remain_df.Y.values[8]
            if x_8 == 0 and y_8 == 0:
                f_8_vehicles += 1
                #print("frame 8 filter")
            else:
                vehicle_pd = pd.concat([vehicle_pd, remain_df], axis=0)
            #vehicle_list.append(remain_df)
            
        
        # trajs = np.concatenate((
        #     df.X.to_numpy().reshape(-1, 1),
        #     df.Y.to_numpy().reshape(-1, 1)), 1)
        # obs_horizon = 8
        # filter_flag = 0
        # for key in keys:
        #     idcs = objs[key]
        #     if trajs[idcs][obs_horizon - 1][0] == 0 and trajs[idcs][obs_horizon - 1][1] == 0:

        columns_to_update = ['X', 'Y', 'V', 'YAW']
        vehicle_pd[columns_to_update] = vehicle_pd[columns_to_update].applymap(replace_less_than_001)
        if if_quintic_no_collision_filter:
            save_folder = './yaw_distribution/'
            stop_ego_flag = 0
            frame_num = len(set(vehicle_pd.TIMESTAMP.values))
            frame_multiple = 5
            further_df = vehicle_pd
            for track_id, remain_df in vehicle_pd.groupby("TRACK_ID"):
                ##### only for analyze #####
                if track_id == 'ego':
                    ego_last_f_yaw = remain_df.iloc[frame_num-1, 5] + 360.0 if remain_df.iloc[frame_num-1, 5] < 0 else remain_df.iloc[frame_num-1, 5]
                elif track_id == attacker_id:
                    attacker_last_f_yaw = remain_df.iloc[frame_num-1, 5] + 360.0 if remain_df.iloc[frame_num-1, 5] < 0 else remain_df.iloc[frame_num-1, 5]
                ##### only for analyze #####
                #print(track_id, tp_index[s].cpu().numpy()[0], remain_df)
                more_frames = 4
                # trajectory may be a curve, so it should rely on last frame and the previous frame
                dis_x = (remain_df.iloc[frame_num-1, 3] - remain_df.iloc[frame_num-2, 3]) / frame_multiple
                dis_y = (remain_df.iloc[frame_num-1, 4] - remain_df.iloc[frame_num-2, 4]) / frame_multiple
                # print("x:", remain_df.iloc[frame_num-1, 3], remain_df.iloc[frame_num-2, 3])
                for frame_index in range(frame_num):
                    # if track_id == 'ego':
                    #     print("v:", remain_df.iloc[frame_index-1, 2])
                        # print("dis:", dis_x, dis_y)
                    if track_id == 'ego' and remain_df.iloc[frame_index-1, 2] < 2: #dis_x < 0.001 and dis_y < 0.001:
                        stop_ego_flag = 1
                # for padding vehicle
                if dis_x > 10 or dis_y > 10:
                    dis_x, dis_y = 0, 0 
                all_x, all_y, all_v, all_yaw = [], [], [], []
                for f_index in range(more_frames * frame_multiple):
                    all_v.append(remain_df.iloc[frame_num-1, 2])
                    all_x.append(remain_df.iloc[frame_num-1, 3] + dis_x * (f_index + 1))
                    all_y.append(remain_df.iloc[frame_num-1, 4] + dis_y * (f_index + 1))
                    all_yaw.append(remain_df.iloc[frame_num-1, 5])
                for further_t in range(more_frames * frame_multiple):
                    b = {'TIMESTAMP': [remain_df.TIMESTAMP.values[-1] + (further_t + 1) * 500000 / frame_multiple], 'TRACK_ID': [track_id],
                        # 'V': [all_v[further_t]], 'X': [x[further_t].cpu().numpy()], 'Y': [y[further_t].cpu().numpy()],
                        'V': [all_v[further_t]], 'X': [all_x[further_t]], 'Y': [all_y[further_t]],
                        'YAW': [all_yaw[further_t]]}
                    df_insert = pd.DataFrame(b)
                    further_df = pd.concat([further_df, df_insert], ignore_index=True)
                # if track_id == attacker_id:
                #     print(vehicle_pd, further_df)
                #     exit()
            if stop_ego_flag:
                stop_ego_from_origin += 1
                continue
            plt.close()
            collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = cal_cr_and_similarity(further_df, attacker_id)
            
            

            last_f_yaw_distance = ego_last_f_yaw - attacker_last_f_yaw
            while last_f_yaw_distance < 0:
                last_f_yaw_distance = (last_f_yaw_distance + 360.0)
            last_f_yaw_distance = abs(last_f_yaw_distance - 360.0) if last_f_yaw_distance > 180 else last_f_yaw_distance
            
            # print("Type:", type_name, "last_f_yaw_distance:", last_f_yaw_distance)
            if type_name == 'JC':
                JC_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 90))
            elif type_name == 'LTAP':
                LTAP_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 90))
            elif type_name == 'LC':
                LC_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 15))
            elif type_name == 'HO':
                HO_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 180))
            elif type_name == 'RE':
                RE_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 0))
            # print("Type:", type_name, "last_f_yaw_distance:", last_f_yaw_distance)
            if collision_flag:
                while record_yaw_distance < 0:
                    record_yaw_distance = (record_yaw_distance + 360.0)
                record_yaw_distance = abs(record_yaw_distance - 360.0) if record_yaw_distance > 180 else record_yaw_distance
                print(collision_flag)
                if attacker_right_flag:
                    print(attacker_right_flag)
                    new_type_name = None
                    # if type_name == 'JC':
                    #     JC_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = JC_ideal_yaw
                    # elif type_name == 'LTAP':
                    #     LTAP_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = LTAP_ideal_yaw
                    # elif type_name == 'LC':
                    #     LC_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = LC_ideal_yaw
                    # elif type_name == 'HO':
                    #     HO_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = HO_ideal_yaw
                    # elif type_name == 'RE':
                    #     RE_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = RE_ideal_yaw
                    # # yaw_distance = abs(ideal_yaw_offset - record_yaw_distance)
                    # to_ideal_yaw_dist = abs(record_yaw_distance - ideal_offset)
                    # all_yaw_distribution.append(to_ideal_yaw_dist)
                    # if to_ideal_yaw_dist > 10:
                    if True:
                        if record_yaw_distance < 10:
                            new_type_name = 'RE'
                        elif 10 < record_yaw_distance and record_yaw_distance < 30:
                            new_type_name = 'LC'
                        elif 170 < record_yaw_distance:
                            new_type_name = 'HO'
                        elif 80 < record_yaw_distance and record_yaw_distance < 100:
                            frame_num = len(set(vehicle_pd.TIMESTAMP.values))

                            intersection_threshold = 20
                            ego_list = []
                            for track_id, remain_df in vehicle_pd.groupby("TRACK_ID"):
                                if str(track_id) == 'ego':
                                    ego_list.append(remain_df.reset_index())
                            for track_id, remain_df in vehicle_pd.groupby("TRACK_ID"):
                                if track_id == attacker_id:
                                    for i in range(frame_num):
                                        
                                        x = remain_df.iloc[i, 3]
                                        y = remain_df.iloc[i, 4]
                                        yaw = remain_df.iloc[i, 5]
                                        # print(abs(yaw - ego_list[0].loc[i, 'YAW']), is_intersection(x, y, town_name=scenario.split('_')[4]))
                                        if is_intersection(x, y, town_name=scenario.split('_')[4]):
                                            continue
                                        diff_yaw = abs(yaw - ego_list[0].loc[i, 'YAW'])
                                        
                                        if abs(diff_yaw - 180) < intersection_threshold:
                                            new_LTAP_intersection_filter += 1
                                            new_type_name = 'LTAP'
                                        elif abs(diff_yaw - 90) < intersection_threshold:
                                            new_JC_intersection_filter += 1
                                            new_type_name = 'JC'
                                        else:
                                            intersection_filtered += 1
                                            new_type_name = 'filter'
                        else:
                            # print("Not in the 5 categories")
                            not_in_5 += 1
                            new_type_name = 'filter'
                            collsion_cnt_filtered[type_name] -= 1
                            continue

                    # print(scenario, new_type_name)
                    if new_type_name == None or new_type_name == 'filter':
                        # print("B", new_type_name, record_yaw_distance)
                        collsion_cnt_filtered[type_name] -= 1
                        continue
                    if new_type_name != None:
                        scenario = scenario.replace(scenario_type, new_type_name)
                    # print(new_type_name)
                    # exit()


                        
                else:
                    filter_quintic_no_collision += 1
                    collsion_cnt_filtered[type_name] -= 1
                    continue
            else:
                filter_quintic_no_collision += 1
                collsion_cnt_filtered[type_name] -= 1
                continue
            
            
            # print("real yaw:", record_yaw_distance, " ideal yaw:", last_f_yaw_distance, type_name)

            title = str(attacker_right_flag) + " real yaw:" + str(round(record_yaw_distance, 1)) + " ideal yaw:" + str(round(last_f_yaw_distance, 1))
            plt.title(title)
            plt.savefig('./yaw_distribution/figures/' + scenario + '.png')
            plt.close()

            

        # print("Now filter_quintic_no_collision:", filter_quintic_no_collision)
        scenario_num += 1
        scenario_type = new_type_name
        now_cnt[scenario_type] += 1
        source_path = data_path + 'all_traj/' + scenario
        scenario = scenario.replace("-" + str(variant_id.split('-')[-1]) + "_", "-" + str(normalize_ttc) + "-" + str(variant_id.split('-')[-1]) + "_")
        
        scenario = scenario.replace("_" + attacker_id, "_" + str(round(record_yaw_distance, 2)) + "_" + attacker_id)
        # print("filter_quintic_no_collision:", filter_quintic_no_collision)
        # print("new_LTAP_intersection_filter:", new_LTAP_intersection_filter)
        # print("new_JC_intersection_filter:", new_JC_intersection_filter)
        # print("intersection_filtered:", intersection_filtered)
        # print("stop_ego_from_origin:", stops_ego_from_origin)
        traj_path = './yaw_distribution/4.5_after_classify/' + scenario
        vehicle_pd.to_csv(traj_path, index=False)
        
        # if type_name == 'LC':
        #     if now_cnt[scenario_type] < collsion_cnt_filtered[scenario_type] * ratio_cnt[scenario_type] * train_split:
        #         traj_path = output_path + 'trajectory/train/' + scenario
        #         vehicle_pd.to_csv(traj_path, index=False)
        #     elif collsion_cnt_filtered[scenario_type] * ratio_cnt[scenario_type] * train_split < now_cnt[scenario_type] < collsion_cnt_filtered[scenario_type] * ratio_cnt[scenario_type] * (train_split + val_split):
        #         traj_path = output_path + 'trajectory/val/' + scenario
        #         vehicle_pd.to_csv(traj_path, index=False)
        #     else:
        #         traj_path = output_path + 'trajectory/test/' + scenario
        #         vehicle_pd.to_csv(traj_path, index=False)
        # shutil.copyfile(source_path, traj_path)
    print("filter:", filter_num)
    print("f_8_vehicles_average:", f_8_vehicles / scenario_num)
    print("nearest_target_to_ego_collision_dist:", filter_num_nearest_target_to_ego_collision_dist_too_far)
    print("filter_num_max_tar_candts_dist_too_far_dist:", filter_num_max_tar_candts_dist_too_far_dist)
    print("filter_quintic_no_collision:", filter_quintic_no_collision)
    print("new_LTAP_intersection_filter:", new_LTAP_intersection_filter)
    print("new_JC_intersection_filter:", new_JC_intersection_filter)
    print("intersection_filtered:", intersection_filtered)
    print("stop_ego_from_origin:", stop_ego_from_origin)
    print("not_in_5:", not_in_5)
    
    print("Origin:", collsion_cnt)
    print("After filter:", collsion_cnt_filtered)
    print("final:", now_cnt)

    # print("all_yaw_last_f_distribution:", all_yaw_last_f_distribution)
    
    plt.close()
    bins = np.linspace(-25, 25, num=50)
    # bins = np.linspace(min(all_yaw_distribution), max(all_yaw_distribution), num=50)
    hist, bins = np.histogram(all_yaw_distribution, bins=bins, density=False)
    plt.bar(bins[:-1], hist, align='center', width=0.02)
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('All Yaw Distribution')
    plt.savefig(save_folder + 'all_yaw_distribution.png')
    plt.close()

    bins = np.linspace(-1, 1, num=50)
    hist, bins = np.histogram(all_yaw_last_f_distribution, bins=bins, density=False)
    plt.bar(bins[:-1], hist, align='center', width=0.02)
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('All Ideal Yaw Distribution')
    plt.savefig(save_folder + 'all_yaw_last_f_distribution.png')
    plt.close()

    bins = np.linspace(-25, 25, num=50)
    plt.hist(all_yaw_distribution, bins=bins, alpha=0.7, label='Real')
    plt.hist(all_yaw_last_f_distribution, bins=bins, alpha=0.7, label='Ideal')
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('Yaw Distribution Comparison')
    plt.legend()
    # plt.show()
    plt.savefig(save_folder + '/Yaw Distribution Comparison.png')
    plt.close()


    bins = np.linspace(min(LC_yaw_distribution), max(LC_yaw_distribution), num=100)
    # bins = np.linspace(-25, 25, num=50)
    hist, bins = np.histogram(LC_yaw_distribution, bins=bins, density=False)
    plt.close()
    plt.bar(bins[:-1], hist, align='center', width=0.1)
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('Lane Change Yaw Distribution')
    plt.savefig(save_folder + 'LC_yaw_distribution.png')
    plt.close()

    bins = np.linspace(min(LC_yaw_last_f_distribution), max(LC_yaw_last_f_distribution), num=100)
    # bins = np.linspace(-10, 10, num=10)
    hist, bins = np.histogram(LC_yaw_last_f_distribution, bins=bins, density=False)
    plt.close()
    plt.bar(bins[:-1], hist, align='center', width=0.1)
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('Lane Change Ideal Yaw Distribution')
    plt.savefig(save_folder + 'LC_yaw_last_f_distribution.png')
    plt.close()

def angle_vectors(v1, v2):
    """ Returns angle between two vectors.  """
    # 若是車輛靜止不動 ego_vec為[0 0]
    if v1[0] < 0.0001 and v1[1] < 0.0001:
        v1_u = [1.0, 0.1]
    else:
        v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # 因為np.arccos給的範圍只有0~pi (180度)，但若cos值為-0.5，不論是120度或是240度，都會回傳120度，因此考慮三四象限的情況，強制轉180度到一二象限(因車輛和行人面積的對稱性，故可這麼做)
    #if v1_u[1] < 0:
    #    v1_u = v1_u * (-1)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if math.isnan(angle):
        return 0.0
    else:
        return angle
    
def cal_cr_and_similarity(traj_df, attacker_id_gt):
    vehicle_length = 4.7 
    vehicle_width = 2
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
    scenario_length = len(vehicle_list[0])
    for t in range(1, scenario_length):
        ego_x = ego_list[0].loc[t - 1, 'X']
        ego_x_next = ego_list[0].loc[t, 'X']
        ego_y = ego_list[0].loc[t - 1, 'Y']
        ego_y_next = ego_list[0].loc[t, 'Y']
        ego_vec = [ego_y_next - ego_y,
                            ego_x_next - ego_x]
        ego_angle = np.rad2deg(
                        angle_vectors(ego_vec, [1, 0])) * np.pi / 180
        # real_ego_angle = ego_list[0].loc[t - 1, 'YAW']
        real_ego_angle = ego_list[0].loc[t - 1, 'YAW'] + 360.0 if ego_list[0].loc[t - 1, 'YAW'] < 0 else ego_list[0].loc[t - 1, 'YAW']
        real_ego_angle = (real_ego_angle + 90.0) * np.pi / 180
        ego_rec = [ego_x_next, ego_y_next, vehicle_width
                                        , vehicle_length, real_ego_angle]
        # print("EGO:", ego_x_next, ego_y_next)
        # print(ego_list)
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
        attacker_x_next = attacker_list[0].loc[t, 'X']
        attacker_y_next = attacker_list[0].loc[t, 'Y']
        real_attacker_angle = attacker_list[0].loc[t - 1, 'YAW'] + 360.0 if attacker_list[0].loc[t - 1, 'YAW'] < 0 else attacker_list[0].loc[t - 1, 'YAW']
        real_attacker_angle = (real_attacker_angle + 90.0) * np.pi / 180
        ego_rec = [attacker_x_next, attacker_y_next, vehicle_width
                                        , vehicle_length, real_attacker_angle]
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
            real_pred_x_next = vl[t][3]
            real_pred_y = vl[t - 1][4]
            real_pred_y_next = vl[t][4]
            other_vec = [real_pred_y_next - real_pred_y,
                                    real_pred_x_next - real_pred_x]
            other_angle = np.rad2deg(
                        angle_vectors(other_vec, [1, 0])) * np.pi / 180
            real_other_angle = vl[t - 1][5] + 360.0 if vl[t - 1][5] < 0 else vl[t - 1][5]
            real_other_angle = (real_other_angle + 90.0) * np.pi / 180
            # other_angle = vl[past_len][4]
            # ego_angle = ego_list[0][4][int(filename_t) + past_len]
            #print(ego_x, ego_y, real_pred_x, real_pred_y)
            ego_rec = [real_pred_x_next, real_pred_y_next, vehicle_width
                                        , vehicle_length, real_other_angle]
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
            if str(now_id) == str(attacker_id_gt):
                # print("attacker:", real_pred_x_next, real_pred_y_next)
                plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='darkgreen', alpha=1)
            else:
                attacker_other_iou = attacker_polygon.intersection(other_polygon).area / attacker_polygon.union(other_polygon).area
                if attacker_other_iou > VEH_COLL_THRESH:
                    collision_flag = 1
                    # print(now_id, attacker_other_iou, real_pred_x_next, real_pred_y_next)
            cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
            # print(t, now_id, ego_polygon.intersection(other_polygon).area, "GT:", attacker_id_gt)
            
            if cur_iou > VEH_COLL_THRESH:
                # print(attacker_id_gt, "COLLIDE!", now_id)
                collision_flag = 1
                if str(now_id) == str(attacker_id_gt):
                    # print("attacker", real_pred_x_next, real_pred_y_next)
                    # plt.close()
                    # fig,ax = plt.subplots()
                    # ax.add_patch(ego_pg)
                    # ax.add_patch(other_pg)
                    # ax.set_xlim([1821,1835])
                    # ax.set_ylim([2529,2544])
                    #plt.show()

                    # print("COLLIDE! GT!!!!!!!! ", cur_iou)
                    
                    right_attacker_flag = 1
                    # Must collide with GT attacker
                    
                    real_yaw_distance = angle_vectors(other_vec, ego_vec) * 180 / np.pi
                    record_yaw_distance = (real_ego_angle - real_other_angle) * 180 / np.pi
                    plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='darkgreen', alpha=1)
                    #print(record_yaw_distance)
                else:
                    # print(x_1, y_1)
                    plt.fill([x_1, x_2, x_4, x_3], [y_1, y_2, y_4, y_3], '-',  color='green', alpha=1)
                    #plt.plot([x_1, x_2, x_4, x_3, x_1], [
                    #                 y_1, y_2, y_4, y_3, y_1], '-',  color='violet', markersize=3)

                
            if collision_flag:
                break
        if collision_flag:
            break
    plt.xlim(ego_list[0].loc[scenario_length - 1, 'X'] - 50,
                ego_list[0].loc[scenario_length - 1, 'X'] + 50)
    plt.ylim(ego_list[0].loc[scenario_length - 1, 'Y'] - 50,
                ego_list[0].loc[scenario_length - 1, 'Y'] + 50)
    # plt.show()
    
    return collision_flag, real_yaw_distance, right_attacker_flag, record_yaw_distance

def only_split():
    train_split = 0.7
    val_split = 0.1
    # data_path = './yaw_distribution/4.5_4.6_4.7_after_classify' 
    data_path = './yaw_distribution/4.5_4.6_4.7_after_collide_with_other_check'
    output_path = 'nuscenes_data/'
    ttc_all_distribution = []
    collide_with_other_num = 0
    collsion_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    now_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    ratio_cnt = {"JC": 0.6, "LTAP": 1.0, "LC": 1.0, "HO": 0.4, "RE": 0.7}
    for scenario in tqdm(sorted(os.listdir(data_path))):
        type_name = scenario.split('_')[5]
        # collsion_cnt[type_name] += 1
        # attacker_id = scenario.split('_')[-1].split('.')[0]
        # interpolation_for_collision_checking = True
        # if interpolation_for_collision_checking:
        #     ############## Interpolation for collision checking (exclude collide with other first) ##############
        #     frame_multiple = 5
        #     traj_df = pd.read_csv(data_path + '/' + scenario)
        #     frame_num = len(set(traj_df.TIMESTAMP.values))
        #     inter_df = pd.DataFrame(columns=['TRACK_ID', 'TIMESTAMP', 'V', 'X', 'Y', 'YAW'])
        #     for track_id, remain_df in traj_df.groupby("TRACK_ID"):
        #         # E.g. 20 frames * 5 times => (20-1)*5+1 = 96 frames
        #         # for origin_f_index in range(1, frame_num + 1):
        #         for origin_f_index in range(1, frame_num):
        #             # print(frame_num, origin_f_ttcindex-1, remain_df.iloc[origin_f_index-1, 3])
        #             # last frame in original df => further seems to be add last frame
        #             frame_multiple_for_loop = frame_multiple
        #             if origin_f_index == frame_num:
        #                 frame_multiple_for_loop = 1
        #                 dis_x, dis_y = 0, 0
        #             else:
        #                 dis_x = (remain_df.iloc[origin_f_index, 3] - remain_df.iloc[origin_f_index-1, 3]) / frame_multiple
        #                 dis_y = (remain_df.iloc[origin_f_index, 4] - remain_df.iloc[origin_f_index-1, 4]) / frame_multiple
        #             # for those padding zero vehicle
        #             if dis_x > 10 or dis_y > 10:
        #                 dis_x, dis_y = 0, 0 
        #             # if track_id == 'f5df5ef1e5624a029ce64dd462556de5':
        #                 # print(dis_x, dis_y)
                    
                    
        #             for fps_f_index in range(frame_multiple_for_loop):
        #                 # inter_df.insert(track_id, remain_df.iloc[origin_f_index-1, 1] + fps_f_index * 500000 / frame_multiple,
        #                 #                 remain_df.iloc[origin_f_index-1, 2], remain_df.iloc[origin_f_index-1, 3] + fps_f_index * dis_x,
        #                 #                 remain_df.iloc[origin_f_index-1, 4] + fps_f_index * dis_y, remain_df.iloc[origin_f_index-1, 5])
        #                 # t = {'TRACK_ID':[track_id], 'TIMESTAMP':[remain_df.iloc[origin_f_index-1, 1] + fps_f_index * 500000 / frame_multiple],
        #                 #       'V':[remain_df.iloc[origin_f_index-1, 2]], 'X':[remain_df.iloc[origin_f_index-1, 3] + fps_f_index * dis_x],
        #                 #       'Y':[remain_df.iloc[origin_f_index-1, 4]] + fps_f_index * dis_y, 'YAW':[remain_df.iloc[origin_f_index-1, 5]]}
                        
        #                 t = {'TRACK_ID':[track_id], 'TIMESTAMP':[remain_df.iloc[origin_f_index-1, 1] + fps_f_index * 500000 / frame_multiple],
        #                         'V':[remain_df.iloc[origin_f_index-1, 2]], 'X':[remain_df.iloc[origin_f_index-1, 3] + fps_f_index * dis_x],
        #                         'Y':[remain_df.iloc[origin_f_index-1, 4] + fps_f_index * dis_y], 'YAW':[remain_df.iloc[origin_f_index-1, 5]]}
        #                 df_insert = pd.DataFrame(t)
        #                 inter_df = pd.concat([inter_df, df_insert], ignore_index=True)
        #     ### Interpolation for moving foward with higher FPS###
        #     for track_id, remain_df in inter_df.groupby("TRACK_ID"):
        #         more_frames = 4 * frame_multiple
        #         # trajectory may be a curve, so it should rely on last frame and the previous frame
        #         dis_x = (remain_df.iloc[(frame_num-1)*frame_multiple-1, 3] - remain_df.iloc[(frame_num-1)*frame_multiple-2, 3])
        #         dis_y = (remain_df.iloc[(frame_num-1)*frame_multiple-1, 4] - remain_df.iloc[(frame_num-1)*frame_multiple-2, 4])
        #         all_x, all_y, all_v, all_yaw = [], [], [], []
        #         for f_index in range(more_frames):
        #             all_v.append(remain_df.iloc[(frame_num-1)*frame_multiple-1, 2])
        #             all_x.append(remain_df.iloc[(frame_num-1)*frame_multiple-1, 3] + dis_x * (f_index + 1))
        #             all_y.append(remain_df.iloc[(frame_num-1)*frame_multiple-1, 4] + dis_y * (f_index + 1))
        #             all_yaw.append(remain_df.iloc[(frame_num-1)*frame_multiple-1, 5])
        #         for further_t in range(more_frames):
        #             b = {'TIMESTAMP': [remain_df.TIMESTAMP.values[-1] + (further_t + 1) * 500000 / frame_multiple], 'TRACK_ID': [track_id],
        #                 # 'V': [all_v[further_t]ttc], 'X': [x[further_t].cpu().numpy()], 'Y': [y[further_t].cpu().numpy()],
        #                 'V': [all_v[further_t]], 'X': [all_x[further_t]], 'Y': [all_y[further_t]],
        #                 'YAW': [all_yaw[further_t]]}
        #             df_insert = pd.DataFrame(b)
        #             inter_df = pd.concat([inter_df, df_insert], ignore_index=True)
            
        #     collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = cal_cr_and_similarity(inter_df, attacker_id)
        #     if collision_flag:
        #         if not attacker_right_flag:
        #             collide_with_other_num += 1
        #             continue
        #     shutil.copyfile(data_path + '/' + scenario, './yaw_distribution/4.5_4.6_4.7_after_collide_with_other_check/' + scenario)
        collsion_cnt[type_name] += 1
    # print("collide_with_other_num:", collide_with_other_num)
    # print("collsion_cnt:", collsion_cnt)
    # exit()

    for scenario in tqdm(os.listdir(data_path)):
        scenario_type = scenario.split('_')[5]
        if now_cnt[scenario_type] > collsion_cnt[scenario_type] * ratio_cnt[scenario_type]:
            continue
        ttc = int(scenario.split('_')[7].split('-')[-1])
        ttc_all_distribution.append(ttc)
        now_cnt[scenario_type] += 1
        if now_cnt[scenario_type] < collsion_cnt[scenario_type] * ratio_cnt[scenario_type] * train_split:
            traj_path = output_path + 'trajectory/train/' + scenario
        elif collsion_cnt[scenario_type] * ratio_cnt[scenario_type] * train_split < now_cnt[scenario_type] < collsion_cnt[scenario_type] * ratio_cnt[scenario_type] * (train_split + val_split):
            traj_path = output_path + 'trajectory/val/' + scenario
        else:
            traj_path = output_path + 'trajectory/test/' + scenario
        # shutil.copyfile(data_path + '/' + scenario, traj_path)
    all_num = now_cnt['HO'] + now_cnt['JC'] + now_cnt['LC'] + now_cnt['LTAP'] + now_cnt['RE']
    
    bins = np.linspace(min(ttc_all_distribution), max(ttc_all_distribution), num=50)
    hist, bins = np.histogram(ttc_all_distribution, bins=bins, density=False)
    plt.bar(bins[:-1], hist, align='center', width=0.8)
    plt.xlabel('TTC (frame)')
    plt.ylabel('Freq')
    plt.title('TTC distribution')
    # plt.show()
    plt.savefig('./yaw_distribution/TTC distribution.png')
    plt.close()
    print("only split!! now_cnt:", now_cnt)
    print("All scenario:", all_num)

# 4.6 version / & 4.7
def data_proc_more_TTC():
    if_quintic_no_collision_filter = True
    train_split = 0.7
    val_split = 0.1
    # data_path = '../csvs_4.6_more_ttc'
    data_path = '../csvs_4.7_more_LTAP/'
    output_path = 'nuscenes_data/'
    filter_num = 0
    nearest_target_to_ego_collision_dist_too_far_dist = 2
    filter_num_nearest_target_to_ego_collision_dist_too_far = 0
    max_tar_candts_dist_too_far_dist = 150
    filter_num_max_tar_candts_dist_too_far_dist = 0
    f_8_vehicles = 0
    scenario_num = 0
    filter_quintic_no_collision = 0
    JC_yaw_distribution = []
    LTAP_yaw_distribution = []
    LC_yaw_distribution = []
    HO_yaw_distribution = []
    RE_yaw_distribution = []
    all_yaw_distribution = []

    JC_yaw_last_f_distribution = []
    LTAP_yaw_last_f_distribution = []
    LC_yaw_last_f_distribution = []
    HO_yaw_last_f_distribution = []
    RE_yaw_last_f_distribution = []
    all_yaw_last_f_distribution = []

    JC_ideal_yaw = 90
    LTAP_ideal_yaw = 90
    LC_ideal_yaw = 20
    HO_ideal_yaw = 180
    RE_ideal_yaw = 0

    yaw_threshold = 10

    new_LTAP_intersection_filter = 0
    new_JC_intersection_filter = 0
    intersection_filtered = 0
    stop_ego_from_origin = 0
    not_in_5 = 0

    collsion_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    now_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    # 4.5: 'JC': 10895, 'LTAP': 1610, 'LC': 2949, 'HO': 3788, 'RE': 4204
    # ratio_cnt = {"JC": 0.15, "LTAP": 1.0, "LC": 1.0, "HO": 0.5, "RE": 0.4}
    ratio_cnt = {"JC": 1.0, "LTAP": 1.0, "LC": 1.0, "HO": 1.0, "RE": 1.0}
    ttc_list = []
    for scenario in sorted(os.listdir(data_path)):
        scenario = scenario[10:]
        variant_id = scenario.split('_')[7]#scenario.split('.')[1].split('_')[5]
        ttc = int(variant_id.split('-')[-1]) - 8 + 1
        if ttc <= 0:
            continue
        type_name = scenario.split('_')[5]
        collsion_cnt[type_name] += 1
        ttc_list.append(ttc)
    # print(collsion_cnt)
    
    collsion_cnt_filtered = collsion_cnt 
    ttc_range_len = max(ttc_list) - min(ttc_list)
    print("4.5 min ttc:", min(ttc_list), " ttc_range_length:", ttc_range_len)
    for scenario in tqdm(os.listdir(data_path)):
        origin_file_name = scenario
        scenario = scenario[10:]
        type_name = scenario.split('_')[5]        
        scenario_type = scenario.split('_')[5]
        variant_id = scenario.split('.')[1].split('_')[5]
        now_ttc = int(variant_id.split('-')[-1]) - 8 + 1
        if now_ttc <= 0:
            continue
        normalize_ttc = (now_ttc - min(ttc_list)) / ttc_range_len

        # if type_name != 'LC':
        #     continue
        # if scenario != 'data_hcis_v4.5_trainval_boston-seaport_HO_scene-0262_5-0-15_9cc46b3cbc3c4efdb2a46aa4d7b6833b.csv':
        #     continue
        # print(scenario, scenario_num)

        
        

        if now_cnt[scenario_type] > collsion_cnt_filtered[scenario_type] * ratio_cnt[scenario_type]:
            continue

        df = pd.read_csv(data_path + origin_file_name)

        ######################
        # obs_horizon = 8
        # for track_id, remain_df in df.groupby('TRACK_ID'):
        #     if track_id == 'ego':
        #         ego_traj = np.concatenate((
        #             remain_df.X.to_numpy().reshape(-1, 1),
        #             remain_df.Y.to_numpy().reshape(-1, 1)), 1)
        #         orig = ego_traj[obs_horizon - 1]
        # pre = (orig - ego_traj[obs_horizon-4]) / 2.0
        # theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
        # rot = np.asarray([
        #     [np.cos(theta), -np.sin(theta)],
        #     [np.sin(theta), np.cos(theta)]], np.float32)
        # topology_id = scenario.split('_')[3] + '_' + scenario.split('_')[4] + '_' + scenario.split('_')[6]
        # topology = np.load("./nuscenes_data/initial_topology/" + topology_id + ".npy", allow_pickle=True)
        # ctr_line_candts_list = []
        # if len(topology) == 0:
        #     continue
        # t_sum = 0
        # for i, _ in enumerate(topology):
        #     t_sum += topology[:, 2][i][:, :2].shape[0]
        #     t = np.matmul(rot, (topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T).T            
        #     if len(t) <= 1:
        #         tmp = (np.matmul(rot, (topology[:, 2][i][:, 3:5] - orig.reshape(-1, 2)).T).T)[0]
        #         t = np.vstack((t[0], tmp))
        #     ctr_line_candts_list.append(t)
        # ctr_line_candts = np.array(ctr_line_candts_list, dtype=object)
        
        # candidates = []
        # for line_id, line in enumerate(ctr_line_candts):
        #     for i in range(len(line) - 1):
        #         # print(i, "line:", line[i])
        #         # if np.any(np.isnan(line[i])) or np.any(np.isnan(line[i+1])):
        #         #     continue
        #         [x_diff, y_diff] = line[i+1] - line[i]
        #         if x_diff == 0.0 and y_diff == 0.0:
        #             continue
        #         candidates.append(line[i])
        
        # if len(candidates) == 0:
        #     continue
        
        # agt_traj_fut = ego_traj[obs_horizon:].copy().astype(np.float32)
        # agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T

        # candidates = np.unique(np.asarray(candidates), axis=0)
        # max_tar_candts_dist = np.max(candidates)
        # if max_tar_candts_dist > max_tar_candts_dist_too_far_dist:
        #     filter_num_max_tar_candts_dist_too_far_dist += 1
        #     collsion_cnt_filtered[type_name] -= 1
        #     continue

        # displacement = candidates - agt_traj_fut[-1]
        # dist = np.sqrt(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))
        # nearest_target_to_ego_collision_dist = min(dist)
        # if nearest_target_to_ego_collision_dist > nearest_target_to_ego_collision_dist_too_far_dist:
        #     filter_num_nearest_target_to_ego_collision_dist_too_far += 1
        #     collsion_cnt_filtered[type_name] -= 1
        #     continue

        
        #print(scene_name)
        # if scene_name != 'scene-0163':
        #     continue
        # if variant_name != '5-0-25':
        #     continue
        with open('/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/filter_agent.json') as f:
            data = json.load(f)
        attacker_id = scenario.split('_')[-1].split('.')[0]
        scene_id = scenario.split('_')[6]
        variant_id = scenario.split('_')[7]
        sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
        #df = pd.read_csv(data_path + 'all_traj/' + scenario, index_col=0)
        
        # for track_id, remain_df in df.groupby('TRACK_ID'):
        #     if track_id == attacker_id:
        #         t = np.array((remain_df.TIMESTAMP.values, remain_df.X.values)).T
        #         print(t)
        #print(df)
        df.iloc[:,5] *= 180
        df.iloc[:,5] /= np.pi
        df = (df.pivot_table(columns='TIMESTAMP', index=['TRACK_ID'], fill_value=0)
                .stack('TIMESTAMP')
                .sort_index(level=['TRACK_ID','TIMESTAMP'])
                .reset_index())
        if sce_temp in data:
            #print("filter:", list(data[sce_temp]), df)
            for idx in range(len(list(data[sce_temp]))):
                #print(list(data[sce_temp])[idx])
                parked_idx = df[df["TRACK_ID"] == list(data[sce_temp])[idx]].index
                df = df.drop(parked_idx).reset_index(drop=True)
        objs = df.groupby(['TRACK_ID']).groups
        keys = list(objs.keys())
        if attacker_id not in keys:
            filter_num += 1
            collsion_cnt_filtered[type_name] -= 1
            # print("filter")
            continue
        vehicle_list = []
        vehicle_pd = pd.DataFrame()
        for track_id, remain_df in df.groupby('TRACK_ID'):
            x_8 = remain_df.X.values[8]
            y_8 = remain_df.Y.values[8]
            if x_8 == 0 and y_8 == 0:
                f_8_vehicles += 1
                #print("frame 8 filter")
            else:
                vehicle_pd = pd.concat([vehicle_pd, remain_df], axis=0)
            #vehicle_list.append(remain_df)
            
        
        # trajs = np.concatenate((
        #     df.X.to_numpy().reshape(-1, 1),
        #     df.Y.to_numpy().reshape(-1, 1)), 1)
        # obs_horizon = 8
        # filter_flag = 0
        # for key in keys:
        #     idcs = objs[key]
        #     if trajs[idcs][obs_horizon - 1][0] == 0 and trajs[idcs][obs_horizon - 1][1] == 0:

        columns_to_update = ['X', 'Y', 'V', 'YAW']
        vehicle_pd[columns_to_update] = vehicle_pd[columns_to_update].applymap(replace_less_than_001)
        if if_quintic_no_collision_filter:
            save_folder = './yaw_distribution/'
            stop_ego_flag = 0
            frame_num = len(set(vehicle_pd.TIMESTAMP.values))
            frame_multiple = 5
            further_df = vehicle_pd
            for track_id, remain_df in vehicle_pd.groupby("TRACK_ID"):
                ##### only for analyze #####
                if track_id == 'ego':
                    ego_last_f_yaw = remain_df.iloc[frame_num-1, 5] + 360.0 if remain_df.iloc[frame_num-1, 5] < 0 else remain_df.iloc[frame_num-1, 5]
                elif track_id == attacker_id:
                    attacker_last_f_yaw = remain_df.iloc[frame_num-1, 5] + 360.0 if remain_df.iloc[frame_num-1, 5] < 0 else remain_df.iloc[frame_num-1, 5]
                ##### only for analyze #####
                #print(track_id, tp_index[s].cpu().numpy()[0], remain_df)
                more_frames = 4
                # trajectory may be a curve, so it should rely on last frame and the previous frame
                dis_x = (remain_df.iloc[frame_num-1, 3] - remain_df.iloc[frame_num-2, 3]) / frame_multiple
                dis_y = (remain_df.iloc[frame_num-1, 4] - remain_df.iloc[frame_num-2, 4]) / frame_multiple
                # print("x:", remain_df.iloc[frame_num-1, 3], remain_df.iloc[frame_num-2, 3])
                for frame_index in range(frame_num):
                    # if track_id == 'ego':
                    #     print("v:", remain_df.iloc[frame_index-1, 2])
                        # print("dis:", dis_x, dis_y)
                    if track_id == 'ego' and remain_df.iloc[frame_index-1, 2] < 2: #dis_x < 0.001 and dis_y < 0.001:
                        stop_ego_flag = 1
                # for padding vehicle
                if dis_x > 10 or dis_y > 10:
                    dis_x, dis_y = 0, 0 
                all_x, all_y, all_v, all_yaw = [], [], [], []
                for f_index in range(more_frames * frame_multiple):
                    all_v.append(remain_df.iloc[frame_num-1, 2])
                    all_x.append(remain_df.iloc[frame_num-1, 3] + dis_x * (f_index + 1))
                    all_y.append(remain_df.iloc[frame_num-1, 4] + dis_y * (f_index + 1))
                    all_yaw.append(remain_df.iloc[frame_num-1, 5])
                for further_t in range(more_frames * frame_multiple):
                    b = {'TIMESTAMP': [remain_df.TIMESTAMP.values[-1] + (further_t + 1) * 500000 / frame_multiple], 'TRACK_ID': [track_id],
                        # 'V': [all_v[further_t]], 'X': [x[further_t].cpu().numpy()], 'Y': [y[further_t].cpu().numpy()],
                        'V': [all_v[further_t]], 'X': [all_x[further_t]], 'Y': [all_y[further_t]],
                        'YAW': [all_yaw[further_t]]}
                    df_insert = pd.DataFrame(b)
                    further_df = pd.concat([further_df, df_insert], ignore_index=True)
                # if track_id == attacker_id:
                #     print(vehicle_pd, further_df)
                #     exit()
            if stop_ego_flag:
                stop_ego_from_origin += 1
                continue
            plt.close()
            collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = cal_cr_and_similarity(further_df, attacker_id)
            
            

            last_f_yaw_distance = ego_last_f_yaw - attacker_last_f_yaw
            while last_f_yaw_distance < 0:
                last_f_yaw_distance = (last_f_yaw_distance + 360.0)
            last_f_yaw_distance = abs(last_f_yaw_distance - 360.0) if last_f_yaw_distance > 180 else last_f_yaw_distance
            
            # print("Type:", type_name, "last_f_yaw_distance:", last_f_yaw_distance)
            if type_name == 'JC':
                JC_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 90))
            elif type_name == 'LTAP':
                LTAP_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 90))
            elif type_name == 'LC':
                LC_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 15))
            elif type_name == 'HO':
                HO_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 180))
            elif type_name == 'RE':
                RE_yaw_last_f_distribution.append(last_f_yaw_distance)
                all_yaw_last_f_distribution.append(abs(last_f_yaw_distance - 0))
            # print("Type:", type_name, "last_f_yaw_distance:", last_f_yaw_distance)
            if collision_flag:
                while record_yaw_distance < 0:
                    record_yaw_distance = (record_yaw_distance + 360.0)
                record_yaw_distance = abs(record_yaw_distance - 360.0) if record_yaw_distance > 180 else record_yaw_distance
                
                if attacker_right_flag:
                    new_type_name = None
                    # if type_name == 'JC':
                    #     JC_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = JC_ideal_yaw
                    # elif type_name == 'LTAP':
                    #     LTAP_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = LTAP_ideal_yaw
                    # elif type_name == 'LC':
                    #     LC_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = LC_ideal_yaw
                    # elif type_name == 'HO':
                    #     HO_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = HO_ideal_yaw
                    # elif type_name == 'RE':
                    #     RE_yaw_distribution.append(record_yaw_distance)
                    #     ideal_offset = RE_ideal_yaw
                    # # yaw_distance = abs(ideal_yaw_offset - record_yaw_distance)
                    # to_ideal_yaw_dist = abs(record_yaw_distance - ideal_offset)
                    # all_yaw_distribution.append(to_ideal_yaw_dist)
                    # if to_ideal_yaw_dist > 10:
                    if True:
                        if record_yaw_distance < 10:
                            new_type_name = 'RE'
                        elif 10 < record_yaw_distance and record_yaw_distance < 30:
                            new_type_name = 'LC'
                        elif 170 < record_yaw_distance:
                            new_type_name = 'HO'
                        elif 80 < record_yaw_distance and record_yaw_distance < 100:
                            frame_num = len(set(vehicle_pd.TIMESTAMP.values))

                            intersection_threshold = 20
                            ego_list = []
                            for track_id, remain_df in vehicle_pd.groupby("TRACK_ID"):
                                if str(track_id) == 'ego':
                                    ego_list.append(remain_df.reset_index())
                            for track_id, remain_df in vehicle_pd.groupby("TRACK_ID"):
                                if track_id == attacker_id:
                                    for i in range(frame_num):
                                        
                                        x = remain_df.iloc[i, 3]
                                        y = remain_df.iloc[i, 4]
                                        yaw = remain_df.iloc[i, 5]
                                        # print(abs(yaw - ego_list[0].loc[i, 'YAW']), is_intersection(x, y, town_name=scenario.split('_')[4]))
                                        if is_intersection(x, y, town_name=scenario.split('_')[4]):
                                            continue
                                        diff_yaw = abs(yaw - ego_list[0].loc[i, 'YAW'])
                                        
                                        if abs(diff_yaw - 180) < intersection_threshold:
                                            new_LTAP_intersection_filter += 1
                                            new_type_name = 'LTAP'
                                        elif abs(diff_yaw - 90) < intersection_threshold:
                                            new_JC_intersection_filter += 1
                                            new_type_name = 'JC'
                                        else:
                                            intersection_filtered += 1
                                            new_type_name = 'filter'
                        else:
                            # print("Not in the 5 categories")
                            not_in_5 += 1
                            new_type_name = 'filter'
                            collsion_cnt_filtered[type_name] -= 1
                            continue

                    # print(scenario, new_type_name)
                    if new_type_name == None or new_type_name == 'filter':
                        # print("B", new_type_name, record_yaw_distance)
                        collsion_cnt_filtered[type_name] -= 1
                        continue
                    if new_type_name != None:
                        scenario = scenario.replace(scenario_type, new_type_name)
                    # print(new_type_name)
                    # exit()


                        
                else:
                    filter_quintic_no_collision += 1
                    collsion_cnt_filtered[type_name] -= 1
                    continue
            else:
                filter_quintic_no_collision += 1
                collsion_cnt_filtered[type_name] -= 1
                continue
            
            
            # print("real yaw:", record_yaw_distance, " ideal yaw:", last_f_yaw_distance, type_name)

            title = str(attacker_right_flag) + " real yaw:" + str(round(record_yaw_distance, 1)) + " ideal yaw:" + str(round(last_f_yaw_distance, 1))
            plt.title(title)
            plt.savefig('./yaw_distribution/figures/' + scenario + '.png')
            plt.close()

            

        # print("Now filter_quintic_no_collision:", filter_quintic_no_collision)
        scenario_num += 1
        scenario_type = new_type_name
        now_cnt[scenario_type] += 1
        source_path = data_path + 'all_traj/' + scenario
        scenario = scenario.replace("-" + str(variant_id.split('-')[-1]) + "_", "-" + str(normalize_ttc) + "-" + str(variant_id.split('-')[-1]) + "_")
        
        scenario = scenario.replace("_" + attacker_id, "_" + str(round(record_yaw_distance, 2)) + "_" + attacker_id)
        # print("filter_quintic_no_collision:", filter_quintic_no_collision)
        # print("new_LTAP_intersection_filter:", new_LTAP_intersection_filter)
        # print("new_JC_intersection_filter:", new_JC_intersection_filter)
        # print("intersection_filtered:", intersection_filtered)
        # print("stop_ego_from_origin:", stops_ego_from_origin)
        traj_path = './yaw_distribution/4.7_after_classify/' + scenario
        vehicle_pd.to_csv(traj_path, index=False)
        
        # if type_name == 'LC':
        #     if now_cnt[scenario_type] < collsion_cnt_filtered[scenario_type] * ratio_cnt[scenario_type] * train_split:
        #         traj_path = output_path + 'trajectory/train/' + scenario
        #         vehicle_pd.to_csv(traj_path, index=False)
        #     elif collsion_cnt_filtered[scenario_type] * ratio_cnt[scenario_type] * train_split < now_cnt[scenario_type] < collsion_cnt_filtered[scenario_type] * ratio_cnt[scenario_type] * (train_split + val_split):
        #         traj_path = output_path + 'trajectory/val/' + scenario
        #         vehicle_pd.to_csv(traj_path, index=False)
        #     else:
        #         traj_path = output_path + 'trajectory/test/' + scenario
        #         vehicle_pd.to_csv(traj_path, index=False)
        # shutil.copyfile(source_path, traj_path)
    print("filter:", filter_num)
    print("f_8_vehicles_average:", f_8_vehicles / scenario_num)
    print("nearest_target_to_ego_collision_dist:", filter_num_nearest_target_to_ego_collision_dist_too_far)
    print("filter_num_max_tar_candts_dist_too_far_dist:", filter_num_max_tar_candts_dist_too_far_dist)
    print("filter_quintic_no_collision:", filter_quintic_no_collision)
    print("new_LTAP_intersection_filter:", new_LTAP_intersection_filter)
    print("new_JC_intersection_filter:", new_JC_intersection_filter)
    print("intersection_filtered:", intersection_filtered)
    print("stop_ego_from_origin:", stop_ego_from_origin)
    print("not_in_5:", not_in_5)
    
    print("Origin:", collsion_cnt)
    print("After filter:", collsion_cnt_filtered)
    print("final:", now_cnt)

    # print("all_yaw_last_f_distribution:", all_yaw_last_f_distribution)
    
    plt.close()
    bins = np.linspace(-25, 25, num=50)
    # bins = np.linspace(min(all_yaw_distribution), max(all_yaw_distribution), num=50)
    hist, bins = np.histogram(all_yaw_distribution, bins=bins, density=False)
    plt.bar(bins[:-1], hist, align='center', width=0.02)
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('All Yaw Distribution')
    plt.savefig(save_folder + 'all_yaw_distribution.png')
    plt.close()

    bins = np.linspace(-1, 1, num=50)
    hist, bins = np.histogram(all_yaw_last_f_distribution, bins=bins, density=False)
    plt.bar(bins[:-1], hist, align='center', width=0.02)
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('All Ideal Yaw Distribution')
    plt.savefig(save_folder + 'all_yaw_last_f_distribution.png')
    plt.close()

    bins = np.linspace(-25, 25, num=50)
    plt.hist(all_yaw_distribution, bins=bins, alpha=0.7, label='Real')
    plt.hist(all_yaw_last_f_distribution, bins=bins, alpha=0.7, label='Ideal')
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('Yaw Distribution Comparison')
    plt.legend()
    # plt.show()
    plt.savefig(save_folder + '/Yaw Distribution Comparison.png')
    plt.close()


    bins = np.linspace(min(LC_yaw_distribution), max(LC_yaw_distribution), num=100)
    # bins = np.linspace(-25, 25, num=50)
    hist, bins = np.histogram(LC_yaw_distribution, bins=bins, density=False)
    plt.close()
    plt.bar(bins[:-1], hist, align='center', width=0.1)
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('Lane Change Yaw Distribution')
    plt.savefig(save_folder + 'LC_yaw_distribution.png')
    plt.close()

    bins = np.linspace(min(LC_yaw_last_f_distribution), max(LC_yaw_last_f_distribution), num=100)
    # bins = np.linspace(-10, 10, num=10)
    hist, bins = np.histogram(LC_yaw_last_f_distribution, bins=bins, density=False)
    plt.close()
    plt.bar(bins[:-1], hist, align='center', width=0.1)
    plt.xlabel('yaw_distance')
    plt.ylabel('Freq')
    plt.title('Lane Change Ideal Yaw Distribution')
    plt.savefig(save_folder + 'LC_yaw_last_f_distribution.png')
    plt.close()
    
def data_proc_miniset():
    data_path = './nuscenes_data/'
    output_path = 'nuscenes_data/'
    train_split = 0.8
    val_split = 0.1
    average_vehicle_num = 0
    average_filtered_out_vehicle_num = 0
    average_same_start_frame_vehicle = 0
    average_ideal_vehicle = 0
    scenario_num = 0
    collsion_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    now_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    with open('/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/filter_agent_miniset.json') as f:
        data = json.load(f)
    mini_scene = []
    for i in range(len(data.keys())):
        #print(list(data.keys()))
        mini_scene.append(list(data.keys())[i].split('_')[0])
    print(list(set(mini_scene)))
    exit()
    for scenario in sorted(os.listdir(data_path + 'all_traj/')):
        type_name = scenario.split('_')[5]
        collsion_cnt[type_name] += 1
    print(collsion_cnt)
    for scenario in tqdm(sorted(os.listdir(data_path + 'all_traj/'))):
        #print(scenario.split('.')[1])
        scenario_num += 1
        scenario_type = scenario.split('_')[5]
        now_cnt[scenario_type] += 1
        source_path = data_path + 'all_traj/' + scenario
        if now_cnt[scenario_type] < collsion_cnt[scenario_type] * train_split:
            traj_path = output_path + 'trajectory/train/' + scenario
        elif collsion_cnt[scenario_type] * train_split < now_cnt[scenario_type] < collsion_cnt[scenario_type] * (train_split + val_split):
            traj_path = output_path + 'trajectory/val/' + scenario
        else:
            traj_path = output_path + 'trajectory/test_obs/' + scenario
        shutil.copyfile(source_path, traj_path)

def data_proc_initial_padding_zero():
    data_path = '../init-trainval/'
    data_name_path = 'nuscenes_data/trajectory/test_obs/'
    output_path = 'nuscenes_data/trajectory/test/'
    train_split = 0.8
    val_split = 0.1
    filter_num = 0
    f_8_vehicles = 0
    scenario_num = 0
    collsion_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    now_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    for scenario in tqdm(sorted(os.listdir(data_name_path))):
        
        scene_name = scenario.split('_')[6]
        #if scene_name != 'scene-0366':
        #    continue
        df = pd.read_csv(data_path + scene_name)
        with open('/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/filter_agent.json') as f:
            data = json.load(f)
        attacker_id = scenario.split('_')[-1].split('.')[0]
        scene_id = scenario.split('_')[6]
        variant_id = scenario.split('_')[7]
        sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
        #print(df)
        df.iloc[:,5] *= 180
        df.iloc[:,5] /= np.pi
        # padding 0
        df = (df.pivot_table(columns='TIMESTAMP', index=['TRACK_ID'], fill_value=0)
                .stack('TIMESTAMP')
                .sort_index(level=['TRACK_ID','TIMESTAMP'])
                .reset_index())
        # df.to_csv('temp.csv',index=False)
        # filter parking
        if sce_temp in data:
            #print("filter:", list(data[sce_temp]), df)
            for idx in range(len(list(data[sce_temp]))):
                #print(list(data[sce_temp])[idx])
                parked_idx = df[df["TRACK_ID"] == list(data[sce_temp])[idx]].index
                df = df.drop(parked_idx).reset_index(drop=True)
        # if GT attacker not in the rest df, filter
        # objs = df.groupby(['TRACK_ID']).groups
        # keys = list(objs.keys())
        # scenario_num += 1
        # if attacker_id not in keys:
        #     filter_num += 1
        #     continue
        vehicle_list = []
        vehicle_pd = pd.DataFrame()
        for track_id, remain_df in df.groupby('TRACK_ID'):
            x_8 = remain_df.X.values[8]
            y_8 = remain_df.Y.values[8]
            #print(track_id, x_8, y_8, remain_df)
            if x_8 == 0 and y_8 == 0:
                f_8_vehicles += 1
                #print("frame 8 filter")
            else:
                #print(remain_df)
                vehicle_pd = pd.concat([vehicle_pd, remain_df], axis=0)
        objs = vehicle_pd.groupby(['TRACK_ID']).groups
        keys = list(objs.keys())
        scenario_num += 1
        if attacker_id not in keys:
            filter_num += 1
            continue

        vehicle_pd.to_csv(output_path + scenario, index=False)
    print("filter:", filter_num)
    print("f_8_vehicles_average:", f_8_vehicles / scenario_num)

def data_proc_initial_early_attack():
    data_path = '../init-trainval/'
    data_name_path = 'nuscenes_data/trajectory/test_obs/'
    output_path = 'nuscenes_data/trajectory/test/'
    train_split = 0.8
    val_split = 0.1
    filter_num = 0
    f_8_vehicles = 0
    scenario_num = 0
    collsion_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    now_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    for scenario in tqdm(sorted(os.listdir(data_name_path))):
        
        scene_name = scenario.split('_')[6]
        #if scene_name != 'scene-0794':
        #    continue
        df = pd.read_csv(data_path + scene_name)
        with open('/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/filter_agent.json') as f:
            data = json.load(f)
        attacker_id = scenario.split('_')[-1].split('.')[0]
        scene_id = scenario.split('_')[6]
        variant_id = scenario.split('_')[7]
        sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
        for track_id, remain_df in df.groupby("TRACK_ID"):
            if track_id == attacker_id:
                attacker_time_sequence = remain_df.TIMESTAMP.values
            elif track_id == 'ego':
                all_time_sequence = remain_df.TIMESTAMP.values
        attacker_length = len(attacker_time_sequence)
        forward_frames = all_time_sequence.tolist().index(attacker_time_sequence[0])
        df.iloc[:,5] *= 180
        df.iloc[:,5] /= np.pi
        # padding 0
        df = (df.pivot_table(columns='TIMESTAMP', index=['TRACK_ID'], fill_value=0)
                .stack('TIMESTAMP')
                .sort_index(level=['TRACK_ID','TIMESTAMP'])
                .reset_index())
        vehicle_list = []
        for track_id, remain_df in df.groupby("TRACK_ID"):
            if track_id == attacker_id:
                v_list = remain_df.V.values
                remain_df.iloc[:attacker_length, 2] = v_list[forward_frames:forward_frames+attacker_length]
                
                x_list = remain_df.X.values
                remain_df.iloc[:attacker_length, 3] = x_list[forward_frames:forward_frames+attacker_length]
                
                y_list = remain_df.Y.values
                remain_df.iloc[:attacker_length, 4] = y_list[forward_frames:forward_frames+attacker_length]
                yaw_list = remain_df.YAW.values
                remain_df.iloc[:attacker_length, 5] = yaw_list[forward_frames:forward_frames+attacker_length]
                remain_df.iloc[attacker_length:, 2:] = 0
                #remain_df.iloc[:attacker_length, 3:] = remain_df[forward_frames:forward_frames+attacker_length, 3:]
            vehicle_list.append(remain_df)
        df = pd.concat(vehicle_list)
        #traj_df.to_csv(scenario + '_temp.csv',index=False)
        # df.to_csv('temp.csv',index=False)
        # filter parking
        if sce_temp in data:
            #print("filter:", list(data[sce_temp]), df)
            for idx in range(len(list(data[sce_temp]))):
                #print(list(data[sce_temp])[idx])
                parked_idx = df[df["TRACK_ID"] == list(data[sce_temp])[idx]].index
                df = df.drop(parked_idx).reset_index(drop=True)
        # if GT attacker not in the rest df, filter
        # objs = df.groupby(['TRACK_ID']).groups
        # keys = list(objs.keys())
        # scenario_num += 1
        # if attacker_id not in keys:
        #     filter_num += 1
        #     continue
        vehicle_list = []
        vehicle_pd = pd.DataFrame()
        for track_id, remain_df in df.groupby('TRACK_ID'):
            x_8 = remain_df.X.values[8]
            y_8 = remain_df.Y.values[8]
            #print(track_id, x_8, y_8, remain_df)
            if x_8 == 0 and y_8 == 0:
                f_8_vehicles += 1
                #print("frame 8 filter")
            else:
                #print(remain_df)
                vehicle_pd = pd.concat([vehicle_pd, remain_df], axis=0)
        objs = vehicle_pd.groupby(['TRACK_ID']).groups
        keys = list(objs.keys())
        scenario_num += 1
        if attacker_id not in keys:
            filter_num += 1
            continue

        vehicle_pd.to_csv(output_path + scenario, index=False)
    print("filter:", filter_num)
    print("f_8_vehicles_average:", f_8_vehicles / scenario_num)

def csv_shorter_ttc_angle():
    exit()
    data_path = '../csvs/'
    output_path = '../csvs/'
    train_split = 0.7
    val_split = 0.1
    filter_num = 0
    f_8_vehicles = 0
    scenario_num = 0
    collsion_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    now_cnt = {"JC": 0, "LTAP": 0, "LC": 0, "HO": 0, "RE": 0}
    for scenario in sorted(os.listdir(data_path)):
        type_name = scenario.split('_')[5]
        collsion_cnt[type_name] += 1
    print(collsion_cnt)
    for scenario in tqdm(sorted(os.listdir(data_path))):
        #print(scenario.split('.')[1])
        #print(scenario)
        scene_name = scenario.split('.')[1].split('_')[4]
        variant_name = scenario.split('.')[1].split('_')[5]
        
        #print(scene_name)
        # if scene_name != 'scene-0163':
        #     continue
        # if variant_name != '5-0-25':
        #     continue
        with open('/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/filter_agent.json') as f:
            data = json.load(f)
        attacker_id = scenario.split('_')[-1].split('.')[0]
        scene_id = scenario.split('_')[6]
        variant_id = scenario.split('_')[7]
        sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
        #df = pd.read_csv(data_path + 'all_traj/' + scenario, index_col=0)
        df = pd.read_csv(data_path + scenario)
        # for track_id, remain_df in df.groupby('TRACK_ID'):
        #     if track_id == attacker_id:
        #         t = np.array((remain_df.TIMESTAMP.values, remain_df.X.values)).T
        #         print(t)
        #print(df)
        df.iloc[:,5] *= 180
        df.iloc[:,5] /= np.pi
        df = (df.pivot_table(columns='TIMESTAMP', index=['TRACK_ID'], fill_value=0)
                .stack('TIMESTAMP')
                .sort_index(level=['TRACK_ID','TIMESTAMP'])
                .reset_index())
        if sce_temp in data:
            #print("filter:", list(data[sce_temp]), df)
            for idx in range(len(list(data[sce_temp]))):
                #print(list(data[sce_temp])[idx])
                parked_idx = df[df["TRACK_ID"] == list(data[sce_temp])[idx]].index
                df = df.drop(parked_idx).reset_index(drop=True)
        objs = df.groupby(['TRACK_ID']).groups
        keys = list(objs.keys())
        if attacker_id not in keys:
            filter_num += 1
            continue
        vehicle_list = []
        vehicle_pd = pd.DataFrame()
        for track_id, remain_df in df.groupby('TRACK_ID'):
            x_8 = remain_df.X.values[8]
            y_8 = remain_df.Y.values[8]
            if x_8 == 0 and y_8 == 0:
                f_8_vehicles += 1
                #print("frame 8 filter")
            else:
                vehicle_pd = pd.concat([vehicle_pd, remain_df], axis=0)
            #vehicle_list.append(remain_df)
            
        
        # trajs = np.concatenate((
        #     df.X.to_numpy().reshape(-1, 1),
        #     df.Y.to_numpy().reshape(-1, 1)), 1)
        # obs_horizon = 8
        # filter_flag = 0
        # for key in keys:
        #     idcs = objs[key]
        #     if trajs[idcs][obs_horizon - 1][0] == 0 and trajs[idcs][obs_horizon - 1][1] == 0:

        columns_to_update = ['X', 'Y', 'V', 'YAW']
        vehicle_pd[columns_to_update] = vehicle_pd[columns_to_update].applymap(replace_less_than_001)
        
        scenario_num += 1
        scenario_type = scenario.split('_')[5]
        now_cnt[scenario_type] += 1
        source_path = data_path + 'all_traj/' + scenario
        if now_cnt[scenario_type] < collsion_cnt[scenario_type] * train_split:
            traj_path = output_path + 'trajectory/train/' + scenario
            vehicle_pd.to_csv(traj_path, index=False)
        elif collsion_cnt[scenario_type] * train_split < now_cnt[scenario_type] < collsion_cnt[scenario_type] * (train_split + val_split):
            traj_path = output_path + 'trajectory/val/' + scenario
            vehicle_pd.to_csv(traj_path, index=False)
        else:
            traj_path = output_path + 'trajectory/test/' + scenario
            vehicle_pd.to_csv(traj_path, index=False)
        #shutil.copyfile(source_path, traj_path)
    print("filter:", filter_num)
    print("f_8_vehicles_average:", f_8_vehicles / scenario_num)

if __name__ == "__main__":
    #data_proc_initial_padding_zero()
    #data_proc_initial_early_attack()
    
    # first data_proc, then only_split
    # data_proc()
    # data_proc_more_TTC()
    only_split()
    