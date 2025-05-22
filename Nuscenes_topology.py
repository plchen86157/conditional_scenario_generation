from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import argparse
import csv
import pandas as pd
import numpy as np
import math
import shutil
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.base import CAP_STYLE

def vectorize_lane(nusc_map, lane, lane_id, ego_x, ego_y, resolution_meters=2):
    arclines = nusc_map.get_arcline_path(lane)
    ret = []
    #print("arclines:", arclines, " ego:", ego_x, ego_y)
    for arc in arclines:
        dist_start_x = abs(arc['start_pose'][0] - ego_x)
        dist_start_y = abs(arc['start_pose'][1] - ego_y)
        dist_end_x = abs(arc['end_pose'][0] - ego_x)
        dist_end_y = abs(arc['end_pose'][1] - ego_y)
        dist_mid_x = abs((arc['start_pose'][0] + arc['end_pose'][0]) / 2 - ego_x)
        dist_mid_y = abs((arc['start_pose'][1] + arc['end_pose'][1]) / 2 - ego_y)
        # if (dist_start_x < 75 and dist_start_y < 75) or (dist_end_x < 75 and dist_end_y < 75):
        if (dist_start_x < 37.5 and dist_start_y < 37.5) or (dist_end_x < 37.5 and dist_end_y < 37.5):
            #print(dist_start_x,dist_start_y,dist_mid_x,dist_mid_y,dist_end_x,dist_end_y)
            #if (dist_start_x < 37.5 and dist_start_y < 37.5) or (dist_end_x < 37.5 and dist_end_y < 37.5) or (dist_mid_x < 37.5 and dist_mid_y < 37.5):
            poses = arcline_path_utils.discretize(arc, resolution_meters)
            #poses.append(lane_id)
            #sys.exit()
            ret.extend(poses)
    return ret

def is_intersection(nusc_map, x, y):
    rstk = nusc_map.record_on_point(x, y, "road_segment")
    if rstk == "":
        return True
    rs = nusc_map.get("road_segment", rstk)
    return rs["is_intersection"]

def has_traffic_light(nusc_map, x, y, rb_with_tls):
    rbtk = nusc_map.record_on_point(x, y, "road_block")
    if rbtk == "":
        return False
    return rbtk in rb_with_tls

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

def main(nusc_map):
    pts = []
    lanes = nusc_map.arcline_path_3
    print(len(lanes))
    for lane in lanes:
        pts.extend(vectorize_lane(nusc_map, lane, resolution_meters=10))

    rb_with_tls = set()
    for tl in nusc_map.traffic_light:
        rb_with_tls.add(tl["from_road_block_token"])

    xlim = [300, 500]
    ylim = [1000, 1200]
    # xlim = [0, 10000]
    # ylim = [0, 10000]
    crop = Polygon(
        [[xlim[0], ylim[0]], [xlim[0], ylim[1]], [xlim[1], ylim[1]], [xlim[1], ylim[0]]]
    )
    pts = [pt for pt in pts if crop.contains(Point(pt[0], pt[1]))]

    for pt in tqdm(pts):
        intsec = is_intersection(nusc_map, pt[0], pt[1])
        traf = has_traffic_light(nusc_map, pt[0], pt[1], rb_with_tls)
        color = None
        if intsec and traf:
            color = "purple"
        elif intsec and not traf:
            color = "red"
        elif not intsec and traf:
            color = "orange"
        else:
            color = "green"
        plt.scatter(pt[0], pt[1], s=0.1, c=color)

    #plt.savefig("tmp.png")
    plt.show()

def save_topology(args):
    for now_data in tqdm(sorted(os.listdir(args.input_path))):
        town_name = now_data.split('_')[4]
        scene_name = now_data.split('.')[0] + '.' + now_data.split('.')[1]
        print(now_data.split('.')[1])
        sce_df = pd.read_csv(args.input_path + now_data)
        ego_pos_x = 0
        ego_pos_y = 0
        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
            if track_id == 'ego':
                ego_pos_x = remain_df["X"].values[7]
                ego_pos_y = remain_df["Y"].values[7]
        nusc_map = NuScenesMap(dataroot=args.nuscenes_map_path, map_name=town_name)
        pts = []
        lanes = nusc_map.arcline_path_3
        rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
        lane_width = 3
        rb_with_tls = set()
        for tl in nusc_map.traffic_light:
            rb_with_tls.add(tl["from_road_block_token"])
        lane_feature_ls = []
        for lane_id, lane in enumerate(lanes):
            lane_info = vectorize_lane(nusc_map, lane, lane_id, ego_pos_x, ego_pos_y, resolution_meters=2)
            if len(lane_info) == 0:
                continue
            halluc_lane_1, halluc_lane_2 = np.empty(
                (0, 3*2)), np.empty((0, 3*2))
            center_lane = np.empty((0, 3*2))
            is_traffic_control = False
            is_junction = False
            turn_direction = None

            for i in range(len(lane_info)-1):
                #print(i,len(lane_info) - 1, lane_info[i])
                before = [lane_info[i][0], lane_info[i][1]]
                after = [lane_info[i+1][0], lane_info[i+1][1]]
                
                if is_junction == False:
                    is_junction = is_intersection(nusc_map, before[0], before[1])
                if is_traffic_control == False:
                    is_traffic_control = has_traffic_light(nusc_map, before[0], before[1], rb_with_tls)

                if i < (len(lane_info) - 2):
                    ego_vec = [after[1] - before[1],
                                after[0] - before[0]]
                    ego_angle = np.rad2deg(
                            angle_vectors(ego_vec, [1, 0]))
                    next = [lane_info[i+2][0], lane_info[i+2][1]]
                    next_ego_vec = [next[1] - after[1],
                                next[0] - after[0]]
                    next_ego_angle = np.rad2deg(
                            angle_vectors(next_ego_vec, [1, 0]))
            
                    if (ego_angle < -360.0):
                        ego_angle = ego_angle + 360.0
                    if (next_ego_angle < -360.0):
                        next_ego_angle = next_ego_angle + 360.0
                    if (next_ego_angle > ego_angle):
                        turn_direction = "right"  # right
                    elif (next_ego_angle < ego_angle):
                        turn_direction = "left"  # left
                np_distance = np.array(
                    [after[t] - before[t] for t in range(len(before))])

                norm = np.linalg.norm(np_distance)
                e1, e2 = rotate_quat @ np_distance / norm, rotate_quat.T @ np_distance / norm
                lane_1 = np.hstack((before + e1 * lane_width/2, lane_info[i][2],
                                    after + e1 * lane_width/2, lane_info[i][2]))
                lane_2 = np.hstack((before + e2 * lane_width/2, lane_info[i][2],
                                    after + e2 * lane_width/2, lane_info[i][2]))
                lane_c = np.hstack((before, lane_info[i][2],
                                    after, lane_info[i][2]))
                halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
                halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
                center_lane = np.vstack((center_lane, lane_c))
            lane_feature_ls.append(
                [halluc_lane_1, halluc_lane_2, center_lane, turn_direction, is_traffic_control, is_junction, (lane_id, lane_id)])
                
            #plt.scatter(lane_info[i][0], lane_info[i][1], s=0.1, c="red")
            pts.extend(lane_info)
        np.save(args.stored_path + scene_name + '.npy', np.array(lane_feature_ls))
        # xlim = [300, 500]
        # ylim = [1000, 1200]
        # crop = Polygon(
        #     [[xlim[0], ylim[0]], [xlim[0], ylim[1]], [xlim[1], ylim[1]], [xlim[1], ylim[0]]]
        # )
        # pts = [pt for pt in pts if crop.contains(Point(pt[0], pt[1]))]        
        # intsec = is_intersection(ego_pos_x, ego_pos_y)
        # traf = has_traffic_light(ego_pos_x, ego_pos_y, rb_with_tls)


        # for pt in tqdm(pts):
        #     intsec = is_intersection(pt[0], pt[1])
        #     traf = has_traffic_light(pt[0], pt[1], rb_with_tls)
        #     color = None
        #     if intsec and traf:
        #         color = "purple"
        #     elif intsec and not traf:
        #         color = "red"
        #     elif not intsec and traf:
        #         color = "orange"
        #     else:
        #         color = "green"
        #     plt.scatter(pt[0], pt[1], s=0.1, c=color)

        # #plt.savefig("tmp.png")
        # plt.show()

def initial_topology(args):
    for now_data in tqdm(sorted(os.listdir(args.stored_path))):
        if now_data == '.npy':
            continue
        split_name = now_data.split('_')
        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[5] + '_' + split_name[6]
        source_path = args.stored_path + '/' + now_data
        destination_path = args.initial_topology_path + '/' + initial_name + '.npy'
        if not str(initial_name + '.npy') in os.listdir(args.initial_topology_path):
            shutil.copyfile(source_path, destination_path)

def collect_initial_topology(args):
    more_topology_num = 0
    for now_data in tqdm(sorted(os.listdir(args.input_path))):
        # attacker_id = now_data.split('_')[-1].split('.')[0]
        # scene_id = now_data.split('_')[6]
        # variant_id = now_data.split('_')[7]
        # sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
        split_name = now_data.split('_')
        # initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[5] + '_' + split_name[6]
        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
        
        # if initial_name != 'trainval_boston-seaport_scene-0681':
        #     continue
        if str(initial_name + '.npy') in os.listdir(args.initial_topology_path):
          continue
        print(initial_name)
        town_name = now_data.split('_')[4]
        scene_name = now_data.split('.')[0] + '.' + now_data.split('.')[1]
        
        more_topology_num += 1
        sce_df = pd.read_csv(args.input_path + now_data)
        ego_pos_x = 0
        ego_pos_y = 0
        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
            if track_id == 'ego':
                ego_pos_x = remain_df["X"].values[7]
                ego_pos_y = remain_df["Y"].values[7]
        nusc_map = NuScenesMap(dataroot=args.nuscenes_map_path, map_name=town_name)
        lanes = nusc_map.arcline_path_3
        rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
        lane_width = 3
        rb_with_tls = set()
        for tl in nusc_map.traffic_light:
            rb_with_tls.add(tl["from_road_block_token"])
        lane_feature_ls = []
        now_points = 0
        t_sum = 0
        for lane_id, lane in enumerate(lanes):
            lane_info = vectorize_lane(nusc_map, lane, lane_id, ego_pos_x, ego_pos_y, resolution_meters=args.sample_range)
            # print("lane_info:", lanes)
            plt.scatter(ego_pos_x, ego_pos_y, s=3, c="green")
            if len(lane_info) == 0:
                continue
            halluc_lane_1, halluc_lane_2 = np.empty(
                (0, 3*2)), np.empty((0, 3*2))
            center_lane = np.empty((0, 3*2))
            is_traffic_control = False
            is_junction = False
            turn_direction = "right"

            for i in range(len(lane_info)-1):
                #print(i,len(lane_info) - 1, lane_info[i])
                now_points += 1
                before = [lane_info[i][0], lane_info[i][1]]
                after = [lane_info[i+1][0], lane_info[i+1][1]]
                
                if is_junction == False:
                    is_junction = is_intersection(nusc_map, before[0], before[1])
                if is_traffic_control == False:
                    is_traffic_control = has_traffic_light(nusc_map, before[0], before[1], rb_with_tls)

                if i < (len(lane_info) - 2):
                    ego_vec = [after[1] - before[1],
                                after[0] - before[0]]
                    ego_angle = np.rad2deg(
                            angle_vectors(ego_vec, [1, 0]))
                    next = [lane_info[i+2][0], lane_info[i+2][1]]
                    next_ego_vec = [next[1] - after[1],
                                next[0] - after[0]]
                    next_ego_angle = np.rad2deg(
                            angle_vectors(next_ego_vec, [1, 0]))
            
                    if (ego_angle < -360.0):
                        ego_angle = ego_angle + 360.0
                    if (next_ego_angle < -360.0):
                        next_ego_angle = next_ego_angle + 360.0
                    if (next_ego_angle > ego_angle):
                        turn_direction = "right"  # right
                    elif (next_ego_angle < ego_angle):
                        turn_direction = "left"  # left
                np_distance = np.array(
                    [after[t] - before[t] for t in range(len(before))])

                norm = np.linalg.norm(np_distance)
                e1, e2 = rotate_quat @ np_distance / norm, rotate_quat.T @ np_distance / norm
                lane_1 = np.hstack((before + e1 * lane_width/2, lane_info[i][2],
                                    after + e1 * lane_width/2, lane_info[i][2]))
                lane_2 = np.hstack((before + e2 * lane_width/2, lane_info[i][2],
                                    after + e2 * lane_width/2, lane_info[i][2]))
                lane_c = np.hstack((before, lane_info[i][2],
                                    after, lane_info[i][2]))
                halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
                halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
                center_lane = np.vstack((center_lane, lane_c))
                plt.scatter(lane_info[i][0], lane_info[i][1], s=1, c="red")
                #plt.text(lane_info[i][0], lane_info[i][1], lane_id, c="black", fontsize =  8)
            t_sum += center_lane.shape[0]
            # print("center_lane:", t_sum)
            lane_feature_ls.append(
                [halluc_lane_1, halluc_lane_2, center_lane, turn_direction, is_traffic_control, is_junction, (lane_id, lane_id)])
                
                
            # print(lane_feature_ls)
        title = "Points:", str(now_points)
        plt.title(title)
        # plt.show()
        plt.savefig('./nuscenes_data/topo_fig/' + initial_name + '.png')
        plt.close()
        np.save(args.initial_topology_path + initial_name + '.npy', np.array(lane_feature_ls))

        
    print("more_topology_num:", more_topology_num)
    # t_sum = 0
    # topology = np.load("/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/initial_topology_4.3_1m/trainval_boston-seaport_scene-0681.npy", allow_pickle=True)
    # for i, _ in enumerate(topology):
    #     t_sum += topology[:, 2][i][:, :2].shape[0]
    #     print(i, topology[:, 2][i][:, :2].shape, t_sum)
    # print(center_lane.shape, topology[:, 2][16].shape)

def copy_initial_topology_without_scenario_type(args):
    for now_data in tqdm(sorted(os.listdir(args.original_initial_path))):
        split_name = now_data.split('_')
        initial_name = split_name[0] + '_' + split_name[1] + '_' + split_name[3]
        if str(initial_name + '.npy') in os.listdir(args.initial_topology_path):
            continue
        print(initial_name)
        shutil.copyfile(args.original_initial_path + now_data, args.initial_topology_path + initial_name)


# def collect_ego_lateral_offset_proposal(args):
#     more_topology_num = 0
#     for now_data in tqdm(sorted(os.listdir(args.input_path))):
#         # attacker_id = now_data.split('_')[-1].split('.')[0]
#         # scene_id = now_data.split('_')[6]
#         # variant_id = now_data.split('_')[7]
#         # sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
#         split_name = now_data.split('_')
#         # initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[5] + '_' + split_name[6]
#         initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
        
#         if initial_name != 'trainval_boston-seaport_scene-0093':
#             continue
#         # if str(initial_name + '.npy') in os.listdir(args.initial_topology_path):
#         #   continue
#         print(initial_name)
#         town_name = now_data.split('_')[4]
#         scene_name = now_data.split('.')[0] + '.' + now_data.split('.')[1]
        
#         more_topology_num += 1
#         sce_df = pd.read_csv(args.input_path + now_data)
#         ego_pos_x = 0
#         ego_pos_y = 0
#         for track_id, remain_df in sce_df.groupby("TRACK_ID"):
#             if track_id == 'ego':
#                 ego_pos_x = remain_df["X"].values[7]
#                 ego_pos_y = remain_df["Y"].values[7]
#         nusc_map = NuScenesMap(dataroot=args.nuscenes_map_path, map_name=town_name)
#         lanes = nusc_map.arcline_path_3
#         rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
#         lane_width = 3
#         lateral_offset = 1
#         rb_with_tls = set()
#         for tl in nusc_map.traffic_light:
#             rb_with_tls.add(tl["from_road_block_token"])
#         lane_feature_ls = []
#         now_points = 0
#         t_sum = 0
#         for lane_id, lane in enumerate(lanes):
#             lane_info = vectorize_lane(nusc_map, lane, lane_id, ego_pos_x, ego_pos_y, resolution_meters=args.sample_range)
#             # print("lane_info:", lanes)
#             # print(ego_pos_x, ego_pos_y)
#             # plt.scatter(ego_pos_x, ego_pos_y, s=3, c="green")
#             if len(lane_info) == 0:
#                 continue
#             halluc_lane_1, halluc_lane_2 = np.empty(
#                 (0, 3*2)), np.empty((0, 3*2))
#             proposal_left, proposal_right = np.empty(
#                 (0, 3*2)), np.empty((0, 3*2))
#             center_lane = np.empty((0, 3*2))
#             is_traffic_control = False
#             is_junction = False
#             turn_direction = "right"

#             for i in range(len(lane_info)-1):
#                 #print(i,len(lane_info) - 1, lane_info[i])
#                 now_points += 1
#                 before = [lane_info[i][0], lane_info[i][1]]
#                 after = [lane_info[i+1][0], lane_info[i+1][1]]
                
#                 if is_junction == False:
#                     is_junction = is_intersection(nusc_map, before[0], before[1])
#                 if is_traffic_control == False:
#                     is_traffic_control = has_traffic_light(nusc_map, before[0], before[1], rb_with_tls)

#                 if i < (len(lane_info) - 2):
#                     ego_vec = [after[1] - before[1],
#                                 after[0] - before[0]]
#                     ego_angle = np.rad2deg(
#                             angle_vectors(ego_vec, [1, 0]))
#                     next = [lane_info[i+2][0], lane_info[i+2][1]]
#                     next_ego_vec = [next[1] - after[1],
#                                 next[0] - after[0]]
#                     next_ego_angle = np.rad2deg(
#                             angle_vectors(next_ego_vec, [1, 0]))
            
#                     if (ego_angle < -360.0):
#                         ego_angle = ego_angle + 360.0
#                     if (next_ego_angle < -360.0):
#                         next_ego_angle = next_ego_angle + 360.0
#                     if (next_ego_angle > ego_angle):
#                         turn_direction = "right"  # right
#                     elif (next_ego_angle < ego_angle):
#                         turn_direction = "left"  # left
#                 np_distance = np.array(
#                     [after[t] - before[t] for t in range(len(before))])

#                 norm = np.linalg.norm(np_distance)
#                 e1, e2 = rotate_quat @ np_distance / norm, rotate_quat.T @ np_distance / norm
#                 lane_1 = np.hstack((before + e1 * lane_width/2, lane_info[i][2],
#                                     after + e1 * lane_width/2, lane_info[i][2]))
#                 lane_2 = np.hstack((before + e2 * lane_width/2, lane_info[i][2],
#                                     after + e2 * lane_width/2, lane_info[i][2]))
#                 lane_c = np.hstack((before, lane_info[i][2],
#                                     after, lane_info[i][2]))
#                 lane_l = np.hstack((before + e1 * lateral_offset/2, lane_info[i][2],
#                                     after + e1 * lateral_offset/2, lane_info[i][2]))
#                 lane_r = np.hstack((before + e2 * lateral_offset/2, lane_info[i][2],
#                                     after + e2 * lateral_offset/2, lane_info[i][2]))
#                 halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
#                 halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
#                 center_lane = np.vstack((center_lane, lane_c))
#                 proposal_left = np.vstack((proposal_left, lane_l))
#                 proposal_right = np.vstack((proposal_right, lane_r))
#                 # plt.scatter(lane_info[i][0], lane_info[i][1], s=1, c="red")
#                 plt.plot(halluc_lane_1[:, 0], halluc_lane_1[:, 1], color='black', label='lane_1')
#                 plt.plot(halluc_lane_2[:, 0], halluc_lane_2[:, 1], color='black', label='lane_2')
#                 plt.plot(center_lane[:, 0], center_lane[:, 1],
#                     color='gray', linestyle='--', label='centerline')
#                 plt.plot(proposal_left[:, 0], proposal_left[:, 1],
#                         color='gray', linestyle='--', label='centerline - 1m')
#                 plt.plot(proposal_right[:, 0], proposal_right[:, 1],
#                         color='gray', linestyle='--', label='centerline + 1m')

#                 #plt.text(lane_info[i][0], lane_info[i][1], lane_id, c="black", fontsize =  8)
#             t_sum += center_lane.shape[0]
#             # print("center_lane:", t_sum)
#             lane_feature_ls.append(
#                 [halluc_lane_1, halluc_lane_2, center_lane, turn_direction, is_traffic_control, is_junction, (lane_id, lane_id)])
#             # print(lane_feature_ls)
#         plt.scatter(ego_pos_x, ego_pos_y, s=100, c="red")
#         plt.xlim(ego_pos_x - 10, ego_pos_x + 10)
#         plt.ylim(ego_pos_y - 10, ego_pos_y + 10)
#         title = "Points:", str(now_points)
#         plt.title(title)
#         plt.show()
#         # plt.savefig('./nuscenes_data/topo_fig/' + initial_name + '.png')
#         plt.close()
#         # np.save(args.initial_topology_path + initial_name + '.npy', np.array(lane_feature_ls))
#     print("more_topology_num:", more_topology_num)

def aggregate_lane_error(lane_points, ego_traj):
    # 建立該車道的 LineString
    lane_line = LineString(lane_points)
    # 計算每個 ego 點到該車道的距離
    distances = [Point(pt).distance(lane_line) for pt in ego_traj]
    # 計算平均距離或均方誤差
    avg_distance = np.mean(distances)
    return avg_distance

def line_substring(line, start_dist, end_dist, num_points=100):
    """
    從 LineString 中取得子線段 (substring)，從 start_dist 到 end_dist 的部分。
    :param line: 待提取子線段的 LineString
    :param start_dist: 子線段起始距離 (從線首測量)
    :param end_dist: 子線段結束距離 (從線首測量)
    :param num_points: 用於插值的點數（數值越大結果越平滑）
    :return: 一個新的 LineString 物件代表子線段
    """
    # 保證距離值在合理範圍內
    if start_dist < 0:
        start_dist = 0
    if end_dist > line.length:
        end_dist = line.length
    if start_dist >= end_dist:
        return LineString([])

    # 生成 num_points 個均勻分布的距離值
    distances = np.linspace(start_dist, end_dist, num_points)
    points = [line.interpolate(d) for d in distances]
    return LineString(points)

def collect_ego_lateral_offset_proposal(args):
    more_topology_num = 0
    for now_data in tqdm(sorted(os.listdir(args.input_path))):
        sce_name = now_data[:-4]
        split_name = now_data.split('_')
        # 以 split_name[3], [4], [6] 組成 initial_name
        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
        # 例如，只處理特定場景
        # if initial_name != 'trainval_boston-seaport_scene-0093':
        #     continue
        if str(sce_name + '.npy') in os.listdir(args.lateral_offset_path):
            continue
        print("Processing scene:", initial_name)
        town_name = now_data.split('_')[4]
        scene_name = now_data.split('.')[0] + '.' + now_data.split('.')[1]
        more_topology_num += 1
        
        # 讀取 CSV 檔案
        sce_df = pd.read_csv(os.path.join(args.input_path, now_data))
        
        # 提取 ego 車的所有軌跡，並只保留從第7個 frame 開始
        ego_traj_df = sce_df[sce_df["TRACK_ID"] == "ego"].copy()
        if len(ego_traj_df) < 8:
            print("Warning: ego trajectory frames < 8, skipping ...")
            continue
        ego_traj_df = ego_traj_df.iloc[7:, :].sort_values(by=["TIMESTAMP"])
        
        ego_pos_x = ego_traj_df["X"].values[0]
        ego_pos_y = ego_traj_df["Y"].values[0]

        # 提取 ego 的 (x,y) 軌跡
        ego_traj_xy = list(zip(ego_traj_df["X"].values, ego_traj_df["Y"].values))
        # 使用 ego 軌跡的第一個點作為參考 (亦可取多點平均，但此處示例取首點)
        ref_x, ref_y = ego_traj_xy[0]
        
        # 取得 NuScenes Map
        nusc_map = NuScenesMap(dataroot=args.nuscenes_map_path, map_name=town_name)
        lanes = nusc_map.arcline_path_3
        rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
        lane_width = 3
        lateral_offset = args.lateral_offset  # 1 meter offset
        
        rb_with_tls = set()
        for tl in nusc_map.traffic_light:
            rb_with_tls.add(tl["from_road_block_token"])
        
        # 收集所有候選車道 (離散點，僅2D)
        all_lane_info = []
        for lane_id, lane in enumerate(lanes):
            lane_points = vectorize_lane(nusc_map, lane, lane_id, ref_x, ref_y, resolution_meters=args.sample_range)
            if len(lane_points) == 0:
                continue
            discrete_xy = [(pt[0], pt[1]) for pt in lane_points]
            all_lane_info.append((lane_id, discrete_xy))
        
        # 對所有候選車道計算與整段 ego 軌跡的平均距離，選出最佳車道
        best_lane_id = None
        min_error = float('inf')
        best_lane_points = None
        for lane_id, lane_xy in all_lane_info:
            error = aggregate_lane_error(lane_xy, ego_traj_xy)
            if error < min_error:
                min_error = error
                best_lane_id = lane_id
                best_lane_points = lane_xy
        print("Selected best_lane_id:", best_lane_id, "with avg error:", min_error)
        if best_lane_points is None:
            continue
        best_lane_array = np.array(best_lane_points)  # shape (N,2)
        
        # 生成車道線提案：我們希望僅使用自車前進方向的部分，
        # 可利用 LineString 的 project 方法取得 ego 在最佳車道上的進度，
        # 並僅取大於該進度的部分。
        lane_line = LineString(best_lane_array)
        ego_proj = lane_line.project(Point(ref_x, ref_y))
        # 取 substring: [ego_proj, lane_line.length] 即前方
        # front_lane_line = lane_line.substring(ego_proj, lane_line.length)
        front_lane_line = line_substring(lane_line, ego_proj, lane_line.length)
        # 將前方部分離散化 (可以用 front_lane_line.coords)
        front_lane_points = list(front_lane_line.coords)
        front_lane_array = np.array(front_lane_points)
        
        # 此時定義：
        # lane_c 為前方部分（中心線）
        lane_c = front_lane_array
        
        # 根據每個連續區段計算法向量生成左右偏移路徑
        proposal_left_points = []
        proposal_right_points = []
        # lane_l = []
        # lane_r = []
        for i in range(len(front_lane_array)-1):
            before = front_lane_array[i]
            after = front_lane_array[i+1]
            seg = after - before
            norm = np.linalg.norm(seg)
            if norm < 1e-6:
                continue
            seg_dir = seg / norm
            # 計算法向量（逆時針旋轉90度）
            normal = np.array([-seg_dir[1], seg_dir[0]])
            proposal_left_points.append(before + normal * (lateral_offset/2))
            proposal_right_points.append(before - normal * (lateral_offset/2))
            # lane_l.append(before + normal * (lane_width/2))
            # lane_r.append(before - normal * (lane_width/2))
        # 加上最後一個點
        if len(front_lane_array) > 0:
            proposal_left_points.append(front_lane_array[-1] + normal * (lateral_offset/2))
            proposal_right_points.append(front_lane_array[-1] - normal * (lateral_offset/2))
        
        proposal_left = np.array(proposal_left_points)
        proposal_right = np.array(proposal_right_points)

        candidate_boundaries = []  # 保存每個候選車道的 (boundary_left, boundary_right)
        for lane_id, lane_xy in all_lane_info:
            lane_arr = np.array(lane_xy)
            lane_line_candidate = LineString(lane_arr)
            front_lane_array_candidate = np.array(lane_line_candidate.coords)
            # front_lane_array_candidate = lane_line_candidate.project(Point(ref_x, ref_y))
            # # 只取前方部分
            # front_lane_line_candidate = line_substring(lane_line_candidate, ego_proj_candidate, lane_line_candidate.length)
            # front_lane_points_candidate = list(front_lane_line_candidate.coords)
            # front_lane_array_candidate = np.array(front_lane_points_candidate)
            
            # 若不足以計算，跳過該 lane
            # if front_lane_array_candidate.shape[0] < 2:
            #     continue
            
            boundary_left_points = []
            boundary_right_points = []
            for i in range(len(front_lane_array_candidate)-1):
                before = front_lane_array_candidate[i]
                after = front_lane_array_candidate[i+1]
                seg = after - before
                norm = np.linalg.norm(seg)
                if norm < 1e-6:
                    continue
                seg_dir = seg / norm
                normal = np.array([-seg_dir[1], seg_dir[0]])
                boundary_left_points.append(before + normal * (lane_width/2))
                boundary_right_points.append(before - normal * (lane_width/2))
            if len(front_lane_array_candidate) > 0:
                boundary_left_points.append(front_lane_array_candidate[-1] + normal * (lane_width/2))
                boundary_right_points.append(front_lane_array_candidate[-1] - normal * (lane_width/2))
            boundary_left = np.array(boundary_left_points)
            boundary_right = np.array(boundary_right_points)
            candidate_boundaries.append((lane_id, boundary_left, boundary_right))
        
        # 繪圖
        plt.figure()
        # 畫出所有候選車道邊界 (黑色實線)
        for lane_id, b_left, b_right in candidate_boundaries:
            if b_left.shape[0] > 1:
                plt.plot(b_left[:,0], b_left[:,1], color='black', linewidth=1, label="Lane Boundary (Left)" if lane_id == candidate_boundaries[0][0] else None)
            if b_right.shape[0] > 1:
                plt.plot(b_right[:,0], b_right[:,1], color='black', linewidth=1, label="Lane Boundary (Right)" if lane_id == candidate_boundaries[0][0] else None)
        
        
        # 畫出 ego 軌跡（紅色點）
        plt.scatter([pt[0] for pt in ego_traj_xy], [pt[1] for pt in ego_traj_xy], s=25, color='red', label="Ego trajectory")
        
        # 畫出所選車道中心線 (紅色虛線)
        if lane_c.shape[0] > 1:
            plt.plot(lane_c[:,0], lane_c[:,1], color='purple', linestyle='--', label="Centerline")
        # 畫出左右偏移 proposal (紅色虛線)
        if proposal_left.shape[0] > 1:
            plt.plot(proposal_left[:,0], proposal_left[:,1], color='violet', linestyle='--', label="Proposal Left (+1m)")
        if proposal_right.shape[0] > 1:
            plt.plot(proposal_right[:,0], proposal_right[:,1], color='violet', linestyle='--', label="Proposal Right (-1m)")

        plt.xlim(ego_pos_x - 25, ego_pos_x + 25)
        plt.ylim(ego_pos_y - 25, ego_pos_y + 25)

        # plt.title(f"Ego Lateral Offset Proposals for {initial_name} (Best Lane: {best_lane_id})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        # plt.show()
        # print(os.path.join(args.lateral_offset_img_path, initial_name + '.png'))
        # exit()
        
        plt.savefig(os.path.join(args.lateral_offset_img_path, sce_name + '.png'))
        plt.close()
        
        # 若需要存檔則可將 lane_c, proposal_left, proposal_right 存為 .npy 檔案
        np.save(os.path.join(args.lateral_offset_path, sce_name + '.npy'),
            np.array([lane_c, proposal_left, proposal_right]))
        
    print("more_topology_num:", more_topology_num)

# def collect_ego_lateral_offset_proposal(args):
#     more_topology_num = 0
#     for now_data in tqdm(sorted(os.listdir(args.input_path))):
#         # attacker_id = now_data.split('_')[-1].split('.')[0]
#         # scene_id = now_data.split('_')[6]
#         # variant_id = now_data.split('_')[7]
#         # sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
#         split_name = now_data.split('_')
#         # initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[5] + '_' + split_name[6]
#         initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
        
#         if initial_name != 'trainval_boston-seaport_scene-0093':
#             continue
#         # if str(initial_name + '.npy') in os.listdir(args.initial_topology_path):
#         #   continue
#         print(initial_name)
#         town_name = now_data.split('_')[4]
#         scene_name = now_data.split('.')[0] + '.' + now_data.split('.')[1]
        
#         more_topology_num += 1
#         sce_df = pd.read_csv(args.input_path + now_data)

#         ego_traj = sce_df[sce_df["TRACK_ID"] == "ego"].copy()
#         if len(ego_traj) < 8:
#             print("Warning: ego trajectory frames < 8, skipping ...")
#             continue
#         # 將 index=7 以前的捨棄
#         ego_traj = ego_traj.iloc[7:, :]  # 只從第7行(含)開始
#         # 由於我們可能要計算方向，需要知道相鄰點
#         # 建議按時間或 frame 排序，避免雜亂
#         ego_traj = ego_traj.sort_values(by=["TIMESTAMP"])
#         all_lane_info = []
#         ego_pos_x = ego_traj["X"].iloc[0]
#         ego_pos_y = ego_traj["Y"].iloc[0]

#         # ego_pos_x = 0
#         # ego_pos_y = 0
#         # for track_id, remain_df in sce_df.groupby("TRACK_ID"):
#         #     if track_id == 'ego':
#         #         ego_pos_x = remain_df["X"].values[7]
#         #         ego_pos_y = remain_df["Y"].values[7]
#         nusc_map = NuScenesMap(dataroot=args.nuscenes_map_path, map_name=town_name)
#         lanes = nusc_map.arcline_path_3
#         rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
#         lane_width = 3
#         lateral_offset = 1
#         rb_with_tls = set()
#         for tl in nusc_map.traffic_light:
#             rb_with_tls.add(tl["from_road_block_token"])
#         lane_feature_ls = []
#         now_points = 0
#         t_sum = 0
#         for lane_id, lane in enumerate(lanes):
#             lane_info = vectorize_lane(nusc_map, lane, lane_id, ego_pos_x, ego_pos_y, resolution_meters=args.sample_range)
#             lane_points = vectorize_lane(nusc_map, lane, lane_id,
#                                          ego_traj["X"].iloc[0], ego_traj["Y"].iloc[0],
#                                          resolution_meters=args.sample_range)
#             if len(lane_points) == 0:
#                 continue
#             # lane_points ~ [(x0,y0,z0), (x1,y1,z1), ...]
#             # 我們只要 x,y => 2D
#             discrete_xy = [(pt[0], pt[1]) for pt in lane_points]
#             all_lane_info.append((lane_id, discrete_xy))
#         best_lane_id = None
#         min_error = float('inf')
#         best_lane_points = None
#         ego_traj_xy = list(zip(ego_traj['X'], ego_traj['Y']))
#         for lane_id, lane_xy in all_lane_info:
#             error = aggregate_lane_error(lane_xy, ego_traj_xy)
#             if error < min_error:
#                 min_error = error
#                 best_lane_id = lane_id
#                 best_lane_points = lane_xy
                
#         print("Selected best_lane_id:", best_lane_id, "with avg error:", min_error)
        
#         # 如果找不到合適車道則跳過
#         if best_lane_points is None:
#             continue
#         best_lane_array = np.array(best_lane_points)  # shape = (N,2)
        
#         # 生成中心線 proposal (lane_c)
#         lane_c = best_lane_array  # 中心線即為該車道離散點
        
#         # 產生左右偏移 proposal
#         # 假設我們需要將每個連續點段計算出法向量，並平移 lateral_offset/2
#         lateral_offset = 1  # 1 m
        
#         proposal_left_points = []
#         proposal_right_points = []
#         # 遍歷連續兩點計算，並以線性方式生成偏移線
#         for i in range(len(best_lane_array)-1):
#             before = best_lane_array[i]
#             after = best_lane_array[i+1]
#             seg = after - before
#             norm = np.linalg.norm(seg)
#             if norm < 1e-6:
#                 continue
#             seg_dir = seg / norm
#             # 求法向量（逆時針旋轉90度）
#             normal = np.array([-seg_dir[1], seg_dir[0]])
#             proposal_left_points.append(before + normal * (lateral_offset/2))
#             proposal_right_points.append(before - normal * (lateral_offset/2))
#         # 加上最後一個點
#         proposal_left_points.append(best_lane_array[-1] + normal * (lateral_offset/2))
#         proposal_right_points.append(best_lane_array[-1] - normal * (lateral_offset/2))
        
#         proposal_left = np.array(proposal_left_points)
#         proposal_right = np.array(proposal_right_points)

#         plt.plot(lane_c[:,0], lane_c[:,1], color='gray', linestyle='--', label="Centerline")
#         plt.plot(proposal_left[:,0], proposal_left[:,1], color='gray', linestyle='--', label="Centerline - 1m")
#         plt.plot(proposal_right[:,0], proposal_right[:,1], color='gray', linestyle='--', label="Centerline + 1m")
#         plt.scatter(ego_pos_x, ego_pos_y, s=100, c="red")
#         plt.xlim(ego_pos_x - 10, ego_pos_x + 10)
#         plt.ylim(ego_pos_y - 10, ego_pos_y + 10)
#         title = "Points:", str(now_points)
#         plt.title(title)
#         plt.show()
#         # plt.savefig('./nuscenes_data/topo_fig/' + initial_name + '.png')
#         plt.close()
#         # np.save(args.initial_topology_path + initial_name + '.npy', np.array(lane_feature_ls))
#     print("more_topology_num:", more_topology_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_map_path', type=str, default='./NuScenes/')
    # parser.add_argument('--input_path', type=str, default='./nuscenes_data/all_traj/')
    # parser.add_argument('--input_path', type=str, default='../csvs_4.5/')
    parser.add_argument('--input_path', type=str, default='nuscenes_data/trajectory/test/') # test
    parser.add_argument('--stored_path', type=str, default='./nuscenes_data/topology/')
    parser.add_argument('--original_initial_path', type=str, default='./nuscenes_data/initial_topology_4.3/')
    parser.add_argument('--initial_topology_path', type=str, default='./nuscenes_data/initial_topology/')
    parser.add_argument('--lateral_offset_path', type=str, default='./nuscenes_data/lateral_offset_np_test/')
    parser.add_argument('--lateral_offset_img_path', type=str, default='./nuscenes_data/lateral_offset_img_test/')
    parser.add_argument('--sample_range', type=float, default=2.0)
    parser.add_argument('--lateral_offset', type=float, default=2.0)
    
    args = parser.parse_args()

    #save_topology(args)
    #initial_topology(args)

    # default
    # collect_initial_topology(args)
    collect_ego_lateral_offset_proposal(args)

    #copy_initial_topology_without_scenario_type(args)
    # main(NuScenesMap(dataroot="./NuScenes/", map_name="singapore-onenorth"))
