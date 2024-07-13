from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point, Polygon
import sys
import os
import argparse
import csv
import pandas as pd
import numpy as np
import math
import shutil
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
            print("lane_info:", lanes)
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
        exit()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_map_path', type=str, default='./NuScenes/')
    # parser.add_argument('--input_path', type=str, default='./nuscenes_data/all_traj/')
    parser.add_argument('--input_path', type=str, default='../csvs_4.5/')
    parser.add_argument('--stored_path', type=str, default='./nuscenes_data/topology/')
    parser.add_argument('--original_initial_path', type=str, default='./nuscenes_data/initial_topology_4.3/')
    parser.add_argument('--initial_topology_path', type=str, default='./nuscenes_data/initial_topology/')
    parser.add_argument('--sample_range', type=float, default=2.0)
    args = parser.parse_args()

    #save_topology(args)
    #initial_topology(args)
    collect_initial_topology(args)
    #copy_initial_topology_without_scenario_type(args)
    # main(NuScenesMap(dataroot="./NuScenes/", map_name="singapore-onenorth"))
