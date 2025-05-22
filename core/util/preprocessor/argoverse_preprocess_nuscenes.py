# About: script to processing argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.07.16

import os
import argparse
from os.path import join as pjoin
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse
import csv
import sys
import warnings
import json
# import torch
from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point, Polygon

from core.util.preprocessor.base import Preprocessor
from core.util.cubic_spline import Spline2D
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence, Union
import math
import random
warnings.filterwarnings("ignore")

nusc_map = NuScenesMap(dataroot="./NuScenes/", map_name="singapore-onenorth")

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
    
def vectorize_lane(lane, resolution_meters=1):
    arclines = nusc_map.get_arcline_path(lane)
    ret = []
    for arc in arclines:
        poses = arcline_path_utils.discretize(arc, resolution_meters)
        ret.extend(poses)
    return ret

def is_intersection(x, y):
    rstk = nusc_map.record_on_point(x, y, "road_segment")
    if rstk == "":
        return True
    rs = nusc_map.get("road_segment", rstk)
    return rs["is_intersection"]

def has_traffic_light(x, y, rb_with_tls):
    rbtk = nusc_map.record_on_point(x, y, "road_block")
    if rbtk == "":
        return False
    return rbtk in rb_with_tls

@lru_cache(128)
def _read_csv(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """A caching CSV reader

    Args:
        path: Path to the csv file
        *args, **kwargs: optional arguments to be used while data loading

    Returns:
        pandas DataFrame containing the loaded csv
    """
    #print(str(path).split('.')[1])
    print(str(path).split('.')[1] + '.' + str(path).split('.')[2])
    return pd.read_csv(path, *args, **kwargs)

class NuScenesLoader:
    def __init__(self, root_dir: Union[str, Path]):
        """Initialization function for the class.

        Args:
            root_dir: Path to the folder having sequence csv files
        """
        self.counter: int = 0

        root_dir = Path(root_dir)
        self.seq_list: Sequence[Path] = [(root_dir / x).absolute() for x in os.listdir(root_dir)]

        self.current_seq: Path = self.seq_list[self.counter]
        print(self.current_seq)


    @property
    def track_id_list(self) -> List[int]:
        """Get the track ids in the current sequence.

        Returns:
            list of track ids in the current sequence
        """
        print(self.seq_df)
        _track_id_list: List[int] = np.unique(self.seq_df["TRACK_ID"].values).tolist()
        return _track_id_list

    @property
    def city(self) -> str:
        """Get the city name for the current sequence.

        Returns:
            city name, i.e., either 'PIT' or 'MIA'
        """
        # _city: str = self.seq_df["CITY_NAME"].values[0]
        _city: str = str(self.current_seq).split('_')[4]
        return _city

    @property
    def num_tracks(self) -> int:
        """Get the number of tracks in the current sequence.

        Returns:
            number of tracks in the current sequence
        """
        return len(self.track_id_list)

    @property
    def seq_df(self) -> pd.DataFrame:
        """Get the dataframe for the current sequence.

        Returns:
            pandas DataFrame for the current sequence
        """
        return _read_csv(self.current_seq)

    @property
    def agent_traj(self) -> np.ndarray:
        """Get the trajectory for the track of type 'AGENT' in the current sequence.

        Returns:
            numpy array of shape (seq_len x 2) for the agent trajectory
        """
        agent_x = self.seq_df[self.seq_df["TRACK_ID"] == "ego"]["X"]
        agent_y = self.seq_df[self.seq_df["TRACK_ID"] == "ego"]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y))
        return agent_traj

    def __iter__(self) -> "ArgoverseForecastingLoader":
        """Iterator for enumerating over sequences in the root_dir specified.

        Returns:
            Data Loader object for the first sequence in the data
        """
        self.counter = 0
        return self

    def __next__(self) -> "ArgoverseForecastingLoader":
        """Get the Data Loader object for the next sequence in the data.

        Returns:
            Data Loader object for the next sequence in the data
        """
        if self.counter >= len(self):
            raise StopIteration
        else:
            self.current_seq = self.seq_list[self.counter]
            self.counter += 1
            return self

    def __len__(self) -> int:
        """Get the number of sequences in the data

        Returns:
            Number of sequences in the data
        """
        return len(self.seq_list)

    def __str__(self) -> str:
        """Decorator that returns a string storing some stats of the current sequence

        Returns:
            A string storing some stats of the current sequence
        """
        return f"""Seq : {self.current_seq}
        ----------------------
        || City: {self.city}
        || # Tracks: {len(self.track_id_list)}
        ----------------------"""

    def __getitem__(self, key: int) -> "ArgoverseForecastingLoader":
        """Get the DataLoader object for the sequence corresponding to the given index.

        Args:
            key: index of the element

        Returns:
            Data Loader object for the given index
        """

        self.counter = key
        self.current_seq = self.seq_list[self.counter]
        return self

    def get(self, seq_id: Union[Path, str]) -> "ArgoverseForecastingLoader":
        """Get the DataLoader object for the given sequence path.

        Args:
            seq_id: Fully qualified path to the sequence

        Returns:
            Data Loader object for the given sequence path
        """
        self.current_seq = Path(seq_id).absolute()
        return self

class ArgoversePreprocessor(Preprocessor):
    def __init__(self,
                 root_dir,
                 split="train",
                 algo="tnt",
                 obs_horizon=8,
                 obs_range=100,
                 pred_horizon=30,
                 normalized=True,
                 save_dir=None):
        super(ArgoversePreprocessor, self).__init__(root_dir, algo, obs_horizon, obs_range, pred_horizon)

        #self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        #self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "walker": "#d3e8ef", "vehicle": "#007672"}

        self.split = split
        self.normalized = normalized
        self.obs_horizon = obs_horizon

        self.augment = False
        self.data_multiple = 1
        self.max_degree = 10

        #self.am = ArgoverseMap()
        #self.loader = NuScenesLoader(pjoin(self.root_dir, self.split+"_obs" if split == "test" else split))
        self.loader = NuScenesLoader(pjoin(self.root_dir, split))


        self.save_dir = save_dir

    def __getitem__(self, idx):
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        path, seq_f_name_ext = os.path.split(f_path)
        seq_f_name, ext = os.path.splitext(seq_f_name_ext)
        df = copy.deepcopy(seq.seq_df)
        ################################
        ################################
        debug = True
        ################################
        ################################
        if debug:
            return self.process_and_save(df, seq_id=seq_f_name, dir_=self.save_dir, data_num = self.data_multiple)
        else:
            if str(f_path).split('.')[-1] == 'csv':
                if not 'features_' +  str(f_path).split('/')[-1].replace('csv', 'pkl') in os.listdir(self.save_dir + '/' + self.split + '_intermediate/raw/'):
                    return self.process_and_save(df, seq_id=seq_f_name, dir_=self.save_dir, data_num = self.data_multiple)
                else:
                    return True
            else:
                return True

    def process(self, dataframe: pd.DataFrame,  seq_id, map_feat=True, data_index=0):
        data = self.read_argo_data(dataframe, seq_id, self.obs_horizon, self.split)
        data = self.get_obj_feats(data, seq_id)

        if self.augment:
            last_underscore_index = seq_id.rfind("_")
            if last_underscore_index != -1:
                number_to_insert = str(data_index) + "_"
                seq_id = seq_id[:last_underscore_index + 1] + number_to_insert + seq_id[last_underscore_index + 1:]
        data['graph'] = self.get_lane_graph(data, seq_id)
        data['seq_id'] = seq_id
        # visualization for debug purpose
        # self.visualize_data(data)
        return seq_id, pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def read_argo_data(df: pd.DataFrame, seq_id, obs_horizon, split):
        ###############################
        attacker_id = seq_id.split('_')[-1].split('.')[0]
        scene_id = seq_id.split('_')[6]
        variant_id = seq_id.split('_')[7]
        sce_temp = scene_id + '_' + attacker_id + '_' + variant_id
        city = seq_id.split('_')[4]

        # with open('/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/filter_agent.json') as f:
        #     data = json.load(f)
        # df = (df.pivot_table(columns='TIMESTAMP', index=['TRACK_ID'], fill_value=0)
        #         .stack('TIMESTAMP')
        #         .sort_index(level=['TRACK_ID','TIMESTAMP'])
        #         .reset_index())
        # if sce_temp in data:
        #     #print("filter:", list(data[sce_temp]), df)
        #     for idx in range(len(list(data[sce_temp]))):
        #         #print(list(data[sce_temp])[idx])
        #         parked_idx = df[df["TRACK_ID"] == list(data[sce_temp])[idx]].index
        #         df = df.drop(parked_idx).reset_index(drop=True)
        #         #print(df)
        # need_csv = False
        # if need_csv:
        #     df.to_csv('./nuscenes_data/padding_trajectory/' + split + '/' + seq_id + '.csv', index=False)
        
        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)
        #print(len(trajs), len(trajs[0]), df) #100 2

        
        
        
        yaws = df.YAW.to_numpy()

        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        
        
        objs = df.groupby(['TRACK_ID']).groups
        keys = list(objs.keys())
        #track_id = [x[0] for x in keys]
        #print(seq_id, keys)
        

        agt_idx = keys.index('ego')
        idcs = objs[keys[agt_idx]]

        agt_traj = trajs[idcs]
        agt_step = steps[idcs]
        
        ###############################
        agt_yaw = yaws[idcs]

        #attacker_type = [x[0] for x in keys]
        attacker_idx = keys.index(attacker_id)
        attacker_idcs = objs[keys[attacker_idx]]
        attacker_traj = trajs[attacker_idcs]
        attacker_yaw = yaws[attacker_idcs]
        attacker_pos_offset = attacker_traj[-1] - agt_traj[-1]
        attacker_yaw_offset = attacker_yaw[-1] - agt_yaw[-1]
        # print(agt_traj, agt_yaw)

        # -360~360 => 0~180
        # atr_origin_to_col = attacker_traj[0] - agt_traj[-2]
        atr_origin_to_col = attacker_traj[-2] - agt_traj[-1]
        u = agt_traj[-1] - agt_traj[-2]
        cross = np.cross(atr_origin_to_col, u)
        attacker_yaw_offset = (attacker_yaw_offset + 360.0) if attacker_yaw_offset < 0 else attacker_yaw_offset
        attacker_yaw_offset = abs(attacker_yaw_offset - 360.0) if attacker_yaw_offset > 180 else attacker_yaw_offset

        ###############################
        ## ctx_trajs: other's all traj
        del keys[agt_idx]
        ctx_trajs, ctx_steps, ctx_yaws, relative_pos, tp_gt_list = [], [], [], [], []
        #print("trajs:", trajs.shape)
        filter_num = 0
        l = []
        for key in keys:
            
            # if key in data:
            #     filter_num+=1
            #     continue
            
            idcs = objs[key]
            #print(key, idcs)
            # if trajs[idcs][obs_horizon - 1][0] == 0 and trajs[idcs][obs_horizon - 1][1] == 0:
            #     filter_num+=1
            #     continue
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])
            #print(key, trajs[idcs])
            ################################
            ctx_yaws.append(yaws[idcs])
            #print(trajs[idcs].shape)
            
            relative_pos_now = trajs[idcs][obs_horizon - 1] - agt_traj[obs_horizon - 1]
            
            #print(relative_pos_now)
            relative_pos.append(relative_pos_now)
            
            tp_gt_list.append([1] if key == attacker_id else [0])
        #print(filter_num, "tp_cand:", len(ctx_trajs))
        #print(ctx_trajs)
        #print("other:", agt_traj)
        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps

        data['attacker_traj'] = [attacker_traj]
        data['attacker_yaw'] = [attacker_yaw]
        data['attacker_pos_offset'] = [attacker_pos_offset]
        data['attacker_yaw_offset'] = [attacker_yaw_offset]
        data['cross'] = [cross]
        data['yaws'] = [agt_yaw] + ctx_yaws
        data['other_vehicle_num'] = len(ctx_trajs)
        data['attacker_id'] = attacker_id
        data['relative_pos'] = [relative_pos]
        data['tp_gt'] = tp_gt_list
        # print(len(ctx_trajs), len(ctx_trajs[0]))# 4 20
        # print(data['city'])
        # print(data['trajs'])
        # print(data['steps'])
        # print(data['attacker_traj'])
        # print(data['attacker_yaw'])
        # print(data['attacker_pos_offset'])
        # print(data['attacker_yaw_offset'])
        # print(data['yaws'])
        # print(data['other_vehicle_num'])
        # print(data['attacker_id'])
        # print(data['relative_pos'])
        # print(data['tp_gt'])
        return data

    def get_obj_feats(self, data, id):
        # get the origin and compute the oritentation of the target agent
        #print(id, np.array(data['trajs']).shape)
        #print("data['trajs'][0]:",data, id, np.array(data['trajs']).shape, data['trajs'][0])
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float32) # [0] means "EGO"
        #print("obs:", data['trajs'][0][self.obs_horizon-1])

        # print("orig:", orig) # the position of Agent after 2 s (70.4, -11.37)

        # comput the rotation matrix
        if self.normalized:
            #pre, conf, _ = self.am.get_lane_direction(traj=data['trajs'][0][:self.obs_horizon], city_name=data['city'])
            #if conf <= 0.1:
            #    pre = (orig - data['trajs'][0][self.obs_horizon-4]) / 2.0
            pre = (orig - data['trajs'][0][self.obs_horizon-4]) / 2.0
            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32)

        # get the target candidates and candidate gt
        agt_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float32)
        #print("agt_traj_obs:", agt_traj_obs)
        
        #agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32)
        ###############################################
        #agt_traj_fut = data['trajs'][0][self.obs_horizon:-1].copy().astype(np.float32)
        agt_traj_fut = data['trajs'][0][self.obs_horizon:].copy().astype(np.float32)
        #agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+3].copy().astype(np.float32)
        ###############################################
        #ctr_line_candts = self.am.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'])

        #topology = np.load("/home/carla/experiments/TNT-Trajectory-Predition/carla_dataset/topology_data/" + id + ".npy", allow_pickle=True)
        
        ###############################################
        #4.2: topology_id = id.split('_')[3] + '_' + id.split('_')[4] + '_' + id.split('_')[5] + '_' + id.split('_')[6]
        topology_id = id.split('_')[3] + '_' + id.split('_')[4] + '_' + id.split('_')[6]
        topology = np.load("/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/initial_topology/" + topology_id + ".npy", allow_pickle=True)
        # print("topology:", topology_id, topology)
        route_num = id.split('_')[0]
        #topology = np.load("/home/yoyo/sgan/initial_scenario/agents_4/" + "RouteScenario_" + route_num + "_to_" + route_num + "/topology_150x150/" + str(self.obs_horizon * 2 - 1) + ".npy", allow_pickle=True)
        ###############################################

        
        #ctr_line_candts = np.empty([len(topology)])
        #print(topology)
        #print(len(topology[:, :2]))yaws
        #print(len(topology[2]))
        #print(topology[2].shape)
        #for i, _ in enumerate(topology):
        #    topology[:, 2][i][:, :2] = np.matmul(rot, (topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T).T
        
        ############
        agt_traj_obs = np.matmul(rot, (agt_traj_obs - orig.reshape(-1, 2)).T).T
        ############

        ############ if yaw augment ############
        if self.augment:
            agt_traj_obs_no_rot = agt_traj_obs # for plot
            random_number = random.uniform(-self.max_degree / 2, self.max_degree / 2)
            angle_radians = np.radians(random_number)
            cos_theta = np.cos(angle_radians)
            sin_theta = np.sin(angle_radians)
            rotation_matrix_augment = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float32)
            agt_traj_obs = np.matmul(rotation_matrix_augment, (agt_traj_obs).T).T
        ############ if yaw augment ############

        # rotate the center lines and find the reference center line
        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T
        ctr_line_candts_list = []
        #print("topology:", topology)
        t_sum = 0
        for i, _ in enumerate(topology):
            #temp = np.empty([topology[:, 2][i][:, :2].shape[0], 2])
            #ctr_line_candts_list.append(temp)
            #print((topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T.shape)
            #print(np.matmul(rot, (topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T).T)
            
            t_sum += topology[:, 2][i][:, :2].shape[0]
            # print(i, topology[:, 2][i][:, :2].shape, t_sum) # (12, 2)
            t = np.matmul(rot, (topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T).T
            #print(agt_traj_obs)
            # t_sum += len(t)
            # print(t, len(t), t_sum)
            
            if len(t) <= 1:
                tmp = (np.matmul(rot, (topology[:, 2][i][:, 3:5] - orig.reshape(-1, 2)).T).T)[0]
                t = np.vstack((t[0], tmp))
            ctr_line_candts_list.append(t)

        ctr_line_candts = np.array(ctr_line_candts_list)
        #print("c:", ctr_line_candts.shape) # lanes * center_points * 2(x, y)
        #tar_candts = self.lane_candidate_sampling(ctr_line_candts, viz=False)
        tar_candts, tar_candts_with_id = self.lane_candidate_sampling(ctr_line_candts, distance=0, viz=False)

        # print(tar_candts.shape, tar_candts_with_id.shape)

        # print(topology)
        tar_candts_dist = np.sqrt(tar_candts[:, 0]**2 + tar_candts[:, 1]**2)
        max_tar_candts_dist = np.max(tar_candts_dist)
        ego_collision_dist = np.sqrt(agt_traj_fut[-1][0]**2 + agt_traj_fut[-1][1]**2)
        ego_collision_dist_x = abs(agt_traj_fut[-1][0])
        ego_collision_dist_y = abs(agt_traj_fut[-1][1])
        displacement = tar_candts - agt_traj_fut[-1]
        dist = np.sqrt(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))
        nearest_target_to_ego_collision_dist = min(dist)

        # print(np.max(tar_candts_dist), ego_collision_dist, ego_collision_dist_x, ego_collision_dist_y) # 812 points for 2m interval
        # np.sqrt(agt_traj_fut[-1][0]**2 + agt_traj_fut[-1][1]**2)
        ######## for lane id ########
        # tar_candts = tar_candts_with_id
        ######## for lane id ########
        
        #total_elements = np.sum([len(subarr) for subarr in ctr_line_candts])
        #print(total_elements, len(tar_candts)) # 188 164
        # exit()
        #plt.plot(agt_traj_obs[:, 0], agt_traj_obs[:, 1], '-', color='green')
        #plt.plot(agt_traj_fut[:, 0], agt_traj_fut[:, 1], '-', color='blue', markersize=20)
        dx = agt_traj_obs[-1, 0] - agt_traj_obs[0, 0]
        dy = agt_traj_obs[-1, 1] - agt_traj_obs[0, 1]
        #plt.arrow(agt_traj_obs[0, 0], agt_traj_obs[0, 1], dx, dy, width=2, head_width=5, color='green')
        
        # new_tar_candts no use
        new_tar_candts = []
        for i in range(tar_candts.shape[0]):
            candidate_dx = tar_candts[i, 0] - agt_traj_obs[0, 0]
            candidate_dy = tar_candts[i, 1] - agt_traj_obs[0, 1]
            angle = angle_vectors([dx, dy], [candidate_dx, candidate_dy]) * 180 / np.pi
            if angle < 90:
                #plt.scatter(tar_candts[i, 0], tar_candts[i, 1], color='red', s=1)
                new_tar_candts.append([tar_candts[i, 0], tar_candts[i, 1]])
            # else:
            #     plt.scatter(tar_candts[i, 0], tar_candts[i, 1], color='purple', s=1)
        new_tar_candts = np.array(new_tar_candts)
        #tar_candts = new_tar_candts
        # if self.split == "test":
        #     tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
        #     splines, ref_idx = None, None
        # else:
        
        multi_target_GT = False
        multi_target_GT_num = 50
        target_GT_on_distance = True
        target_GT_on_distance_scale = 4 #(m)
        
        splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)
        tar_candts_gt, tar_offse_gt, tar_offse_gt_each = self.get_candidate_gt_each_point_with_offset_GT(tar_candts, agt_traj_fut[-1], multi_target_GT, multi_target_GT_num, target_GT_on_distance, target_GT_on_distance_scale)
        # print(agt_traj_fut[-1])
        # print(tar_candts.shape, tar_candts)
        # print(tar_candts_gt.shape, tar_candts_gt)
        # print(int(np.where(tar_candts_gt==1)[0]), tar_offse_gt)
        # exit()
        #traj_converted = np.matmul(np.linalg.inv(rot), agt_traj_fut.T).T + orig.reshape(-1, 2)

        
        ### debug ###
        
        debug = False
        if debug:
            #print(data['yaws'][0][-1])
            scatter = True
            now_points = 0

            # plot past trajectory
            # plt.plot(agt_traj_obs_no_rot[:, 0], agt_traj_obs_no_rot[:, 1], '-', color='red')
            # plt.plot(agt_traj_obs[:, 0], agt_traj_obs[:, 1], '-', color='orange', alpha=0.5)
            
            # yaw = (data['yaws'][0][-1] - 55.0) * np.pi / 180
            yaw = (data['yaws'][0][-1] + 35.0) * np.pi / 180
            index_tuple = np.where(tar_candts_gt == 1)
            # print(len(index_tuple[0]), tar_offse_gt)
            preprocess_target_point= tar_candts[index_tuple[0], :]
            for i in range(tar_candts.shape[0]):
                plt.scatter(tar_candts[i][0], tar_candts[i][1], s=1, c="black")
                now_points += 1
                
                # plt.arrow(tar_candts[i][0], tar_candts[i][1], tar_offse_gt_each[i][0], tar_offse_gt_each[i][1], head_width = 0.1, color = 'cyan')

                # plt.text(tar_candts_with_id[i][0], tar_candts_with_id[i][1],
                        #   str(int(tar_candts_with_id[i][2])) + " " + str(int(tar_candts_with_id[i][3])), fontsize=8)
            # print(agt_traj_fut)
            if scatter:
               plt.scatter(agt_traj_fut[-1][0], agt_traj_fut[-1][1], s=10, c="red", marker="s")
            else:
                ego_vec = [agt_traj_fut[-1][1] - agt_traj_fut[0][1], agt_traj_fut[-1][0] - agt_traj_fut[0][0]]
                angle = np.rad2deg(angle_vectors(ego_vec, [1, 0]))
                transform_angle = (float(angle)) * np.pi / 180
                # print(angle_vectors(ego_vec, [1, 0]))
                # print(data['yaws'][0][-1] * np.pi / 180)
                # ego_rec = [agt_traj_fut[-1][0], agt_traj_fut[-1][1], 2, 4.7, yaw]
                ego_rec = [agt_traj_fut[-1][0], agt_traj_fut[-1][1], 4.7, 2, yaw]
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
                plt.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color="red", markersize=3)
            if scatter:
                for point_index in range(preprocess_target_point.shape[0]):
                    plt.scatter(preprocess_target_point[point_index][0], preprocess_target_point[point_index][1], s=20, c="cyan", marker="*")
            else:
                for point_index in range(preprocess_target_point.shape[0]):
                    # ego_rec = [preprocess_target_point[point_index][0], preprocess_target_point[point_index][1], 2, 4.7, yaw]
                    ego_rec = [preprocess_target_point[point_index][0], preprocess_target_point[point_index][1], 4.7, 2, yaw]
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
                    plt.plot([x_1, x_2, x_4, x_3, x_1], [y_1, y_2, y_4, y_3, y_1], '-',  color="blue", markersize=3)
            plt.xlim( - 37,
                     + 37)
            plt.ylim( - 37,
                     + 37)
            title = "Points:", str(now_points)
            plt.title(title)
            plt.show()

        # for i in range(len(tar_candts_gt)):
        #     if tar_candts_gt[i][0] != 0:
        #         print(i, tar_candts_gt[i]) # 3083, 1
        # sys.exit()

        #print(ctr_line_candts[0][0], agt_traj_obs[0], agt_traj_fut[0])#, tar_candts[0])
        #self.plot_target_candidates(ctr_line_candts, agt_traj_obs, agt_traj_fut, tar_candts)
        # if not np.all(offse_gt < self.LANE_WIDTH[data['city']]):
        #     self.plot_target_candidates(ctr_line_candts, agt_traj_obs, agt_traj_fut, tar_candts)

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        yaw_feats, gt_preds_yaw, has_preds_yaw = [], [], []
        attacker_traj, attacker_yaw, atr_pos_offset, atr_yaw_offset, cross = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        for traj, step, yaws, attacker_pos_offset, attacker_yaw_offset, cross_ in zip(data['trajs'], data['steps'], data['yaws'], data['attacker_pos_offset'], data['attacker_yaw_offset'], data['cross']):
            # if id.split('_')[0] != '612':
            #     continue
            
            
            if self.obs_horizon-1 not in step:
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

            rot_atr_pos_offset = np.matmul(rot, attacker_pos_offset.T).T

            #####################################################################
            max_length = 0
            for i in range(np.array(data['trajs']).shape[0]):
                if len(data['trajs'][i]) > max_length:
                    max_length = len(data['trajs'][i])
            self.pred_horizon = max_length - self.obs_horizon
            #self.pred_horizon = 3
            #self.pred_horizon = np.array(data['trajs']).shape[1] - self.obs_horizon
            #####################################################################
        
            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            has_pred = np.zeros(self.pred_horizon, np.bool)
            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            post_step = step[future_mask] - self.obs_horizon
            post_traj = traj_nd[future_mask]
            #print(post_traj.shape, post_traj) #12 2
            gt_pred[post_step] = post_traj
            has_pred[post_step] = True

            #yaw
            gt_pred_yaw = np.zeros(self.pred_horizon, np.float32)
            has_pred_yaw = np.zeros(self.pred_horizon, np.bool)
            post_yaw = yaws[future_mask]
            gt_pred_yaw[post_step] = post_yaw
            has_pred_yaw[post_step] = True

            # if id.split('_')[0] == '612':
            #     print("yaw:", yaws)
            #     print("mask:", post_yaw)
            #     print("gt_pred_yaw:", gt_pred_yaw)

            # colect the observation
            obs_mask = step < self.obs_horizon
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]

            # print("traj_obs:", traj_obs)
            # print("gt_pred:", gt_pred)
            # print(tar_candts_gt)
            # sys.exit()

            #yaw
            yaw_obs = yaws[obs_mask]
            yaw_obs = yaw_obs[idcs]

            #print(len(post_traj), post_traj) 30
            #print(len(traj_obs), traj_obs) 10

            # print(len(post_yaw), post_yaw)
            # print(len(yaw_obs), yaw_obs)

            for i in range(len(step_obs)):
                if step_obs[i] == self.obs_horizon - len(step_obs) + i:
                    break
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:]

            yaw_obs = yaw_obs[i:]

            if len(step_obs) <= 1:
                continue

            feat = np.zeros((self.obs_horizon, 3), np.float32)
            has_obs = np.zeros(self.obs_horizon, np.bool)

            yaw_feat = np.zeros(self.obs_horizon, np.float32)
            yaw_feat[step_obs] = yaw_obs

            #print(traj_obs)
            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            
            feats.append(feat)                  # displacement vectors
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

            yaw_feats.append(yaw_feat)                  # displacement vectors
            gt_preds_yaw.append(gt_pred_yaw)
            has_preds_yaw.append(has_pred_yaw)

            atr_pos_offset.append(rot_atr_pos_offset)
            atr_yaw_offset.append(attacker_yaw_offset)
            cross.append(cross_)
        # if len(feats) < 1:
        #     raise Exception()
        feats = np.asarray(feats, np.float32)
        has_obss = np.asarray(has_obss, np.bool)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        yaw_feats = np.asarray(yaw_feats, np.float32)
        gt_preds_yaw = np.asarray(gt_preds_yaw, np.float32)
        has_preds_yaw = np.asarray(has_preds_yaw, np.bool)

        atr_pos_offset = np.asarray(atr_pos_offset, np.float32)
        atr_yaw_offset = np.asarray(atr_yaw_offset, np.float32)
        cross = np.asarray(cross, np.float32)
        #print(gt_preds_yaw.shape, atr_pos_offset.shape, atr_yaw_offset.shape) #(1, 24) (1, 2) (1,)

        # plot the splines
        # self.plot_reference_centerlines(ctr_line_candts, splines, feats[0], gt_preds[0], ref_idx)

        # # target candidate filtering
        # tar_candts = np.matmul(rot, (tar_candts - orig.reshape(-1, 2)).T).T
        # inlier = np.logical_and(np.fabs(tar_candts[:, 0]) <= self.obs_range, np.fabs(tar_candts[:, 1]) <= self.obs_range)
        # if not np.any(candts_gt[inlier]):
        #     raise Exception("The gt of target candidate exceeds the observation range!")

        data['yaw_feats'] = yaw_feats
        data['gt_preds_yaw'] = gt_preds_yaw
        data['has_preds_yaw'] = has_preds_yaw
        
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt
        data['gt_tar_offset_each'] = tar_offse_gt_each

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        
        data['atr_pos_offset'] = atr_pos_offset
        data['atr_yaw_offset'] = atr_yaw_offset
        data['cross'] = cross

        data['max_tar_candts_dist'] = max_tar_candts_dist
        data['ego_collision_dist'] = ego_collision_dist
        data['ego_collision_dist_x'] = ego_collision_dist_x
        data['ego_collision_dist_y'] = ego_collision_dist_y
        data['nearest_target_to_ego_collision_dist'] = nearest_target_to_ego_collision_dist
        data['tar_candts_with_id'] = tar_candts_with_id

        return data

    def get_lane_graph(self, data, id):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        #radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        #lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius * 1.5)
        #lane_ids = copy.deepcopy(lane_ids)

        #topology = np.load("/home/carla/experiments/TNT-Trajectory-Predition/carla_dataset/topology_data/" + id + ".npy", allow_pickle=True)
        ###############################################
        # 4.2: topology_id = id.split('_')[3] + '_' + id.split('_')[4] + '_' + id.split('_')[5] + '_' + id.split('_')[6]
        topology_id = id.split('_')[3] + '_' + id.split('_')[4] + '_' + id.split('_')[6]
        topology = np.load("/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data/initial_topology/" + topology_id + ".npy", allow_pickle=True)
        route_num = id.split('_')[0]
        #topology = np.load("/home/yoyo/sgan/initial_scenario/agents_4/" + "RouteScenario_" + route_num + "_to_" + route_num + "/topology/7.npy", allow_pickle=True)
        #topology = np.load("/home/yoyo/sgan/initial_scenario/agents_4/" + "RouteScenario_" + route_num + "_to_" + route_num + "/topology_150x150/" + str(self.obs_horizon * 2 - 1) + ".npy", allow_pickle=True)
        ###############################################

        #print("topo:", topology.shape)
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for i, _ in enumerate(topology):
            ctrln = np.matmul(data['rot'], (topology[:, 2][i][:, :2] - data['orig'].reshape(-1, 2)).T).T
            if len(ctrln) < 2:
                tmp = (np.matmul(data['rot'], (topology[:, 2][i][:, 3:5] - data['orig'].reshape(-1, 2)).T).T)[0]
                ctrln = np.vstack((ctrln[0], tmp))
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))
            x = np.zeros((num_segs, 2), np.float32)
            if topology[:, 3][i] == 'left':
                x[:, 0] = 1
            elif topology[:, 3][i] == 'right':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(topology[:, 4][i] * np.ones(num_segs, np.float32))
            intersect.append(topology[:, 5][i] * np.ones(num_segs, np.float32))


        '''
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))
        '''

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count
        lane_idcs = np.concatenate(lane_idcs, 0)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['lane_idcs'] = lane_idcs

        # print("ctrs:", graph['ctrs'].shape)
        # print("num_nodes:", graph['num_nodes'])
        # print("feats:", graph['feats'].shape)
        # print("turn:", graph['turn'].shape)
        # print("control:", graph['control'].shape)
        # print("intersect:", graph['intersect'].shape)
        # print("lane_idcs:", len(lane_idcs))
        # exit()

        return graph


    @staticmethod
    def get_ref_centerline(cline_list, pred_gt):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx


    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "AGENT" if traj_id == 0 else "OTHERS"

        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], "d-", color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)

        plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))

        if len(pred) == 0:
            plt.text(obs[-1, 0], obs[-1, 1], "{}_e".format(traj_na))
        else:
            plt.text(pred[-1, 0], pred[-1, 1], "{}_e".format(traj_na))


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument("-r", "--root", type=str, default="/home/carla/experiments/TNT-Trajectory-Predition/carla_dataset")
    #parser.add_argument("-d", "--dest", type=str, default="/home/carla/experiments/TNT-Trajectory-Predition/carla_dataset")
    parser.add_argument("-r", "--root", type=str, default="/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data")
    parser.add_argument("-d", "--dest", type=str, default="/home/yoyo/Documents/TNT_Nuscenes/nuscenes_data")

    parser.add_argument("-s", "--small", action='store_true', default=False)
    args = parser.parse_args()


    #raw_dir = os.path.join(args.root, "raw_trajectory_data")
    raw_dir = os.path.join(args.root, "trajectory")
    interm_dir = os.path.join(args.dest, "interm_data" if not args.small else "interm_data_small")


    
    # for split in ["train", "val", "test"]:
    for split in ["test"]:
        # construct the preprocessor and dataloader
        argoverse_processor = ArgoversePreprocessor(root_dir=raw_dir, split=split, save_dir=interm_dir)
        print(split)
        loader = DataLoader(argoverse_processor,
                            batch_size=512,
                            num_workers=0,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
        
        for i, data in enumerate(tqdm(loader)):
            #break
            if args.small:
                if split == "train" and i >= 200:
                    break
                elif split == "val" and i >= 50:
                    break
                elif split == "test" and i >= 50:
                    break
        
