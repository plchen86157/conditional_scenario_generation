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

# import torch
from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline

from core.util.preprocessor.base import Preprocessor
from core.util.cubic_spline import Spline2D

warnings.filterwarnings("ignore")


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

        

        #self.am = ArgoverseMap()
        self.loader = ArgoverseForecastingLoader(pjoin(self.root_dir, self.split+"_obs" if split == "test" else split))

        print(self.loader)

        self.save_dir = save_dir

    def __getitem__(self, idx):
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        path, seq_f_name_ext = os.path.split(f_path)
        seq_f_name, ext = os.path.splitext(seq_f_name_ext)

        df = copy.deepcopy(seq.seq_df)
        #print("df:", df)
        return self.process_and_save(df, seq_id=seq_f_name, dir_=self.save_dir)

    def process(self, dataframe: pd.DataFrame,  seq_id, map_feat=True):
        #if seq_id != '5-s-7_TCD_violation_1037_999':
        #    continue
        data = self.read_argo_data(dataframe, seq_id, self.obs_horizon)
        data = self.get_obj_feats(data, seq_id)

        data['graph'] = self.get_lane_graph(data, seq_id)
        data['seq_id'] = seq_id
        # visualization for debug purpose
        # self.visualize_data(data)
        return pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def read_argo_data(df: pd.DataFrame, seq_id, obs_horizon):
        ###############################
        attacker_id = int(seq_id.split('_')[-1])
        ###############################

        city = df["CITY_NAME"].values[0]

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

        

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]
        #print(keys, idcs, objs)

        agt_traj = trajs[idcs]
        agt_step = steps[idcs]
        ###############################
        agt_yaw = yaws[idcs]

        attacker_type = [x[0] for x in keys]
        attacker_idx = attacker_type.index(attacker_id)
        attacker_idcs = objs[keys[attacker_idx]]
        attacker_traj = trajs[attacker_idcs]
        attacker_yaw = yaws[attacker_idcs]
        attacker_pos_offset = attacker_traj[-1] - agt_traj[-1]
        attacker_yaw_offset = attacker_yaw[-1] - agt_yaw[-1]
        #print(agt_traj, agt_yaw)


        ###############################
        ## ctx_trajs: other's all traj
        del keys[agt_idx]
        ctx_trajs, ctx_steps, ctx_yaws, relative_pos, tp_gt_list = [], [], [], [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])
            ################################
            ctx_yaws.append(yaws[idcs])
            relative_pos_now = trajs[idcs][obs_horizon - 1] - agt_traj[obs_horizon - 1]
            relative_pos.append(relative_pos_now)
            tp_gt_list.append([1] if key[0] == attacker_idx else [0])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps

        data['attacker_traj'] = [attacker_traj]
        data['attacker_yaw'] = [attacker_yaw]
        data['attacker_pos_offset'] = [attacker_pos_offset]
        data['attacker_yaw_offset'] = [attacker_yaw_offset]
        data['yaws'] = [agt_yaw] + ctx_yaws
        data['other_vehicle_num'] = len(ctx_trajs)
        data['attacker_id'] = attacker_id
        data['relative_pos'] = [relative_pos]
        data['tp_gt'] = tp_gt_list
        # print(len(ctx_trajs), len(ctx_trajs[0]))# 4 20
        # sys.exit()
        return data

    def get_obj_feats(self, data, id):
        # get the origin and compute the oritentation of the target agent
        print(id, np.array(data['trajs']).shape)
        #print("data['trajs'][0]:",data, id, np.array(data['trajs']).shape, data['trajs'][0])
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float32) # [0] means "EGO"
        #print("obs:", data['trajs'][0][self.obs_horizon-1])

        #print("orig:", orig) # the position of Agent after 2 s (70.4, -11.37)

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
        agt_traj_fut = data['trajs'][0][self.obs_horizon:-1].copy().astype(np.float32)
        ###############################################
        #ctr_line_candts = self.am.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'])

        #topology = np.load("/home/carla/experiments/TNT-Trajectory-Predition/carla_dataset/topology_data/" + id + ".npy", allow_pickle=True)
        
        ###############################################
        topology = np.load("/home/yoyo/Documents/TNT_input_ttc/carla_all_data/topology_150x150/" + id + ".npy", allow_pickle=True)
        route_num = id.split('_')[0]
        #topology = np.load("/home/yoyo/sgan/initial_scenario/agents_4/" + "RouteScenario_" + route_num + "_to_" + route_num + "/topology_150x150/" + str(self.obs_horizon * 2 - 1) + ".npy", allow_pickle=True)
        ###############################################

        
        #ctr_line_candts = np.empty([len(topology)])
        #print(topology)
        #print(len(topology[:, :2]))
        #print(len(topology[2]))
        #print(topology[2].shape)
        #for i, _ in enumerate(topology):
        #    topology[:, 2][i][:, :2] = np.matmul(rot, (topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T).T
        
        ############
        agt_traj_obs = np.matmul(rot, (agt_traj_obs - orig.reshape(-1, 2)).T).T
        ############
        # rotate the center lines and find the reference center line
        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T
        ctr_line_candts_list = []
        #print("topology:", topology)
        for i, _ in enumerate(topology):
            #temp = np.empty([topology[:, 2][i][:, :2].shape[0], 2])
            #ctr_line_candts_list.append(temp)
            #print(topology[:, 2][i][:, :2])
            #print((topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T.shape)
            #print(np.matmul(rot, (topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T).T)
            #print(topology[:, 2][i][:, :2].shape) # (12, 2)
            t = np.matmul(rot, (topology[:, 2][i][:, :2] - orig.reshape(-1, 2)).T).T
            
            if len(t) <= 1:
                tmp = (np.matmul(rot, (topology[:, 2][i][:, 3:5] - orig.reshape(-1, 2)).T).T)[0]
                t = np.vstack((t[0], tmp))
            ctr_line_candts_list.append(t)

        ctr_line_candts = np.array(ctr_line_candts_list)
        #print("c:", ctr_line_candts.shape) # lanes * center_points * 2(x, y)
        tar_candts = self.lane_candidate_sampling(ctr_line_candts, viz=False)
        if self.split == "test":
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])
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
        attacker_traj, attacker_yaw, atr_pos_offset, atr_yaw_offset = [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        for traj, step, yaws, attacker_pos_offset, attacker_yaw_offset in zip(data['trajs'], data['steps'], data['yaws'], data['attacker_pos_offset'], data['attacker_yaw_offset']):
            # if id.split('_')[0] != '612':
            #     continue
            
            
            if self.obs_horizon-1 not in step:
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

            rot_atr_pos_offset = np.matmul(rot, attacker_pos_offset.T).T

            #####################################################################
            self.pred_horizon = np.array(data['trajs']).shape[1] - self.obs_horizon
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
        #print(gt_preds_yaw.shape, atr_pos_offset.shape, atr_yaw_offset.shape) (1, 24) (1, 2) (1,)

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

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        
        data['atr_pos_offset'] = atr_pos_offset
        data['atr_yaw_offset'] = atr_yaw_offset
        return data

    def get_lane_graph(self, data, id):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        #radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        #lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius * 1.5)
        #lane_ids = copy.deepcopy(lane_ids)

        #topology = np.load("/home/carla/experiments/TNT-Trajectory-Predition/carla_dataset/topology_data/" + id + ".npy", allow_pickle=True)
        ###############################################
        topology = np.load("/home/yoyo/Documents/TNT_input_ttc/carla_all_data/topology_150x150/" + id + ".npy", allow_pickle=True)
        route_num = id.split('_')[0]
        #topology = np.load("/home/yoyo/sgan/initial_scenario/agents_4/" + "RouteScenario_" + route_num + "_to_" + route_num + "/topology/7.npy", allow_pickle=True)
        #topology = np.load("/home/yoyo/sgan/initial_scenario/agents_4/" + "RouteScenario_" + route_num + "_to_" + route_num + "/topology_150x150/" + str(self.obs_horizon * 2 - 1) + ".npy", allow_pickle=True)
        ###############################################

        #print(topology)
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

        return graph

    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_ctrs = data['graph']['ctrs']
        lines_feats = data['graph']['feats']
        lane_idcs = data['graph']['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss']
        preds = data['gt_preds']
        has_preds = data['has_preds']
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

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

    def plot_reference_centerlines(self, cline_list, splines, obs, pred, ref_line_idx):
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        for centerline_coords in cline_list:
            visualize_centerline(centerline_coords)

        for i, spline in enumerate(splines):
            xy = np.stack([spline.x_fine, spline.y_fine], axis=1)
            if i == ref_line_idx:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="r", alpha=0.7, linewidth=1, zorder=10)
            else:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="b", alpha=0.5, linewidth=1, zorder=10)

        self.plot_traj(obs, pred)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

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
    parser.add_argument("-r", "--root", type=str, default="/home/yoyo/Documents/TNT_input_ttc/carla_all_data")
    parser.add_argument("-d", "--dest", type=str, default="/home/yoyo/Documents/TNT_input_ttc/carla_all_data")

    parser.add_argument("-s", "--small", action='store_true', default=False)
    args = parser.parse_args()


    #raw_dir = os.path.join(args.root, "raw_trajectory_data")
    raw_dir = os.path.join(args.root, "trajectory")
    interm_dir = os.path.join(args.dest, "interm_data" if not args.small else "interm_data_small")


    for split in ["train", "val", "test"]:
        # construct the preprocessor and dataloader
        argoverse_processor = ArgoversePreprocessor(root_dir=raw_dir, split=split, save_dir=interm_dir)
        print(split)
        loader = DataLoader(argoverse_processor,
                            batch_size=16,
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
        
