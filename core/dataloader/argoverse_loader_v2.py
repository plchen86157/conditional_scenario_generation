import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import gc
from copy import deepcopy, copy

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
# from torch.utils.data import DataLoader

sys.path.append("core/dataloader")

class_list = ["junction_crossing", "LTAP", "lane_change", "opposite_direction", "rear_end"]
class_num = len(class_list)

def get_fc_edge_index(node_indices):
    """
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index
    """
    xx, yy = np.meshgrid(node_indices, node_indices)
    xy = np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)
    return xy


def get_traj_edge_index(node_indices):
    """
    generate the polyline graph for traj, each node are only directionally connected with the nodes in its future
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index
    """
    edge_index = np.empty((2, 0))
    for i in range(len(node_indices)):
        xx, yy = np.meshgrid(node_indices[i], node_indices[i:])
        edge_index = np.hstack([edge_index, np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)])
    return edge_index


class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0

# %%


# dataset loader which loads data into memory
class ArgoverseInMem(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ArgoverseInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        gc.collect()

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith(".pkl")]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """ transform the raw data and store in GraphData """
        # loading the raw data
        traj_lens = []
        valid_lens = []
        candidate_lens = []
        tp_candidate_lens = []
        tp_all = 0
        all_scene = 0

        max_tar_candts_dist_list = []
        ego_collision_dist_list = []
        nearest_target_to_ego_collision_dist_list = []
        check_ego_in_farest_target_range_sum = 0

        for raw_path in tqdm(self.raw_paths, desc="Loading Raw Data..."):
            raw_data = pd.read_pickle(raw_path)

            #print("path:", raw_path.split('/')[-1])
            # print("data:", raw_data)

            # statistics
            traj_num = raw_data['feats'].values[0].shape[0]
            traj_lens.append(traj_num)

            lane_num = raw_data['graph'].values[0]['lane_idcs'].max() + 1
            valid_lens.append(traj_num + lane_num)
            #print(raw_data['tar_candts'].values[0])
            
            candidate_num = raw_data['tar_candts'].values[0].shape[0]
            candidate_lens.append(candidate_num)
            ####################################################
            tp_candidate_lens.append(raw_data['other_vehicle_num'])
            ####################################################
            all_scene += 1
            tp_all += int(raw_data['other_vehicle_num'])

            max_tar_candts_dist_list.append(raw_data['max_tar_candts_dist'].values[0])
            ego_collision_dist_list.append(raw_data['ego_collision_dist'].values[0])
            nearest_target_to_ego_collision_dist_list.append(raw_data['nearest_target_to_ego_collision_dist'].values[0])
            if raw_data['max_tar_candts_dist'].values[0] > raw_data['ego_collision_dist'].values[0]:
                check_ego_in_farest_target_range_sum += 1

        # print(max(max_tar_candts_dist_list)) #265.3163932552843
        # print(max(ego_collision_dist_list)) #71.69916403243349
        # print("all scenes:", all_scene, "check_ego_in_farest_target_range_sum:", check_ego_in_farest_target_range_sum)
        # exit()

        figure = False
        if figure:
            import matplotlib.pyplot as plt
            bins = np.linspace(min(max_tar_candts_dist_list), max(max_tar_candts_dist_list), num=50)
            hist, bins = np.histogram(max_tar_candts_dist_list, bins=bins, density=False)
            plt.bar(bins[:-1], hist, align='center', width=0.8)
            plt.xlabel('Max tar candts dist')
            plt.ylabel('Freq')
            plt.title('Max tar candts dist distribution')
            plt.show()

            print("Max:", max(nearest_target_to_ego_collision_dist_list))
            bins = np.linspace(min(nearest_target_to_ego_collision_dist_list), 4, num=10)
            hist, bins = np.histogram(nearest_target_to_ego_collision_dist_list, bins=bins, density=False)
            plt.bar(bins[:-1], hist, align='center', width=0.8)
            plt.xlabel('Nearest target to ego collision dist')
            plt.ylabel('Freq')
            plt.title('Nearest target to ego collision distance distribution')
            plt.show()

        #print(tp_all/all_scene) # 19
        num_valid_len_max = np.max(valid_lens)
        #print(valid_lens, candidate_lens)
        num_candidate_max = np.max(candidate_lens)
        num_tp_candidate_max = np.max(tp_candidate_lens) ###########71
        
        # print("\n[Argoverse]: The maximum of valid length is {}.".format(num_valid_len_max))
        # print("[Argoverse]: The maximum of no. of candidates is {}.".format(num_candidate_max))

        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        data_list = []
        tar_candidate_lens = []
        for ind, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData...")):
            #print(ind, raw_path)
            raw_data = pd.read_pickle(raw_path)
            #print(num_tp_candidate_max, len(raw_data['tp_gt'].values[0]), raw_data['tp_gt'].values[0], raw_data['seq_id'][0].split('_')[-1].split('.')[0])
            # input data
            #print("raw_data:",raw_data['atr_pos_offset'], raw_data['atr_yaw_offset'])
            #print("raw_data:", raw_data['seq_id'], raw_data['feats'].values[0].shape, raw_data['feats'].values[0]) 1, 8, 3
            #print(raw_data['rot'].shape, torch.from_numpy(raw_data['rot'].values[0]).float().unsqueeze(0).shape) #(1,) -> (1,2,2)
            #print(raw_data['atr_yaw_offset'].shape, torch.from_numpy(raw_data['atr_yaw_offset'].values[0]).float().shape)
            x, cluster, edge_index, identifier = self._get_x(raw_data)
            # print("edge_index:", edge_index)
            # exit()
            y = self._get_y(raw_data)
            y_yaw = self._get_y_yaw(raw_data)
            obs_yaw = self._get_obs_yaw(raw_data)
            vector_for_ttc = self._get_vector_for_ttc(raw_data)
            obs_traj_for_ttc = self._get_obs_traj_for_ttc(raw_data)
            one_hot_class = self._get_one_hot_vector(raw_data['seq_id'])
            tar_candidate_lens.append(raw_data['tar_candts'].values[0].shape[0])
            future_traj = self._get_future_traj(raw_data)

            # print(raw_data['tar_candts'].values[0].shape, raw_data['tar_candts_with_id'].values[0].shape)


            #print("one_hot_class:", one_hot_class)
            # print(raw_data['tar_candts'].shape, raw_data['tar_candts'].values[0])
            # print(torch.from_numpy(raw_data['tar_candts'].values[0]).float().shape, torch.from_numpy(raw_data['tar_candts'].values[0]).float())
            # print(raw_data['relative_pos'].shape, raw_data['relative_pos'].values[0])
            # print(torch.from_numpy(np.array(raw_data['relative_pos'].values[0])).float().squeeze(0).shape)
            graph_input = GraphData(
                x=torch.from_numpy(x).float(),
                y=torch.from_numpy(y).float(),

                y_yaw=torch.from_numpy(y_yaw).float(),
                obs_yaw=torch.from_numpy(obs_yaw).float(),
                y_ttc=torch.from_numpy(vector_for_ttc).float(),
                obs_traj=torch.from_numpy(obs_traj_for_ttc).float(),
                future_traj=torch.from_numpy(future_traj).float(),

                cluster=torch.from_numpy(cluster).short(),
                edge_index=torch.from_numpy(edge_index).long(),
                identifier=torch.from_numpy(identifier).float(),    # the identify embedding of global graph completion

                traj_len=torch.tensor([traj_lens[ind]]).int(),            # number of traj polyline
                valid_len=torch.tensor([valid_lens[ind]]).int(),          # number of valid polyline
                time_step_len=torch.tensor([num_valid_len_max]).int(),    # the maximum of no. of polyline

                candidate_len_max=torch.tensor([num_candidate_max]).int(),
                candidate_mask=[],
                candidate=torch.from_numpy(raw_data['tar_candts'].values[0]).float(),
                candidate_with_id=torch.from_numpy(raw_data['tar_candts_with_id'].values[0]).float(),
                candidate_gt=torch.from_numpy(raw_data['gt_candts'].values[0]).bool(),
                offset_gt=torch.from_numpy(raw_data['gt_tar_offset'].values[0]).float(),
                target_gt=torch.from_numpy(raw_data['gt_preds'].values[0][0][-1, :]).float(),
                offset_gt_each=torch.from_numpy(raw_data['gt_tar_offset_each'].values[0]).float(),
                candidate_lens=raw_data['tar_candts'].values[0].shape[0],
                

                orig=torch.from_numpy(raw_data['orig'].values[0]).float().unsqueeze(0),
                rot=torch.from_numpy(raw_data['rot'].values[0]).float().unsqueeze(0),
                #seq_id=torch.tensor([int(raw_data['seq_id'])]).int()
                seq_id=raw_data['seq_id'][0],
                ##################################
                #horizon=y_yaw.shape[0],
                ##################################
                horizon=float(raw_data['seq_id'][0].split('_')[7].split('-')[-1]),
                ttc=float(raw_data['seq_id'][0].split('_')[7].split('-')[-2]),
                one_hot=one_hot_class,
                attacker_id=raw_data['seq_id'][0].split('_')[-1].split('.')[0],
                atr_pos_offset=torch.from_numpy(raw_data['atr_pos_offset'].values[0]).float(),
                atr_yaw_offset=torch.from_numpy(raw_data['atr_yaw_offset'].values[0]).float(),
                tp_candidate_len_max=torch.tensor([num_tp_candidate_max]).int(),
                tp_candidate_mask=[],
                tp_candidate=torch.from_numpy(np.array(raw_data['relative_pos'].values[0])).float().squeeze(0),
                tp_gt=torch.from_numpy(np.array(raw_data['tp_gt'].values[0])).bool(),
                cross=torch.from_numpy(raw_data['cross'].values[0]).float(), 
                max_tar_candts_dist=torch.tensor(raw_data['max_tar_candts_dist'].values[0]).float(),
            )
            
            data_list.append(graph_input)

        # print("data_list:", data_list[0])

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(ArgoverseInMem, self).get(idx).clone()
        #print(data.candidate.shape, data.candidate_len_max[0].item())


        feature_len = data.x.shape[1]
        index_to_pad = data.time_step_len[0].item()
        valid_len = data.valid_len[0].item()

        # pad feature with zero nodes
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len, index_to_pad)])
        data.identifier = torch.cat([data.identifier, torch.zeros((index_to_pad - valid_len, 2), dtype=data.x.dtype)])

        # pad candidate and candidate_gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate, torch.zeros((num_cand_max - len(data.candidate), 2))])
        data.candidate_gt = torch.cat([data.candidate_gt, torch.zeros((num_cand_max - len(data.candidate_gt), 1))])

        # pad candidate_with_id
        data.candidate_with_id = torch.cat([data.candidate_with_id, torch.zeros((num_cand_max - len(data.candidate_with_id), 4))])
        
        #########################################
        tp_num_cand_max = data.tp_candidate_len_max[0].item()
        data.tp_candidate_mask = torch.cat([torch.ones((len(data.tp_candidate), 1)),
                                         torch.zeros((tp_num_cand_max - len(data.tp_candidate), 1))])
        data.tp_candidate = torch.cat([data.tp_candidate, torch.zeros((tp_num_cand_max - len(data.tp_candidate), 2))])
        data.tp_gt = torch.cat([data.tp_gt, torch.zeros((tp_num_cand_max - len(data.tp_gt), 1))])
        #########################################
        return data

    @staticmethod
    def _get_x(data_seq):
        """
        feat: [xs, ys, vec_x, vec_y, step(timestamp), traffic_control, turn, is_intersection, polyline_id];
        xs, ys: the control point of the vector, for trajectory, it's start point, for lane segment, it's the center point;
        vec_x, vec_y: the length of the vector in x, y coordinates;
        step: indicating the step of the trajectory, for the lane node, it's always 0;
        traffic_control: feature for lanes
        turn: twon binary indicator representing is the lane turning left or right;
        is_intersection: indicating whether the lane segment is in intersection;
        polyline_id: the polyline id of this node belonging to;
        """
        #feats = np.empty((0, 10))
        feats = np.empty((0, 11))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))

        yaw_feats = data_seq['yaw_feats'].values[0]

        # get traj features
        traj_feats = data_seq['feats'].values[0]
        traj_has_obss = data_seq['has_obss'].values[0]
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        
        traj_cnt = 0
        #for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
        for _, [feat, has_obs, yaw_feat] in enumerate(zip(traj_feats, traj_has_obss, yaw_feats)):
            # print(feat)
            xy_s = feat[has_obs][:-1, :2]
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2]
            #print(len(feat), len(xy_s), len(vec)) # 10 9 9
            yaw_s = yaw_feat[has_obs][:-1].reshape((-1,1)) * np.pi / 180
            
            traffic_ctrl = np.zeros((len(xy_s), 1))
            is_intersect = np.zeros((len(xy_s), 1))
            is_turn = np.zeros((len(xy_s), 2))
            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt
            # print(feats.shape, xy_s.shape, yaw_s.shape, step[has_obs][:-1].shape, traffic_ctrl.shape, is_turn.shape, is_intersect.shape, polyline_id.shape)
            # print(xy_s, vec, yaw_s)
            feats = np.vstack([feats, np.hstack([xy_s, vec, yaw_s, step[has_obs][:-1], traffic_ctrl, is_turn, is_intersect, polyline_id])])
            # print(feats.shape)
            # print(feats)
        
            traj_cnt += 1
        # exit()
        # get lane features
        graph = data_seq['graph'].values[0]
        ctrs = graph['ctrs']
        vec = graph['feats']
        traffic_ctrl = graph['control'].reshape(-1, 1)
        is_turns = graph['turn']
        is_intersect = graph['intersect'].reshape(-1, 1)
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt
        steps = np.zeros((len(lane_idcs), 1))
        # print(traj_cnt, lane_idcs)
        # exit()
        yaw_lane = np.zeros((len(lane_idcs), 1))
        # print(ctrs.shape, vec.shape, yaw_lane.shape, steps.shape)
        # print(is_turn)
        feats = np.vstack([feats, np.hstack([ctrs, vec, yaw_lane, steps, traffic_ctrl, is_turns, is_intersect, lane_idcs])])
        # print(feats.shape)
        # exit()
        # get the cluster and construct subgraph edge_index
        cluster = copy(feats[:, -1].astype(np.int64))
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])
            if len(indices) <= 1:
                continue                # skip if only 1 node
            if cluster_idc < traj_cnt:
                edge_index = np.hstack([edge_index, get_traj_edge_index(indices)])
            else:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
        #print(feats.shape, feats)


        return feats, cluster, edge_index, identifier

    @staticmethod
    def _get_y(data_seq):
        traj_obs = data_seq['feats'].values[0][0]
        traj_fut = data_seq['gt_preds'].values[0][0]
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        # print(traj_obs.astype(np.int), traj_fut.astype(np.int))
        # print(offset_fut.astype(np.int))
        # print(traj_fut[0, :] - traj_obs[-1, :2])
        #print(data_seq['feats'].values.shape)
        # exit()
        return offset_fut.reshape(-1).astype(np.float32)
    
    @staticmethod
    def _get_vector_for_ttc(data_seq):
        traj_obs = data_seq['feats'].values[0][0]
        traj_fut = data_seq['gt_preds'].values[0][0]
        offset_fut = traj_fut[-1, :] - traj_obs[:, :2]
        offset_ = np.sqrt(offset_fut[:, 0] ** 2 + offset_fut[:, 1] ** 2)
        return offset_.astype(np.float32)
    
    @staticmethod
    def _get_obs_traj_for_ttc(data_seq):
        traj_obs = data_seq['feats'].values[0][0]
        #print(torch.flatten(traj_obs))
        #print(traj_obs.flatten(0).astype(np.float32))
        return traj_obs[:, :2].flatten().astype(np.float32)
        #return np.concatenate(traj_obs[:, 0], traj_obs[:, 1]).astype(np.float32)
    
    @staticmethod
    def _get_y_yaw(data_seq):
        yaw_fut = data_seq['gt_preds_yaw'].values[0][0]
        return yaw_fut
    
    @staticmethod
    def _get_obs_yaw(data_seq):
        yaw_fut = data_seq['yaw_feats'].values[0][0]
        return yaw_fut
    
    @staticmethod
    def _get_future_traj(data_seq):
        future_traj = data_seq['gt_preds'].values[0][0]
        return future_traj
    
    @staticmethod
    def _get_one_hot_vector(seq_id):
        scenario_type = seq_id[0].split('_')[5]
        #print("scenario_type:", scenario_type)
        if scenario_type == 'JC':
            return 0
        elif scenario_type == 'LTAP': 
            return 1
        elif scenario_type == 'LC': 
            return 2
        elif scenario_type == 'HO': 
            return 3
        elif scenario_type == 'RE': 
            return 4



if __name__ == "__main__":

    # for folder in os.listdir("./data/interm_data"):
    INTERMEDIATE_DATA_DIR = "/home/carla/experiments/TNT-Trajectory-Predition/carla_all_data/interm_data"
    # INTERMEDIATE_DATA_DIR = "../../dataset/interm_tnt_n_s_0804"
    # INTERMEDIATE_DATA_DIR = "/media/Data/autonomous_driving/Argoverse/intermediate"

    for folder in ["train", "val", "test"]:
    # for folder in ["test"]:
        dataset_input_path = os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")

        # dataset = Argoverse(dataset_input_path)
        dataset = ArgoverseInMem(dataset_input_path).shuffle()
        batch_iter = DataLoader(dataset, batch_size=16, num_workers=16, shuffle=True, pin_memory=True)
        for k in range(1):
            for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
                pass

            # print("{}".format(i))
            # candit_len = data.candidate_len_max[0]
            # print(candit_len)
            # target_candite = data.candidate[candit_gt.squeeze(0).bool()]
            # try:
            #     # loss = torch.nn.functional.binary_cross_entropy(candit_gt, candit_gt)
            #     target_candite = data.candidate[candit_gt.bool()]
            # except:
            #     print(torch.argmax())
            #     print(candit_gt)
            # # print("type: {}".format(type(candit_gt)))
            # print("max: {}".format(candit_gt.max()))
            # print("min: {}".format(candit_gt.min()))

