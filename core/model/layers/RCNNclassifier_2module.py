# trajectory prediction layer of TNT algorithm with Binary CE Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys
from core.model.layers.utils import masked_softmax
from core.model.layers.basic_module import MLP
import numpy as np
from torch.autograd import Variable
from shapely.geometry.polygon import Polygon
    
class RCNNClassifier_2module(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 m: int = 5,
                 device=torch.device("cpu"),
                 nearest_subgraph=1):
        """"""
        super(RCNNClassifier_2module, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        #print("TargetPred hidden:", self.hidden_dim, "in_channels:", self.in_channels)
        self.M = m          # output candidate target

        self.device = device
        #################
        #################
        #################
        self.num_classes = 2
        # RCNN_xy, PointNet, local_feature
        self.mode = "local_feature" #"local_feature"
        self.nearest_subgraph = nearest_subgraph
        #################
        #################
        #################
        
        #################
        # self.classifier = nn.Linear(hidden_dim + 1, num_classes)
        # self.classifier = nn.Linear(hidden_dim + 2, num_classes)
        # self.classifier = nn.Linear(hidden_dim + 4, num_classes)
        
        #################
        self.dimension_raising = nn.Sequential(
            # nn.ReLU(),
            MLP(2, hidden_dim, hidden_dim),
            #nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            # nn.ReLU(),
            MLP(in_channels * 2, hidden_dim, hidden_dim),
            #nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, self.num_classes)
        )

        # self.classifier = nn.Sequential(
        #     nn.ReLU(),
        #     MLP(in_channels + 2, hidden_dim, hidden_dim),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(hidden_dim, num_classes)
        # )

        ############# PointNet #############
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = False
        self.feature_transform = False
        # if self.feature_transform:
            # self.fstn = STNkd(k=64)
        self.down_dimension = torch.nn.Conv1d(1024, 64, 1)
        # self.down_dimension_to_local_fea = torch.nn.Conv1d(1024, nearest_subgraph * in_channels, 1)
        ############# PointNet #############

        self.ego_final_feature = nn.Linear(self.hidden_dim, nearest_subgraph * in_channels)
        self.local_fea_classifier = nn.Sequential(
                # nn.ReLU(),
                MLP(nearest_subgraph * in_channels * 3, self.hidden_dim, self.hidden_dim),
                #nn.Dropout(p=0.1),
                nn.Linear(self.hidden_dim, self.num_classes))
        self.dimension_reduction_from_1024 = nn.Sequential(
            # nn.ReLU(),
            MLP(1024, hidden_dim, hidden_dim),
            #nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, nearest_subgraph * in_channels)
        )
        self.loss_weight_by_dist = True


    def forward(self, proposal_feat, target_candidate, data, RCNN_cls_result, offset, yaw_pred):
        batch_size, n, _ = target_candidate.size()

        iou_threshold = 0.001
        vehicle_length = 4.7
        vehicle_width = 2
        candidate_pos = data.candidate.view(-1, n, 2)
        RCNN_labels = torch.argmax(RCNN_cls_result, dim=2)
        
        RCNN_cls_result_clone = RCNN_cls_result.clone()
        #RCNN_cls_result_clone.requires_grad_(True)
        RCNN_cls_2module_loss = 0
        
        confidence_scores = RCNN_cls_result[..., 1]
        confidence_scores = confidence_scores[:, confidence_scores[0]>=0.5]
        sorted_indices = torch.argsort(confidence_scores, descending=True)
        pred_positive_proposal_num = confidence_scores.shape[1]
        # pred_RCNN_target_point_final = torch.zeros((batch_size, 2))
        # pred_RCNN_target_point_yaw = torch.zeros((batch_size, 1))
        pred_RCNN_target_point_final = torch.zeros((batch_size, pred_positive_proposal_num, 2))
        pred_RCNN_target_point_yaw = torch.zeros((batch_size, pred_positive_proposal_num, 1))
        horizon_sum = 0
        data_gt_list = []
        data_gt_yaw_list = []
        for s in range(batch_size):
            data_horizon_in_train = int(data.horizon[s])
            # print(s, data.horizon[s], horizon_sum)
            # pred_index_tuple = torch.where(RCNN_labels[s][0] == 1)
            # print(RCNN_cls_result.shape, RCNN_labels.shape, pred_index_tuple[0].shape)#, RCNN_cls_result, pred_index_tuple)
            # continue
            # pred_RCNN_target_point = candidate_pos[s][sorted_indices[s][0], :]
            # pred_RCNN_target_point_offset = offset[s][sorted_indices[s][0], :]
            pred_RCNN_target_point = candidate_pos[s][sorted_indices[s][:pred_positive_proposal_num], :]
            pred_RCNN_target_point_offset = offset[s][sorted_indices[s][:pred_positive_proposal_num], :]
            pred_RCNN_target_point_yaw[s] = yaw_pred[s][0]
            pred_RCNN_target_point_final[s] = pred_RCNN_target_point + pred_RCNN_target_point_offset

            data_horizon_in_train -= 7
            each_traj = data.y[horizon_sum * 2:(horizon_sum + data_horizon_in_train) * 2].cpu().view(-1, 2).cumsum(axis=0) #int(data.y[horizon_sum + int(data.horizon[i]) - 1].cpu())
            each_yaw = int(data.y_yaw[horizon_sum + data_horizon_in_train - 1].cpu())
            horizon_sum += data_horizon_in_train
            data_gt_list.append(each_traj)
            data_gt_yaw_list.append(each_yaw)
        gt_np = np.array(data_gt_list) # 128 
        gt_yaw_np = np.array(data_gt_yaw_list) # 128
        pred_RCNN_target_point_yaw = pred_RCNN_target_point_yaw * 180 / np.pi
        for s in range(batch_size):
            if not self.loss_weight_by_dist:
                ego_rec = [gt_np[s][-1][0], gt_np[s][-1][1], vehicle_width
                                        , vehicle_length, (gt_yaw_np[s] + 90.0) * np.pi / 180]
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

                #for target_index in range(n):
                for target_index in range(pred_positive_proposal_num):
                    # print(pred_RCNN_target_point_final[s][target_index][0], pred_RCNN_target_point_final[s][target_index][1], pred_RCNN_target_point_yaw[s][0])
                    # exit()
                    # speed up
                    # if sorted_indices[s][target_index] in RCNN_labels:
                    ego_rec = [pred_RCNN_target_point_final[s][target_index][0], pred_RCNN_target_point_final[s][target_index][1], vehicle_width
                                            , vehicle_length, (pred_RCNN_target_point_yaw[s][0] + 90.0) * np.pi / 180]
                    x_1 = float(torch.cos(
                        ego_rec[4])*(-ego_rec[2]/2) - torch.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                    x_2 = float(torch.cos(
                        ego_rec[4])*(ego_rec[2]/2) - torch.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                    x_3 = float(torch.cos(
                        ego_rec[4])*(-ego_rec[2]/2) - torch.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                    x_4 = float(torch.cos(
                        ego_rec[4])*(ego_rec[2]/2) - torch.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                    y_1 = float(torch.sin(
                        ego_rec[4])*(-ego_rec[2]/2) + torch.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                    y_2 = float(torch.sin(
                        ego_rec[4])*(ego_rec[2]/2) + torch.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                    y_3 = float(torch.sin(
                        ego_rec[4])*(-ego_rec[2]/2) + torch.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                    y_4 = float(torch.sin(
                        ego_rec[4])*(ego_rec[2]/2) + torch.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                    pred_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])

                    now_iou = ego_polygon.intersection(pred_polygon).area / ego_polygon.union(pred_polygon).area
                    if now_iou < iou_threshold:
                        # IOU not enough => filter out
                        # print(s, sorted_indices[s][target_index], now_iou, RCNN_cls_result[s][target_index])
                        RCNN_cls_2module_loss += (1.0 - RCNN_cls_result_clone[s][target_index][0])
                        RCNN_cls_2module_loss += (RCNN_cls_result_clone[s][target_index][0])
                        RCNN_cls_result_clone[s][target_index][0] = 1.0
                        RCNN_cls_result_clone[s][target_index][1] = 0.0
                        
                        # print(RCNN_cls_result_clone[s][target_index])
            else:
                for target_index in range(pred_positive_proposal_num):
                    target_dist = (gt_np[s][-1][0] - pred_RCNN_target_point_final[s][target_index][0]) ** 2 + (gt_np[s][-1][1] - pred_RCNN_target_point_final[s][target_index][1]) ** 2
                    RCNN_cls_2module_loss += torch.tensor([target_dist], dtype=torch.float, device=self.device)
        # RCNN_cls_result_clone.requires_grad_(True)
        return RCNN_cls_result_clone, RCNN_cls_2module_loss
        
