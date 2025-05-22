# TNT model
import os
from tqdm import tqdm
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
# from core.model.backbone.vectornet import VectorNetBackbone
from core.model.backbone.vectornet_v2 import VectorNetBackbone
from core.model.layers.target_prediction import TargetPred
from core.model.layers.RCNNclassifier import RCNNClassifier
from core.model.layers.RCNNclassifier_2module import RCNNClassifier_2module
from core.model.layers.target_regression import TargetReg
from core.model.layers.atr_offset_prediction import AtrOffsetPred
from core.model.layers.ttc_prediction import TTCPred
from core.model.layers.tp_selector import TPSelector
from core.model.layers.final_feature import FinalFeat 
from core.model.layers.trajectory_encoder import TrajEncoder
# from core.model.layers.target_prediction_v2 import TargetPred
from core.model.layers.motion_etimation import MotionEstimation
from core.model.layers.scoring_and_selection import TrajScoreSelection, distance_metric
from core.loss import TNTLoss
import matplotlib.pyplot as plt
from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem


class TNT(nn.Module):
    def __init__(self,
                 in_channels=8,
                 horizon=0,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 with_aux=True,
                 aux_width=64,
                 target_pred_hid=64,
                 m=5,
                 motion_esti_hid=64,
                 score_sel_hid=64,
                 temperature=0.01,
                 k=6,
                 lambda1=0.1,
                 lambda2=1.0,
                 lambda3=0.1,
                 device=torch.device("cpu"),
                 multi_gpu: bool = False,
                 positive_weight=10,
                 nearest_subgraph=10):
        """
        TNT algorithm for trajectory prediction
        :param in_channels: int, the number of channels of the input node features
        :param horizon: int, the prediction horizon (prediction length)
        :param num_subgraph_layers: int, the number of subgraph layer
        :param num_global_graph_layer: the number of global interaction layer
        :param subgraph_width: int, the channels of the extrated subgraph features
        :param global_graph_width: int, the channels of extracted global graph feature
        :param with_aux: bool, with aux loss or not
        :param aux_width: int, the hidden dimension of aux recovery mlp
        :param n: int, the number of sampled target candidate
        :param target_pred_hid: int, the hidden dimension of target prediction
        :param m: int, the number of selected candidate
        :param motion_esti_hid: int, the hidden dimension of motion estimation
        :param score_sel_hid: int, the hidden dimension of score module
        :param temperature: float, the temperature when computing the score
        :param k: int, final output trajectories
        :param lambda1: float, the weight of candidate prediction loss
        :param lambda2: float, the weight of motion estimation loss
        :param lambda3: float, the weight of trajectory scoring loss
        :param device: the device for computation
        :param multi_gpu: the multi gpu setting
        """
        super(TNT, self).__init__()
        self.horizon = horizon
        self.m = m
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.with_aux = with_aux

        self.positive_weight = positive_weight

        # print(in_channels, horizon,num_subgraph_layers,
        #          num_global_graph_layer,
        #          subgraph_width,
        #          global_graph_width,
        #          with_aux,
        #          aux_width,
        #          target_pred_hid )
        # exit()

        self.device = device
        self.criterion = TNTLoss(
            self.lambda1, self.lambda2, self.lambda3, temperature, aux_loss=self.with_aux, device=self.device
        )

        # feature extraction backbone
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layres=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            aux_mlp_width=aux_width,
            device=device
        )
        # self.trajectory_encoder_layer = TrajEncoder(
        #     in_channels=global_graph_width,
        #     hidden_dim=target_pred_hid,
        #     device=device
        # )
        self.final_feature_layer = FinalFeat(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            device=device
        )
        self.target_regression_layer = TargetReg(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        
        self.target_pred_layer = TargetPred(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        self.atr_offset_layer = AtrOffsetPred(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        self.tp_selector_layer = TPSelector(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        self.ttc_pred_layer = TTCPred(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        self.motion_estimator = MotionEstimation(
            in_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=motion_esti_hid
        )
        self.traj_score_layer = TrajScoreSelection(
            feat_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=score_sel_hid,
            temper=temperature,
            device=self.device
        )
        self._init_weight()
        self.class_list = ["junction_crossing", "LTAP", "lane_change", "opposite_direction", "rear_end"]
        self.class_num = len(self.class_list)
        ###################
        self.m = 50
        self.LSTM_setting = False
        self.attacker_only_regression = False
        self.regression = False
        self.object_detection_first_classification = True
        self.local_fea_subgraph_num = nearest_subgraph #10
        self.object_detection_2module = False
        ###################
        self.RCNNClassifier_layer = RCNNClassifier(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device,
            nearest_subgraph=self.local_fea_subgraph_num
        )
        # self.RCNNClassifier_2module_layer = RCNNClassifier_2module(
        #     in_channels=global_graph_width,
        #     hidden_dim=target_pred_hid,
        #     m=m,
        #     device=device,
        #     nearest_subgraph=self.local_fea_subgraph_num
        # )

    def forward(self, data):
        """
        output prediction for training
        :param data: observed sequence data
        :return: dict{
                        "target_prob":  the predicted probability of each target candidate,
                        "offset":       the predicted offset of the target position from the gt target candidate,
                        "traj_with_gt": the predicted trajectory with the gt target position as the input,
                        "traj":         the predicted trajectory without the gt target position,
                        "score":        the predicted score for each predicted trajectory,
                     }
        """
        n = data.candidate_len_max[0]
        #print(n)
        #exit()
        
        target_candidate = data.candidate.view(-1, n, 2)   # [batch_size, N, 2]
        batch_size, _, _ = target_candidate.size()
        candidate_mask = data.candidate_mask.view(-1, n)

        target_candidate_with_id = data.candidate_with_id.view(-1, n, 4)   # [batch_size, N, 4]
        #print(n, target_candidate.shape)

        # feature encoding
        global_feat, aux_out, aux_gt, target_candidate_fea = self.backbone(data, self.local_fea_subgraph_num)             # [batch_size, time_step_len, global_graph_width]
        
        need_save_scenario_map_feat = False
        if need_save_scenario_map_feat:
            global_feat_np = global_feat.cpu().detach().numpy()
            np.save("map_feat/" + data['seq_id'][0], global_feat_np)

        
        #print(aux_gt.shape) #128, 64 # batch_size * vectornet_width
        #print(data.atr_pos_offset, data.atr_yaw_offset) (128, 2) (128)
        
        #self.class_list
        target_feat = global_feat[:, 0].unsqueeze(1) 
        # print(global_feat.shape, target_feat.shape) # 128,131,64 => 128,1,64
        # exit()
        if self.LSTM_setting:
            final_feat = self.trajectory_encoder_layer(data, data.ttc, data.one_hot, batch_size)
        else:
            final_feat = self.final_feature_layer(target_feat, data.ttc, data.one_hot, self.class_num)
        #print(target_feat.shape, final_feat.shape) #(128, 1, 64) (128, 1, 70)
        # final_feat = self.final_feature_layer(target_feat, data.attacker_id, data.one_hot, self.class_num)
        #print("horizon:", data.horizon.sum())
        #print("data.y:", data.y.shape, data.y) #data.y.shape == data.horizon.sum() * 2
        #print("y:", data.y.view(-1, self.horizon * 2).shape) #original: 64, 60

        # horizon_sum = 0
        # data_y_list = []
        # #print(data.y_ttc.shape) #640 (64*10)
        # for i in range(batch_size):
        #     print(horizon_sum, int(data.horizon[i]))
        #     each_list = data.y_ttc[horizon_sum:horizon_sum + int(data.horizon[i])].cpu()
        #     horizon_sum += int(data.horizon[i])
        #     data_y_list.append(each_list)
        #     #print(each_list)
        # #print(data_y_list)
        # # tensor seems to be unable to concat different size 
        # #print(torch.from_numpy(np.array(data_y_list)).float().shape)
        # data_y = np.array(data_y_list)

        # print("target_feat:", target_feat.shape) 64, 1, 64
        # print("target_candidate:", target_candidate.shape) 64, 2485, 2
        # print("candidate_mask:", candidate_mask.shape) 64, 2485
        
        #print(target_candidate.shape, candidate_mask.shape)
        #sys.exit()
        # predict prob. for each target candidate, and corresponding offest
        #target_prob, offset, yaw_pred = self.target_pred_layer(target_feat, target_candidate, candidate_mask)

        if self.regression:
            offset, yaw_pred = self.target_regression_layer(final_feat, data)
            # only for not bug
            # _, _, _ = self.target_pred_layer(final_feat, target_candidate, candidate_mask)
            RCNN_cls_result, _ = self.RCNNClassifier_layer(final_feat, target_candidate, target_candidate_fea, self.local_fea_subgraph_num, target_candidate_with_id)
            target_prob = torch.zeros((batch_size, n))
        elif self.object_detection_first_classification:
            RCNN_cls_result, proposal_feat = self.RCNNClassifier_layer(final_feat, target_candidate, target_candidate_fea, self.local_fea_subgraph_num, target_candidate_with_id)
            _, offset, yaw_pred = self.target_pred_layer(final_feat, target_candidate, candidate_mask)
            target_prob = torch.zeros((batch_size, n))
        else:
            target_prob, offset, yaw_pred = self.target_pred_layer(final_feat, target_candidate, candidate_mask)
        tar_offset_pred, atr_yaw_pred = self.atr_offset_layer(final_feat)
        tp_n = data.tp_candidate_len_max[0]

        tp_candidate = data.tp_candidate.view(-1, tp_n, 2)   # [batch_size, N, 2]
        #batch_size, _, _ = target_candidate.size()
        tp_candidate_mask = data.tp_candidate_mask.view(-1, tp_n)
        tp_target_prob = self.tp_selector_layer(final_feat, tp_candidate, tp_candidate_mask)
        # print(tar_offset_pred.shape, atr_yaw_pred.shape, tp_target_prob.shape) # 128, 2 # 128, 1 # 128, 76
        #pred_ttc = self.ttc_pred_layer(data.y_ttc)
        #print(pred_ttc.shape) #64, 1

        # print(offset.shape, yaw_pred.shape) # 128, 525, 2 # 128, 1, 1
        # exit()

        # print("target_prob:", target_prob.shape) 64, 2485
        # print("offset:", offset.shape) 64, 2485, 2
        
        # # predict the trajectory given the target gt
        # target_gt = data.target_gt.view(-1, 1, 2)
        # traj_with_gt = self.motion_estimator(target_feat, target_gt)

        # # predict the trajectories for the M most-likely predicted target, and the score
        # _, indices = target_prob.topk(self.m, dim=1)
        # batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.m)]).T
        # target_pred_se, offset_pred_se = target_candidate[batch_idx, indices], offset[batch_idx, indices]
        # trajs = self.motion_estimator(target_feat, target_pred_se + offset_pred_se)
        # score = self.traj_score_layer(target_feat, trajs)

        # No need to choose top 50 point, because yaw is attached to each target point
        #target_pred_yaw = yaw_pred[batch_idx, indices]
        
        RCNN_cls_result_clone = None
        RCNN_cls_2module_loss = 0
        if self.object_detection_2module:
            RCNN_cls_result_clone, RCNN_cls_2module_loss = self.RCNNClassifier_2module_layer(proposal_feat, target_candidate, data, RCNN_cls_result, offset, yaw_pred)
        
        return {
            "target_prob": target_prob,
            "offset": offset,
            #"traj_with_gt": traj_with_gt,
            #"traj": trajs,
            #"score": score,
            "target_pred_yaw": yaw_pred,
            "tar_offset_pred": tar_offset_pred,
            "atr_yaw_pred": atr_yaw_pred,
            "tp_prob": tp_target_prob,
            "final_feat": final_feat,
            "RCNN_cls": RCNN_cls_result,
            "RCNN_cls_clone": RCNN_cls_result_clone,
            "RCNN_cls_2module_loss": RCNN_cls_2module_loss,
        }, aux_out, aux_gt

    def loss(self, data):
        """
        compute loss according to the gt
        :param data: node feature data
        :return: loss
        """
        n = data.candidate_len_max[0]
        pred, aux_out, aux_gt = self.forward(data)
        # print("aux:", aux_gt)

        tp_num = data.tp_candidate_len_max[0]
        #print(data.horizon.shape, data.horizon) 64
        #print(data.y_yaw.shape)# 3204 can't % 30 == 0
        #print("atr_pos_offset:", data.atr_pos_offset.shape)
        gt = {
            "target_prob": data.candidate_gt.view(-1, n),
            "offset": data.offset_gt.view(-1, 2),
            "offset_each": data.offset_gt_each.view(-1, 2),
            "target_candidate_lens": data.candidate_lens,
            #"y": data.y.view(-1, self.horizon * 2),
            "y": data.y,
            "y_yaw": data.y_yaw,
            "ttc": data.horizon,
            "atr_pos_offset": data.atr_pos_offset,
            "atr_yaw_offset": data.atr_yaw_offset,
            "tp_gt": data.tp_gt.view(-1, tp_num),
            "candidate": data.candidate.view(-1, n, 2),
            "obs_yaw": data.obs_yaw,
        }

        return self.criterion(pred, gt, aux_out, aux_gt, self.positive_weight)

    def inference(self, data):
        """
        predict the top k most-likely trajectories
        :param data: observed sequence data
        :return:
        """
        n = data.candidate_len_max[0]
        target_candidate = data.candidate.view(-1, n, 2)    # [batch_size, N, 2]


        target_candidate_with_id = data.candidate_with_id.view(-1, n, 4)   # [batch_size, N, 4]
        # target_candidate_with_id = None
        batch_size, _, _ = target_candidate.size()

        global_feat, _, _, target_candidate_fea = self.backbone(data, self.local_fea_subgraph_num)     # [batch_size, time_step_len, global_graph_width]
        target_feat = global_feat[:, 0].unsqueeze(1)
        final_feat = self.final_feature_layer(target_feat, data.ttc, data.one_hot, self.class_num)

        
        if self.regression:
            print("In only regression")
            offset_pred, yaw_pred = self.target_regression_layer(final_feat, data)
            
            # for no bug
            target_prob, _, _ = self.target_pred_layer(final_feat, target_candidate)
            RCNN_cls_result, _ = self.RCNNClassifier_layer(final_feat, target_candidate, target_candidate_fea, self.local_fea_subgraph_num, target_candidate_with_id)
            #print(data.orig.shape, target_candidate)
            #print(data.obs_traj.reshape(batch_size, 8, 2)[:, -1, :])
            #exit()
            ### don't need to modify metric if we repeat like target-based method ###
            target_candidate = torch.zeros((target_candidate.shape)).cuda()
            offset_pred = offset_pred.repeat(1, n, 1)
            yaw_pred = yaw_pred.repeat(1, n, 1)

            #### if norm ####
            scale = 1 # 17 for 128 scenes
            offset_pred *= scale
            #### if norm ####

        elif self.object_detection_first_classification:
            RCNN_cls_result, _ = self.RCNNClassifier_layer(final_feat, target_candidate, target_candidate_fea, self.local_fea_subgraph_num, target_candidate_with_id)
            # predicted_labels = torch.argmax(RCNN_cls_result, dim=2)
            # print(predicted_labels)
            
            target_prob, offset_pred, yaw_pred = self.target_pred_layer(final_feat, target_candidate)
            # print("Predicted Labels:", np.sum(predicted_labels[0].cpu().numpy()), predicted_labels[0].cpu().numpy())
            # exit()
        else:
            target_prob, offset_pred, yaw_pred = self.target_pred_layer(final_feat, target_candidate)
            # print(offset_pred[0][:20])
            # exit()
        #_, indices = data.candidate_gt.view(-1, n).topk(1, dim=1)
        tar_offset_pred, atr_yaw_pred = self.atr_offset_layer(final_feat)
        #print("GT bbox:", indices, data.candidate_gt.view(-1, n), data.candidate_gt.view(-1, n)[0][indices], target_candidate[0, indices])
        # print(data.candidate_gt.view(-1, n)[0])
        #print(target_candidate[0, :, 0])

        # print(offset_pred.shape, offset_pred)
        # print(data.offset_gt_each.shape, data.offset_gt_each)
        # exit()

        
        
        #for i in range(target_candidate.shape[1]):
        #    plt.plot(target_candidate[0, :, 0].cpu().numpy(), target_candidate[0, :, 1].cpu().numpy(), '-', color='black')
        #plt.show()
        
        
        

        # if object_detection_first_classification:
        #     indices = torch.nonzero(target_prob == 1, as_tuple=False)
        #     indices = torch.argmax(target_prob, dim=1)
        _, indices = target_prob.topk(self.m, dim=1)
        batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.m)]).T
        target_pred_se, offset_pred_se = target_candidate[batch_idx, indices], offset_pred[batch_idx, indices]
        
        # GT
        target_gt = data.candidate_gt.view(-1, n)
        _, indices_gt = target_gt.topk(1, dim=1)
        #print(indices_gt.shape, indices_gt)
        #comp_np = np.array((indices_gt.cpu().numpy(), indices.cpu().numpy())).T
        comp_np = torch.cat((indices_gt, indices), 1)
        #print(comp_np)
        target_gt_pos = target_candidate[batch_idx, indices_gt]

        
        
        debug = False
        
        if debug:
            print(target_gt_pos.shape, target_gt_pos)
            for i in range(indices.shape[0]):
                print(data.seq_id[i])
                exit()
                for each_target in range(n):
                    plt.text(target_candidate[i][each_target][0], target_candidate[i][each_target][1], each_target, c="black", fontsize = 5)
                    if 1665 < each_target < 1680:
                        plt.text(target_candidate[i][each_target][0], target_candidate[i][each_target][1], each_target, c="green")
                for j in range(self.m):
                    #print(target_pred_se[i][j][0], target_pred_se[i][j][1], indices[i][j])
                    plt.text(target_pred_se[i][j][0], target_pred_se[i][j][1], indices[i][j].cpu().numpy(), c="blue")
                plt.text(target_gt_pos[i][0][0], target_gt_pos[i][0][1], indices_gt[i][0].cpu().numpy(), c="red")
                plt.xlim(target_gt_pos[i][0][0].cpu().numpy() - 25,
                            target_gt_pos[i][0][0].cpu().numpy() + 25)
                plt.ylim(target_gt_pos[i][0][1].cpu().numpy() - 25,
                            target_gt_pos[i][0][1].cpu().numpy() + 25)
                plt.show()
                plt.close()
            exit()
        
        # predict only 1 value instead of each value per position
        only_predict_1_ego_yaw = True
        if only_predict_1_ego_yaw:
            pred_target_point_yaw = yaw_pred[batch_idx, 0]
        else:
            pred_target_point_yaw = yaw_pred[batch_idx, indices]

        tp_n = data.tp_candidate_len_max[0]
        #print("data.tp_candidate_len_max:", data.candidate_gt)#data.tp_candidate_len_max)
        # print(yaw_pred[batch_idx].shape, pred_target_point_yaw.shape)
        # exit()
        tp_candidate = data.tp_candidate.view(-1, tp_n, 2)   # [batch_size, N, 2]
        #batch_size, _, _ = target_candidate.size()
        tp_candidate_mask = data.tp_candidate_mask.view(-1, tp_n)
        tp_target_prob = self.tp_selector_layer(final_feat, tp_candidate, tp_candidate_mask)
        _, tp_indices = tp_target_prob.topk(1, dim=1)
        tp_batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(1)]).T
        tp_pred = tp_candidate[tp_batch_idx, tp_indices]

        # print("target_pred_se:", target_pred_se.shape)
        # print("tp_pred:", tp_pred.shape)
        # print("target_pred_se:", indices)
        # print("tp_pred:", tp_indices)
        
        #tp_index = [tp_batch_idx, tp_indices]
        tp_index = tp_indices

        #print(data.obs_traj.shape) 1280 2
        #####################
        obs_traj = data.obs_traj.reshape((-1, 8, 2))
        
        # print(target_pred_se, offset_pred_se)
        pred_target_point = target_pred_se + offset_pred_se
        # print(target_pred_se, offset_pred_se, pred_target_point)
        # print(offset_pred_se.shape, offset_pred_se)
        
        #print(offset_pred.shape, offset_pred[0][:700])
        #for i in range(738):
        #    print(offset_pred[0][i])
        #exit()
        
        #print(target_pred_se[:10, :1, :], offset_pred_se[:10, :1, :], data.offset_gt.view(-1, 2))
        #print(offset_pred_se.shape, offset_pred_se)
        #exit()
        # print(data.offset_gt.view(-1, 2))
        # sys.exit()
        #print(obs_traj[:, :, :2].shape) #128 10 2
        #print(pred_target_point[:, 0, :].shape) #128 50 2
        
        #############################################
        # dist_fro_ttc = []
        # for i in range(batch_size):
        #     offset_fut = pred_target_point[i, 0, :] - obs_traj[i, :, :2]
        #     dist_fro_ttc.append(torch.sqrt(offset_fut[:, 0] ** 2 + offset_fut[:, 1] ** 2).cpu().numpy())
        # pred_target_for_ttc = torch.from_numpy(np.array(dist_fro_ttc)).cuda()
        # pred_ttc = self.ttc_pred_layer(pred_target_for_ttc)
        #############################################
        
        # # DEBUG
        # gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1)

        # trajectory estimation for the m predicted target location
        #traj_pred = self.motion_estimator(target_feat, target_pred_se + offset_pred_se)

        # score the predicted trajectory and select the top k trajectory
        #score = self.traj_score_layer(target_feat, traj_pred)
        if self.object_detection_first_classification:
            offset_pred_se = offset_pred
        

        if self.attacker_only_regression:
            zeros_tensor = torch.zeros_like(pred_target_point)
            pred_target_point[:] = zeros_tensor
        return target_prob, target_pred_se, offset_pred_se, yaw_pred, pred_target_point, pred_target_point_yaw, tar_offset_pred, atr_yaw_pred, tp_index, RCNN_cls_result
        return self.traj_selection(traj_pred, score), target_prob, offset_pred, yaw_pred, pred_ttc

    def candidate_sampling(self, data):
        """
        sample candidates given the test data
        :param data:
        :return:print
        """
        raise NotImplementedError

    # todo: determine appropiate threshold
    def traj_selection(self, traj_in, score, threshold=0.01):
        """
        select the top k trajectories according to the score and the distance
        :param traj_in: candidate trajectories, [batch, M, horizon * 2]
        :param score: score of the candidate trajectories, [batch, M]
        :param threshold: float, the threshold for exclude traj prediction
        :return: [batch_size, k, horizon * 2]
        """
        # re-arrange trajectories according the the descending order of the score
        _, batch_order = score.sort(descending=True)
        traj_pred = torch.cat([traj_in[i, order] for i, order in enumerate(batch_order)], dim=0).view(-1, self.m, self.horizon * 2)
        traj_selected = traj_pred[:, :self.k]                                   # [batch_size, k, horizon * 2]

        # check the distance between them, NMS, stop only when enough trajs collected
        for batch_id in range(traj_pred.shape[0]):                              # one batch for a time
            traj_cnt = 1
            while traj_cnt < self.k:
                for j in range(1, self.m):
                    dis = distance_metric(traj_selected[batch_id, :traj_cnt], traj_pred[batch_id, j].unsqueeze(0))
                    if not torch.any(dis < threshold):
                        traj_selected[batch_id, traj_cnt] = traj_pred[batch_id, j]

                        traj_cnt += 1
                    if traj_cnt >= self.k:
                        break
                threshold /= 2.0

        return traj_selected

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == "__main__":
    batch_size = 32
    DATA_DIR = "/home/carla/experiments/TNT-Trajectory-Predition/carla_all_data/interm_data"
    # DATA_DIR = "../../dataset/interm_tnt_n_s_0804"
    TRAIN_DIR = os.path.join(DATA_DIR, 'train_intermediate')
    # TRAIN_DIR = os.path.join(DATA_DIR, 'val_intermediate')
    # TRAIN_DIR = os.path.join(DATA_DIR, 'test_intermediate')

    dataset = ArgoverseInMem(TRAIN_DIR)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    m, k = 50, 6
    pred_len = 30

    # device = torch.device("cuda:1")
    device = torch.device("cpu")

    model = TNT(in_channels=dataset.num_features,
                horizon=pred_len,
                m=m,
                k=k,
                with_aux=True,
                device=device).to(device)

    # train mode
    model.train()
    for i, data in enumerate(tqdm(data_iter)):
        loss, _ = model.loss(data.to(device))
        print("Training Pass! loss: {}".format(loss))

        if i == 2:
            break

    # eval mode
    model.eval()
    for i, data in enumerate(tqdm(data_iter)):
        pred = model(data.to(device))
        print("Evaluation Pass! Shape of out: {}".format(pred.shape))

        if i == 2:
            break
