import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import sys
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
import math
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from argoverse.evaluation.competition_util import generate_forecasting_h5
from Quintic import quintic_polynomials_planner
from core.trainer.trainer import Trainer
from core.model.TNT import TNT
from core.optim_schedule import ScheduledOptim
from core.util.viz_utils import show_pred_and_gt
from pytp.utils.evaluate import get_ade, get_fde
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Polygon as pg
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
import csv

VEH_COLL_THRESH = 0.02

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

def get_iou(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)

    iou_area = iou_w * iou_h
    all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area

    return max(iou_area/all_area, 0)

class TNTTrainer(Trainer):
    """
    VectorNetTrainer, train the vectornet with specified hyperparameters and configurations
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 num_global_graph_layer=1,
                 num_subgraph_layers=3,
                 subgraph_width=64,
                 global_graph_width=64,
                 target_pred_hid=64,
                 horizon: int = 0,
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=20,
                 lr_update_freq=5,
                 lr_decay_rate=0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
                 enable_log=True,
                 log_freq: int = 2,
                 save_folder: str = "./tnt",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True,
                 positive_weight = 10,
                 ):
        """
        trainer class for vectornet
        :param train_loader: see parent class
        :param eval_loader: see parent class
        :param test_loader: see parent class
        :param lr: see parent class
        :param betas: see parent class
        :param weight_decay: see parent class
        :param warmup_steps: see parent class
        :param with_cuda: see parent class
        :param multi_gpu: see parent class
        :param log_freq: see parent class
        :param model_path: str, the path to a trained model
        :param ckpt_path: str, the path to a stored checkpoint to be resumed
        :param verbose: see parent class
        """
        super(TNTTrainer, self).__init__(
            trainset=trainset,
            evalset=evalset,
            testset=testset,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            cuda_device=cuda_device,
            enable_log=enable_log,
            log_freq=log_freq,
            save_folder=save_folder,
            verbose=verbose,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.positive_weight = positive_weight
        # init or load model
        self.aux_loss = aux_loss
        # input dim: (20, 8); output dim: (30, 2)
        # model_name = VectorNet
        model_name = TNT
        self.model = model_name(
            self.trainset.num_features if hasattr(self.trainset, 'num_features') else self.testset.num_features,
            horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device,
            multi_gpu=self.multi_gpu,
            positive_weight=positive_weight,
        )
        
        # resume from model file or maintain the original
        if model_path:
            self.load(model_path, 'm')
        #print("tnt_trainer")
        if self.multi_gpu:
            # self.model = DataParallel(self.model)
            if self.verbose:
                print("[TNTTrainer]: Train the mode with multiple GPUs: {}.".format(self.cuda_id))
        else:
            if self.verbose:
                print("[TNTTrainer]: Train the mode with single device on {}.".format(self.device))
        self.model = self.model.to(self.device)

        # init optimizer
        self.optim = AdamW(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optm_schedule = ScheduledOptim(
            self.optim,
            self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )
        # record the init learning rate
        self.write_log("LR", self.lr, 0)
        # df = pd.DataFrame(columns=['epoch', 'train_loss', 'cls_loss', 'offset_loss', 'yaw_loss', 'ttc_loss'])
        # df.to_csv(save_folder + '/training_loss.csv', index=False)
        # df.to_csv(save_folder + '/eval_loss.csv', index=False)
        self.save_folder = save_folder

        # resume training from ckpt
        if ckpt_path:
            self.load(ckpt_path, 'c')

    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        #print("before iteration!!!!!")

        data_iter = tqdm(
            enumerate(dataloader),
            desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("train" if training else "eval",
                                                                   epoch,
                                                                   0.0,
                                                                   avg_loss),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        train_loss = 0
        cls_loss = 0
        offset_loss = 0
        yaw_loss = 0
        #ttc_loss = 0
        atr_pos_loss = 0
        atr_yaw_loss = 0
        tp_cls_loss = 0
        aux_loss = 0
        for i, data in data_iter:
            n_graph = data.num_graphs
            # ################################### DEBUG ################################### #
            # if epoch > 0:
            #     print("\nsize of x: {};".format(data.x.shape))
            #     print("size of cluster: {};".format(data.cluster.shape))
            #     print("valid_len: {};".format(data.valid_len))
            #     print("time_step_len: {};".format(data.time_step_len))
            #
            #     print("size of candidate: {};".format(data.candidate.shape))
            #     print("size of candidate_mask: {};".format(data.candidate_mask.shape))
            #     print("candidate_len_max: {};".format(data.candidate_len_max))
            # ################################### DEBUG ################################### #self.write_log
            if training:
                if self.multi_gpu:
                    # loss, loss_dict = self.model.module.loss(data.to(self.device))
                    loss, loss_dict = self.model.loss(data.to(self.device))
                    # loss, loss_dict = self.model.module.loss(data)
                else:
                    loss, loss_dict = self.model.loss(data.to(self.device))

                self.optm_schedule.zero_grad()
                loss.backward()
                self.optim.step()

                # writing loss
                self.write_log("Train_Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Target_Cls_Loss",
                               loss_dict["tar_cls_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Target_Offset_Loss",
                               loss_dict["tar_offset_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Yaw_Loss",
                               loss_dict["yaw_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Atr_Pos_Loss",
                               loss_dict["atr_pos_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Atr_Yaw_Loss",
                               loss_dict["atr_yaw_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("TP_Cls_loss",
                               loss_dict["tp_cls_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Aux_loss",
                               loss_dict["aux_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Reg_loss",
                               loss_dict["reg_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("RCNN_loss",
                               loss_dict["RCNN_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                # self.write_log("TTC_Loss",
                #                loss_dict["ttc_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                # self.write_log("Traj_Loss",
                #                loss_dict["traj_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                # self.write_log("Score_Loss",
                #                loss_dict["score_loss"].detach().item() / n_graph, i + epoch * len(dataloader))

                # train_loss = loss.detach().item() / n_graph, i + epoch * len(dataloader)
                # cls_loss = loss_dict["tar_cls_loss"].detach().item() / n_graph, i + epoch * len(dataloader)
                # offset_loss = loss_dict["tar_offset_loss"].detach().item() / n_graph, i + epoch * len(dataloader)
                # yaw_loss = loss_dict["yaw_loss"].detach().item() / n_graph, i + epoch * len(dataloader)
                # loss_list = [epoch, train_loss, cls_loss, offset_loss, yaw_loss]
                # loss_df = pd.DataFrame([loss_list])
                # loss_df.to_csv(self.save_folder + '/loss.csv', mode='a', header=False, index=False)

                train_loss += loss.detach().item() / n_graph
                cls_loss += loss_dict["tar_cls_loss"].detach().item() / n_graph
                offset_loss += loss_dict["tar_offset_loss"].detach().item() / n_graph
                yaw_loss += loss_dict["yaw_loss"].detach().item() / n_graph
                # ttc_loss += loss_dict["ttc_loss"].detach().item() / n_graph
                atr_pos_loss += loss_dict["atr_pos_loss"].detach().item() / n_graph
                atr_yaw_loss += loss_dict["atr_yaw_loss"].detach().item() / n_graph
                tp_cls_loss += loss_dict["tp_cls_loss"].detach().item() / n_graph
                aux_loss += loss_dict["aux_loss"].detach().item() / n_graph
                

            else:
                with torch.no_grad():
                    if self.multi_gpu:
                        # loss, loss_dict = self.model.module.loss(data.to(self.device))
                        loss, loss_dict = self.model.loss(data.to(self.device))
                    else:
                        loss, loss_dict = self.model.loss(data.to(self.device))

                    # writing loss
                    self.write_log("Eval_Loss", loss.item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Target_Cls_Loss(Eval)",
                                   loss_dict["tar_cls_loss"].item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Target_Offset_Loss(Eval)",
                                   loss_dict["tar_offset_loss"].item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Yaw_Loss(Eval)",
                                   loss_dict["yaw_loss"].item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Atr_Pos_Loss(Eval)",
                                   loss_dict["atr_pos_loss"].item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Atr_Yaw_Loss(Eval)",
                                loss_dict["atr_yaw_loss"].item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("TP_Cls_loss(Eval)",
                                loss_dict["tp_cls_loss"].item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Reg_loss(Eval)",
                                loss_dict["reg_loss"].item() / n_graph, i + epoch * len(dataloader))


                    train_loss += loss.detach().item() / n_graph
                    cls_loss += loss_dict["tar_cls_loss"].detach().item() / n_graph
                    offset_loss += loss_dict["tar_offset_loss"].detach().item() / n_graph
                    yaw_loss += loss_dict["yaw_loss"].detach().item() / n_graph
                    # ttc_loss += loss_dict["ttc_loss"].detach().item() / n_graph
                    atr_pos_loss += loss_dict["atr_pos_loss"].detach().item() / n_graph
                    atr_yaw_loss += loss_dict["atr_yaw_loss"].detach().item() / n_graph
                    tp_cls_loss += loss_dict["tp_cls_loss"].detach().item() / n_graph

            num_sample += n_graph
            avg_loss += loss.detach().item()

            desc_str = "[Info: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format("train" if training else "eval",
                                                                                 epoch,
                                                                                 loss.detach().item() / n_graph,
                                                                                 avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)
        
        if training:
            learning_rate = self.optm_schedule.step_and_update_lr()
            #learning_rate = self.lr
            self.write_log("LR", learning_rate, epoch)

            loss_csv_file = self.save_folder + '/training_loss.csv'
        else:
            loss_csv_file = self.save_folder + '/eval_loss.csv'
        # loss_list = [epoch, train_loss / len(dataloader), cls_loss / len(dataloader), offset_loss / len(dataloader), yaw_loss / len(dataloader), ttc_loss / len(dataloader)]
        # loss_df = pd.DataFrame([loss_list])
        # loss_df.to_csv(loss_csv_file, mode='a', header=False, index=False)

        return avg_loss

    def test(self,
             m=1,
             split=None,
             save_folder='./tnt',
             miss_threshold=2.0,
             compute_metric=False,
             convert_coordinate=False,
             plot=True,
             save_pred=False):
        """
        test the testset,
        :param miss_threshold: float, the threshold for the miss rate, default 2.0m
        :param compute_metric: bool, whether compute the metric
        :param convert_coordinate: bool, True: under original coordinate, False: under the relative coordinate
        :param save_pred: store the prediction or not, store in the Argoverse benchmark format
        """
        print("in test!!")
        self.model.eval()

        forecasted_trajectories, gt_trajectories = {}, {}

        # k = self.model.k if not self.multi_gpu else self.model.module.k
        k = self.model.k
        # horizon = self.model.horizon if not self.multi_gpu else self.model.module.horizon
        horizon = self.model.horizon
        self.m = m

        # debug
        out_dict = {}
        out_cnt = 0
        collision_ratio = 1.0

        only_1_batch_flag = 0
        ideal_yaw_offset = 0
        vehicle_length = 4.7 * collision_ratio
        vehicle_width = 2 * collision_ratio

        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width

        all_iou = 0
        all_iou_no_0 = 0
        ego_iou_2 = 0
        ego_iou_50 = 0
        ego_iou_70 = 0
        attacker_iou_2 = 0
        attacker_iou_50 = 0
        all_data_num = 0
        sum_fde = 0
        sum_attacker_de = 0
        fde_list = []
        gt_trajectories_distribution = []
        pred_trajectories_distribution = []
        padding_number_list = []
        positive_num_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        gt_positive_proposal_num = []
        ego_iou_list = []
        atr_iou_list = []
        all_TP = 0
        all_FP = 0
        all_FN = 0
        no_guess_list = []
        ttc_distance_sum = 0
        batch_num = 0
        all_cls_acc = 0
        all_top5_cls_acc = 0
        all_yaw_distance = 0
        over50_yaw_distance = 0
        below50_yaw_distance = 0
        gt_collision_rate = 0
        pred_collision_rate = 0
        all_tp_cls_acc = 0
        attacker_right_collision_rate = 0

        tolerance_degree_10 = 0
        tolerance_degree_20 = 0
        tolerance_degree_30 = 0
        ideal_yaw_dist_average = 0
        real_yaw_dist_average = 0

        col_degree_30 = 0

        lane_change_num = 0
        junction_crossing_num = 0
        LTAP_num = 0
        opposite_direction_num = 0
        rear_end_num = 0

        lane_change_all = 0
        junction_crossing_all = 0
        LTAP_all = 0
        opposite_direction_all = 0
        rear_end_all = 0

        lane_change_yaw_distance = 0
        junction_crossing_yaw_distance = 0
        LTAP_yaw_distance = 0
        opposite_direction_yaw_distance = 0
        rear_end_yaw_distance = 0

        col_lane_change_num = 0.0000001
        col_junction_crossing_num = 0.0000001
        col_LTAP_num = 0.0000001
        col_opposite_direction_num = 0.0000001
        col_rear_end_num = 0.0000001

        sim_lane_change_num = 0
        sim_junction_crossing_num = 0
        sim_LTAP_num = 0
        sim_opposite_direction_num = 0
        sim_rear_end_num = 0

        lane_change_IOU50 = 0
        junction_crossing_IOU50 = 0
        ltap_IOU50 = 0
        opposite_direction_IOU50 = 0
        rear_end_IOU50 = 0

        attacker_IOU50 = 0
        ego_tp_pred_IOU50 = 0
        ego_tp_pred_average_IOU = 0

        ###########
        plot_atr = False
        plot_target_point_before_regression = False # fixed pos num, e.g. 50
        plot_RCNN_target = False # only target point classification
        plot_regression_based_on_RCNN = True # whole EGO BBOX prediction
        mulit_gt_target_point = True
        only_regression = False 
        EGO_steer_angle = False # seemed no use before 4/2
        plot_most_confident_bbox = True
        RCNN_threshold = 0 #-1 #0.9
        augment = False
        ###########
        # print(self.lr, self.weight_decay)
        # exit()

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                
                batch_size = data.num_graphs
                #print(data.horizon, data.y.shape) #data.y: 11746 // data.y_ttc: 1280
                horizon_sum = 0
                data_gt_list = []
                data_gt_yaw_list = []
                for i in range(batch_size):
                    #print(i, horizon_sum, data.horizon[i], data.y.shape)
                    ###########################################
                    #data.horizon[i] = data.horizon[i] * 10 + 13
                    data.horizon[i] -= 7

                    #data.horizon[i] = 3

                    ###########################################
                    # gt_traj used to validate each_traj
                    #gt_traj = data.future_traj[horizon_sum:(horizon_sum + int(data.horizon[i]))].cpu().view(-1, 2)
                    each_traj = data.y[horizon_sum * 2:(horizon_sum + int(data.horizon[i])) * 2].cpu().view(-1, 2).cumsum(axis=0) #int(data.y[horizon_sum + int(data.horizon[i]) - 1].cpu())
                    each_yaw = int(data.y_yaw[horizon_sum + int(data.horizon[i]) - 1].cpu())
                    horizon_sum += int(data.horizon[i])
                    
                    data_gt_list.append(each_traj)
                    data_gt_yaw_list.append(each_yaw)
                gt_np = np.array(data_gt_list) # 128 
                gt_yaw_np = np.array(data_gt_yaw_list) # 128 
                #gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()
                
                origs = data.orig.numpy()
                #print(data.rot, type(data.rot)) torch.tensor
                rots = data.rot.numpy()
                #print(data.seq_id)
                atr_pos_offset = data.atr_pos_offset.numpy()
                atr_yaw_offset = data.atr_yaw_offset.numpy()

                
                

                #data.seq_id[0] = counter
                #counter += 1
                seq_ids = np.array(data.seq_id)
                for s in range(batch_size):
                    if augment:
                        parts = seq_ids[s].split("_")
                        new_parts = parts[:-2]
                        new_parts.append(parts[-1])
                        seq_ids[s] = "_".join(new_parts)
                #seq_ids = data.seq_id.numpy()

                if gt_np is None:
                    compute_metric = False

                # inference and transform dimension
                if self.multi_gpu:
                    # out = self.model.module(data.to(self.device))
                    pred_target, target_point_pred_pos, offset_pred, yaw_pred, pred_target_point, pred_target_point_yaw, tar_offset_pred, atr_yaw_pred, tp_index, RCNN_cls_result = self.model.inference(data.to(self.device))
                else:
                    pred_target, target_point_pred_pos, offset_pred, yaw_pred, pred_target_point, pred_target_point_yaw, tar_offset_pred, atr_yaw_pred, tp_index, RCNN_cls_result = self.model.inference(data.to(self.device))

                #print(gt_np, "pred:", offset_pred)
                #print("ttc_pred:", ttc_pred.shape) #128 1
                ################################################################
                gt_target = data.candidate_gt.view(-1, data.candidate_len_max[0])
                #pred_offset = offset_pred[gt_target.bool()]
                #print(gt[:, -1, :].shape)
                gt_offset = data.offset_gt.view(-1, 2)
                pred_target_point = pred_target_point.cpu()
                target_point_pred_pos = target_point_pred_pos.cpu()
                #print(pred_target_point)
                #exit()
                pred_target_point_yaw = pred_target_point_yaw.cpu()
                tar_offset_pred = tar_offset_pred.cpu()
                atr_yaw_pred = atr_yaw_pred.cpu()

                n = data.candidate_len_max[0]
                candidate_pos = data.candidate.view(-1, n, 2).cpu()
                candidate_gt = data.candidate_gt.view(-1, n).cpu()
                
                
                
                
                if not RCNN_threshold:
                    RCNN_labels = torch.argmax(RCNN_cls_result, dim=2)
                    RCNN_labels = RCNN_labels.cpu()
                    confidence_scores = RCNN_cls_result[..., 1]
                    confidence_scores = confidence_scores[:, confidence_scores[0]>=0.0]
                    sorted_indices = torch.argsort(confidence_scores, descending=True)
                else:
                    confidence_scores = RCNN_cls_result[..., 1]
                    confidence_scores = confidence_scores[:, confidence_scores[0]>=RCNN_threshold]
                    sorted_indices = torch.argsort(confidence_scores, descending=True)

                    # predicted_labels = torch.where(RCNN_cls_result[..., 1] >= torch.tensor(0.5, torch.tensor(1), torch.tensor(0))
                    RCNN_labels = torch.where(RCNN_cls_result[..., 1] >= torch.tensor(RCNN_threshold, device=self.device),
                                            torch.tensor(1, device=self.device), torch.tensor(0, device=self.device)).cpu()
                    # print(RCNN_labels, torch.argmax(RCNN_cls_result, dim=2))
                # pred_index_tuple = np.where(RCNN_labels[0] == 1)
                # pred_index_tuple_ = np.where(torch.argmax(RCNN_cls_result, dim=2).cpu()[0] == 1)
                # print(len(pred_index_tuple[0]), len(pred_index_tuple_[0]) )
                # exit()


                if EGO_steer_angle:
                    ego_yaw_last_frame_list = [data.obs_yaw[i].cpu().numpy().item() * np.pi / 180 for i in range(7, data.obs_yaw.shape[0], 8)]
                    ego_yaw_last_frame_list = [angle + 2*np.pi if angle < 0 else angle for angle in ego_yaw_last_frame_list]
                    ego_yaw_last_frame_list = np.tile(np.array(ego_yaw_last_frame_list).reshape(len(ego_yaw_last_frame_list), 1),(1, 50)).reshape(len(ego_yaw_last_frame_list), 50, 1) # 256, 50, 1
                    # print(max(pred_target_point_yaw[:, 0, 0]), min(pred_target_point_yaw[:, 0, 0]))
                    # print(max(ego_yaw_last_frame_list[:, 0, 0]), min(ego_yaw_last_frame_list[:, 0, 0]))
                    pred_target_point_yaw_origin = ego_yaw_last_frame_list
                    pred_target_point_yaw += ego_yaw_last_frame_list
                    pred_target_point_yaw[pred_target_point_yaw > 6.28] -= 6.28
                    # print(max(pred_target_point_yaw[:, 0, 0]), min(pred_target_point_yaw[:, 0, 0]))

                #############################################
                #atr_yaw_pred = atr_yaw_pred * np.pi / 180
                #############################################

                #print(tar_offset_pred.shape, atr_yaw_pred.shape)
                #pred_yaw = yaw_pred[gt_target.bool()].cpu()
                #print(pred_target.shape, pred_yaw.shape)
                #yaw_gt = data.y_yaw.unsqueeze(1).view(batch_size, -1, 1).cpu().numpy()[:, -1]#.unsqueeze(1)
                # print(pred_target.shape, gt_target.shape) 128, 1545
                # print(pred_offset.shape, gt_offset.shape) 128, 2
                # print(pred_yaw.shape, yaw_gt.shape) 128, 1
                #print(data.candidate.view(-1, data.candidate_len_max[0], 2).shape) 128, 1545, 2
                # sys.exit()
                _, indices = pred_target.topk(1, dim=1)
                batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(1)]).T
                # target_pred_se, offset_pred_se = data.candidate.view(-1, data.candidate_len_max[0], 2)[batch_idx, indices].cpu(), offset_pred[batch_idx, indices].cpu()
                #print(target_pred_se.shape, offset_pred_se.shape) 128, 1, 2
                _, indices_GT = data.candidate_gt.view(-1, data.candidate_len_max[0]).topk(1, dim=1) #128, 1748 -> 128, 1
                #print("PRED:", indices.shape, indices) #128, 1
                batch_cls_acc = accuracy_score(indices_GT.cpu(), indices.cpu())
                all_cls_acc += batch_cls_acc
                batch_num += 1

                batch_top5_cls_acc = top_k_accuracy_score(indices_GT.flatten().cpu().numpy(), pred_target.cpu().numpy(), k=5, labels=np.arange(data.candidate_len_max[0].cpu()))
                _, indices_5 = pred_target.topk(5, dim=1)
                all_top5_cls_acc += batch_top5_cls_acc

                #print(data.tp_candidate.shape) #9088, 2
                gt_attacker_list = []
                #all_data_num += len(seq_ids)
                for s in range(batch_size):
                    # sce_df = pd.read_csv('nuscenes_data/padding_trajectory/' + split + '/' + seq_ids[s] + '.csv')
                    sce_df = pd.read_csv('nuscenes_data/trajectory/' + split + '/' + seq_ids[s] + '.csv')
                    #print(len(keys), keys.index(seq_ids[s].split('.')[1].split('_')[-1]))
                    objs = sce_df.groupby(['TRACK_ID']).groups
                    keys = list(objs.keys())
                    del keys[keys.index('ego')]
                    # if seq_ids[s].split('.')[1].split('_')[-1] in keys:
                    #     gt_attacker_list.append([keys.index(seq_ids[s].split('.')[1].split('_')[-1])])
                    if seq_ids[s].split('_')[8] in keys:
                        gt_attacker_list.append([keys.index(seq_ids[s].split('_')[8])])
                    else:
                        gt_attacker_list.append([100])
                batch_tp_cls_acc = accuracy_score(np.array(tp_index.cpu()), np.array(gt_attacker_list))           
                all_tp_cls_acc += batch_tp_cls_acc

                

                # print(pred_target[0].shape, pred_target[0].topk(1, dim=0), data.candidate.view(-1, data.candidate_len_max[0], 2)[0][1012])
                # for i in range(data.candidate.view(-1, data.candidate_len_max[0], 2).shape[1]):
                #     print(data.candidate.view(-1, data.candidate_len_max[0], 2)[0][i])
                    
                ################################################################

                ## future traj gt (for plot): gt_np
                ## future traj pred (for plot): {Quintic}
                ## target point gt: x y->  gt_np[-1] // yaw-> gt_yaw_np
                ## target point pred: x y->  target_pred_se + offset_pred_se // yaw-> pred_yaw
                ## ttc gt: data.horizon
                ## ttc pred: ttc_pred

                if not only_1_batch_flag:
                    # only change this flag to test 1 batch
                    only_1_batch_flag = 0
                    for s in range(batch_size):
                        if plot_target_point_before_regression or plot_regression_based_on_RCNN:
                            ego, = plt.plot([0, 0], [0, 0], '-',
                                            color='black', markersize=1)
                            ego_prediction, = plt.plot([0, 0], [0, 0], '-',
                                            color='red', markersize=1)
                            ego_prediction_proposal, = plt.plot([0, 0], [0, 0], '-',
                                                color='orange', markersize=1)
                            plt.legend([ego, ego_prediction, ego_prediction_proposal], [
                                    "ego GT", "final prediction", "proposal prediction"])
                        if plot_RCNN_target:
                            target_candidate, = plt.plot([0, 0], [0, 0], '-',
                                            color='black', markersize=1)
                            prediction_proposal, = plt.plot([0, 0], [0, 0], '-',
                                            color='red', markersize=1)
                            gt_proposal, = plt.plot([0, 0], [0, 0], '-',
                                                color='cyan', markersize=1)
                            plt.legend([target_candidate, prediction_proposal, gt_proposal], [
                                    "Target candidate", "Prediction proposal", "GT proposal"])
                        # print("GT:", data.offset_gt.view(-1, 2).shape)
                        # print("Pred:", offset_pred[:,0,:].shape)
                        # exit()
                        
                        ###### 180 degree for ego yaw ######
                        # gt_yaw_np[s] = gt_yaw_np[s] + np.pi
                        pred_target_point_yaw[s][0][0] = pred_target_point_yaw[s][0][0] - np.pi
                        #print(pred_target_point_yaw[s][0][0] * 180 / np.pi, gt_yaw_np[s])
                        ###### 180 degree for ego yaw ######
                        
                        candidate_pos = data.candidate.view(-1, n, 2).cpu()
                        
                        candidate_pos[s] = self.convert_coord(candidate_pos[s], origs[s], rots[s])
                        gt_index_tuple = np.where(candidate_gt[s] == 1)
                        #print("GT positive:", len(gt_index_tuple[0]))
                        gt_positive_proposal_num.append(len(gt_index_tuple[0]))



                        
                        
                        gt_target_point = candidate_pos[s][gt_index_tuple, :]
                        pred_index_tuple = np.where(RCNN_labels[s] == 1)

                        
                        # pred_RCNN_target_point = np.zeros((candidate_pos.shape))
                        
                        # positive_num: how many positive proposal predict
                        positive_num = candidate_pos[s][pred_index_tuple, :].shape[1]
                        positive_num_list.append(positive_num)
                        
                        # If plot biggest confident bbox
                        if plot_most_confident_bbox:
                            pred_RCNN_target_point_copy = torch.zeros((1, 1, 2)) #torch.zeros((batch_size, 1, 2))
                            pred_RCNN_target_point_copy[0][0] = candidate_pos[s][sorted_indices[s][0], :]
                            pred_RCNN_target_point_offset = offset_pred[s][sorted_indices[s][0], :].unsqueeze(0).unsqueeze(0).cpu()
                            pred_target_point[s] = pred_RCNN_target_point_copy
                            pred_target_point[s][:, 0] += pred_RCNN_target_point_offset[0][0][0]
                            pred_target_point[s][:, 1] += pred_RCNN_target_point_offset[0][0][1]
                        
                        pred_RCNN_target_point = candidate_pos[s][pred_index_tuple, :]
                        pred_RCNN_target_point_offset = offset_pred[s][pred_index_tuple, :].cpu()

                        #print(candidate_pos.shape, gt_target_point.shape, pred_RCNN_target_point.shape)
                        # print(sorted_indices, pred_index_tuple, pred_RCNN_target_point)
                        # exit()
                        if plot_RCNN_target:
                            for point_index in range(candidate_pos.shape[0]):
                                plt.scatter(candidate_pos[point_index][0], candidate_pos[point_index][1], s=5, c="black")
                            for point_index in range(gt_target_point.shape[1]): #1,50,2
                                plt.scatter(gt_target_point[0][point_index][0], gt_target_point[0][point_index][1], s=5, c="cyan", marker="s")
                        
                        if positive_num != 0:
                            if plot_RCNN_target:
                                for point_index in range(positive_num):
                                    plt.scatter(pred_RCNN_target_point[0][point_index][0] + 1, pred_RCNN_target_point[0][point_index][1] + 1, s=10, c="red", marker="s")
                            if plot_regression_based_on_RCNN:
                                alpha_value = 0.3
                                for point_index in range(positive_num):
                                    ego_target_point_pred_rec = [pred_RCNN_target_point[0][point_index][0], pred_RCNN_target_point[0][point_index][1],
                                    vehicle_width, vehicle_length, pred_target_point_yaw[s][0][0] + 90 * np.pi / 180]
                                    ego_target_point_pred_polygon = self.build_polygon(ego_target_point_pred_rec, 'orange', alpha_value)
                                    #final_pred_pos_x = pred_RCNN_target_point[0][point_index][0] + pred_RCNN_target_point_offset[0][point_index][0]
                                    #final_pred_pos_y = pred_RCNN_target_point[0][point_index][1] + pred_RCNN_target_point_offset[0][point_index][1]
                                    dx = pred_RCNN_target_point_offset[0][point_index][0]
                                    dy = pred_RCNN_target_point_offset[0][point_index][1]
                                    plt.arrow(pred_RCNN_target_point[0][point_index][0], pred_RCNN_target_point[0][point_index][1], dx, dy, head_width = 0.1, color = 'cyan')
                                    
                                    final_pred_rec = [pred_RCNN_target_point[0][point_index][0] + dx, pred_RCNN_target_point[0][point_index][1] + dy,
                                    vehicle_width, vehicle_length, pred_target_point_yaw[s][0][0] + 90 * np.pi / 180]
                                    final_pred_polygon = self.build_polygon(final_pred_rec, 'orange', alpha_value)

                            #print(pred_index_tuple[0], gt_index_tuple[0])
                            if only_regression:
                                accuracy, recall, precision, f1, TP, FP, FN = 1, 1, 1 ,1 ,1 ,1, 1
                            else:
                                accuracy, recall, precision, f1, TP, FP, FN = self.calculate_F1(pred_index_tuple[0], gt_index_tuple[0])
                            all_TP += TP
                            all_FP += FP
                            all_FN += FN
                            precision_list.append(precision)
                            recall_list.append(recall)
                            f1_list.append(f1)
                        else:
                            FN = len(gt_index_tuple[0])
                            print("No guess, FN: ", FN)
                            no_guess_list.append(FN)
                            all_FP += 0
                            all_TP += 0
                            all_FN += FN
                            precision_list.append(0)
                            recall_list.append(0)
                            f1_list.append(0)


                        all_data_num += 1
                        now_scenario = seq_ids[s]
                        #print("GT before rot:", gt_np[s], origs[s], rots[s])
                        gt_trajectories[now_scenario] = self.convert_coord(gt_np[s], origs[s], rots[s])
                        plt.plot(gt_trajectories[now_scenario][:, 0], gt_trajectories[now_scenario][:, 1], '-', color='black')
                        split_name = now_scenario.split('_')
                        # initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[5] + '_' + split_name[6]
                        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
                        lane_feature = np.load('nuscenes_data/initial_topology/' + initial_name + '.npy', allow_pickle=True)
                        for features in lane_feature:
                            xs, ys = np.vstack((features[0][:, :2], features[0][-1, 3:5]))[
                                :, 0], np.vstack((features[0][:, :2], features[0][-1, 3:5]))[:, 1]
                            plt.plot(xs, ys, '-', color='lightgray')
                        # sce_df = pd.read_csv('nuscenes_data/padding_trajectory/' + split + '/' + now_scenario + '.csv')
                        sce_df = pd.read_csv('nuscenes_data/trajectory/' + split + '/' + now_scenario + '.csv')
                        # vehicle_num = len(set(sce_df.TRACK_ID.values))
                        
                        
                        #print(lane_feature)
                        #route_num = now_scenario.split('_')[0]
                        #gt_ttc = now_scenario.split('_')[-2]
                        objs = sce_df.groupby(['TRACK_ID']).groups
                        keys = list(objs.keys())
                        del keys[keys.index('ego')]
                        
                        #print(now_scenario, tp_index[s].cpu().numpy()[0], data.horizon[s].cpu().numpy())
                        
                        #print(tp_index[s].cpu().numpy(), keys)
                        if keys[tp_index[s].cpu().numpy()[0]] in keys:
                            guess_attacker_id = keys[tp_index[s].cpu().numpy()[0]]
                        elif len(keys) != 0:
                            ### random guess
                            guess_attacker_id = keys[0]
                        else:
                            continue

                        ##################################
                        # ttc = int(float(now_scenario.split('_')[7].split('-')[2]) * 10 + 13)
                        ttc = int(now_scenario.split('_')[7].split('-')[-1])
                        ##################################
                        condition = now_scenario.split('_')[5]
                        #attacker_id = now_scenario.split('.')[1].split('_')[-1]
                        attacker_id = now_scenario.split('_')[8]
                        right_guess_tp = 0
                        if str(attacker_id) == str(guess_attacker_id):
                            right_guess_tp = 1
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
                            ideal_yaw_offset = 15
                        #lane_feature = np.load("/home/yoyo/sgan/initial_scenario/agents_4/" + "RouteScenario_" + route_num + "_to_" + route_num + "/topology/7.npy", allow_pickle=True)

                        
                        
                        # angle = np.rad2deg(angle_vectors(data.rot[s], [1, 0]))
                        
                        #if now_scenario.split('_')[0] != '597':
                        #    continue
                        
                        #ego_rot = np.rad2deg(angle_vectors(data.rot[s][0].cpu().numpy(), [0, 1]))
                        ego_rot = math.acos(np.linalg.inv(rots[s])[0][0]) * 180 / np.pi

                        #for i in range(k):
                        #    plt.plot(gt_np[s][:, 0], gt_np[s][:, 1], '-', color='gray')
                        
                        # seq_id = seq_ids[s][0]
                        # now_pred_attacker = tp_index[s][0].cpu().numpy()[0]
                        # now_gt_attacker = seq_id.split('_')[-1]
                        # print(now_gt_attacker, now_pred_attacker)
                        # sys.exit()


                        #print(pred_target_point[s])
                        
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            if str(track_id) == 'ego':
                                continue
                            #print(track_id, guess_attacker_id, attacker_id)
                            if str(track_id) == str(guess_attacker_id) and str(track_id) == str(attacker_id):
                                #print("RIGHT")
                                # ego_rec = [remain_df.X.values[8], remain_df.Y.values[8], vehicle_width
                                #         , vehicle_length, (remain_df.YAW.values[8] + 90.0) * np.pi / 180]
                                ego_rec = [remain_df.X.values[8], remain_df.Y.values[8], vehicle_width
                                        , vehicle_length, (remain_df.YAW.values[8] + 90.0) * np.pi / 180]
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
                                # plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                                #     y_1, y_2, y_4, y_3, y_1], '-.',  color='purple', markersize=3)
                            else:
                                ego_rec = [remain_df.X.values[8], remain_df.Y.values[8], vehicle_width
                                        , vehicle_length, (remain_df.YAW.values[8] + 90.0) * np.pi / 180]
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
                                if plot_atr:
                                    plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                                        y_1, y_2, y_4, y_3, y_1], ':',  color='dimgray', markersize=3)
                        #print(pred_target_point[s], offset_pred[s])
                        
                        # beacause already convert when most confident setting
                        if not plot_most_confident_bbox:
                            pred_target_point[s] = self.convert_coord(pred_target_point[s], origs[s], rots[s])
                        
                        target_point_pred_pos[s] = self.convert_coord(target_point_pred_pos[s], origs[s], rots[s])
                        #print("After rot pred:", pred_target_point[s])
                        #print(atr_pos_offset[s])
                        atr_pos_offset[s] = np.matmul(np.linalg.inv(rots[s]), atr_pos_offset[s].T).T
                        #print(atr_pos_offset[s])
                        #sys.exit()
                        tar_offset_pred[s] = np.matmul(np.linalg.inv(rots[s]), tar_offset_pred[s].T).T
                        #for i in range(k):
                        #print(gt_trajectories[now_scenario][:, 0], gt_trajectories[now_scenario][:, 1])
                        
                        
                        # print(math.acos(np.linalg.inv(rots[s])[0][0]))
                        # print(math.asin(np.linalg.inv(rots[s])[1][0]))
                        #print(ego_rot)
                        

                        # print("gt:", gt_trajectories[now_scenario][-1])
                        # print("pred:", target_pred_se[s][0][0] + offset_pred_se[s][0][0] + origs[s][0], target_pred_se[s][0][1] + offset_pred_se[s][0][1] + origs[s][1])
                        # print(gt_trajectories[now_scenario])
                        # print("origs:", target_pred_se[s][0], offset_pred_se[s][0], origs[s])
                        
                        # last pos of traj
                        #print("GT:", gt_np[s][-1][0], gt_np[s][-1][1], origs[s])
                        #print(target_pred_se.shape, target_pred_se)
                        #print("Pred:", target_pred_se[s][0][0] + offset_pred_se[s][0][0], target_pred_se[s][0][1] + offset_pred_se[s][0][1])
                        # ego_rec = [gt_np[s][-1][0], gt_np[s][-1][1], vehicle_width
                        #             , vehicle_length, (gt_yaw_np[s]) * np.pi / 180]
                        ego_rec = [gt_trajectories[now_scenario][-1][0], gt_trajectories[now_scenario][-1][1], vehicle_width
                                    , vehicle_length, (gt_yaw_np[s] + 90.0) * np.pi / 180]
                        # print(pred_RCNN_target_point[s][0][0] + 1, pred_RCNN_target_point[s][0][1] + 1)
                        # print(gt_trajectories[now_scenario][-1][0], gt_trajectories[now_scenario][-1][1])
                        # exit()
                        # ego_rec = [gt[s][-1][0], gt[s][-1][1], vehicle_width
                        #             , vehicle_length, (np.array(yaw_gt)[s][0] + ego_rot) * np.pi / 180]
                        # ego_rec = [gt_trajectories[now_scenario][-1][0], gt_trajectories[now_scenario][-1][1], vehicle_width
                        #             , vehicle_length, (np.array(yaw_gt)[s][0] + ego_rot) * np.pi / 180]
                        # maybe need real target + offset?
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
                        plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                            y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=5)
                        ego_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])

                        if data.cross[s] < 0:
                            atr_yaw_offset[s] = abs(atr_yaw_offset[s] - 360.0)
                            #print(atr_yaw_offset[s])
                        else:
                            atr_yaw_offset[s] = atr_yaw_offset[s]

                        attacker_gt_rec = [gt_trajectories[now_scenario][-1][0] + atr_pos_offset[s][0], gt_trajectories[now_scenario][-1][1] + atr_pos_offset[s][1], vehicle_width
                                    , vehicle_length, (gt_yaw_np[s] + 90.0 + atr_yaw_offset[s]) * np.pi / 180]
                        x_1 = float(np.cos(
                            attacker_gt_rec[4])*(-attacker_gt_rec[2]/2) - np.sin(attacker_gt_rec[4])*(-attacker_gt_rec[3]/2) + attacker_gt_rec[0])
                        x_2 = float(np.cos(
                            attacker_gt_rec[4])*(attacker_gt_rec[2]/2) - np.sin(attacker_gt_rec[4])*(-attacker_gt_rec[3]/2) + attacker_gt_rec[0])
                        x_3 = float(np.cos(
                            attacker_gt_rec[4])*(-attacker_gt_rec[2]/2) - np.sin(attacker_gt_rec[4])*(attacker_gt_rec[3]/2) + attacker_gt_rec[0])
                        x_4 = float(np.cos(
                            attacker_gt_rec[4])*(attacker_gt_rec[2]/2) - np.sin(attacker_gt_rec[4])*(attacker_gt_rec[3]/2) + attacker_gt_rec[0])
                        y_1 = float(np.sin(
                            attacker_gt_rec[4])*(-attacker_gt_rec[2]/2) + np.cos(attacker_gt_rec[4])*(-attacker_gt_rec[3]/2) + attacker_gt_rec[1])
                        y_2 = float(np.sin(
                            attacker_gt_rec[4])*(attacker_gt_rec[2]/2) + np.cos(attacker_gt_rec[4])*(-attacker_gt_rec[3]/2) + attacker_gt_rec[1])
                        y_3 = float(np.sin(
                            attacker_gt_rec[4])*(-attacker_gt_rec[2]/2) + np.cos(attacker_gt_rec[4])*(attacker_gt_rec[3]/2) + attacker_gt_rec[1])
                        y_4 = float(np.sin(
                            attacker_gt_rec[4])*(attacker_gt_rec[2]/2) + np.cos(attacker_gt_rec[4])*(attacker_gt_rec[3]/2) + attacker_gt_rec[1])
                        if plot_atr:
                            plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                                y_1, y_2, y_4, y_3, y_1], '-',  color='brown', markersize=3)
                        attacker_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
                        
                        # _, indices = target_prob.topk(self.m, dim=1)
                        # batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.m)]).T
                        # target_pred_se, offset_pred_se = target_candidate[batch_idx, indices], offset_pred[batch_idx, indices]

                        ego_rec = [pred_target_point[s][0][0],
                                pred_target_point[s][0][1], vehicle_width,
                                vehicle_length, pred_target_point_yaw[s][0][0] + 90 * np.pi / 180]
                        # maybe need real target + offset?
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
                        plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                            y_1, y_2, y_4, y_3, y_1], '-',  color='red', markersize=4)
                        pred_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])

                        if plot_target_point_before_regression:
                            ego_target_point_pred_rec = [target_point_pred_pos[s][0][0], target_point_pred_pos[s][0][1],
                                                            vehicle_width, vehicle_length, pred_target_point_yaw[s][0][0] + 90 * np.pi / 180]
                            ego_target_point_pred_polygon = self.build_polygon(ego_target_point_pred_rec, 'green')
                            dx = pred_target_point[s][0][0] - target_point_pred_pos[s][0][0]
                            dy = pred_target_point[s][0][1] - target_point_pred_pos[s][0][1]
                            plt.arrow(target_point_pred_pos[s][0][0], target_point_pred_pos[s][0][1], dx, dy, head_width = 0.1, color = 'cyan')



                        ### check left or right using cross between ego and attacker
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            if str(track_id) == str(guess_attacker_id):
                                # tp_start_x = remain_df.X.values[8]
                                # tp_start_y = remain_df.Y.values[8]
                                tp_start_x = remain_df.X.values[-3]
                                tp_start_y = remain_df.Y.values[-3]
                            elif str(track_id) == 'ego':
                                # ego_start_x = remain_df.X.values[8]
                                # ego_start_y = remain_df.Y.values[8]
                                ego_start_x = remain_df.X.values[-3]
                                ego_start_y = remain_df.Y.values[-3]
                        cross_ego_vec = np.array([pred_target_point[s][0][0] - ego_start_x, pred_target_point[s][0][1]- ego_start_y])
                        cross_tp_vec = np.array([tp_start_x - ego_start_x, tp_start_y- ego_start_y])
                        cross = np.cross(cross_tp_vec, cross_ego_vec)
                        if cross < 0:
                            yaw_offset = abs(atr_yaw_pred[s] - 6.28)
                        else:
                            yaw_offset = atr_yaw_pred[s]

                        # attacker_pred_rec = [pred_target_point[s][0][0] + tar_offset_pred[s][0], pred_target_point[s][0][1] + tar_offset_pred[s][1], vehicle_width
                        #             , vehicle_length, pred_target_point_yaw[s][0][0] + (90.0 + atr_yaw_pred[s]) * np.pi / 180]
                        attacker_pred_rec = [pred_target_point[s][0][0] + tar_offset_pred[s][0], pred_target_point[s][0][1] + tar_offset_pred[s][1], vehicle_width
                                    , vehicle_length, pred_target_point_yaw[s][0][0] + yaw_offset + (90.0) * np.pi / 180]

                        x_1 = float(np.cos(
                            attacker_pred_rec[4])*(-attacker_pred_rec[2]/2) - np.sin(attacker_pred_rec[4])*(-attacker_pred_rec[3]/2) + attacker_pred_rec[0])
                        x_2 = float(np.cos(
                            attacker_pred_rec[4])*(attacker_pred_rec[2]/2) - np.sin(attacker_pred_rec[4])*(-attacker_pred_rec[3]/2) + attacker_pred_rec[0])
                        x_3 = float(np.cos(
                            attacker_pred_rec[4])*(-attacker_pred_rec[2]/2) - np.sin(attacker_pred_rec[4])*(attacker_pred_rec[3]/2) + attacker_pred_rec[0])
                        x_4 = float(np.cos(
                            attacker_pred_rec[4])*(attacker_pred_rec[2]/2) - np.sin(attacker_pred_rec[4])*(attacker_pred_rec[3]/2) + attacker_pred_rec[0])
                        y_1 = float(np.sin(
                            attacker_pred_rec[4])*(-attacker_pred_rec[2]/2) + np.cos(attacker_pred_rec[4])*(-attacker_pred_rec[3]/2) + attacker_pred_rec[1])
                        y_2 = float(np.sin(
                            attacker_pred_rec[4])*(attacker_pred_rec[2]/2) + np.cos(attacker_pred_rec[4])*(-attacker_pred_rec[3]/2) + attacker_pred_rec[1])
                        y_3 = float(np.sin(
                            attacker_pred_rec[4])*(-attacker_pred_rec[2]/2) + np.cos(attacker_pred_rec[4])*(attacker_pred_rec[3]/2) + attacker_pred_rec[1])
                        y_4 = float(np.sin(
                            attacker_pred_rec[4])*(attacker_pred_rec[2]/2) + np.cos(attacker_pred_rec[4])*(attacker_pred_rec[3]/2) + attacker_pred_rec[1])
                        if plot_atr:
                            plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                                y_1, y_2, y_4, y_3, y_1], '-',  color='blue', markersize=3)
                        attacker_pred_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])

                        # print("Pred:", tar_offset_pred[s])
                        # print("GT:", atr_pos_offset[s])

                        
                        

                        
                        now_iou = ego_polygon.intersection(pred_polygon).area / ego_polygon.union(pred_polygon).area
                        attacker_iou = attacker_polygon.intersection(attacker_pred_polygon).area / attacker_polygon.union(attacker_pred_polygon).area
                        together_gt_iou = ego_polygon.intersection(attacker_polygon).area / ego_polygon.union(attacker_polygon).area
                        together_pred_iou = pred_polygon.intersection(attacker_pred_polygon).area / pred_polygon.union(attacker_pred_polygon).area
                        # together_yaw_distance = (atr_yaw_pred[s].cpu().numpy() * 180 / np.pi + 360.0) if atr_yaw_pred[s].cpu().numpy() < 0 else atr_yaw_pred[s].cpu().numpy() * 180 / np.pi
                        # together_yaw_distance = abs(together_yaw_distance - 360.0) if together_yaw_distance > 180 else together_yaw_distance
                        together_yaw_distance = (yaw_offset * 180 / np.pi + 360.0) if yaw_offset < 0 else yaw_offset * 180 / np.pi
                        together_yaw_distance = abs(together_yaw_distance - 360.0) if together_yaw_distance > 180 else together_yaw_distance
                        #print(pred_target_point_yaw[s][0][0].cpu().numpy() * 180 / np.pi, gt_yaw_np[s])
                        #print("Pred yaw dist:", together_yaw_distance)
                        ideal_yaw_dist = abs(ideal_yaw_offset - together_yaw_distance[0])


                        ideal_yaw_dist_average += ideal_yaw_dist
                        if ideal_yaw_dist < 10:
                            tolerance_degree_10 += 1
                        if ideal_yaw_dist < 20:
                            tolerance_degree_20 += 1
                        if ideal_yaw_dist < 30:
                            tolerance_degree_30 += 1
                            if condition == 'LTAP': 
                                LTAP_num += 1
                            elif condition == 'JC':
                                junction_crossing_num += 1
                            elif condition == 'HO':
                                opposite_direction_num += 1
                            elif condition == 'RE':
                                rear_end_num += 1
                            elif condition == 'LC':
                                lane_change_num += 1
                        #now_iou = get_iou([gt_trajectories[now_scenario][-1][0], gt_trajectories[now_scenario][-1][1], vehicle_width, vehicle_length],
                        #                      [pred_target_point[s][0][0],
                        #        pred_target_point[s][0][1], vehicle_width,vehicle_length])
                        
                        
                        
                        # target_candiate = data.candidate.view(-1, data.candidate_len_max[0], 2)
                        # _, indices_5 = pred_target[s].topk(5, dim=0)
                        # now_scenario_padding_number = 0
                        # for now_t in range(target_candiate[s][indices_5].shape[0]):
                        #     if target_candiate[s][indices_5][now_t][0] == 0 and target_candiate[s][indices_5][now_t][1] == 0:
                        #         now_scenario_padding_number += 1
                        # padding_number_list.append(now_scenario_padding_number)
                        
                        plot_other_ego_bbox = False
                        if self.m > 1 and plot_other_ego_bbox:
                            for t in range(1, self.m):
                                alpha_value = 1-0.2*t
                                ego_rec = [pred_target_point[s][t][0],
                                        pred_target_point[s][t][1], vehicle_width,
                                        vehicle_length, pred_target_point_yaw[s][t][0] + 90 * np.pi / 180]# + 90 * np.pi / 180]
                                #print(ego_rec)
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
                                plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                                    y_1, y_2, y_4, y_3, y_1], '-', alpha=alpha_value,  color='red', markersize=3)
                                if plot_target_point_before_regression:                                    
                                    ego_target_point_pred_rec = [target_point_pred_pos[s][t][0], target_point_pred_pos[s][t][1],
                                                            vehicle_width, vehicle_length, pred_target_point_yaw[s][0][0] + 90 * np.pi / 180]
                                    ego_target_point_pred_polygon = self.build_polygon(ego_target_point_pred_rec, "orange", alpha_value)
                                    dx = pred_target_point[s][t][0] - target_point_pred_pos[s][t][0]
                                    dy = pred_target_point[s][t][1] - target_point_pred_pos[s][t][1]
                                    plt.arrow(target_point_pred_pos[s][t][0], target_point_pred_pos[s][t][1], dx, dy, head_width = 1, alpha=alpha_value, color = 'cyan')

                        
                        start_x = gt_trajectories[now_scenario][0][0].cpu()
                        start_y = gt_trajectories[now_scenario][0][1].cpu()
                        start_yaw = gt_yaw_np[s] * np.pi / 180 #(gt_yaw_np[s] + 90.0) * np.pi / 180
                        
                        sa = 0.1
                        ga = 0.0
                        max_accel = 100.0 
                        max_jerk = 10.0
                        #######################
                        dt = 0.5
                        #######################
                        # min_t = ttc_pred[s].cpu().numpy()[0] / 10
                        # max_t = min_t + dt
                        gt_ttc = ttc - 8###data.horizon[s].cpu().numpy()
                        min_t = gt_ttc * dt - dt / 2
                        max_t = min_t + 0.1
                        # gx = pred_target_point[s][0][0]
                        # gy = pred_target_point[s][0][1]
                        # gyaw = pred_target_point_yaw[s][0][0]# + 90 * np.pi / 180# + ego_rot * np.pi / 180
                        # constant_v = math.sqrt((gx - start_x) ** 2 + (gy - start_y) ** 2) / min_t
                        # sv = 4
                        # gv = constant_v
                        # time, x, y, all_yaw, all_v, a, j = quintic_polynomials_planner(
                        #     start_x, start_y, start_yaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt, min_t, max_t)
                        # pos = np.array((x, y)).T
                        # ego_yaw_v = np.array((all_v, all_yaw)).T #[1:]
                        #print(start_x, start_y, pos[0]) # Same
                        
                        # attacker quintic
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            if str(track_id) == str(guess_attacker_id):
                                tp_start_x = remain_df.X.values[8]
                                tp_start_y = remain_df.Y.values[8]
                                tp_start_yaw = remain_df.YAW.values[8] * np.pi / 180
                        tp_gx = pred_target_point[s][0][0] + tar_offset_pred[s][0]
                        tp_gy = pred_target_point[s][0][1] + tar_offset_pred[s][1]
                        # tp_gyaw = pred_target_point_yaw[s][0][0] + atr_yaw_pred[s]
                        tp_gyaw = pred_target_point_yaw[s][0][0] + yaw_offset  
                        tp_constant_v = math.sqrt((tp_gx - tp_start_x) ** 2 + (tp_gy - tp_start_y) ** 2) / min_t
                        tp_sv = 4
                        tp_gv = tp_constant_v
                        tp_time, tp_x, tp_y, tp_all_yaw, tp_all_v, tp_a, tp_j = quintic_polynomials_planner(
                            tp_start_x, tp_start_y, tp_start_yaw, tp_sv, sa, tp_gx, tp_gy, tp_gyaw, tp_gv, ga, max_accel, max_jerk, dt, min_t, max_t)
                        tp_pos = np.array((tp_x, tp_y)).T
                        tp_yaw_v = np.array((tp_all_v, tp_all_yaw)).T
                        #print("pos:", pos.shape)
                        tp_pos = tp_pos
                        vehicle_list = []
                        # print(remain_df)
                        # print(tp_pos, tp_gx, tp_gy, tp_start_x, tp_start_y)
                        

                        
                        ##################### initial scenario is 4 frames
                        collision_moment = sce_df.TIMESTAMP.values[8+gt_ttc]
                        sce_df = sce_df[(sce_df.TIMESTAMP <= collision_moment)]
                        #####################
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            
                            # if str(track_id) == 'ego':
                            #     remain_df.iloc[8:, 3] = pd_pos[:, 0]
                            #     remain_df.iloc[8:, 4] = pd_pos[:, 1]
                            #     remain_df.iloc[8:, 2] = ego_yaw_v[:, 0]
                            #     remain_df.iloc[8:, 5] = ego_yaw_v[:, 1]
                            if str(track_id) == str(guess_attacker_id):
                                #print(len(remain_df.X.values), tp_pos.shape, 8+gt_ttc)
                                remain_df.iloc[8:, 3] = tp_pos[:, 0]
                                remain_df.iloc[8:, 4] = tp_pos[:, 1]
                                remain_df.iloc[8:, 2] = tp_yaw_v[:, 0]
                                remain_df.iloc[8:, 5] = tp_yaw_v[:, 1] * 180 / np.pi
                            vehicle_list.append(remain_df)
                        traj_df = pd.concat(vehicle_list)
                        traj_df.to_csv('output_csv/' + now_scenario + '.csv', index=False)
                        frame_num = len(set(traj_df.TIMESTAMP.values))
                        further_list = []
                        further_df = traj_df
                        ########### moving foward by quintic for collision checking ###########
                        # for track_id, remain_df in traj_df.groupby("TRACK_ID"):
                        #     dis_x = remain_df.iloc[frame_num-1, 3] - remain_df.iloc[frame_num-2, 3]
                        #     dis_y = remain_df.iloc[frame_num-1, 4] - remain_df.iloc[frame_num-2, 4]
                        #     gx = remain_df.iloc[frame_num-1, 3]
                        #     gy = remain_df.iloc[frame_num-1, 4]
                        #     gv = remain_df.iloc[frame_num-1, 2]
                        #     gyaw = remain_df.iloc[frame_num-1, 5] * np.pi / 180
                        #     steps = 4
                        #     new_dt = 0.125
                        #     origin_dt = 0.5
                        #     new_gx = gx + dis_x * steps
                        #     new_gy = gy + dis_y * steps
                        #     new_min_t = origin_dt * steps
                        #     new_max_t = new_min_t + origin_dt
                        #     time, x, y, all_yaw, all_v, a, j = quintic_polynomials_planner(
                        #         gx, gy, gyaw, gv, ga, new_gx, new_gy, gyaw, gv, ga, max_accel, max_jerk, new_dt, new_min_t, new_max_t)
                        #     further_pos = np.array((x, y)).T
                        #     for further_t in range(len(x)):
                        #         # b = {'TIMESTAMP': remain_df.TIMESTAMP.values[-1] + (further_t + 1) * new_dt, 'TRACK_ID': track_id,
                        #         #                         'OBJECT_TYPE': remain_df.OBJECT_TYPE.values[0], 'X': x[further_t].cpu().numpy(), 'Y': y[further_t].cpu().numpy(),
                        #         #                         'YAW': all_yaw[further_t], 'V': all_v[further_t], 'CITY_NAME': remain_df.CITY_NAME.values[0]}
                        #         # further_df = further_df.append(b, ignore_index=True)
                        #         b = {'TIMESTAMP': [remain_df.TIMESTAMP.values[-1] + (further_t + 1) * new_dt * 500000], 'TRACK_ID': [track_id],
                        #             # 'V': [all_v[further_t]], 'X': [x[further_t].cpu().numpy()], 'Y': [y[further_t].cpu().numpy()],
                        #             'V': [all_v[further_t]], 'X': [x[further_t]], 'Y': [y[further_t]],
                        #             'YAW': [all_yaw[further_t] * 180 / np.pi]}
                        #         df_insert = pd.DataFrame(b)
                        #         further_df = pd.concat([further_df, df_insert], ignore_index=True)
                        # further_df.to_csv('output_csv_moving_foward/' + now_scenario + '_foward.csv', index=False)
                        # collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = self.cal_cr_and_similarity(further_df, attacker_id)
                        ########### moving foward for collision checking ###########

                        ### Interpolation for moving foward ###
                        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
                            #print(track_id, tp_index[s].cpu().numpy()[0], remain_df)
                            more_frames = 4
                            # trajectory may be a curve, so it should rely on last frame and the previous frame
                            dis_x = remain_df.iloc[frame_num-1, 3] - remain_df.iloc[frame_num-2, 3]
                            dis_y = remain_df.iloc[frame_num-1, 4] - remain_df.iloc[frame_num-2, 4]
                            all_x, all_y, all_v, all_yaw = [], [], [], []
                            for f_index in range(more_frames):
                                all_v.append(remain_df.iloc[frame_num-1, 2])
                                all_x.append(remain_df.iloc[frame_num-1, 3] + dis_x * (f_index + 1))
                                all_y.append(remain_df.iloc[frame_num-1, 4] + dis_y * (f_index + 1))
                                all_yaw.append(remain_df.iloc[frame_num-1, 5])
                            for further_t in range(more_frames):
                                b = {'TIMESTAMP': [remain_df.TIMESTAMP.values[-1] + (further_t + 1) * 500000], 'TRACK_ID': [track_id],
                                    # 'V': [all_v[further_t]], 'X': [x[further_t].cpu().numpy()], 'Y': [y[further_t].cpu().numpy()],
                                    'V': [all_v[further_t]], 'X': [all_x[further_t]], 'Y': [all_y[further_t]],
                                    'YAW': [all_yaw[further_t]]}
                                df_insert = pd.DataFrame(b)
                                further_df = pd.concat([further_df, df_insert], ignore_index=True)
                        further_df.to_csv('output_csv_moving_foward/' + now_scenario + '_foward.csv', index=False)
                        collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = self.cal_cr_and_similarity(further_df, attacker_id)

                        ### Interpolation for moving foward ###

                        if collision_flag:
                            pred_collision_rate += 1
                            
                            while record_yaw_distance < 0:
                                record_yaw_distance = (record_yaw_distance + 360.0)
                            record_yaw_distance = abs(record_yaw_distance - 360.0) if record_yaw_distance > 180 else record_yaw_distance

                            # record_yaw_distance = (record_yaw_distance + 360.0) if record_yaw_distance < 0 else record_yaw_distance
                            # record_yaw_distance = abs(record_yaw_distance - 360.0) if record_yaw_distance > 180 else record_yaw_distance
                            #print("REAL yaw dist:", real_yaw_dist)
                            #print("Record yaw dist:", record_yaw_distance) 
                            
                            # metric only calculate on GT collision
                            if attacker_right_flag:
                                attacker_right_collision_rate += 1
                                yaw_distance = abs(ideal_yaw_offset - record_yaw_distance)
                                real_yaw_dist_average += yaw_distance
                                if yaw_distance < 30:
                                    col_degree_30 += 1
                                if condition == 'LTAP':
                                    LTAP_yaw_distance += yaw_distance
                                    col_LTAP_num += 1
                                    if yaw_distance < 30:
                                        sim_LTAP_num += 1
                                    
                                elif condition == 'JC':
                                    junction_crossing_yaw_distance += yaw_distance
                                    col_junction_crossing_num += 1
                                    if yaw_distance < 30:
                                        sim_junction_crossing_num += 1
                                elif condition == 'HO':
                                    opposite_direction_yaw_distance += yaw_distance
                                    col_opposite_direction_num += 1
                                    if yaw_distance < 30:
                                        sim_opposite_direction_num += 1
                                elif condition == 'RE':
                                    rear_end_yaw_distance += yaw_distance
                                    col_rear_end_num += 1
                                    if yaw_distance < 30:
                                        sim_rear_end_num += 1
                                elif condition == 'LC':
                                    lane_change_yaw_distance += yaw_distance
                                    col_lane_change_num += 1
                                    if yaw_distance < 30:
                                        sim_lane_change_num += 1

                        ### attacker IOU ###
                        if condition == 'LTAP':
                            if attacker_iou > 0.5:
                                attacker_IOU50 += 1
                                ltap_IOU50 += 1
                        elif condition == 'JC':
                            if attacker_iou > 0.5:
                                attacker_IOU50 += 1
                                junction_crossing_IOU50 += 1
                        elif condition == 'HO':
                            if attacker_iou > 0.5:
                                attacker_IOU50 += 1
                                opposite_direction_IOU50 += 1
                        elif condition == 'RE':
                            if attacker_iou > 0.5:
                                attacker_IOU50 += 1
                                rear_end_IOU50 += 1
                        elif condition == 'LC':
                            if attacker_iou > 0.5:
                                attacker_IOU50 += 1
                                lane_change_IOU50 += 1
                        atr_iou_list.append(attacker_iou)
                        


                        # Calculate Metrics
                        #fde = math.sqrt((traj_df.iloc[frame_num-1, 3] - gt_trajectories[now_scenario][-1][0]) ** 2 + (traj_df.iloc[frame_num-1, 4] - gt_trajectories[now_scenario][-1][1]) ** 2)
                        
                        #gt_yaw = gt_yaw_np[s]
                        #p_yaw = gyaw#.cpu().numpy()# * 180 / np.pi
                        #print(yaw_offset, atr_yaw_offset[s])
                        ego_pred_target_yaw = pred_target_point_yaw[s][0][0].cpu().numpy() * 180 / np.pi
                        ego_gt_target_yaw = gt_yaw_np[s]
                        
                        now_yaw_distance = abs(ego_pred_target_yaw - ego_gt_target_yaw)
                        # print(ego_yaw_last_frame_list[s][0][0] * 180 / np.pi, pred_target_point_yaw_origin[s][0][0], ego_gt_target_yaw)
                        all_yaw_distance += now_yaw_distance

                        if now_iou > 0.02:
                            all_iou_no_0 += now_iou
                            ego_iou_2 += 1
                        if now_iou > 0.5:
                            ego_iou_50 += 1
                            over50_yaw_distance += now_yaw_distance
                        else:
                            below50_yaw_distance += now_yaw_distance
                        all_iou += now_iou

                        if now_iou > 0.7:
                            ego_iou_70 += 1
                        ego_iou_list.append(now_iou)
                        

                        # if attacker_iou > 0.02:
                        #     attacker_iou_2 += 1
                        if attacker_iou > 0.5:
                            attacker_iou_50 += 1
                        
                        # if together_gt_iou > 0.02:
                        #     gt_collision_rate += 1
                        if together_pred_iou > 0.02:
                            ego_tp_pred_IOU50 += 1
                        ego_tp_pred_average_IOU += together_pred_iou

                        # print("atr point:", pred_target_point[s][0][0] + tar_offset_pred[s][0], pred_target_point[s][0][1] + tar_offset_pred[s][1])
                        # print("tp_pos:", tp_pos)
                        if plot_atr:
                            #for i in range(tp_pos.shape[0]):
                            plt.plot(tp_pos[:, 0], tp_pos[:, 1], '-', color='purple')

                        
                        # plt.xlim(gt_np[s][0][0] - 35,
                        #             gt_np[s][0][0] + 35)
                        # plt.ylim(gt_np[s][0][1] - 35,
                        #             gt_np[s][0][1] + 35)
                       
                        #"""
                        plt.xlim(gt_trajectories[now_scenario][-1][0] - 50,
                                    gt_trajectories[now_scenario][-1][0] + 50)
                        plt.ylim(gt_trajectories[now_scenario][-1][1] - 50,
                                    gt_trajectories[now_scenario][-1][1] + 50)
                        # plt.xlim(gt_trajectories[now_scenario][-1][0] - 25,
                        #             gt_trajectories[now_scenario][-1][0] + 25)
                        # plt.ylim(gt_trajectories[now_scenario][-1][1] - 25,
                        #             gt_trajectories[now_scenario][-1][1] + 25)

                        # gt_ttc = data.horizon[s].cpu().numpy()
                        # pred_ttc = ttc_pred[s].cpu().numpy()[0]
                        # ttc_distance = abs(gt_ttc - pred_ttc)
                        # ttc_distance_sum += ttc_distance
                        # title = 'TTC loss: ' + str(round(ttc_distance, 2))
                        #title = 'GT TTC:' + str(gt_ttc) + 'Pred TTC:' + str(pred_ttc) + 'loss:' + str(ttc_distance)
                        #plt.title(title)

                        # if right_guess_tp:
                        #     right_guess_tp = "T"
                        # else:
                        #     right_guess_tp = "F"

                        pred_target_indices = indices[s].cpu().numpy()
                        GT_target_indices = indices_GT[s].cpu().numpy()
                        if pred_target_indices == GT_target_indices:
                            target_guess = "T"
                        else:
                            target_guess = "F"
                        # if attacker_right_flag:
                        #     attacker_right = "T"
                        # else:
                        #     attacker_right = "F"

                        #print(atr_yaw_pred)
                        #exit()
                        #title = 'Yaw offset:' + str(abs(round(atr_yaw_pred[s].cpu().numpy()[0] * 180 / np.pi, 1))) + ' TP: ' + right_guess_tp + ' Target: ' + target_guess
                        
                        #title = 'TP: ' + right_guess_tp + ' Target: ' + target_guess
                        
                        #print("GT:", data.offset_gt.view(-1, 2).shape)
                        #print("Pred:", offset_pred[:,0,:].shape)
                        #print("GT:", gt_trajectories[now_scenario][-1], origs[s])
                        #exit()
                        sum_target_candidate_len = 0
                        #(pred_target.repeat(1, 2).reshape(batch_size, -1, 2).shape) # 256, 738, 2
                        all_gt = torch.zeros(pred_target.repeat(1, 2).reshape(batch_size, -1, 2).shape, device=self.device)
                        for i in range(batch_size):
                            #print(data.candidate_lens.shape, i, sum_target_candidate_len)
                            temp = data.offset_gt_each[sum_target_candidate_len:sum_target_candidate_len + data.candidate_lens[i]]
                            #print(all_gt.shape, temp.shape)
                            all_gt[i, :data.candidate_lens[i], :] = temp
                            sum_target_candidate_len += data.candidate_lens[i]
                        # print(indices[s])
                        # print(offset_pred.shape, offset_pred[s][0])
                        # print(all_gt.shape, all_gt[s][indices[s]])
                        # print(all_gt[s][indices[s]][0][0] , offset_pred[s][0][0])
                        offset_dist_each = round(math.sqrt((all_gt[s][indices[s]][0][0] - offset_pred[s][0][0]) ** 2 + (all_gt[s][indices[s]][0][1] - offset_pred[s][0][1]) ** 2), 2) 
                        #offset_dist = round(math.sqrt((data.offset_gt.view(-1, 2)[s][0] - offset_pred[s][0][0]) ** 2 + (data.offset_gt.view(-1, 2)[s][1] - offset_pred[s][0][1]) ** 2), 2) 

                        if mulit_gt_target_point:
                            mean_dist = 0
                            for gt_index in range(pred_target_point[s].shape[0]):
                                mean_dist += round(math.sqrt((gt_trajectories[now_scenario][-1][0] - pred_target_point[s][gt_index][0]) ** 2 + (gt_trajectories[now_scenario][-1][1] - pred_target_point[s][gt_index][1]) ** 2), 1) 
                            dist = mean_dist / pred_target_point[s].shape[0]
                        else:
                            dist = round(math.sqrt((gt_trajectories[now_scenario][-1][0] - pred_target_point[s][0][0]) ** 2 + (gt_trajectories[now_scenario][-1][1] - pred_target_point[s][0][1]) ** 2), 1) 
                        
                        # print(gt_trajectories[now_scenario][-1], pred_target_point[s][0])
                        
                        #print("1st future:", gt_trajectories[now_scenario][0], "origs:", origs[s])
                        offset_pred_convert = self.convert_coord(offset_pred[s][0].cpu().numpy(), np.array([0, 0]), rots[s])[0]
                        
                        #print("origin point:", origs[s])
                        #print("final gt point:", gt_trajectories[now_scenario][-1], "gt dist:", gt_trajectories[now_scenario][-1] - origs[s])
                        #print("offset:", offset_pred_convert, "origin + offset:", origs[s] + offset_pred_convert )
                        #print("final pred point:", pred_target_point[s][0])
                        #print(dist)
                        
                        sum_fde += dist
                        fde_list.append(dist)
                        #ideal_yaw_offset - record_yaw_distance

                        attacker_pred_x = pred_target_point[s][0][0] + tar_offset_pred[s][0]
                        attacker_pred_y = pred_target_point[s][0][1] + tar_offset_pred[s][1]
                        attacker_gt_x = gt_trajectories[now_scenario][-1][0] + atr_pos_offset[s][0]
                        attacker_gt_y = gt_trajectories[now_scenario][-1][1] + atr_pos_offset[s][1]

                        attacker_dist = np.sqrt((attacker_gt_x - attacker_pred_x)**2 + (attacker_gt_y - attacker_pred_y)**2)
                        #print(attacker_dist)
                        sum_attacker_de += attacker_dist

                        gt_start_to_end_distance = np.sqrt((gt_trajectories[now_scenario][-1][0] - origs[s][0])**2 + (gt_trajectories[now_scenario][-1][1] - origs[s][1])**2)
                        pred_start_to_end_distance = np.sqrt(offset_pred_convert[0] ** 2 + offset_pred_convert[1] ** 2)
                        gt_trajectories_distribution.append(gt_start_to_end_distance)
                        pred_trajectories_distribution.append(pred_start_to_end_distance)
                        #title = 'Pred: ' + str(pred_target_indices) + ' GT: ' + str(GT_target_indices) + ' Target: ' + target_guess + ' Dist: ' + str(dist) + ' Offset: ' + str(offset_dist_each)
                        #title = 'Pred: ' + str(pred_target_indices) + ' GT: ' + str(GT_target_indices) + ' Offset GT: ' + str(np.around(all_gt[s][indices[s]][0].cpu().numpy(), 1)) + ' Offset Pred: ' + str(np.around(offset_pred[s][0].cpu().numpy(), 1))
                        #title = 'Pred: ' + str(pred_target_indices) + ' GT: ' + str(GT_target_indices) + ' Predict Padding: ' + str(now_scenario_padding_number)# + ' Offset Pred: ' + str(np.around(offset_pred[s][0].cpu().numpy(), 1))
                        #print(offset_pred[s][0].cpu().numpy(), pred_start_to_end_distance)

                        ### RCNN module
                        if positive_num == 0:
                            title = 'Prediction No Positive Proposal'
                        else:
                            title = 'P: ' + str(round(precision, 2)) + ' R: ' + str(round(recall, 2)) + ' F1: ' + str(round(f1, 2)) + ' P num: ' + str(positive_num)
                        ### RCNN module

                        if only_regression:
                            title = 'Offset GT: ' + str(np.around(gt_start_to_end_distance.cpu().numpy(), 1)) + ' Offset Pred: ' + str(np.around(pred_start_to_end_distance, 1)) + \
                                ' Ideal Yaw: ' + str(ideal_yaw_offset) + ' Record Yaw: ' + str(abs(round(record_yaw_distance, 2)))
                        else:
                            title = 'Offset GT: ' + str(np.around(all_gt[s][indices[s]][0].cpu().numpy(), 1)) + ' Offset Pred: ' + str(np.around(offset_pred[s][0].cpu().numpy(), 1))
                        
                        #title = 'Collide GT: ' + attacker_right + ' Yaw dist: ' + str(abs(round(yaw_distance, 2)))
                        
                        #title = 'Collide GT: ' + attacker_right + ' Ideal Yaw: ' + str(ideal_yaw_offset) + ' Record Yaw: ' + str(abs(round(record_yaw_distance, 2)))

                        plt.title(title)
                        #plt.savefig('figures/' + np.array(data.seq_id)[s] + '.png')
                        if not os.path.isdir(save_folder + '/'+ 'figures'):
                            os.mkdir(save_folder + '/figures')
                        # plt.show()
                        plt.savefig(save_folder + '/figures/' + np.array(data.seq_id)[s] + '_' + str(round(now_iou, 1)) + '.png')
                        plt.close()
                        

                #print("pred_y:", pred_y)
                #print("gt:", gt)

                # record the prediction and ground truth
                # for batch_id in range(batch_size):
                #     seq_id = seq_ids[batch_id]
                #     forecasted_trajectories[seq_id] = [self.convert_coord(pred_y_k, origs[batch_id], rots[batch_id])
                #                                        if convert_coordinate else pred_y_k
                #                                        for pred_y_k in pred_y[batch_id]]
                #     gt_trajectories[seq_id] = self.convert_coord(gt[batch_id], origs[batch_id], rots[batch_id]) \
                #         if convert_coordinate else gt[batch_id]
        
        
        bins = np.linspace(0.0, 1.0, num=20)
        # bins = np.linspace(min(ego_iou_list), max(ego_iou_list), num=20)
        hist, bins = np.histogram(ego_iou_list, bins=bins, density=False)
        plt.bar(bins[:-1], hist, align='center', width=0.01)
        plt.xlabel('iou')
        plt.ylabel('Freq')
        plt.title('EGO iou distribution')
        # plt.show()
        plt.savefig(save_folder + '/EGO iou distribution.png')
        plt.close()

        if plot_atr:
            bins = np.linspace(0.0, 1.0, num=20)
            hist, bins = np.histogram(atr_iou_list, bins=bins, density=False)
            plt.bar(bins[:-1], hist, align='center', width=0.01)
            plt.xlabel('iou')
            plt.ylabel('Freq')
            plt.title('Attacker iou distribution')
            # plt.show()
            plt.savefig(save_folder + '/Attacker iou distribution.png')
            plt.close()

        # bins = np.linspace(min(fde_list), max(fde_list), num=20)
        bins = np.linspace(min(positive_num_list), max(positive_num_list), num=50)
        hist, bins = np.histogram(positive_num_list, bins=bins, density=False)
        plt.bar(bins[:-1], hist, align='center', width=0.8)
        plt.xlabel('Positive proposal')
        plt.ylabel('Freq')
        plt.title('Positive proposal distribution')
        # plt.show()
        plt.savefig(save_folder + '/Positive proposal distribution.png')
        plt.close() 

        bins = np.linspace(min(gt_positive_proposal_num), max(gt_positive_proposal_num), num=50)
        hist, bins = np.histogram(gt_positive_proposal_num, bins=bins, density=False)
        plt.bar(bins[:-1], hist, align='center', width=0.8)
        plt.xlabel('Positive proposal')
        plt.ylabel('Freq')
        plt.title('GT Positive proposal distribution')
        # plt.show()
        plt.savefig(save_folder + '/GT Positive proposal distribution.png')
        plt.close()

        # bar_width = 0.25
        # index = np.arange(len(precision_list))
        # bar1 = plt.bar(index, precision_list, bar_width, label='Precision')
        # bar2 = plt.bar(index + bar_width, recall_list, bar_width, label='Recall')
        # bar3 = plt.bar(index + 2 * bar_width, f1_list, bar_width, label='F1 Score')
        # plt.xlabel('Data Points')
        # plt.ylabel('Scores')
        # plt.title('Distribution of Precision, Recall, and F1 Score')
        # plt.xticks(index + bar_width)
        # # plt.xticklabels([f'Data {i+1}' for i in range(len(precision_list))])
        # plt.legend()
        # plt.show()
        # plt.close()
        bins = np.arange(0, 1.1, 0.1)
        #bins = np.linspace(0, 1, 11)
        plt.hist(precision_list, bins=bins, alpha=0.7, label='Precision')
        plt.hist(recall_list, bins=bins, alpha=0.7, label='Recall')
        plt.hist(f1_list, bins=bins, alpha=0.7, label='F1 Score')
        plt.xlabel('Score')
        plt.ylabel('Number of Data Points')
        plt.title('Distribution of Precision, Recall, and F1 Score')
        plt.legend()
        # plt.show()
        plt.savefig(save_folder + '/F1 Distribution.png')
        plt.close()

        
        if only_regression:
            print("average displacement:", sum(gt_trajectories_distribution) / all_data_num)
            bins = np.arange(min(fde_list), max(fde_list), 0.1)
            #bins = np.linspace(min(fde_list), 5, num=50)
            hist, bins = np.histogram(fde_list, bins=bins, density=False)
            plt.bar(bins[:-1], hist, align='center', width=0.8)
            plt.xlabel('Displacement Error (m)')
            plt.ylabel('Freq')
            plt.title('distribution')
            # plt.show()
            plt.savefig(save_folder + '/DE Distribution.png')
            plt.close()
            
            counts1, bins = np.histogram(gt_trajectories_distribution, bins=np.linspace(min(gt_trajectories_distribution), max(gt_trajectories_distribution), num=20))
            counts2, _ = np.histogram(pred_trajectories_distribution, bins=bins)
            plt.bar(bins[:-1], counts1, width=np.diff(bins), align='edge', label='GT')
            plt.bar(bins[:-1], counts2, width=np.diff(bins), align='edge', label='Pred', alpha=0.5)
            plt.xlabel('Distance(m)')
            plt.ylabel('Total Count')
            plt.title('Trajectory Distribution')
            plt.legend()
            # plt.show()
            plt.savefig(save_folder + '/Trajectory Distribution.png')
            plt.close()

        # bins = np.linspace(min(padding_number_list), max(padding_number_list), num=6)
        # hist, bins = np.histogram(padding_number_list, bins=bins, density=False)
        # plt.bar(bins[:-1], hist, align='center', width=0.8)
        # plt.xlabel('Prediction on padding number')
        # plt.ylabel('Freq')
        # plt.title('Target point prediction')
        # plt.show()
        # plt.close()


        
        print("TP:", all_TP, "FP:", all_FP, "FN:", all_FN)
        recall = all_TP / (all_TP + all_FN) if (all_TP + all_FN) != 0 else 0
        precision = all_TP / (all_TP + all_FP) if (all_TP + all_FP) != 0 else 0
        print("Recall:", recall, "Precision:", precision)
        F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        print("F1 score:", F1_score)
        print("Positive Prediction Num:", sum(positive_num_list) / all_data_num)
        print("No guess scenarios:", len(no_guess_list))
        # compute the metric
        print("cls accuracy:", all_cls_acc / batch_num) 
        print("Top 5 cls accuracy:", all_top5_cls_acc / batch_num)
        #print("average IOU:", all_iou / all_data_num)
        #print("2 percent ego data:", ego_iou_2 / all_data_num)
        print("50 percent ego data:", ego_iou_50 / all_data_num)
        print("70 percent ego data:", ego_iou_70 / all_data_num)
        #print("average IOU with threshold:", all_iou_no_0 / all_data_num)
        print("average ego FDE:", sum_fde / all_data_num)
        print("ego yaw distance:", all_yaw_distance / all_data_num)
        print("50 percent attacker BBOX:", attacker_IOU50 / all_data_num)
        print("average attacker DE:", sum_attacker_de / all_data_num)
        # print("over 50 percent yaw distance:", over50_yaw_distance / collision_data_50)
        # print("below 50 percent yaw distance:", below50_yaw_distance / (all_data_num - collision_data_50))
        #print("2 percent attacker data:", attacker_iou_2 / all_data_num)
        #print("50 percent attacker data:", attacker_iou_50 / all_data_num)
        #print("GT collision rate:", gt_collision_rate / all_data_num)
        # print("EGO TP overlap IOU 50:", ego_tp_pred_IOU50 / all_data_num)
        # print("EGO TP overlap average IOU:", ego_tp_pred_average_IOU / all_data_num)
        print("Pred any collision rate:", pred_collision_rate / all_data_num) 
        print("Right GT collision rate:", attacker_right_collision_rate / all_data_num)
        print("Theoretical similarity(10):", tolerance_degree_10 / all_data_num)
        #print("Theoretical similarity(20):", tolerance_degree_20 / all_data_num)
        print("Theoretical similarity(30):", tolerance_degree_30 / all_data_num)
        print("Ideal attacker yaw distance:", ideal_yaw_dist_average / all_data_num)
        print("Real attacker yaw distance:", real_yaw_dist_average / attacker_right_collision_rate)
        
        print("LC ideal sim:", lane_change_num, "all:", lane_change_all, " Real collision:", col_lane_change_num, " Real sim:", sim_lane_change_num)
        print("HO ideal sim:", opposite_direction_num, "all:", opposite_direction_all, " Real collision:", col_opposite_direction_num, " Real sim:", sim_opposite_direction_num)
        print("RE ideal sim:", rear_end_num, "all:", rear_end_all, " Real collision:", col_rear_end_num, " Real sim:", sim_rear_end_num)
        print("JC ideal sim:", junction_crossing_num, "all:", junction_crossing_all, " Real collision:", col_junction_crossing_num, " Real sim:", sim_junction_crossing_num)
        print("LTAP ideal sim:", LTAP_num, "all:", LTAP_all, " Real collision:", col_LTAP_num, " Real sim:", sim_LTAP_num)

        print("LC ratio:", round(lane_change_num / lane_change_all, 2), " Real collision:", round(col_lane_change_num / lane_change_all, 2), " Similarity:", round(sim_lane_change_num / col_lane_change_num, 2))
        print("HO ratio:", round(opposite_direction_num / opposite_direction_all, 2), " Real collision:", round(col_opposite_direction_num / opposite_direction_all, 2), " Similarity:", round(sim_opposite_direction_num / col_opposite_direction_num, 2))
        print("RE ratio:", round(rear_end_num / rear_end_all, 2), " Real collision:", round(col_rear_end_num / rear_end_all, 2), " Similarity:", round(sim_rear_end_num / col_rear_end_num, 2))
        print("JC ratio:", round(junction_crossing_num / junction_crossing_all, 2), " Real collision:", round(col_junction_crossing_num / junction_crossing_all, 2), " Similarity:", round(sim_junction_crossing_num / col_junction_crossing_num, 2))
        print("LTAP:", round(LTAP_num / LTAP_all, 2) if col_LTAP_num != 0 else 0, " Real collision:", round(col_LTAP_num / LTAP_all, 2) if col_LTAP_num != 0 else 0, " Similarity:", round(sim_LTAP_num / col_LTAP_num, 2) if col_LTAP_num != 0 else 0)

        print("LC TP IOU ratio:", round(lane_change_IOU50 / lane_change_all, 2), "Yaw distance:", round(lane_change_yaw_distance / col_lane_change_num, 2))
        print("HO TP IOU ratio:", round(opposite_direction_IOU50 / opposite_direction_all, 2), "Yaw distance:", round(opposite_direction_yaw_distance / col_opposite_direction_num, 2))
        print("RE TP IOU ratio:", round(rear_end_IOU50 / rear_end_all, 2), "Yaw distance:", round(rear_end_yaw_distance / col_rear_end_num, 2))
        print("JC TP IOU ratio:", round(junction_crossing_IOU50 / junction_crossing_all, 2), "Yaw distance:", round(junction_crossing_yaw_distance / col_junction_crossing_num, 2))
        print("LTAP TP IOU ratio:", round(ltap_IOU50 / LTAP_all, 2) if ltap_IOU50 != 0 else 0, "Yaw distance:", round(LTAP_yaw_distance / col_LTAP_num, 2))

        print("Attacker IOU ratio:", round(attacker_IOU50 / all_data_num, 2))

        print("Average Similarity:", (lane_change_num + opposite_direction_num + rear_end_num + junction_crossing_num + LTAP_num) / all_data_num,
              " Real collision:", (col_lane_change_num + col_opposite_direction_num + col_rear_end_num + col_junction_crossing_num + col_LTAP_num) / all_data_num)
        print("TP selection:", all_tp_cls_acc / batch_num)
        print(save_folder)
        cr = (col_lane_change_num + col_opposite_direction_num + col_rear_end_num + col_junction_crossing_num + col_LTAP_num) / all_data_num
        with open(save_folder + '/' + split + '_min_train_loss.txt_' + str(F1_score) + '_F1', 'w') as f:
            f.write(
                f"TP / FP / FN:        {all_TP} / {all_FP} / {all_FN}\n")
            f.write(
                f"Recall / Precision:  {recall:.4f} / {precision:.4f}\n")
            f.write(
                f"F1 score:            {F1_score:.4f}\n")
            f.write(
                f"Pos Prediction Num:  {sum(positive_num_list) / all_data_num:.4f}\n")
            f.write(
                "=======================================================\n")
            f.write(
                f"cls accuracy:        {all_cls_acc / batch_num:.4f}\n")
            f.write(
                f"Top 5 cls accuracy:  {all_top5_cls_acc / batch_num:.4f}\n")
            f.write(
                f"2 percent ego data:  {ego_iou_2 / all_data_num:.4f}\n")
            f.write(
                f"50 percent ego data: {ego_iou_50 / all_data_num:.4f}\n")
            f.write(
                f"ego yaw distance:    {all_yaw_distance / all_data_num:.4f}\n")
            f.write(
                "=======================================================\n")
            f.write(
                f"2 percent attacker data:    {sum_fde / all_data_num:.4f}\n")
            f.write(
                f"2 percent attacker data:    {attacker_iou_2 / all_data_num:.4f}\n")
            f.write(
                f"50 percent attacker data:   {attacker_iou_50 / all_data_num:.4f}\n")
            f.write(
                f"Pred any collision rate:    {pred_collision_rate / all_data_num:.4f}\n")
            f.write(
                f"Right GT collision rate:    {attacker_right_collision_rate / all_data_num:.4f}\n")
            f.write(
                f"Theoretical similarity(10): {tolerance_degree_10 / all_data_num:.4f}\n")
            f.write(
                f"Theoretical similarity(20): {tolerance_degree_20 / all_data_num:.4f}\n")
            f.write(
                f"Theoretical similarity(30): {tolerance_degree_30 / all_data_num:.4f}\n")
            f.write(
                f"Ideal attacker yaw distance: {ideal_yaw_dist_average / all_data_num:.4f}\n")
            f.write(
                f"Real attacker yaw distance: {real_yaw_dist_average / attacker_right_collision_rate:.4f}\n")
            f.write(
                f"Ideal similarity / Real collision rate / Real Similarity / Real yaw distance \n")
            f.write(
                f"{lane_change_num / lane_change_all:.4f} / {col_lane_change_num / lane_change_all:.4f} / {sim_lane_change_num / col_lane_change_num:.4f} / {lane_change_yaw_distance / col_lane_change_num:.4f}\n")
            f.write(
                f"{opposite_direction_num / opposite_direction_all:.4f} / {col_opposite_direction_num / opposite_direction_all:.4f} / {sim_opposite_direction_num / col_opposite_direction_num:.4f} / {opposite_direction_yaw_distance / col_opposite_direction_num:.4f}\n")
            f.write(
                f"{rear_end_num / rear_end_all:.4f} / {col_rear_end_num / rear_end_all:.4f} / {sim_rear_end_num / col_rear_end_num:.4f} / {rear_end_yaw_distance / col_rear_end_num:.4f}\n")
            f.write(
                f"{junction_crossing_num / junction_crossing_all:.4f} / {col_junction_crossing_num / junction_crossing_all:.4f} / {sim_junction_crossing_num / col_junction_crossing_num:.4f} / {junction_crossing_yaw_distance / col_junction_crossing_num:.4f}\n")
            f.write(
                f"{LTAP_num / LTAP_all:.4f} / {col_LTAP_num / LTAP_all:.4f} / {sim_LTAP_num / col_LTAP_num:.4f} / {LTAP_yaw_distance / col_LTAP_num:.4f}\n")
            f.write(
                "=======================================================\n")
            Average_Similarity = (lane_change_num + opposite_direction_num + rear_end_num + junction_crossing_num + LTAP_num) / all_data_num
            f.write(
                f"Average Similarity: {Average_Similarity:.4f}\n")
            f.write(
                f"Real collision:     {(col_lane_change_num + col_opposite_direction_num + col_rear_end_num + col_junction_crossing_num + col_LTAP_num) / all_data_num:.4f}\n")
            f.write(
                f"TP selection:       {all_tp_cls_acc / batch_num:.4f}\n")
        csv_file = 'run/' + split + '_performance.csv'
        if not os.path.exists(csv_file):
            print("not exist")
            with open(csv_file, 'a+') as f:
                writer = csv.writer(f)
                # writer.writerow(['Time', 'LR', 'Adam_weight_decay', 'Ego iou>0.5', 'FDE', 'Ego yaw', 'TP iou>0.5', 'TP DE', 'TP yaw', 'TP ID', 'CR', 'Sim'])
                writer.writerow(['Time', 'LR', 'Adam_weight_decay', 'Positive_weight(RCNN)', 'Recall', 'Precision', 'F1', 'iou>0.5']) # RCNN version
                f.close()
        with open(csv_file, 'a+') as f:
            writer = csv.writer(f)
            # writer.writerow([save_folder.split('/')[-1], self.lr, self.weight_decay, ego_iou_50 / all_data_num, sum_fde / all_data_num, all_yaw_distance / all_data_num, 
            #                  attacker_iou_50 / all_data_num, sum_attacker_de / all_data_num, ideal_yaw_dist_average / all_data_num,
            #                  all_tp_cls_acc / all_data_num, cr, Average_Similarity])

            writer.writerow([save_folder.split('/')[-1], self.lr, self.weight_decay, self.positive_weight, recall, precision, F1_score, ego_iou_50 / all_data_num])  # RCNN version
            f.close()
        # all_metric_csv_directory = save_folder + '/all_metric_' + split + '_performance.csv'
        all_metric_csv_directory = 'run/all_metric_' + split + '_performance.csv'
        if not os.path.exists(all_metric_csv_directory):
            print("not exist")
            with open(all_metric_csv_directory, 'a+') as all_metric_f:
                all_metric_writer = csv.writer(all_metric_f)
                all_metric_writer.writerow(['Time', 'LR', 'Adam_weight_decay', 'Ego iou>0.5', 'FDE', 'Ego yaw', 'TP iou>0.5', 'TP DE', 'TP yaw', 'TP ID', 'CR', 'Sim'])
                #writer.writerow(['Time', 'LR', 'Adam_weight_decay', 'Positive_weight(RCNN)', 'Recall', 'Precision', 'F1', 'iou>0.5']) # RCNN version
                all_metric_f.close()
        with open(all_metric_csv_directory, 'a+') as all_metric_f:
            print("writing at ", all_metric_csv_directory)
            all_metric_writer = csv.writer(all_metric_f)
            all_metric_writer.writerow([save_folder.split('/')[-1], self.lr, self.weight_decay, ego_iou_50 / all_data_num, sum_fde / all_data_num, all_yaw_distance / all_data_num, 
                             attacker_iou_50 / all_data_num, sum_attacker_de / all_data_num, ideal_yaw_dist_average / all_data_num,
                             all_tp_cls_acc / batch_num, cr, Average_Similarity])
            all_metric_f.close()
        return ego_iou_50 / all_data_num

    def cal_cr_and_similarity(self, traj_df, attacker_id_gt):
        collision_flag = 0
        right_attacker_flag = 0
        real_yaw_distance = -999
        record_yaw_distance = -999
        vehicle_list = []
        for track_id, remain_df in traj_df.groupby('TRACK_ID'):
            vehicle_list.append(remain_df)
        ego_list = []
        for track_id, remain_df in traj_df.groupby('TRACK_ID'):
            if str(track_id) == 'ego':
                ego_list.append(remain_df.reset_index())
        #print(traj_df)

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
            real_ego_angle = ego_list[0].loc[t - 1, 'YAW']
            ego_rec = [ego_x_next, ego_y_next, self.vehicle_width
                                            , self.vehicle_length, ego_angle]
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
            # plt.plot([x_1, x_2, x_4, x_3, x_1], [
            #                             y_1, y_2, y_4, y_3, y_1], '-',  color='lime', markersize=3)
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
                real_other_angle = real_pred_x = vl[t - 1][5]
                # other_angle = vl[past_len][4]
                # ego_angle = ego_list[0][4][int(filename_t) + past_len]
                #print(ego_x, ego_y, real_pred_x, real_pred_y)
                ego_rec = [real_pred_x_next, real_pred_y_next, self.vehicle_width
                                            , self.vehicle_length, other_angle]
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
                
                cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
                #print(t, now_id, ego_polygon.intersection(other_polygon).area, "GT:", attacker_id_gt)
                
                if cur_iou > VEH_COLL_THRESH:
                    #print(attacker_id_gt, "COLLIDE!", now_id)
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
                        # Must collide with GT attacker
                        
                        real_yaw_distance = angle_vectors(other_vec, ego_vec) * 180 / np.pi
                        record_yaw_distance = real_ego_angle - real_other_angle
                        #print(record_yaw_distance)
                    
                if collision_flag:
                    break
            if collision_flag:
                break
        return collision_flag, real_yaw_distance, right_attacker_flag, record_yaw_distance
    def eval_when_training(self,
             m=1,
             split=None,
             iter_epoch=0,
             output_dir='',
             miss_threshold=2.0,
             compute_metric=False,
             convert_coordinate=False,
             plot=True,
             save_pred=False):

        print("eval_when_training!!")

        if split == 'val':
            now_loader = self.eval_loader
        elif split == 'train':
            now_loader = self.train_loader

        self.model.eval()

        gt_trajectories = {}

        k = self.model.k
        self.m = m
        collision_ratio = 1.0

        only_1_batch_flag = 0
        ideal_yaw_offset = 0
        vehicle_length = 4.7 * collision_ratio
        vehicle_width = 2 * collision_ratio

        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width

        all_iou = 0
        all_iou_no_0 = 0
        ego_iou_2 = 0
        ego_iou_50 = 0
        ego_iou_70 = 0
        attacker_iou_2 = 0
        attacker_iou_50 = 0
        all_data_num = 0
        sum_fde = 0
        sum_attacker_de = 0
        fde_list = []
        gt_trajectories_distribution = []
        pred_trajectories_distribution = []
        padding_number_list = []
        positive_num_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        all_TP = 0
        all_FP = 0
        all_FN = 0
        ttc_distance_sum = 0
        batch_num = 0
        all_cls_acc = 0
        all_top5_cls_acc = 0
        all_yaw_distance = 0
        over50_yaw_distance = 0
        below50_yaw_distance = 0
        gt_collision_rate = 0
        pred_collision_rate = 0
        all_tp_cls_acc = 0
        attacker_right_collision_rate = 0

        tolerance_degree_10 = 0
        tolerance_degree_20 = 0
        tolerance_degree_30 = 0
        ideal_yaw_dist_average = 0
        real_yaw_dist_average = 0

        col_degree_30 = 0

        lane_change_num = 0
        junction_crossing_num = 0
        LTAP_num = 0
        opposite_direction_num = 0
        rear_end_num = 0

        lane_change_all = 0
        junction_crossing_all = 0
        LTAP_all = 0
        opposite_direction_all = 0
        rear_end_all = 0

        lane_change_yaw_distance = 0
        junction_crossing_yaw_distance = 0
        LTAP_yaw_distance = 0
        opposite_direction_yaw_distance = 0
        rear_end_yaw_distance = 0

        col_lane_change_num = 0
        col_junction_crossing_num = 0
        col_LTAP_num = 0
        col_opposite_direction_num = 0
        col_rear_end_num = 0

        sim_lane_change_num = 0
        sim_junction_crossing_num = 0
        sim_LTAP_num = 0
        sim_opposite_direction_num = 0
        sim_rear_end_num = 0

        ###########
        plot_atr = True
        plot_target_point_before_regression = False # fixed pos num
        plot_RCNN_target = False
        plot_regression_based_on_RCNN = False
        mulit_gt_target_point = False
        only_regression = True
        ###########

        with torch.no_grad():
            for data in tqdm(now_loader):                
                batch_size = data.num_graphs
                horizon_sum = 0
                data_gt_list = []
                data_gt_yaw_list = []
                for i in range(batch_size):
                    data.horizon[i] -= 7
                    each_traj = data.y[horizon_sum * 2:(horizon_sum + int(data.horizon[i])) * 2].cpu().view(-1, 2).cumsum(axis=0)
                    
                    #data.horizon[i] = 3

                    each_yaw = int(data.y_yaw[horizon_sum + int(data.horizon[i]) - 1].cpu())
                    horizon_sum += int(data.horizon[i])
                    data_gt_list.append(each_traj)
                    data_gt_yaw_list.append(each_yaw)
                gt_np = np.array(data_gt_list)
                gt_yaw_np = np.array(data_gt_yaw_list)
                
                origs = data.orig.numpy()
                rots = data.rot.numpy()
                atr_pos_offset = data.atr_pos_offset.numpy()
                atr_yaw_offset = data.atr_yaw_offset.numpy()
                
                seq_ids = np.array(data.seq_id)

                # inference and transform dimension
                if self.multi_gpu:
                    # out = self.model.module(data.to(self.device))
                    pred_target, target_point_pred_pos, offset_pred, yaw_pred, pred_target_point, pred_target_point_yaw, tar_offset_pred, atr_yaw_pred, tp_index, RCNN_cls_result = self.model.inference(data.to(self.device))
                else:
                    pred_target, target_point_pred_pos, offset_pred, yaw_pred, pred_target_point, pred_target_point_yaw, tar_offset_pred, atr_yaw_pred, tp_index, RCNN_cls_result = self.model.inference(data.to(self.device))
                    
                ################################################################
                #gt_target = data.candidate_gt.view(-1, data.candidate_len_max[0])
                pred_target_point = pred_target_point.cpu()
                pred_target_point_yaw = pred_target_point_yaw.cpu()
                
                
                
                tar_offset_pred = tar_offset_pred.cpu()
                atr_yaw_pred = atr_yaw_pred.cpu()
                _, indices = pred_target.topk(1, dim=1)
                #batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(1)]).T
                #target_pred_se, offset_pred_se = data.candidate.view(-1, data.candidate_len_max[0], 2)[batch_idx, indices].cpu(), offset_pred[batch_idx, indices].cpu()
                _, indices_GT = data.candidate_gt.view(-1, data.candidate_len_max[0]).topk(1, dim=1) #128, 1748 -> 128, 1
                batch_cls_acc = accuracy_score(indices_GT.cpu(), indices.cpu())
                all_cls_acc += batch_cls_acc
                batch_num += 1

                RCNN_labels = torch.argmax(RCNN_cls_result, dim=2)
                RCNN_labels = RCNN_labels.cpu()

                batch_top5_cls_acc = top_k_accuracy_score(indices_GT.flatten().cpu().numpy(), pred_target.cpu().numpy(), k=5, labels=np.arange(data.candidate_len_max[0].cpu()))
                #_, indices_5 = pred_target.topk(5, dim=1)
                all_top5_cls_acc += batch_top5_cls_acc

                gt_attacker_list = []
                for s in range(batch_size):
                    sce_df = pd.read_csv('nuscenes_data/trajectory/' + split + '/' + seq_ids[s] + '.csv')
                    objs = sce_df.groupby(['TRACK_ID']).groups
                    keys = list(objs.keys())
                    del keys[keys.index('ego')]
                    if seq_ids[s].split('_')[8] in keys:
                        gt_attacker_list.append([keys.index(seq_ids[s].split('_')[8])])
                    else:
                        gt_attacker_list.append([100])
                batch_tp_cls_acc = accuracy_score(np.array(tp_index.cpu()), np.array(gt_attacker_list))           
                all_tp_cls_acc += batch_tp_cls_acc
                if not only_1_batch_flag:
                    only_1_batch_flag = 0
                    for s in range(batch_size):
                        
                        ###### 180 degree for ego yaw ######
                        pred_target_point_yaw[s][0][0] = pred_target_point_yaw[s][0][0] - np.pi
                        ###### 180 degree for ego yaw ######
                
                        all_data_num += 1
                        now_scenario = np.array(data.seq_id)[s]
                        sce_df = pd.read_csv('nuscenes_data/trajectory/' + split + '/' + now_scenario + '.csv')
                        split_name = now_scenario.split('_')
                        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[6]
                        objs = sce_df.groupby(['TRACK_ID']).groups
                        keys = list(objs.keys())
                        del keys[keys.index('ego')]
                        if keys[tp_index[s].cpu().numpy()[0]] in keys:
                            guess_attacker_id = keys[tp_index[s].cpu().numpy()[0]]
                        elif len(keys) != 0:
                            ### random guess
                            guess_attacker_id = keys[0]
                        else:
                            continue
                        ##################################
                        ttc = int(now_scenario.split('_')[7].split('-')[-1])
                        ##################################
                        condition = now_scenario.split('_')[5]
                        attacker_id = now_scenario.split('_')[8]
                        right_guess_tp = 0
                        if str(attacker_id) == str(guess_attacker_id):
                            right_guess_tp = 1
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
                            ideal_yaw_offset = 15
                        gt_trajectories[now_scenario] = self.convert_coord(gt_np[s], origs[s], rots[s])
                        pred_target_point[s] = self.convert_coord(pred_target_point[s], origs[s], rots[s])
                        atr_pos_offset[s] = np.matmul(np.linalg.inv(rots[s]), atr_pos_offset[s].T).T
                        tar_offset_pred[s] = np.matmul(np.linalg.inv(rots[s]), tar_offset_pred[s].T).T
                        ego_rec = [gt_trajectories[now_scenario][-1][0], gt_trajectories[now_scenario][-1][1], vehicle_width
                                    , vehicle_length, (gt_yaw_np[s] + 90.0) * np.pi / 180]
                        ego_polygon = self.build_polygon(ego_rec)
                        if data.cross[s] < 0:
                            atr_yaw_offset[s] = abs(atr_yaw_offset[s] - 360.0)
                        else:
                            atr_yaw_offset[s] = atr_yaw_offset[s]

                        attacker_gt_rec = [gt_trajectories[now_scenario][-1][0] + atr_pos_offset[s][0], gt_trajectories[now_scenario][-1][1] + atr_pos_offset[s][1], vehicle_width
                                    , vehicle_length, (gt_yaw_np[s] + 90.0 + atr_yaw_offset[s]) * np.pi / 180]
                        attacker_polygon = self.build_polygon(attacker_gt_rec)
                        
                        pred_ego_rec = [pred_target_point[s][0][0],
                                pred_target_point[s][0][1], vehicle_width,
                                vehicle_length, pred_target_point_yaw[s][0][0] + 90 * np.pi / 180]
                        pred_polygon = self.build_polygon(pred_ego_rec)

                        ### check left or right using cross between ego and attacker
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            if str(track_id) == str(guess_attacker_id):
                                tp_start_x = remain_df.X.values[8]
                                tp_start_y = remain_df.Y.values[8]
                            elif str(track_id) == 'ego':
                                ego_start_x = remain_df.X.values[8]
                                ego_start_y = remain_df.Y.values[8]
                        cross_ego_vec = np.array([pred_target_point[s][0][0] - ego_start_x, pred_target_point[s][0][1]- ego_start_y])
                        cross_tp_vec = np.array([tp_start_x - ego_start_x, tp_start_y- ego_start_y])
                        cross = np.cross(cross_tp_vec, cross_ego_vec)
                        if cross < 0:
                            yaw_offset = abs(atr_yaw_pred[s] - np.pi * 2)
                        else:
                            yaw_offset = atr_yaw_pred[s]

                        attacker_pred_rec = [pred_target_point[s][0][0] + tar_offset_pred[s][0], pred_target_point[s][0][1] + tar_offset_pred[s][1], vehicle_width
                                    , vehicle_length, pred_target_point_yaw[s][0][0] + yaw_offset + (90.0) * np.pi / 180]
                        attacker_pred_polygon = self.build_polygon(attacker_pred_rec)
                        
                        now_iou = ego_polygon.intersection(pred_polygon).area / ego_polygon.union(pred_polygon).area
                        attacker_iou = attacker_polygon.intersection(attacker_pred_polygon).area / attacker_polygon.union(attacker_pred_polygon).area
                        together_yaw_distance = (atr_yaw_pred[s].cpu().numpy() * 180 / np.pi + 360.0) if atr_yaw_pred[s].cpu().numpy() < 0 else atr_yaw_pred[s].cpu().numpy() * 180 / np.pi
                        together_yaw_distance = abs(together_yaw_distance - 360.0) if together_yaw_distance > 180 else together_yaw_distance
                        ideal_dist_offset = abs(ideal_yaw_offset - together_yaw_distance[0])
                        ideal_yaw_dist_average += ideal_dist_offset
                        if ideal_dist_offset < 10:
                            tolerance_degree_10 += 1
                        if ideal_dist_offset < 20:
                            tolerance_degree_20 += 1
                        if ideal_dist_offset < 30:
                            tolerance_degree_30 += 1
                            if condition == 'LTAP': 
                                LTAP_num += 1
                            elif condition == 'JC':
                                junction_crossing_num += 1
                            elif condition == 'HO':
                                opposite_direction_num += 1
                            elif condition == 'RE':
                                rear_end_num += 1
                            elif condition == 'LC':
                                lane_change_num += 1                        
                        start_x = gt_trajectories[now_scenario][0][0].cpu()
                        start_y = gt_trajectories[now_scenario][0][1].cpu()
                        start_yaw = gt_yaw_np[s] * np.pi / 180 
                        sa = 0.1
                        ga = 0.0
                        max_accel = 100.0 
                        max_jerk = 10.0
                        #######################
                        dt = 0.5
                        #######################
                        # min_t = ttc_pred[s].cpu().numpy()[0] / 10
                        # max_t = min_t + dt
                        gt_ttc = ttc - 8###data.horizon[s].cpu().numpy()
                        min_t = gt_ttc * dt - dt / 2
                        max_t = min_t + 0.1
                        # attacker quintic
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            if str(track_id) == str(guess_attacker_id):
                                tp_start_x = remain_df.X.values[8]
                                tp_start_y = remain_df.Y.values[8]
                                tp_start_yaw = remain_df.YAW.values[8] * np.pi / 180
                        tp_gx = pred_target_point[s][0][0] + tar_offset_pred[s][0]
                        tp_gy = pred_target_point[s][0][1] + tar_offset_pred[s][1]
                        tp_gyaw = pred_target_point_yaw[s][0][0] + atr_yaw_pred[s]
                        tp_constant_v = math.sqrt((tp_gx - tp_start_x) ** 2 + (tp_gy - tp_start_y) ** 2) / min_t
                        tp_sv = 4
                        tp_gv = tp_constant_v
                        tp_time, tp_x, tp_y, tp_all_yaw, tp_all_v, tp_a, tp_j = quintic_polynomials_planner(
                            tp_start_x, tp_start_y, tp_start_yaw, tp_sv, sa, tp_gx, tp_gy, tp_gyaw, tp_gv, ga, max_accel, max_jerk, dt, min_t, max_t)
                        tp_pos = np.array((tp_x, tp_y)).T
                        tp_yaw_v = np.array((tp_all_v, tp_all_yaw)).T
                        tp_pos = tp_pos
                        vehicle_list = []
                        ##################### initial scenario is 4 frames
                        collision_moment = sce_df.TIMESTAMP.values[8+gt_ttc]
                        sce_df = sce_df[(sce_df.TIMESTAMP <= collision_moment)]
                        #####################
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            if str(track_id) == str(guess_attacker_id):
                                remain_df.iloc[8:, 3] = tp_pos[:, 0]
                                remain_df.iloc[8:, 4] = tp_pos[:, 1]
                                remain_df.iloc[8:, 2] = tp_yaw_v[:, 0]
                                remain_df.iloc[8:, 5] = tp_yaw_v[:, 1] * 180 / np.pi
                            vehicle_list.append(remain_df)
                        traj_df = pd.concat(vehicle_list)
                        frame_num = len(set(traj_df.TIMESTAMP.values))
                        further_df = traj_df
                        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
                            dis_x = remain_df.iloc[frame_num-1, 3] - remain_df.iloc[frame_num-2, 3]
                            dis_y = remain_df.iloc[frame_num-1, 4] - remain_df.iloc[frame_num-2, 4]
                            gx = remain_df.iloc[frame_num-1, 3]
                            gy = remain_df.iloc[frame_num-1, 4]
                            gv = remain_df.iloc[frame_num-1, 2]
                            gyaw = remain_df.iloc[frame_num-1, 5]
                            steps = 4
                            new_dt = 0.125
                            origin_dt = 0.5
                            new_gx = gx + dis_x * steps
                            new_gy = gy + dis_y * steps
                            new_min_t = origin_dt * steps
                            new_max_t = new_min_t + origin_dt
                            
                            time, x, y, all_yaw, all_v, a, j = quintic_polynomials_planner(
                                gx, gy, gyaw, gv, ga, new_gx, new_gy, gyaw, gv, ga, max_accel, max_jerk, new_dt, new_min_t, new_max_t)
                            further_pos = np.array((x, y)).T
                            for further_t in range(len(x)):
                                b = {'TIMESTAMP': [remain_df.TIMESTAMP.values[-1] + (further_t + 1) * new_dt * 500000], 'TRACK_ID': [track_id],
                                    # 'V': [all_v[further_t]], 'X': [x[further_t].cpu().numpy()], 'Y': [y[further_t].cpu().numpy()],
                                    'V': [all_v[further_t]], 'X': [x[further_t]], 'Y': [y[further_t]],
                                    'YAW': [all_yaw[further_t] * 180 / np.pi]}
                                df_insert = pd.DataFrame(b)
                                further_df = pd.concat([further_df, df_insert], ignore_index=True)
                        collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = self.cal_cr_and_similarity(further_df, attacker_id)
                        
                        if collision_flag:
                            pred_collision_rate += 1
                            
                            while record_yaw_distance < 0:
                                record_yaw_distance = (record_yaw_distance + 360.0)
                            record_yaw_distance = abs(record_yaw_distance - 360.0) if record_yaw_distance > 180 else record_yaw_distance
                            
                            # metric only calculate on GT collision
                            if attacker_right_flag:
                                attacker_right_collision_rate += 1
                                yaw_distance = abs(ideal_yaw_offset - record_yaw_distance)
                                real_yaw_dist_average += yaw_distance
                                if yaw_distance < 30:
                                    col_degree_30 += 1
                                if condition == 'LTAP':
                                    LTAP_yaw_distance += yaw_distance
                                    col_LTAP_num += 1
                                    if yaw_distance < 30:
                                        sim_LTAP_num += 1
                                elif condition == 'JC':
                                    junction_crossing_yaw_distance += yaw_distance
                                    col_junction_crossing_num += 1
                                    if yaw_distance < 30:
                                        sim_junction_crossing_num += 1
                                elif condition == 'HO':
                                    opposite_direction_yaw_distance += yaw_distance
                                    col_opposite_direction_num += 1
                                    if yaw_distance < 30:
                                        sim_opposite_direction_num += 1
                                elif condition == 'RE':
                                    rear_end_yaw_distance += yaw_distance
                                    col_rear_end_num += 1
                                    if yaw_distance < 30:
                                        sim_rear_end_num += 1
                                elif condition == 'LC':
                                    lane_change_yaw_distance += yaw_distance
                                    col_lane_change_num += 1
                                    if yaw_distance < 30:
                                        sim_lane_change_num += 1

                        


                        # Calculate Metrics
                        fde = round(math.sqrt((gt_trajectories[now_scenario][-1][0] - pred_target_point[s][0][0]) ** 2 + (gt_trajectories[now_scenario][-1][1] - pred_target_point[s][0][1]) ** 2), 2) 
                        fde_list.append(fde)
                        sum_fde += fde

                        attacker_pred_x = pred_target_point[s][0][0] + tar_offset_pred[s][0]
                        attacker_pred_y = pred_target_point[s][0][1] + tar_offset_pred[s][1]
                        attacker_gt_x = gt_trajectories[now_scenario][-1][0] + atr_pos_offset[s][0]
                        attacker_gt_y = gt_trajectories[now_scenario][-1][1] + atr_pos_offset[s][1]

                        attacker_dist = np.sqrt((attacker_gt_x - attacker_pred_x)**2 + (attacker_gt_y - attacker_pred_y)**2)
                        sum_attacker_de += attacker_dist

                        gt_yaw = gt_yaw_np[s]
                        #p_yaw = gyaw * 180 / np.pi
                        p_yaw = pred_target_point_yaw[s][0][0].cpu().numpy() * 180 / np.pi
                        #p_yaw = gyaw.cpu().numpy() * 180 / np.pi
                        now_yaw_distance = abs(gt_yaw - p_yaw)
                        #print("now_yaw_distance:", now_yaw_distance)
                        gt_start_to_end_distance = np.sqrt((gt_trajectories[now_scenario][-1][0] - gt_trajectories[now_scenario][0][0] + 0.001)**2 + (gt_trajectories[now_scenario][-1][1] - gt_trajectories[now_scenario][0][1] + 0.001)**2)
                        pred_start_to_end_distance = np.sqrt(offset_pred[s][0].cpu().numpy()[0] ** 2 + offset_pred[s][0].cpu().numpy()[1] ** 2)
                        gt_trajectories_distribution.append(gt_start_to_end_distance)
                        pred_trajectories_distribution.append(pred_start_to_end_distance)
                        all_yaw_distance += now_yaw_distance
                        if now_iou > 0.02:
                            all_iou_no_0 += now_iou
                            ego_iou_2 += 1
                        if now_iou > 0.5:
                            ego_iou_50 += 1
                            over50_yaw_distance += now_yaw_distance
                        else:
                            below50_yaw_distance += now_yaw_distance
                        all_iou += now_iou

                        if now_iou > 0.7:
                            ego_iou_70 += 1
                        #ego_iou_list.append(now_iou)

                        if attacker_iou > 0.02:
                            attacker_iou_2 += 1
                        if attacker_iou > 0.5:
                            attacker_iou_50 += 1                        
        if False:
            #print("fde_list:", fde_list)
            # bins = np.linspace(min(fde_list), max(fde_list), num=20)
            bins = np.arange(0, 1.1, 0.1)
            plt.hist(precision_list, bins=bins, alpha=0.7, label='Precision')
            plt.hist(recall_list, bins=bins, alpha=0.7, label='Recall')
            plt.hist(f1_list, bins=bins, alpha=0.7, label='F1 Score')
            plt.xlabel('Score')
            plt.ylabel('Number of Data Points')
            plt.title('Distribution of Precision, Recall, and F1 Score')
            plt.legend()
            plt.show()
            plt.close()


            bins = np.arange(min(fde_list), 5, 0.1)
            #bins = np.linspace(min(fde_list), 5, num=20)
            hist, bins = np.histogram(fde_list, bins=bins, density=False)
            plt.bar(bins[:-1], hist, align='center', width=0.8)
            plt.xlabel('fde')
            plt.ylabel('freq')
            plt.title('distribution')
            plt.savefig(output_dir + '/' + split + '_' + str(iter_epoch) + '.png')
            plt.close()

            print("Pred:", pred_trajectories_distribution)
            counts1, bins = np.histogram(gt_trajectories_distribution, bins=np.linspace(min(gt_trajectories_distribution), max(gt_trajectories_distribution) / 2, num=20))
            counts2, _ = np.histogram(pred_trajectories_distribution, bins=bins)
            plt.bar(bins[:-1], counts1, width=np.diff(bins), align='edge', label='GT')
            plt.bar(bins[:-1], counts2, width=np.diff(bins), align='edge', label='Pred', alpha=0.5)
            plt.xlabel('Distance(m)')
            plt.ylabel('Total Count')
            plt.title('Trajectory Distribution')
            plt.legend()
            plt.savefig(output_dir + '/' + split + '_' + str(iter_epoch) + '_traj.png')
            plt.close()

            # compute the metric
        cr = (col_lane_change_num + col_opposite_direction_num + col_rear_end_num + col_junction_crossing_num + col_LTAP_num) / all_data_num
        with open(output_dir + '/' + split + '_' + str(iter_epoch) + '.txt', 'a+') as f:
            f.write(
                f"cls accuracy:        {all_cls_acc / batch_num:.4f}\n")
            f.write(
                f"Top 5 cls accuracy:  {all_top5_cls_acc / batch_num:.4f}\n")
            f.write(
                f"50 percent ego data: {ego_iou_50 / all_data_num:.4f}\n")
            f.write(
                f"70 percent ego data: {ego_iou_70 / all_data_num:.4f}\n")
            f.write(
                f"ego FDE:    {sum_fde / all_data_num:.4f}\n")
            f.write(
                f"ego yaw distance:    {all_yaw_distance / all_data_num:.4f}\n")
            f.write(
                "=======================================================\n")
            f.write(
                f"2 percent attacker data:    {attacker_iou_2 / all_data_num:.4f}\n")
            f.write(
                f"50 percent attacker data:   {attacker_iou_50 / all_data_num:.4f}\n")
            f.write(
                f"Pred any collision rate:    {pred_collision_rate / all_data_num:.4f}\n")
            f.write(
                f"Right GT collision rate:    {attacker_right_collision_rate / all_data_num:.4f}\n")
            f.write(
                f"Theoretical similarity(10): {tolerance_degree_10 / all_data_num:.4f}\n")
            f.write(
                f"Theoretical similarity(20): {tolerance_degree_20 / all_data_num:.4f}\n")
            f.write(
                f"Theoretical similarity(30): {tolerance_degree_30 / all_data_num:.4f}\n")
            f.write(
                f"Ideal attacker yaw distance: {ideal_yaw_dist_average / all_data_num:.4f}\n")
            f.write(
                f"Real attacker yaw distance: {real_yaw_dist_average / attacker_right_collision_rate:.4f}\n")
            f.write(
                f"Ideal similarity / Real collision rate / Real Similarity / All || (scenario number) \n")
            
            f.write(f"{lane_change_num}, {col_lane_change_num}, {sim_lane_change_num}, {lane_change_all}\n")
            f.write(f"{opposite_direction_num}, {col_opposite_direction_num}, {sim_opposite_direction_num}, {opposite_direction_all}\n")
            f.write(f"{rear_end_num:}, {col_rear_end_num}, {sim_rear_end_num}, {rear_end_all}\n")
            f.write(f"{junction_crossing_num:}, {col_junction_crossing_num}, {sim_junction_crossing_num}, {junction_crossing_all}\n")
            f.write(f"{LTAP_num}, {col_LTAP_num}, {sim_LTAP_num}, {LTAP_all}\n")
            
            # f.write(
            #     f"Ideal similarity / Real collision rate / Real Similarity / Real yaw distance || (ratio) \n")
            # f.write(
            #     f"{lane_change_num / lane_change_all:.4f} / {col_lane_change_num / lane_change_all:.4f} / {sim_lane_change_num / col_lane_change_num:.4f} / {lane_change_yaw_distance / col_lane_change_num:.4f}\n")
            # f.write(
            #     f"{opposite_direction_num / opposite_direction_all:.4f} / {col_opposite_direction_num / opposite_direction_all:.4f} / {sim_opposite_direction_num / col_opposite_direction_num:.4f} / {opposite_direction_yaw_distance / col_opposite_direction_num:.4f}\n")
            # f.write(
            #     f"{rear_end_num / rear_end_all:.4f} / {col_rear_end_num / rear_end_all:.4f} / {sim_rear_end_num / col_rear_end_num:.4f} / {rear_end_yaw_distance / col_rear_end_num:.4f}\n")
            # f.write(
            #     f"{junction_crossing_num / junction_crossing_all:.4f} / {col_junction_crossing_num / junction_crossing_all:.4f} / {sim_junction_crossing_num / col_junction_crossing_num:.4f} / {junction_crossing_yaw_distance / col_junction_crossing_num:.4f}\n")
            # f.write(
            #     f"{LTAP_num / LTAP_all:.4f} / {col_LTAP_num / LTAP_all:.4f} / {sim_LTAP_num / col_LTAP_num:.4f} / {LTAP_yaw_distance / col_LTAP_num:.4f}\n")
            f.write(
                "=======================================================\n")
            Average_Similarity = (lane_change_num + opposite_direction_num + rear_end_num + junction_crossing_num + LTAP_num) / all_data_num
            f.write(
                f"Average Similarity: {Average_Similarity:.4f}\n")
            f.write(
                f"Real collision:     {cr:.4f}\n")
            f.write(
                f"TP selection:       {all_tp_cls_acc / batch_num:.4f}\n")
        print("folder:", output_dir)
        csv_directory = output_dir + '/' + split + '_performance.csv'
        if not os.path.exists(csv_directory):
            print("not exist")
            with open(csv_directory, 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'Epoch', 'LR', 'Adam_weight_decay', 'Ego iou>0.5', 'FDE', 'Ego yaw', 'TP iou>0.5', 'TP DE', 'TP yaw', 'TP ID', 'CR', 'Sim'])
                #writer.writerow(['Time', 'LR', 'Adam_weight_decay', 'Positive_weight(RCNN)', 'Recall', 'Precision', 'F1', 'iou>0.5']) # RCNN version
        with open(csv_directory, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([output_dir.split('/')[-1], iter_epoch, self.lr, self.weight_decay, ego_iou_50 / all_data_num, sum_fde / all_data_num, all_yaw_distance / all_data_num, 
                             attacker_iou_50 / all_data_num, sum_attacker_de / all_data_num, ideal_yaw_dist_average / all_data_num,
                             all_tp_cls_acc / batch_num, cr, Average_Similarity])
            
            #     writer.writerow(['Time', 'Epoch', 'LR', 'Adam_weight_decay', 'Positive_weight', 'Ego iou>0.5', 'FDE', 'Ego yaw', 'TP iou>0.5', 'TP ID', 'CR'])
            # writer.writerow([output_dir.split('/')[-1], iter_epoch, self.lr, self.weight_decay, self.positive_weight, ego_iou_50 / all_data_num,
            #                  sum_fde / all_data_num, all_yaw_distance / all_data_num, attacker_iou_50 / all_data_num, all_tp_cls_acc / all_data_num, cr])
        self.write_log("Ego iou>0.5", ego_iou_50 / all_data_num, iter_epoch)
        #self.write_log("Ego iou>0.7", ego_iou_70 / all_data_num, iter_epoch)
        self.write_log("FDE", sum_fde / all_data_num, iter_epoch)
        self.write_log("Ego yaw", all_yaw_distance / all_data_num, iter_epoch)
        self.write_log("TP iou>0.5", attacker_iou_50 / all_data_num, iter_epoch)
        self.write_log("TP ID", all_tp_cls_acc / batch_num, iter_epoch)
        self.write_log("CR", cr, iter_epoch)

        # self.logger.add_scalar('Ego metric', {
        #                             'Ego iou>0.5': ego_iou_50 / all_data_num,
        #                           'FDE': sum_fde / all_data_num,
        #                           'Ego yaw': all_yaw_distance / all_data_num}, iter_epoch)
        
        # ego_iou = ego_iou_50 / all_data_num
        # fde = sum_fde / all_data_num
        # ego_yaw = all_yaw_distance / all_data_num
        # self.logger.add_scalar('Ego metric', f'Ego iou>0.5: {ego_iou}, FDE: {fde}, Ego yaw: {ego_yaw}', iter_epoch)

        
        return ego_iou_50 / all_data_num, attacker_iou_50 / all_data_num, all_tp_cls_acc / batch_num
        #return ego_iou_50 / all_data_num, Average_Similarity, all_tp_cls_acc / batch_num


    # function to convert the coordinates of trajectories from relative to world
    def convert_coord(self, traj, orig, rot):
        traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
        return traj_converted
    
    def build_polygon(self, ego_rec, plot=None, alpha_value=None):
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
        pred_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
        if plot:
            plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                        y_1, y_2, y_4, y_3, y_1], '-', alpha=alpha_value,  color=plot, markersize=3)
        return pred_polygon
    def calculate_F1(self, prediction, ground_truth):
        predicted_set = set(prediction)
        ground_truth_set = set(ground_truth)

        true_positives = len(predicted_set.intersection(ground_truth_set))
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)

        accuracy = true_positives / len(ground_truth_set) if len(ground_truth_set) != 0 else 0

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0

        # print(prediction, ground_truth)
        print("TP:", true_positives, "FP:", false_positives, "FN:", false_negatives)
        print("Precision:", precision, "Recall:", recall)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return accuracy, recall, precision, f1, true_positives, false_positives, false_negatives
    def other_visualization(self):
        self.model.eval()
        gt_trajectories = {}
        count = 0
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                batch_size = data.num_graphs
                horizon_sum = 0
                data_gt_list = []
                data_gt_yaw_list = []
                origs = data.orig.numpy()
                rots = data.rot.numpy()
                traj = data.obs_traj.reshape(batch_size, -1, 2)
                for i in range(batch_size):
                    data.horizon[i] -= 7
                    each_traj = data.y[horizon_sum * 2:(horizon_sum + int(data.horizon[i])) * 2].cpu().view(-1, 2).cumsum(axis=0) #int(data.y[horizon_sum + int(data.horizon[i]) - 1].cpu())
                    each_yaw = int(data.y_yaw[horizon_sum + int(data.horizon[i]) - 1].cpu())
                    horizon_sum += int(data.horizon[i])
                    data_gt_list.append(each_traj)
                    data_gt_yaw_list.append(each_yaw)
                gt_np = np.array(data_gt_list) # 128 
                gt_yaw_np = np.array(data_gt_yaw_list) # 128
                
                for s in range(batch_size):
                    now_scenario = np.array(data.seq_id)[s]
                    split_name = now_scenario.split('_')
                    now_ttc = split_name[7].split('-')[-1]
                    scenario_type = split_name[5]
                    #print(scenario_type)
                    # exit()
                    alpha_value = 1
                    route_color = "gray"
                    scatter_color = "red"
                    if_plot_with_dif_color = False
                    
                    if if_plot_with_dif_color:
                        if int(now_ttc) == 16:
                            route_color = "cyan"
                            scatter_color = "blue"
                        elif int(now_ttc) == 10:
                            route_color = "orange"
                            scatter_color = "red"
                        else:
                            continue
                    if_plot_with_dif_alpha = False
                    if if_plot_with_dif_alpha:
                        alpha_value = (17 - int(now_ttc)) / 7
                    # if_plot_
                    if scenario_type != "LC":
                        continue
                    count += 1
                    
                    
                    gt_trajectories[now_scenario] = self.convert_coord(gt_np[s], origs[s], rots[s])
                    # print(gt_np[s], traj[s])
                    traj_rot = self.convert_coord(traj[s], origs[s], rots[s])
                    # print(traj_rot, gt_trajectories[now_scenario])
                    # exit()
                    plt.plot(traj[s][:, 0], traj[s][:, 1], '-', alpha=alpha_value, color=route_color) # not convert
                    # plt.plot(traj_rot[:, 0], traj_rot[:, 1], '-', color='gray') # convert
                    # plt.plot(gt_trajectories[now_scenario][:, 0], gt_trajectories[now_scenario][:, 1], '-', color='gray') # future
                    
                    # plt.scatter(gt_trajectories[now_scenario][-1][0], gt_trajectories[now_scenario][-1][1], s=5, c="black")  # convert
                    plt.scatter(gt_np[s][-1][0], gt_np[s][-1][1], s=5, alpha=alpha_value, c=scatter_color)
                    #title = "Type: All " + "TTC: " + now_ttc
                    #plt.title(title)
                    # exit()
            print(count)
            #plt.show()
            

