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
                 save_folder: str = "",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True
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
            verbose=verbose
        )
        self.lr = lr
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
            multi_gpu=self.multi_gpu
        )

        # resume from model file or maintain the original
        if model_path:
            self.load(model_path, 'm')

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
            # ################################### DEBUG ################################### #
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

        flag = 0
        ideal_yaw_offset = 0
        vehicle_length = 4.7 * collision_ratio
        vehicle_width = 2 * collision_ratio

        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width

        all_iou = 0
        all_iou_no_0 = 0
        ego_iou_2 = 0
        ego_iou_50 = 0
        attacker_iou_2 = 0
        attacker_iou_50 = 0
        all_data_num = 0
        sum_fde = 0
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

        lane_change_only_col = 0
        junction_crossing_only_col = 0
        LTAP_only_col = 0
        opposite_direction_only_col = 0
        rear_end_only_col = 0

        col_lane_change_num = 0
        col_junction_crossing_num = 0
        col_LTAP_num = 0
        col_opposite_direction_num = 0
        col_rear_end_num = 0

        

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                
                batch_size = data.num_graphs
                #print(data.horizon, data.y.shape) data.y: 11746 // data.y_ttc: 1280
                horizon_sum = 0
                data_gt_list = []
                data_gt_yaw_list = []
                for i in range(batch_size):
                    #print(i, horizon_sum, int(data.horizon[i]))
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
                #seq_ids = data.seq_id.numpy()

                if gt_np is None:
                    compute_metric = False

                # inference and transform dimension
                if self.multi_gpu:
                    # out = self.model.module(data.to(self.device))
                    pred_target, offset_pred, yaw_pred, pred_target_point, pred_target_point_yaw, tar_offset_pred, atr_yaw_pred, tp_index = self.model.inference(data.to(self.device))
                else:
                    pred_target, offset_pred, yaw_pred, pred_target_point, pred_target_point_yaw, tar_offset_pred, atr_yaw_pred, tp_index = self.model.inference(data.to(self.device))

                #print("ttc_pred:", ttc_pred.shape) #128 1
                ################################################################
                gt_target = data.candidate_gt.view(-1, data.candidate_len_max[0])
                pred_offset = offset_pred[gt_target.bool()]
                #print(gt[:, -1, :].shape)
                gt_offset = data.offset_gt.view(-1, 2)
                pred_target_point = pred_target_point.cpu()
                pred_target_point_yaw = pred_target_point_yaw.cpu()
                tar_offset_pred = tar_offset_pred.cpu()
                atr_yaw_pred = atr_yaw_pred.cpu()
                #print(tar_offset_pred.shape, atr_yaw_pred.shape)
                pred_yaw = yaw_pred[gt_target.bool()].cpu()
                #print(pred_target.shape, pred_yaw.shape)
                #yaw_gt = data.y_yaw.unsqueeze(1).view(batch_size, -1, 1).cpu().numpy()[:, -1]#.unsqueeze(1)
                # print(pred_target.shape, gt_target.shape) 128, 1545
                # print(pred_offset.shape, gt_offset.shape) 128, 2
                # print(pred_yaw.shape, yaw_gt.shape) 128, 1
                #print(data.candidate.view(-1, data.candidate_len_max[0], 2).shape) 128, 1545, 2
                # sys.exit()
                _, indices = pred_target.topk(1, dim=1)
                batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(1)]).T
                target_pred_se, offset_pred_se = data.candidate.view(-1, data.candidate_len_max[0], 2)[batch_idx, indices].cpu(), offset_pred[batch_idx, indices].cpu()
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
                    if seq_ids[s].split('.')[1].split('_')[-1] in keys:
                        gt_attacker_list.append([keys.index(seq_ids[s].split('.')[1].split('_')[-1])])
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

                if not flag:
                    flag = 1
                    for s in range(batch_size):
                        all_data_num += 1
                        now_scenario = np.array(data.seq_id)[s]
                        #if now_scenario.split('_')[0] != '543':
                        #    continue
                        # sce_df = pd.read_csv('nuscenes_data/padding_trajectory/' + split + '/' + now_scenario + '.csv')
                        sce_df = pd.read_csv('nuscenes_data/trajectory/' + split + '/' + now_scenario + '.csv')
                        # vehicle_num = len(set(sce_df.TRACK_ID.values))
                        split_name = now_scenario.split('_')
                        initial_name = split_name[3] + '_' + split_name[4] + '_' + split_name[5] + '_' + split_name[6]
                        lane_feature = np.load('nuscenes_data/initial_topology/' + initial_name + '.npy', allow_pickle=True)
                        #route_num = now_scenario.split('_')[0]
                        #gt_ttc = now_scenario.split('_')[-2]
                        objs = sce_df.groupby(['TRACK_ID']).groups
                        keys = list(objs.keys())
                        del keys[keys.index('ego')]
                        print(now_scenario, tp_index[s].cpu().numpy()[0], data.horizon[s].cpu().numpy())
                        #print(tp_index[s].cpu().numpy(), keys)
                        if tp_index[s].cpu().numpy()[0] in keys:
                            guess_attacker_id = keys[tp_index[s].cpu().numpy()[0]]
                        elif len(keys) != 0:
                            ### random guess
                            guess_attacker_id = keys[0]
                        else:
                            continue

                        ttc = int(now_scenario.split('_')[7].split('-')[2])
                        condition = now_scenario.split('_')[5]
                        attacker_id = now_scenario.split('.')[1].split('_')[-1]
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

                        for features in lane_feature:
                            xs, ys = np.vstack((features[0][:, :2], features[0][-1, 3:5]))[
                                :, 0], np.vstack((features[0][:, :2], features[0][-1, 3:5]))[:, 1]
                            plt.plot(xs, ys, '-', color='lightgray')

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



                        gt_trajectories[now_scenario] = self.convert_coord(gt_np[s], origs[s], rots[s])
                        #print(pred_target_point[s])
                        
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            if str(track_id) == 'ego':
                                continue
                            if str(track_id) == str(guess_attacker_id) and str(track_id) == str(attacker_id):
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
                                plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                                    y_1, y_2, y_4, y_3, y_1], '-.',  color='purple', markersize=3)
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
                                plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                                    y_1, y_2, y_4, y_3, y_1], ':',  color='dimgray', markersize=3)

                        pred_target_point[s] = self.convert_coord(pred_target_point[s], origs[s], rots[s])
                        #print(atr_pos_offset[s])
                        atr_pos_offset[s] = np.matmul(np.linalg.inv(rots[s]), atr_pos_offset[s].T).T
                        #print(atr_pos_offset[s])
                        #sys.exit()
                        tar_offset_pred[s] = np.matmul(np.linalg.inv(rots[s]), tar_offset_pred[s].T).T
                        for i in range(k):
                            plt.plot(gt_trajectories[now_scenario][:, 0], gt_trajectories[now_scenario][:, 1], '-', color='black')
                        # print(math.acos(np.linalg.inv(rots[s])[0][0]))
                        # print(math.asin(np.linalg.inv(rots[s])[1][0]))
                        #print(ego_rot)
                        
                        #plt.plot(gt_trajectories[now_scenario], '-', color='gray')

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
                            y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                        ego_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])

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
                        plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                            y_1, y_2, y_4, y_3, y_1], '-',  color='brown', markersize=3)
                        attacker_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
                        
                        ego_rec = [pred_target_point[s][0][0],
                                pred_target_point[s][0][1], vehicle_width,
                                vehicle_length, pred_target_point_yaw[s][0][0] + 90 * np.pi / 180]
                        # ego_rec = [target_pred_se[s][0][0] + offset_pred_se[s][0][0] + origs[s][0],
                        #         target_pred_se[s][0][1] + offset_pred_se[s][0][1] + origs[s][1], vehicle_width,
                        #         vehicle_length, pred_yaw[s][0] + 90 * np.pi / 180]
                        
                        # ego_rec = [target_pred_se[s][0][0] + offset_pred_se[s][0][0],
                        #         target_pred_se[s][0][1] + offset_pred_se[s][0][1], vehicle_width,
                        #         vehicle_length, pred_yaw[s][0] + ego_rot * np.pi / 180]
                        # ego_rec = [target_pred_se[s][0][0] + offset_pred_se[s][0][0] + origs[s][0],
                        #         target_pred_se[s][0][1] + offset_pred_se[s][0][1] + origs[s][1], vehicle_width,
                        #         vehicle_length, pred_yaw[s][0] + ego_rot * np.pi / 180]
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
                        #print(now_scenario, gt_yaw_np[s], pred_yaw[s][0] * 180 / np.pi)
                        plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                            y_1, y_2, y_4, y_3, y_1], '-',  color='red', markersize=3)
                        pred_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])

                        # attacker_pred_rec = [pred_target_point[s][0][0] + tar_offset_pred[s][0], pred_target_point[s][0][1] + tar_offset_pred[s][1], vehicle_width
                        #             , vehicle_length, pred_target_point_yaw[s][0][0] + (90.0 + atr_yaw_pred[s]) * np.pi / 180]
                        attacker_pred_rec = [pred_target_point[s][0][0] + tar_offset_pred[s][0], pred_target_point[s][0][1] + tar_offset_pred[s][1], vehicle_width
                                    , vehicle_length, pred_target_point_yaw[s][0][0] + atr_yaw_pred[s] + (90.0) * np.pi / 180]

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
                        plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                            y_1, y_2, y_4, y_3, y_1], '-',  color='blue', markersize=3)
                        attacker_pred_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
                        
                        now_iou = ego_polygon.intersection(pred_polygon).area / ego_polygon.union(pred_polygon).area
                        attacker_iou = attacker_polygon.intersection(attacker_pred_polygon).area / attacker_polygon.union(attacker_pred_polygon).area
                        together_gt_iou = ego_polygon.intersection(attacker_polygon).area / ego_polygon.union(attacker_polygon).area
                        together_pred_iou = pred_polygon.intersection(attacker_pred_polygon).area / pred_polygon.union(attacker_pred_polygon).area
                        together_yaw_distance = (atr_yaw_pred[s].cpu().numpy() * 180 / np.pi + 360.0) if atr_yaw_pred[s].cpu().numpy() < 0 else atr_yaw_pred[s].cpu().numpy() * 180 / np.pi
                        together_yaw_distance = abs(together_yaw_distance - 360.0) if together_yaw_distance > 180 else together_yaw_distance
                        #print(pred_target_point_yaw[s][0][0].cpu().numpy() * 180 / np.pi, gt_yaw_np[s])
                        #print("Pred yaw dist:", together_yaw_distance)
                        #print(ideal_yaw_offset - together_yaw_distance)
                        if abs(ideal_yaw_offset - together_yaw_distance) < 10:
                            tolerance_degree_10 += 1
                        if abs(ideal_yaw_offset - together_yaw_distance) < 20:
                            tolerance_degree_20 += 1
                        if abs(ideal_yaw_offset - together_yaw_distance) < 30:
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
                        
                        
                        
                        if self.m > 1:
                            for t in range(1, self.m):
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
                                    y_1, y_2, y_4, y_3, y_1], ':',  color='orange', markersize=3)
                        
                        
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
                        gx = pred_target_point[s][0][0]
                        gy = pred_target_point[s][0][1]
                        gyaw = pred_target_point_yaw[s][0][0]# + 90 * np.pi / 180# + ego_rot * np.pi / 180
                        constant_v = math.sqrt((gx - start_x) ** 2 + (gy - start_y) ** 2) / min_t
                        sv = 4
                        gv = constant_v
                        time, x, y, all_yaw, all_v, a, j = quintic_polynomials_planner(
                            start_x, start_y, start_yaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt, min_t, max_t)
                        pos = np.array((x, y)).T
                        ego_yaw_v = np.array((all_v, all_yaw)).T #[1:]
                        #print(start_x, start_y, pos[0]) # Same

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
                        #print("pos:", pos.shape)
                        pd_pos = pos
                        tp_pos = tp_pos
                        vehicle_list = []
                        ##################### initial scenario is 4 frames
                        collision_moment = sce_df.TIMESTAMP.values[8+gt_ttc]
                        sce_df = sce_df[(sce_df.TIMESTAMP <= collision_moment)]
                        #####################
                        for track_id, remain_df in sce_df.groupby("TRACK_ID"):
                            
                            if str(track_id) == 'ego':
                                remain_df.iloc[8:, 3] = pd_pos[:, 0]
                                remain_df.iloc[8:, 4] = pd_pos[:, 1]
                                remain_df.iloc[8:, 2] = ego_yaw_v[:, 0]
                                remain_df.iloc[8:, 5] = ego_yaw_v[:, 1]
                            elif str(track_id) == str(guess_attacker_id):
                                #print(len(remain_df.X.values), tp_pos.shape, 8+gt_ttc)
                                remain_df.iloc[8:, 3] = tp_pos[:, 0]
                                remain_df.iloc[8:, 4] = tp_pos[:, 1]
                                remain_df.iloc[8:, 2] = tp_yaw_v[:, 0]
                                remain_df.iloc[8:, 5] = tp_yaw_v[:, 1]
                            vehicle_list.append(remain_df)
                        traj_df = pd.concat(vehicle_list)
                        traj_df.to_csv('output_csv/' + now_scenario + '.csv', index=False)
                        ################################
                        # moving foward for collision checking
                        frame_num = len(set(traj_df.TIMESTAMP.values))
                        further_list = []
                        further_df = traj_df
                        for track_id, remain_df in traj_df.groupby("TRACK_ID"):
                            dis_x = remain_df.iloc[frame_num-1, 3] - remain_df.iloc[frame_num-2, 3]
                            dis_y = remain_df.iloc[frame_num-1, 4] - remain_df.iloc[frame_num-2, 4]
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
                                # b = {'TIMESTAMP': remain_df.TIMESTAMP.values[-1] + (further_t + 1) * new_dt, 'TRACK_ID': track_id,
                                #                         'OBJECT_TYPE': remain_df.OBJECT_TYPE.values[0], 'X': x[further_t].cpu().numpy(), 'Y': y[further_t].cpu().numpy(),
                                #                         'YAW': all_yaw[further_t], 'V': all_v[further_t], 'CITY_NAME': remain_df.CITY_NAME.values[0]}
                                # further_df = further_df.append(b, ignore_index=True)
                                b = {'TIMESTAMP': [remain_df.TIMESTAMP.values[-1] + (further_t + 1) * new_dt * 500000], 'TRACK_ID': [track_id],
                                    'V': [all_v[further_t]], 'X': [x[further_t].cpu().numpy()], 'Y': [y[further_t].cpu().numpy()],
                                    'YAW': [all_yaw[further_t]]}
                                df_insert = pd.DataFrame(b)
                                further_df = pd.concat([further_df, df_insert], ignore_index=True)
                        further_df.to_csv('output_csv_moving_foward/' + now_scenario + '_foward.csv', index=False)
                        collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = self.cal_cr_and_similarity(further_df, attacker_id)
                        # collision_flag, real_yaw_dist, attacker_right_flag, record_yaw_distance = self.cal_cr_and_similarity(traj_df, attacker_id)
                        #print(collision_flag, real_yaw_dist)
                        if collision_flag:
                            pred_collision_rate += 1
                            if attacker_right_flag:
                                attacker_right_collision_rate += 1
                            real_yaw_dist = (real_yaw_dist + 360.0) if real_yaw_dist < 0 else real_yaw_dist
                            record_yaw_distance = (record_yaw_distance + 360.0) if record_yaw_distance < 0 else record_yaw_distance
                            record_yaw_distance = abs(record_yaw_distance - 360.0) if record_yaw_distance > 180 else record_yaw_distance
                            #print("REAL yaw dist:", real_yaw_dist)
                            #print("Record yaw dist:", record_yaw_distance) 
                            if condition == 'LTAP': 
                                LTAP_only_col += 1
                            elif condition == 'JC':
                                junction_crossing_only_col += 1
                            elif condition == 'HO':
                                opposite_direction_only_col += 1
                            elif condition == 'RE':
                                rear_end_only_col += 1
                            elif condition == 'LC':
                                lane_change_only_col += 1
                            if abs(ideal_yaw_offset - record_yaw_distance) < 30:
                                col_degree_30 += 1
                                if condition == 'LTAP': 
                                    col_LTAP_num += 1
                                elif condition == 'JC':
                                    col_junction_crossing_num += 1
                                elif condition == 'HO':
                                    col_opposite_direction_num += 1
                                elif condition == 'RE':
                                    col_rear_end_num += 1
                                elif condition == 'LC':
                                    col_lane_change_num += 1
                        


                        # Calculate Metrics
                        fde = math.sqrt((gx - gt_trajectories[now_scenario][-1][0]) ** 2 + (gy - gt_trajectories[now_scenario][-1][1]) ** 2)
                        sum_fde += fde
                        gt_yaw = gt_yaw_np[s]
                        p_yaw = gyaw.cpu().numpy() * 180 / np.pi
                        #print(gt_yaw, p_yaw)
                        now_yaw_distance = abs(gt_yaw - p_yaw)
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

                        if attacker_iou > 0.02:
                            attacker_iou_2 += 1
                        if attacker_iou > 0.5:
                            attacker_iou_50 += 1
                        # if together_gt_iou > 0.02:
                        #     gt_collision_rate += 1
                        # if together_pred_iou > 0.02:
                        #     pred_collision_rate += 1
                        
                        for i in range(pos.shape[0]):
                            plt.plot(pos[:, 0], pos[:, 1], '-', color='green')
                            plt.plot(tp_pos[:, 0], tp_pos[:, 1], '-', color='purple')

                        # plt.xlim(gt_np[s][0][0] - 35,
                        #             gt_np[s][0][0] + 35)
                        # plt.ylim(gt_np[s][0][1] - 35,
                        #             gt_np[s][0][1] + 35)
                        plt.xlim(gt_trajectories[now_scenario][-1][0] - 35,
                                    gt_trajectories[now_scenario][-1][0] + 35)
                        plt.ylim(gt_trajectories[now_scenario][-1][1] - 35,
                                    gt_trajectories[now_scenario][-1][1] + 35)
                        # gt_ttc = data.horizon[s].cpu().numpy()
                        # pred_ttc = ttc_pred[s].cpu().numpy()[0]
                        # ttc_distance = abs(gt_ttc - pred_ttc)
                        # ttc_distance_sum += ttc_distance
                        # title = 'TTC loss: ' + str(round(ttc_distance, 2))
                        #title = 'GT TTC:' + str(gt_ttc) + 'Pred TTC:' + str(pred_ttc) + 'loss:' + str(ttc_distance)
                        #plt.title(title)
                        plt.savefig('figures/' + np.array(data.seq_id)[s] + '.png')
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

        # compute the metric
        print("cls accuracy:", all_cls_acc / batch_num) 
        print("Top 5 cls accuracy:", all_top5_cls_acc / batch_num)
        #print("average IOU:", all_iou / all_data_num)
        print("2 percent ego data:", ego_iou_2 / all_data_num)
        print("50 percent ego data:", ego_iou_50 / all_data_num)
        #print("average IOU with threshold:", all_iou_no_0 / all_data_num)
        #print("average TTC loss:", ttc_distance_sum / all_data_num)
        print("average ego FDE:", sum_fde / all_data_num)
        print("ego yaw distance:", all_yaw_distance / all_data_num)
        # print("over 50 percent yaw distance:", over50_yaw_distance / collision_data_50)
        # print("below 50 percent yaw distance:", below50_yaw_distance / (all_data_num - collision_data_50))
        print("2 percent attacker data:", attacker_iou_2 / all_data_num)
        print("50 percent attacker data:", attacker_iou_50 / all_data_num)
        #print("GT collision rate:", gt_collision_rate / all_data_num)
        print("Pred collision rate:", pred_collision_rate / all_data_num) 
        print("Right GT collision rate:", attacker_right_collision_rate / all_data_num)
        print("similarity(10):", tolerance_degree_10 / all_data_num)
        print("similarity(20):", tolerance_degree_20 / all_data_num)
        print("similarity(30):", tolerance_degree_30 / all_data_num) 

        print("lane change ratio:", lane_change_num / lane_change_all, " Real collision:", col_lane_change_num / lane_change_only_col)
        print("opposite direction ratio:", opposite_direction_num / opposite_direction_all, " Real collision:", col_opposite_direction_num / opposite_direction_only_col)
        print("rear end ratio:", rear_end_num / rear_end_all, " Real collision:", col_rear_end_num / rear_end_only_col)
        print("junction crossing ratio:", junction_crossing_num / junction_crossing_all, " Real collision:", col_junction_crossing_num / junction_crossing_only_col)
        print("LTAP:", LTAP_num / LTAP_all, " Real collision:", col_LTAP_num / LTAP_only_col)

        print("Average Similarity:", (lane_change_num + opposite_direction_num + rear_end_num + junction_crossing_num + LTAP_num) / all_data_num,
              " Real collision:", (col_lane_change_num + col_opposite_direction_num + col_rear_end_num + col_junction_crossing_num + col_LTAP_num) / all_data_num)
        print("TP selection:", all_tp_cls_acc / batch_num)
        # if compute_metric:
        #     metric_results = get_displacement_errors_and_miss_rate(
        #         forecasted_trajectories,
        #         gt_trajectories,
        #         k,
        #         horizon,
        #         miss_threshold
        #     )
        #     print("[TNTTrainer]: The test result: {};".format(metric_results))

        # # plot the result
        # if plot:
        #     fig, ax = plt.subplots()
        #     for key in forecasted_trajectories.keys():
        #         print("key:", key)
        #         ax.set_xlim(-15, 15)
        #         show_pred_and_gt(ax, gt_trajectories[key], forecasted_trajectories[key])
        #         plt.pause(3)
        #         ax.clear()

        # # todo: save the output in argoverse format
        # if save_pred:
        #     for key in forecasted_trajectories.keys():
        #         forecasted_trajectories[key] = np.asarray(forecasted_trajectories[key])
        #     generate_forecasting_h5(forecasted_trajectories, self.save_folder)
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
        scenario_length = len(vehicle_list[0])
        for t in range(1, scenario_length):
            ego_x = ego_list[0].loc[t - 1, 'X']
            ego_x_next = ego_list[0].loc[t, 'X']
            ego_y = ego_list[0].loc[t - 1, 'Y']
            ego_y_next = ego_list[0].loc[t, 'Y']
            ego_vec = [ego_y_next - ego_y,
                                ego_x_next * (-1) - ego_x * (-1)]
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
                                        real_pred_x_next * (-1) - real_pred_x * (-1)]
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
                        plt.close()
                        fig,ax = plt.subplots()
                        
                        
                        ax.add_patch(ego_pg)
                        ax.add_patch(other_pg)
                        ax.set_xlim([1821,1835])
                        ax.set_ylim([2529,2544])
                        #plt.show()

                        print("COLLIDE! GT!!!!!!!! ", cur_iou)
                        
                        right_attacker_flag = 1
                        # Must collide with GT attacker
                        
                        real_yaw_distance = ego_angle * 180 / np.pi - other_angle * 180 / np.pi
                        record_yaw_distance = real_ego_angle - real_other_angle
                    
                if collision_flag:
                    break
            if collision_flag:
                break
        return collision_flag, real_yaw_distance, right_attacker_flag, record_yaw_distance
    def test_in_train(self,
             m=1,
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

        flag = 0
        vehicle_length = 4.7
        vehicle_width = 2

        collision_data_50 = 0
        all_data_num = 0
        batch_num = 0
        all_cls_acc = 0
        all_top5_cls_acc = 0
        all_tp_cls_acc = 0

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                batch_size = data.num_graphs
                horizon_sum = 0
                data_gt_list = []
                data_gt_yaw_list = []
                for i in range(batch_size):
                    #print(i, horizon_sum, int(data.horizon[i]))
                    each_traj = data.y[horizon_sum * 2:(horizon_sum + int(data.horizon[i])) * 2].cpu().view(-1, 2).cumsum(axis=0) #int(data.y[horizon_sum + int(data.horizon[i]) - 1].cpu())
                    each_yaw = int(data.y_yaw[horizon_sum + int(data.horizon[i]) - 1].cpu())
                    horizon_sum += int(data.horizon[i])
                    data_gt_list.append(each_traj)
                    data_gt_yaw_list.append(each_yaw)
                gt_np = np.array(data_gt_list) # 128 
                gt_yaw_np = np.array(data_gt_yaw_list) # 128 
                #gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()
                
                origs = data.orig.numpy()
                rots = data.rot.numpy()
                seq_ids = np.array(data.seq_id)
                #seq_ids = data.seq_id.numpy()

                if gt_np is None:
                    compute_metric = False

                # inference and transform dimension
                if self.multi_gpu:
                    # out = self.model.module(data.to(self.device))
                    pred_target, offset_pred, yaw_pred, ttc_pred, pred_target_point, pred_target_point_yaw = self.model.inference(data.to(self.device))
                else:
                    pred_target, offset_pred, yaw_pred, ttc_pred, pred_target_point, pred_target_point_yaw = self.model.inference(data.to(self.device))

                #print("ttc_pred:", ttc_pred.shape) #128 1
                ################################################################
                gt_target = data.candidate_gt.view(-1, data.candidate_len_max[0])
                pred_offset = offset_pred[gt_target.bool()]
                #print(gt[:, -1, :].shape)
                gt_offset = data.offset_gt.view(-1, 2)
                pred_target_point = pred_target_point.cpu()
                pred_target_point_yaw = pred_target_point_yaw.cpu()
                pred_yaw = yaw_pred[gt_target.bool()].cpu()
                _, indices = pred_target.topk(1, dim=1)
                batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(1)]).T
                target_pred_se, offset_pred_se = data.candidate.view(-1, data.candidate_len_max[0], 2)[batch_idx, indices].cpu(), offset_pred[batch_idx, indices].cpu()
                #print(target_pred_se.shape, offset_pred_se.shape) 128, 1, 2
                _, indices_GT = data.candidate_gt.view(-1, data.candidate_len_max[0]).topk(1, dim=1) #128, 1748 -> 128, 1
                #print("PRED:", indices.shape, indices) #128, 1
                batch_cls_acc = accuracy_score(indices_GT.cpu(), indices.cpu())
                all_cls_acc += batch_cls_acc
                batch_num += 1

                batch_top5_cls_acc = top_k_accuracy_score(indices_GT.flatten().cpu().numpy(), pred_target.cpu().numpy(), k=5, labels=np.arange(data.candidate_len_max[0].cpu()))
                _, indices_5 = pred_target.topk(5, dim=1)
                all_top5_cls_acc += batch_top5_cls_acc
                if not flag:
                    flag = 1
                    for s in range(batch_size):
                        all_data_num += 1
                        now_scenario = np.array(data.seq_id)[s][0]
                        seq_id = seq_ids[s]
                        gt_trajectories[now_scenario] = self.convert_coord(gt_np[s], origs[s], rots[s])
                        #print(pred_target_point[s])
                        pred_target_point[s] = self.convert_coord(pred_target_point[s], origs[s], rots[s])
                        #print(pred_target_point[s])

                        for i in range(k):
                            plt.plot(gt_trajectories[now_scenario][:, 0], gt_trajectories[now_scenario][:, 1], '-', color='black')

                        ego_rec = [gt_trajectories[now_scenario][-1][0], gt_trajectories[now_scenario][-1][1], vehicle_width
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
                        plt.plot([x_1 , x_2 , x_4 , x_3 , x_1 ], [
                            y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                        
                        ego_rec = [pred_target_point[s][0][0],
                                pred_target_point[s][0][1], vehicle_width,
                                vehicle_length, pred_target_point_yaw[s][0][0] + 90 * np.pi / 180]

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
                        #print(now_scenario, gt_yaw_np[s], pred_yaw[s][0] * 180 / np.pi)
                        now_iou = get_iou([gt_trajectories[now_scenario][-1][0], gt_trajectories[now_scenario][-1][1], vehicle_width, vehicle_length],
                                              [pred_target_point[s][0][0],
                                pred_target_point[s][0][1], vehicle_width,vehicle_length])
                        if now_iou > 0.5:
                            collision_data_50 += 1

        return collision_data_50 / all_data_num

    # function to convert the coordinates of trajectories from relative to world
    def convert_coord(self, traj, orig, rot):
        traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
        return traj_converted

