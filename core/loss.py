# loss function for train the vector net

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from core.model.layers.scoring_and_selection import distance_metric
import numpy as np
import time

class VectorLoss(nn.Module):
    """
        The loss function for train vectornet, Loss = L_traj + alpha * L_node
        where L_traj is the negative Gaussian log-likelihood loss, L_node is the huber loss
    """
    def __init__(self, alpha=1.0, aux_loss=False, reduction='sum'):
        super(VectorLoss, self).__init__()

        self.alpha = alpha
        self.aux_loss = aux_loss
        if reduction in ["mean", "sum"]:
            self.reduction = reduction
        else:
            raise NotImplementedError("[VectorLoss]: The reduction has not been implemented!")

    def forward(self, pred, gt, aux_pred=None, aux_gt=None):
        batch_size = pred.size()[0]
        loss = 0.0

        l_traj = F.mse_loss(pred, gt, reduction='sum')

        if self.reduction == 'mean':
            l_traj /= batch_size

        loss += l_traj
        if self.aux_loss:
            # return nll loss if pred is None
            if not isinstance(aux_pred, torch.Tensor) or not isinstance(aux_gt, torch.Tensor):
                return loss
            assert aux_pred.size() == aux_gt.size(), "[VectorLoss]: The dim of prediction and ground truth don't match!"

            l_node = F.smooth_l1_loss(aux_pred, aux_gt, reduction=self.reduction)
            if self.reduction == 'mean':
                l_node /= batch_size
            loss += self.alpha * l_node
        return loss


class TNTLoss(nn.Module):
    """
        The loss function for train TNT, loss = a1 * Targe_pred_loss + a2 * Traj_reg_loss + a3 * Score_loss
    """
    def __init__(self,
                 lambda1,
                 lambda2,
                 lambda3,
                 temper=0.01,
                 aux_loss=False,
                 reduction='sum',
                 device=torch.device("cpu")):
        """
        lambda1, lambda2, lambda3: the loss coefficient;
        temper: the temperature for computing the score gt;
        aux_loss: with the auxiliary loss or not;
        reduction: loss reduction, "sum" or "mean" (batch mean);
        """
        super(TNTLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.aux_loss = aux_loss
        self.reduction = reduction
        self.temper = temper

        self.device = device

    def forward(self, pred_dict, gt_dict, aux_pred=None, aux_gt=None, classfication_pos_weight=10):
        """
            pred_dict: the dictionary containing model prediction,
                {
                    "target_prob":  the predicted probability of each target candidate,
                    "offset":       the predicted offset of the target position from the gt target candidate,
                    "traj_with_gt": the predicted trajectory with the gt target position as the input,
                    "traj":         the predicted trajectory without the gt target position,
                    "score":        the predicted score for each predicted trajectory,
                }
            gt_dict: the dictionary containing the prediction gt,
                {
                    "target_prob":  the one-hot gt of traget candidate;
                    "offset":       the gt for the offset of the nearest target candidate to the target position;
                    "y":            the gt trajectory of the target agent;
                }
        """
        start_time = time.time()
        batch_size = pred_dict['target_prob'].size()[0]
        loss = 0.0


        # # 12/7
        # self.lambda0 = 0.5 * 10
        # self.lambda1 = 0.25 # offset: x, y, yaw
        # self.lambda2 = 0.25 
        # self.lambda3 = 0.25 
        # self.lambda_yaw = 0.5 
        # 
        only_regression = False
        regress_on_multi_GT_fix_num = False
        regress_on_multi_GT_variable_num = True
        attacker_only_regression = False
        if_add_highest_score_pos = True
        multi_positive_sample = 50 #250
        RCNN_positive_sample = 50
        self.RCNN_threshold = 0.0 #0.9
        EGO_steer_angle = False # replace directly predict ego position angle
        object_detection_2module = False
        self.lambda_object_detection_2module = 0.00002 # 0.01 for IOU filter
        # classfication_pos_weight = 0#10
        self.RCNN_cls = 0.002#0.125 / classfication_pos_weight
        self.lambda_cls = 0#0.05 / classfication_pos_weight
        # self.lambda_cls = 0.25 # cross-entropy no weight
        # self.lambda_offset = 0.0004#1
        self.lambda_offset = 0.004#0.0004#0.04#1 # only regress on multi GT
        self.lambda_yaw = 1#2
        self.lambda_atr_yaw = 2 
        self.lambda_atr_offset = 0.04 #0.4 
        self.lambda_tp = 10
        self.lambda_aux = 0#0.005 

        self.lambda_highest_score_dist = 0.1
        
        if only_regression:
            self.lambda_cls = 0
            self.lambda_offset = 0.5
            self.lambda_yaw = 10
            self.lambda_atr_yaw = 10     #2 
            self.lambda_atr_offset = 0.5 # 1 #0.4 
            self.lambda_tp = 10
            self.lambda_aux = 0         #0.1#0.005 
            self.RCNN_cls = 0

        
        



        ############################# pos_weight #############################
        # compute target prediction loss
        # weight = torch.tensor([1.0, 2.0], dtype=torch.float, device=self.device)
        # cls_loss = F.cross_entropy(
        #     pred_dict['target_prob'].transpose(1, 2),
        #     gt_dict['target_prob'].long(),
        #     weight=weight,
        #     reduction='sum')
        # #w_p = torch.FloatTensor([5], device=self.device)
        # # w_p = torch.tensor([1748], dtype=torch.float, device=self.device)
        # # print("w_p:", w_p)
        # pos_weight = torch.tensor([2.0])
        # _, indices = gt_dict['target_prob'].topk(1, dim=1)
        # batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(1)]).T
        # each_loss = F.binary_cross_entropy(
        #     pred_dict['target_prob'], gt_dict['target_prob'].float(), reduction='none')
        # # index = indices[0][0].cpu()
        # # print(index)
        # # print("pred:", pred_dict['target_prob'][0][index - 2:index + 2])
        # # print("GT:", gt_dict['target_prob'][0][index - 2:index + 2])
        # # print("origin each loss:", each_loss[0][index - 2:index + 2])
        # each_loss[batch_idx, indices] *= 1
        # # print("after *2:", each_loss[0][index - 2:index + 2])
        # # print("origin loss:", F.binary_cross_entropy(
        # #     pred_dict['target_prob'], gt_dict['target_prob'].float(), reduction='sum'))
        # print("*2 loss:", torch.sum(each_loss))
        # cls_loss = torch.sum(each_loss)
        
        ######## cross_entropy no_weight #########
        # gt_index = torch.argmax(gt_dict['target_prob'], dim=1)
        # cls_loss = F.cross_entropy(pred_dict['target_prob'], gt_index, reduction='sum')
        ######## cross_entropy no_weight #########
        
        ######## binary_cross_entropy manual * pos_weight #########
        ######## ***** Mostly Used ***** #########
        if_cls_ranking = False
        if if_cls_ranking:
            _, indices = gt_dict['target_prob'].topk(multi_positive_sample, dim=1)
            batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(1)]).T
            each_loss = F.binary_cross_entropy(
                pred_dict['target_prob'], gt_dict['target_prob'].float(), reduction='none')
            each_loss[batch_idx, indices] *= classfication_pos_weight
            cls_loss = torch.sum(each_loss)
        else:
            cls_loss = torch.tensor(0)
        ######## ***** Mostly Used ***** #########
        ######## binary_cross_entropy manual * pos_weight #########


        
        ######## binary_cross_entropy manual * pos_weight, speed up by ChatGPT #########
        # _, indices = gt_dict['target_prob'].topk(1, dim=1)
        # batch_idx = torch.arange(0, batch_size, device=self.device).unsqueeze(1).expand(batch_size, 1)
        # raw_loss = F.binary_cross_entropy_with_logits(
        #     pred_dict['target_prob'], gt_dict['target_prob'].float(), reduction='none')
        # raw_loss[batch_idx, indices] *= 2
        # cls_loss = raw_loss.sum()
        ######## binary_cross_entropy manual * pos_weight, speed up by ChatGPT #########


        # your logits and targets
        # logits = pred_dict['target_prob']
        # targets = gt_dict['target_prob'].long()

        # # compute raw cross-entropy loss
        # raw_loss = F.cross_entropy(logits, targets.view(-1), reduction='none')

        # # get the indices of the top-1 predictions
        # _, top1_indices = logits.topk(1, dim=1)

        # # create a binary mask indicating the top-1 predictions
        # top1_mask = torch.zeros_like(logits, dtype=torch.float)
        # top1_mask.scatter_(1, top1_indices, 1)

        # # define a weight for the top-1 predictions
        # weight_for_top1 = torch.tensor([your_weight_for_top1], device=logits.device)

        # # apply the weight only to the top-1 predictions
        # weighted_loss = raw_loss * (1 + (top1_mask - 1) * weight_for_top1)

        # # compute the sum of weighted losses
        # cls_loss = torch.sum(weighted_loss)




        # w_p = torch.tensor([1], dtype=torch.float, device=self.device)
        # cls_loss = F.binary_cross_entropy_with_logits(
        #    pred_dict['target_prob'], gt_dict['target_prob'].float(), reduction='sum', pos_weight=w_p)
        ############################# pos_weight #############################


        # Target point, offset, yaw prediction
        # cls_loss = F.binary_cross_entropy(
        #     pred_dict['target_prob'], gt_dict['target_prob'].float(), reduction='sum')
        
        #################### Original Offset Predictor ####################
        # offset = pred_dict['offset'][gt_dict['target_prob'].bool()]
        # offset_loss = F.smooth_l1_loss(offset, gt_dict['offset'], reduction='sum')
        # offset_loss = F.l1_loss(offset, gt_dict['offset'], reduction='sum')
        #################### Original Offset Predictor ####################
        # loss += self.lambda1 * (cls_loss + offset_loss) / (1.0 if self.reduction == "sum" else batch_size)
        
        
        ###### RCNN classifier ######
        # _, indices = gt_dict['target_prob'].topk(RCNN_positive_sample, dim=1)
        # target_pred_se = gt_dict['candidate'][batch_idx, indices]
        # print(target_pred_se)
        # exit()
        # labels = torch.zeros((indices.shape), dtype=torch.long)
        # labels[indices] = 1

        # labels = torch.zeros((batch_size, gt_dict['target_prob'].shape[1]), dtype=torch.long, device=self.device)
        # labels[torch.arange(batch_size).unsqueeze(1), indices] = 1
        # #print(torch.sum(labels[0]))
        # labels_flat = labels.view(-1)
        
        if object_detection_2module:
            outputs_flat = pred_dict['RCNN_cls_clone'].view(-1, 2)
            RCNN_cls_2module_loss = self.lambda_object_detection_2module * pred_dict['RCNN_cls_2module_loss']
            # print("\nRCNN_cls_2module_loss:", RCNN_cls_2module_loss)
            loss += RCNN_cls_2module_loss
            
        else:
            RCNN_cls_2module_loss = 0
            outputs_flat = pred_dict['RCNN_cls'].view(-1, 2)
        
        # weight = torch.tensor([1.0, classfication_pos_weight], device=self.device)
        # RCNN_cls_loss = F.cross_entropy(outputs_flat, labels_flat, weight=weight, reduction='sum')
        # RCNN_cls_loss = self.RCNN_cls * RCNN_cls_loss
        # loss += RCNN_cls_loss
        outputs_flat = outputs_flat.view(batch_size, -1, 2) # 128, 525, 2
        predicted_labels = torch.argmax(outputs_flat, dim=2)
        # predicted_labels = torch.where(outputs_flat[..., 1] >= torch.tensor(self.RCNN_threshold, device=self.device),
        #                                 torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))

        #print("Predicted Labels:", np.sum(predicted_labels.cpu().numpy()))
        
        # labels = torch.zeros((batch_size, gt_dict['target_prob'].shape[1]), device=self.device)
        # labels[torch.arange(batch_size).unsqueeze(1), indices.squeeze(1)] = 1
        labels = gt_dict['target_prob']
        outputs = pred_dict['RCNN_cls'] # 128, 525, 2
        
        
        # print(labels.unsqueeze(2).shape, outputs.shape)
        debug = torch.cat((labels.unsqueeze(2), outputs), dim=2)
        # print(debug[0][:200])
        

        # for ix in range(batch_size):
        #     print(np.sum(gt_dict['target_prob'][ix].cpu().numpy()))
        
        #binary_labels = (labels == 1).float()
        positive_confidence = torch.index_select(outputs, dim=2, index=torch.tensor([1], device=outputs.device))
        positive_confidence = torch.squeeze(positive_confidence, dim=2)
        # print(positive_confidence.shape, positive_confidence) # 128, 525
        w_p = torch.tensor([classfication_pos_weight], dtype=torch.float, device=self.device)
        # print(sum(labels), positive_confidence)
        # exit()
        RCNN_cls_loss = F.binary_cross_entropy_with_logits(positive_confidence, labels.float(), reduction='sum', pos_weight=w_p)
        
        RCNN_cls_loss = self.RCNN_cls * RCNN_cls_loss
        loss += RCNN_cls_loss


        

        #print(labels.shape, labels[0][:400])
        ###### RCNN classifier ######
        
        if not only_regression:
            #################### Each Offset Predictor ####################
            _, indices = gt_dict['target_prob'].topk(RCNN_positive_sample, dim=1)
            sum_target_candidate_len = 0
            all_gt = torch.zeros(pred_dict['offset'].shape, device=self.device)
            ######################
            batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(RCNN_positive_sample)]).T
            target_pred_se = gt_dict['candidate'][batch_idx, indices]
            ######################
            for i in range(batch_size):
                pred_index_tuple = torch.where(predicted_labels[i] == 1)
                # print(len(pred_index_tuple[0]))
                #print(i, sum_target_candidate_len, gt_dict['target_candidate_lens'][i], indices.shape[1])
                temp = gt_dict['offset_each'][sum_target_candidate_len:sum_target_candidate_len + gt_dict['target_candidate_lens'][i]]
                temp_m = torch.zeros(temp.shape, device=self.device)
                if regress_on_multi_GT_variable_num:
                    for j in range(len(pred_index_tuple[0])):
                        if pred_index_tuple[0][j] >= gt_dict['target_candidate_lens'][i]:
                            # print("pred index > target candidate number")
                            continue
                        temp_m[pred_index_tuple[0][j]][0] = temp[pred_index_tuple[0][j]][0]
                        temp_m[pred_index_tuple[0][j]][1] = temp[pred_index_tuple[0][j]][1]
                    # print(temp_m)
                    # exit()
                    all_gt[i, :gt_dict['target_candidate_lens'][i], :] = temp_m
                elif regress_on_multi_GT_fix_num:
                    for j in range(indices.shape[1]): # 5
                        temp_m[indices[i][j]][0] = temp[indices[i][j]][0]
                        temp_m[indices[i][j]][1] = temp[indices[i][j]][1]
                    all_gt[i, :gt_dict['target_candidate_lens'][i], :] = temp_m
                else:
                    all_gt[i, :gt_dict['target_candidate_lens'][i], :] = temp
                sum_target_candidate_len += gt_dict['target_candidate_lens'][i]
                # print("\ntarget_pred_se:", target_pred_se[i])
                # print("regression:", temp[indices[i]])
                # print(target_pred_se[i] + temp[indices[i]])
            # offset_loss = F.l1_loss(pred_dict['offset'], all_gt, reduction='none')
            # total_loss = torch.sum(offset_loss)
            offset_loss = F.l1_loss(pred_dict['offset'], all_gt, reduction='sum')
            # print("RCNN_cls_loss:", RCNN_cls_loss, "offset_loss:", offset_loss)
            # print(pred_dict['offset'].shape, gt_dict['offset_each'].shape, gt_dict['y_yaw'].shape, gt_dict['target_candidate_lens'].shape, all_gt.shape)
            # print(pred_dict['offset'].shape)
            # exit()
            #################### Each Offset Predictor ####################
            cls_loss = self.lambda_cls * cls_loss
            loss += cls_loss
            offset_loss = self.lambda_offset * offset_loss
            loss += offset_loss

            #yaw_pred = pred_dict['target_pred_yaw'][gt_dict['target_prob'].bool()]# * 180 / np.pi
            yaw_pred = pred_dict['target_pred_yaw'].squeeze(1)

            horizon_sum = 0
            data_yaw_list = []
            data_last_point_list = []
            #print(data.y_ttc.shape) #640 (64*10)

            sorted_indices = torch.argsort(positive_confidence, descending=True)
            highest_score_dist_list = []

            for i in range(batch_size):
                gt_dict['ttc'][i] -= 7
                #print(i, horizon_sum, int(gt_dict['ttc'][i]), gt_dict['y_yaw'].shape)
                ##### debug for fix pred horizon
                #gt_dict['ttc'][i] = 3
                #####
                each_yaw = float(gt_dict['y_yaw'][horizon_sum + int(gt_dict['ttc'][i]) - 1].cpu())
                each_traj = gt_dict['y'][horizon_sum * 2:(horizon_sum + int(gt_dict['ttc'][i])) * 2].cpu().view(-1, 2).cumsum(axis=0)
                data_last_point_list.append(each_traj[-1])
                horizon_sum += int(gt_dict['ttc'][i])
                data_yaw_list.append([each_yaw])

                # most predict point VS gt ego pos
                highest_score_dist = torch.sqrt((gt_dict['candidate'][i][sorted_indices[i]][0][0] - each_traj[-1][0]) ** 2 + (gt_dict['candidate'][i][sorted_indices[i]][0][1] - each_traj[-1][1]) ** 2)
                highest_score_dist_list.append(highest_score_dist)

            stacked_tensor = torch.stack(data_last_point_list)

            
            gt_dict['y_yaw'] = torch.from_numpy(np.array(data_yaw_list)).float().cuda()
            #print(yaw_pred.shape, gt_dict['y_yaw'].shape) 64, 1
            gt_dict['y_yaw'] = gt_dict['y_yaw'] * np.pi / 180 + np.pi
            # print(yaw_pred.shape, gt_dict['y_yaw'].shape)
            yaw_loss = F.smooth_l1_loss(yaw_pred, gt_dict['y_yaw'], reduction='sum')
            yaw_loss = self.lambda_yaw * yaw_loss
            loss += yaw_loss
            # print(gt_dict['y_yaw'])
            # exit()

            

        else:
            #cls_loss = 0 * cls_loss
            #### if only regression ###
            horizon_sum = 0
            data_yaw_list = []
            data_last_point_list = []
            # data_y_tensor = torch.zeros((batch_size, 2))
            #print(gt_dict['y_yaw'])
            for i in range(batch_size):
                gt_dict['ttc'][i] -= 7
                #print(i, horizon_sum, int(gt_dict['ttc'][i]), horizon_sum + int(gt_dict['ttc'][i]) - 1, gt_dict['y'].shape, gt_dict['y_yaw'].shape)
                each_yaw = float(gt_dict['y_yaw'][horizon_sum + int(gt_dict['ttc'][i]) - 1].cpu())            
                #print(i, each_yaw, gt_dict['y_yaw'][horizon_sum: horizon_sum + int(gt_dict['ttc'][i])])
                #each_y = gt_dict['y'].reshape(-1, 2)[horizon_sum + int(gt_dict['ttc'][i]) - 1].cpu().float()

                each_traj = gt_dict['y'][horizon_sum * 2:(horizon_sum + int(gt_dict['ttc'][i])) * 2].cpu().view(-1, 2).cumsum(axis=0)
                data_last_point_list.append(each_traj[-1])
                # data_y_tensor[i] = gt_dict['y'].reshape(-1, 2)[horizon_sum + int(gt_dict['ttc'][i]) - 1].cumsum(axis=0)
                
                horizon_sum += int(gt_dict['ttc'][i])
                data_yaw_list.append([each_yaw])
            if EGO_steer_angle:
                ego_yaw_last_frame_list = [gt_dict['obs_yaw'][i].cpu().numpy().item() for i in range(7, gt_dict['obs_yaw'].shape[0], 8)]
                data_yaw_list = [item for sublist in data_yaw_list for item in sublist]

                data_yaw_list = [x - y for x, y in zip(data_yaw_list, ego_yaw_last_frame_list)]
                data_yaw_list = [angle + 360 if angle < 0 else angle for angle in data_yaw_list]
                gt_dict['y_yaw'] = torch.from_numpy(np.array(data_yaw_list)).float().cuda()
                gt_dict['y_yaw'] = gt_dict['y_yaw'] * np.pi / 180
            else:
                gt_dict['y_yaw'] = torch.from_numpy(np.array(data_yaw_list)).float().cuda()
                gt_dict['y_yaw'] = gt_dict['y_yaw'] * np.pi / 180 + np.pi
            #print(gt_dict['y_yaw'])
            # stacked_tensor: based on origin point, ego's GT pos
            stacked_tensor = torch.stack(data_last_point_list)
            # print(pred_dict['target_pred_yaw'].shape, gt_dict['y_yaw'].shape)
            
            
            #### if norm ####
            scale = 1 # 17 for 128 scenes
            stacked_tensor_scale = stacked_tensor / scale
            #### if norm ####
            
            
            #### xy ####
            offset_loss = F.l1_loss(pred_dict['offset'].squeeze(1), stacked_tensor_scale.cuda(), reduction='sum')
            offset_loss = self.lambda_offset * offset_loss
            loss += offset_loss
            #### xy ####

            #### L2 dist + theta ####
            # pred_xy = pred_dict['offset'].squeeze(1)
            # x = pred_xy[:, 0]
            # y = pred_xy[:, 1]
            # angle = torch.atan2(y, x).unsqueeze(-1)
            # for yaw_index in range(batch_size):
            #     angle[yaw_index] = angle[yaw_index] + np.pi if angle[yaw_index] < 0 else angle[yaw_index]
            # squared_sum = torch.sum(pred_xy ** 2, dim=-1)
            # euclidean_norm = torch.sqrt(squared_sum)
            # pred_regression = euclidean_norm.unsqueeze(-1)
            # pred_regression = torch.cat((pred_regression, angle), dim=1)

            # gt_x = data_y_tensor[:, 0]
            # gt_y = data_y_tensor[:, 1]
            # gt_angle = torch.atan2(gt_y, gt_x).unsqueeze(-1)
            # for yaw_index in range(batch_size):
            #     gt_angle[yaw_index] = gt_angle[yaw_index] + np.pi if gt_angle[yaw_index] < 0 else gt_angle[yaw_index]
            # gt_squared_sum = torch.sum(data_y_tensor ** 2, dim=-1)
            # gt_euclidean_norm = torch.sqrt(gt_squared_sum)
            # gt_regression = gt_euclidean_norm.unsqueeze(-1)
            # gt_regression = torch.cat((gt_regression, gt_angle), dim=1).to(self.device)
            
            # offset_loss = F.l1_loss(pred_regression, gt_regression, reduction='sum')
            # offset_loss = self.lambda_offset * offset_loss
            # loss += offset_loss
            #### L2 dist + theta ####


            yaw_loss = F.smooth_l1_loss(pred_dict['target_pred_yaw'].squeeze(1), gt_dict['y_yaw'].unsqueeze(1), reduction='sum')
            yaw_loss = self.lambda_yaw * yaw_loss
            loss += yaw_loss
            #### if only regression ###

        ############################# if predict TTC #####################################
        # ttc_loss = F.smooth_l1_loss(pred_dict['ttc'].float(), gt_dict['ttc'].float().unsqueeze(1), reduction='sum')
        # loss += self.lambda1 * ttc_loss
        ##################################################################

        # Attacker position offset of xy, yaw
        #print("in loss:", pred_dict['tar_offset_pred'].shape, gt_dict['atr_pos_offset'].shape) #(128, 2) (128, 1)
        
        # print(gt_dict['atr_pos_offset'].shape, pred_dict['offset'].shape)
        if attacker_only_regression:
            gt_dict['atr_pos_offset'] += stacked_tensor.cuda()
        atr_pos_loss = F.smooth_l1_loss(pred_dict['tar_offset_pred'].float(), gt_dict['atr_pos_offset'].float(), reduction='sum')
        atr_pos_loss = self.lambda_atr_offset * atr_pos_loss
        loss += atr_pos_loss

        if if_add_highest_score_pos:
            highest_score_dist_loss = self.lambda_highest_score_dist * sum(highest_score_dist_list).cuda()
            loss += highest_score_dist_loss
        else:
            highest_score_dist_loss = 0
        
        atr_yaw_pred = pred_dict['atr_yaw_pred']# * 180 / np.pi
        gt_dict['atr_yaw_offset'] = gt_dict['atr_yaw_offset'] * np.pi / 180
        atr_yaw_loss = F.smooth_l1_loss(atr_yaw_pred.float(), gt_dict['atr_yaw_offset'].unsqueeze(1).float(), reduction='sum')
        atr_yaw_loss = self.lambda_atr_yaw * atr_yaw_loss
        loss += atr_yaw_loss
        
        # Attacker ID prediction
        tp_cls_loss = F.binary_cross_entropy(
            pred_dict['tp_prob'], gt_dict['tp_gt'].float(), reduction='sum')
        tp_cls_loss = self.lambda3 * tp_cls_loss
        tp_cls_loss = self.lambda_tp * tp_cls_loss
        loss += tp_cls_loss
        ############################# original trajectory prediction #############################
        # # compute motion estimation loss
        # reg_loss = F.smooth_l1_loss(pred_dict['traj_with_gt'].squeeze(1), gt_dict['y'], reduction='sum')
        # loss += self.lambda2 * reg_loss
        # use L1 to calculate distance
        #print(pred_dict['traj_with_gt'].squeeze(1).shape, gt_dict['y'].shape) 64, 60(30*2)
        # # compute scoring gt and loss
        # score_gt = F.softmax(-distance_metric(pred_dict['traj'], gt_dict['y'])/self.temper, dim=-1)
        # score_loss = torch.sum(torch.mul(- torch.log(pred_dict['score']), score_gt))
        # loss += self.lambda3 * score_loss
        ############################# original trajectory prediction #############################

        ############################# regularization #############################
        reg_strength = 0.1
        reg_loss = 0.5 * reg_strength * torch.sum(pred_dict['final_feat']**2)
        # loss += reg_loss
        ############################# regularization #############################

        end_time = time.time()
        # print("time:", end_time - start_time)
        # exit()


        # aux loss
        if self.aux_loss:
            if not isinstance(aux_pred, torch.Tensor) or not isinstance(aux_gt, torch.Tensor):
                loss_dict = {"tar_cls_loss": cls_loss, "tar_offset_loss": offset_loss, "yaw_loss": yaw_loss,
                      "atr_pos_loss": atr_pos_loss, "atr_yaw_loss": atr_yaw_loss, "tp_cls_loss": tp_cls_loss, "reg_loss": reg_loss}
                return loss, loss_dict
            assert aux_pred.size() == aux_gt.size(), "[TNTLoss]: The dim of prediction and ground truth don't match!"
            aux_loss = F.smooth_l1_loss(aux_pred, aux_gt, reduction="sum")
            # loss += aux_loss / (1.0 if self.reduction == "sum" else batch_size)
            # loss += aux_loss / batch_size
            aux_loss = self.lambda_aux * aux_loss
            loss += aux_loss
        loss_dict = {"tar_cls_loss": cls_loss, "tar_offset_loss": offset_loss, "yaw_loss": yaw_loss,
                      "atr_pos_loss": atr_pos_loss, "atr_yaw_loss": atr_yaw_loss, "tp_cls_loss": tp_cls_loss,
                        "aux_loss": aux_loss, "reg_loss": reg_loss, "RCNN_loss": RCNN_cls_loss, "h_dist_loss": highest_score_dist_loss,
                        "RCNN_cls_2module_loss": RCNN_cls_2module_loss}
        return loss, loss_dict
