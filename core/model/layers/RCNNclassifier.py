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

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)  # 3 => 2
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # 3 => 4
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 1]).astype(np.float32))).view(1, 4).repeat(batchsize, 1)  # 3x3 => 2x2
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 2, 2)
        return x
    
class RCNNClassifier(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 m: int = 5,
                 device=torch.device("cpu"),
                 nearest_subgraph=1):
        """"""
        super(RCNNClassifier, self).__init__()
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
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        # self.conv1_with_lane_id = torch.nn.Conv1d(4, 64, 1)
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


    def forward(self, feat_in: torch.Tensor, tar_candidate: torch.Tensor, target_candidate_fea: torch.Tensor, nearest_subgraph, target_candidate_with_id):
        assert feat_in.dim() == 3, "[TNT-TargetPred]: Error input feature dimension"
        #batch_size = feat_in.shape[0]
        batch_size, n, _ = tar_candidate.size()
        
        if self.mode == 'RCNN_xy':
            #### L2 dist, unlock these 3 lines ####
            # squared_sum = torch.sum(tar_candidate ** 2, dim=-1)
            # euclidean_norm = torch.sqrt(squared_sum)
            # tar_candidate = euclidean_norm.unsqueeze(-1)
            #print(tar_candidate[0][:200])
            #### L2 dist ####

            #### theta, unlock these 3 lines ####
            # x = tar_candidate[:, :, 0]
            # y = tar_candidate[:, :, 1]
            # tar_candidate = torch.atan2(y, x).unsqueeze(-1)
            #### theta ####

            
            #### L2 dist + theta, unlock these 7 lines #### For now, the best performance ####
            x = tar_candidate[:, :, 0]
            y = tar_candidate[:, :, 1]
            angle = torch.atan2(y, x).unsqueeze(-1)
            squared_sum = torch.sum(tar_candidate ** 2, dim=-1)
            euclidean_norm = torch.sqrt(squared_sum)
            tar_candidate = euclidean_norm.unsqueeze(-1)
            tar_candidate = torch.cat((tar_candidate, angle), dim=2) # batch_size, max_point, 2
            #### L2 dist + theta ####
            tar_candidate = self.dimension_raising(tar_candidate)
            
            

            #### L2 dist + theta + xy, unlock these 8 lines ####
            # x = tar_candidate[:, :, 0]
            # y = tar_candidate[:, :, 1]
            # angle = torch.atan2(y, x).unsqueeze(-1)
            # squared_sum = torch.sum(tar_candidate ** 2, dim=-1)
            # euclidean_norm = torch.sqrt(squared_sum)
            # tar_candidate_l2 = euclidean_norm.unsqueeze(-1)
            # tar_candidate_l2 = torch.cat((tar_candidate_l2, angle), dim=2) # batch_size, max_point, 2
            # tar_candidate = torch.cat((tar_candidate_l2, tar_candidate), dim=2) # batch_size, max_point, 4
            #### L2 dist + theta ####

            feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), tar_candidate], dim=2) # 128, 525, 128
            result = self.classifier(feat_in_repeat)

        elif self.mode == 'PointNet':
            #
            x = tar_candidate.transpose(1, 2)
            
            n_pts = x.size()[2]
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = F.relu(self.bn1(self.conv1(x)))
            if self.feature_transform:
                trans_feat = self.fstn(x)
                x = x.transpose(2,1)
                x = torch.bmm(x, trans_feat)
                x = x.transpose(2,1)
            else:
                trans_feat = None

            pointfeat = x
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            tar_candidate = self.down_dimension(x).transpose(1, 2)
            
            feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), tar_candidate], dim=2) # 128, 525, 128
            result = self.classifier(feat_in_repeat)
            # x = torch.max(x, 2, keepdim=True)[0]
            # x = x.view(-1, 1024)
            # if self.global_feat:
            #     print("global_feat")
            #     # return x, trans, trans_feat
            # else:
            #     x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            #     tar_candidate = self.down_dimension(x).transpose(1, 2)
            #     # return torch.cat([x, pointfeat], 1), trans, trans_feat
            # #
        elif self.mode == 'local_feature':

            # xy, polar
            coordinate = 'polar' #'polar'
            # gloabl, batch, None
            norm = 'global'
            use_lane_id = False

            
            if coordinate == 'polar': 
                #### L2 dist + theta ####
                x = tar_candidate[:, :, 0]
                y = tar_candidate[:, :, 1]
                angle = torch.atan2(y, x).unsqueeze(-1)
                squared_sum = torch.sum(tar_candidate ** 2, dim=-1)
                euclidean_norm = torch.sqrt(squared_sum)
                tar_candidate = euclidean_norm.unsqueeze(-1)
                if norm == 'global':
                    # tar_candidate /= 265.3 # max value of whole dataset
                    tar_candidate /= 150
                tar_candidate = torch.cat((tar_candidate, angle), dim=2)
                #### L2 dist + theta ####
            
            if use_lane_id:
                lane_id_norm = True
                if lane_id_norm:
                    lane_id = target_candidate_with_id[:, :, 2] / 83
                    point_id = target_candidate_with_id[:, :, 3] / 122
                    lane_ids = torch.cat((lane_id.unsqueeze(2), point_id.unsqueeze(2)), dim=2)
                else:
                    lane_ids = target_candidate_with_id[:, :, 2:]
                
                # print(torch.max(target_candidate_with_id[:, :, 2]), torch.max(target_candidate_with_id[:, :, 3]))
                tar_candidate = torch.cat((tar_candidate, lane_ids), dim=2)
                x = tar_candidate.transpose(1, 2)
                x = F.relu(self.bn1(self.conv1_with_lane_id(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.bn3(self.conv3(x))
            else:
                x = tar_candidate.transpose(1, 2)
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.bn3(self.conv3(x))
            
            x = self.dimension_reduction_from_1024(x.transpose(1, 2))

            result = torch.cat([target_candidate_fea, x], dim=2)
            ego_final_fea = self.ego_final_feature(feat_in.repeat(1, n, 1))
            feat_in_repeat = torch.cat([ego_final_fea, result], dim=2)

            plot_tsne = False
            if plot_tsne:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2)
                features_2d = tsne.fit_transform(target_candidate_fea[0].cpu().detach().numpy())
                import matplotlib.pyplot as plt
                plt.scatter(features_2d[:, 0], features_2d[:, 1], marker='.')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.title('t-SNE for nearest subgraph ' + str(self.nearest_subgraph))
                plt.show()

            result = self.local_fea_classifier(feat_in_repeat)
        
        elif self.mode == 'local_feature_Pointnet':
            x = tar_candidate.transpose(1, 2)
            
            n_pts = x.size()[2]
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = F.relu(self.bn1(self.conv1(x)))
            if self.feature_transform:
                trans_feat = self.fstn(x)
                x = x.transpose(2,1)
                x = torch.bmm(x, trans_feat)
                x = x.transpose(2,1)
            else:
                trans_feat = None

            pointfeat = x
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            tar_candidate = self.down_dimension_to_local_fea(x).transpose(1, 2)
            feat_in_repeat = torch.cat([target_candidate_fea, tar_candidate], dim=2)
            result = self.local_fea_classifier(feat_in_repeat)

        # print("nearest_subgraph:", self.nearest_subgraph)
        # exit()
        result = F.softmax(result, 2)
        return result, feat_in_repeat
    def loss(self,
             feat_in: torch.Tensor,
             tar_candidate: torch.Tensor,
             candidate_gt: torch.Tensor,
             offset_gt: torch.Tensor,
             candidate_mask=None):
        """
        compute the loss for target prediction, classification gt is binary labels,
        only the closest candidate is labeled as 1
        :param feat_in: encoded feature for the target candidate, [batch_size, inchannels]
        :param tar_candidate: the target candidates for predicting the end position of the target agent, [batch_size, N, 2]
        :param candidate_gt: target prediction ground truth, classification gt and offset gt, [batch_size, N]
        :param offset_gt: the offset ground truth, [batch_size, 2]
        :param candidate_mask:
        :return:
        """
        print("targepoint loss")
        sys.exit()
        batch_size, n, _ = tar_candidate.size()
        _, num_cand = candidate_gt.size()

        assert num_cand == n, "The num target candidate and the ground truth one-hot vector is not aligned: {} vs {};".format(n, num_cand)

        # pred prob and compute cls loss
        tar_candit_prob, tar_offset_mean = self.forward(feat_in, tar_candidate, candidate_mask)

        # classfication loss in n candidates
        n_candidate_loss = F.cross_entropy(tar_candit_prob.transpose(1, 2), candidate_gt.long(), reduction='sum')

        # classification loss in m selected candidates
        _, indices = tar_candit_prob[:, :, 1].topk(self.M, dim=1)
        batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.M)]).T
        # tar_pred_prob_selected = F.normalize(tar_candit_prob[batch_idx, indices], dim=-1)
        # tar_pred_prob_selected = tar_candit_prob[batch_idx, indices]
        # candidate_gt_selected = candidate_gt[batch_idx, indices]
        # m_candidate_loss = F.binary_cross_entropy(tar_pred_prob_selected, candidate_gt_selected, reduction='sum') / batch_size

        # pred offset with gt candidate and compute regression loss
        # feat_in_offset = torch.cat([feat_in.squeeze(1), tar_candidate[candidate_gt]], dim=-1)
        # offset_loss = F.smooth_l1_loss(self.mean_mlp(feat_in_offset), offset_gt, reduction='sum')

        # isolate the loss computation from the candidate target offset prediction
        offset_loss = F.smooth_l1_loss(tar_offset_mean[candidate_gt.bool()], offset_gt, reduction='sum')

        # ====================================== DEBUG ====================================== #
        # # select the M output and check corresponding gt
        # _, indices = tar_candit_prob.topk(self.M, dim=1)
        # batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.M)]).T
        # tar_pred_prob_selected = F.normalize(tar_candit_prob[batch_idx, indices], dim=-1)
        # tar_pred_selected = tar_candidate[batch_idx, indices]
        # candidate_gt_selected = candidate_gt[batch_idx, indices]
        #
        # tar_candit_prob_cpu = tar_pred_prob_selected.detach().cpu().numpy()
        # candidate_gt_cpu = candidate_gt_selected.detach().cpu().numpy()
        #
        # print("\n[DEBUG]: tar_pred_prob_selected: \n{};\n[DEBUG]: candidate_gt_selected: \n{};".format(tar_candit_prob_cpu,
        #                                                                                                candidate_gt_cpu))
        # print("[DEBUG]: tar_pred_selected: \n{};\n[DEBUG]: tar_gt: \n{};".format(tar_pred_selected.detach().cpu().numpy(),
        #                                                                          tar_candidate[candidate_gt.bool()].detach().cpu().numpy()))
        # # check offset
        # tar_offset_mean_cpu = tar_offset_mean.detach().cpu().numpy()
        # offset_gt_cpu = offset_gt.detach().cpu().numpy()
        # print("[DEBUG]: tar_offset_mean: {};\n[DEBUG]: offset_gt: {};".format(tar_offset_mean_cpu, offset_gt_cpu))
        #
        # # check destination
        # dst_gt = tar_candidate[candidate_gt.bool()] + offset_gt
        # offset = torch.normal(self.mean_mlp(feat_in_prob), std=1.0)[batch_idx, indices]
        # dst_pred = tar_pred_selected + offset
        # print("[DEBUG]: dst_pred: \n{};\n[DEBUG]: dst_gt: \n{};".format(dst_pred.detach().cpu().numpy(),
        #                                                                 dst_gt.detach().cpu().numpy()))
        # ====================================== DEBUG ====================================== #
        # return n_candidate_loss + m_candidate_loss + offset_loss, tar_candidate[batch_idx, indices], tar_offset_mean[batch_idx, indices]
        return n_candidate_loss + offset_loss, tar_candidate[batch_idx, indices], tar_offset_mean[batch_idx, indices]
        # return m_candidate_loss + offset_loss, tar_candidate[batch_idx, indices], tar_offset_mean[batch_idx, indices]

    def inference(self,
                  feat_in: torch.Tensor,
                  tar_candidate: torch.Tensor,
                  candidate_mask=None):
        """
        output only the M predicted propablity of the predicted target
        :param feat_in:        the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate:  tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask: the mask of valid target candidate
        :return:
        """
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in: the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask:
        :return:
        """
        return self.forward(feat_in, tar_candidate, candidate_mask)


if __name__ == "__main__":
    batch_size = 4
    in_channels = 64
    N = 1000
    layer = TargetPred(in_channels)
    print("total number of params: ", sum(p.numel() for p in layer.parameters()))

    # forward
    print("test forward")
    feat_tensor = torch.randn((batch_size, 1, in_channels)).float()
    tar_candi_tensor = torch.randn((batch_size, N, 2)).float()
    tar_pred, offset_pred = layer(feat_tensor, tar_candi_tensor)
    print("shape of pred prob: ", tar_pred.size())
    print("shape of dx and dy: ", offset_pred.size())

    # loss
    print("test loss")
    candid_gt = torch.zeros((batch_size, N), dtype=torch.bool)
    candid_gt[:, 5] = 1.0
    offset_gt = torch.randn((batch_size, 2))
    loss = layer.loss(feat_tensor, tar_candi_tensor, candid_gt, offset_gt)

    # inference
    print("test inference")
    tar_candidate, offset = layer.inference(feat_tensor, tar_candi_tensor)
    print("shape of tar_candidate: ", tar_candidate.size())
    print("shape of offset: ", offset.size())


