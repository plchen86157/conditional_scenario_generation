# trajectory prediction layer of TNT algorithm with Binary CE Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys
from core.model.layers.utils import masked_softmax
from core.model.layers.basic_module import MLP
from sklearn import preprocessing

class FinalFeat(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 device=torch.device("cpu")):
        """"""
        super(FinalFeat, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.before_concat_dim = 32

        self.device = device

        self.class_list = ["junction_crossing", "LTAP", "lane_change", "opposite_direction", "rear_end"]
        self.class_num = len(self.class_list)
        # self.label_emb = nn.Embedding(self.class_num, self.class_num)
        self.label_emb = nn.Sequential(
            nn.Embedding(self.class_num, self.class_num),
            nn.ReLU(),
            nn.Linear(self.class_num, self.before_concat_dim),
            nn.ReLU(),
        )
        # self.down_dimension = nn.Sequential(
        #     MLP(in_channels + 2, hidden_dim, hidden_dim),
        #     nn.Linear(hidden_dim, 1)
        # )
        self.down_dimension = nn.Sequential(
            # nn.Linear(self.in_channels + 1 + 5, hidden_dim), ### concat TTC version ###
            # nn.Linear(self.in_channels + 5, hidden_dim), ### add TTC version ###
            nn.Linear(self.in_channels + self.before_concat_dim + self.before_concat_dim, hidden_dim), ### concat TTC version ### modify after hank
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.ttc_emb = nn.Sequential(
            nn.Linear(1, self.before_concat_dim),
            nn.ReLU(),
        )

        ##################################################################
        self.observe_frame = 8
        ##################################################################


    def forward(self, target_feat, ttc, one_hot, class_num):
        target_feat.requires_grad_(True)
        one_hot = self.label_emb(one_hot)#.unsqueeze(1)

        # each_softmax = False
        # if each_softmax:
        #     target_feat = F.softmax(target_feat, dim=-1)
        #     one_hot = F.softmax(one_hot, dim=-1)
        #     t = torch.cat([one_hot, ttc.unsqueeze(1)], dim=1).unsqueeze(1)
        # else:
        #     t = torch.cat([one_hot, (ttc/10).unsqueeze(1)], dim=1).unsqueeze(1)
        
        ### add TTC version ###
        # ttc_in_repeat = ttc.unsqueeze(1).unsqueeze(1).repeat(1, 1, 64) + target_feat
        # final = torch.cat([ttc_in_repeat, one_hot.unsqueeze(1)], dim=2)
        # final = self.down_dimension(final.squeeze(1)).unsqueeze(1)
        ### add TTC version ###
        
        ### concat TTC version ###
        # t = torch.cat([one_hot, ttc.unsqueeze(1)], dim=1).unsqueeze(1)
        # target_feat = self.relu(target_feat)
        # final = torch.cat([target_feat, t], dim=2)
        # final = self.down_dimension(final.squeeze(1)).unsqueeze(1)
        ### concat TTC version ###

        ### concat TTC version ### modify after hank
        ttc = self.ttc_emb(ttc.unsqueeze(1))
        t = torch.cat([one_hot, ttc], dim=1)
        target_feat = self.relu(target_feat).squeeze(1)
        final = torch.cat([target_feat, t], dim=1)
        final = self.down_dimension(final).unsqueeze(1)
        ### concat TTC version ###

        return final
