# trajectory prediction layer of TNT algorithm with Binary CE Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys
from core.model.layers.utils import masked_softmax
from core.model.layers.basic_module import MLP
from sklearn import preprocessing

class TrajEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 device=torch.device("cpu")):
        """"""
        super(TrajEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.before_concat_dim = 32

        self.device = device

        self.class_list = ["junction_crossing", "LTAP", "lane_change", "opposite_direction", "rear_end"]
        self.class_num = len(self.class_list)
        # self.label_emb = nn.Embedding(self.class_num, self.class_num)

        self.num_layers = 1
        self.dropout = 0

        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim, self.num_layers, dropout=self.dropout
        )

        self.spatial_embedding = nn.Linear(2, hidden_dim)



        self.LSTMencoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            # nn.ReLU(),
            # nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout),
            # nn.ReLU(),
        )
        
        self.label_emb = nn.Sequential(
            nn.Embedding(self.class_num, self.class_num),
            nn.ReLU(),
            nn.Linear(self.class_num, self.before_concat_dim),
            nn.ReLU(),
        )

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

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.hidden_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.hidden_dim).cuda()
        )

        
    def forward(self, data, ttc, one_hot, batch_size):
        one_hot = self.label_emb(one_hot)#.unsqueeze(1)

        obs_traj = data.obs_traj.reshape(batch_size, 8, 2)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch_size, self.hidden_dim
        )
        state_tuple = self.init_hidden(batch_size)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        target_feat = final_h.squeeze(0)


        ### concat TTC version ### modify after hank
        ttc = self.ttc_emb(ttc.unsqueeze(1))
        t = torch.cat([one_hot, ttc], dim=1)
        target_feat = self.relu(target_feat).squeeze(1)
        final = torch.cat([target_feat, t], dim=1)
        final = self.down_dimension(final).unsqueeze(1)
        ### concat TTC version ###

        return final
