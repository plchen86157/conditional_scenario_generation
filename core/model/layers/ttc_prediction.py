# trajectory prediction layer of TNT algorithm with Binary CE Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys
from core.model.layers.utils import masked_softmax
from core.model.layers.basic_module import MLP

class Encoder(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=1, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        ##################################################################
        ##################################################################
        ##################################################################
        self.observe_frame = 8
        ##################################################################
        ##################################################################
        ##################################################################

        self.spatial_embedding = nn.Linear(self.observe_frame, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        # observe 10 frames
        batch = obs_traj.reshape(-1, self.observe_frame).shape[0] #64, 10
        
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, self.observe_frame))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        #print(obs_traj_embedding.shape, obs_traj_embedding) # 1, 64, 64
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h

class TTCPred(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 m: int = 50,
                 device=torch.device("cpu")):
        """"""
        super(TTCPred, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.M = m          # output candidate target

        self.device = device

        self.encoder = Encoder()

        
        # self.ttc_pred = nn.Sequential(
        #     MLP(in_channels, hidden_dim, hidden_dim),
        #     nn.Linear(hidden_dim, 1)
        # )
        
        ##################################################################
        self.observe_frame = 8
        ##################################################################
        self.ttc_pred = nn.Sequential(
            MLP(self.observe_frame, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, y_ttc):
        y_ttc.requires_grad_(True)
                
        #final_h = self.encoder(y_ttc)
        y_ttc = y_ttc.reshape(-1, self.observe_frame)#.unsqueeze(1)
        #print(y_ttc.shape)
        ttc_pred = self.ttc_pred(y_ttc)
        #print(ttc_pred.shape, ttc_pred) 64, 1

        #return final_h[0]
        
        return ttc_pred

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

