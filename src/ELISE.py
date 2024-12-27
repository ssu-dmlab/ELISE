# not checked
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class Elise(nn.Module):
    def __init__(self,
                 num_layer: int,
                 c: float,
                 device: str,
                 input_dim: int,
                 **params):
        """_summary_

        Args:
            num_layer (int): # of layer
            c (float): ratio of personalized injection
            device (str): GPU or CPU device
            input_dim (int): dimension size
        """
        super(Elise, self).__init__()

        # hyper-parameters
        self.layer_num = num_layer
        self.c = c
        self.input_dim = input_dim

        # the alpha for final layer
        self.alpha = 1/(self.layer_num+1)

        # device
        self.device = device

    def build_structure(self, dataset):
        init_emb = dataset["init_emb"]
        # randomly initialized representation
        self.P_0_v = nn.Parameter(init_emb[1])
        self.P_0_u = nn.Parameter(init_emb[0])

        self.M_0_v = nn.Parameter(torch.zeros(
            init_emb[1].shape).to(self.device))
        self.M_0_u = nn.Parameter(torch.zeros(
            init_emb[0].shape).to(self.device))
        
        # list of l-th layer for main view
        self.P_l_v = [None] * (self.layer_num+1)
        self.M_l_v = [None] * (self.layer_num+1)
        self.P_l_v[0] = self.P_0_v
        self.M_l_v[0] = self.M_0_v

        self.P_l_u = [None] * (self.layer_num+1)
        self.M_l_u = [None] * (self.layer_num+1)
        self.P_l_u[0] = self.P_0_u
        self.M_l_u[0] = self.M_0_u

        # list of l-th layer for augmented view
        self.hat_P_l_v = [None] * (self.layer_num+1)
        self.hat_M_l_v = [None] * (self.layer_num+1)
        self.hat_P_l_v[0] = self.P_0_v
        self.hat_M_l_v[0] = self.M_0_v

        self.hat_P_l_u = [None] * (self.layer_num+1)
        self.hat_M_l_u = [None] * (self.layer_num+1)
        self.hat_P_l_u[0] = self.P_0_u
        self.hat_M_l_u[0] = self.M_0_u

        # candidate of final layer
        self.agg_P_v = None
        self.agg_M_v = None
        self.agg_P_u = None
        self.agg_M_u = None

        self.hat_agg_P_v = None
        self.hat_agg_M_v = None
        self.hat_agg_P_u = None
        self.hat_agg_M_u = None
        
        # bi-adjacency matrix
        self.A_pos = dataset["A_pos"]
        self.A_neg = dataset["A_neg"]
        self.B_pos = dataset["B_pos"]
        self.B_neg = dataset["B_neg"]

        # components of R-SVD
        self.svd_A = dataset["svd_A"]
        self.svd_B = dataset["svd_B"]
        
        
    def forward(self):
        """_summary_

        Args:
            None

        Returns <- concatenate_emb:
            user_emb (torch.tensor) : user_emb
            item_emb (torch.tensor) : item_emb
        """
        for l in range(1, self.layer_num+1):

            # ------------Singed Personalized Message Passing------------
            # U to V
            plv = (1-self.c) * (torch.mm(self.B_pos, self.P_l_u[l-1]) + torch.mm(self.B_neg, self.M_l_u[l-1])) \
                + (self.c * self.P_l_v[0])
            mlv = (1-self.c) * (torch.mm(self.B_pos,
                                         self.M_l_u[l-1]) + torch.mm(self.B_neg, self.P_l_u[l-1]))

            # V to U
            plu = (1-self.c) * (torch.mm(self.A_pos, self.P_l_v[l-1]) + torch.mm(self.A_neg, self.M_l_v[l-1])) \
                + (self.c * self.P_l_u[0])
            mlu = (1-self.c) * torch.mm(self.A_pos,
                                        self.M_l_v[l-1]) + torch.mm(self.A_neg, self.P_l_v[l-1])

            self.P_l_v[l] = plv
            self.M_l_v[l] = mlv

            self.P_l_u[l] = plu
            self.M_l_u[l] = mlu

            # ------------Refined Messaged Passing------------
            # U to V
            lram_plv = self.compute_rmp(direction_to='v',
                                        P=self.hat_P_l_u[l-1],
                                        M=self.hat_M_l_u[l-1])
            self.hat_P_l_v[l] = ((1-self.c) * lram_plv) + \
                (self.c * self.hat_P_l_v[0])

            lram_mlv = self.compute_rmp(direction_to='v',
                                        P=self.hat_M_l_u[l-1],
                                        M=self.hat_P_l_u[l-1])
            self.hat_M_l_v[l] = (1-self.c) * lram_mlv
            # V to U
            lram_plu = self.compute_rmp(direction_to='u',
                                        P=self.hat_P_l_v[l-1],
                                        M=self.hat_M_l_v[l-1])
            self.hat_P_l_u[l] = ((1-self.c) * lram_plu) + \
                (self.c * self.hat_P_l_u[0])

            lram_mlu = self.compute_rmp(direction_to='u',
                                        P=self.hat_M_l_v[l-1],
                                        M=self.hat_P_l_v[l-1])
            self.hat_M_l_u[l] = (1-self.c) * lram_mlu
        
        # ------------Layer-wise aggregation------------
        # Signed Personalized Message Passing
        self.agg_P_v = self.alpha * sum(self.P_l_v)
        self.agg_M_v = self.alpha * sum(self.M_l_v)

        self.agg_P_u = self.alpha * sum(self.P_l_u)
        self.agg_M_u = self.alpha * sum(self.M_l_u)

        # Refined Message Passing
        self.agg_hat_P_v = self.alpha * sum(self.hat_P_l_v)
        self.agg_hat_M_v = self.alpha * sum(self.hat_M_l_v)

        self.agg_hat_P_u = self.alpha * sum(self.hat_P_l_u)
        self.agg_hat_M_u = self.alpha * sum(self.hat_M_l_u)
        
        return self.concatnate_emb()

    def compute_rmp(self,
                    direction_to: str,
                    P: Tensor,
                    M: Tensor):
        """_summary_

        Args:
            direction_to (str): explicitly direction, you can choose to u -> v, v -> u 
            P (Tensor): node representation for positive sign link
            M (Tensor): node representation for negative sign link

        Returns:
            usve_p + usve_m(Tensor): sum of the positive and negative representations
        """

        if direction_to == 'v':
            svd = self.svd_B
        elif direction_to == 'u':
            svd = self.svd_A

        # P: positive sign / M: negative
        ve_p = svd['v_pos'].t() @ P  # (N x q).t() x (N x d) -> (q x d)
        sve_p = torch.diag(svd['s_pos']) @ ve_p  # (q x q) x (q x d) -> (q x d)
        # (M x q) x (q x d) -> (M x q), and vice versa
        usve_p = svd['u_pos'] @ sve_p

        # P: negative sign / M: positive
        ve_m = svd['v_neg'].t() @ M
        sve_m = torch.diag(svd['s_neg']) @ ve_m
        usve_m = svd['u_neg'] @ sve_m

        return usve_p + usve_m

    def concatnate_emb(self):
        """_summary_

        Returns:
            [u_emb, v_emb]: the listed embeddings of each node type
        """
        pm_v_cat = torch.cat([self.agg_P_v, self.agg_M_v], dim=-1)
        hat_pm_v_cat = torch.cat([self.agg_hat_P_v, self.agg_hat_M_v], dim=-1)
        v_emb = torch.cat([pm_v_cat, hat_pm_v_cat], dim=-1)  # N X N
        
        pm_u_cat = torch.cat([self.agg_P_u, self.agg_M_u], dim=-1)
        hat_pm_u_cat = torch.cat([self.agg_hat_P_u, self.agg_hat_M_u], dim=-1)
        u_emb = torch.cat([pm_u_cat, hat_pm_u_cat], dim=-1)  # M X M

        return  [u_emb, v_emb]
    

