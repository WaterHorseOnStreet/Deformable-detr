# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
#from turtle import hideturtle
import torch
from torch import nn

from util.misc import NestedTensor


class SupportProjector(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
        channel_list = [256,512,1024,2048]
        self.input_proj_local = []
        self.local_post_1  = nn.Conv2d(channel_list[0], hidden_dim//4, kernel_size=16)
        self.local_post_2  = nn.Conv2d(channel_list[1], hidden_dim//4, kernel_size=8)
        self.local_post_3  = nn.Conv2d(channel_list[2], hidden_dim//4, kernel_size=4)
        self.local_post_4  = nn.Conv2d(channel_list[3], hidden_dim//4, kernel_size=2)
        self.init_weights()

    def forward(self, support_feature):
        transformer_input_local = None
        transformer_input_local_pos = None

        for i,s in enumerate(support_feature):
            src_l = s.decompose()[0]
            
            if i == 0:
                src_l = self.local_post_1(src_l)
            elif i == 1:
                src_l = self.local_post_2(src_l)
            elif i == 2:
                src_l = self.local_post_3(src_l)
            else:
                src_l = self.local_post_4(src_l)


            src_l = src_l.reshape(src_l.shape[0],src_l.shape[1],src_l.shape[2]*src_l.shape[3])
            if i == 0:
                transformer_input_local = src_l
            else:
                transformer_input_local = torch.cat([transformer_input_local,src_l], dim=1)
        transformer_input_local = transformer_input_local.permute(2,0,1)

        return transformer_input_local

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.local_post_1.weight)
        torch.nn.init.xavier_uniform_(self.local_post_2.weight)
        torch.nn.init.xavier_uniform_(self.local_post_3.weight)
        torch.nn.init.xavier_uniform_(self.local_post_4.weight)


def build_support_projector(args):
    hidden_dim = args.hidden_dim

    return SupportProjector(hidden_dim)
