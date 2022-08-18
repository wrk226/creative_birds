import os
import pdb
import copy
import numpy as np
import os.path as osp
from absl import app, flags
import math

import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F
from ..utils import mesh


class GateBlock(nn.Module):
    def __init__(self, gate_momentum=0.5, residual=True, do_forward = True):
        super(GateBlock, self).__init__()
        self.conv_share = nn.Conv1d(1024, 512, 1)
        self.gate_momentum = gate_momentum
        self.residual = residual
        self.do_forward = do_forward
        if self.do_forward:
            self.forward_layer = nn.Sequential(nn.Conv1d(512, 512, 1),
                                                nn.BatchNorm1d(512),
                                                nn.ReLU(),
                                                )

    def forward(self, s, t, gate_prev=None):
        g_specific = torch.sigmoid(self.conv_share(torch.cat([s, t], dim=1)))
        if gate_prev is not None:
            g_specific = g_specific * (1-self.gate_momentum) + gate_prev * self.gate_momentum
        s_diff = s*g_specific
        t_diff = t*g_specific
        if self.do_forward:
            s_diff = self.forward_layer(s_diff)
            t_diff = self.forward_layer(t_diff)
        if self.residual:
            s_diff = (s_diff + s)/2
            t_diff = (t_diff + t)/2
        return s_diff, t_diff, g_specific


class FuseBlock(nn.Module):
    def __init__(self):
        super(FuseBlock, self).__init__()
        self.conv_fuse = nn.Conv1d(1024, 512, 1)

    def forward(self, s, t, alpha_weight = 0):
        g_fuse = self.conv_fuse(torch.cat([s*(1+alpha_weight), t*(1-alpha_weight)], dim=1))
        return g_fuse

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=256):
        super(FlowHead, self).__init__()
        self.num_verts = output_dim
        self.conv1 = nn.Conv1d(input_dim, 2 * input_dim, 1)
        self.conv2 = nn.Conv1d(2 * input_dim, output_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))).view(x.shape[0], -1, 3)


class ShapePredictor(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128, output_dim=372):
        super(ShapePredictor, self).__init__()
        self.flow_head = FlowHead(input_dim=512, output_dim=output_dim)

    def forward(self, fused_inp):
        delta_flow = self.flow_head(fused_inp)
        # scale mask to balence gradients
        return delta_flow


class Dense_Gated_Net(nn.Module):
    def __init__(self, opts, num_half_verts):
        super(Dense_Gated_Net, self).__init__()
        self.opts = opts
        self.symmetric = opts.symmetric
        self.num_half_verts = num_half_verts
        self.init_networks()
        verts, faces = mesh.create_sphere(opts.subdivide)
        if self.symmetric:
            _, _, _, num_sym, _, _ = mesh.make_symmetric(verts, faces,
                                                                                                  axis=0)
            self.num_sym = num_sym
        self.flip = torch.ones(1, 3).cuda()
        self.flip[0, 1] = -1


    def init_networks(self, dim=512):
        # point_encoder
        self.projector = nn.Sequential(nn.Conv1d(350, 512, 1),
                                       nn.Tanh(),
                                       nn.Conv1d(512, 512, 1),
                                       nn.Tanh(),)
        self.gate_layer_1 = GateBlock()
        self.gate_layer_2 = GateBlock()
        self.gate_layer_3 = GateBlock()
        self.gate_layer_4 = GateBlock()
        self.gate_layer_5 = GateBlock()
        self.gate_layer_6 = GateBlock()
        self.gate_layer_7 = GateBlock()
        self.gate_layer_8 = GateBlock()

        self.fuse_layer = FuseBlock()
        self.shape_predictor = ShapePredictor(hidden_dim=dim * 2, input_dim=dim, output_dim=self.num_half_verts * 3)

    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip * V[-self.num_sym:]
                return torch.cat([V, V_left], 0)
            else:
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                return torch.cat([V, V_left], 1)
        else:
            return V
            # 337, 3

    def forward(self, source_feat, target_feat, mean_shape_half):
        self.target_feat = target_feat
        self.source_feat = source_feat
        self.delta_res = torch.autograd.Variable(torch.zeros(source_feat.shape[0], self.num_half_verts, 3),
                                                 requires_grad=True)
        self.delta_res = self.delta_res.cuda()
        bs, _, _ = self.delta_res.shape
        mean_shape_half = mean_shape_half.unsqueeze(dim=0).repeat(bs, 1, 1).transpose(1, 2)
        s = self.projector(self.source_feat)
        t = self.projector(self.target_feat)
        s, t, gate_1 = self.gate_layer_1(s, t)
        s, t, gate_2 = self.gate_layer_2(s, t, gate_prev = gate_1)
        s, t, gate_3 = self.gate_layer_3(s, t, gate_prev = gate_2)
        s, t, gate_4 = self.gate_layer_4(s, t, gate_prev = gate_3)
        s, t, gate_5 = self.gate_layer_5(s, t, gate_prev = gate_4)
        s, t, gate_6 = self.gate_layer_6(s, t, gate_prev = gate_5)
        s, t, gate_7 = self.gate_layer_7(s, t, gate_prev = gate_6)
        s, t, gate_8 = self.gate_layer_8(s, t, gate_prev = gate_7)
        fused_feat = nn.ReLU()(self.fuse_layer(s,t))
        delta_flow = self.shape_predictor(fused_feat)
        self.delta_res = self.delta_res + delta_flow

        self.deformed_shapes = self.symmetrize(self.delta_res + mean_shape_half.transpose(1, 2))


        outputs = {
            "deformed_shape": self.deformed_shapes,  # was called "delta_v"
        }

        return outputs
