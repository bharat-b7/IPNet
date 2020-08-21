'''
Takes in smpl parms and initialises a smpl object with optimizable params.
class th_SMPL currently does not take batch dim.
If code works:
    Author: Bharat
else:
    Author: Anonymous
'''
import torch
import torch.nn as nn
from lib.smpl_layer import SMPL_Layer


class th_batch_SMPL(nn.Module):
    def __init__(self, batch_sz, betas=None, pose=None, trans=None, offsets=None, faces=None, gender='male', device='cuda'):
        super(th_batch_SMPL, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, 300))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(batch_sz, 72))
        else:
            assert pose.ndim == 2
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890,3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.faces = faces
        self.gender = gender
        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender,
                               model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models')

    def forward(self):
        verts, jtr, tposed, naked = self.smpl(self.pose,
                                              th_betas=self.betas,
                                              th_trans=self.trans,
                                              th_offsets=self.offsets)
        return verts, jtr, tposed, naked


class th_SMPL(nn.Module):
    def __init__(self, betas=None, pose=None, trans=None, offsets=None, gender='male'):
        super(th_SMPL, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(300,))
        else:
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(72,))
        else:
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(3,))
        else:
            self.trans = nn.Parameter(trans)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(6890,3))
        else:
            self.offsets = nn.Parameter(offsets)

        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender,
                          model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models')

    def forward(self):
        verts, Jtr, tposed, naked = self.smpl(self.pose.unsqueeze(axis=0),
                                              th_betas=self.betas.unsqueeze(axis=0),
                                              th_trans=self.trans.unsqueeze(axis=0),
                                              th_offsets=self.offsets.unsqueeze(axis=0))
        return verts[0]
