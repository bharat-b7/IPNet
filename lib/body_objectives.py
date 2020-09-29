"""
Objectives used in single mesh registrations
Original author: Garvita
Edited by: Bharat
"""

import os
from os.path import join, split, exists
import sys
import numpy as np
import ipdb
import torch
from psbody.mesh import Mesh
import pickle as pkl
# from psbody.smpl import load_model
# from psbody.smpl.serialization import backwards_compatibility_replacements
from lib.smpl_paths import ROOT
from lib.torch_functions import batch_sparse_dense_matmul
# from lib.mesh_distance import point_to_surface_vec, batch_point_to_surface_vec_signed

HAND_VISIBLE = 0.2

part2num = {
'global': 0, 'leftThigh': 1, 'rightThigh': 2, 'spine': 3, 'leftCalf': 4, 'rightCalf': 5, 'spine1': 6, 'leftFoot': 7,
    'rightFoot': 8, 'spine2': 9, 'leftToes': 10, 'rightToes': 11, 'neck': 12, 'leftShoulder': 13, 'rightShoulder': 14,
    'head': 15, 'leftUpperArm': 16, 'rightUpperArm': 17, 'leftForeArm': 18, 'rightForeArm': 19, 'leftHand': 20,
    'rightHand': 21, 'leftFingers': 22, 'rightFingers': 23
}

def get_prior_weight(no_right_hand_batch, no_left_hand_batch):
    pr_w = np.ones((len(no_right_hand_batch), 69)).astype('float32')

    #for (no_right_hand, no_left_hand) in zip(no_right_hand_batch,no_left_hand_batch )
    for i in range(len(no_right_hand_batch)):
        if no_right_hand_batch[i]:
            pr_w[i, (part2num['rightFingers'] - 1) * 3:part2num['rightFingers'] * 3] = 1e5
            pr_w[i, (part2num['rightHand'] - 1) * 3:part2num['rightHand'] * 3] = 1e3

        if no_left_hand_batch[i]:
            pr_w[i, (part2num['leftFingers'] - 1) * 3:part2num['leftFingers'] * 3] = 1e5
            pr_w[i,(part2num['leftHand'] - 1) * 3:part2num['leftHand'] * 3] = 1e3

        pr_w[i, (part2num['rightToes'] - 1) * 3:part2num['rightToes'] * 3] = 1e3
        pr_w[i, (part2num['leftToes'] - 1) * 3:part2num['leftToes'] * 3] = 1e3
    #pr_w = np.ones((len(no_right_hand_batch), 69)).astype('float32')
    return  torch.from_numpy(pr_w)


def torch_pose_obj_data(batch_size=1):
    """
    Keypoint operators on SMPL verts.
    """
    body25_reg = pkl.load(open('assets/body25_regressor.pkl', 'rb'), encoding="latin1").T
    face_reg = pkl.load(open(join(ROOT, 'assets/face_regressor.pkl'), 'rb'), encoding="latin1").T
    hand_reg = pkl.load(open(join(ROOT, 'assets/hand_regressor.pkl'), 'rb'), encoding="latin1").T
    body25_reg_torch = torch.sparse_coo_tensor(body25_reg.nonzero(), body25_reg.data , body25_reg.shape)
    face_reg_torch = torch.sparse_coo_tensor(face_reg.nonzero(), face_reg.data , face_reg.shape)
    hand_reg_torch = torch.sparse_coo_tensor(hand_reg.nonzero(), hand_reg.data , hand_reg.shape)

    return torch.stack([body25_reg_torch]*batch_size), torch.stack([face_reg_torch]*batch_size),\
           torch.stack([hand_reg_torch]*batch_size)

def batch_get_pose_obj(th_pose_3d, smpl,  init_pose=False):
    """
    Comapre landmarks/keypoints ontained from the existing SMPL against those observed on the scan.
    Naive implementation as batching currently implies just looping.
    """
    batch_size = len(th_pose_3d)
    verts, _, _, _ = smpl.forward()
    J, face, hands = smpl.get_landmarks()

    J_observed = torch.stack([th_pose_3d[i]['pose_keypoints_3d'] for i in range(batch_size)]).cuda()
    face_observed = torch.stack([th_pose_3d[i]['face_keypoints_3d'] for i in range(batch_size)]).cuda()
    
    # Bharat: Why do we need to loop? Shouldn't we structure th_pose_3d as [key][batch, ...] as opposed to current [batch][key]?
    # This would allow us to remove the loop here.
    hands_observed = torch.stack(
        [torch.cat((th_pose_3d[i]['hand_left_keypoints_3d'], th_pose_3d[i]['hand_right_keypoints_3d']), dim=0) for i in
        range(batch_size)]).cuda()

    idx_mask = hands_observed[:, :, 3] < HAND_VISIBLE
    hands_observed[:, :, :3][idx_mask] = 0.

    if init_pose:
        pose_init_idx = torch.LongTensor([0, 2, 5, 8, 11])
        return (((J[:, pose_init_idx, : ] - J_observed[:, pose_init_idx, : 3])
         *J_observed[:, pose_init_idx,  3].unsqueeze(-1)) ** 2).mean()
    else:
        return ( (((J - J_observed[:, :, :3]) *J_observed[:, :, 3].unsqueeze(-1))**2).mean() +\
                   (((face - face_observed[:, :,: 3]) *face_observed[:, :,  3].unsqueeze(-1))**2).mean() +\
                   (((hands - hands_observed[:, :, :3]) *hands_observed[:, :,  3].unsqueeze(-1))**2).mean() ).unsqueeze(0)/3
        # return (((J - J_observed[:, :, :3]) *J_observed[:, :, 3].unsqueeze(-1))**2).mean().unsqueeze(0)   #only joints

def get_pose_obj(pose_3d, smpl):
    # Bharat: Why do we have torch, chumpy and numpy in the same function. It seems all the ops can be done in torch.

    body25_reg_torch, face_reg_torch, hand_reg_torch = torch_pose_obj_data()
    ipdb.set_trace()
    verts, _, _, _ = smpl.forward()
    J = batch_sparse_dense_matmul(body25_reg_torch, verts)
    face = batch_sparse_dense_matmul(face_reg_torch, verts)
    hands = batch_sparse_dense_matmul(hand_reg_torch, verts)

    J_observed = pose_3d['pose_keypoints_3d']
    face_observed = pose_3d['face_keypoints_3d']
    hands_observed = np.vstack((pose_3d['hand_left_keypoints_3d'], pose_3d['hand_right_keypoints_3d']))

    hands_observed[:, 3][hands_observed[:, 3] < HAND_VISIBLE] = 0.

    #
    return ch.vstack((
        (J - J_observed[:, :3]) * J_observed[:, 3].reshape((-1, 1)),
        (face - face_observed[:, :3]) * face_observed[:, 3].reshape((-1, 1)),
        (hands - hands_observed[:, :3]) * hands_observed[:, 3].reshape((-1, 1)),
        ))
