"""
Code to test IPNet with a pointcloud as input. To run IPNet on entire dataset see generate.py
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""

import torch
import models.local_model_body_full as model
import numpy as np
import argparse
from utils.preprocess_scan import func
from utils.voxelized_pointcloud_sampling import voxelize
from models.generator import GeneratorIPNet, GeneratorIPNetMano, Generator
import trimesh
import os


def pc2vox(pc, res):
    """Convert PC to voxels for IPNet"""
    # preprocess the pointcloud
    pc = func(pc)
    vox = voxelize(pc, res)
    return vox


def main(args):
    # Load PC
    pc = trimesh.load(args.pc)
    pc_vox = pc2vox(pc, args.res)
    pc_vox = np.reshape(pc_vox, (args.res,) * 3).astype('float32')

    # Load network
    if args.model == 'IPNet':
        net = model.IPNet(hidden_dim=args.decoder_hidden_dim, num_parts=14)
        gen = GeneratorIPNet(net, 0.5, exp_name=None, checkpoint=args.checkpoint, resolution=args.retrieval_res,
                             batch_points=args.batch_points)
    elif args.model == 'IPNetMano':
        net = model.IPNetMano(hidden_dim=args.decoder_hidden_dim, num_parts=7)
        gen = GeneratorIPNetMano(net, 0.5, exp_name=None, checkpoint=args.checkpoint, resolution=args.retrieval_res,
                                 batch_points=args.batch_points)
    elif args.model == 'IPNetSingleSurface':
        net = model.IPNetSingleSurface(hidden_dim=args.decoder_hidden_dim, num_parts=14)
        gen = Generator(net, 0.5, exp_name=None, checkpoint=args.checkpoint, resolution=args.retrieval_res,
                        batch_points=args.batch_points)
    else:
        print('Wow watch where u goin\' with that model')
        exit()

    # Load weights
    checkpoint_ = torch.load(args.weights)
    net.load_state_dict(checkpoint_['model_state_dict'])

    # Run IPNet and Save
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    data = {'input': pc_vox[np.newaxis]}    # add a batch dimension
    if args.model == 'IPNet':
        full, body, parts = gen.generate_meshs_all_parts(data)
        body.set_vertex_colors_from_weights(parts)
        body.write_ply(args.out_path + '/body.ply')
        np.save(args.out_path + '/parts.npy', parts)

    elif args.model == 'IPNetMano':
        full, parts = gen.generate_meshs_all_parts(data)
        np.save(args.out_path + '/parts.npy', parts)

    elif args.model == 'IPNetSingleSurface':
        full = gen.generate_mesh_all(data)

    full.write_ply(args.out_path + '/full.ply')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run Model'
    )
    # Path to PC mesh
    parser.add_argument('pc', type=str)
    # path to pretrained weights
    parser.add_argument('weights', type=str)
    # path to save result
    parser.add_argument('out_path', type=str)
    # the resolution of the input
    parser.add_argument('-res', default=128, type=int)
    # keep this fixed
    parser.add_argument('-h_dim', '--decoder_hidden_dim', default=256, type=int)
    # number of points queried for to produce the result
    parser.add_argument('-retrieval_res', default=256, type=int)
    # number of points from the querey grid which are put into the batch at once
    parser.add_argument('-batch_points', default=500000, type=int)
    # which model to use, e.g. "-m IPNet"
    parser.add_argument('-m', '--model', default='IPNetSingleSurface', type=str)
    args = parser.parse_args()

    main(args)


"""
python test_IPNet.py <pc_path> <network_weights_path> <out_path> -m IPNetSingleSurface
"""