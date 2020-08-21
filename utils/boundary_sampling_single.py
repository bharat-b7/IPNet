"""
Code to generate input data for the implicit network.
This code is inspired from: https://github.com/jchibane/if-net/blob/master/data_processing/boundary_sampling.py
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""

import numpy as np
import glob
import os
from os.path import join, split, exists
import argparse
import pickle as pkl
from scipy.spatial import cKDTree as KDTree
import trimesh
from utils import implicit_waterproofing as iw
from utils.boundary_sampling_double import part_labelling
from psbody.mesh import Mesh


def boundary_sampling(full_path, name, out_path, sigma=0.05, sample_num=100000, bounds=(-1., 1.), ext=''):
    out_file = join(out_path, name + '_boundary_all_sigma_{}_{}.npz'.format(sigma, ext))

    if exists(out_file) and REDO == False:
        print('File already exists, ', out_file)
        return out_file

    if not exists(full_path):
        print('Mesh not found, ', full_path)
        return False

    full = trimesh.load(full_path)
    points_full = full.sample(sample_num)

    boundary_points = points_full + sigma * np.random.randn(sample_num, 3)

    # coordinates transformation for torch.nn.functional.grid_sample grid interpolation method
    # for indexing of grid_sample function: swaps x and z coordinates
    grid_coords = boundary_points.copy()
    grid_coords[:, 0], grid_coords[:, 2] = grid_coords[:, 2], grid_coords[:, 0].copy()

    ce, sc = bounds[0] + bounds[1], bounds[1] - bounds[0]
    grid_coords = 2 * grid_coords - ce
    grid_coords = grid_coords / sc

    ## Also add uniform points
    n_samps = 30
    uniform_points = np.array(iw.create_grid_points_from_bounds(bounds[0], bounds[1], n_samps))
    uniform_points_scaled = (2. * uniform_points.copy() - ce) / sc
    uniform_points_scaled[:, 0], uniform_points_scaled[:, 2] = uniform_points_scaled[:, 2], \
                                                               uniform_points_scaled[:, 0].copy()

    grid_coords = np.append(grid_coords, uniform_points_scaled, axis=0)
    boundary_points = np.append(boundary_points, uniform_points, axis=0)

    occupancies_full = iw.implicit_waterproofing(full, boundary_points)[0]

    labels = np.ones((len(occupancies_full, )))
    labels[~occupancies_full] = 0

    np.savez(out_file, points=boundary_points, occupancies=labels, grid_coords=grid_coords)
    print('Done sampling points, ', out_file)

    return out_file


REDO = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('full_path', type=str)
    parser.add_argument('reg_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--ext_in', type=str, default='')
    parser.add_argument('--ext_out', type=str, default='')
    parser.add_argument(
        '--REDO', dest='REDO', action='store_true')
    parser.add_argument('--sigma', default=0.05, type=float)
    parser.add_argument('--sample_num', default=100000, type=np.int32)
    parser.set_defaults(parts=False)
    parser.add_argument('--parts', dest='parts', action='store_true')

    args = parser.parse_args()

    REDO = args.REDO
    min = -1.
    max = 1.

    name = split(args.full_path)[1][:-4]

    if not exists(args.out_path):
        os.makedirs(args.out_path)

    out_file = boundary_sampling(args.full_path, name, args.out_path, sigma=args.sigma, sample_num=args.sample_num,
                                 bounds=(min, max), ext=args.ext_in)
    part_labelling(out_file, args.reg_path, name, args.out_path, sigma=args.sigma, ext=args.ext_out)
