"""
Code inspired from: https://github.com/jchibane/if-net/blob/master/data_processing/voxelized_pointcloud_sampling.py
"""

import utils.implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
import glob
import os
from os.path import join, split, exists
import argparse
from utils.voxelized_pointcloud_sampling import voxelize


def get_3DSV(mesh):
    from opendr.camera import ProjectPoints
    from opendr.renderer import DepthRenderer
    WIDTH, HEIGHT = 250, 250

    camera = ProjectPoints(v=mesh.vertices, f=np.array([WIDTH, WIDTH]), c=np.array([WIDTH, HEIGHT]) / 2.,
                           t=np.array([0, 0, 2.5]), rt=np.array([np.pi, 0, 0]), k=np.zeros(5))
    frustum = {'near': 1., 'far': 10., 'width': WIDTH, 'height': HEIGHT}
    rn = DepthRenderer(camera=camera, frustum=frustum, f=mesh.faces, overdraw=False)

    points3d = camera.unproject_depth_image(rn.r)
    points3d = points3d[points3d[:, :, 2] > np.min(points3d[:, :, 2]) + 0.01]

    # print('sampled {} points.'.format(points3d.shape[0]))
    return points3d


def voxelized_pointcloud_sampling(mesh_path, name, out_path, res, bounds=(-0.837726, 1.10218), ext=''):
    if not exists(mesh_path):
        print('Mesh not found, ', mesh_path)
        return

    out_file = join(out_path, name + '_voxelized_point_cloud_res_{}_points_{}_{}.npz'.format(res, -1, ext))
    if exists(out_file) and REDO == False:
        print('Already exists, ', split(out_file)[1])
        return

    mesh = trimesh.load(mesh_path)
    point_cloud = get_3DSV(mesh)

    _ = voxelize(point_cloud, res, bounds, save_path=out_path)

    print('Done, ', split(out_file)[1])


REDO = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxelized sampling'
    )
    parser.add_argument(
        'mesh_path',
        type=str)
    parser.add_argument(
        'out_path',
        type=str)
    parser.add_argument(
        '--ext',
        type=str, default='')
    parser.add_argument(
        '--REDO', dest='REDO', action='store_true')
    parser.add_argument('--res', type=int, default=32)
    args = parser.parse_args()

    bb_min = -1.
    bb_max = 1.

    REDO = args.REDO
    name = split(args.mesh_path)[1][:-4]
    voxelized_pointcloud_sampling(args.mesh_path, name, args.out_path, args.res, bounds=(bb_min, bb_max), ext=args.ext)
