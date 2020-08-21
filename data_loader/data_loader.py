'''
Modified version of Julian's voxelized_data.py
'''
import cv2
import codecs
import numpy as np

try:
    import cPickle as pkl
except:
    import _pickle as pkl
import sys
from torch.utils.data import Dataset
import os
from os.path import join, split, exists
# from psbody.mesh import Mesh, MeshViewer
import torch


class DataLoader(object):
    def __init__(self, mode, res=32, pointcloud_samples=3000, data_path='/BS/bharat-2/static00/implicit',
                 split_file='/BS/bharat/work/DoubleImplicit/test_data/data_split_01.pkl', suffix='',
                 batch_size=64, num_sample_points=1024, num_workers=12, sample_distribution=[1], sample_sigmas=[0.005],
                 ext=''):
        # sample distribution should contain the percentage of uniform samples at index [0]
        # and the percentage of N(0,sample_sigma[i-1]) samples at index [i] (>0).
        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.mode = mode
        self.path = data_path
        with open(split_file, "rb") as f:
            self.split = pkl.load(f)

        self.data = self.split[mode]
        self.res = res
        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pointcloud_samples = pointcloud_samples
        self.ext = ext
        self.suffix = suffix

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * self.num_sample_points).astype(np.uint32)

    def __len__(self):
        return len(self.data)

    def load_sampling_points(self, file):
        points, occupancies, coords, parts = [], [], [], []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = file + '_{}_{}.npz'.format(self.sample_sigmas[i], self.ext)
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']

            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

            if 'parts' in boundary_samples_npz.keys():
                boundary_sample_parts = boundary_samples_npz['parts']
                parts.extend(boundary_sample_parts[subsample_indices])
            else:
                boundary_sample_parts = None


        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points
        assert (len(parts) == self.num_sample_points) or (boundary_sample_parts is None)

        return points, coords, occupancies, parts

    def __getitem__(self, idx):
        path = self.data[idx]
        name = split(path)[1]

        voxel_path_full = join(path, 'voxelized', name + '_scaled_voxelized_point_cloud_res_{}_points_{}_{}.npz'.format(
            self.res, self.pointcloud_samples, self.suffix))

        input_full = self.load_voxel_input(voxel_path_full)

        if self.mode == 'train' or self.mode == 'val':
            boundary_samples_path = join(path, 'boundary_sampling',
                                         name + '_scaled_boundary_all_sigma')
            points, coords, occupancies, parts = self.load_sampling_points(boundary_samples_path)
        else:  # for testing we might not have annotations
            points, coords, occupancies, parts = None, None, None, None

        return {'grid_coords': np.array(coords, dtype=np.float32),
                'occupancies': np.array(occupancies, dtype=np.float32),
                'points': np.array(points, dtype=np.float32),
                'inputs': np.array(input_full, dtype=np.float32),
                'path': path
                }

    def load_voxel_input(self, file):
        occupancies = np.unpackbits(np.load(file)['compressed_occupancies'])
        input = np.reshape(occupancies, (self.res,) * 3)
        return input

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        ''' Worker init function to ensure true randomness.
        '''
        # base_seed = int(os.urandom(4).encode('hex'), 16)
        base_seed = int(codecs.encode(os.urandom(4), 'hex'), 16)
        np.random.seed(base_seed + worker_id)


class DataLoaderFullBodyParts(DataLoader):
    def __init__(self, mode, res=32, pointcloud_samples=3000,
                 data_path='/BS/bharat-2/static00/implicit',
                 split_file='/BS/bharat/work/DoubleImplicit/test_data/data_split_01.pkl', suffix='',
                 batch_size=64, num_sample_points=1024, num_workers=12, sample_distribution=[1], sample_sigmas=[0.005],
                 ext=''):
        # sample distribution should contain the percentage of uniform samples at index [0]
        # and the percentage of N(0,sample_sigma[i-1]) samples at index [i] (>0).
        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.mode = mode
        self.path = data_path
        with open(split_file, "rb") as f:
            self.split = pkl.load(f)

        self.data = self.split[mode]
        self.res = res
        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pointcloud_samples = pointcloud_samples
        self.ext = ext
        self.suffix = suffix

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * self.num_sample_points).astype(np.uint32)

    def __getitem__(self, idx):
        path = self.data[idx]
        name = split(path)[1]

        voxel_path_full = join(path, 'voxelized', name + '_scaled_voxelized_point_cloud_res_{}_points_{}_{}.npz'.format(
            self.res, self.pointcloud_samples, self.suffix))

        input_full = self.load_voxel_input(voxel_path_full)

        if self.mode == 'train' or self.mode == 'val':
            boundary_samples_path = join(path, 'boundary_sampling',
                                         name + '_scaled_boundary_all_parts_sigma')
            points, coords, occupancies, parts = self.load_sampling_points(boundary_samples_path)
        else:  # for test me might not have annotations
            points, coords, occupancies, parts = None, None, None, None

        return {'grid_coords': np.array(coords, dtype=np.float32),
                'occupancies': np.array(occupancies, dtype=np.float32),
                'points': np.array(points, dtype=np.float32),
                'inputs': np.array(input_full, dtype=np.float32),
                'path': path,
                'parts': np.array(parts, dtype=np.float32)
                }


class DataLoaderFullBodyPartsSV(DataLoaderFullBodyParts):
    def __getitem__(self, idx):
        path = self.data[idx]
        name = split(path)[1]

        voxel_path_full = join(path, 'voxelized_SV', name + \
                               '_scaled_voxelized_point_cloud_res_{}_points_{}_{}.npz'.format(self.res,
                                                                                              self.pointcloud_samples,
                                                                                              self.suffix))
        input_full = self.load_voxel_input(voxel_path_full)

        if self.mode == 'train':
            boundary_samples_path = join(path, 'boundary_sampling',
                                         name + '_scaled_boundary_all_parts_sigma')
            points, coords, occupancies, parts = self.load_sampling_points(boundary_samples_path)
        else:
            boundary_samples_path, points, coords, occupancies, parts = None, None, None, None, None

        return {'grid_coords': np.array(coords, dtype=np.float32),
                'occupancies': np.array(occupancies, dtype=np.float32),
                'points': np.array(points, dtype=np.float32),
                'inputs': np.array(input_full, dtype=np.float32),
                'path': path,
                'parts': np.array(parts, dtype=np.float32)
                }


if __name__ == "__main__":
    args = lambda: None
    args.pc_samples = 5000
    args.res = 128
    args.sample_distribution = [0.5, 0.5]
    args.sample_sigmas = [0.15, 0.015]
    args.num_sample_points = 40000
    args.batch_size = 4
    args.suffix = '01'

    # args.ext = '01s'
    # args.split_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/data_split_single_surface.pkl'
    # train_dataset = DataLoader('train', pointcloud_samples=args.pc_samples, res=args.res,
    #                            sample_distribution=args.sample_distribution,
    #                            sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
    #                            batch_size=args.batch_size, num_workers=0,
    #                            suffix=args.suffix, ext=args.ext,
    #                            split_file=args.split_file).get_loader()

    args.ext = '01'
    args.split_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/data_split_double_surface.pkl'
    train_dataset = DataLoaderFullBodyParts('train', pointcloud_samples=args.pc_samples, res=args.res,
                                            sample_distribution=args.sample_distribution,
                                            sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
                                            batch_size=args.batch_size, num_workers=0,
                                            suffix=args.suffix, ext=args.ext,
                                            split_file=args.split_file).get_loader()

    for b in train_dataset:
        break

    # import ipdb; ipdb.set_trace()
    print('Done')
