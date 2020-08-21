"""
Code to produces results from a trained IPNet.
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""

import models.local_model_body_full as model
from data_loader.data_loader import DataLoaderFullBodyParts, DataLoaderFullBodyPartsSV
import numpy as np
import argparse
from models.generator import GeneratorIPNet, GeneratorIPNetMano, Generator
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

parser = argparse.ArgumentParser(
    description='Run Model'
)
# number of points in input in case of pointcloud input
parser.add_argument('-pc_samples', default=3000, type=int)
# distribution of samples used constructed via different standard devations
parser.add_argument('-dist', '--sample_distribution', default=[1], nargs='+', type=float)
# the standard deviations from the surface used to compute inside/outside samples
parser.add_argument('-std_dev', '--sample_sigmas', default=[0.005], nargs='+', type=float)
# the resolution of the input
parser.add_argument('-res', default=128, type=int)
# keep this fixed
parser.add_argument('-h_dim', '--decoder_hidden_dim', default=256, type=int)
# what data should be used for generation? test,val,train?
parser.add_argument('-mode', default='val', type=str)
# number of points queried for to produce the result
parser.add_argument('-retrieval_res', default=256, type=int)
# which checkpoint of the experiment should be used?
parser.add_argument('-checkpoint', type=int)
# number of points from the querey grid which are put into the batch at once
parser.add_argument('-batch_points', default=500000, type=int)
# how many samples should be generated?
parser.add_argument('-samples_num', default=20, type=int)
# which model to use, e.g. "-m IPNet"
parser.add_argument('-m', '--model', default='IPNetSingleSurface', type=str)
# data suffix
parser.add_argument('-suffix', '--suffix', default='', type=str)
# ext for data suffix
parser.add_argument('-ext', '--ext', default='', type=str)
## experiment id for folder suffix
parser.add_argument('-exp_id', '--exp_id', default='', type=str)
# Select singleView mode
parser.add_argument('-SV', dest='SV', action='store_true', default=False)
args = parser.parse_args()

if args.SV:
    exp_name = '{}{}_{}_exp_id{}'.format(
        args.model + '_SV',
        '_p{}'.format(args.pc_samples),
        args.ext,
        args.exp_id
    )
else:
    exp_name = '{}{}_{}_exp_id{}'.format(
        args.model,
        '_p{}'.format(args.pc_samples),
        args.ext,
        args.exp_id
    )

if args.model == 'IPNet':
    net = model.IPNet(hidden_dim=args.decoder_hidden_dim, num_parts=14)
    gen = GeneratorIPNet(net, 0.5, exp_name, checkpoint=args.checkpoint, resolution=args.retrieval_res,
                         batch_points=args.batch_points)
    args.split_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/data_split_double_surface.pkl'
elif args.model == 'IPNetMano':
    net = model.IPNetMano(hidden_dim=args.decoder_hidden_dim, num_parts=7)
    gen = GeneratorIPNetMano(net, 0.5, exp_name, checkpoint=args.checkpoint, resolution=args.retrieval_res,
                             batch_points=args.batch_points)
    args.split_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/mano_data_split_01.pkl'
elif args.model == 'IPNetSingleSurface':
    net = model.IPNetSingleSurface(hidden_dim=args.decoder_hidden_dim, num_parts=14)
    gen = Generator(net, 0.5, exp_name, checkpoint=args.checkpoint, resolution=args.retrieval_res,
                    batch_points=args.batch_points)
    args.split_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/data_split_single_surface.pkl'
else:
    print('Wow watch where u goin\' with that model')
    exit()

if args.SV:
    dataset = DataLoaderFullBodyPartsSV(args.mode, pointcloud_samples=args.pc_samples,
                                        res=args.res, sample_distribution=args.sample_distribution,
                                        sample_sigmas=args.sample_sigmas,
                                        batch_size=1,
                                        num_workers=30, suffix=args.suffix, ext=args.ext,
                                        split_file=args.split_file)
else:
    dataset = DataLoaderFullBodyParts(args.mode, pointcloud_samples=args.pc_samples,
                                      res=args.res, sample_distribution=args.sample_distribution,
                                      sample_sigmas=args.sample_sigmas,
                                      batch_size=1,
                                      num_workers=30, suffix=args.suffix, ext=args.ext,
                                      split_file=args.split_file)

loader = dataset.get_loader(shuffle=False)

if args.checkpoint is None:
    args.checkpoint = gen.get_last_checkpoint()

out_path = 'experiments/{}/visualization_@{}_{}_res{}'.format(exp_name, args.checkpoint, args.mode, args.retrieval_res)

if not os.path.exists(out_path):
    os.makedirs(out_path)
print(out_path)

i = 0
for data in loader:
    name = os.path.basename(data['path'][0])
    if args.model == 'IPNet':
        full, body, parts = gen.generate_meshs_all_parts(data)

        # import ipdb; ipdb.set_trace()
        body.set_vertex_colors_from_weights(parts)
        body.write_ply(out_path + '/{}_body.ply'.format(name))
        np.save(out_path + '/{}_parts.npy'.format(name), parts)

    elif args.model == 'IPNetMano':
        full, parts = gen.generate_meshs_all_parts(data)
        np.save(out_path + '/{}_parts.npy'.format(name), parts)

    elif args.model == 'IPNetSingleSurface':
        full = gen.generate_mesh_all(data)

    full.write_ply(out_path + '/{}_full.ply'.format(name))
    print('Generated ', name)
    i += 1
    if i == args.samples_num:
        break

"""
python generate.py -dist 0.5 0.5 -std_dev 0.15 0.015 -res 128 -m IPNetSingleSurface -ext 01s -suffix 01 -pc_samples 5000 -exp_id 01 -samples_num 20

python generate.py -dist 0.5 0.5 -std_dev 0.15 0.015 -res 128 -m IPNet -ext 01 -suffix 01 -pc_samples 5000 -exp_id 01 -samples_num 20
"""