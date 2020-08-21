"""
Code to train IPNet.
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""

import models.local_model_body_full as model
from data_loader.data_loader import DataLoaderFullBodyParts, DataLoaderFullBodyPartsSV, DataLoader
from models.trainer import TrainerIPNet, TrainerIPNetMano, Trainer
import argparse
import torch

import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser(
    description='Run Model'
)
# number of points in input in case of pointcloud input
parser.add_argument('-pc_samples', default=3000, type=int)
# number of points to predict as output
parser.add_argument('-num_sample_points', default=25000, type=int)
# distribution of samples used constructed via different standard devations
parser.add_argument('-dist', '--sample_distribution', default=[1], nargs='+', type=float)
# the standard deviations from the surface used to compute inside/outside samples
parser.add_argument('-std_dev', '--sample_sigmas', default=[0.005], nargs='+', type=float)
# defines how much input data is unsed as a batch.
parser.add_argument('-batch_size', default=30, type=int)
# the resolution of the input
parser.add_argument('-res', default=32, type=int)
# keep this fixed
parser.add_argument('-h_dim', '--decoder_hidden_dim', default=256, type=int)
# which model to use, e.g. "-m IPNet"
parser.add_argument('-m', '--model', default='IPNetSingleSurface', type=str)
# keep this fixed
parser.add_argument('-o', '--optimizer', default='Adam', type=str)
# data suffix
parser.add_argument('-suffix', '--suffix', default='', type=str)
# ext for data suffix
parser.add_argument('-ext', '--ext', default='', type=str)
# experiment id for folder suffix
parser.add_argument('-exp_id', '--exp_id', default='', type=str)
# Select singleView mode
parser.add_argument('-SV', dest='SV', action='store_true', default=False)
# Epochs
parser.add_argument('-epochs', default=150, type=int)

args = parser.parse_args()


if args.model == 'IPNet':
    net = model.IPNet(hidden_dim=args.decoder_hidden_dim, num_parts=14)
elif args.model == 'IPNetMano':
    net = model.IPNetMano(hidden_dim=args.decoder_hidden_dim, num_parts=7)
elif args.model == 'IPNetSingleSurface':
    net = model.IPNetSingleSurface(hidden_dim=args.decoder_hidden_dim, num_parts=14)
else:
    print('Wow watch where u goin\' with that model')
    exit()

if args.model == 'IPNetMano':
    args.split_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/mano_data_split_01.pkl'
elif args.model == 'IPNetSingleSurface':
    args.split_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/data_split_single_surface.pkl'
else:
    args.split_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/data_split_double_surface.pkl'

if args.SV:
    train_dataset = DataLoaderFullBodyPartsSV('train', pointcloud_samples=args.pc_samples, res=args.res,
                                              sample_distribution=args.sample_distribution,
                                              sample_sigmas=args.sample_sigmas,
                                              num_sample_points=args.num_sample_points,
                                              batch_size=args.batch_size, num_workers=30,
                                              suffix=args.suffix, ext=args.ext,
                                              split_file=args.split_file)

    val_dataset = DataLoaderFullBodyPartsSV('val', pointcloud_samples=args.pc_samples,
                                            res=args.res, sample_distribution=args.sample_distribution,
                                            sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
                                            batch_size=args.batch_size, num_workers=30,
                                            suffix=args.suffix, ext=args.ext,
                                            split_file=args.split_file)
elif args.model == 'IPNet':
    train_dataset = DataLoaderFullBodyParts('train', pointcloud_samples=args.pc_samples, res=args.res,
                                            sample_distribution=args.sample_distribution,
                                            sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
                                            batch_size=args.batch_size, num_workers=30,
                                            suffix=args.suffix, ext=args.ext,
                                            split_file=args.split_file)

    val_dataset = DataLoaderFullBodyParts('val', pointcloud_samples=args.pc_samples,
                                          res=args.res, sample_distribution=args.sample_distribution,
                                          sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
                                          batch_size=args.batch_size, num_workers=30,
                                          suffix=args.suffix, ext=args.ext, split_file=args.split_file)

elif args.model == 'IPNetSingleSurface':
    train_dataset = DataLoader('train', pointcloud_samples=args.pc_samples, res=args.res,
                               sample_distribution=args.sample_distribution,
                               sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
                               batch_size=args.batch_size, num_workers=30,
                               suffix=args.suffix, ext=args.ext,
                               split_file=args.split_file)

    val_dataset = DataLoader('val', pointcloud_samples=args.pc_samples,
                             res=args.res, sample_distribution=args.sample_distribution,
                             sample_sigmas=args.sample_sigmas, num_sample_points=args.num_sample_points,
                             batch_size=args.batch_size, num_workers=30,
                             suffix=args.suffix, ext=args.ext, split_file=args.split_file)

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

# Load pre-trained model. This model was trained for single layered predictions.
# We use this pre-training because inside surface is not available for all the scans.
# Skip if this model is not available.
if args.model != 'IPNetMano' and args.model != 'IPNetSingleSurface':
    pre_path = 'IPNetSingleSurface_p5000_01s_exp_id01'
    pre_trained = model.IPNetSingleSurface(hidden_dim=args.decoder_hidden_dim, num_parts=14)
    pre_trainer = TrainerIPNet(pre_trained, torch.device("cuda"), None, None, pre_path,
                               optimizer=args.optimizer)
    pre_trainer.load_checkpoint()
    print('Loaded pretrained model from: ', pre_path)

    # Copy weights for initial layers
    import copy

    for i, (src, tgt) in enumerate(zip(pre_trained.children(), net.children())):
        if i > 9:
            break
        tgt.weight.data = copy.deepcopy(src.weight.data)
        tgt.bias.data = copy.deepcopy(src.bias.data)

if args.model == 'IPNetMano':
    exp_name += '_mano'
    trainer = TrainerIPNetMano(net, torch.device("cuda"), train_dataset, val_dataset, exp_name,
                               optimizer=args.optimizer)
elif args.model == 'IPNet':
    trainer = TrainerIPNet(net, torch.device("cuda"), train_dataset, val_dataset, exp_name,
                           optimizer=args.optimizer)
else:  # single surface model no parts
    trainer = Trainer(net, torch.device("cuda"), train_dataset, val_dataset, exp_name,
                      optimizer=args.optimizer)

trainer.train_model(args.epochs)

"""
python train.py -dist 0.5 0.5 -std_dev 0.15 0.015 -batch_size 4 -res 128 -m IPNetSingleSurface -ext 01s -suffix 01 -pc_samples 5000 -num_sample_points 20000 -exp_id 01

python train.py -dist 0.5 0.5 -std_dev 0.15 0.015 -batch_size 4 -res 128 -m IPNet -ext 01 -suffix 01 -pc_samples 5000 -num_sample_points 20000 -exp_id 01
"""
