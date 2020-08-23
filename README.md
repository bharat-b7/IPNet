# Implicit Part Network (IP-Net)
Repo for **"Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV'20 (Oral)"**

Link to paper: http://arxiv.org/abs/2007.11432

## Prerequisites
1. Cuda 10.0
2. Cudnn 7.6.5
3. Kaolin (https://github.com/NVIDIAGameWorks/kaolin) - for SMPL registration
4. MPI mesh library (https://github.com/MPI-IS/mesh)
5. Trimesh
6. Python 3.7.6
7. Tensorboard 1.15
8. Pytorch 1.4

## Download pre-trained models
Coming soon

## Preprocess data for training
1. Normalize scans: `python utils/preprocess scan.py <scan.obj> <body_shape.obj> <save_name> <save_path>`
2. Register SMPL+D to the  scan: `smpl_registration/fit_SMPLD.py <scan_path.obj> <save_path>`\
Note that SMPL fitting is much more stable with correct gender.
3. Generate query points: `python boundary_sampling_double.py <scan_scaled.obj> <body_shape_scaled.py> <smpld_registration.obj> <save_path> --sigma <sigma> --sample_num 100000 --ext_in 01 --ext_out 01`\
We used sigma=0.15 and 0.015, ext_in and ext_out are just suffix for naming files.
4. Generate voxelized input : `python voxelized_pointcloud_sampling.py <scan_scaled.obj> <save_path> --ext 01 --res 128 --num_points 5000`

## Run IP-Net
1. Test on single scan/PC: `python test_IPNet.py <pc_path> <network_weights_path> <out_path> -m IPNet`
2. Test on a dataset: `python generate.py -dist 0.5 0.5 -std_dev 0.15 0.015 -res 128 -m IPNet -ext 01 -suffix 01 -pc_samples 5000 -exp_id 01`

For training/ testing on dataset, you'd need the following directory structure if you'd like to use our dataloaders:

[DATASETS]\
-[dataset]\
--[subject_01]\
---[scan.obj]\
---[smpld_registration.obj]\
---[boundary_sampling]\
---- <query points for implicit function, see boundary_samplin_double.py, we use sigma=[0.15, 0.015]>\
---[voxels]\
---- <voxelized scan, see voxelized_pointcloud_sampling.py>\
--[subject_02]

## Train IP-Net
`python train.py -dist 0.5 0.5 -std_dev 0.15 0.015 -batch_size 4 -res 128 -m IPNet -ext 01 -suffix 01 -pc_samples 5000 -num_sample_points 20000 -exp_id 01`

## Fit SMPL to IP-net predictions
`python smpl_registration/fit_SMPL_IPNet.py <scan_path.obj> <scan_labels.npy> <scale_file.npy> <save_path>`

Cite us:
```
@inproceedings{bhatnagar2020ipnet,
    title = {Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction},
    author = {Bhatnagar, Bharat Lal and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision ({ECCV})},
    month = {aug},
    organization = {{Springer}},
    year = {2020},
}
```

## License

Copyright (c) 2020 Bharat Lal Bhatnagar, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction** paper in documents and papers that report on research using this Software.