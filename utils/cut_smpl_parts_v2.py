"""
Code to cut SMPL into near symmetric parts.
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""

import numpy as np
from psbody.mesh import Mesh
from lib.smpl_paths import SmplPaths
import pickle as pkl


def get_tpose_smpl():
    sp = SmplPaths(gender='neutral')
    smpl = sp.get_smpl()
    smpl.trans[:] = 0
    smpl.pose[:] = 0

    return smpl


def cut_right_forearm(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[verts[:, 0] < -0.6] = 1  # right hand

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('right_forearm ', np.where(col)[0].shape)
    return col


def cut_left_forearm(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[verts[:, 0] > 0.6] = 1  # left hand

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('left_forearm ', np.where(col)[0].shape)
    return col


def cut_right_midarm(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 0] >= -0.6) & (verts[:, 0] < -0.4)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('right_midarm ', np.where(col)[0].shape)
    return col


def cut_right_upperarm(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 0] >= -0.4) & (verts[:, 0] < -0.2)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('right_upperarm ', np.where(col)[0].shape)
    return col


def cut_left_midarm(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 0] <= 0.6) & (verts[:, 0] > 0.4)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('left_midarm ', np.where(col)[0].shape)
    return col


def cut_left_upperarm(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 0] <= 0.4) & (verts[:, 0] > 0.2)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('left_upperarm ', np.where(col)[0].shape)
    return col


def cut_head(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[verts[:, 1] > 0.3] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('head ', np.where(col)[0].shape)
    return col


def cut_upper_right_leg(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 1] < -0.3) & (verts[:, 0] < 0) & (verts[:, 1] >= -0.7)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('upper_right_leg ', np.where(col)[0].shape)
    return col


def cut_right_leg(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 1] < -0.7) & (verts[:, 0] < 0) & (verts[:, 1] > -1)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('right_leg ', np.where(col)[0].shape)
    return col


def cut_right_foot(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 1] < -1) & (verts[:, 0] < 0)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('left_foot ', np.where(col)[0].shape)
    return col


def cut_upper_left_leg(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 1] < -0.3) & (verts[:, 0] >= 0) & (verts[:, 1] >= -0.7)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('upper_left_leg ', np.where(col)[0].shape)
    return col


def cut_left_leg(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 1] < -0.7) & (verts[:, 0] >= 0) & (verts[:, 1] > -1)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('left_leg ', np.where(col)[0].shape)
    return col


def cut_left_foot(display=False):
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)
    col = np.zeros(verts.shape[0])
    col[(verts[:, 1] < -1) & (verts[:, 0] >= 0)] = 1

    if display:
        ms.set_vertex_colors_from_weights(col)
        ms.show()

    print('left_foot ', np.where(col)[0].shape)
    return col


if __name__ == "__main__":
    smpl = get_tpose_smpl()
    verts = smpl.r
    faces = smpl.f

    ms = Mesh(v=verts, f=faces)

    col = np.zeros((6890,))
    display = False
    rfa = cut_right_forearm(display)
    col += (rfa * 0.1)

    rma = cut_right_midarm(display)
    col += (rma * 0.2)

    lfa = cut_left_forearm(display)
    col += (lfa * 0.3)

    lma = cut_left_midarm(display)
    col += (lma * 0.4)

    rua = cut_right_upperarm(display)
    col += (rua * 0.5)

    lua = cut_left_upperarm(display)
    col += (lua * 0.6)

    h = cut_head(display)
    col += (h * 0.7)

    url = cut_upper_right_leg(display)
    col += (url * 0.8)

    rl = cut_right_leg(display)
    col += (rl * 0.9)

    ull = cut_upper_left_leg(display)
    col += (ull * 1)

    ll = cut_left_leg(display)
    col += (ll * 1.1)

    lf = cut_left_foot(display)
    col += (lf * 1.2)

    rf = cut_right_foot(display)
    col += (rf * 1.3)

    print('torso ', len(ms.v) - np.where(col)[0].shape[0])

    parts = {'right_forearm': np.where(rfa)[0], 'left_forearm': np.where(lfa)[0],
             'right_upperarm': np.where(rua)[0], 'left_upperarm': np.where(lua)[0],
             'head': np.where(h)[0], 'right_leg': np.where(rl)[0], 'left_leg': np.where(ll)[0],
             'torso': np.where(col == 0)[0],
             'right_midarm': np.where(rma)[0], 'left_midarm': np.where(lma)[0],
             'upper_left_leg': np.where(ull)[0], 'upper_right_leg': np.where(url)[0],
             'right_foot': np.where(rf)[0], 'left_foot': np.where(lf)[0]}

    import collections

    parts = collections.OrderedDict(sorted(parts.items()))

    col = np.zeros((6890,))
    for n, k in enumerate(parts):
        col[parts[k]] = n
    ms.set_vertex_colors_from_weights(col)
    ms.show()

    import ipdb; ipdb.set_trace()

    pkl.dump(parts, open('/BS/bharat-3/work/IPNet/assets/smpl_parts_dense.pkl', 'wb'))
    print('Done')
