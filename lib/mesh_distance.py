"""
Original Author: Garvita
Edited by: Bharat
"""

import torch
import kaolin as kal
from kaolin.rep import Mesh
import kaolin.cuda.tri_distance as td
import numpy as np
from kaolin.metrics.mesh import _compute_edge_dist, _compute_planar_dist, TriangleDistance, point_to_surface
from kaolin.rep import SDF as sdf
from lib.torch_functions import batch_gather

def point_to_surface_vec(points,mesh):
    """Computes the minimum distances from a set of points to a mesh
    Args:
            points (torch.Tensor): set of points
            mesh (Mesh): mesh to calculate distance

    Returns:
            distance: distance between points and surface (not averaged like Kaolin point_to_surface)
    """

    # extract triangle defs from mesh
    v1 = torch.index_select(mesh.vertices.clone(), 0, mesh.faces[:, 0])
    v2 = torch.index_select(mesh.vertices.clone(), 0, mesh.faces[:, 1])
    v3 = torch.index_select(mesh.vertices.clone(), 0, mesh.faces[:, 2])

    # if quad mesh the separate the triangles
    if mesh.faces.shape[-1] == 4:
        v4 = torch.index_select(mesh.vertices.clone(), 0, mesh.faces[:, 3])
        temp1 = v1.clone()
        temp2 = v2.clone()
        temp3 = v3.clone()
        v1 = torch.cat((v1, v1), dim=0)
        v2 = torch.cat((v2, v4), dim=0)
        v3 = torch.cat((v3, v3), dim=0)

    if points.is_cuda:

        tri_minimum_dist = TriangleDistance()
        # pass to cuda
        distance, indx, dist_type = tri_minimum_dist(points, v1, v2, v3)
        indx = indx.data.cpu().numpy()
        dist_type = torch.LongTensor(dist_type.data.cpu().numpy())
        # reconpute distances to define gradient
        grad_dist = _recompute_point_to_surface_vec(
            [v1, v2, v3], points, indx, dist_type)
        # sanity check
        # print(distance.mean(), grad_dist)
    else:
        raise NotImplementedError

    return grad_dist


def _recompute_point_to_surface_vec(verts, p, indecies, dist_type):
    # recompute surface based the calcualted correct assignments of points and triangles
    # and the type of distacne, type 1 to 3 idicates which edge to calcualte to,
    # type 4 indicates the distance is from a point on the triangle not an edge
    v1, v2, v3 = verts
    v1 = v1[indecies]
    v2 = v2[indecies]
    v3 = v3[indecies]

    type_1 = (dist_type == 0)
    type_2 = (dist_type == 1)
    type_3 = (dist_type == 2)
    type_4 = (dist_type == 3)

    v21 = v2 - v1
    v32 = v3 - v2
    v13 = v1 - v3

    p1 = p - v1
    p2 = p - v2
    p3 = p - v3

    dists = []
    dists.append(_compute_edge_dist(v21[type_1], p1[type_1]).view(-1))
    dists.append(_compute_edge_dist(v32[type_2], p2[type_2]).view(-1))
    dists.append(_compute_edge_dist(v13[type_3], p3[type_3]).view(-1))

    if len(np.where(type_4)[0]) > 0:
        nor = torch.cross(v21[type_4], v13[type_4])
        dists.append(_compute_planar_dist(nor, p1[type_4]))

    distances = torch.cat(dists)
    return distances


def chamfer_distance(s1, s2, w1=1., w2=1.):
    """
    :param s1: B x N x 3
    :param s2: B x M x 3
    :param w1: weight for distance from s1 to s2
    :param w2: weight for distance from s2 to s1
    """
    from kaolin.metrics.point import SidedDistance

    assert s1.is_cuda and s2.is_cuda
    sided_minimum_dist = SidedDistance()
    closest_index_in_s2 = sided_minimum_dist(s1, s2)
    closest_index_in_s1 = sided_minimum_dist(s2, s1)
    closest_s2 = batch_gather(s2, closest_index_in_s2)
    closest_s1 = batch_gather(s1, closest_index_in_s1)

    dist_to_s2 = (((s1 - closest_s2) ** 2).sum(dim=-1)).mean() * w1
    dist_to_s1 = (((s2 - closest_s1) ** 2).sum(dim=-1)).mean() * w2

    return dist_to_s2 + dist_to_s1

def batch_point_to_surface(points, meshes):
    """
    Naive implementation. Just loops over the set of points and meshes.
    This is a bit tricky to batch-ify because number of points and
        mesh structure could be different for each entry in the batch.
    """
    distance = [point_to_surface(p, m) for p, m in zip(points, meshes)]
    return torch.stack(distance)

def batch_point_to_surface_vec(points, meshes):
    distance = [point_to_surface_vec(p, m) for p, m in zip(points, meshes)]
    return torch.stack(distance)

def batch_point_to_surface_vec_signed(meshes, points):
    prelu = torch.nn.PReLU(init=25. *25.).cuda()
    dist = []
    for m, p in zip(meshes, points):
        dist_val = point_to_surface_vec(p, m)
        sign_val = torch.ones_like(dist_val)
        sign_bool = sdf.check_sign(m,p)[0] == 0
        sign_val[sign_bool] = -1.
        signed_dist = prelu(sign_val*dist_val)
        dist.append(torch.mean(signed_dist*signed_dist))

    return torch.stack(dist)