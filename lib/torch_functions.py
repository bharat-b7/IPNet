"""
Author: Bharat
"""
import torch


def batch_gather(arr, ind):
    """
    :param arr: B x N x D
    :param ind: B x M
    :return: B x M x D
    """
    dummy = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), arr.size(2))
    out = torch.gather(arr, 1, dummy)
    return out

def batch_sparse_dense_matmul(S, D):
    """
    Batch sparse-dense matrix multiplication

    :param torch.SparseTensor S: a sparse tensor of size (batch_size, p, q)
    :param torch.Tensor D: a dense tensor of size (batch_size, q, r)
    :return: a dense tensor of size (batch_size, p, r)
    :rtype: torch.Tensor
    """

    num_b = D.shape[0]
    S_shape = S.shape
    if not S.is_coalesced():
        S = S.coalesce()

    indices = S.indices().view(3, num_b, -1)
    values = S.values().view(num_b, -1)
    ret = torch.stack([
        torch.sparse.mm(
            torch.sparse_coo_tensor(indices[1:, i], values[i], S_shape[1:], device=D.device),
            D[i]
        )
        for i in range(num_b)
    ])
    return ret

