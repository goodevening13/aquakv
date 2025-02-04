from typing import Optional

import torch
import torch.nn as nn
from tqdm import trange


@torch.no_grad()
def fit_linear_regression(
        X: torch.Tensor,
        Y: torch.Tensor, *,
        reg_rate: float = 0,
        fit_intercept: bool = True,
        compute_device: Optional[torch.device] = None,
        compute_dtype: torch.dtype = None,
        chunk_size: Optional[int] = None,
        verbose: bool = False,
):
    """
    Multivariate linear regression with exactsolution
    :param X: input, shape: [nsamples, in_features]
    :param Y: target, shape: [nsamples, num_targets]
    :param reg_rate: regularize weights by this value, in proportion to mean square input
    :param fit_intercept: if True (default), learn the bias term as in the regular least-squares
        if False, force bias term to be zeros -- but still return the resulting all-zero vector
    :param compute_device: optionally transfer the covariance matrix to this device and compute inverse there
        Computing pinverse of large matrices on CPU can take hours
    :param compute_dtype: optionally cast tensors to this dtype for computations (memory-efficiently)
    :returns: (W, b) such that Y ~ (X @ W.T + b)
    """
    assert X.ndim == Y.ndim == 2 and X.shape[0] == Y.shape[0], "X, Y must be [nsamples, in/out_features]"
    (nsamples, in_features), (_, out_features), source_device = X.shape, Y.shape, Y.device
    orig_dtype = X.dtype

    if chunk_size is None:
        X = X.to(compute_dtype)
        if fit_intercept:
            X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        CXX = (X.T @ X).to(compute_device)  # [in_features + 1, in_features + 1], aka MSE hessian
        # add column of ones
    else:
        CXX = torch.zeros(in_features + 1, in_features + 1, device=compute_device, dtype=compute_dtype or X.dtype)
        for chunk_start in trange(0, nsamples, chunk_size, desc='fit_linear_regression::CXX',
                                  leave=False, disable=not verbose):
            xb = X[chunk_start: chunk_start + chunk_size].to(
                device=compute_device, dtype=compute_dtype, non_blocking=True)
            if fit_intercept:
                xb = torch.cat([xb, torch.ones(xb.shape[0], 1, device=xb.device)], dim=1)
            CXX = torch.addmm(CXX, xb.T, xb, out=CXX)
            del xb

    if reg_rate > 0:
        ix = torch.arange(len(CXX), device=compute_device or source_device)
        CXX[ix, ix] += reg_rate * abs(torch.diag(CXX)).mean()
        del ix

    CXX_pinv = torch.pinverse(CXX)
    del CXX

    if chunk_size is None:
        CXY = (X.T @ Y).to(compute_device)  # [in_features, out_features]
        del X, Y
    else:
        CXY = torch.zeros(in_features + 1, out_features, device=compute_device, dtype=compute_dtype or X.dtype)
        for chunk_start in trange(0, nsamples, chunk_size, desc='fit_linear_regression::CXY',
                                  leave=False, disable=not verbose):
            xb, yb = [tensor[chunk_start: chunk_start + chunk_size].to(
                device=compute_device, dtype=compute_dtype, non_blocking=True)
                for tensor in (X, Y)]
            if fit_intercept:
                xb = torch.cat([xb, torch.ones(xb.shape[0], 1, device=xb.device)], dim=1)
            CXY = torch.addmm(CXY, xb.T, yb, out=CXY)
            del xb, yb
        del X, Y

    W = (CXX_pinv @ CXY).T.to(source_device, dtype=orig_dtype)

    if fit_intercept:
        W, bias = W[:, :-1], W[:, -1]
    else:
        bias = None

    return W, bias


def reduced_rank_regression(
        X: torch.Tensor, Y: torch.Tensor, rank: int, *,
        reg_rate: float = 0, svd_niter: Optional[int] = 100, fit_intercept: bool = True,
        compute_device: Optional[torch.device] = None,
        compute_dtype: torch.dtype = None,
        chunk_size: Optional[int] = None,
):
    """
    Multivariate linear regression with a low-rank weight matrix, based on
    https://web.math.ku.dk/~sjo/papers/ReducedRankRegression.pdf

    :param X: input[nsamples, in_features]
    :param Y: target[nsamples, num_targets]
    :param rank: the required rank of the weight matrix
    :param reg_rate: regularize weights by this value, in proportion to mean square input
    :param svd_niter: if specified, estimate SVD with this many steps of Algorithm 5.1 from Halko et al, 2009
        If None, compute SVD exactly and take the first k components
    :note: you can also compute partial SVD with arpack from scipy.sparse.linalg.svds; this is not implemented
    :param fit_intercept: if True (default), learn the bias term as in the regular least-squares
        if False, force bias term to be zeros -- but still return the resulting all-zero vector
    :param compute_device: optionally transfer the covariance matrix to this device and compute inverse there
        Computing pinverse of large matrices on CPU can take hours
    :param compute_dtype: optionally cast tensors to this dtype for computations (memory-efficiently)
    :param chunk_size: process this many rows at a time when accumulating X^T @ X and X^T @ Y
    :returns: (W, V, b) such that Y ~ (X @ W @ V + b)


    :note: on using sample weights -- you can incorporate weights with pre/post-processing steps
      - when using per-dimension weights of shape [out_features], multiply only Y by dim_weight
        ... and then, divide both second matrix (VT) and intercept by dim_weight
      - when using per-sample weights of shape [nsamples], multiply both X and Y by
        ... sqrt(sample_weight)[:, None] / mean(sqrt(sample_weight)), and you should be fine
      - when using per-item weights of shape [nsamples, out_features], you're fucked and will probably need SGD,
        ... consider starting from a non_weighted solution (or use 1d weights), then fine-tune with SGD
    """
    assert X.ndim == Y.ndim == 2 and X.shape[0] == Y.shape[0], "X, Y must be [nsamples, in/out_features]"
    assert rank <= min(X.shape[1], Y.shape[1]), "rank must be less than num features / outputs"
    (nsamples, in_features), (_, out_features) = X.shape, Y.shape
    source_device = X.device

    if chunk_size is None:
        X = X.to(compute_dtype)
        CXX = (X.T @ X).to(compute_device)  # [in_features, in_features], aka MSE hessian
    else:
        CXX = torch.zeros(in_features, in_features, device=compute_device, dtype=compute_dtype or X.dtype)
        for chunk_start in range(0, nsamples, chunk_size):
            print(end=',')
            xb = X[chunk_start: chunk_start + chunk_size].to(
                device=compute_device, dtype=compute_dtype, non_blocking=True)
            CXX = torch.addmm(CXX, xb.T, xb, out=CXX)
            del xb

    if reg_rate > 0:
        ix = torch.arange(len(CXX), device=CXX.device)
        CXX[ix, ix] += reg_rate * abs(torch.diag(CXX)).mean()
        del ix

    # TODO this can be made a lot more efficient if you know the original weight
    CXX_pinv = torch.pinverse(CXX.to(compute_device))
    del CXX  # note: CXX can be computed on GPU by accumulating Xbatch.T@Xbatch products

    bias = Y.mean(0) if fit_intercept else None

    if chunk_size is None:
        Y_centered = Y.to(compute_dtype) - bias if fit_intercept else Y
        CXY = (X.T @ Y_centered).to(compute_device)  # [in_features, out_features]
        del X, Y, Y_centered
    else:
        device_bias = bias.to(compute_device) if fit_intercept else None
        CXY = torch.zeros(in_features, out_features, device=compute_device, dtype=compute_dtype or X.dtype)
        for chunk_start in range(0, nsamples, chunk_size):
            xb, yb = [tensor[chunk_start: chunk_start + chunk_size].to(
                device=compute_device, dtype=compute_dtype, non_blocking=True)
                for tensor in (X, Y)]
            if fit_intercept:
                yb = yb - device_bias
            CXY = torch.addmm(CXY, xb.T, yb, out=CXY)
            del xb, yb
        del X, Y

    A = torch.linalg.multi_dot((CXY.T, CXX_pinv, CXY))  # [out_features, out_features]
    if svd_niter is not None:
        _, _, V = torch.svd_lowrank(A, q=rank, niter=svd_niter)
    else:
        _, _, VT = torch.linalg.svd(A)
        V = VT[:rank, :].T.clone()
        del VT
    # VT: [out_features, rank]
    W = torch.linalg.multi_dot((CXX_pinv, CXY, V))
    W, VT = W.to(source_device), V.T.to(source_device).contiguous()
    if fit_intercept:
        bias = bias.to(source_device)
    return W, VT, bias


def convert_rrr_to_module(W: torch.Tensor, V: torch.Tensor, bias: Optional[torch.Tensor]) -> nn.Module:
    first = nn.Linear(*W.shape, dtype=W.dtype, device=W.device, bias=False)
    second = nn.Linear(*V.shape, dtype=V.dtype, device=V.device, bias=bias is not None)
    with torch.no_grad():
        first.weight[...] = W.T
        second.weight[...] = V.T
        if bias is not None:
            second.bias[...] = bias
    return nn.Sequential(first, second)


if __name__ == '__main__':
    X = torch.randn(10_000, 32)
    y = X @ torch.randn(32, 20) + torch.randn(20)

    X = X.half()
    y = y.half()

    for rank in (1, 2, 4, 8, 16, 20):
        model = convert_rrr_to_module(*reduced_rank_regression(
            X, y, rank=rank, reg_rate=0.0000001,
            compute_device=torch.device('cuda'), compute_dtype=torch.float32, chunk_size=32, fit_intercept=True))
        model.half()
        print(rank, (model(X) - y).square().mean() / y.square().mean().item())
