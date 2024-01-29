# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:31:20 2021

@author: Peter Clark
"""
import numpy as np
from typing import List

def interpolate(fin, x, y, xd, yd, interp_order=3, 
                wrap: List[bool] = [True, False]):
    """
    Interpolate from 2D gridded data.

    Parameters
    ----------
    fin : list(numpy.ndarray)
        List of input data on 2D array.
    x : numpy.ndarray
        2D array of x coordinate points corresponding to fin data.
    y : numpy.ndarray
        2D array of x coordinate points corresponding to fin data.
    xd : numpy.ndarray
        x coordinates of required output points.
    yd : numpy.ndarray
        y coordinates of required output points.
    interp_order : int (optional default=3)
        order of Lagrange interpolation
    wrap : list[bool], optional
        True for any dimension means periodic wrapping. Otherwise fixed
        values outside boundaries. The default is [True, False].

    Returns
    -------
    f_int : list(numpy.ndarray)
        fin interpolated to output points.

    """   
    dx = x[1,0] - x[0,0]
    dy = y[0,1] - y[0,0]

    (nx, ny) = np.shape(fin[0])
    pos = np.ndarray(( nx, ny, 2))
    pos[..., 0] = (xd - x[0, 0]) / dx
    pos[..., 1] = (yd - y[0, 0]) / dy
    f_int = multi_dim_lagrange_interp(fin, pos, order=interp_order,
                wrap=wrap)

    return f_int

def multi_dim_lagrange_interp(data: List[np.ndarray], pos: np.ndarray,
                              order: int = 3,
                              wrap: List[bool] = None) -> List[np.ndarray]:
    """
    Multidimensional arbitrary order Lagrange interpolation.

    Parameters
    ----------
    data : list[np.ndarray]
        List of N-dimensional numpy arrays with data to interpolate.
    pos : np.ndarray [..., N]
        Positions in N-dimensional space to interpolate to in grid units.
    order : int, optional
        Lagrange polynomial order. The default is 3.
    wrap : list[bool], optional
        True for any dimension means periodic wrapping. Otherwise fixed
        values outside boundaries. The default is None, i.e. wrap all dims.

    Returns
    -------
    List of nump arrays containing data interpolated to pos, retaing structure
    of pos apart from last dimension.

    @author : Peter Clark (C) 2021
    """
    if type(data) is not list:
        raise TypeError('Argument data should be a list of numpy arrays.')
    if type(pos) is not np.ndarray:
        raise TypeError('Argument pos should be numpy array.')

# Stencil of points to use assuming x between 0 and 1.
# So for order=1, [0,1], order=3, [-1,0,1,2] etc.

    local_grid = np.arange(order+1)-order//2

# Weights in 1D for each stencil point.
    grid_weight = np.ones(order+1)
    for i in range(order+1):
        for j in range(order+1):
            if i==j:
                continue
            else:
                grid_weight[i] *=  1.0/(i-j)

    npts = np.shape(data[0])
    ndims = len(npts)

# Default is wrap if not supplied.
    if wrap is None:
        wrap = [True for i in range(ndims)]

# Make sure points to interpolate to are in acceptable range.
    for dim in range(ndims):
        if wrap[dim]:
            pos[..., dim] %= npts[dim]
        else:
            pos[..., dim] = np.clip(pos[..., dim], 0, npts[dim]-1)

# Split x into integer and fractional parts.

    idim = np.floor(pos).astype(int)
    xdim = pos - idim

    def compute_interp(weight, inds, off, dim):
        """
        Recursive function to compute Lagrange polynomial interpolation.

        Parameters
        ----------
        weight : float or numpy array.
            Weight for current gridpoint in stencil.
        inds : list of numpy arrays of interger indices.
            Actual gridpoints for this point in stencil.
        off : int
            Position in stencil for current dimension.
        dim : int
            Dimension.

        Returns
        -------
        Either contribution from this gridpoint or final result,
        data list interpolated to pos.

        """
        if dim >= 0:
#            print('Finding weight')
            # Find indices for stencil position in this dimension.
            ii = (idim[..., dim] + local_grid[off])
            if wrap[dim]:
                ii %= npts[dim]
            else:
                ii = np.clip(ii, 0, npts[dim]-1)
            inds.append(ii)

            # Find weight for this stencil position u=in this dimension.
            w = grid_weight[off]
            for woffset in range(order+1):
                if woffset == off:
                    continue
                else:
                    w *= (xdim[..., dim] - local_grid[woffset])

            weight *= w

        if dim == ndims-1:
            # Weight is now weight over all dimensions, so we can find
            # contribution to final result.
            contrib = []
            for d in data:
                o = d[tuple(inds)] * weight
                contrib.append(o)

            return contrib

        else:
            # Find contributions from each dimension and add in.
            interpolated_data = None
            for offset in range(order+1):
                contrib = compute_interp(weight.copy(), inds.copy(),
                                         offset, dim+1)
                if contrib is not None:
                    if interpolated_data is not None:
                        for l, c in enumerate(contrib):
                            interpolated_data[l] += c
                    else:
                        interpolated_data = contrib
            return interpolated_data

    weight = np.ones_like(xdim[..., 0])
    inds = []
    off = -1
    dim = -1
    output = compute_interp(weight, inds, off, dim)

    return output
