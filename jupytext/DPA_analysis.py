# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # The Density Peak Advanced clustering algorithm
#
# ----------------
# Load the package:

# +
import io
import sys
import pandas as pd
import numpy as np
from Pipeline import DPA
import time

# %load_ext autoreload
# %autoreload 2
# -

# Read input csv file:

data_F1 = pd.read_csv("Pipeline/tests/benchmarks/Fig1.dat", sep=" ", header=None)

# How to run Density Peak Advanced clustering:
#
#     The default pipeline makes use of the PAk density estimator and of the TWO-NN intristic dimension estimator.
#     The densities and the corresponding errors can also be provided as precomputed arrays.
#
#     Parameters
#     ----------
#
#     Z : float, default = 1
#         The number of standard deviations, which fixes the level of statistical confidence at which
#         one decides to consider a cluster meaningful.
#
#     metric : string, or callable
#         The distance metric to use.
#         If metric is a string, it must be one of the options allowed by
#         scipy.spatial.distance.pdist for its metric parameter, or a metric listed in
#         pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is "precomputed", X is assumed to
#         be a distance matrix. Alternatively, if metric is a callable function, it is
#         called on each pair of instances (rows) and the resulting value recorded. The
#         callable should take two arrays from X as input and return a value indicating
#         the distance between them. Default is 'euclidean'.
#         
#     densities : array [n_samples], default = None
#         The logarithm of the density at each point. If provided, the following parameters are ignored:
#         density_algo, k_max, D_thr.
#
#     err_densities : array [n_samples], default = None
#         The uncertainty in the density estimation, obtained by computing
#         the inverse of the Fisher information matrix.
#     
#     k_hat : array [n_samples], default = None
#         The optimal number of neighbors for which the condition of constant density holds.
#         
#     nn_distances  : array [n_samples, k_max+1]
#         Distances to the k_max neighbors of each points.
#
#     nn_indices : array [n_samples, k_max+1]
#         Indices of the k_max neighbors of each points.
#
#     affinity : string or callable, default 'precomputed'
#         How to construct the affinity matrix.
#          - ``nearest_neighbors`` : construct the affinity matrix by computing a
#            graph of nearest neighbors.
#          - ``rbf`` : construct the affinity matrix using a radial basis function
#            (RBF) kernel.
#          - ``precomputed`` : interpret ``X`` as a precomputed affinity matrix.
#          - ``precomputed_nearest_neighbors`` : interpret ``X`` as a sparse graph
#            of precomputed nearest neighbors, and constructs the affinity matrix
#            by selecting the ``n_neighbors`` nearest neighbors.
#          - one of the kernels supported by
#            :func:`~sklearn.metrics.pairwise_kernels`.
#
#
#     Parameters specific of the PAk estimator:
#     -----------------------------------------
#
#     density_algo : string, default = "PAk"
#         Define the algorithm to use as density estimator. It mast be one of the options allowed by
#         VALID_DENSITY.
#
#     k_max : int, default=1000
#         This parameter is considered if density_algo is "PAk" or "kNN", it is ignored otherwise.
#         k_max set the maximum number of nearest-neighbors considered by the density estimator.
#         If density_algo="PAk", k_max is used by the algorithm in the search for the
#         largest number of neighbors ``\hat{k}`` for which the condition of constant density
#         holds, within a given level of confidence.
#         If density_algo="kNN", k_max set the number of neighbors to be used by the standard
#         k-Nearest Neighbor algorithm.
#         If the number of points in the sample N is
#         less than the default value, k_max will be set automatically to the value ``N/2``.
#         
#     D_thr : float, default=23.92812698
#         This parameter is considered if density_algo is "PAk", it is ignored otherwise.
#         Set the level of confidence in the PAk density estimator. The default value corresponds to a p-value of
#         ``10**{-6}`` for a ``\chiˆ2`` distribution with one degree of freedom.
#
#     dim : int, default = None
#         Intrinsic dimensionality of the sample. If dim is provided, the following parameters are ignored:
#         dim_algo, blockAn, block_ratio, frac.
#
#     dim_algo : string, or callable, default="twoNN"
#         Method for intrinsic dimensionality calculation. If dim_algo is "auto", dim is assumed to be
#         equal to n_samples. If dim_algo is a string, it must be one of the options allowed by VALID_DIM.
#
#     Parameters specific of the TWO-NN estimator:
#     --------------------------------------------
#
#     blockAn : bool, default=True
#         This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
#         If blockAn is True the algorithm perform a block analysis that allows discriminating the relevant 
#         dimensions as a function of the block size. This allows to study the stability of the estimation 
#         with respect to changes in the neighborhood size, which is crucial for ID estimations when the 
#         data lie on a manifold perturbed by a high-dimensional noise.
#
#     block_ratio : int, default=20
#         This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
#         Set the minimum size of the blocks as n_samples/block_ratio. If blockAn=False, block_ratio is ignored.
#         
#     frac : float, default=1
#         This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
#         Define the fraction of points in the data set used for ID calculation. By default the full 
#         data set is used.
#
#
#
#     Attributes
#     ----------
#     labels_ : array [Nclus]
#         The clustering labels assigned to each point in the data set.
#
#     halos_ : array [Nclus]
#         The clustering labels assigned to each point in the data set. Points identified as halos have
#         label equal to zero.
#
#     topography_ : array [Nclus, Nclus]
#         Let be Nclus the number of clusters, the topography consists in a Nclus × Nclus symmetric matrix,
#         in which the diagonal entries are the heights of the peaks and the off-diagonal entries are the
#         heights of the saddle points.
#
#     nn_distances_ : array [n_samples, k_max+1]
#         Distances to the k_max neighbors of each points. The point itself is included in the array.
#
#     nn_indices_ : array [n_samples, k_max+1]
#         Indices of the k_max neighbors of each points. The point itself is included in the array.
#
#     k_hat_ : array [n_samples], default = None
#         The optimal number of neighbors for which the condition of constant density holds.
#
#     centers_ :array [Nclus]
#         The clustering labels assigned to each point in the data set.
#         
#     dim_ : int,
#         Intrinsic dimensionality of the sample. If ``dim`` is not provided, ``dim_`` is set
#         to the number of features in the input file.
#
#     k_max_ : int
#         The maximum number of nearest-neighbors considered by the procedure that returns the
#         largest number of neighbors ``k_hat`` for which the condition of constant density
#         holds, within a given level of confidence. If the number of points in the sample `N` is
#         less than the default value, k_max_ will be set automatically to the value ``N/2``.
#
#     densities_ : array [n_samples]
#         If not provided by the parameter ``densities``, it is computed by using the `PAk` density estimator.
#
#     err_densities_ : array [n_samples]
#         The uncertainty in the density estimation. If not provided by the parameter ``densities``, it is
#         computed by using the `PAk` density estimator.
#

est = DPA.DensityPeakAdvanced(Z=1.5)

start=time.time()
est.fit(data_F1)
end=time.time()
print(end-start)

est.topography_

# Running again with a different Z without the need of recomputing the neighbors-densities

params = est.get_computed_params()
est.set_params(**params)
est.set_params(Z=1)
start=time.time()
est.fit(data_F1)
end=time.time()
print(end-start)

# The PAk and twoNN estimator can be used indipendently from the DPA clustering method.

# +
from Pipeline import PAk
from Pipeline import twoNN

rho_est = PAk.PointAdaptive_kNN()
d_est = twoNN.twoNearestNeighbors()


# +
results = rho_est.fit(data_F1)
print(results.densities_[:10])

dim = d_est.fit(data_F1).dim_
print(dim)
# -


