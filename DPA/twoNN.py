"""
TWO-NN: Intrinsic dimension estimator by a minimal neighborhood information.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, DensityMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from math import log, sqrt, exp, lgamma, pi, pow

def _twoNearestNeighbors(distances, indices, blockAn=True, block_ratio=20, frac=1):
    """
    Parameters
    ----------
    distances: array [n_samples, k_max]
        Distances to the k_max neighbors of each points.
 
    indices : array [n_samples, k_max]
        Indices of the k_max neighbors of each points.

    blockAn : bool, default=True
        If blockAn is True the algorithm perform a block analysis that allows discriminating the relevant dimensions 
        as a function of the block size. This allows to study the stability of the estimation with respect to
        changes in the neighborhood size, which is crucial for ID estimations when the data lie on a manifold perturbed 
        by a high-dimensional noise. 

    block_ratio : int, default=20
        Set the minimum size of the blocks as n_samples/block_ratio. If blockAn=False, block_ratio is ignored.

    frac : float, default=1
        Define the fraction of points in the data set used for ID calculation. By default the full data set is used.   
    """

    return 2

class twoNearestNeighbors(BaseEstimator, DensityMixin):
    """ID-estimator that employs only the distances to the first two nearest neighbors of each point.


    Parameters
    ----------
    metric : string, or callable
        The distance metric to use. 
        If metric is a string, it must be one of the options allowed by 
        scipy.spatial.distance.pdist for its metric parameter, or a metric listed in 
        pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is "precomputed", X is assumed to
        be a distance matrix. Alternatively, if metric is a callable function, it is 
        called on each pair of instances (rows) and the resulting value recorded. The 
        callable should take two arrays from X as input and return a value indicating 
        the distance between them. Default is 'euclidean'.

    blockAn : bool, default=True
        If blockAn is True the algorithm perform a block analysis that allows discriminating the relevant dimensions 
        as a function of the block size. This allows to study the stability of the estimation with respect to
        changes in the neighborhood size, which is crucial for ID estimations when the data lie on a manifold perturbed 
        by a high-dimensional noise. 

    block_ratio : int, default=20
        Set the minimum size of the blocks as n_samples/block_ratio. If blockAn=False, block_ratio is ignored.

    frac : float, default=1
        Define the fraction of points in the data set used for ID calculation. By default the full data set is used.   

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    dim_ = int
        The intrinsic dimensionality
 
    """
    def __init__(self, metric='euclidean',  blockAn=True, block_ratio=20, frac=1, n_jobs=None):
        self.metric = 'euclidean'
        self.blockAn = blockAn
        self.block_ratio = block_ratio
        self.frac = frac
        self.n_jobs = n_jobs

        if self.frac > 1:
            raise ValueError("frac should be between 0 and 1.")
    
    
    def fit(self, X, y=None):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : array [n_samples, n_samples] if metric == “precomputed”, or,
            [n_samples, n_features] otherwise
            The input samples.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = check_array(X, order='C', accept_sparse=True)

        if self.block_ratio >= X.shape[0]:
            raise ValueError("block_ratio is larger than the sample size, the minimum size for block analysis \
                        would be zero. Please set a lower value.")

        if self.metric == "precomputed":
            nbrs = NearestNeighbors(n_neighbors=3, # Only two neighbors used; the point i is counted in its neighborhood
                                          algorithm="auto",
                                        metric=self.metric,
                                        n_jobs=self.n_jobs).fit(X)
        else:
            if self.frac<1:
                X = X[np.random.choice(X.shape[0], int(round(X.shape[0]*self.frac)), replace=False), :]
            nbrs = NearestNeighbors(n_neighbors=3, # Only two neighbors used; the point i is counted in its neighborhood
                                         algorithm="brute",
                                        metric=self.metric, 
                                        n_jobs=self.n_jobs).fit(X)
            #self.matrix_ = kneighbors_graph(X, n_neighbors=self.k_max,
            #                                        mode="distance",
            #                                        include_self=True)
        self.distances_, self.indices_ = nbrs.kneighbors(X) 

     
        self.dim_ = _twoNearestNeighbors(self.distances_, 
                                           self.indices_,
                                    blockAn=self.blockAn, 
                             block_ratio=self.block_ratio, 
                                          frac=self.frac)
        self.is_fitted_ = True
        return self


