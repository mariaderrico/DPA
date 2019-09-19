"""
The pointwise-adaptive k-NN density estimator.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
#from . import twoNN


VALID_METRIC = ['precomputed', 'euclidean']
VALID_DIM = ['auto', 'two_nn']

class PointAdaptive_kNN(BaseEstimator):
    """ The pointwise-adaptive k-NN density estimator.

    Parameters
    ----------
    k_max : int, default=1000
        The maximum number of nearest-neighbors considered by the procedure that returns the
        largest number of neighbors ``\hat{k}`` for which the condition of constant density 
        holds, within a given level of confidence. If the number of points in the sample N is 
        less than the default value, k_max will be set automatically to the value ``N/2``.
    
    D_thr : float, default=23.92812698
        Set the level of confidence. The default value corresponds to a p-value of 
        ``10**{-6}`` for a ``\chiˆ2`` distribution with one degree of freedom.

    metric : string, or callable
        The distance metric to use. 
        If metric is a string, it must be one of the options allowed by 
        scipy.spatial.distance.pdist for its metric parameter, or a metric listed in 
        pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is "precomputed", X is assumed to
        be a distance matrix. Alternatively, if metric is a callable function, it is 
        called on each pair of instances (rows) and the resulting value recorded. The 
        callable should take two arrays from X as input and return a value indicating 
        the distance between them. Default is 'euclidean'.

    dim_algo : string, or callable
        Method for intrinsic dimensionality calculation. If dim_algo is "auto", dim is assumed to be
        equal to n_samples. If dim_algo is a string, it must be one of the options allowed by VALID_DIM. 

    dim : int
        Intrinsic dimensionality of the sample. If dim is provaded, dim_algo is ignored.
    """
    def __init__(self, k_max=1000, D_thr=23.92812698, metric="euclidean", dim_algo="auto", dim=None):
        self.k_max = k_max
        self.D_thr = D_thr
        self.metric = metric
        self.dim_algo = dim_algo
        self.dim = dim

        if metric not in VALID_METRIC:
            raise ValueError("invalid metric: '{0}'".format(metric))

        if dim_algo not in VALID_DIM:
            raise ValueError("invalid dim_algo: '{0}'".format(dim_algo))

       

    def fit(self, X, y=None):
        """Fit the PAk estimator on the data.

        Parameters
        ----------
        X : array [n_samples, n_samples] if metric == “precomputed”, or, 
            [n_samples, n_features] otherwise
            #{array-like, sparse matrix}, shape (n_samples, n_features)
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

        if not self.dim:
            if self.dim_algo == "auto":
                self.dim = X.shape[0]
            elif self.dim_algo == "two_nn":
                self.dim = X.shape[0] #two_nn().fit(X)
            else:
                pass
     
        if self.k_max > X.shape[0]:
            self.k_max = int(X.shape[0]/2)

        if self.metric == "precomputed":
            nbrs = NearestNeighbors(n_neighbors=self.k_max, algorithm="auto", metric=self.metric).fit(X)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k_max, algorithm="brute", metric=self.metric).fit(X)
        self.distances_, self.indices_ = nbrs.kneighbors(X) 

        self.is_fitted_ = True
        return self


    def score(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        density : ndarray, shape (n_samples,)
            The array of log(density) evaluations. 

        err_density : ndarray, shape (n_samples,)
            The uncertainty on the density estimation.

        k_hat : ndarray, shape (n_samples,)
            The optimal neighborhood size.
 
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64)

