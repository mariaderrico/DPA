"""
The pointwise-adaptive k-NN density estimator.
"""
import numpy as np
from sklearn.base import BaseEstimator, DensityMixin, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from math import log, sqrt, exp, lgamma, pi, pow
#from . import twoNN


VALID_METRIC = ['precomputed', 'euclidean']
VALID_DIM = ['auto', 'two_nn']

def _PointAdaptive_kNN(distances, indices, k_max=1000, D_thr=23.92812698, dim=None):
    """
    Parameters
    ----------
    distances: array [n_samples, k_max]
        Distances to the k_max neighbors of each points.
  
    indices : array [n_samples, k_max]
        Indices of the k_max neighbors of each points. 

    k_max : int, default=1000
        The maximum number of nearest-neighbors considered by the procedure that returns the
        largest number of neighbors ``\hat{k}`` for which the condition of constant density 
        holds, within a given level of confidence. If the number of points in the sample N is 
        less than the default value, k_max will be set automatically to the value ``N/2``.
    
    D_thr : float, default=23.92812698
        Set the level of confidence. The default value corresponds to a p-value of 
        ``10**{-6}`` for a ``\chiˆ2`` distribution with one degree of freedom.

    dim : int
        Intrinsic dimensionality of the sample.

    Results
    -------
    densities : array [n_samples]
        The logarithm of the density at each point.
    
    err_densities : array [n_samples]
        The uncertainty in the density estimation, obtained by computing 
        the inverse of the Fisher information matrix.

    k_hat : array [n_samples]
        The optimal number of neighbors for which the condition of constant density holds.

    dc : array [n_sample]
        The radius of the optimal neighborhood for each point.
    
    References
    ----------
 

    """
    # Compute the volume of the dim-sphere with unitary radius
    V1 = exp(dim/2.*log(pi)-lgamma((dim+2)/2.))    
    N = distances.shape[0]
    k_hat = []
    dc = []
    densities = []
    err_densities = []
    for i in range(0,N):
        k = 3 # Minimum number of neighbors required
        Dk = 0 
        while (k<k_max and Dk<=D_thr):
            j = k+1
            # The point i is counted in its neighborhood, so the k-th neighbor is at position k, and not k-1
            vi = V1*exp(dim*log(distances[i][k])) 
            vj = V1*exp(dim*log(distances[indices[i][j]][k])) 
            Dk = -2.*k*(log(vi)+log(vj)-2.*log(vi+vj)+log(4.))
            k += 1
        k_hat.append(k-1)
        dc.append(distances[i][k-1]) #(k-1)-th neighbor is at position k-1
        densities.append(log(k-1)-(log(V1)+dim*log(dc[i]))) 
        err_densities.append(sqrt((4.*(k-1)+2.)/((k-1)*(k-2))))

    return densities, err_densities, k_hat, dc


class PointAdaptive_kNN(BaseEstimator, DensityMixin):
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
        Intrinsic dimensionality of the sample. If dim is provided, dim_algo is ignored.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    distances_ : array [n_samples, k_max]
        Distances to the k_max neighbors of each points.

    indices_ : array [n_samples, k_max]
        Indices of the k_max neighbors of each points.
 
    densities_ : array [n_samples]
        The logarithm of the density at each point.
    
    err_densities_ : array [n_samples]
        The uncertainty in the density estimation, obtained by computing 
        the inverse of the Fisher information matrix.

    k_hat_ : array [n_samples]
        The optimal number of neighbors for which the condition of constant density holds.

    dc_ : array [n_sample]
        The radius of the optimal neighborhood for each point.


    Examples
    --------
    >>> from DPApipe.DPA import PAk
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> est = PAk.PointAdaptive_kNN().fit(X)
    >>> est.distances_
    array([[0.        , 1.        , 1.        ],
           [0.        , 1.        , 1.41421356],
           [0.        , 1.        , 1.41421356],
           [0.        , 1.41421356, 2.23606798],
           [0.        , 1.        , 2.23606798],
           [0.        , 1.        , 1.41421356]])
    >>> est.indices_
    array([[0, 1, 2],
           [1, 0, 2],
           [2, 0, 1],
           [3, 5, 4],
           [4, 5, 3],
           [5, 4, 3]])    
    
    References
    ----------

        
    """
    def __init__(self, k_max=1000, D_thr=23.92812698, metric="euclidean", dim_algo="auto", dim=None, n_jobs=None):
        self.k_max = k_max
        self.D_thr = D_thr
        self.metric = metric
        self.dim_algo = dim_algo
        self.dim = dim
        self.n_jobs = n_jobs

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
                self.dim = X.shape[1]
            elif self.dim_algo == "two_nn":
                self.dim = X.shape[1] #two_nn().fit(X)
            else:
                pass
     
        if self.k_max > X.shape[0]:
            self.k_max = int(X.shape[0]/2)
        if self.k_max < 3:
            raise ValueError("k_max is below 3, the minimum value required for \
                        statistical significance. Please use a larger datasets.")

        if self.metric == "precomputed":
            nbrs = NearestNeighbors(n_neighbors=self.k_max+1, # The point i is counted in its neighborhood 
                                          algorithm="auto", 
                                        metric=self.metric,
                                        n_jobs=self.n_jobs).fit(X)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k_max+1, # The point i is counted in its neighborhood 
                                         algorithm="brute", 
                                        metric=self.metric, 
                                        n_jobs=self.n_jobs).fit(X)
            #self.matrix_ = kneighbors_graph(X, n_neighbors=self.k_max,
            #                                        mode="distance",
            #                                        include_self=True)
        self.distances_, self.indices_ = nbrs.kneighbors(X) 

        self.densities_, self.err_densities_, self.k_hat_, self.dc_ = _PointAdaptive_kNN(self.distances_, 
                                                                                 self.indices_,
                                                                              k_max=self.k_max, 
                                                                              D_thr=self.D_thr, 
                                                                                  dim=self.dim)
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

