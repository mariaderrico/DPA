"""
Non-parametric Density Peak clustering: 
Automatic topography of high-dimensional data sets 
"""
import numpy as np
from sklearn.base import BaseEstimator, DensityMixin, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from math import log, sqrt, exp, lgamma, pi, pow
from .twoNN import twoNearestNeighbors
from .PAk import PointAdaptive_kNN
import heapq


VALID_METRIC = ['precomputed', 'euclidean']
VALID_DIM = ['auto', 'twoNN']
VALID_DENSITY = ['PAk', 'kNN']

def _DensityPeakAdvanced(densities, err_densities, k_hat, distances, indices, Z):  
    """Main function implementing the Density Peak Advanced clustering algorithm: 

    * Automatic detection of cluster centers
    * Point assignament to clusters in order of decreasing g
    * Topography reconstruction: search of saddle points and cluster merging

    Parameters
    ----------
    densities : array [n_samples]
        The logarithm of the density at each point.
    
    err_densities : array [n_samples]
        The uncertainty in the density estimation, obtained by computing 
        the inverse of the Fisher information matrix.

    k_hat : array [n_samples]
        The optimal number of neighbors for which the condition of constant density holds.

    distances: array [n_samples, k_max+1]
        Distances to the k_max neighbors of each points. The point itself is included in the array. 
  
    indices : array [n_samples, k_max+1]
        Indices of the k_max neighbors of each points. The point itself is included in the array.

    Z : float, default = 1
        The number of standard deviations, which fixes the level of statistical confidence at which 
        one decides to consider a cluster meaningful.

    Results
    -------
    labels : array [Nclus]
        The clustering labels assigned to each point in the data set.
    
    topography : array [Nclus, Nclus]
        Let be Nclus the number of clusters, the topography consists in a Nclus × Nclus symmetric matrix,
        in which the diagonal entries are the heights of the peaks and the off-diagonal entries are the
        heights of the saddle points.

    centers : array [Nclus]
        The list of points identified as the centers of the Nclus statistically significant clusters.

    """

    # We define as cluster centers the local maxima of g, where g is defined as density-err_density.
    g = [densities[i]-err_densities[i] for i in range(0,len(densities))]

    # Automatic detection of cluster centers
    #---------------------------------------
    # Criterion 1 from Heuristic 1
    centers = []
    N = len(densities)
    for i in range(0, N):
        putative_center = True
        for j in indices[i][1:k_hat[i]+1]:
            if g[j]>g[i]:
                putative_center = False
                break
        if putative_center:
            centers.append(i)
    # Criterion 2 from Heuristic 1
    for c in centers:
        for i in range(0,N):
            if g[c]<g[i] and c in indices[i][1:k_hat[i]+1]:
                centers.remove(c)
                break
    Nclus = len(centers)

    # Sort index by decreasing g
    #---------------------------
    ig_sort = np.argsort([-x for x in g])

    # Assign points to clusters
    #--------------------------
    # Assign all the points that are not centers to the same cluster as the nearest point with higher g. 
    # This assignation is performed in order of decreasing g
    clu_labels = [-1]*N
    for c in centers:
        clu_labels[c] = centers.index(c)
    for i in range(0,N):
        el = ig_sort[i]
        k = 0
        while (clu_labels[el]==-1):
            k=k+1
            clu_labels[el] = clu_labels[indices[el][k]] # the point with higher g is already assigned by construction

    # Topography reconstruction
    #--------------------------
    # Finding saddle points between pair of clusters c and c'.
    # Criterion 1 from Heuristic 2:
    # point i belonging to c is at the border if its closest point j belonging to c′ is within a distance k_hat[i] 
    border_dict = {}
    g_saddle = {}
    for i in range(0,N):
        for k in range(0,k_hat[i]):
            j = indices[i][k+1]
            if clu_labels[j]!=clu_labels[i]:
                if (i, clu_labels[i]) not in border_dict.keys():
                    border_dict[(i, clu_labels[i])] = [-1]*Nclus
                    border_dict[(i, clu_labels[i])][clu_labels[j]] = j
                elif border_dict[(i, clu_labels[i])][clu_labels[j]]==-1:
                    border_dict[(i, clu_labels[i])][clu_labels[j]] = j
                else:
                    pass
    # Criterion 2 from Heuristic 2:
    # check if i is the closest point to j among those belonging to c.
    for i,c in border_dict.keys():
        for cp in range(Nclus):
            j = border_dict[(i,c)][cp]
            if j!=-1:
                if (j,cp) in border_dict.keys() and border_dict[(j,cp)][c] == i:
                    m_c = min(c,cp)
                    M_c = max(c,cp)
                    if (m_c, M_c) in g_saddle.keys():
                        g_saddle[(m_c,M_c)] = (i, max(g_saddle[(m_c,M_c)][1], g[i]))
                    else:
                        g_saddle[(m_c,M_c)] = (i, g[i])

    Rho_bord = np.zeros((Nclus,Nclus),dtype=float)
    Rho_bord_err = np.zeros((Nclus,Nclus),dtype=float)
    for c,cp in g_saddle.keys():
        i = g_saddle[(c,cp)][0]
        Rho_bord[c][cp] = densities[i]
        Rho_bord[cp][c] = densities[i]
        Rho_bord_err[c][cp] = err_densities[i]
        Rho_bord_err[cp][c] = err_densities[i]
    #for c in range(0,Nclus):
    #    Rho_bord[c][c] = -1
    #    Rho_bord_err[c][c] = 0

    topography_temp = []
    for i in range(0, Nclus-1):
        for j in range(i+1, Nclus):
            topography_temp.append([i,j, Rho_bord[i][j], Rho_bord_err[i][j]])

    # Merging:
    # TODO

    labels = clu_labels

    return labels, topography_temp, g, centers

   
class DensityPeakAdvanced(BaseEstimator, DensityMixin):
    """Class definition for the non-parametric Density Peak clustering.

    The default pipeline makes use of the PAk density estimator and of the TWO-NN intristic dimension estimator.
    The densities and the corresponding errors can also be provided as precomputed arrays.
 
    Parameters
    ----------

    Z : float, default = 1
        The number of standard deviations, which fixes the level of statistical confidence at which 
        one decides to consider a cluster meaningful.

    metric : string, or callable
        The distance metric to use. 
        If metric is a string, it must be one of the options allowed by 
        scipy.spatial.distance.pdist for its metric parameter, or a metric listed in 
        pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is "precomputed", X is assumed to
        be a distance matrix. Alternatively, if metric is a callable function, it is 
        called on each pair of instances (rows) and the resulting value recorded. The 
        callable should take two arrays from X as input and return a value indicating 
        the distance between them. Default is 'euclidean'.

    densities : array [n_samples], default = None
        The logarithm of the density at each point. If provided, the following parameters are ignored:
        density_algo, k_max, D_thr.

    err_densities : array [n_samples], default = None
        The uncertainty in the density estimation, obtained by computing 
        the inverse of the Fisher information matrix.

    k_hat : array [n_samples], default = None
        The optimal number of neighbors for which the condition of constant density holds.

    density_algo : string, default = "PAk"
        Define the algorithm to use as density estimator. It mast be one of the options allowed by 
        VALID_DENSITY. 
        
    k_max : int, default=1000
        This parameter is considered if density_algo is "PAk" or "kNN", it is ignored otherwise.
        k_max set the maximum number of nearest-neighbors considered by the density estimator.
        If density_algo="PAk", k_max is used by the algorithm in the search for the 
        largest number of neighbors ``\hat{k}`` for which the condition of constant density 
        holds, within a given level of confidence. 
        If density_algo="kNN", k_max set the number of neighbors to be used by the standard
        k-Nearest Neighbor algorithm.
        If the number of points in the sample N is 
        less than the default value, k_max will be set automatically to the value ``N/2``, if .

    D_thr : float, default=23.92812698
        This parameter is considered if density_algo is "PAk", it is ignored otherwise.
        Set the level of confidence in the PAk density estimator. The default value corresponds to a p-value of 
        ``10**{-6}`` for a ``\chiˆ2`` distribution with one degree of freedom.

    dim : int, default = None
        Intrinsic dimensionality of the sample. If dim is provided, the following parameters are ignored:
        dim_algo, blockAn, block_ratio, frac.

    dim_algo : string, or callable, default="twoNN"
        Method for intrinsic dimensionality calculation. If dim_algo is "auto", dim is assumed to be
        equal to n_samples. If dim_algo is a string, it must be one of the options allowed by VALID_DIM. 

    blockAn : bool, default=True
        This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
        If blockAn is True the algorithm perform a block analysis that allows discriminating the relevant dimensions 
        as a function of the block size. This allows to study the stability of the estimation with respect to
        changes in the neighborhood size, which is crucial for ID estimations when the data lie on a 
        manifold perturbed by a high-dimensional noise. 

    block_ratio : int, default=20
        This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
        Set the minimum size of the blocks as n_samples/block_ratio. If blockAn=False, block_ratio is ignored.

    frac : float, default=1
        This parameter is considered if dim_algo is "twoNN", it is ignored otherwise.
        Define the fraction of points in the data set used for ID calculation. By default the full data set is used.   
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    labels_ : array [Nclus]
        The clustering labels assigned to each point in the data set.

    topography_ : array [Nclus, Nclus]
        Let be Nclus the number of clusters, the topography consists in a Nclus × Nclus symmetric matrix, 
        in which the diagonal entries are the heights of the peaks and the off-diagonal entries are the 
        heights of the saddle points. 

    distances_ : array [n_samples, k_max+1]
        Distances to the k_max neighbors of each points. The point itself is included in the array.

    indices_ : array [n_samples, k_max+1]
        Indices of the k_max neighbors of each points. The point itself is included in the array.
 
    k_hat_ : array [n_samples], default = None
        The optimal number of neighbors for which the condition of constant density holds.

    centers_ :array [Nclus]
        The clustering labels assigned to each point in the data set.

    Example
    -------
    # TODO


    References
    ----------
    # TODO
    """

    def __init__(self, Z=1, metric="euclidean", densities=None, err_densities=None, k_hat=None,
                      density_algo="PAk", k_max=1000, D_thr=23.92812698, dim=None, dim_algo="twoNN", 
                       blockAn=True, block_ratio=20, frac=1, n_jobs=None):
        self.Z = Z
        self.metric = metric
        self.densities = densities
        self.err_densities = err_densities
        self.k_hat = k_hat
        self.density_algo = density_algo
        self.k_max = k_max
        self.D_thr = D_thr
        self.dim = dim
        self.dim_algo = dim_algo
        self.blockAn = blockAn
        self.block_ratio = block_ratio
        self.frac = frac
        self.n_jobs = n_jobs

        if metric not in VALID_METRIC:
            raise ValueError("invalid metric: '{0}'".format(metric))

        if dim_algo not in VALID_DIM:
            raise ValueError("invalid dim_algo: '{0}'".format(dim_algo))

        if density_algo not in VALID_DENSITY:
            raise ValueError("invalid dim_algo: '{0}'".format(density_algo))

        if self.densities or self.err_densities  or self.k_hat:
            # TODO: decide whether to raise a worning instead and automatically run PAk. 
            raise ValueError("DPA requires the error estimation and optimal neighborhood along \
                              with the densities. If not available, use the default PAk estimator") 

        if self.dim_algo == "twoNN" and self.frac > 1:
            raise ValueError("frac should be between 0 and 1.")

        
    def fit(self, X, y=None):
        """Fit the DPA clustering on the data.

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

        if not self.dim:
            if self.dim_algo == "auto":
                self.dim = X.shape[1]
            elif self.dim_algo == "twoNN":
                if self.block_ratio >= X.shape[0]:
                    raise ValueError("block_ratio is larger than the sample size, the minimum size for \
                                      block analysis would be zero. Please set a lower value.")
                self.dim = twoNearestNeighbors(blockAn=self.blockAn, block_ratio=self.block_ratio, 
                                               frac=self.frac, n_jobs=self.n_jobs).fit(X).dim_
            else:
                pass

        """
        if self.k_max > X.shape[0]:
            self.k_max = int(X.shape[0]/2)
        if self.k_max < 3:
            raise ValueError("k_max is below 3, the minimum value required for \
                        statistical significance. Please use a larger datasets.")
        """
    
        # If densities, uncertainties and k_hat are provided as input, compute only the
        # matrix of nearest neighbor: 
        
        if self.densities and self.err_densities and self.k_hat:
            self.k_max = max(self.k_hat)
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
            self.distances_, self.indices_ = nbrs.kneighbors(X) 
        elif self.density_algo == "PAk":
            PAk = PointAdaptive_kNN(k_max=self.k_max, D_thr=self.D_thr, metric=self.metric, 
                                               dim_algo=self.dim_algo, blockAn=self.blockAn, 
                                               block_ratio=self.block_ratio,
                                               frac=self.frac, dim=self.dim, n_jobs=self.n_jobs).fit(X)
            self.distances_ = PAk.distances_
            self.indices_ = PAk.indices_ 
            self.densities = PAk.densities_
            self.err_densities = PAk.err_densities_
            self.k_hat = PAk.k_hat_

        else:
            # TODO: implement option for kNN
            pass

        self.labels_, self.topography_, self.g_, self.centers_ = _DensityPeakAdvanced(self.densities, 
                                                              self.err_densities, self.k_hat,
                                                              self.distances_, self.indices_, self.Z)
                                                              

        self.is_fitted_ = True
        return self



