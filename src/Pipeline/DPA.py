# coding=utf-8
# Non-parametric Density Peak clustering: 
# Automatic topography of high-dimensional data sets 
#
# Author: Maria d'Errico <mariaderr@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from sklearn.base import BaseEstimator, ClusterMixin, DensityMixin, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from math import log, sqrt, exp, lgamma, pi, pow
from Pipeline.twoNN import twoNearestNeighbors
from Pipeline.PAk import PointAdaptive_kNN


VALID_METRIC = ['precomputed', 'euclidean','cosine']
VALID_DIM = ['auto', 'twoNN']
VALID_DENSITY = ['PAk', 'kNN']

def _DensityPeakAdvanced(densities, err_densities, k_hat, distances, indices, Z):  
    """Main function implementing the Density Peak Advanced clustering algorithm: 

    * Automatic detection of cluster centers
    * Point assignament to clusters in order of decreasing `g`
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


    Attributes
    ----------

    labels : array [Nclus]
        The clustering labels assigned to each point in the data set.

    halos : array [Nclus]
        The clustering labels assigned to each point in the data set. Points identified as halos have 
        clustering lable equal to ``-1``.
    
    topography : array [Nclus, Nclus]
        Let be Nclus the number of clusters, the topography consists in a Nclus × Nclus symmetric matrix,
        in which the diagonal entries are the heights of the peaks and the off-diagonal entries are the
        heights of the saddle points.

    centers : array [Nclus]
        The list of points identified as the centers of the Nclus statistically significant clusters.

    """
    from Pipeline import _DPA

    # We define as cluster centers the local maxima of g, where g is defined as density-err_density.
    g = [densities[i]-err_densities[i] for i in range(0,len(densities))]

    # Automatic detection of cluster centers
    #---------------------------------------
    N = len(densities)
    centers = _DPA.get_centers(N, indices, k_hat, g)
    Nclus = len(centers)

    # Assign points to clusters
    #--------------------------
    # Assign all the points that are not centers to the same cluster as the nearest point with higher g. 
    # This assignation is performed in order of decreasing g
    clu_labels = _DPA.initial_assignment(g, N, indices, centers)

    # Topography reconstruction
    #--------------------------
    # Finding saddle points between pair of clusters c and c'.
    # Halo points are also dentified as the points whose density is lower than 
    # the density of the lowest saddle point, manely the set of points 
    # whose assignation is not reliable. The clustering labels for halo point is set to -1.
    Rho_bord, Rho_bord_err, clu_labels, clu_halos, Nclus, centers_m = _DPA.get_borders(N, k_hat, indices, 
                                                                            clu_labels, Nclus, 
                                                                  g, densities, err_densities,
                                                                                   Z, centers)
    topography = []
    for i in range(0, Nclus-1):
        for j in range(i+1, Nclus):
            topography.append([i,j, Rho_bord[i][j], Rho_bord_err[i][j]])

    labels = clu_labels
    halos = clu_halos
    return labels, halos, topography, g, centers_m

   
class DensityPeakAdvanced(ClusterMixin, BaseEstimator): 
    """Class definition for the non-parametric Density Peak clustering.

    The default pipeline makes use of the `PAk` density estimator and of the `TWO-NN` intristic dimension estimator.
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
        :obj:`VALID_METRIC = [precomputed, euclidean,cosine]`. If metric is ``precomputed``, X is assumed to
        be a distance matrix. Alternatively, if metric is a callable function, it is 
        called on each pair of instances (rows) and the resulting value recorded. The 
        callable should take two arrays from X as input and return a value indicating 
        the distance between them. Default is ``euclidean``.

    densities : array [n_samples], default = None
        The logarithm of the density at each point. If provided, the following parameters are ignored:
        ``density_algo``, ``k_max``, ``D_thr``.

    err_densities : array [n_samples], default = None
        The uncertainty in the density estimation, obtained by computing 
        the inverse of the Fisher information matrix.

    k_hat : array [n_samples], default = None
        The optimal number of neighbors for which the condition of constant density holds.

    nn_distances  : array [n_samples, k_max+1]
        Distances to the k_max neighbors of each points.

    nn_indices : array [n_samples, k_max+1]
        Indices of the k_max neighbors of each points.

    affinity : string or callable, default 'precomputed'
        How to construct the affinity matrix.
         - ``nearest_neighbors`` : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - ``rbf`` : construct the affinity matrix using a radial basis function
           (RBF) kernel.
         - ``precomputed`` : interpret ``X`` as a precomputed affinity matrix.
         - ``precomputed_nearest_neighbors`` : interpret ``X`` as a sparse graph
           of precomputed nearest neighbors, and constructs the affinity matrix
           by selecting the ``n_neighbors`` nearest neighbors.
         - one of the kernels supported by
           :func:`~sklearn.metrics.pairwise_kernels`.

    density_algo : string, default = "PAk"
        Define the algorithm to use as density estimator. It mast be one of the options allowed by 
        :obj:`VALID_DENSITY = [PAk, kNN]`. 
        
    k_max : int, default=1000
        This parameter is considered if density_algo is ``PAk`` or ``kNN``, it is ignored otherwise.
        k_max set the maximum number of nearest-neighbors considered by the density estimator.
        If ``density_algo=PAk``, k_max is used by the algorithm in the search for the 
        largest number of neighbors ``k_hat`` for which the condition of constant density 
        holds, within a given level of confidence. 
        If ``density_algo=kNN``, k_max set the number of neighbors to be used by the standard
        k-Nearest Neighbor algorithm.
        If the number of points in the sample N is 
        less than the default value, k_max will be set automatically to the value ``N/2``.

    D_thr : float, default=23.92812698
        This parameter is considered if density_algo is ``PAk``, it is ignored otherwise.
        Set the level of confidence in the PAk density estimator. The default value corresponds to a p-value of 
        :math:`10^{-6}` for a :math:`\chiˆ2` distribution with one degree of freedom.

    dim : int, default = None
        Intrinsic dimensionality of the sample. If dim is provided, the following parameters are ignored:
        ``dim_algo``, ``blockAn``, ``block_ratio``, ``frac``.

    dim_algo : string, or callable, default="twoNN"
        Method for intrinsic dimensionality calculation. If dim_algo is ``auto``, dim is assumed to be
        equal to n_samples. If dim_algo is a string, it must be one of the options allowed by :obj:`VALID_DIM = [auto, twoNN]`. 

    blockAn : bool, default=True
        This parameter is considered if dim_algo is ``twoNN``, it is ignored otherwise.
        If blockAn is True the algorithm perform a block analysis that allows discriminating the relevant dimensions 
        as a function of the block size. This allows to study the stability of the estimation with respect to
        changes in the neighborhood size, which is crucial for ID estimations when the data lie on a 
        manifold perturbed by a high-dimensional noise. 

    block_ratio : int, default=5
        This parameter is considered if dim_algo is ``twoNN``, it is ignored otherwise.
        Set the minimum size of the blocks as `n_samples/block_ratio`. If ``blockAn=False``, ``block_ratio`` is ignored.

    frac : float, default=1
        This parameter is considered if dim_algo is ``twoNN``, it is ignored otherwise.
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

    halos_ : array [Nclus]
        The clustering labels assigned to each point in the data set. Points identified as halos have 
        label equal to zero.

    topography_ : array [Nclus, Nclus]
        Let be Nclus the number of clusters, the topography consists in a Nclus × Nclus symmetric matrix, 
        in which the diagonal entries are the heights of the peaks and the off-diagonal entries are the 
        heights of the saddle points. 

    nn_distances_ : array [n_samples, k_max+1]
        Distances to the k_max neighbors of each points. The point itself is included in the array.

    nn_indices_ : array [n_samples, k_max+1]
        Indices of the k_max neighbors of each points. The point itself is included in the array.
 
    k_hat_ : array [n_samples]
        The optimal number of neighbors for which the condition of constant density holds.

    centers_ :array [Nclus]
        The clustering labels assigned to each point in the data set.

    dim_ : int, 
        Intrinsic dimensionality of the sample. If ``dim`` is not provided, ``dim_`` is set 
        to the number of features in the input file.

    k_max_ : int
        The maximum number of nearest-neighbors considered by the procedure that returns the
        largest number of neighbors ``k_hat`` for which the condition of constant density
        holds, within a given level of confidence. If the number of points in the sample `N` is
        less than the default value, k_max_ will be set automatically to the value ``N/2``.

    densities_ : array [n_samples]
        If not provided by the parameter ``densities``, it is computed by using the `PAk` density estimator. 

    err_densities_ : array [n_samples]
        The uncertainty in the density estimation. If not provided by the parameter ``densities``, it is 
        computed by using the `PAk` density estimator. 

    Example
    -------


    References
    ----------
    
    M. d'Errico, E. Facco, A. Laio, A. Rodriguez, Information Sciences, Volume 560, June 2021, 476-492, https://www.sciencedirect.com/science/article/pii/S0020025521000116?dgcid=author (also available on arXiv: https://arxiv.org/abs/1802.10549)
    """

    def __init__(self, Z=1, metric="euclidean", densities=None, err_densities=None, k_hat=None,
                      nn_distances=None, nn_indices=None, affinity='precomputed',
                      density_algo="PAk", k_max=1000, D_thr=23.92812698, dim=None, dim_algo="twoNN", 
                       blockAn=True, block_ratio=5, frac=1, n_jobs=None):
        self.Z = Z
        self.metric = metric
        self.densities = densities
        self.err_densities = err_densities
        self.k_hat = k_hat
        self.nn_distances = nn_distances
        self.nn_indices = nn_indices
        self.affinity = affinity
        self.density_algo = density_algo
        self.k_max = k_max
        self.D_thr = D_thr
        self.dim = dim
        self.dim_algo = dim_algo
        self.blockAn = blockAn
        self.block_ratio = block_ratio
        self.frac = frac
        self.n_jobs = n_jobs

        if metric not in VALID_METRIC and not callable(metric):
            raise ValueError("invalid metric: '{0}'".format(metric))

        if dim_algo not in VALID_DIM:
            raise ValueError("invalid dim_algo: '{0}'".format(dim_algo))

        if density_algo not in VALID_DENSITY:
            raise ValueError("invalid dim_algo: '{0}'".format(density_algo))

        #if not (self.densities and self.err_densities  and self.k_hat):
        #    # TODO: decide whether to raise a worning instead and automatically run PAk. 
        #    raise ValueError("DPA requires the error estimation and optimal neighborhood along \
        #                      with the densities. If not available, use the default PAk estimator") 

        if self.dim_algo == "twoNN" and self.frac > 1:
            raise ValueError("frac should be between 0 and 1.")

        if self.nn_distances is not None and self.nn_indices is not None:        
            if self.nn_distances.shape[1] != self.nn_indices.shape[1]:
                raise ValueError("check nn_distances and nn_indices. Mismatch in array dimension.")
 
    def fit(self, X, y=None):
        """Fit the DPA clustering on the data.

        Parameters
        ----------
        X : array [n_samples, n_samples] if metric == “precomputed”, or, 
            [n_samples, n_features] otherwise
            The input samples. Similarities / affinities between
            instances if ``affinity='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64, ensure_min_samples=2)
        
        allow_squared = self.affinity in ["precomputed",
                                          "precomputed_nearest_neighbors"]
        if X.shape[0] == X.shape[1] and not allow_squared:
            warnings.warn("The DPA clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        
        self.k_max_ = self.k_max
        self.dim_ = self.dim
        if not self.dim:
            if self.dim_algo == "auto":
                self.dim_ = X.shape[1]
            elif self.dim_algo == "twoNN":
                if self.block_ratio >= X.shape[0]:
                    raise ValueError("block_ratio is larger than the sample size, the minimum size for \
                                      block analysis would be zero. Please set a value lower than "+str(X.shape[0]))
                self.dim_ = twoNearestNeighbors(blockAn=self.blockAn, block_ratio=self.block_ratio, metric=self.metric,
                                               frac=self.frac, n_jobs=self.n_jobs).fit(X).dim_
            else:
                pass

    
            
        # If densities, uncertainties and k_hat are provided as input, compute only the
        # matrix of nearest neighbor: 
        self.densities_ = self.densities
        self.err_densities_ = self.err_densities
        self.k_hat_ = self.k_hat
        if self.densities_ is not None and self.err_densities_ is not None and self.k_hat_ is not None:
            # If the nearest neighbors matrix is precomputed:
            if self.nn_distances is not None and self.nn_indices is not None:
                self.k_max_ = max(self.k_hat_) 
                self.nn_distances_ = self.nn_distances
                self.nn_indices_ = self.nn_indices
            else:
                self.k_max_ = max(self.k_hat_)
                if self.metric == "precomputed":
                    nbrs = NearestNeighbors(n_neighbors=self.k_max_+1, # The point i is counted in its neighborhood 
                                                  algorithm="brute", 
                                                metric=self.metric,
                                                n_jobs=self.n_jobs).fit(X)
                else:
                    nbrs = NearestNeighbors(n_neighbors=self.k_max_+1, # The point i is counted in its neighborhood 
                                                 algorithm="auto", 
                                                metric=self.metric, 
                                                n_jobs=self.n_jobs).fit(X)
                self.nn_distances_, self.nn_indices_ = nbrs.kneighbors(X) 
        elif self.density_algo == "PAk":
            # If the nearest neighbors matrix is precomputed:
            if self.nn_distances is not None and self.nn_indices is not None:
                self.k_max_ = self.nn_distances.shape[1]-1
                PAk = PointAdaptive_kNN(k_max=self.k_max_, D_thr=self.D_thr, metric=self.metric,
                                                   nn_distances=self.nn_distances, nn_indices=self.nn_indices,
                                                   dim_algo=self.dim_algo, blockAn=self.blockAn,
                                                   block_ratio=self.block_ratio,
                                                   frac=self.frac, dim=self.dim_, n_jobs=self.n_jobs).fit(X)
            else:
                PAk = PointAdaptive_kNN(k_max=self.k_max_, D_thr=self.D_thr, metric=self.metric, 
                                                   dim_algo=self.dim_algo, blockAn=self.blockAn, 
                                                   block_ratio=self.block_ratio,
                                                   frac=self.frac, dim=self.dim_, n_jobs=self.n_jobs).fit(X)
            self.nn_distances_ = PAk.distances_
            self.nn_indices_ = PAk.indices_ 
            self.densities_ = PAk.densities_
            self.err_densities_ = PAk.err_densities_
            self.k_hat_ = PAk.k_hat_
            self.k_max_ = max(self.k_hat_)
        else:
            # TODO: implement option for kNN
            pass
        self.labels_, self.halos_, self.topography_, self.g_, self.centers_ = _DensityPeakAdvanced(self.densities_, 
                                                              self.err_densities_, self.k_hat_, 
                                                              self.nn_distances_, self.nn_indices_, self.Z)
                                                              

        self.is_fitted_ = True
        
        return self


    def fit_predict(self, X, y=None):
        """Perform DPA clustering from features or distance matrix,
        and return cluster labels.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X)
        return self.labels_

    def get_computed_params(self):
        params = self.get_params()
        for x in params:
            if x+"_" in self.__dict__:
                params[x] = self.__dict__[x+"_"]
        return params
    def get_histogram(self):
# Generation of SL dendrogram
# Prepare some auxiliary lists
        e1=[]
        e2=[]
        d12=[]
        L=[]
        Li1=[]
        Li2=[]
        Ldis=[]
        Fmax=max(self.densities_)
        Nclus=np.max(self.labels_)+1
# Obtain distances in list format from topography
        for row in self.topography_:
            dis12=Fmax-row[2]
            e1.append(row[0])
            e2.append(row[1])
            d12.append(dis12)

# Obtain the dendrogram in form of links
        nlinks=0
        clnew=Nclus
        for j in range(Nclus-1):
            aa=np.argmin(d12)
            nlinks=nlinks+1
            L.append(clnew+nlinks)
            Li1.append(e1[aa])
            Li2.append(e2[aa])
            Ldis.append(d12[aa])
    #update distance matrix
            t=0
            fe=Li1[nlinks-1]
            fs=Li2[nlinks-1]
            newname=L[nlinks-1]
    # list of untouched clusters
            unt=[]
            for r in d12:
                if ((e1[t]!=fe)&(e1[t]!=fs)):
                    unt.append(e1[t])
                if ((e2[t]!=fe)&(e2[t]!=fs)):
                    unt.append(e2[t])
                t=t+1
            myset = set(unt)
            unt=list(myset)
    # Build a new distance matrix
            e1new=[]
            e2new=[]
            d12new=[]
            for j in unt:
                t=0
                dmin=9.9E99
                for r in d12:
                    if ((e1[t]==j)|(e2[t]==j)):
                        if ((e1[t]==fe)|(e2[t]==fe)|(e1[t]==fs)|(e2[t]==fs)):
                            if (d12[t]<dmin):
                                dmin=d12[t]
                    t=t+1
                e1new.append(j)
                e2new.append(newname)
                d12new.append(dmin)
        
            t=0
            for r in d12:
                if ((unt.count(e1[t]))&(unt.count(e2[t]))):
                    e1new.append(e1[t])
                    e2new.append(e2[t])
                    d12new.append(d12[t])
                t=t+1
        
            e1=e1new
            e2=e2new
            d12=d12new

# Get the order in which the elements should be displayed
        sorted_elements=[]
        sorted_elements.append(L[nlinks-1])

        for jj in range(len(L)):
            j=len(L)-jj-1
            for i in range(len(sorted_elements)):
                if (sorted_elements[i]==L[j]):
                    sorted_elements[i]=Li2[j]
                    sorted_elements.insert(i,Li1[j])
# Get coordinates for the plot
        pop=np.zeros((Nclus),dtype=int)
        for i in range (len(self.labels_)):
            pop[self.labels_[i]]=pop[self.labels_[i]]+1
#print (pop)
        add=0.
        x=[]
        y=[]
        label=[]
        for i in range(len(sorted_elements)):
            label.append(sorted_elements[i])
            j=self.centers_[label[i]]
            y.append(self.densities_[j])
            x.append(add+0.5*np.log(pop[i]))
            add=add+np.log(pop[i])

        xs=x.copy()
        ys=y.copy()
        labels=label.copy()
        zorder=0
        for jj in range(len(L)):
            c1=label.index(Li1[jj])
            c2=label.index(Li2[jj])
            label.append(L[jj])
            x.append((x[c1]+x[c2])/2.)
            ynew=Fmax-Ldis[jj]
            y.append(ynew)
            x1=x[c1]
            y1=y[c1]
            x2=x[c2]
            y2=y[c2]
            zorder=zorder+1
            plt.plot([x1, x1], [y1, ynew], color='k', linestyle='-', linewidth=2,zorder=zorder)
            zorder=zorder+1
            plt.plot([x2, x2], [y2, ynew], color='k', linestyle='-', linewidth=2,zorder=zorder)
            zorder=zorder+1
            plt.plot([x1, x2], [ynew, ynew], color='k', linestyle='-', linewidth=2,zorder=zorder)

        zorder=zorder+1
        viridis = cm.get_cmap('viridis', Nclus)
        #print(viridis.colors)
        plt.scatter (xs,ys,c=labels,s=100,zorder=zorder,cmap="viridis")
        for i in range (Nclus):
            zorder=zorder+1
            cc='k'
            r=viridis.colors[labels[i]][0]
            g=viridis.colors[labels[i]][1]
            b=viridis.colors[labels[i]][2]
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            luma=luma*255
            if (luma<156):
                    cc='w'
            plt.annotate(labels[i],(xs[i],ys[i]),horizontalalignment='center',verticalalignment='center',zorder=zorder,c=cc,weight='bold')
        plt.show()
