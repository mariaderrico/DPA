# file '_PAk.pyx'. 
import numpy as np
cimport numpy as c_np
from math import log, sqrt, exp, lgamma, pi, pow

def ratio_test(i, N, V1, V_dic, dim, distances, k_max, D_thr, indices):
    # Compute the volume of the dim-sphere with unitary radius
    cdef double Dk, vi, vj
    cdef long k, j 
    k = 3 # Minimum number of neighbors required
    Dk = 0
    # Stop when the k+1-th neighbors has a significantly different density from point i 
    while (k<k_max and Dk<=D_thr):
        # Note: the k-th neighbor is at position k in the distances and indices arrays
        # Volumes are stored in V_dic dictionary to avoid double computations 
        if i in V_dic.keys() and V_dic[i][k-1] != -1:
            vi = V_dic[i][k-1]
        elif i not in V_dic.keys():
            V_dic[i] = [-1]*k_max
            V_dic[i][k-1] = V1*exp(dim*log(distances[i][k]))
            vi = V_dic[i][k-1]
        else:
            V_dic[i][k-1] = V1*exp(dim*log(distances[i][k]))
            vi = V_dic[i][k-1]
        # Check on the k+1-th neighbor of i
        j = indices[i][k+1]
        if j in V_dic.keys() and V_dic[j][k-1] != -1:
            vj = V_dic[j][k-1]
        elif j not in V_dic.keys():
            V_dic[j] = [-1]*k_max
            V_dic[j][k-1] = V1*exp(dim*log(distances[j][k]))
            vj = V_dic[j][k-1]
        else:
            V_dic[j][k-1] = V1*exp(dim*log(distances[j][k]))
            vj = V_dic[j][k-1]

        Dk = -2.*k*(log(vi)+log(vj)-2.*log(vi+vj)+log(4.))
        k += 1
    return k, distances[i][k-1]


def get_densities(dim, distances, k_max, D_thr, indices):
    cdef float V1 = exp(dim/2.*log(pi)-lgamma((dim+2)/2.))    
    cdef int N = distances.shape[0]
    cdef list k_hat = []
    cdef list dc = []
    cdef list densities = []
    cdef list err_densities = []
    cdef dict V_dic = {}
    for i in range(0,N):
        k, dc_i = ratio_test(i, N, V1, V_dic, dim, distances, k_max, D_thr, indices)
        k_hat.append(k-1)
        dc.append(dc_i)
        densities.append(log(k-1)-(log(V1)+dim*log(dc[i]))) 
        err_densities.append(sqrt((4.*(k-1)+2.)/((k-1)*(k-2))))

        # Apply a correction to the density estimation if no neighbors are at the same distance from point i 
        # Check if neighbors with identical distances from point i
        identical = np.unique(distances[i]) == len(distances[i])
        # TODO:
        #if not identical:
        #    densities[i] = NR.nrmax(densities[i], k_hat[i], V_dic[i])
        #else:
        #    pass
    return k_hat, dc, densities, err_densities, V_dic

def _PointAdaptive_kNN(distances, indices, k_max=1000, D_thr=23.92812698, dim=None):
    """Main function implementing the Pointwise Adaptive k-NN density estimator.

    Parameters
    ----------
    distances: array [n_samples, k_max+1]
        Distances to the k_max neighbors of each points. The point is included. 
  
    indices : array [n_samples, k_max+1]
        Indices of the k_max neighbors of each points. The point is included.

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
    # TODO 

    """
    # Compute the volume of the dim-sphere with unitary radius
    cdef float V1 = exp(dim/2.*log(pi)-lgamma((dim+2)/2.))    
    cdef int N = distances.shape[0]
    #cdef c_np.ndarray[long, ndim=1] k_hat #= []
    #cdef c_np.ndarray[double, ndim=1] dc #= []
    #cdef c_np.ndarray[double, ndim=1] densities #= []
    #cdef c_np.ndarray[double, ndim=1] err_densities #= []
    cdef list k_hat = []
    cdef list dc = []
    cdef list densities = []
    cdef list err_densities = []
    cdef dict V_dic = {}
    cdef double Dk, vi, vj
    cdef long i, k, j 
    for i in range(0,N):
        k = 3 # Minimum number of neighbors required
        Dk = 0
        # Stop when the k+1-th neighbors has a significantly different density from point i 
        while (k<k_max and Dk<=D_thr):
            # Note: the k-th neighbor is at position k in the distances and indices arrays
            # Volumes are stored in V_dic dictionary to avoid double computations 
            if i in V_dic.keys() and V_dic[i][k-1] != -1:
                vi = V_dic[i][k-1]
            elif i not in V_dic.keys():
                V_dic[i] = [-1]*k_max
                V_dic[i][k-1] = V1*exp(dim*log(distances[i][k]))
                vi = V_dic[i][k-1]
            else:
                V_dic[i][k-1] = V1*exp(dim*log(distances[i][k]))
                vi = V_dic[i][k-1]
            # Check on the k+1-th neighbor of i
            j = indices[i][k+1]
            if j in V_dic.keys() and V_dic[j][k-1] != -1:
                vj = V_dic[j][k-1]
            elif j not in V_dic.keys():
                V_dic[j] = [-1]*k_max
                V_dic[j][k-1] = V1*exp(dim*log(distances[j][k]))
                vj = V_dic[j][k-1]
            else:
                V_dic[j][k-1] = V1*exp(dim*log(distances[j][k]))
                vj = V_dic[j][k-1]

            Dk = -2.*k*(log(vi)+log(vj)-2.*log(vi+vj)+log(4.))
            k += 1
        k_hat.append(k-1)
        dc.append(distances[i][k-1]) #(k-1)-th neighbor is at position k-1
        densities.append(log(k-1)-(log(V1)+dim*log(dc[i]))) 
        err_densities.append(sqrt((4.*(k-1)+2.)/((k-1)*(k-2))))

        # Apply a correction to the density estimation if no neighbors are at the same distance from point i 
        # Check if neighbors with identical distances from point i
        identical = np.unique(distances[i]) == len(distances[i])
        # TODO:
        #if not identical:
        #    densities[i] = NR.nrmax(densities[i], k_hat[i], V_dic[i])
        #else:
        #    pass
    # Apply shift to have all densities as positive values 
    cdef double m = min(densities)
    densities = [x-m+1 for x in densities]

    return densities, err_densities, k_hat, dc

