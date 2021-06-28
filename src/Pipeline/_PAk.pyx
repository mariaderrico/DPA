# !python
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: profile=False

# file '_PAk.pyx'. 
#
# Author: Maria d'Errico <mariaderr@gmail.com>
#
# Licence: BSD 3 clause

from libc.math cimport log, sqrt, exp, lgamma, pi, pow
from libc.stdint cimport int32_t, int64_t
import sys
import numpy as np
cimport cython

fepsilon = sys.float_info.epsilon
DBL_MIN = sys.float_info.min
DBL_MAX = sys.float_info.max

cdef double get_volume(double V1, double dist, int dim):
    # Check if the distance is not zero and return the corresponding volume
    if dist > 0:
        return V1*exp(dim*(log(dist)))
    else:
        return 0

cdef tuple ratio_test(int i, int N, double V1, dict V_dic, int dim,
                      double[:,:]  distances, int k_max,
                      double D_thr, int64_t[:,:] indices):
    # Compute the volume of the dim-sphere with unitary radius
    cdef float Dk, vi, vj
    cdef int k, j 
    # Volumes are stored in V_dic dictionary to avoid double computations
    if i not in V_dic.keys():
        V_dic[i] = [-1]*k_max
        V_dic[i][0] = get_volume(V1, distances[i][1], dim)
        V_dic[i][1] = get_volume(V1, distances[i][2], dim)
        V_dic[i][2] = get_volume(V1, distances[i][3], dim)
    k = 3 # Minimum number of neighbors required
    Dk = 0
    # Stop when the k+1-th neighbors has a significantly different density from point i 
    while (k<k_max and Dk<=D_thr):
        # Note: the k-th neighbor is at position k in the distances and indices arrays
        if (i in V_dic.keys() and V_dic[i][k-1] != -1):
            vi = V_dic[i][k-1]
        elif i not in V_dic.keys():
            V_dic[i] = [-1]*k_max
            V_dic[i][0] = get_volume(V1, distances[i][1], dim)
            V_dic[i][1] = get_volume(V1, distances[i][2], dim)
            V_dic[i][2] = get_volume(V1, distances[i][3], dim)
            V_dic[i][k-1] = get_volume(V1, distances[i][k], dim)
            vi = V_dic[i][k-1]
        else:
            V_dic[i][k-1] = get_volume(V1, distances[i][k], dim)
            vi = V_dic[i][k-1]
        # Check on the k+1-th neighbor of i
        j = indices[i][k+1]
        if (j in V_dic.keys() and V_dic[j][k-1] != -1):
            vj = V_dic[j][k-1]
        elif j not in V_dic.keys():
            V_dic[j] = [-1]*k_max
            V_dic[j][0] = get_volume(V1, distances[j][1], dim)
            V_dic[j][1] = get_volume(V1, distances[j][2], dim)
            V_dic[j][2] = get_volume(V1, distances[j][3], dim)
            V_dic[j][k-1] = get_volume(V1, distances[j][k], dim)
            vj = V_dic[j][k-1]
        else:
            V_dic[j][k-1] = get_volume(V1, distances[j][k], dim)
            vj = V_dic[j][k-1]
        # In case of identical points vi and/or vj are set to zero
        if (vi>0 and vj>0):
            Dk = -2.*k*(log(vi)+log(vj)-2.*log(vi+vj)+log(4.))
        else:
            pass
        k += 1
    V_dic[i][k-1] = V1*exp(dim*log(distances[i][k]))
    return k, distances[i][k-1], V_dic


cpdef tuple get_densities(int dim, double[:,:] distances,
                          int k_max, double D_thr, int64_t[:,:] indices):
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
    
    """
    cdef float V1 = exp(dim/2.*log(pi)-lgamma((dim+2)/2.))    
    cdef int N = distances.shape[0]
    cdef list k_hat = []
    cdef list dc = []
    cdef list densities = []
    cdef list err_densities = []
    cdef dict V_dic = {}
    cdef int k, identical
    cdef double dc_i
    cdef float rho_min = DBL_MAX
    for i in range(N):
        k, dc_i, V_dic = ratio_test(i, N, V1, V_dic, dim, distances, k_max, D_thr, indices)
        k_hat.append(k-1)
        dc.append(dc_i)
        densities.append(log(k-1)-(log(V1)+dim*log(dc[i]))) 
        err_densities.append(sqrt((4.*(k-1)+2.)/((k-1)*(k-2))))
        # Apply a correction to the density estimation if no neighbors are at the same distance from point i 
        densities[i] = nrmaxl(densities[i], k_hat[i], V_dic[i], k_max)
        if densities[i] < rho_min:
            rho_min = densities[i]
    # Apply shift to have all densities as positive values 
    densities = [x-rho_min+1 for x in densities]
    
    return k_hat, dc, densities, err_densities


cdef double nrmaxl(double rinit, int kopt,
                   list V_dic, int maxk):
    cdef int j, niter, jf
    cdef double a, b, ga, gb, stepmax, func, sigma, sa, sb
    cdef double[:] vi = np.zeros(len(V_dic), dtype=np.double)
    cdef double[:,:] Covinv2 = np.zeros((2,2), dtype=np.double)
    cdef double[:,:] Cov2 = np.zeros((2,2), dtype=np.double)
    kNN = False
    # Initialization of the parameters in the log-Likelihod function
    # b corresponds to -F=log(density)
    b = rinit
    a = 0.
    stepmax = 0.1*abs(b)
    # Compute volumes of the shells enclosed by consecutive neighbors
    vi[0] = V_dic[0]
    for j in range(1,kopt):
        vi[j] = V_dic[j]-V_dic[j-1]
        if vi[j] < 1e-100:
           kNN = True
    if kNN:
        return b
    ga, gb, Cov2 = get_derivatives(a, b, kopt, vi)
    Covinv2 = get_inverse(Cov2)
    func = 100.
    niter = 0
    # NR maximization loop
    while ((func>1e-3) and (niter < 1000)):
        sb = (Covinv2[0,0]*gb+Covinv2[0,1]*ga)
        sa = (Covinv2[1,0]*gb+Covinv2[1,1]*ga)
        niter = niter+1
        sigma = 0.1
        if abs(sigma*sb) > stepmax:
            sigma = abs(stepmax/sb)
        b = b-sigma*sb
        a = a-sigma*sa
        ga, gb, Cov2 = get_derivatives(a, b, kopt, vi)
        Covinv2 = get_inverse(Cov2)
        if (abs(a) <= fepsilon or abs(b) <= fepsilon):
            func = max(gb,ga)
        else:
            func = max(abs(gb/b),abs(ga/a))
    return b


cdef tuple get_derivatives(double a, double b, int kopt, double[:] vi):
    cdef double L0, gb, ga, tt, t, s
    cdef double[:,:] Cov2 = np.zeros((2,2), dtype=np.double)
    cdef int j, jf
    L0 = 0.
    gb = kopt
    ga = (kopt+1)*(kopt)/2.
    Cov2[0,0] = 0.
    Cov2[0,1] = 0.
    Cov2[1,1] = 0.
    for j in range(kopt):
        jf = (j+1)
        t = b+a*jf
        s = exp(t)
        tt = vi[j]*s
        L0 = L0+t-tt
        gb = gb-tt
        ga = ga-jf*tt
        Cov2[0,0] = Cov2[0,0]-tt
        Cov2[0,1] = Cov2[0,1]-jf*tt
        Cov2[1,1] = Cov2[1,1]-jf*jf*tt
    Cov2[1,0] = Cov2[0,1]
    return ga, gb, Cov2

cdef double[:,:] get_inverse(double[:,:] Cov2):
    cdef double detinv
    cdef double[:,:] Covinv2 = np.zeros((2,2), dtype=np.double)
    detinv = 1./(Cov2[0,0]*Cov2[1,1] - Cov2[0,1]*Cov2[1,0])
    Covinv2[0,0] = detinv * Cov2[1,1]
    Covinv2[1,0] = -1.*detinv * Cov2[1,0]
    Covinv2[0,1] = -1.*detinv * Cov2[0,1]
    Covinv2[1,1] = detinv * Cov2[0,0]
    return Covinv2
