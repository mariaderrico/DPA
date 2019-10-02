# file '_PAk.pyx'. 
import numpy as np
cimport numpy as c_np
from math import log, sqrt, exp, lgamma, pi, pow
from scipy.optimize import minimize


def ratio_test(i, N, V1, V_dic, dim, distances, k_max, D_thr, indices):
    # Compute the volume of the dim-sphere with unitary radius
    cdef float Dk, vi, vj
    cdef int k, j 
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
    return k, distances[i][k-1], V_dic

def ML_fun(params, args):
    '''
    The function returns the log-Likelihood expression to be minimized.
   
    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:
   
    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    '''
    #cdef c_np.ndarray[float, ndim=1] g= np.zeros((2),dtype=float)
    cdef float b=params[0]
    cdef float a=params[1]
    cdef int kopt = args[0]
    cdef float gb=kopt
    cdef float ga=(kopt+1)*kopt*0.5
    cdef float L0=b*gb+a*ga
    cdef c_np.ndarray[double, ndim=1] Vi=args[1]
    cdef int k
    cdef float t, tt
    cdef float s
    for k in range(1,kopt):
        jf=float(k)
        t=b+a*jf
        s=exp(t)
        tt=Vi[k-1]*s
        L0=L0-tt
    return -L0

def ML_hess_fun(params, args):
    '''
    The function returns the expressions for the asymptotic variances of the estimated parameters.
   
    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    '''
    cdef c_np.ndarray[double, ndim=1] g = np.zeros(2)
    cdef float b=params[0]
    cdef float a=params[1]
    cdef int kopt = args[0]
    cdef float gb=kopt
    cdef float ga=(kopt+1)*kopt*0.5
    cdef float L0=b*gb+a*ga
    cdef c_np.ndarray[double, ndim=1] Vi=args[1]
    cdef c_np.ndarray[double, ndim=2] Cov2 = np.zeros((2,2))
    cdef c_np.ndarray[double, ndim=2] Covinv2
    cdef int k
    cdef float jf, t, tt, s

    for k in range(1,kopt):
        jf=float(k)
        t=b+a*jf
        s=exp(t)
        tt=Vi[k-1]*s
        L0=L0-tt
        gb=gb-tt
        ga=ga-jf*tt
        Cov2[0][0]=Cov2[0][0]-tt
        Cov2[0][1]=Cov2[0][1]-jf*tt
        Cov2[1][1]=Cov2[1][1]-jf*jf*tt
    Cov2[1][0]=Cov2[0][1]
    Cov2 = Cov2*(-1)
    Covinv2 = np.linalg.inv(Cov2)
    g[0]=sqrt(Covinv2[0][0])
    g[1]=sqrt(Covinv2[1][1])
    return g

def MLmax(rr, kopt, Vi):
    '''
    This function uses the scipy.optimize package to minimize the function returned by ''ML_fun'', and
    the ''ML_hess_fun'' for the analytical calculation of the Hessian for errors estimation.
    It returns the value of the density which minimize the log-Likelihood in Eq. (S1)

    Requirements:

    * **rr**: is the initial value for the density, by using the standard k-NN density estimator
    * **kopt**: is the optimal neighborhood size k as return by the Likelihood Ratio test
    * **Vi**: is the list of the ''kopt'' volumes of the shells defined by two successive nearest neighbors of the current point

    '''
    cdef float a_err
    results = minimize(ML_fun, [rr,0.], method='Nelder-Mead', args=([kopt,Vi],), options={'maxiter':1000})
    err = ML_hess_fun(results.x, [kopt,Vi])
    a_err = err[1]
    rr = results.x[0] #b
    return rr



def get_densities(dim, distances, k_max, D_thr, indices):
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
    cdef c_np.ndarray[double, ndim=1] dc = np.array([])
    cdef c_np.ndarray[double, ndim=1] densities = np.array([])
    cdef c_np.ndarray[double, ndim=1] err_densities = np.array([])
    cdef dict V_dic = {}
    cdef int k, identical
    cdef double dc_i
    cdef c_np.ndarray[double, ndim=1] Vi 
    for i in range(0,N):
        k, dc_i, V_dic = ratio_test(i, N, V1, V_dic, dim, distances, k_max, D_thr, indices)
        k_hat.append(k-1)
        dc = np.append(dc, dc_i)
        densities = np.append(densities, log(k-1)-(log(V1)+dim*log(dc[i]))) 
        err_densities = np.append(err_densities, sqrt((4.*(k-1)+2.)/((k-1)*(k-2))))

        # Apply a correction to the density estimation if no neighbors are at the same distance from point i 
        # Check if neighbors with identical distances from point i
        Vi =  np.array([])
        identical = 0
        Vi = np.append(Vi, V_dic[i][0])
        for k in range(1, len(V_dic[i])):
            Vi = np.append(Vi, V_dic[i][k]-V_dic[i][k-1])
            #if Vi[k]<1.0E-300:
            #    identical = 1 
        #identical = len(np.unique(distances[i])) == len(distances[i])
        # TODO:
        if not identical:
            densities[i] = MLmax(densities[i], k_hat[i], Vi)
        #    densities[i] = NR.nrmax(densities[i], k_hat[i], V_dic[i])
        #else:
        #    pass
    return k_hat, dc, densities, err_densities, V_dic

