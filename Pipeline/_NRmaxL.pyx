import cython
import sys
from libc.math cimport exp

fepsilon = sys.float_info.epsilon


@cython.boundscheck(False)
@cython.cdivision(True)
def nrmaxl(float rinit,
            int kopt,
            float [:] V_dic,
            int maxk):

    cdef int j,niter
    cdef float a,b,stepmax,jf,t,s,tt,func,sigma,sa,sb,detinv
    cdef float [:] vi = [0 for i in range(len(V_dic))]

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
        if vi[j] < 1e-100 :
           kNN = True
    if kNN:
        return b
    # Compute the first and second derivatives, (gb, ga) and Cov2 respectively
    ga, gb, Cov2 = get_derivatives(a,b,kopt,vi)
    Covinv2 = get_inverse(Cov2)
    func=100.
    niter=0
    # NR maximization loop
    while ( (func>1e-6) and (niter < 1000) ):
        sb=(Covinv2[0,0]*gb+Covinv2[0,1]*ga)
        sa=(Covinv2[1,0]*gb+Covinv2[1,1]*ga)
        niter=niter+1
        sigma=0.1
        if (abs(sigma*sb) > stepmax) :
            sigma=abs(stepmax/sb)
        b=b-sigma*sb
        a=a-sigma*sa
        ga, gb, Cov2 = get_derivatives(a,b,kopt,vi)
        Covinv2 = get_inverse(Cov2)
        if ((abs(a) <= fepsilon ) or (abs(b) <= fepsilon )):
            func=max(gb,ga)
        else:
            func=max(abs(gb/b),abs(ga/a))
    return b

def get_derivatives(float a, float b, int kopt, float [:] vi):
    cdef float L0, gb, ga
    cdef float [:, :] Cov2 = [[0]*2 for i in range(2)]
    L0=0.
    gb=float(kopt)
    ga=float(kopt+1)*float(kopt)/2.
    Cov2[0,0]=0.
    Cov2[0,1]=0.
    Cov2[1,1]=0.
    for j in range(kopt):
        jf=float(j+1)
        t=b+a*jf
        s=exp(t)
        tt=vi[j]*s
        L0=L0+t-tt
        gb=gb-tt
        ga=ga-jf*tt
        Cov2[0,0]=Cov2[0,0]-tt
        Cov2[0,1]=Cov2[0,1]-jf*tt
        Cov2[1,1]=Cov2[1,1]-jf*jf*tt
    Cov2[1,0]=Cov2[0,1]
    return ga,gb,Cov2

def get_inverse(float [:,:] Cov2):
    cdef float detinv
    cdef float [:, :] Covinv2 = [[0]*2 for i in range(2)]

    detinv = 1./(Cov2[0,0]*Cov2[1,1] - Cov2[0,1]*Cov2[1,0])
    Covinv2[0,0] = detinv * Cov2[1,1]
    Covinv2[1,0] = -1.*detinv * Cov2[1,0]
    Covinv2[0,1] = -1.*detinv * Cov2[0,1]
    Covinv2[1,1] = detinv * Cov2[0,0]
    return Covinv2
