# file '_DPA.pyx'. 
import numpy as np
cimport numpy as c_np

def get_centers(N, indices, k_hat, g):
    cdef int i, j, c
    cdef list centers = []
    # Criterion 1 from Heuristic 1
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
    return centers

def initial_assignment(g, N, indices, centers):
    cdef long c, i, el, k
    cdef c_np.ndarray[long, ndim=1] ig_sort = np.argsort([-x for x in g])

    # Assign points to clusters
    #--------------------------
    # Assign all the points that are not centers to the same cluster as the nearest point with higher g. 
    # This assignation is performed in order of decreasing g
    cdef c_np.ndarray[long, ndim=1] clu_labels = np.zeros(N,dtype=int)-1
    for c in centers:
        clu_labels[c] = centers.index(c)
    for i in range(0,N):
        el = ig_sort[i]
        k = 0
        while (clu_labels[el]==-1):
            k=k+1
            clu_labels[el] = clu_labels[indices[el][k]] # the point with higher g is already assigned by construction
    return clu_labels


def get_borders( N, k_hat, indices, clu_labels, Nclus, g, densities, err_densities):
    cdef dict border_dict = {}
    cdef dict g_saddle = {}
    cdef int i, k, j, c, cp, m_c, M_c

    for i in range(0,N):
        for k in range(0,k_hat[i]):
            j = indices[i][k+1]
            if clu_labels[j]!=clu_labels[i]:
                if (i, clu_labels[i]) not in border_dict.keys():
                    border_dict[(i, clu_labels[i])] = [-1]*Nclus
                    border_dict[(i, clu_labels[i])][clu_labels[j]] = j
                    break
                elif border_dict[(i, clu_labels[i])][clu_labels[j]]==-1:
                    border_dict[(i, clu_labels[i])][clu_labels[j]] = j
                    break
                else:
                    break

    # Criterion 2 from Heuristic 2:
    # check if i is the closest point to j among those belonging to c.
    for i,c in border_dict.keys():
        for cp in range(Nclus):
            j = border_dict[(i,c)][cp]
            if j!=-1:
                if (j,cp) in border_dict.keys() and border_dict[(j,cp)][c] == i:
                    m_c = min(c,cp)
                    M_c = max(c,cp)
                    if (m_c, M_c) not in g_saddle.keys() or g[i] > g_saddle[(m_c,M_c)][1]:
                        g_saddle[(m_c,M_c)] = (i, g[i])

    cdef c_np.ndarray[double, ndim=2] Rho_bord = np.zeros((Nclus,Nclus),dtype=float)
    cdef c_np.ndarray[double, ndim=2] Rho_bord_err = np.zeros((Nclus,Nclus),dtype=float)
    for c,cp in g_saddle.keys():
        i = g_saddle[(c,cp)][0]
        Rho_bord[c][cp] = densities[i]
        Rho_bord[cp][c] = densities[i]
        Rho_bord_err[c][cp] = err_densities[i]
        Rho_bord_err[cp][c] = err_densities[i]
    for c in range(0,Nclus):
        Rho_bord[c][c] = -1
        Rho_bord_err[c][c] = 0
    return Rho_bord, Rho_bord_err
