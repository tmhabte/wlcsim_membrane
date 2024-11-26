""" 

ALT f bind were there is an unfavorable binding energy when bound to unmarked tails

HP1 binds to H3K9me3, PRC1 binds to H3K27me3, but only 2 proteins max bound per nucleosome

Nuclear/looping scaling behavior at intermediate lenthscales (rubenstein)

Matching chromo/wlcsim (PNAS) protein interaction by defining NUMBER DENISTIES not volume fractions, and multipling v_int interaction parameters by Vol_int interaction volumes

Contains all functions needed to generate spinodal diagrams for a chromosome solution with two reader proteins
USES np.float16 for certain functions to reduce memory usage

Inlcudes:
Average binding state
Strucuture factor (2D and 1D)
Gamma 2
Spinodal

"""
import numpy as np
from scipy import signal
import scipy as sp

# DATA_TYPE = np.float16
# DATA_TYPE = np.float32
DATA_TYPE = np.float64

def def_chrom(n_bind, v_int, e_m, rho_c, rho_s, poly_marks, mu_max, mu_min, del_mu, chrom_type = "test"):
    # fraction of nucleosomes with 0,1,2 marks per protein type, calculated form marks1, marks2: 
    [marks_1, marks_2] = poly_marks # becomes probabilty of a given mark for averaged polymer
    M = len(marks_1)
#     [marks_1.astype(DATA_TYPE), marks_2.astype(DATA_TYPE)] = poly_marks
    f_om = np.array([(np.array(marks_1)==0).sum(),(np.array(marks_1)==1).sum(),(np.array(marks_1)==2).sum(), \
                        (np.array(marks_2)==0).sum(),(np.array(marks_2)==1).sum(),(np.array(marks_2)==2).sum()])/len(marks_1)
    
    if chrom_type == "DNA":
#         l_p = 53 # 53 nm bare DNA
        l_p = 20 # 20 nm chromosomal DNA
        bp_p_b = 45 # base pairs per bond
        nm_p_bp = 0.34 # nanometetrs per base pair
        b = l_p * 2 #kuhn length

        N = (len(marks_1)-1) * bp_p_b * nm_p_bp * (1/b)
        N_m = N/(len(marks_1)-1)
    
    elif chrom_type == "test":
        b = 1
        N_m = 1000
        N = N_m * len(marks_1)
        
    elif chrom_type == "diblock":
        b = 1
        N_m = 1000
        N = N_m * len(marks_1)
        
    r_int = 3 #nm
    Vol_int = (4/3) * np.pi * r_int**3
    
    return [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b]

##################################################### BINDING STATE CALC ##################################################
# from hetero_bind_ave
def eval_tmat(mu_a, mu_b, pa1 = 1, pa2 = 1, ea = 0, eb = 0, j_aa = 0, j_bb = 0, j_ab = 0, f_ref = 0):
    r"""
    eval_tmat - Evaluate the transfer matrix or the nucleosome
    T: considereing all the possible binding state combinations of the nucleosome to the left and the right
        , all the contributions to the binding state partition function
    Parameters
    ----------
    mu : float
        HP1 chemical potential
    nm1 : int
        Number of methylated tails in the left-side nucleosome
    nm2 : int
        Number of methylated tails in the right-side nucleosome
    nu : int
        Indicator for nucleosomes within the interaction length
    j_int : float
        Strength of the HP1 interactions
    
    Returns
    -------
    tmat : 3x3 float array
        Transfer matrix for the nucleosome    
    
    """
    
    v1 = np.array([1, np.sqrt(pa1) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa1) * np.exp(mu_b / 2), np.sqrt(1 - pa1) * np.exp(mu_a / 2), np.sqrt(1 - pa1) * np.exp(mu_b / 2 - eb / 2)])
    v2 = np.array([1, np.sqrt(pa2) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa2) * np.exp(mu_b / 2), np.sqrt(1 - pa2) * np.exp(mu_a / 2), np.sqrt(1 - pa2) * np.exp(mu_b / 2 - eb / 2)])
    tmat = np.outer(v1, v2) * np.exp(f_ref)
    # T: tmat is all possible combinations of (un-normalized) probability of adjacent nucleosomes binding state (boltzmann weightings)
    #   essentially joint probability matrix
    
    # Add the interaction terms
    # T: all the matrix elements with a 0 have at max one protein bound, therefore no interaction
    # T: QUESTION: tmat[1,1]] means left nucleosome and right nucleosome have A bound. why assume interaction between them then?
    # ANSWER: tmat is bbetween two nucleosomes total: left-side and right-side. propogating from left-side to right-side
    tmat[1, 1] *= np.exp(-j_aa)
    tmat[1, 2] *= np.exp(-j_ab)
    tmat[1, 3] *= np.exp(-j_aa)
    tmat[1, 4] *= np.exp(-j_ab)
    tmat[2, 1] *= np.exp(-j_ab)
    tmat[2, 2] *= np.exp(-j_bb)
    tmat[2, 3] *= np.exp(-j_ab)
    tmat[2, 4] *= np.exp(-j_bb)
    tmat[3, 1] *= np.exp(-j_aa)
    tmat[3, 2] *= np.exp(-j_ab)
    tmat[3, 3] *= np.exp(-j_aa)
    tmat[3, 4] *= np.exp(-j_ab)
    tmat[4, 1] *= np.exp(-j_ab)
    tmat[4, 2] *= np.exp(-j_bb)
    tmat[4, 3] *= np.exp(-j_ab)
    tmat[4, 4] *= np.exp(-j_bb)
                    
    return tmat

def eval_tend(mu_a, mu_b, pa = 1, ea = 0, eb = 0, f_ref = 0):
    r"""
    eval_tmat - Evaluate the transfer matrix or the nucleosome
    T: only have one interaction if nucleosome is first or last genomic position
    Parameters
    ----------
    mu : float
        HP1 chemical potential
    nm1 : int
        Number of methylated tails in the left-side nucleosome
    nm2 : int
        Number of methylated tails in the right-side nucleosome
    nu : int
        Indicator for nucleosomes within the interaction length
    j_int : float
        Strength of the HP1 interactions
    
    Returns
    -------
    tmat : 3x3 float array
        Transfer matrix for the nucleosome    
    
    """

    v1 = np.array([1, np.sqrt(pa) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa) * np.exp(mu_b / 2), np.sqrt(1 - pa) * np.exp(mu_a / 2), np.sqrt(1 - pa) * np.exp(mu_b / 2 - eb / 2)])    
    
    tend = v1 * np.exp(f_ref)
                    
    return tend

def eval_dtdmu(mu_a, mu_b, pa1 = 1, pa2 = 1, ea = 0, eb = 0, j_aa = 0, j_bb = 0, j_ab = 0, f_ref = 0):
    r"""
    eval_tmat - Evaluate the transfer matrix or the nucleosome
    
    Parameters
    ----------
    mu : float
        HP1 chemical potential
    nm1 : int
        Number of methylated tails in the left-side nucleosome
    nm2 : int
        Number of methylated tails in the right-side nucleosome
    nu : int
        Indicator for nucleosomes within the interaction length
    j_int : float
        Strength of the HP1 interactions
    
    Returns
    -------
    tmat : 3x3 float array
        Transfer matrix for the nucleosome    
    
    """

    
    v1 = np.array([1, np.sqrt(pa1) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa1) * np.exp(mu_b / 2), np.sqrt(1 - pa1) * np.exp(mu_a / 2), np.sqrt(1 - pa1) * np.exp(mu_b / 2 - eb / 2)])
    # change in transfer matrix as change mu_a, mu_b wrt left nucleosome
    dv1da = np.array([0, 0.5 * np.sqrt(pa1) * np.exp(mu_a / 2 - ea / 2), 0, 0.5 * np.sqrt(1 - pa1) * np.exp(mu_a / 2), 0])
    dv1db = np.array([0, 0, 0.5 * np.sqrt(pa1) * np.exp(mu_b / 2), 0, 0.5 * np.sqrt(1 - pa1) * np.exp(mu_b / 2 - eb / 2)])
    
    v2 = np.array([1, np.sqrt(pa2) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa2) * np.exp(mu_b / 2), np.sqrt(1 - pa2) * np.exp(mu_a / 2), np.sqrt(1 - pa2) * np.exp(mu_b / 2 - eb / 2)])
    dv2da = np.array([0, 0.5 * np.sqrt(pa2) * np.exp(mu_a / 2 - ea / 2), 0, 0.5 * np.sqrt(1 - pa2) * np.exp(mu_a / 2), 0])
    dv2db = np.array([0, 0, 0.5 * np.sqrt(pa2) * np.exp(mu_b / 2), 0, 0.5 * np.sqrt(1 - pa2) * np.exp(mu_b / 2 - eb / 2)])
    
    dtda1 = np.outer(dv1da, v2) * np.exp(f_ref)
    dtdb1 = np.outer(dv1db, v2) * np.exp(f_ref)
    dtda2 = np.outer(v1, dv2da) * np.exp(f_ref)
    dtdb2 = np.outer(v1, dv2db) * np.exp(f_ref)
    
    # Add the interaction terms
    
    dtda1[1, 1] *= np.exp(-j_aa)
    dtda1[1, 2] *= np.exp(-j_ab)
    dtda1[1, 3] *= np.exp(-j_aa)
    dtda1[1, 4] *= np.exp(-j_ab)
    dtda1[2, 1] *= np.exp(-j_ab)
    dtda1[2, 2] *= np.exp(-j_bb)
    dtda1[2, 3] *= np.exp(-j_ab)
    dtda1[2, 4] *= np.exp(-j_bb)
    dtda1[3, 1] *= np.exp(-j_aa)
    dtda1[3, 2] *= np.exp(-j_ab)
    dtda1[3, 3] *= np.exp(-j_aa)
    dtda1[3, 4] *= np.exp(-j_ab)
    dtda1[4, 1] *= np.exp(-j_ab)
    dtda1[4, 2] *= np.exp(-j_bb)
    dtda1[4, 3] *= np.exp(-j_ab)
    dtda1[4, 4] *= np.exp(-j_bb)
    
    dtdb1[1, 1] *= np.exp(-j_aa)
    dtdb1[1, 2] *= np.exp(-j_ab)
    dtdb1[1, 3] *= np.exp(-j_aa)
    dtdb1[1, 4] *= np.exp(-j_ab)
    dtdb1[2, 1] *= np.exp(-j_ab)
    dtdb1[2, 2] *= np.exp(-j_bb)
    dtdb1[2, 3] *= np.exp(-j_ab)
    dtdb1[2, 4] *= np.exp(-j_bb)
    dtdb1[3, 1] *= np.exp(-j_aa)
    dtdb1[3, 2] *= np.exp(-j_ab)
    dtdb1[3, 3] *= np.exp(-j_aa)
    dtdb1[3, 4] *= np.exp(-j_ab)
    dtdb1[4, 1] *= np.exp(-j_ab)
    dtdb1[4, 2] *= np.exp(-j_bb)
    dtdb1[4, 3] *= np.exp(-j_ab)
    dtdb1[4, 4] *= np.exp(-j_bb)
                    
    dtda2[1, 1] *= np.exp(-j_aa)
    dtda2[1, 2] *= np.exp(-j_ab)
    dtda2[1, 3] *= np.exp(-j_aa)
    dtda2[1, 4] *= np.exp(-j_ab)
    dtda2[2, 1] *= np.exp(-j_ab)
    dtda2[2, 2] *= np.exp(-j_bb)
    dtda2[2, 3] *= np.exp(-j_ab)
    dtda2[2, 4] *= np.exp(-j_bb)
    dtda2[3, 1] *= np.exp(-j_aa)
    dtda2[3, 2] *= np.exp(-j_ab)
    dtda2[3, 3] *= np.exp(-j_aa)
    dtda2[3, 4] *= np.exp(-j_ab)
    dtda2[4, 1] *= np.exp(-j_ab)
    dtda2[4, 2] *= np.exp(-j_bb)
    dtda2[4, 3] *= np.exp(-j_ab)
    dtda2[4, 4] *= np.exp(-j_bb)
    
    dtdb2[1, 1] *= np.exp(-j_aa)
    dtdb2[1, 2] *= np.exp(-j_ab)
    dtdb2[1, 3] *= np.exp(-j_aa)
    dtdb2[1, 4] *= np.exp(-j_ab)
    dtdb2[2, 1] *= np.exp(-j_ab)
    dtdb2[2, 2] *= np.exp(-j_bb)
    dtdb2[2, 3] *= np.exp(-j_ab)
    dtdb2[2, 4] *= np.exp(-j_bb)
    dtdb2[3, 1] *= np.exp(-j_aa)
    dtdb2[3, 2] *= np.exp(-j_ab)
    dtdb2[3, 3] *= np.exp(-j_aa)
    dtdb2[3, 4] *= np.exp(-j_ab)
    dtdb2[4, 1] *= np.exp(-j_ab)
    dtdb2[4, 2] *= np.exp(-j_bb)
    dtdb2[4, 3] *= np.exp(-j_ab)
    dtdb2[4, 4] *= np.exp(-j_bb)
        
    return dtda1, dtda2, dtdb1, dtdb2
    
def eval_dtenddmu(mu_a, mu_b, pa = 1, ea = 0, eb = 0, f_ref = 0):
    r"""
    eval_tmat - Evaluate the transfer matrix or the nucleosome
    
    Parameters
    ----------
    mu : float
        HP1 chemical potential
    nm1 : int
        Number of methylated tails in the left-side nucleosome
    nm2 : int
        Number of methylated tails in the right-side nucleosome
    nu : int
        Indicator for nucleosomes within the interaction length
    j_int : float
        Strength of the HP1 interactions
    
    Returns
    -------
    tmat : 3x3 float array
        Transfer matrix for the nucleosome    
    
    """

    
    dv1da = np.array([0, 0.5 * np.sqrt(pa) * np.exp(mu_a / 2 - ea / 2), 0, 0.5 * np.sqrt(1 - pa) * np.exp(mu_a / 2), 0])
    dv1db = np.array([0, 0, 0.5 * np.sqrt(pa) * np.exp(mu_b / 2), 0, 0.5 * np.sqrt(1 - pa) * np.exp(mu_b / 2 - eb / 2)])
    
    dtendda = dv1da * np.exp(f_ref)
    dtenddb = dv1db * np.exp(f_ref)
                    
    return dtendda, dtenddb
    
def eval_phi(pa_vec, mu_a = 0, mu_b = 0, ea = 0, eb = 0, j_aa = 0, j_bb = 0, j_ab = 0, f_ref = 0):
    
    nm = len(pa_vec)
    phiva = np.zeros((nm, 5))
    phivb = np.zeros((nm, 5))
    phia = np.zeros((nm))
    phib = np.zeros((nm))
    
    # Evaluate binding for the first bead
    
    pa2 = pa_vec[0]
    tend = eval_tend(mu_a, mu_b, pa2, ea, eb, f_ref)
    dtendda, dtenddb = eval_dtenddmu(mu_a, mu_b, pa2, ea, eb, f_ref)

    q_vec = tend
    phiva[0, :] = dtendda
    phivb[0, :] = dtenddb
    for j in range(1, nm):
        phiva[j, :] = tend
        phivb[j, :] = tend
    
    # Evaluate binding for the intermediate beads
    
    for i in range(0, nm - 1):

        # update mark probabilty of left and right nucleosome
        pa1 = pa2
        pa2 = pa_vec[i + 1]
        
        tmat = eval_tmat(mu_a, mu_b, pa1, pa2, ea, eb, j_aa, j_bb, j_ab, f_ref)
        dtda1, dtda2, dtdb1, dtdb2 = eval_dtdmu(mu_a, mu_b, pa1, pa2, ea, eb, j_aa, j_bb, j_ab, f_ref)
        
        q_vec = np.matmul(q_vec, tmat)

        for j in range(0, nm):
            if j == i:
                phiva[j, :] = np.matmul(phiva[j, :], tmat) + np.matmul(phiva[i + 1, :], dtda1)
                phivb[j, :] = np.matmul(phivb[j, :], tmat) + np.matmul(phivb[i + 1, :], dtdb1)
            elif j == (i + 1):
                # only condiser neighbor in one direction- whole point of transfer matrix method
                phiva[j, :] = np.matmul(phiva[j, :], dtda2)
                phivb[j, :] = np.matmul(phivb[j, :], dtdb2)
            else:
                phiva[j, :] = np.matmul(phiva[j, :], tmat)
                phivb[j, :] = np.matmul(phivb[j, :], tmat)
    
    # Evaluate binding for the last bead

    pa1 = pa2
    tend = eval_tend(mu_a, mu_b, pa1, ea, eb, f_ref)
    dtendda, dtenddb = eval_dtenddmu(mu_a, mu_b, pa1, ea, eb, f_ref)

    # calculate average binding fractions
    q = np.matmul(q_vec, tend) #part func
    phia[nm - 1] = (np.matmul(q_vec, dtendda) + np.matmul(phiva[nm - 1, :], tend)) / q
    phib[nm - 1] = (np.matmul(q_vec, dtenddb) + np.matmul(phivb[nm - 1, :], tend)) / q
    for j in range(0, nm - 1):
        phia[j] = np.matmul(phiva[j, :], tend) / q
        phib[j] = np.matmul(phivb[j, :], tend) / q
    
    return phia, phib
    

##################################################### BINDING STATE CALC ##################################################

def calc_binding_states(chrom):
    
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    [pa_vec, marks_2] = poly_marks

    mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]

    s_bind_1_soln_arr = np.zeros((len(mu1_array), len(mu2_array), M))
    s_bind_2_soln_arr = np.zeros((len(mu1_array), len(mu2_array), M))
    
    f_ref = np.min(np.array([v_int[0,0], v_int[1,1], v_int[0,1], e_m[0] / 2,  e_m[1] / 2]))

    for i, mu1 in enumerate(mu1_array):
        for j, mu2 in enumerate(mu2_array):
            s_bind_ave_a, s_bind_ave_b = eval_phi(pa_vec, mu1, mu2, e_m[0], e_m[1], v_int[0,0], v_int[1,1], v_int[0,1], f_ref)
            s_bind_1_soln_arr[i,j,:] = s_bind_ave_a
            s_bind_2_soln_arr[i,j,:] = s_bind_ave_b
    
    return s_bind_1_soln_arr, s_bind_2_soln_arr

def calc_sf2_chromo_shlk(chrom, M2s, k_vec = np.logspace(-3, -1, 30)):
    # calculates sf2 using rank 1 monomer correlation tensor 
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    M2_AA, M2_AB, M2_BA, M2_BB, M1_cgam0, M1_cgam1, M2_cc = M2s

    M = np.shape(M2_AA)[0]
    nk = len(k_vec)
    N = M*N_m
    
    S2_AA_arr = np.zeros(nk)
    S2_AB_arr = np.zeros(nk)
    S2_BA_arr = np.zeros(nk)
    S2_BB_arr = np.zeros(nk)
    
    S2_cgam0_arr = np.zeros(nk)
    S2_cgam1_arr = np.zeros(nk)
    S2_cc_arr = np.zeros(nk)


    for i, k in enumerate(k_vec):
        C = np.zeros(M)
        k = np.linalg.norm(k)
        x_m = (1/6) * N_m * b**2 * k**2

        #j1 = j2, s1 > s2
        index = 0#(j1 == j2)
        constant = 1
        debye = (2/(x_m**2)) * (x_m + np.exp(-x_m) - 1) 

    #     C[np.where((index) != 0)] += debye
        C[0] += debye

        #j1 > j2, s1 s2 any
        index = np.arange(0, M, 1)#(j1 > j2) #index = del!
        constant = np.exp(-x_m*index)
        integral = (1/(x_m**2)) * (np.exp(x_m) + np.exp(-x_m) - 2) #for off-diagonal terms

        C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral

        S2_AA_arr[i] = np.sum((1/M**2) * C * M2_AA)
        S2_AB_arr[i] = np.sum((1/M**2) * C * M2_AB)
        S2_BA_arr[i] = np.sum((1/M**2) * C * M2_BA)
        S2_BB_arr[i] = np.sum((1/M**2) * C * M2_BB)
        
        S2_cgam0_arr[i] = np.sum((1/M**2) * C * M1_cgam0)
        S2_cgam1_arr[i] = np.sum((1/M**2) * C * M1_cgam1)
        S2_cc_arr[i] = np.sum((1/M**2) * C * M2_cc)
        
    return S2_AA_arr*N**2, S2_AB_arr*N**2, S2_BA_arr*N**2, S2_BB_arr*N**2, S2_cgam0_arr*N**2, S2_cgam1_arr*N**2, S2_cc_arr*N**2

def inc_gamma(a, x): 
    # got rid of a = 0 condition!
    return sp.special.gamma(a)*sp.special.gammaincc(a, x)

def erf(n):
    return sp.special.erf(n)


def eval_and_reduce_cc(len_marks_1):
    res = np.zeros(len_marks_1)
    res[0] = len_marks_1
    res[1:] = np.arange(2, 2*(len_marks_1), 2)[::-1]
    return res

def eval_and_reduce_cgam(s_bnd_vec):
#     [marks_1, marks_2] = poly_marks
    
    # s_bnd = s_bnd.astype(DATA_TYPE)
    M = len(s_bnd_vec)
    
    sisj_tens = np.zeros(M, dtype = DATA_TYPE)

    # s_bnd_vec = s_bnd[pairs_ind] 
    
#         sisj_tens[0] = np.sum(s_bnd_vec)
#         for i in range(M-1):
#             sisj_tens[-(i+1)] = np.sum(s_bnd_vec[:(i+1)]) + np.sum(s_bnd_vec[-(i+1):])
    
    forward_cumsum = np.cumsum(s_bnd_vec)
    backward_cumsum = np.cumsum(s_bnd_vec[::-1])#[::-1]

    sisj_tens[0] = forward_cumsum[-1]
    sisj_tens[1:] = (forward_cumsum[:-1] + backward_cumsum[:-1])[::-1]
            
          
    return sisj_tens
        
        
def eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B):
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

#     [marks_1, marks_2] = poly_marks

    # s_bnd_A = s_bnd#[pairs_ind+9*gam1_ind]
    # s_bnd_B = s_bnd#[pairs_ind+9*gam2_ind]
    
    # M = len(s_bnd_A)
    sisj_tens = np.zeros(M, dtype = DATA_TYPE)
    
#     ind = np.arange(0,M,1)

#     for i in range(M):
#         if i == 0:
#             sisj_tens[i] = np.sum(s_bnd_A * s_bnd_B)
#         else:
#             sisj_tens[-i] = np.sum(s_bnd_A[:i]*s_bnd_B[-i:]) + np.sum(s_bnd_A[-i:]*s_bnd_B[:i])

#     ind = np.arange(0,M,1)
    
#     sisj_tens[0] = np.sum(s_bnd_A * s_bnd_B)
#     for i in range(M-1):
#         sisj_tens[-(i+1)] = np.sum(s_bnd_A[:(i+1)]*s_bnd_B[-(i+1):]) + np.sum(s_bnd_A[-(i+1):]*s_bnd_B[:(i+1)])

    sisj_tens[0] = np.sum(s_bnd_A * s_bnd_B)
#     conv = np.convolve(s_bnd_A, s_bnd_B[::-1])
    conv = signal.convolve(s_bnd_A, s_bnd_B[::-1])
    sisj_tens[1:] = conv[:M-1][::-1] + conv[:M-1:-1][::-1]
    
    return sisj_tens

import psutil
import os

# import multiprocessing

def calc_sf_mats(chrom, s_bind_1_soln_arr, s_bind_2_soln_arr, k_vec = np.logspace(-3, -1, 30) ):
    # returns rank 3 tensor of mu1, mu2 , k, each value is S2 matrix    
    print(" I USING DATA TYPE " + str(DATA_TYPE))
            
#     start_time = time.time()
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom
    [marks_1, marks_2] = poly_marks
    len_marks_1 = len(marks_1)
    
    # sig_to_frac_dic={}
    # sig_to_int_dic={}
    # i=0
    # for om1 in range(3):
    #     for om2 in range(3):
    #         key = repr([om1,om2])
    #         frac = np.sum((marks_1 == om1)*1 * (marks_2 == om2)*1)/len(marks_1) #fraction of nucleosomes with sigma = [sig_1, sig_2]

    #         sig_to_frac_dic[key] = frac
    #         sig_to_int_dic[key] = i
    #         i+=1

    # pairs = np.column_stack(poly_marks)

    mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    
    sf_mat = np.zeros((len(mu1_array[:]), len(mu2_array[:]), len(k_vec)), dtype = "object")
    
    # pairs = np.column_stack(poly_marks)
    # pairs_ind = np.zeros(len(marks_1), dtype = int) # for each nucleosome (2 mark values), the corresponding index in s_bnd

    # for i,sig_pair in enumerate(pairs):
    #     key = repr(list(sig_pair))
    #     ind = sig_to_int_dic[key]
    # #     print(type(ind))
    #     pairs_ind[i] = ind 
    
    for i, mu1 in enumerate(mu1_array[:]):
        for j, mu2 in enumerate(mu2_array[:]):
            
            mu = [mu1, mu2]
            # f1 = f_gam_soln_arr[0][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]
            # f2 = f_gam_soln_arr[1][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]
            # f_bars = [f1, f2]
            
            # s_bnd = np.zeros(n_bind*9)
            # for ib in range(n_bind*9):
            #     s_bnd[ib] = s_bind_soln_arr[ib][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]
            s_bnd_A = s_bind_1_soln_arr[i,j,:]
            s_bnd_B = s_bind_2_soln_arr[i,j,:]
            
            cc_red = eval_and_reduce_cc(len_marks_1)

            s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)

            s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)

            sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
            
            sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
        
            sisj_BA_red = sisj_AB_red
            
            sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)
            
            M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]

            
            for ik, k in enumerate(k_vec):
                # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = phi_c * np.array(calc_sf2_chromo_shlk(chrom, M2s, [k]))
                # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = rho_c * np.array(calc_sf2_nuclear(chrom, M2s, [k]))
                g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = rho_c * np.array(calc_sf2_chromo_shlk(chrom, M2s, [k]))
                
                ss = rho_s#1-phi_c
                S2_mat = 1/N**2 * np.array([[cc[0], 0, cg1[0], cg2[0]],\
                                [0, ss*N**2, 0, 0], \
                                [cg1[0], 0, g1g1[0], g1g2[0]],\
                                [cg2[0], 0, g2g1[0], g2g2[0]]])
                sf_mat[i][j][ik] = S2_mat
    return sf_mat
