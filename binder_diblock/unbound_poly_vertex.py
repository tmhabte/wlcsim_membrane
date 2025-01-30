""" 
1/29/2025
PABUS - Polymer, A, B, unbound, solvent theory 
Ensured to generate gamma fucntions, THEN apply dimensionality reduction using fluctuation constraints!

1/10/2025
CLEANED UP STRUCUTER FACTOR prefactors
each sf has rho_p / M prefactor, and at solvent sf position = alpha = (\rho_s M) / \rho_p = [-M / (v_s * \rho_P)] * [ 1 - (v_m \rho_p)] 
this form of alpha is due to applying incompressibilty condition

this file combines binder_diblock gamma234 and binder_diblock_vertex_competitive files

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

from itertools import permutations as perms
from itertools import product


# DATA_TYPE = np.float16
# DATA_TYPE = np.float32
DATA_TYPE = np.float64

def def_chrom(n_bind, v_int, e_m, rho_c, rho_s, poly_marks, mu_max, mu_min, del_mu, v_s, v_m, chrom_type = "test"):
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
        
    r_int = 3#1#3 #nm
    Vol_int = (4/3) * np.pi * r_int**3

    # v_m = N_m * 1 # cross_secional area = 1
    alpha = (M / (rho_c*v_s)) * (1 - (v_m * rho_c))  # alpha = (rho_s * M) / rho_p, then apply incompressibilty
    # return [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b]
    return [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b]

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

    # M2_AA, M2_AB, M2_BA, M2_BB, M1_cgam0, M1_cgam1, M2_cc = M2s
    M2_AA, M2_AB, M2_BA, M2_BB, M1_cgam0, M1_cgam1, M2_cc, M1_cu, M2_UA, M2_UB, M2_UU = M2s

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

    S2_cu_arr = np.zeros(nk)
    S2_UA_arr = np.zeros(nk)
    S2_UB_arr = np.zeros(nk)
    S2_UU_arr = np.zeros(nk)


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

        S2_cu_arr[i] = np.sum((1/M**2) * C * M1_cu)
        S2_UA_arr[i] = np.sum((1/M**2) * C * M2_UA)
        S2_UB_arr[i] = np.sum((1/M**2) * C * M2_UB)
        S2_UU_arr[i] = np.sum((1/M**2) * C * M2_UU)

    return S2_AA_arr*N**2, S2_AB_arr*N**2, S2_BA_arr*N**2, S2_BB_arr*N**2, S2_cgam0_arr*N**2, S2_cgam1_arr*N**2, S2_cc_arr*N**2, S2_cu_arr*N**2, S2_UA_arr*N**2, S2_UB_arr*N**2, S2_UU_arr*N**2
    # return S2_AA_arr, S2_AB_arr, S2_BA_arr, S2_BB_arr, S2_cgam0_arr, S2_cgam1_arr, S2_cc_arr

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
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
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
    
    sf2_mat = np.zeros((len(mu1_array[:]), len(mu2_array[:]), len(k_vec)), dtype = "object")

    # sf3_mat = np.zeros((len(mu1_array[:]), len(mu2_array[:]), len(k_vec)), dtype = "object")
    # sf4_mat = np.zeros((len(mu1_array[:]), len(mu2_array[:]), len(k_vec)), dtype = "object")

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
            s_bnd_U = 1 - s_bnd_A - s_bnd_B

            cc_red = eval_and_reduce_cc(len_marks_1)

            s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)

            s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)

            sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
            
            sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
        
            sisj_BA_red = sisj_AB_red
            
            sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)
            
            s_cu_red = eval_and_reduce_cgam(s_bnd_U)

            sisj_UA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_A)

            sisj_UB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_B)

            sisj_UU_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_U)

            M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red, s_cu_red, sisj_UA_red, sisj_UB_red, sisj_UU_red]

            # M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)
            # M4 = calc_mon_mat_4(s_bnd_A, s_bnd_B)
            
            for ik, k in enumerate(k_vec):
                # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = phi_c * np.array(calc_sf2_chromo_shlk(chrom, M2s, [k]))
                # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = rho_c * np.array(calc_sf2_nuclear(chrom, M2s, [k]))
                g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc, cu, ug1, ug2, uu = np.array(calc_sf2_chromo_shlk(chrom, M2s, [k]))
                
                # ss = alpha#1-phi_c
                # S2_mat = (rho_c / M) *  np.array([[cc[0], 0, cg1[0], cg2[0]],\
                #                 [0, ss, 0, 0], \
                #                 [cg1[0], 0, g1g1[0], g1g2[0]],\
                #                 [cg2[0], 0, g2g1[0], g2g2[0]]])
                
                # ss = alpha#1-phi_c
                # S2_mat = (rho_c / M) *  np.array([[cc[0], 0, cg1[0], cg2[0], cu[0]],\
                #                 [0, ss, 0, 0, 0], \
                #                 [cg1[0], 0, g1g1[0], g1g2[0], ug1[0]],\
                #                 [cg2[0], 0, g2g1[0], g2g2[0], ug2[0]],\
                #                 [cu[0], 0, ug1[0], ug2[0], uu[0]]])
                # sf2_mat[i][j][ik] = S2_mat

                ss = alpha#1-phi_c
                S2_mat = (rho_c / M) *  np.array([[cc[0], cg1[0], cg2[0], cu[0], 0],\
                                    [cg1[0], g1g1[0], g1g2[0], ug1[0], 0],\
                                    [cg2[0], g2g1[0], g2g2[0], ug2[0], 0],\
                                    [cu[0], ug1[0], ug2[0], uu[0], 0],\
                                    [0, 0, 0, 0, ss]])
                sf2_mat[i][j][ik] = S2_mat

                # NOT necessary to do- this code is to get the kstar/ spinodal!
                # s4 = calc_sf4(chrom, M4, [K1], [K2], [K3]) 
                # s31 = calc_sf3(chrom, M3, [K1], [K2])
    return sf2_mat































######################################################### S3,4 ######################################################################################
######################################################### S3,4 ######################################################################################
######################################################### S3,4 ######################################################################################
######################################################### S3,4 ######################################################################################
######################################################### S3,4 ######################################################################################
######################################################### S3,4 ######################################################################################
######################################################### S3,4 ######################################################################################




























############################################


def calc_single_monomer_matrix_3(s_bind_A, s_bind_B, alphas):
    # calculates the alph1 aph2 alph3 monomer identity cross correlation matrix
    # assumes all polymers consist of monomers with the same length N_m

    # alphas is the seqence that identifies the correlation. 0 indicates polymer, 1, indicates protein 1 ,etc
    s_bind_U = 1 - s_bind_A - s_bind_B
    a1, a2, a3 = alphas
    poly = np.ones(len(s_bind_A))
    sig_arr = [poly, s_bind_A, s_bind_B, s_bind_U]
    return np.einsum('i,j,k',sig_arr[a1],sig_arr[a2],sig_arr[a3])

def calc_single_monomer_matrix_4(s_bind_A, s_bind_B, alphas):
    # calculates the alph1 aph2 alph3 monomer identity cross correlation matrix
    # assumes all polymers consist of monomers with the same length N_m

    # alphas is the seqence that identifies the correlation. 0 indicates polymer, 1, indicates protein 1 ,etc
    s_bind_U = 1 - s_bind_A - s_bind_B

    a1, a2, a3, a4 = alphas
    poly = np.ones(len(s_bind_A))
    sig_arr = [poly, s_bind_A, s_bind_B, s_bind_U]
    return np.einsum('i,j,k,l',sig_arr[a1],sig_arr[a2],sig_arr[a3], sig_arr[a4])

def calc_mon_mat_3(s_bind_A, s_bind_B):
    nm = len(s_bind_A)
    sig_inds = [0,1,2,3] # polymer, gama1, gamma2, U
    M3_arr = np.zeros((len(sig_inds), len(sig_inds), len(sig_inds)), dtype= "object")
    for a1, a2, a3 in product(sig_inds, repeat=3):
        # print([a1, a2, a3])
        M3_arr[a1][a2][a3] = calc_single_monomer_matrix_3(s_bind_A, s_bind_B, [a1, a2, a3])
    return M3_arr

def calc_mon_mat_4(s_bind_A, s_bind_B):
    nm = len(s_bind_A)
    sig_inds = [0,1,2,3] # polymer, gama1, gamma2, U
    M4_arr = np.zeros((len(sig_inds), len(sig_inds), len(sig_inds), len(sig_inds)), dtype= "object")
    for a1, a2, a3, a4 in product(sig_inds, repeat=4):
        M4_arr[a1][a2][a3][a4] = calc_single_monomer_matrix_4(s_bind_A, s_bind_B, [a1, a2, a3, a4])
    return M4_arr


# def s3wlc_zeroq3(chrom, K1):
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom

    # # k1, k2, k3 = Ks
    # s3 = np.zeros((3,3,3),dtype=type(1+1j))

    # FA, FB = [0.5,0.5]#f_om
    # print("ASSUMING FA = 0.5 in s3 0q!")
    # # s2 = s2wlc(pset, N, FA, norm(k1))

    # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = np.array(calc_sf2_chromo_shlk(chrom, M2s, [K1]))
    # ss = alpha#1-phi_c
    # S2_mat = (rho_c / M) * np.array([[cc[0], 0, cg1[0], cg2[0]],\
    #                 [0, ss, 0, 0], \
    #                 [cg1[0], 0, g1g1[0], g1g2[0]],\
    #                 [cg2[0], 0, g2g1[0], g2g2[0]]])
    # k_val = [1, FA, FB]
    # for i in [0,1,2]:
    #     for j in [0,1,2]:
    #         for k in[0,1,2]:
    #             s3[i][j][k] = k_val[k] * S2_mat[i][j]
    
    # return s3

# ADAPTED FROM gaus_vertex_pd_mix.py
def calc_sf3(chrom, M3_arr, k_vec, k_vec_2, plotting = False):
    # for a gaussian chain of M monomers, each of length N_m
    # N = mix.N
    # M = mix.M_ave
    # M_max = mix.M_max
    # N_m = mix.N_m
    # b = mix.b
    # M3_AAA, M3_AAB, M3_ABA, M3_BAA, M3_ABB, M3_BAB,  M3_BBA, M3_BBB = mix.M3s
    # nk = len(k_vec)
    
    # if np.linalg.norm(k_vec[0] + k_vec_2[0]) < 1e-5:
    #     # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  rho_p * ((N_m**2 * M)/n_p) * np.array(s2wlc_zeroq(chrom))
    #     return s3wlc_zeroq3(chrom, k_vec)

        # return sf2_inv_zeroq(chrom, rho_p)
        
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom


    # if np.linalg.norm(k_vec[0] + k_vec_2[0]) < 1e-5:
    #     return s3wlc_zeroq3(chrom, k_vec, 

    # M = np.shape(M2_AA)[0]
    nk = len(k_vec)
    assert nk == 1

    N = M*N_m

    sig_inds = [0,1,2,3,4] #polymer, prot1, prot2, U, S
    S3_arr = np.zeros((len(sig_inds),len(sig_inds),len(sig_inds))) #polymer, prot1, prot2

    grid = np.indices((M,M,M))
    j1 = grid[0]
    j2 = grid[1]
    j3 = grid[2]
    
    # grid = np.indices((M_max,M_max,M_max))
    # j1 = grid[0]
    # j2 = grid[1]
    # j3 = grid[2]

    # S3_AAA_arr =  np.zeros(nk)
    # S3_BAA_arr = np.zeros(nk)
    # S3_BBA_arr = np.zeros(nk)
    # S3_BBB_arr = np.zeros(nk)
    
    # S3_ABA_arr = np.zeros(nk)
    # S3_BAB_arr = np.zeros(nk)
    
    # S3_AAB_arr = np.zeros(nk)
    # S3_ABB_arr = np.zeros(nk) 
    
    for i, k_1 in enumerate(k_vec):
        k_2 = k_vec_2[i]
        k_12 = k_1 + k_2

        # CASE 1; kA = k1 + k2, kB = k_1; S3 > S2 > S1 and S1 > S2 > S3
        case1 = [[k_12, k_1], [j3, j2, j1]]
        case1_deg = [[k_1, k_12], [j1, j2, j3]]

        # CASE 2; kA = k2, kB = k1 + k2; S2 > S1 > S3 and S3 > S1 > S2
        case2 = [[k_2, k_12], [j2, j1, j3]]
        case2_deg = [[k_12, k_2], [j3, j1, j2]]
        
        # CASE 3; kA = k2, kB = -k1; S2 > S3 > S1 and S1 > S3 > S2
        case3 = [[-k_2, k_1], [j2, j3, j1]] # SWITCHED negatives from -k_1
        case3_deg = [[k_1, -k_2], [j1, j3, j2]] # SWITCHED negatives from -k_1
        
        case_arr = [case1, case2, case3, case1_deg, case2_deg, case3_deg]
        # need to consider degenerate cases. flipping each element in array, then appending to original case array
        # case_arr = np.vstack((case_arr, [[np.flipud(el) for el in cse] for cse in case_arr]))
        
#        for each case and sub case, add to a matrix C(j1, j2, j3) which contains the contribution to the overall S3
#        then sum over all indices. Need to keep track of js so that aproiate multiplications with cross corr matrix M3        
        C = np.zeros((M,M,M))

        for cse in case_arr:
            kA, kB = cse[0]
            ordered_js = cse[1]
            
            xm_A = (1/6) * N_m * b**2 * np.linalg.norm(kA)**2
            xm_B = (1/6) * N_m * b**2 * np.linalg.norm(kB)**2
            
            C = calc_case_s3(C, xm_A, xm_B, ordered_js)

        for a1, a2, a3 in product(sig_inds, repeat=3):
            # print([a1, a2, a3])
            # M3_arr[a1][a2][a3] = calc_single_monomer_matrix_3(s_bind_A, s_bind_B, [a1, a2, a3])
            if (a1 == a2 == a3 == 4): #at S^{(3)}_{SSS}
                S3_arr[a1][a2][a3] += alpha
            elif (a1 == 4 or a2 == 4 or a3 == 4): #at S^{(3)}_{Sxx}
                S3_arr [a1][a2][a3] += 0
            else:
                S3_arr[a1][a2][a3] += np.sum((1/M**3) * M3_arr[a1][a2][a3] * C)*(N**3)
        
        # S3_AAA_arr[i] += np.sum((1/M**3) * M3_AAA * C)*(N**3)
        # S3_BAA_arr[i] += np.sum((1/M**3) * M3_BAA * C)*(N**3)
        # S3_BBA_arr[i] += np.sum((1/M**3) * M3_BBA * C)*(N**3)
        # S3_BBB_arr[i] += np.sum((1/M**3) * M3_BBB * C)*(N**3)
        
        # S3_ABA_arr[i] += np.sum((1/M**3) * M3_ABA * C)*(N**3)
        # S3_BAB_arr[i] += np.sum((1/M**3) * M3_BAB * C)*(N**3)
        
        # S3_AAB_arr[i] += np.sum((1/M**3) * M3_AAB * C)*(N**3)
        # S3_ABB_arr[i] += np.sum((1/M**3) * M3_ABB * C)*(N**3)
        
    # s3 = np.zeros((2,2,2)) 
    # s3[0][0][0] = S3_AAA_arr[0]
    # s3[1][0][0] = S3_BAA_arr[0]
    # s3[0][1][0] = S3_ABA_arr[0]
    # s3[0][0][1] = S3_AAB_arr[0]
    # s3[0][1][1] = S3_ABB_arr[0]
    # s3[1][0][1] = S3_BAB_arr[0]
    # s3[1][1][0] = S3_BBA_arr[0]
    # s3[1][1][1] = S3_BBB_arr[0]
    
    # if plotting: # matrix only contains single value, for calculating gamma functions
    #     return S3_AAA_arr, S3_AAB_arr, S3_ABA_arr, S3_BAA_arr, S3_ABB_arr, S3_BAB_arr,  S3_BBA_arr, S3_BBB_arr
    
    return S3_arr

def calc_case_s3(C, xm_A, xm_B, ordered_js):

    jmax, jmid, jmin = ordered_js
    
    cylindrical = False
    epsilon = 0.0000001
    if xm_A + epsilon > xm_B and xm_A - epsilon < xm_B:
        cylindrical = True
    
    xm_A_eq_0 = False
    if xm_A < 1e-5:
        xm_A_eq_0 = True
        
    xm_B_eq_0 = False
    if xm_B < 1e-5:
        xm_B_eq_0 = True

    #for each sub case, looking at the degenerate case where 1 and 2 are switched
    constant = np.exp(-xm_A*(jmax - jmid)) * np.exp(-xm_B*(jmid - jmin)) 

    # sub case 1; jmax > jmid > jmin, {s1, s2, s3} any 
    index = (jmax > jmid) * (jmid > jmin)
    
    if cylindrical == True:
        integral = (1 / xm_A**2) * 2 * (-1 + np.cosh(xm_A))
    elif xm_B_eq_0:
        integral = (2*(-1+np.cosh(xm_A)))/ (xm_A**2)
    elif xm_A_eq_0:
        integral = (2*(-1+np.cosh(xm_B)))/ (xm_B**2)
    else:
        integral = (-2 / (xm_A * (xm_A - xm_B) * xm_B)) \
        * (-np.sinh(xm_A) + np.sinh(xm_A - xm_B) + np.sinh(xm_B))

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    # sub case 2; jmax = jmid > jmin, s3 > s2, {s1} any
    index = (jmax == jmid) * (jmid > jmin)
    
    if cylindrical == True:
        integral = (1 / xm_A**3) *( (2 + xm_A) * (-1 + np.cosh(xm_A)) - (xm_A * np.sinh(xm_A)) )
    elif xm_B_eq_0:
        integral = (-1 + xm_A + np.cosh(xm_A) - np.sinh(xm_A))/ (xm_A**2)
    elif xm_A_eq_0:
        integral = (np.exp(-xm_B)*(-1 + np.exp(xm_B))*(1+np.exp(xm_B)*(-1 + xm_B))) / (xm_B**3)   
    else:
        integral = ((-1 + np.exp(xm_B))/(xm_A * (xm_A - xm_B)*xm_B**2)) \
        * (xm_A + (-1 + np.exp(-xm_A))*xm_B - xm_A*np.cosh(xm_B) + xm_A*np.sinh(xm_B))

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral 

    # BONUS sub case 4; jmax > jmid = jmin, s2 > s1, {s3} any 
    index = (jmax > jmid) * (jmid == jmin)
    
    if cylindrical == True:
        integral = (1 / xm_A**3) *( (2 + xm_A) * (-1 + np.cosh(xm_A)) - (xm_A * np.sinh(xm_A)) )
    elif xm_B_eq_0:
        integral = ((-2+xm_A)*(-1+np.cosh(xm_A))+ (xm_A*np.sinh(xm_A)))/ (xm_A**3)
    elif xm_A_eq_0:
        integral = (-1+xm_B+np.cosh(xm_B) - np.sinh(xm_B))/ (xm_B**2)
    else:
        integral = (((-1 + np.exp(xm_A))*(np.exp(-xm_A - xm_B)))/(xm_B * (xm_A - xm_B)*xm_A**2)) \
        * (-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B) * (xm_A -xm_B) + np.exp(xm_B)*xm_B)

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral 

    # sub case 3; jmax = jmid = jmin, s3 > s2 > s1
    index = (jmax == jmid) * (jmid == jmin)

    if cylindrical == True:
        integral = (1 / xm_A**3) * (np.exp(-xm_A) * (2 + np.exp(xm_A)*(-2 + xm_A) + xm_A))
    elif xm_B_eq_0:
        integral = (2-2*np.exp(-xm_A) - 2*xm_A + xm_A**2)/ (2*xm_A**3)
    elif xm_A_eq_0:
        integral = (2-2*np.exp(-xm_B) - 2*xm_B + xm_B**2)/ (2*xm_B**3)
    else:
        integral = (1 / (xm_A**2 * xm_B - xm_A * xm_B**2))\
        * ( xm_A + (((-1 + np.exp(-xm_B)) * xm_A)/(xm_B)) - xm_B + ((xm_B - np.exp(-xm_A)*xm_B)/(xm_A)) )

    C[np.where(index != 0)] += 1\
                                    * constant[np.where(index != 0)]\
                                    * integral
    return C

def calc_sf4(chrom, M4_arr, k_vec, k_vec_2, k_vec_3, plotting = False):
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom

    nk = len(k_vec)
    N = M*N_m
    sig_inds = [0,1,2,3, 4] #polymer, prot1, prot2, U, Solv
    S4_arr = np.zeros((len(sig_inds),len(sig_inds),len(sig_inds),len(sig_inds))) 

    grid = np.indices((M,M,M,M))
    j1 = grid[0]
    j2 = grid[1]
    j3 = grid[2]
    j4 = grid[3]
    
    for i, k1 in enumerate(k_vec):
        k2 = k_vec_2[i]
        k3 = k_vec_3[i]
        k12 = k1 + k2
        k13 = k1 + k3
        k23 = k2 + k3
        k123 = k1 + k2 + k3
        
        # CASE 1; kA = k1 + k2 + k3; kB = k_1 + k_2; kC = k_1  S4 > S3 > S2 > S1 (and reverse). All cases on wlcstat
        case1 = [[k123, k12, k1], [j4, j3, j2, j1]]
        case2 = [[k123, k12, k2], [j4, j3, j1, j2]]
        case3 = [[k123, k13, k1], [j4, j2, j3, j1]]
        case4 = [[k123, k23, k2], [j4, j1, j3, j2]]
        case5 = [[k123, k13, k3], [j4, j2, j1, j3]]
        case6 = [[k123, k23, k3], [j4, j1, j2, j3]]
        case7 = [[-k3, k12, k1], [j3, j4, j2, j1]]
        case8 = [[-k3, k12, k2], [j3, j4, j1, j2]]
        case9 = [[-k2, k13, k1], [j2, j4, j3, j1]]
        case10 = [[-k1, k23, k2], [j1, j4, j3, j2]]
        case11 = [[-k2, k13, k3], [j2, j4, j1, j3]]
        case12 = [[-k1, k23, k3], [j1, j4, j2, j3]]
        
        case1_deg = [[k1, k12, k123], [j1, j2, j3, j4]]
        case2_deg = [[k2, k12, k123], [j2, j1, j3, j4]]
        case3_deg = [[k1, k13, k123], [j1, j3, j2, j4]]
        case4_deg = [[k2, k23, k123], [j2, j3, j1, j4]]
        case5_deg = [[k3, k13, k123], [j3, j1, j2, j4]]
        case6_deg = [[k3, k23, k123], [j3, j2, j1, j4]]
        case7_deg = [[k1, k12, -k3], [j1, j2, j4, j3]]
        case8_deg = [[k2, k12, -k3], [j2, j1, j4, j3]]
        case9_deg = [[k1, k13, -k2], [j1, j3, j4, j2]]
        case10_deg = [[k2, k23, -k1], [j2, j3, j4, j1]]
        case11_deg = [[k3, k13, -k2], [j3, j1, j4, j2]]
        case12_deg = [[k3, k23, -k1], [j3, j2, j4, j1]]



        case_arr = [case1, case2, case3, case4, case5, case6, \
                   case7, case8, case9, case10, case11, case12, \
                    case1_deg, case2_deg, case3_deg, case4_deg, case5_deg, case6_deg, \
                   case7_deg, case8_deg, case9_deg, case10_deg, case11_deg, case12_deg]
#         print("FASTER??") nope
        # need to consider degenerate cases. flipping each element in array, then appending to original case array
        # case_arr = np.vstack((case_arr, [[np.flipud(el) for el in cse] for cse in case_arr]))
        
#        for each case and sub case, add to a matrix C(j1, j2, j3, j4) which contains the contribution to the overall S4
#        then sum over all indices. Need to keep track of js so that aproiate multiplications with cross corr matrix M4 
        C = np.zeros((M,M,M,M))
        for cse in case_arr:
            kA, kB, kC = cse[0]
            ordered_js = cse[1]
            
            xm_A = (1/6) * N_m * b**2 * np.linalg.norm(kA)**2
            xm_B = (1/6) * N_m * b**2 * np.linalg.norm(kB)**2
            xm_C = (1/6) * N_m * b**2 * np.linalg.norm(kC)**2
            # print("-----------------")
            # print("ks: ", kA, kB, kC)

            C = calc_case_s4(C, xm_A, xm_B, xm_C, ordered_js)
            
        for a1, a2, a3, a4 in product(sig_inds, repeat=4):
            if (a1 == a2 == a3 == a4 == 4): #at S^{(4)}_{SSSS}
                S4_arr[a1][a2][a3][a4] += alpha
            elif (a1 == 4 or a2 == 4 or a3 == 4 or a4 == 4): #at S^{(4)}_{Sxxx}
                S4_arr[a1][a2][a3][a4] += 0
            else:
                S4_arr[a1][a2][a3][a4] += np.sum((1/M**4) * M4_arr[a1][a2][a3][a4] * C)*(N**4)     
    
#     if plotting: # matrix only contains single value, for calculating gamma functions
# #         raise Exception("need to fix return value")
#         return S4_AAAA_arr, S4_AAAB_arr, S4_AABA_arr, S4_ABAA_arr, S4_BAAA_arr, S4_AABB_arr, S4_BBAA_arr, S4_BAAB_arr, S4_ABBA_arr, S4_BABA_arr, S4_ABAB_arr, S4_BBBA_arr, S4_BBAB_arr, S4_BABB_arr, S4_ABBB_arr, S4_BBBB_arr 
    
    return S4_arr

def calc_case_s4(C, xm_A, xm_B, xm_C, ordered_js):

    jmax, jupp, jlow, jmin = ordered_js
    
    xmA_eq_xmB = False
    xmA_eq_xmC = False
    xmB_eq_xmC = False
    epsilon = 0.0000001
    if xm_A + epsilon > xm_B and xm_A - epsilon < xm_B:
        xmA_eq_xmB = True
    if xm_A + epsilon > xm_C and xm_A - epsilon < xm_C:
        xmA_eq_xmC = True
    if xm_B + epsilon > xm_C and xm_B - epsilon < xm_C:
        xmB_eq_xmC = True
        
    xm_B_eq_0 = False
    if xm_B < 1e-5:
        xm_B_eq_0 = True
    # print("xms: ", xm_A, xm_B, xm_C)
    #for each sub case, looking at the degenerate case where 1 and 2 are switched
    constant = np.exp(-xm_A*(jmax - jupp)- xm_B*(jupp - jlow) - xm_C*(jlow - jmin))
    
    # print(constant)
    # sub case 1; jmax > jupp > jlow > jmin, {s1234} any
    index = (jmax > jupp) * (jupp > jlow) * (jlow > jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = 2*(-1 + np.cosh(xm_A)) / xm_A**2
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = (16 / (xm_A**2 * xm_C**2) )* \
                    np.sinh(xm_A / 2)**2 *np.sinh(xm_C / 2)**2
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (16 / xm_A**4 )* \
                    np.sinh(xm_A / 2)**4
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = (16*np.sinh(xm_A / 2) * np.sinh((xm_A - xm_B)/2) * np.sinh((xm_B - xm_C)/2) * np.sinh(xm_C/2))\
                    / (xm_A * (xm_A - xm_B) * (xm_B - xm_C) * xm_C)
    elif xmA_eq_xmB:
        integral = -(2 / (xm_A*xm_C*(xm_A - xm_C))) * \
                    (-np.sinh(xm_A) + np.sinh(xm_A - xm_C) + np.sinh(xm_C))
    elif xmB_eq_xmC:
        integral = -(2 / (xm_A*xm_B*(xm_A - xm_B))) * \
                    (-np.sinh(xm_A) + np.sinh(xm_A - xm_B) + np.sinh(xm_B))
    elif xmA_eq_xmC:
        integral = (16 / (xm_A**2 * (xm_A - xm_B)**2) )* \
                    np.sinh(xm_A / 2)**2 *np.sinh((xm_A - xm_B) / 2)**2

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    # sub case 2; jmax = jupp > jlow > jmin
    index = (jmax == jupp) * (jupp > jlow) * (jlow > jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (np.exp(-xm_A) * (-1 + np.exp(xm_A)) * (-1 + np.exp(xm_A) - xm_A))\
        / (xm_A**3)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = ((np.exp(-xm_A - xm_C)) * (-1+np.exp(xm_C))**2 * (1+np.exp(xm_A) *(-1 + xm_A)))\
        /(xm_A**2*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = ((np.exp(-2*xm_A)) * (-1+np.exp(xm_A))**2 * (1+np.exp(xm_A) *(-1 + xm_A)))\
        /(xm_A**2*xm_A**2)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((np.exp(-xm_A-xm_B-xm_C)) * (-1+np.exp(xm_C)) * (-np.exp(xm_B)+np.exp(xm_C))\
                    * (-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B))\
                    / (xm_A*xm_B*xm_C*(xm_A - xm_B) * (xm_C - xm_B))
    elif xmA_eq_xmB:
        integral = ((np.exp(-xm_A - xm_C)) * (-1+np.exp(xm_C)) * (-np.exp(xm_A)+np.exp(xm_C)) * (-1+np.exp(xm_A) - xm_A))\
        /(xm_A**2*xm_C*(xm_C - xm_A))
    elif xmB_eq_xmC:
        integral = ((-1+np.exp(xm_B))*(xm_A - xm_A*np.exp(-xm_B) + (-1+np.exp(-xm_A))*xm_B))/(xm_B**2*xm_A*(xm_A - xm_B))
    elif xmA_eq_xmC:
        integral = ( (np.exp(-2*xm_A-xm_B))*(-1+np.exp(xm_A))*(np.exp(xm_A) - np.exp(xm_B))\
                    *(-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B)) / (xm_B*xm_A**2*(xm_A-xm_B)**2)

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    


    # sub case 3; jmax > jupp = jlow > jmin
    index = (jmax > jupp) * (jupp == jlow) * (jlow > jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (-1 + np.cosh(xm_A)) / (xm_A**2)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = ((np.exp(-xm_A - xm_C)) * (-1+np.exp(xm_A)) * (-1+np.exp(xm_C)) * (-np.exp(xm_A)*xm_A + np.exp(xm_A+xm_C)*(xm_A-xm_C) + np.exp(xm_C)*xm_C))\
        /(xm_A**2*xm_C**2*(xm_A - xm_C))
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (np.exp(-xm_A) * (-1+np.exp(xm_A))**2 * (-1 + np.exp(xm_A) - xm_A))\
        /(xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((1-np.exp(-xm_A))*(-1+np.exp(xm_C))*( ( (1-np.exp(xm_A-xm_B))/(xm_A-xm_B) ) + ( (-1+np.exp(xm_A-xm_C))/(xm_A-xm_C) ) ))/(xm_A*xm_C*(xm_B - xm_C))
    elif xmA_eq_xmB:
        integral = ((np.exp(-xm_A - xm_C)) * (-1+np.exp(xm_A)) * (-1+np.exp(xm_C)) * (np.exp(xm_A)+np.exp(xm_C)*(-1-xm_A+xm_C)))\
        /(xm_A*xm_C*(xm_A - xm_C)**2)
    elif xmB_eq_xmC:
        integral = ((np.exp(-xm_A - xm_B)) * (-1+np.exp(xm_A)) * (-1+np.exp(xm_B)) * (np.exp(xm_B)+np.exp(xm_A)*(-1+xm_A-xm_B)))\
        /(xm_A*xm_B*(xm_A - xm_B)**2)
    elif xmA_eq_xmC:
        integral = ((np.exp(-xm_A - xm_B)) * (-1+np.exp(xm_A))**2 * (np.exp(xm_A)+np.exp(xm_B)*(-1-xm_A+xm_B)))/(xm_A**2 * (xm_A - xm_B)**2)
  
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    


    # sub case 4; jmax > jupp > jlow = jmin
    index = (jmax > jupp) * (jupp > jlow) * (jlow == jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = ((2+xm_A)*(-1+np.cosh(xm_A)) - xm_A*np.sinh(xm_A))/(xm_A**3)
    
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = (np.exp(-xm_A - xm_C) * (-1+np.exp(xm_A))**2 * (1 + np.exp(xm_C)*(-1 + xm_C)))\
        /(xm_A**2*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (np.exp(-2*xm_A) * (-1+np.exp(xm_A))**2 * (1 + np.exp(xm_A)*(-1 + xm_A)))\
        /(xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((np.exp(-xm_A-xm_B-xm_C)) * (-1+np.exp(xm_A)) * (-np.exp(xm_B)+np.exp(xm_A))\
                    * (-np.exp(xm_B)*xm_B + np.exp(xm_C + xm_B)*(xm_B-xm_C) + np.exp(xm_C)*xm_C))\
                    / (xm_A*xm_B*xm_C*(xm_A - xm_B) * (xm_B - xm_C))
    elif xmA_eq_xmB:
        integral = ((np.exp(-xm_A-xm_C)) * (-1+np.exp(xm_A)) * (-np.exp(xm_A)*xm_A + np.exp(xm_A+xm_C)*(xm_A-xm_C)+np.exp(xm_C)*xm_C)) / (xm_A**2*xm_C*(xm_A - xm_C))
    elif xmB_eq_xmC:
        integral = ((np.exp(-xm_A - xm_B)) * (-1+np.exp(xm_A)) * (-np.exp(xm_A)+np.exp(xm_B)) * (-1+np.exp(xm_B) - xm_B))\
        /(xm_A*xm_B**2*(xm_B - xm_A))
    elif xmA_eq_xmC:
        integral = ( (np.exp(-2*xm_A-xm_B))*(-1+np.exp(xm_A))*(np.exp(xm_A) - np.exp(xm_B))\
                    *(-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B)) / (xm_B*xm_A**2*(xm_A-xm_B)**2)
  
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    
    # sub case 5; jmax == jupp == jlow > jmin
    index = (jmax == jupp) * (jupp == jlow) * (jlow > jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (np.exp(-xm_A) * (-1 + np.exp(xm_A)) * (-2 + 2*np.exp(xm_A) - 2*xm_A - xm_A**2))\
        / (2*xm_A**4)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = ((-1+np.exp(xm_C))*( -xm_C + (   (xm_A*(-1+np.exp(-xm_C) + xm_C))  / (xm_C)  ) +  ( (xm_C-np.exp(-xm_A)*xm_C)  / (xm_A)  )     )) / (xm_A*(xm_A-xm_C)*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (4-4*np.cosh(xm_A) + 2*xm_A*np.sinh(xm_A)) / (xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((-1+np.exp(xm_C)) * (  ( (np.exp(-xm_B))/((xm_A-xm_B)*xm_B)   )  +  ( (xm_B - xm_C)/(xm_A*xm_B*xm_C)  )  +   (   (np.exp(-xm_C))/(xm_C*(xm_C-xm_A))    )  +    (    (np.exp(-xm_A)*(-xm_B+xm_C)) / ( xm_A*(xm_A-xm_B)*(xm_A-xm_C))  )   ))/((xm_B-xm_C)*xm_C)
    elif xmA_eq_xmB:
        integral = ((-1+np.exp(xm_C))* (  (-np.exp(-xm_C) * xm_A**2)     +    ((xm_A-xm_C)**2)    +     (np.exp(-xm_A)*xm_C*(xm_A**2 - xm_A*(-2+xm_C)-xm_C)) ))/(xm_A**2*xm_C**2*(xm_A-xm_C)**2)
    elif xmB_eq_xmC:
        integral = -((-1+np.exp(xm_B))* (  (np.exp(-xm_A) * xm_B**2)     +    -((xm_A-xm_B)**2)    +     (np.exp(-xm_B)*xm_A*(xm_A*(1+xm_B) - xm_B*(2+xm_B))) ))/(xm_B**3*xm_A*(xm_A-xm_B)**2)
    elif xmA_eq_xmC:
        integral = ((-1+np.exp(xm_A))* (  (-np.exp(-xm_B) * xm_A**2)     +    ((xm_A-xm_B)**2)    +     (np.exp(-xm_A)*xm_B*(xm_A**2 - xm_A*(-2+xm_B)-xm_B)) ))/(xm_A**3*xm_B*(xm_A-xm_B)**2)

        
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    # sub case 6; jmax > jupp == jlow = jmin
    index = (jmax > jupp) * (jupp == jlow) * (jlow == jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = ((-1 + np.exp(-xm_A)) * (2 - 2*np.exp(xm_A) + 2*xm_A + xm_A**2))\
        / (2*xm_A**4)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = (np.exp(-xm_A - xm_C)) * (-1 + np.exp(xm_A)) * ( (np.exp(xm_A)*xm_A**2)  +   (np.exp(xm_A+xm_C)*(xm_A-xm_C)*(xm_A*(-1+xm_C) -xm_C) )   -    (np.exp(xm_C)*xm_C**2) ) / (xm_A**3*(xm_A-xm_C)*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (4-4*np.cosh(xm_A) + 2*xm_A*np.sinh(xm_A)) / (xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ( (np.exp(-xm_A - xm_B - xm_C)) * (-1+np.exp(xm_A)) * \
                   (          (-np.exp(xm_A+xm_B)*xm_A*(xm_A-xm_B)*xm_B)      +         (np.exp(xm_A+xm_C)*xm_A*(xm_A-xm_C)*xm_C)     +       (np.exp(xm_B+xm_C)*(xm_B-xm_C)*(np.exp(xm_A)*(xm_A-xm_B)*(xm_A-xm_C) -xm_B*xm_C) )     ))\
                    / (xm_A**2 * (xm_A - xm_B) * xm_B * (xm_A-xm_C) * (xm_B - xm_C) * xm_C)
    elif xmA_eq_xmB:
        integral = ( (np.exp(-xm_A - xm_C)) * (-1 + np.exp(xm_A)) * ((-np.exp(xm_A)*xm_A**2) +   (np.exp(xm_A+xm_C)*(xm_A-xm_C)**2)    +    -(np.exp(xm_C)*xm_C*(-xm_A**2+xm_A*(-2+xm_C)+xm_C)) ))/ (xm_A**3*(xm_A-xm_C)**2*xm_C)
    elif xmB_eq_xmC:
        integral = ( (np.exp(-xm_A - xm_B)) * (-1 + np.exp(xm_A)) * ((-np.exp(xm_B)*xm_B**2) +   (np.exp(xm_A+xm_B)*(xm_A-xm_B)**2)    +    -(np.exp(xm_A)*xm_A*(xm_A*(1+xm_B)-xm_B*(2+xm_B))) ))/ (xm_A**2*(xm_A-xm_B)**2*xm_B**2)
    elif xmA_eq_xmC:
        integral = ( (np.exp(-xm_A - xm_B)) * (-1 + np.exp(xm_A)) * ((-np.exp(xm_A)*xm_A**2) +   (np.exp(xm_A+xm_B)*(xm_A-xm_B)**2)    +    -(np.exp(xm_B)*xm_B*(-xm_A**2+xm_A*(-2+xm_B)+xm_B)) ))/ (xm_A**3*(xm_A-xm_B)**2*xm_B)
    
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    # sub case 7; jmax == jupp == jlow = jmin
    index = (jmax == jupp) * (jupp == jlow) * (jlow == jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (np.exp(-xm_A) * (6 + 2*(np.exp(xm_A)*(-3+xm_A)) + 4*xm_A + xm_A**2) ) / (2*xm_A**4)
    
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = (  (2*(-1 + np.exp(-xm_A))*xm_C**3) + (2*xm_A*xm_C**3)  + (-xm_A**2*xm_C**3) + (xm_A**3*(2-2*np.exp(-xm_C)-2*xm_C+xm_C**2)) ) / (2*xm_A**3*(xm_A-xm_C)*xm_C**3)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (np.exp(-xm_A) / (2*xm_A**4)) * ( (-2*(3+xm_A)) + (np.exp(xm_A) * (6-4*xm_A + xm_A**2)))

    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ( (np.exp(-xm_A)) / (xm_A**2 * (xm_A-xm_B) * (xm_A - xm_C)) )      +      ( (np.exp(-xm_B)) / (xm_B**2 * (xm_B-xm_A) * (xm_B - xm_C)) )     +      ( (np.exp(-xm_C)) / (xm_C**2 * (xm_C-xm_A) * (xm_C - xm_B)) )      +        -(  (xm_B*xm_C + xm_A*(xm_B+xm_C-xm_B*xm_C))   /    (xm_A**2*xm_B**2*xm_C**2)  )
    elif xmA_eq_xmB:
        integral = (  (-xm_A**3) + (np.exp(-xm_C)*xm_A**3) + (xm_A*(xm_A-xm_C)**2*xm_C) + ((3*xm_A-2*xm_C)*xm_C**2) + ( np.exp(-xm_A)*xm_C**2 *(2*xm_C + xm_A*(-3-xm_A+xm_C))) ) / (xm_A**3 * xm_C**2 * (xm_A - xm_C)**2)
    elif xmB_eq_xmC:
        integral = ( ((xm_A - xm_B)**2 * (xm_A*(-2 + xm_B) - xm_B))     +     (np.exp(-xm_A) * xm_B**3)     +    (np.exp(-xm_B)*xm_A**2 * (xm_A*(2+xm_B) - xm_B*(3 + xm_B))  )      ) / (xm_A**2 * (xm_A - xm_B)**2 * xm_B**3)
    elif xmA_eq_xmC:
        integral =  (  (-xm_A**3) + (np.exp(-xm_B)*xm_A**3) + (xm_A*(xm_A-xm_B)**2*xm_B) + ((3*xm_A-2*xm_B)*xm_B**2) + ( np.exp(-xm_A)*xm_B**2 *(2*xm_B + xm_A*(-3-xm_A+xm_B))) ) / (xm_A**3 * xm_B**2 * (xm_A - xm_B)**2)
 
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral #* 2
    # sub case 8; jmax == jupp > jlow = jmin
    index = (jmax == jupp) * (jupp > jlow) * (jlow == jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (np.exp(-xm_A) * (1 + -np.exp(xm_A) +xm_A)**2)\
        / (xm_A**4)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = ((np.exp(-xm_A - xm_C)) * (1+np.exp(xm_A) *(-1 + xm_A))*(1+np.exp(xm_C)*(-1+xm_C)))\
        /(xm_A**2*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = ( np.exp(-2*xm_A) * (1+np.exp(xm_A)*(-1 + xm_A))**2)\
        /(xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((np.exp(-xm_A-xm_B-xm_C)) * (-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B) *(-np.exp(xm_B)*xm_B + np.exp(xm_B + xm_C)*(xm_B-xm_C) + np.exp(xm_C)*xm_C) )\
                    / (xm_A*xm_B**2*xm_C*(xm_A - xm_B) * (xm_B - xm_C))
    elif xmA_eq_xmB:
        integral = ( (np.exp(-xm_A - xm_C)) * (-1 + np.exp(xm_A)-xm_A) * ( (-np.exp(xm_A)*xm_A) +   (np.exp(xm_A+xm_C)*(xm_A-xm_C))    +  (np.exp(xm_C)*xm_C) ))/ (xm_A**3*(xm_A-xm_C)*xm_C)
    elif xmB_eq_xmC:
        integral = ( (-1+np.exp(xm_B) - xm_B) * (xm_A-np.exp(-xm_B)*xm_A + (-1+np.exp(-xm_A))*xm_B) )/ (xm_A*(xm_A-xm_B)*xm_B**3)
    elif xmA_eq_xmC:
        integral = ( (np.exp(-2*xm_A-xm_B)) * (-np.exp(xm_A)*xm_A + np.exp(xm_A+xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B)**2) / (xm_A**2*xm_B**2*(xm_A-xm_B)**2)
  
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
 
    return C





























######################################################### GAMMAS ######################################################################################
######################################################### GAMMAS ######################################################################################
######################################################### GAMMAS ######################################################################################
######################################################### GAMMAS ######################################################################################
######################################################### GAMMAS ######################################################################################
######################################################### GAMMAS ######################################################################################
######################################################### GAMMAS ######################################################################################





























def gamma2(chrom, s_bnd_A, s_bnd_B, K, chi):
    # chrom object contains polymer parameters
    # s_bind arrays of appropriate mu1, mu2
    # polymer-solv chi

    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    #edit prefactor
    # avo = 6.02e23 # num / mol
    # dens_p = 1 # g/cm^3
    # mol_weight_p = 1e5 # g/mol
    # rho_p = avo*(1/mol_weight_p)*dens_p*M*(100**3)*(1/1e9)**3
    # n_p = 1e8 
    s_bnd_U = 1 - s_bnd_A - s_bnd_B       
    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)
    s_cu_red = eval_and_reduce_cgam(s_bnd_U)
    sisj_UA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_A)
    sisj_UB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_B)
    sisj_UU_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_U)

    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red, s_cu_red, sisj_UA_red, sisj_UB_red, sisj_UU_red]

    #calc sf2
    # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  rho_p * ((N_m**2 * M)/n_p) * np.array(calc_sf2_chromo_shlk(chrom, M2s, [K]))
    # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  np.array(calc_sf2_chromo_shlk(chrom, M2s, [K]))
    g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc, cu, ug1, ug2, uu = np.array(calc_sf2_chromo_shlk(chrom, M2s, [K]))
    
    # ss = alpha#1-phi_c
    # S2_mat = (rho_c / M) * np.array([[cc[0], 0, cg1[0], cg2[0]],\
    #                 [0, ss, 0, 0], \
    #                 [cg1[0], 0, g1g1[0], g1g2[0]],\
    #                 [cg2[0], 0, g2g1[0], g2g2[0]]])
    ss = alpha#1-phi_c
    # S2_mat = (rho_c / M) *  np.array([[cc[0], 0, cg1[0], cg2[0], cu[0]],\
    #                 [0, ss, 0, 0, 0], \
    #                 [cg1[0], 0, g1g1[0], g1g2[0], ug1[0]],\
    #                 [cg2[0], 0, g2g1[0], g2g2[0], ug2[0]],\
    #                 [cu[0], 0, ug1[0], ug2[0], uu[0]]])
    

    S2_mat = (rho_c / M) *  np.array([[cc[0], cg1[0], cg2[0], cu[0], 0],\
                                    [cg1[0], g1g1[0], g1g2[0], ug1[0], 0],\
                                    [cg2[0], g2g1[0], g2g2[0], ug2[0], 0],\
                                    [cu[0], ug1[0], ug2[0], uu[0], 0],\
                                    [0, 0, 0, 0, ss]])

    # S2_mat *= (N_m**2 * M)/n_p
    # S2_mat[1][1] /= (N_m**2 * M)/n_p
    
    # rho_c_test = rho_p
    # S2_mat /= rho_c
    # S2_mat *= rho_c_test
    # S2_mat[1][1] *= (rho_c/rho_c_test)    

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)

    # poly/solv reduction
    S2_inv_ps =  np.array([[S2_inv[0,0] + S2_inv[4,4], S2_inv[0,1], S2_inv[0, 2], S2_inv[0, 3]],\
       [S2_inv[1,0], S2_inv[1,1] , S2_inv[1,2], S2_inv[1,3]],\
       [S2_inv[2,0], S2_inv[2,1] , S2_inv[2,2], S2_inv[2,3]],\
       [S2_inv[3, 0], S2_inv[3, 1], S2_inv[3, 2], S2_inv[3,3]]])
    
    # then apply unbound poly reduction
    T = np.array([[1,0,0], [0,1,0], [0,0,1], [1,-1,-1]]) # \Delta_{unred} = T \Delta_{red}

    S2_inv_full = np.einsum("ij, ik, jl -> kl", S2_inv_ps, T, T) # only in terms of P, A, B

    G2 = np.array([[S2_inv_full[0,0] - 2*chi, S2_inv_full[0,1], S2_inv_full[0, 2]],\
       [S2_inv_full[1,0], S2_inv_full[1,1] + v_int[0,0]*Vol_int, S2_inv_full[1,2] + v_int[0,1]*Vol_int],\
       [S2_inv_full[2,0], S2_inv_full[2,1] + v_int[1,0]*Vol_int, S2_inv_full[2,2] + v_int[1,1]*Vol_int]])
        
    return G2

def calc_fa(phia, phib):
    nm = len(phia)
    
    ind = 0
    for i in range(nm):
        if phia[i] > phib[i]:
            ind += 1
    
    fa = ind / nm
    
    return fa
def calc_fb(phia, phib):
    nm = len(phia)
    
    ind = 0
    for i in range(nm):
        if phib[i] > phia[i]:
            ind += 1
    
    fb = ind / nm
    
    return fb
    
# def s2wlc_zeroq(chrom):
#     [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom
#     #f_om now holds FA, FB
#     FA, FB = f_om
#     return [FA*FA], [FA*FB], [FB*FA], [FB*FB], [FA], [FB], [1]

def sf2_inv_zeroq(chrom, rho_p, s_bnd_A, s_bnd_B):
    print("NEED TO REDO FOR UNBOUND")
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    fa = calc_fa(s_bnd_A, s_bnd_B)
    fb = calc_fb(s_bnd_A, s_bnd_B)
    s2 = np.ones((3,3),dtype='complex') / 9
    s2[0,0] += (N**2 / alpha)
    s2[1,0] *= (1/fa)
    s2[0,1] *= (1/fa)
    s2[2,0] *= (1/fb)
    s2[0,2] *= (1/fb)
    s2[1,1] *= (1/ fa**2)
    s2[1,2] *= (1 / (fa * fb))
    s2[2,1] *= (1 / (fa * fb))
    s2[2,2] *= (1/fb**2)

    s2 *= (M / (2*rho_p*N**2)) 
    return s2#np.zeros((3,3),dtype='complex')#s2
    
# def sf2_inv(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B):
#     [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
#     if np.linalg.norm(K1) < 1e-5:
#         # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  rho_p * ((N_m**2 * M)/n_p) * np.array(s2wlc_zeroq(chrom))
#         return sf2_inv_zeroq(chrom, rho_p, s_bnd_A, s_bnd_B)
    
#     g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc, cu, ug1, ug2, uu = np.array(calc_sf2_chromo_shlk(chrom, M2s, [K1]))
    
#     ss = alpha#1-phi_c
#     S2_mat = (rho_c / M) *  np.array([[cc[0], 0, cg1[0], cg2[0], cu[0]],\
#                     [0, ss, 0, 0, 0], \
#                     [cg1[0], 0, g1g1[0], g1g2[0], ug1[0]],\
#                     [cg2[0], 0, g2g1[0], g2g2[0], ug2[0]],\
#                     [cu[0], 0, ug1[0], ug2[0], uu[0]]])

#     #invert, calc g2
#     S2_inv = np.linalg.inv(S2_mat)

#     # poly/solv reduction
#     S2_inv_ps =  np.array([[S2_inv[0,0] + S2_inv[1,1], S2_inv[0,2], S2_inv[0, 3], S2_inv[0, 4]],\
#        [S2_inv[2,0], S2_inv[2,2] , S2_inv[2,3], S2_inv[2,4]],\
#        [S2_inv[3,0], S2_inv[3,2] , S2_inv[3,3], S2_inv[3,4]],\
#        [S2_inv[4, 0], S2_inv[4, 2], S2_inv[4, 3], S2_inv[4, 4]]])
    
#     # then apply unbound poly reduction
#     T = np.array([[1,0,0], [0,1,0], [0,0,1], [1,-1,-1]]) # \Delta_{unred} = T \Delta_{red}

#     S2_inv_full = np.einsum("ij, ik, jl -> kl", S2_inv_ps, T, T) # only in terms of P, A, B


#     # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  np.array(calc_sf2_chromo_shlk(chrom, M2s, [K1]))
#     # ss = alpha#
#     # S2_mat_k1 = (rho_c / M) * np.array([[cc[0], 0, cg1[0], cg2[0]],\
#                     # [0, ss, 0, 0], \
#                     # [cg1[0], 0, g1g1[0], g1g2[0]],\
#                     # [cg2[0], 0, g2g1[0], g2g2[0]]])
#     # S2_inv = np.linalg.inv(S2_mat_k1)
#     # S2_inv_red = np.array([[S2_inv[0,0] + S2_inv[1,1], S2_inv[0,2], S2_inv[0, 3]],\
#     #    [S2_inv[2,0], S2_inv[2,2], S2_inv[2,3] ],\
#     #    [S2_inv[3,0], S2_inv[3,2] , S2_inv[3,3]]])  
#     return S2_inv_full

def sf2_inv_raw(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B):
    # UNREDUCED
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    if np.linalg.norm(K1) < 1e-5:
        # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  rho_p * ((N_m**2 * M)/n_p) * np.array(s2wlc_zeroq(chrom))
        return sf2_inv_zeroq(chrom, rho_p, s_bnd_A, s_bnd_B)
    
    g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc, cu, ug1, ug2, uu = np.array(calc_sf2_chromo_shlk(chrom, M2s, [K1]))
    
    ss = alpha#1-phi_c
    S2_mat = (rho_c / M) *  np.array([[cc[0], cg1[0], cg2[0], cu[0], 0],\
                                    [cg1[0], g1g1[0], g1g2[0], ug1[0], 0],\
                                    [cg2[0], g2g1[0], g2g2[0], ug2[0], 0],\
                                    [cu[0], ug1[0], ug2[0], uu[0], 0],\
                                    [0, 0, 0, 0, ss]])

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)
    return S2_inv

def gamma3(chrom, s_bnd_A, s_bnd_B, Ks):
    K1, K2, K3 = Ks
    
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise Exception('Qs must add up to zero')

    
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    # avo = 6.02e23 # num / mol
    # dens_p = 1 # g/cm^3
    # mol_weight_p = 1e5 # g/mol
    # rho_p = avo*(1/mol_weight_p)*dens_p*M*(100**3)*(1/1e9)**3
    # n_p = 1e8 
    
    # print("removed prefactors for s3, s4 in gamma 4")
    # n_p = 1
    # rho_p = 1
    # M = 1  
    
    s_bnd_U = 1 - s_bnd_A - s_bnd_B       
    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)
    s_cu_red = eval_and_reduce_cgam(s_bnd_U)
    sisj_UA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_A)
    sisj_UB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_B)
    sisj_UU_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_U, s_bnd_U)

    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red, s_cu_red, sisj_UA_red, sisj_UB_red, sisj_UU_red]
    #calc sf2\
    S2_inv_raw = sf2_inv_raw(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B)
    # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  rho_p * ((N_m**2 * M)/n_p) * np.array(calc_sf2_chromo_shlk(chrom, M2s, [K1]))
    # ss = rho_s#
    # S2_mat_k1 = 1/N**2 * np.array([[cc[0], 0, cg1[0], cg2[0]],\
    #                 [0, ss*N**2, 0, 0], \
    #                 [cg1[0], 0, g1g1[0], g1g2[0]],\
    #                 [cg2[0], 0, g2g1[0], g2g2[0]]])
    # S2_inv = np.linalg.inv(S2_mat_k1)
    # S2_inv_red = np.array([[S2_inv[0,0] + S2_inv[1,1], S2_inv[0,2], S2_inv[0, 3]],\
    #    [S2_inv[2,0], S2_inv[2,2], S2_inv[2,3] ],\
    #    [S2_inv[3,0], S2_inv[3,2] , S2_inv[3,3]]])    

    
    # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  rho_p * ((N_m**2 * M)/n_p) * np.array(calc_sf2_chromo_shlk(chrom, M2s, [K2]))
    # ss = rho_s#
    # S2_mat_k2 = 1/N**2 * np.array([[cc[0], 0, cg1[0], cg2[0]],\
    #                 [0, ss*N**2, 0, 0], \
    #                 [cg1[0], 0, g1g1[0], g1g2[0]],\
    #                 [cg2[0], 0, g2g1[0], g2g2[0]]])  
    # S2_inv = np.linalg.inv(S2_mat_k2)
    # S2_inv_red_2 = np.array([[S2_inv[0,0] + S2_inv[1,1], S2_inv[0,2], S2_inv[0, 3]],\
    #    [S2_inv[2,0], S2_inv[2,2], S2_inv[2,3] ],\
    #    [S2_inv[3,0], S2_inv[3,2] , S2_inv[3,3]]])  
    S2_inv_raw_2 = sf2_inv_raw(chrom, M2s, K2, rho_p, s_bnd_A, s_bnd_B)

    
    # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  rho_p * ((N_m**2 * M)/n_p) * np.array(calc_sf2_chromo_shlk(chrom, M2s, [K3]))
    # ss = rho_s#
    # S2_mat_k3 = 1/N**2 * np.array([[cc[0], 0, cg1[0], cg2[0]],\
    #                 [0, ss*N**2, 0, 0], \
    #                 [cg1[0], 0, g1g1[0], g1g2[0]],\
    #                 [cg2[0], 0, g2g1[0], g2g2[0]]])  
    # S2_inv = np.linalg.inv(S2_mat_k3)
    # S2_inv_red_3 = np.array([[S2_inv[0,0] + S2_inv[1,1], S2_inv[0,2], S2_inv[0, 3]],\
    #    [S2_inv[2,0], S2_inv[2,2], S2_inv[2,3] ],\
    #    [S2_inv[3,0], S2_inv[3,2] , S2_inv[3,3]]])  
    S2_inv_raw_3 = sf2_inv_raw(chrom, M2s, K3, rho_p, s_bnd_A, s_bnd_B)

    #
    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)


    s3 = ( rho_p/(M) ) * calc_sf3(chrom, M3, [K1], [K2])
    # s3 *= n_p # HAIL MARY
    # s3 /= n_p**(1/2) # HAIL MARY
    # s3 /= N_m**(3) # HAIL MARY
    # print("HAIL MARY")

    #s3 prefactor:  goal is 1/V * (N**3 * 1/N**3) = rho_p / (M * np)
    
    # s3[0,0,0] += alpha * (rho_p/M) #solvent sf
    T = np.array([[1,0,0], [0,1,0], [0,0,1], [1,-1,-1], [-1,0,0]]) # \Delta_{unred} = T \Delta_{red}           
        
    G3 = np.einsum("ijk,il,jm,kn-> lmn", -s3, S2_inv_raw, S2_inv_raw_2, S2_inv_raw_3)

    G3_red = np.einsum("ijk, im, jn, ko -> mno", G3, T, T, T) # only in terms of P, A, B

    return G3_red

def gamma4(chrom, s_bnd_A, s_bnd_B, Ks):
    K1, K2, K3, K4 = Ks
    if np.linalg.norm(K1+K2+K3+K4) >= 1e-10:
        raise Exception('Qs must add up to zero')    
    K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)

    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    
    M4 = calc_mon_mat_4(s_bnd_A, s_bnd_B)
    
    s4 = ( rho_c/(M) ) * calc_sf4(chrom, M4, [K1], [K2], [K3]) 

    # print("HAIL MARY")
    # print(alpha)
    # s4[0,0,0,0] += alpha * (rho_c/M)

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)

    # print("LAM s3s mad 0")
    # s3_12 = np.zeros((3,3,3))#0#( rho_c/(M) ) * calc_sf3(chrom, M3, [K1], [K2])
    # s3_13 = ( rho_c/(M) ) * calc_sf3(chrom, M3, [K1], [K3])
    # s3_14 = np.zeros((3,3,3))#0#( rho_c/(M) ) * calc_sf3(chrom, M3, [K1], [K4])
    # s3_23 = np.zeros((3,3,3))#0#( rho_c/(M) ) * calc_sf3(chrom, M3, [K2], [K3])
    # s3_24 = ( rho_c/(M) ) * calc_sf3(chrom, M3, [K2], [K4])
    # s3_34 = np.zeros((3,3,3))#0#( rho_c/(M) ) * calc_sf3(chrom, M3, [K3], [K4])
    
    # #
    # print("TOOK AWAY THIS TOO!!!")
    # # s3_12[0,0,0] += alpha * (rho_c/M)#solvent sf
    # s3_13[0,0,0] += alpha * (rho_c/M)#solvent sf
    # # s3_14[0,0,0] += alpha * (rho_c/M)#solvent sf
    # # s3_23[0,0,0] += alpha * (rho_c/M)#solvent sf
    # s3_24[0,0,0] += alpha * (rho_c/M)#solvent sf
    # # s3_34[0,0,0] += alpha * (rho_c/M)#solvent sf


    s3_12 = ( rho_c/(M) ) * calc_sf3(chrom, M3, [K1], [K2])
    s3_13 = ( rho_c/(M) ) * calc_sf3(chrom, M3, [K1], [K3])
    s3_14 = ( rho_c/(M) ) * calc_sf3(chrom, M3, [K1], [K4])
    s3_23 = ( rho_c/(M) ) * calc_sf3(chrom, M3, [K2], [K3])
    s3_24 = ( rho_c/(M) ) * calc_sf3(chrom, M3, [K2], [K4])
    s3_34 = ( rho_c/(M) ) * calc_sf3(chrom, M3, [K3], [K4])
    
    
    s3_12[0,0,0] += alpha * (rho_c/M)#solvent sf
    s3_13[0,0,0] += alpha * (rho_c/M)#solvent sf
    s3_14[0,0,0] += alpha * (rho_c/M)#solvent sf
    s3_23[0,0,0] += alpha * (rho_c/M)#solvent sf
    s3_24[0,0,0] += alpha * (rho_c/M)#solvent sf
    s3_34[0,0,0] += alpha * (rho_c/M)#solvent sf
    
    rho_p = rho_c
    n_p = np.nan 
    
    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)    
    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]
    
    S2_inv_red = sf2_inv_raw(chrom, M2s, K, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_12 = sf2_inv_raw(chrom, M2s, K12, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_13 = sf2_inv_raw(chrom, M2s, K13, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_14 = sf2_inv_raw(chrom, M2s, K14, rho_p, s_bnd_A, s_bnd_B)

    S2_inv_red_2 = sf2_inv_raw(chrom, M2s, K2, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_3 = sf2_inv_raw(chrom, M2s, K3, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_4 = sf2_inv_raw(chrom, M2s, K4, rho_p, s_bnd_A, s_bnd_B)


    # print("Sfs:")
    # print("S4:", s4)
    # print("s3s:")
    # print(s3_12)
    # print(s3_13)
    # print(s3_14)
    # print(s3_23)
    # print(s3_24)
    # print(s3_34)

    # print("s2s:")
    # print(S2_inv_red)
    # print(S2_inv_red_12) #OUTLIER
    # print(S2_inv_red_13)
    # print(S2_inv_red_14) #OUTLIER
    # print(S2_inv_red_2)
    # print(S2_inv_red_3)
    # print(S2_inv_red_4)

    # S2_inv_red_12 *= 1e-6
    # S2_inv_red_14 *= 1e-6
    

    part1 = np.einsum("ijkl, im, jn, ko, lp-> mnop", s4, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) 
    # print("g4 part1:")
    # print(part1)
    # print("NO PART 2 IN GAM 4")
    part2 = 0

    # # ORIGINAL
    # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)


    # edited index so that alphas match correctly b/w s3s and s2s
    part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 += np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 += np.einsum("abc, def, cf, ai, dk, el, bj -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    
    # print("g4 part2 raw NEW:")
    # print(part2)

    # print("non zero part of part 2:", np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4))

    # print("zero part of p2:", np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4))

    # print("-------------------------------------------------------------------------------------------------")
    # print("swithced index equivalency test")
    # # part2_A = np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    # # part2_B = np.einsum("abc, def, cf, dk, ai, el, bj -> ijkl" ,s3_24, s3_13, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)

    # print("PART A:", part2_A)
    # print("PART B:", part2_B)
    # print(part2_A == part2_B)
    # print("-------------------------------------------------------------------------------------------------")

    # V = (2*np.pi)**3#1e01
    # print("part2 div by V= %s" % V)
    # part2 /= V
    #WRONG::
    # for q in Ks:
    #     s2_summed = sf2_inv(chrom, M2s, q, rho_p, n_p)

    #     part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, s2_summed, S2_inv_red, S2_inv_red_12, S2_inv_red_13, S2_inv_red_14)
    #     part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_13, s3_24, s2_summed, S2_inv_red, S2_inv_red_12, S2_inv_red_13, S2_inv_red_14)
    #     part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_14, s3_23, s2_summed, S2_inv_red, S2_inv_red_12, S2_inv_red_13, S2_inv_red_14)
        
        # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, s2_summed, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
        # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_13, s3_24, s2_summed, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
        # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_14, s3_23, s2_summed, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)

        # part2 += np.einsum("ijk, ijk, il, jm, kn, il, jm -> ijkl" ,s3_12, s3_34, s2_summed, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
        # part2 += np.einsum("ijk, ijk, il, jm, kn, il, jm -> ijkl" ,s3_13, s3_24, s2_summed, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
        # part2 += np.einsum("ijk, ijk, il, jm, kn, il, jm -> ijkl" ,s3_14, s3_23, s2_summed, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4
                               
    # np.einsum("",
    # print("made negative")
    # print("ONLY PART 1")
    G4 = (part1 - part2)
    T = np.array([[1,0,0], [0,1,0], [0,0,1], [1,-1,-1], [-1,0,0]]) # \Delta_{unred} = T \Delta_{red}           
        
    # G3 = np.einsum("ijk,il,jm,kn-> lmn", -s3, S2_inv_raw, S2_inv_raw_2, S2_inv_raw_3)

    G4_red = np.einsum("ijkl, im, jn, ko, lp -> mnop", G4, T, T, T, T) # only in terms of P, A, B
    return G4_red




