from OABS_util import *

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
    

def calc_binding_states(psol):
    # evaluate average binding state for each nucleosome at each mu1,mu2
    
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    [pa_vec, marks_2] = psol.poly_marks

    # mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    # mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]

    s_bind_1_soln_arr = np.zeros((len(psol.mu1_arr), len(psol.mu2_arr), psol.M))
    s_bind_2_soln_arr = np.zeros((len(psol.mu1_arr), len(psol.mu1_arr), psol.M))
    
    f_ref = np.min(np.array([psol.v_int[0,0], psol.v_int[1,1], psol.v_int[0,1], psol.e_m[0] / 2,  psol.e_m[1] / 2]))

    for i, mu1 in enumerate(psol.mu1_arr):
        for j, mu2 in enumerate(psol.mu1_arr):
            s_bind_ave_a, s_bind_ave_b = eval_phi(pa_vec, mu1, mu2, psol.e_m[0], psol.e_m[1], psol.v_int[0,0], psol.v_int[1,1], psol.v_int[0,1], f_ref)
            s_bind_1_soln_arr[i,j,:] = s_bind_ave_a
            s_bind_2_soln_arr[i,j,:] = s_bind_ave_b
    
    return s_bind_1_soln_arr, s_bind_2_soln_arr