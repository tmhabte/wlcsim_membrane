# from OABS_util import *
from OABS_binding_calc import *

def calc_sf2(psol, M2s, k_vec, competitive):
    # calculates sf2 using rank 1 monomer correlation tensor 
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    M = psol.M
    N_m = psol.N_m
    b = psol.b
    alpha = psol.alpha
    N = psol.N

    # M2_AA, M2_AB, M2_BA, M2_BB, M1_cgam0, M1_cgam1, M2_cc = M2s CHANGE

    # M = np.shape(M2_AA)[0]
    # nk = len(k_vec)
    # N = M*N_m
    
    # S2_AA_arr = np.zeros(nk)
    # S2_AB_arr = np.zeros(nk)
    # S2_BA_arr = np.zeros(nk)
    # S2_BB_arr = np.zeros(nk)
    
    # S2_cgam0_arr = np.zeros(nk)
    # S2_cgam1_arr = np.zeros(nk)
    # S2_cc_arr = np.zeros(nk)


    if competitive:
        # explicit competitive binding- sig_AB not considered
        sig_inds = [0,1,2] # O, gamma1, gamma2
    else:
        sig_inds = [0,1,2,3] # O, gamma1, gamma2, gamma1&gamma2

    S2_arr = np.zeros((len(sig_inds)+1,len(sig_inds)+1)) #previous ocmponents + solvent

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

        # S2_AA_arr[i] = np.sum((1/M**2) * C * M2_AA)
        # S2_AB_arr[i] = np.sum((1/M**2) * C * M2_AB)
        # S2_BA_arr[i] = np.sum((1/M**2) * C * M2_BA)
        # S2_BB_arr[i] = np.sum((1/M**2) * C * M2_BB)
        
        # S2_cgam0_arr[i] = np.sum((1/M**2) * C * M1_cgam0)
        # S2_cgam1_arr[i] = np.sum((1/M**2) * C * M1_cgam1)
        # S2_cc_arr[i] = np.sum((1/M**2) * C * M2_cc)

        solvent_index = sig_inds[-1] + 1 # solvent index is always the last one
        for a1, a2 in product(sig_inds+[solvent_index], repeat=2):
            if (a1 == a2 == solvent_index): #at S^{(2)}_{SS}
                S2_arr[a1][a2] += alpha
            elif (a1 == solvent_index or a2 == solvent_index): #at S^{(2)}_{Sx}
                S2_arr [a1][a2] += 0
            else:
                S2_arr[a1][a2] += np.sum((1/M**2) * M2s[a1][a2] * C)*(N**2)
    
    return S2_arr
    # return S2_AA_arr*N**2, S2_AB_arr*N**2, S2_BA_arr*N**2, S2_BB_arr*N**2, S2_cgam0_arr*N**2, S2_cgam1_arr*N**2, S2_cc_arr*N**2
    # return S2_AA_arr, S2_AB_arr, S2_BA_arr, S2_BB_arr, S2_cgam0_arr, S2_cgam1_arr, S2_cc_arr

def inc_gamma(a, x): 
    # got rid of a = 0 condition!
    return sp.special.gamma(a)*sp.special.gammaincc(a, x)

def erf(n):
    return sp.special.erf(n)
        
def eval_and_reduce_sisj_bind_simp(psol, s_bnd_A, s_bnd_B):
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    M = psol.M

    sisj_tens = np.zeros(M, dtype = DATA_TYPE)
    sisj_tens[0] = np.sum(s_bnd_A * s_bnd_B)
    conv = signal.convolve(s_bnd_A, s_bnd_B[::-1])
    sisj_tens[1:] = conv[:M-1][::-1] + conv[:M-1:-1][::-1]
    
    return sisj_tens

# import psutil
# import os


def calc_sf_mats(psol, s_bind_1_soln_arr, s_bind_2_soln_arr, k_vec, competitive):
    # returns rank 3 tensor of mu1, mu2 , k, each value is S2 matrix
    # Used for spinodal calculation    
    # print(" I USING DATA TYPE " + str(DATA_TYPE))
    phi_p = psol.phi_p
    N = psol.N

#     start_time = time.time()
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    # [marks_1, marks_2] = psolpoly_marks
    # len_marks_1 = len(marks_1)
    
    # mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    # mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    
    sf2_mat = np.zeros((len(psol.mu1_arr), len(psol.mu2_arr), len(k_vec)), dtype = "object")
    print("vm=A=1")
    for i, mu1 in enumerate(psol.mu1_arr):
        for j, mu2 in enumerate(psol.mu2_arr):
            
            mu = [mu1, mu2]
            s_bnd_A = s_bind_1_soln_arr[i,j,:]
            s_bnd_B = s_bind_2_soln_arr[i,j,:]
            
            M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)
            for ik, k in enumerate(k_vec):
                S2_mat = (phi_p / N) * np.array(calc_sf2(psol, M2s, [k], competitive))
                # S2_mat[-1,-1] *= (N / M)

                sf2_mat[i][j][ik] = S2_mat

    return sf2_mat

def calc_sisjs(s_bind_A, s_bind_B):
    sig_0 = (1-s_bind_A)*(1-s_bind_B)
    sig_A = s_bind_A * (1-s_bind_B)
    sig_B = s_bind_B * (1 - s_bind_A)
    sig_AB = s_bind_A * s_bind_B   
    sisj_arr = [sig_0, sig_A, sig_B, sig_AB]
    return sisj_arr

def calc_mon_mat_2(s_bind_A, s_bind_B, competitive):
    nm = len(s_bind_A)
   
    sisj_arr =  calc_sisjs(s_bind_A, s_bind_B) #[sig_0, sig_A, sig_B, sig_AB]

    if competitive:
        # explicit competitive binding- sig_AB not considered
        sig_inds = [0,1,2] # O, gamma1, gamma2
    else:
        sig_inds = [0,1,2,3] # O, gamma1, gamma2, gamma1&gamma2

    M2_arr = np.zeros((len(sig_inds), len(sig_inds)), dtype= "object")
    for a1, a2 in product(sig_inds, repeat=2):
        # print([a1, a2, a3])
        # M2_arr[a1][a2] = np.einsum("i,j", sisj_arr[a1],  sisj_arr[a2])

        #calculate reduced monomer tensor 
        sisj_tens = np.zeros(nm)

        sisj_tens[0] = np.sum(sisj_arr[a1] * sisj_arr[a2])
        conv = signal.convolve(sisj_arr[a1], sisj_arr[a2][::-1])
        sisj_tens[1:] = conv[:nm-1][::-1] + conv[:nm-1:-1][::-1]
        M2_arr[a1][a2] = sisj_tens

    return M2_arr

def calc_mon_mat_3(s_bind_A, s_bind_B, competitive):
    nm = len(s_bind_A)

    sisj_arr =  calc_sisjs(s_bind_A, s_bind_B)


    if competitive:
        # explicit competitive binding- sig_AB not considered
        sig_inds = [0,1,2] # O, gamma1, gamma2
    else:
        sig_inds = [0,1,2,3] # O, gamma1, gamma2, gamma1&gamma2

    M3_arr = np.zeros((len(sig_inds), len(sig_inds), len(sig_inds)), dtype= "object")
    for a1, a2, a3 in product(sig_inds, repeat=3):
        # print([a1, a2, a3])
        M3_arr[a1][a2][a3] = np.einsum("i,j, k", sisj_arr[a1], sisj_arr[a2], sisj_arr[a3])

    return M3_arr

def calc_mon_mat_4(s_bind_A, s_bind_B, competitive):
    nm = len(s_bind_A)

    sisj_arr =  calc_sisjs(s_bind_A, s_bind_B)

    if competitive:
        # explicit competitive binding- sig_AB not considered
        sig_inds = [0,1,2] # O, gamma1, gamma2
    else:
        sig_inds = [0,1,2,3] # O, gamma1, gamma2, gamma1&gamma2

    M4_arr = np.zeros((len(sig_inds), len(sig_inds), len(sig_inds), len(sig_inds)), dtype= "object")
    for a1, a2, a3, a4 in product(sig_inds, repeat=4):
        # print([a1, a2, a3])
        M4_arr[a1][a2][a3][a4] = np.einsum("i,j,k,l", sisj_arr[a1], sisj_arr[a2], sisj_arr[a3], sisj_arr[a4])

    return M4_arr

# def calc_mon_mat_3(s_bind_A, s_bind_B):
#     nm = len(s_bind_A)
#     sig_inds = [0,1,2] # polymer, gama1, gamma2
#     M3_arr = np.zeros((len(sig_inds), len(sig_inds), len(sig_inds)), dtype= "object")
#     for a1, a2, a3 in product(sig_inds, repeat=3):
#         # print([a1, a2, a3])
#         M3_arr[a1][a2][a3] = calc_single_monomer_matrix_3(s_bind_A, s_bind_B, [a1, a2, a3])
#     return M3_arr

# def calc_mon_mat_4(s_bind_A, s_bind_B):
#     nm = len(s_bind_A)
#     sig_inds = [0,1,2] # polymer, gama1, gamma2
#     M4_arr = np.zeros((len(sig_inds), len(sig_inds), len(sig_inds), len(sig_inds)), dtype= "object")
#     for a1, a2, a3, a4 in product(sig_inds, repeat=4):
#         M4_arr[a1][a2][a3][a4] = calc_single_monomer_matrix_4(s_bind_A, s_bind_B, [a1, a2, a3, a4])
#     return M4_arr


# def s3wlc_zeroq3(chrom, K1):
    # NOT needed- can just calc with k_3 = 0
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
def calc_sf3(psol, M3_arr, k_vec, k_vec_2, competitive):
    # for a gaussian chain of M monomers, each of length N_m
    # calculates s3 matrix at a single k
        
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    M = psol.M
    N_m = psol.N_m
    b = psol.b
    N = psol.N
    alpha = psol.alpha

    # if np.linalg.norm(k_vec[0] + k_vec_2[0]) < 1e-5:
    #     return s3wlc_zeroq3(chrom, k_vec, 
    if competitive:
        # explicit competitive binding- sig_AB not considered
        sig_inds = [0,1,2] # O, gamma1, gamma2
    else:
        sig_inds = [0,1,2,3] # O, gamma1, gamma2, gamma1&gamma2

    # M = np.shape(M2_AA)[0]
    # nk = len(k_vec)
    # N = M*N_m

    # sig_inds = [0,1,2,3] #polymer, prot1, prot2, solv
    S3_arr = np.zeros((len(sig_inds)+1,len(sig_inds)+1,len(sig_inds)+1)) #polymer, prot1, prot2

    grid = np.indices((M,M,M))
    j1 = grid[0]
    j2 = grid[1]
    j3 = grid[2]
    
    
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
#        then sum over all indices. Need to keep track of js so that appropriate multiplications with cross corr matrix M3        
        C = np.zeros((M,M,M))

        for cse in case_arr:
            kA, kB = cse[0]
            ordered_js = cse[1]
            
            xm_A = (1/6) * N_m * b**2 * np.linalg.norm(kA)**2
            xm_B = (1/6) * N_m * b**2 * np.linalg.norm(kB)**2
            
            C = calc_case_s3(C, xm_A, xm_B, ordered_js)

        solvent_index = sig_inds[-1] + 1 # solvent index is always the last one

        for a1, a2, a3 in product(sig_inds+[solvent_index], repeat=3):
            if (a1 == a2 == a3 == solvent_index): #at S^{(3)}_{SSS}
                S3_arr[a1][a2][a3] += alpha
            elif (a1 == solvent_index or a2 == solvent_index or a3 == solvent_index): #at S^{(3)}_{Sxx}
                S3_arr [a1][a2][a3] += 0
            else:
                S3_arr[a1][a2][a3] += np.sum((1/M**3) * M3_arr[a1][a2][a3] * C)*(N**3)
    
    return S3_arr



def calc_sf4(psol, M4_arr, k_vec, k_vec_2, k_vec_3, competitive):
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    M = psol.M
    N = psol.N
    N_m = psol.N_m
    b = psol.b
    alpha = psol.alpha
    
    # nk = len(k_vec)
    # N = M*N_m
    # sig_inds = [0,1,2,3] #polymer, prot1, prot2, solv

    if competitive:
        # explicit competitive binding- sig_AB not considered
        sig_inds = [0,1,2] # O, gamma1, gamma2
    else:
        sig_inds = [0,1,2,3] # O, gamma1, gamma2, gamma1&gamma2

    S4_arr = np.zeros((len(sig_inds)+1,len(sig_inds)+1,len(sig_inds)+1,len(sig_inds)+1)) 

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
            
        solvent_index = sig_inds[-1] + 1 # solvent index is always the last one

        for a1, a2, a3, a4 in product(sig_inds+[solvent_index], repeat=4):
            if (a1 == a2 == a3 == a4 == solvent_index): #at S^{(4)}_{SSSS}
                S4_arr[a1][a2][a3][a4] += alpha
            elif (a1 == solvent_index or a2 == solvent_index or a3 == solvent_index or a4 == solvent_index): #at S^{(4)}_{Sxxx}
                S4_arr[a1][a2][a3][a4] += 0
            else:
                S4_arr[a1][a2][a3][a4] += np.sum((1/M**4) * M4_arr[a1][a2][a3][a4] * C)*(N**4) 

    return S4_arr



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





















