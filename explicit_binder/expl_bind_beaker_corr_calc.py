# import numpy as np
from expl_bind_beaker_binding_calc import *
from expl_bind_beaker_s3_integrals import *
from expl_bind_beaker_s4_integrals import *


# published theory- each polymer-based sf had an identical sf integral, 
#   only with different binding state correlation average. 
#   this meant that could calculate all nth order sfs identically, then just
#   multiply by the appropriate binding average to get al sfs (AA, AB, etc)

# NOW- multiple differnt forms of the strucutre factor integral and binding correlation
#   cannot just have a single sf integral for n-order strucutre factor

# # ds normalization - BAD!!
# def calc_sf2(psol, corrs, phis, k, plotting=False):
#     v_P = psol.v_p
#     N_P = psol.N_P
#     b_P = psol.b_P
#     v_A = psol.v_A
#     N_A = psol.N_A
#     b_A = psol.b_A
#     v_B = psol.v_B
#     N_B = psol.N_B
#     b_B = psol.b_B
#     M = psol.M
#     bs_per_M = psol.bs_per_M
#     # solv_cons = psol.solv_cons
#     # phi_p = psol.phi_p
#     phi_p, phi_Au, phi_Bu, phi_s = phis

#     sA, sB = corrs
#     sAsA = np.outer(sA, sA)
#     sBsB = np.outer(sB, sB)
#     sAsB = np.outer(sA, sB)

#     k = np.linalg.norm(k)

#     x_p = (1/6)*N_P*b_P**2*k**2
#     x_A = (1/6)*N_A*b_A**2*k**2
#     x_B = (1/6)*N_B*b_B**2*k**2
#     x_del = (1/6)*(N_P/(M-1))*b_P**2*k**2
#     grid = np.indices((M,M))
#     j1 = grid[0]
#     j2 = grid[1]
#     ds = 1# N_P / M # M normaliztion
    
#     S_PP = ((2/x_p**2)*(x_p + np.exp(-x_p) - 1))#[0]
    
#     S_AuAu = ((2/x_A**2)*(x_A + np.exp(-x_A) - 1))#[0]
#     # print("SAuAu: ", S_AuAu)
#     S_BuBu = ((2/x_B**2)*(x_B + np.exp(-x_B) - 1))#[0]


#     S_AA = 0
#     C = np.zeros((M,M))
#     # diagonal
#     index = (j1 == j2)
#     integral =  ((2/x_A**2)*(x_A + np.exp(-x_A) - 1))#[0]
#     corr = sA
#     C[np.where((index) != 0)] += corr * integral
# #     print("removed diag")
#     #off diagonal
#     index = (j2 != j1)
#     delta = np.abs(j1 - j2)
#     integral = (1/x_A)**2 * (1- np.exp(-x_A))**2 * np.exp(-x_del*delta)
#     corr = sAsA
#     C[np.where((index) != 0)] += corr[np.where((index) != 0)] * integral[np.where((index) != 0)]

#     S_AA = ds**2 *np.sum(C)
#     # print("S_AAbound: ", S_AA)

#     S_BB = 0
#     C = np.zeros((M,M))
#     # diagonal
#     index = (j1 == j2)
#     integral =  (2/x_B**2)*(x_B + np.exp(-x_B) - 1)
#     corr = sB
#     C[np.where((index) != 0)] += corr * integral
# #     print("removed diag")
#     #off diagonal
#     index = (j2 != j1)
#     delta = np.abs(j1 - j2)
#     integral = (1/x_B)**2 * (1- np.exp(-x_B))**2 * np.exp(-x_del*delta)
#     corr = sBsB
#     C[np.where((index) != 0)] += corr[np.where((index) != 0)] * integral[np.where((index) != 0)]
#     S_BB = ds**2 * np.sum(C)


#     S_AB = 0
#     C = np.zeros((M,M))
#     #off diagonal
#     index = (j2 != j1)
#     delta = np.abs(j1 - j2)
#     integral = (1/x_B)*(1/x_A) * (1- np.exp(-x_B)) * (1- np.exp(-x_A)) * np.exp(-x_del*delta)
#     corr = sAsB
#     C[np.where((index) != 0)] +=  corr[np.where((index) != 0)] * integral[np.where((index) != 0)]
#     S_AB = ds**2 * np.sum(C)


#     S_AP = 0
#     C = np.zeros((M))
#     j1_arr = np.arange(1, M+1)
#     integral = (1/x_p)*(1/x_A) * (1- np.exp(-x_A)) * \
#           (2 - np.exp(-x_del*(j1_arr-1)) - np.exp(-x_p + x_del*(j1_arr-1)))
#     corr = sA
#     C = corr * integral
#     S_AP = ds**1 * np.sum(C)


#     S_BP = 0
#     C = np.zeros((M))
#     j1_arr = np.arange(1, M+1)
#     integral = (1/x_p)*(1/x_B) * (1- np.exp(-x_B)) * \
#           (2 - np.exp(-x_del*(j1_arr-1)) - np.exp(-x_p + x_del*(j1_arr-1)))
#     corr = sB
#     C = corr * integral
#     S_BP = ds**1 * np.sum(C)

#     S_ss = phi_s

#     # P A_bound B_bound A_unbound B_unbound S
#     # constants: np from z_P; Ns from sf definiton, V_sys form E density
#     # assume v_P = v_A = v_B

#     # apply sadle point result, but only to unbound sfs
#     # bound and unbound COMBINED
#     # AP: n_p N_A N_P / V_sys = N_A phi_p
#     # AA: n_p N_A N_A / V_sys = n_p N_A N_A N_P / V_sys N_P = phi_p N_A^2 / N_P
#     # AuAu: n_Au * N_A**2 / V_sys = phi_Au * N_A
#     S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
#           [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P +  S_AuAu*phi_Au*N_A, S_AB*(phi_p*N_A*N_B)/N_P, 0],\
#           [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P + S_BuBu*phi_Bu*N_B, 0],\
#           [0, 0, 0, S_ss]]
    
#     if plotting == True:
#         # TO PLOT S2s, uncomment this code:

#         # separated bound and unbound components- 6 components total
#         # P Ab Au Bb Bu S
#         S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, 0, S_BP*phi_p*N_B, 0, 0,], \
#             [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P,0 , S_AB*(phi_p*N_A*N_B)/N_P, 0, 0],\
#             [0,0,S_AuAu*phi_Au*N_A,0,0,0],\
#             [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, 0,  S_BB*(phi_p*N_B**2)/N_P, 0, 0],\
#             [0,0,0,0,S_BuBu*phi_Bu*N_B,0],\
#             [0, 0, 0, 0, 0, S_ss]]
#         # print("S2 boud and bound separate")
#     return S2

# 1/M NORMALIZATION- GOOD
def calc_sf2(psol, corrs, phis, k, plotting=False):
    v_P = psol.v_p
    N_P = psol.N_P
    b_P = psol.b_P
    v_A = psol.v_A
    N_A = psol.N_A
    b_A = psol.b_A
    v_B = psol.v_B
    N_B = psol.N_B
    b_B = psol.b_B
    M = psol.M
    bs_per_M = psol.bs_per_M
    # solv_cons = psol.solv_cons
    # phi_p = psol.phi_p
    phi_p, phi_Au, phi_Bu, phi_s = phis

    sA, sB = corrs
    sAsA = np.outer(sA, sA)
    sBsB = np.outer(sB, sB)
    sAsB = np.outer(sA, sB)

    k = np.linalg.norm(k)

    x_p = (1/6)*N_P*b_P**2*k**2
    x_A = (1/6)*N_A*b_A**2*k**2
    x_B = (1/6)*N_B*b_B**2*k**2
    x_del = (1/6)*(N_P/(M-1))*b_P**2*k**2
    grid = np.indices((M,M))
    j1 = grid[0]
    j2 = grid[1]
    # ds = N_P / M # M normaliztion
    
    S_PP = ((2/x_p**2)*(x_p + np.exp(-x_p) - 1))#[0]
    
    S_AuAu = ((2/x_A**2)*(x_A + np.exp(-x_A) - 1))#[0]
    # print("SAuAu: ", S_AuAu)
    S_BuBu = ((2/x_B**2)*(x_B + np.exp(-x_B) - 1))#[0]


    S_AA = 0
    C = np.zeros((M,M))
    # diagonal
    index = (j1 == j2)
    integral =  ((2/x_A**2)*(x_A + np.exp(-x_A) - 1))#[0]
    # print(integral)
    corr = sA
    C[np.where((index) != 0)] += (1/M)**2 * corr * integral
#     print("removed diag")
    #off diagonal
    index = (j2 != j1)
    delta = np.abs(j1 - j2)
    integral = (1/x_A)**2 * (1- np.exp(-x_A))**2 * np.exp(-x_del*delta)
    corr = sAsA
    C[np.where((index) != 0)] += (1/M)**2 * corr[np.where((index) != 0)] * integral[np.where((index) != 0)]

    S_AA = np.sum(C)
    # print("S_AAbound: ", S_AA)

    S_BB = 0
    C = np.zeros((M,M))
    # diagonal
    index = (j1 == j2)
    integral =  (2/x_B**2)*(x_B + np.exp(-x_B) - 1)
    corr = sB
    C[np.where((index) != 0)] += (1/M)**2 * corr * integral
#     print("removed diag")
    #off diagonal
    index = (j2 != j1)
    delta = np.abs(j1 - j2)
    integral = (1/x_B)**2 * (1- np.exp(-x_B))**2 * np.exp(-x_del*delta)
    corr = sBsB
    C[np.where((index) != 0)] += (1/M)**2 * corr[np.where((index) != 0)] * integral[np.where((index) != 0)]
    S_BB = np.sum(C)


    S_AB = 0
    C = np.zeros((M,M))
    #off diagonal
    index = (j2 != j1)
    delta = np.abs(j1 - j2)
    integral = (1/x_B)*(1/x_A) * (1- np.exp(-x_B)) * (1- np.exp(-x_A)) * np.exp(-x_del*delta)
    corr = sAsB
    C[np.where((index) != 0)] += (1/M)**2 * corr[np.where((index) != 0)] * integral[np.where((index) != 0)]
    S_AB = np.sum(C)


    S_AP = 0
    C = np.zeros((M))
    j1_arr = np.arange(1, M+1)
    integral = (1/x_p)*(1/x_A) * (1- np.exp(-x_A)) * \
          (2 - np.exp(-x_del*(j1_arr-1)) - np.exp(-x_p + x_del*(j1_arr-1)))
    corr = sA
    C = corr * integral
    S_AP = (1/M)**1 * np.sum(C)


    S_BP = 0
    C = np.zeros((M))
    j1_arr = np.arange(1, M+1)
    integral = (1/x_p)*(1/x_B) * (1- np.exp(-x_B)) * \
          (2 - np.exp(-x_del*(j1_arr-1)) - np.exp(-x_p + x_del*(j1_arr-1)))
    corr = sB
    C = corr * integral
    S_BP = (1/M)**1 * np.sum(C)

    S_ss = phi_s

    # P A_bound B_bound A_unbound B_unbound S
    # constants: np from z_P; Ns from sf definiton, V_sys form E density
    # assume v_P = v_A = v_B

    # apply sadle point result, but only to unbound sfs
    # bound and unbound COMBINED
    # AP: n_p N_A N_P / V_sys = N_A phi_p
    # AA: n_p N_A N_A / V_sys = n_p N_A N_A N_P / V_sys N_P = phi_p N_A^2 / N_P
    # AuAu: n_Au * N_A**2 / V_sys = phi_Au * N_A
    S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
          [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P +  S_AuAu*phi_Au*N_A, S_AB*(phi_p*N_A*N_B)/N_P, 0],\
          [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P + S_BuBu*phi_Bu*N_B, 0],\
          [0, 0, 0, S_ss]]
    
    if plotting == True:
        # TO PLOT S2s, uncomment this code:

        # separated bound and unbound components- 6 components total
        # P Ab Au Bb Bu S
        S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, 0, S_BP*phi_p*N_B, 0, 0,], \
            [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P,0 , S_AB*(phi_p*N_A*N_B)/N_P, 0, 0],\
            [0,0,S_AuAu*phi_Au*N_A,0,0,0],\
            [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, 0,  S_BB*(phi_p*N_B**2)/N_P, 0, 0],\
            [0,0,0,0,S_BuBu*phi_Bu*N_B,0],\
            [0, 0, 0, 0, 0, S_ss]]
        # print("S2 boud and bound separate")
    return S2

# define a set of integral functions (e.g. S_AAA^(3,1)), and create a function that, when given appropriate
# integral functions for (3,1), (3,2), and (3,3) and the k and b and corr identities, returns the sf3




def calc_sf3(psol, corrs, phis, k1, k2, k12):
    """
    Compute third-order structure factors for a single (k1,k2,k12).
    Returns S3_arr with axes [species1, species2, species3] where
      0 -> P, 1 -> A, 2 -> B
    """
    v_P = psol.v_p
    N_P = psol.N_P
    b_P = psol.b_P
    v_A = psol.v_A
    N_A = psol.N_A
    b_A = psol.b_A
    v_B = psol.v_B
    N_B = psol.N_B
    b_B = psol.b_B
    M = psol.M
    # solv_cons = psol.solv_cons
    # phi_Au, phi_Bu = phius
    # phi_p = psol.phi_p
    phi_p, phi_Au, phi_Bu, phi_s = phis
    solv_cons = phi_s

    # phi_A = psol.phi_A
    # phi_B = psol.phi_B

    # phi_Ab = psol.phi_Ab
    # phi_Au = psol.phi_Au
    # phi_Bb = psol.phi_Bb
    # phi_Bu = psol.phi_Bu
    k1 = np.linalg.norm(k1)
    k2 = np.linalg.norm(k2)
    k12 = np.linalg.norm(k12)


    sA, sB = corrs
    # correlations
    sP = np.ones_like(sA)
    sAsAsA = np.einsum("i,j,k->ijk", sA, sA, sA)
    sAsAsP = np.einsum("i,j,k->ijk", sA, sA, sP)
    sAsPsP = np.einsum("i,j,k->ijk", sA, sP, sP)
    sAsBsP = np.einsum("i,j,k->ijk", sA, sB, sP)

    sBsBsB = np.einsum("i,j,k->ijk", sB, sB, sB)
    sBsBsP = np.einsum("i,j,k->ijk", sB, sB, sP)
    sBsPsP = np.einsum("i,j,k->ijk", sB, sP, sP)

    sAsAsB = np.einsum("i,j,k->ijk", sA, sA, sB)
    sAsBsB = np.einsum("i,j,k->ijk", sA, sB, sB)

    # monomer index grid
    j1, j2, j3 = np.indices((M, M, M))

    # case permutations
    case1     = [[k12, k1],  [j3, j2, j1]]
    case1_deg = [[k1,  k12], [j1, j2, j3]]
    case2     = [[k2,  k12], [j2, j1, j3]]
    case2_deg = [[k12, k2],  [j3, j1, j2]]
    case3     = [[-k2, k1],  [j2, j3, j1]]
    case3_deg = [[k1, -k2],  [j1, j3, j2]]
    case_arr  = [case1, case2, case3, case1_deg, case2_deg, case3_deg]

    #prefactors. assuming N_A = N_B
    # EACH has a divide by N_alpha^3 bc did not include N factors in x definitons in integrals
    #ppp: np N_p^3 / V_sys = N_P^2 phi_P
    ppp_pre = N_P**2 * phi_p / N_P**3 # divide bc did not include N factors in x definitons in integrals
    #ppa: np N_p^2 N_A / V_sys = N_P N_A phi_P
    ppa_pre = N_P * N_A * phi_p / (N_P**2 * N_A) 
    #paa: np N_p N_A^2 / V_sys = N_A^2 phi_P
    paa_pre = N_A**2 * phi_p  / (N_P * N_A**2) 
    # BOUND aaa: np N_A^3 / V_sys = np N_A^3 N_P / (v_sys N_P) = N_A^3 phi_P / N_P
    aaa_pre = (N_A**3 * phi_p) / (N_P * N_A**3)

    # UNBOUND
    # aaaU_pre = phi_Au * N_A**3 / (N_A**3)
    # bbbU_pre = phi_Bu * N_B**3 / (N_B**3)
    # same as above but introduce n_alpha^U to the AuAu along w saddle point result
    # AuAu =  n_au * N_A**3 / V_sys = phi_Au * N_A**2 
    aaaU_pre = phi_Au * N_A**2 / (N_A**3)
    bbbU_pre = phi_Bu * N_B**2 / (N_B**3)

    S3_arr = np.zeros((4, 4, 4), dtype=float)
    S3_Au = 0
    S3_Bu = 0

    # helper for masked sums
    def masked_sum(corr, I, mask):
        m = (mask != 0)
        if not np.any(m):
            return 0.0
        return np.sum(corr[m] * I[m])

    # loop over cases and accumulate into S3_arr
    for cse in case_arr:
        kA, kB = cse[0]
        ordered_js = cse[1]

        # masks
        index1 = (ordered_js[0] == ordered_js[1]) * (ordered_js[0] > ordered_js[-1])
        index2 = (ordered_js[2] > ordered_js[1]) * (ordered_js[1] > ordered_js[0])

        # PPP  -> [0,0,0]
        # S3_arr[0,0,0] += np.sum(S_AAA31(kA, kB, b_P, N_P))
        S3_arr[0,0,0] += S_AAA31(kA, kB, b_P, N_P)

        S3_Au += S_AAA31(kA, kB, b_A, N_A)
        S3_Bu += S_AAA31(kA, kB, b_A, N_A)

        # S3_arr
        # AAA -> [1,1,1]
        S3_arr[1,1,1] += np.sum(sA * S_AAA31(kA, kB, b_A, N_A))
        I = S_AAA32(kA, kB, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr1 = sAsAsA / sA[ordered_js[0]]   # careful: ordered_js[0] indexes into sA
        S3_arr[1,1,1] += masked_sum(corr1, 2*I, index1)
        I = S_AAA33(kA, kB, -kA-kB, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        corr2 = sAsAsA
        S3_arr[1,1,1] += masked_sum(corr2, I, index2)

        # AAB -> [1,1,2]
        I = S_AAA32(kA, kB, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sAsAsB / sA[ordered_js[0]]
        S3_arr[1,1,2] += masked_sum(corr, I, index1)
        I = S_AAA33(kA, kB, -kA-kB, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[1,1,2] += masked_sum(sAsAsB, I, index2)


        # ABB -> [1,2,2]
        I = S_AAA32(kA, kB, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sAsBsB / sB[ordered_js[2]]
        S3_arr[1,2,2] += masked_sum(corr, I, index1)
        I = S_AAA33(kA, kB, -kA-kB, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[1,2,2] += masked_sum(sAsBsB, I, index2)


        # AAP -> [1,1,0]
        S3_arr[1,1,0] += np.sum(sA * S_AAP31(kA, kB, b_A, N_A))
        I = S_AAP32(kA, kB, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sAsAsP / sA[ordered_js[0]]
        S3_arr[1,1,0] += masked_sum(corr, I, index1)
        #already handled below
        # I = S_APA32(kA, kB, b_A, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        # corr = sAsAsP
        # S3_arr[1,1,0] += masked_sum(corr, I, index1)
        I = S_AAP33(kA, kB, -kA-kB, b_A, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[1,1,0] += masked_sum(sAsAsP, I, index2)


        # APA -> [1,0,1]
        S3_arr[1,0,1] += np.sum(sA * S_AAP31(kA, kB, b_A, N_A))
        I = S_APA32(kA, kB, b_A, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sAsAsP
        S3_arr[1,0,1] += masked_sum(corr, 2*I, index1)
        # handling As on different monomers
        S3_arr[1,1,0] += masked_sum(corr, I, index1)
        I = S_APA33(kA, kB, -kA-kB, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[1,0,1] += masked_sum(sAsAsP, I, index2)


        # BBB -> [2,2,2]
        S3_arr[2,2,2] += np.sum(sB * S_AAA31(kA, kB, b_B, N_B))
        I = S_AAA32(kA, kB, b_B, b_P, N_B, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sBsBsB / sB[ordered_js[0]]
        S3_arr[2,2,2] += masked_sum(corr, 2*I, index1)
        I = S_AAA33(kA, kB, -kA-kB, b_B, b_P, N_B, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[2,2,2] += masked_sum(sBsBsB, I, index2)


        # BBP -> [2,2,0]
        S3_arr[2,2,0] += np.sum(sB * S_AAP31(kA, kB, b_B, N_B))
        I = S_AAP32(kA, kB, b_B, b_P, N_B, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sBsBsP / sB[ordered_js[0]]
        S3_arr[2,2,0] += masked_sum(corr, I, index1)
        I = S_AAP33(kA, kB, -kA-kB, b_B, b_B, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[2,2,0] += masked_sum(sBsBsP, I, index2)


        # BPB -> [2,0,2]
        S3_arr[2,0,2] += np.sum(sB * S_AAP31(kA, kB, b_B, N_B))
        I = S_APA32(kA, kB, b_B, b_B, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sBsBsP
        S3_arr[2,0,2] += masked_sum(corr, 2*I, index1)
        # also add to BBP
        S3_arr[2,2,0] += masked_sum(corr, I, index1)
        I = S_APA33(kA, kB, -kA-kB, b_B, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[2,0,2] += masked_sum(sBsBsP, I, index2)


        # ABP -> [1,2,0]
        I = S_APA32(kA, kB, b_A, b_B, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sAsBsP
        S3_arr[1,2,0] += masked_sum(corr, I, index1)
        I = S_AAP33(kA, kB, -kA-kB, b_A, b_B, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[1,2,0] += masked_sum(sAsBsP, I, index2)


        # BPA -> [2,0,1]
        I = S_APA32(kA, kB, b_A, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sAsBsP
        S3_arr[2,0,1] += masked_sum(corr, 2*I, index1)
        I = S_APA33(kA, kB, -kA-kB, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[2,0,1] += masked_sum(sAsBsP, I, index2)


        # APP -> [1,0,0] (only depends on single-k part; averaged over cases in original)
        S3_arr[1,0,0] += np.sum(sA * S_APP31(kA, b_A, N_A)) / len(case_arr)
        I = S_APP32(kA, kB, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        S3_arr[1,0,0] += masked_sum(sAsPsP, I, index1)
        I = S_APP33(kA, kB, -kA-kB, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[1,0,0] += masked_sum(sAsPsP, I, index2)


        # BPP -> [2,0,0]
        S3_arr[2,0,0] += np.sum(sB * S_APP31(kA, b_B, N_B)) / len(case_arr)
        I = S_APP32(kA, kB, b_B, b_P, N_B, N_P, M, ordered_js[0], ordered_js[-1])
        S3_arr[2,0,0] += masked_sum(sBsPsP, I, index1)
        I = S_APP33(kA, kB, -kA-kB, b_B, b_P, N_B, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[2,0,0] += masked_sum(sBsPsP, I, index2)

    S3_arr[0,0,0] *= ppp_pre
    S3_Au *= aaaU_pre
    S3_Bu *= bbbU_pre
    S3_arr[1,1,1] *= aaa_pre

    S3_arr[1,1,2] *= aaa_pre
    S3_arr[1,2,2] *= aaa_pre
    S3_arr[1,1,0] *= paa_pre
    S3_arr[1,0,1] *= paa_pre
    S3_arr[2,2,2] *= aaa_pre
    S3_arr[2,2,0] *= paa_pre
    S3_arr[2,0,2] *= paa_pre
    S3_arr[1,2,0] *= paa_pre
    S3_arr[2,0,1] *= paa_pre
    S3_arr[1,0,0] *= ppa_pre
    S3_arr[2,0,0] *= ppa_pre
    # PAA = AAP
    S3_arr[0,1,1] = S3_arr[1,1,0]

    # PBB = BBP
    S3_arr[0,2,2] = S3_arr[2,2,0]

    # BAP = PAB = PBA = ABP
    # BAP [2,1,0], PAB [0,1,2], PBA [0,2,1], ABP [1,2,0]
    S3_arr[2,1,0] = S3_arr[0,1,2] = S3_arr[0,2,1] = S3_arr[1,2,0]

    # APB = BPA
    # APB [1,0,2], BPA [2,0,1]
    S3_arr[1,0,2] = S3_arr[2,0,1]

    # PPA = PAP = APP
    # PPA [0,0,1], PAP [0,1,0], APP [1,0,0]
    S3_arr[0,0,1] = S3_arr[0,1,0] = S3_arr[1,0,0]

    # PPB = PBP = BPP
    # PPB [0,0,2], PBP [0,2,0], BPP [2,0,0]
    S3_arr[0,0,2] = S3_arr[0,2,0] = S3_arr[2,0,0]

    # ABA = BAA = AAB
    # ABA [1,2,1], BAA [2,1,1], AAB [1,1,2]
    S3_arr[1,2,1] = S3_arr[2,1,1] = S3_arr[1,1,2]

    # BAB = BBA = ABB
    # BAB [2,1,2], BBA [2,2,1], ABB [1,2,2]
    S3_arr[2,1,2] = S3_arr[2,2,1] = S3_arr[1,2,2]

    # ADD UNBOUND CONTRIBUTION
    S3_arr[1,1,1] += S3_Au
    S3_arr[2,2,2] += S3_Bu

    S3_arr[3,3,3] += solv_cons
    return S3_arr



def calc_sf4(psol, corrs, phis, k1, k2, k3, k123):
    # [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    v_P = psol.v_p
    N_P = psol.N_P
    b_P = psol.b_P
    v_A = psol.v_A
    N_A = psol.N_A
    b_A = psol.b_A
    v_B = psol.v_B
    N_B = psol.N_B
    b_B = psol.b_B
    M = psol.M

    phi_p, phi_Au, phi_Bu, phi_s = phis
    solv_cons = phi_s

    k1 = np.linalg.norm(k1)
    k2 = np.linalg.norm(k2)
    k3 = np.linalg.norm(k3)
    k123 = np.linalg.norm(k123)

    sA, sB = corrs
    sP = np.ones_like(sA)
    sAsA = np.outer(sA, sA)
    sBsB = np.outer(sB, sB)
    sAsB = np.outer(sA, sB)

    sAsAsAsA = np.einsum("i,j,k,l->ijkl", sA, sA, sA, sA)
    sAsAsAsP = np.einsum("i,j,k,l->ijkl", sA, sA, sA, sP)
    sAsAsPsP = np.einsum("i,j,k,l->ijkl", sA, sA, sP, sP)
    sAsPsPsP = np.einsum("i,j,k,l->ijkl", sA, sP, sP, sP)

    sAsAsAsB = np.einsum("i,j,k,l->ijkl", sA, sA, sA, sB)
    sAsAsBsB = np.einsum("i,j,k,l->ijkl", sA, sA, sB, sB)
    sAsBsBsB = np.einsum("i,j,k,l->ijkl", sA, sB, sB, sB)
    sBsBsBsB = np.einsum("i,j,k,l->ijkl", sB, sB, sB, sB)

    sAsAsAsP = np.einsum("i,j,k,l->ijkl", sA, sA, sA, sP)
    sAsAsPsP = np.einsum("i,j,k,l->ijkl", sA, sA, sP, sP)
    sAsPsPsP = np.einsum("i,j,k,l->ijkl", sA, sP, sP, sP)

    sAsAsBsP = np.einsum("i,j,k,l->ijkl", sA, sA, sB, sP)
    sAsBsBsP = np.einsum("i,j,k,l->ijkl", sA, sB, sB, sP)
    sBsBsBsP = np.einsum("i,j,k,l->ijkl", sB, sB, sB, sP)
    sAsBsPsP = np.einsum("i,j,k,l->ijkl", sA, sB, sP, sP)

    sBsBsBsP = np.einsum("i,j,k,l->ijkl", sB, sB, sB, sP)
    sBsBsPsP = np.einsum("i,j,k,l->ijkl", sB, sB, sP, sP)
    sBsPsPsP = np.einsum("i,j,k,l->ijkl", sB, sP, sP, sP)


    grid = np.indices((M,M,M,M))
    j1 = grid[0]
    j2 = grid[1]
    j3 = grid[2]
    j4 = grid[3]

    # k2 = k_vec_2[i]
    # k3 = k_vec_3[i]
    k12 = k1 + k2
    k13 = k1 + k3
    k23 = k2 + k3
    # k123 = k1 + k2 + k3
    
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
    

    #prefactors. assuming N_A = N_B

    #pppp: np N_p^4 / V_sys = N_P^3 phi_P
    pppp_pre = N_P**3 * phi_p / N_P**4 # x definition doesnt have N facotrs- and for P need this regardless
    #pppa: np N_p^3 N_A / V_sys = N_P^2 N_A phi_P
    pppa_pre = N_P**2 * N_A * phi_p / (N_P**3 * N_A)
    #ppaa: np N_p^2 N_A^2 / V_sys = N_A^2 N_P phi_P
    ppaa_pre = N_P * N_A**2 * phi_p / (N_P**2 * N_A**2)
    #paaa: np N_p N_A^3 / V_sys = N_A^3 phi_P
    paaa_pre = N_A**3 * phi_p / (N_A**3 * N_P)

    # BOUND aaaa: np N_A^4 / V_sys = np N_A^4 N_P / (v_sys N_P) = N_A^4 phi_P / N_P
    aaaa_pre = (N_A**4 * phi_p) / (N_P * N_A**4)

    # UNBOUND
    # aaaaU_pre = phi_Au * N_A**4 / N_A**4
    # bbbbU_pre = phi_Bu * N_B**4 / N_A**4
    # same as above but introduce n_alpha^U to the AuAu along w saddle point result
    # AuAu = n_au * N_A**4 / V_sys = phi_Au**2 * N_A**3 
    aaaaU_pre = phi_Au * N_A**3 / (N_A**4)
    bbbbU_pre = phi_Bu * N_B**3 / (N_B**4)

    S4_arr = np.zeros((4,4,4,4)) 
    S4_Au = 0
    S4_Bu = 0


    for cse in case_arr:
        kA, kB, kC = cse[0]
        ordered_js = cse[1]
        
        # AAAA
        S4_arr[1,1,1,1] += np.sum(sA*S_AAAA41(kA, kB,kC, -kA-kB-kC, b_A, N_A))
        S4_arr[2,2,2,2] += np.sum(sB*S_AAAA41(kA, kB,kC, -kA-kB-kC, b_A, N_A))

        # S4_arr[0,0,0,0] += np.sum(S_AAAA41(kA, kB,kC, -kA-kB-kC, b_P, N_P))
        # S4_Au += np.sum(S_AAAA41(kA, kB,kC, -kA-kB-kC, b_A, N_A))
        # S4_Bu += np.sum(S_AAAA41(kA, kB,kC, -kA-kB-kC, b_B, N_B))
        S4_arr[0,0,0,0] += S_AAAA41(kA, kB,kC, -kA-kB-kC, b_P, N_P)
        S4_Au += S_AAAA41(kA, kB,kC, -kA-kB-kC, b_A, N_A)
        S4_Bu += S_AAAA41(kA, kB,kC, -kA-kB-kC, b_B, N_B)

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] == ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAAA42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsAsAsA / (sA[ordered_js[1]] * sA[ordered_js[2]])
        S4_arr[1,1,1,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsBsBsB / (sB[ordered_js[1]] * sB[ordered_js[2]])
        S4_arr[2,2,2,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsAsAsB / (sA[ordered_js[1]] * sA[ordered_js[2]])
        S4_arr[1,1,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsB / (sB[ordered_js[1]] * sB[ordered_js[2]])
        S4_arr[1,2,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAAA43(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsAsA / (sA[ordered_js[1]])
        S4_arr[1,1,1,1] += np.sum(corr[np.where(index != 0)]*3*I[np.where(index != 0)])
        corr = sBsBsBsB / (sB[ordered_js[1]])
        S4_arr[2,2,2,2] += np.sum(corr[np.where(index != 0)]*3*I[np.where(index != 0)])
        corr = sAsAsAsB / (sA[ordered_js[1]])
        S4_arr[1,1,1,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsAsAsB / (sA[ordered_js[1]])
        S4_arr[1,1,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsB / (sB[ordered_js[1]])
        S4_arr[1,2,2,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsBsBsB / (sB[ordered_js[1]])
        S4_arr[2,1,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsB / (sA[ordered_js[1]])
        S4_arr[1,1,2,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])

        index = (ordered_js[0] < ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAAA44(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[1], ordered_js[2], ordered_js[3])
        corr = sAsAsAsA 
        S4_arr[1,1,1,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsBsB 
        S4_arr[2,2,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsAsB 
        S4_arr[1,1,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsAsB
        S4_arr[1,1,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsB
        S4_arr[1,2,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsB
        S4_arr[2,1,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsB 
        S4_arr[1,1,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsB 
        S4_arr[1,2,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])


        # AAAP
        S4_arr[1,1,1,0] += np.sum(sA*S_AAAP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))
        S4_arr[2,2,2,0] += np.sum(sB*S_AAAP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] == ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAAP42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsAsAsP / (sA[ordered_js[1]] * sA[ordered_js[2]])
        S4_arr[1,1,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsBsP / (sB[ordered_js[1]] * sB[ordered_js[2]])
        S4_arr[2,2,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        I = S_AAPA42_AAPtriple_Aisolated(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsAsAsP / (sA[ordered_js[1]])
        S4_arr[1,1,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsBsP / (sB[ordered_js[1]])
        S4_arr[2,2,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP / (sA[ordered_js[1]])
        S4_arr[0,1,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP / (sB[ordered_js[1]])
        S4_arr[0,2,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAAP43(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsAsP / (sA[ordered_js[1]])
        S4_arr[1,1,1,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsBsBsP / (sB[ordered_js[1]])
        S4_arr[2,2,2,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsAsBsP / (sA[ordered_js[1]])
        S4_arr[1,1,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP / (sA[ordered_js[1]])
        S4_arr[0,1,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #1,2,1
        corr = sAsAsBsP 
        S4_arr[0,1,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #2,1,1
        corr = sAsBsBsP / (sB[ordered_js[1]])
        S4_arr[0,2,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #1,2,1
        corr = sAsBsBsP
        S4_arr[0,2,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #2,1,1
        corr = sAsBsBsP / (sB[ordered_js[1]])
        S4_arr[2,2,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        I = S_AAAP43_2(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsAsP 
        S4_arr[1,1,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsBsP 
        S4_arr[2,2,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP 
        S4_arr[1,1,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP 
        S4_arr[0,1,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #1,2,1
        corr = sAsAsBsP 
        S4_arr[0,1,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #2,1,1
        corr = sAsBsBsP 
        S4_arr[0,2,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #1,2,1
        corr = sAsBsBsP
        S4_arr[0,2,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #2,1,1
        corr = sAsBsBsP
        S4_arr[2,2,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        index = (ordered_js[0] < ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAAP44(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[1], ordered_js[2], ordered_js[3])
        corr = sAsAsAsP
        S4_arr[1,1,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsBsP 
        S4_arr[2,2,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP
        S4_arr[1,1,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP
        S4_arr[2,2,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP 
        S4_arr[0,1,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP
        S4_arr[0,2,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        # #AAPA → (1,1,0,1)
        # #BBPB → (2,2,0,2)

        S4_arr[1,1,0,1] += np.sum(sA*S_AAAP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))
        S4_arr[2,2,0,2] += np.sum(sB*S_AAAP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))

        # # TODO- should this code block exist? i dont think so, assuming must propogate forward
        # index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] == ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        # I = S_AAAP42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        # corr = sAsAsAsP / (sA[ordered_js[1]] * sA[ordered_js[2]])
        # S4_arr[1,1,0,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # corr = sBsBsBsP / (sB[ordered_js[1]] * sB[ordered_js[2]])
        # S4_arr[2,2,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # corr = sAsAsBsP / (sA[ordered_js[1]])
        # S4_arr[1,1,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # corr = sAsBsBsP / (sB[ordered_js[1]])
        # S4_arr[1,2,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        I = S_AAPA42_AAPtriple_Aisolated(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsAsAsP / (sA[ordered_js[1]])
        S4_arr[1,1,0,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsBsBsP / (sB[ordered_js[1]])
        S4_arr[2,2,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsAsBsP / (sA[ordered_js[1]])
        S4_arr[1,1,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP / (sB[ordered_js[1]])
        S4_arr[1,2,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP / (sA[ordered_js[1]])
        S4_arr[2,1,0,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP / (sB[ordered_js[1]])
        S4_arr[1,0,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAPA43(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsAsP / (sA[ordered_js[1]])
        S4_arr[1,1,0,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsBsP / (sB[ordered_js[1]])
        S4_arr[2,2,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP / (sA[ordered_js[1]])
        S4_arr[1,1,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #2,1,1
        corr = sAsBsBsP / (sB[ordered_js[1]])
        S4_arr[1,0,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #1,1,2

        # I = S_AAPA43_2(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        I = S_AAAP43_2(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsAsP
        S4_arr[1,1,0,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1, 2, 1 and 1,1,2
        corr = sBsBsBsP 
        S4_arr[2,2,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsAsBsP
        S4_arr[1,1,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 1,1,2
        corr = sAsBsBsP 
        S4_arr[1,0,2,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 2,1,1
        corr = sAsBsBsP 
        S4_arr[1,2,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 1,1,2
        corr = sAsAsBsP
        S4_arr[1,0,2,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 2,1,1
        corr = sAsAsBsP
        S4_arr[0,1,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP
        S4_arr[0,2,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP
        S4_arr[2,0,1,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsAsBsP 
        S4_arr[2,1,0,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 1,1,2

        # index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        # I = S_AAPA43(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        # corr = sAsAsAsP / (sA[ordered_js[1]])
        # S4_arr[1,1,0,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # corr = sBsBsBsP / (sB[ordered_js[1]])
        # S4_arr[2,2,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # corr = sAsAsBsP
        # S4_arr[1,1,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 2,1,1
        # corr = sAsAsBsP / (sA[ordered_js[1]])
        # S4_arr[1,1,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #1,1,2
        # corr = sAsBsBsP
        # S4_arr[1,0,2,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 2,1,1
        # corr = sAsBsBsP / (sB[ordered_js[2]])
        # S4_arr[1,0,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)]) #1,1,2
        # corr = sAsBsBsP 
        # S4_arr[1,2,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 1,1,2
        # corr = sAsAsBsP
        # S4_arr[1,0,2,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 1,1,2
        # corr = sAsAsBsP
        # S4_arr[0,1,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # corr = sAsBsBsP
        # S4_arr[0,2,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # corr = sAsBsBsP
        # S4_arr[2,0,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # corr = sAsAsBsP 
        # S4_arr[2,1,0,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)]) #1,2,1 and 1,1,2

        index = (ordered_js[0] < ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAAP44(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[1], ordered_js[2], ordered_js[3])
        corr = sAsAsAsP
        S4_arr[1,1,0,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsBsP
        S4_arr[2,2,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP
        S4_arr[1,1,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP
        S4_arr[1,0,2,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP
        S4_arr[1,2,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP
        S4_arr[1,0,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP
        S4_arr[0,1,2,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP
        S4_arr[2,0,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsBsP
        S4_arr[0,2,1,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsAsBsP 
        S4_arr[2,1,0,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # #AAPP → (1,1,0,0)
        # #BBPP → (2,2,0,0)

        S4_arr[1,1,0,0] += np.sum(sA*S_AAPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))
        S4_arr[2,2,0,0] += np.sum(sB*S_AAPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] == ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAPP42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsAsPsP / (sA[ordered_js[1]])
        S4_arr[1,1,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP / (sB[ordered_js[1]])
        S4_arr[2,2,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        I = S_APPA42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsAsPsP 
        S4_arr[1,1,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP
        S4_arr[2,2,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

      
        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAPP43(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP 
        S4_arr[1,1,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP 
        S4_arr[2,2,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        I = S_AAPP43_2(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP / (sA[ordered_js[1]])
        S4_arr[1,1,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP / (sB[ordered_js[1]])
        S4_arr[2,2,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        I = S_AAPP43_pairP(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP 
        S4_arr[1,1,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP 
        S4_arr[2,2,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        index = (ordered_js[0] < ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_AAPP44(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[1], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP
        S4_arr[1,1,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsPsP
        S4_arr[1,2,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP
        S4_arr[2,2,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        # #APPA → (1,0,0,1)
        # #BPPB → (2,0,0,2)

        S4_arr[1,0,0,1] += np.sum(sA*S_AAPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))
        S4_arr[2,0,0,2] += np.sum(sB*S_AAPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))


        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] == ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_APPA42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsAsPsP 
        S4_arr[1,0,0,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsBsPsP 
        S4_arr[2,0,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsBsPsP
        S4_arr[1,0,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsBsPsP
        S4_arr[1,0,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_APPA43(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP  # As are on different monomers
        S4_arr[1,0,0,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP  # Bs are on different monomers
        S4_arr[2,0,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsPsP
        S4_arr[1,0,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        I = S_APPA43_2(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP  # As are on different monomers
        S4_arr[1,0,0,1] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsBsPsP  # Bs are on different monomers
        S4_arr[2,0,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sAsBsPsP
        S4_arr[1,0,0,2] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        # corr = sAsBsPsP
        # S4_arr[1,0,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        index = (ordered_js[0] < ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_APPA44(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[1], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP
        S4_arr[1,0,0,1] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP
        S4_arr[2,0,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsPsP
        S4_arr[1,0,0,2] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsPsP
        S4_arr[1,0,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        # #PAAP → (0,1,1,0)
        # #PBBP → (0,2,2,0)

        S4_arr[0,1,1,0] += np.sum(sA*S_AAPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))
        S4_arr[0,2,2,0] += np.sum(sB*S_AAPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] == ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_PAAP42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsAsPsP / (sA[ordered_js[1]])
        S4_arr[0,1,1,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsBsPsP / (sB[ordered_js[1]])
        S4_arr[0,2,2,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_PAAP43_pairP(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP  # As are on different monomers
        S4_arr[0,1,1,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsBsPsP  # Bs are on different monomers
        S4_arr[0,2,2,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])

        I = S_PAAP43_pairA(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP / (sA[ordered_js[1]])  # As are on different monomers
        S4_arr[0,1,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP / (sB[ordered_js[1]])  # Bs are on different monomers
        S4_arr[0,2,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        index = (ordered_js[0] < ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_PAAP44(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[1], ordered_js[2], ordered_js[3])
        corr = sAsAsPsP
        S4_arr[0,1,1,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsBsPsP
        S4_arr[0,2,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sAsBsPsP
        S4_arr[0,1,2,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

        #APPP → (1,0,0,0)
        #BPPP → (2,0,0,0)
        
        S4_arr[1,0,0,0] += np.sum(sA*S_APPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))
        S4_arr[2,0,0,0] += np.sum(sB*S_APPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] == ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_APPP42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsPsPsP
        S4_arr[1,0,0,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsPsPsP
        S4_arr[2,0,0,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_APPP43(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsPsPsP
        S4_arr[1,0,0,0] += np.sum(corr[np.where(index != 0)]*3*I[np.where(index != 0)])
        corr = sBsPsPsP
        S4_arr[2,0,0,0] += np.sum(corr[np.where(index != 0)]*3*I[np.where(index != 0)])

        index = (ordered_js[0] < ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_APPP44(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[1], ordered_js[2], ordered_js[3])
        corr = sAsPsPsP
        S4_arr[1,0,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsPsPsP
        S4_arr[2,0,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])


        #PAPP → (0,1,0,0)
        #PBPP → (0,2,0,0)
        S4_arr[0,1,0,0] += np.sum(sA*S_APPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))
        S4_arr[0,2,0,0] += np.sum(sB*S_APPP41(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M))

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] == ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_PAPP42(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[3])
        corr = sAsPsPsP
        S4_arr[0,1,0,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])
        corr = sBsPsPsP
        S4_arr[0,2,0,0] += np.sum(corr[np.where(index != 0)]*2*I[np.where(index != 0)])

        index = (ordered_js[0] == ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_APPP43(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[2], ordered_js[3])
        corr = sAsPsPsP  # As are on different monomers
        S4_arr[0,1,0,0] += np.sum(corr[np.where(index != 0)]*3*I[np.where(index != 0)])
        corr = sBsPsPsP  # As are on different monomers
        S4_arr[0,2,0,0] += np.sum(corr[np.where(index != 0)]*3*I[np.where(index != 0)])

        index = (ordered_js[0] < ordered_js[1]) * (ordered_js[1] < ordered_js[2]) * (ordered_js[2] < ordered_js[3])
        I = S_APPP44(kA, kB, kC, -kA-kB-kC, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[1], ordered_js[2], ordered_js[3])
        corr = sAsPsPsP
        S4_arr[0,1,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])
        corr = sBsPsPsP
        S4_arr[0,2,0,0] += np.sum(corr[np.where(index != 0)]*I[np.where(index != 0)])

    S4_arr[0,0,0,0] *= pppp_pre
    S4_Au *= aaaaU_pre
    S4_Bu *= bbbbU_pre

    S4_arr[1,1,1,1] *= aaaa_pre
    S4_arr[2,2,2,2] *= aaaa_pre
    S4_arr[1,1,1,2] *= aaaa_pre
    S4_arr[1,2,2,2] *= aaaa_pre
    S4_arr[1,1,2,2] *= aaaa_pre
    S4_arr[1,2,1,2] *= aaaa_pre #7

    S4_arr[1,1,1,0] *= paaa_pre
    S4_arr[2,2,2,0] *= paaa_pre
    S4_arr[1,1,2,0] *= paaa_pre
    S4_arr[2,2,1,0] *= paaa_pre
    S4_arr[1,1,0,1] *= paaa_pre
    S4_arr[2,2,0,2] *= paaa_pre #6 T13
    S4_arr[2,0,1,2] *= paaa_pre
    S4_arr[1,0,2,1] *= paaa_pre
    S4_arr[1,1,0,2] *= paaa_pre
    S4_arr[1,2,0,2] *= paaa_pre



    S4_arr[1,1,0,0] *= ppaa_pre
    S4_arr[2,2,0,0] *= ppaa_pre
    S4_arr[1,2,0,0] *= ppaa_pre
    S4_arr[1,0,0,1] *= ppaa_pre
    S4_arr[2,0,0,2] *= ppaa_pre
    S4_arr[1,0,0,2] *= ppaa_pre
    S4_arr[0,1,1,0] *= ppaa_pre
    S4_arr[0,2,2,0] *= ppaa_pre
    S4_arr[0,1,2,0] *= ppaa_pre #9 T22

    S4_arr[1,0,0,0] *= pppa_pre
    S4_arr[2,0,0,0] *= pppa_pre
    S4_arr[0,1,0,0] *= pppa_pre
    S4_arr[0,2,0,0] *= pppa_pre #4 T26


# #   ORIGINAL RELATIONS
#     S4_arr[2,1,1,1] = S4_arr[1,2,1,1] = S4_arr[1,1,2,1] = S4_arr[1,1,1,2]
#     S4_arr[2,2,2,1] = S4_arr[2,2,1,2] = S4_arr[2,1,2,2] = S4_arr[1,2,2,2]
#     S4_arr[2,2,1,1] = S4_arr[1,2,2,1] = S4_arr[2,1,1,2] = S4_arr[1,1,2,2]
#     S4_arr[2,1,2,1] = S4_arr[1,2,1,2] #10 T36



#     S4_arr[0,1,1,1] = S4_arr[1,1,1,0]
#     S4_arr[0,2,2,2] = S4_arr[2,2,2,0] #2 T38

#     S4_arr[1,2,1,0] = S4_arr[0,1,2,1] = S4_arr[2,1,1,0] = S4_arr[0,2,1,1] \
#         = S4_arr[0,1,1,2] = S4_arr[1,1,2,0]
#     S4_arr[2,1,2,0] = S4_arr[0,2,1,2] = S4_arr[1,2,2,0] = S4_arr[0,1,2,2] \
#         = S4_arr[0,2,2,1] = S4_arr[2,2,1,0] #10 T48
    

#     S4_arr[1,0,1,1] = S4_arr[1,1,0,1]
#     S4_arr[2,0,2,2] = S4_arr[2,2,0,2] 
#     S4_arr[1,0,1,2] = S4_arr[2,0,1,1] = S4_arr[2,1,0,1] = S4_arr[1,1,0,2]
#     S4_arr[2,2,0,1] = S4_arr[2,0,2,1] = S4_arr[1,0,2,2] = S4_arr[1,2,0,2] 
#     S4_arr[1,2,0,1] = S4_arr[1,0,2,1]
#     S4_arr[2,1,0,2] = S4_arr[2,0,1,2] #10 T58

#     S4_arr[0,1,0,1] = S4_arr[1,0,1,0] = S4_arr[1,0,0,1]
#     S4_arr[0,2,0,2] = S4_arr[2,0,2,0] = S4_arr[2,0,0,2] #4 T62

#     S4_arr[0,2,0,1] = S4_arr[0,1,0,2] = S4_arr[2,0,1,0] = S4_arr[1,0,2,0] = S4_arr[1,0,0,2]
#     S4_arr[2,0,0,1] = S4_arr[1,0,0,2]
#     S4_arr[0,2,1,0] = S4_arr[0,1,2,0]
#     S4_arr[2,1,0,0] = S4_arr[0,0,1,2] = S4_arr[0,0,2,1] = S4_arr[1,2,0,0] 
#     S4_arr[0,0,1,1] = S4_arr[1,1,0,0]
#     S4_arr[0,0,2,2] = S4_arr[2,2,0,0] #11 T 73


#     S4_arr[0,0,1,0] = S4_arr[0,1,0,0]
#     S4_arr[0,0,2,0] = S4_arr[0,2,0,0]
#     S4_arr[0,0,0,1] = S4_arr[1,0,0,0]
#     S4_arr[0,0,0,2] = S4_arr[2,0,0,0] #4 T77

#     S4_arr[1,1,1,1] += S4_Au
#     S4_arr[2,2,2,2] += S4_Bu


#     S4_arr[3,3,3,3] += solv_cons

#     return S4_arr


    # RIGOROUS RELATTOINS
    # TODO ADD:  S4_arr[1,1,2,1], S4_arr[2,1,2,2], S4_arr[0,1,2,1], S4_arr[0,1,1,2], S4_arr[0,2,1,2], S4_arr[0,2,2,1], S4_arr[2,1,0,1], S4_arr[1,0,2,2], S4_arr[1,0,2,0]
    #                 DONE             DONE                DONE           DONE             DONE            DONE              DONE            DONE             DONE

    S4_arr[2,1,1,1] = S4_arr[1,1,1,2]
    S4_arr[1,2,1,1] = S4_arr[1,1,2,1] # NEW
    S4_arr[2,2,2,1] = S4_arr[1,2,2,2]
    S4_arr[2,2,1,2] = S4_arr[2,1,2,2] # NEW
    S4_arr[2,2,1,1] = S4_arr[1,2,2,1] = S4_arr[2,1,1,2] = S4_arr[1,1,2,2]
    S4_arr[2,1,2,1] = S4_arr[1,2,1,2] #10 T36



    S4_arr[0,1,1,1] = S4_arr[1,1,1,0]
    S4_arr[0,2,2,2] = S4_arr[2,2,2,0] #2 T38

    S4_arr[1,2,1,0] = S4_arr[0,1,2,1] # NEW
    S4_arr[2,1,1,0] = S4_arr[0,1,1,2] # NEW
    S4_arr[0,2,1,1] = S4_arr[1,1,2,0]
    S4_arr[2,1,2,0] = S4_arr[0,2,1,2] # NEW
    S4_arr[1,2,2,0] = S4_arr[0,2,2,1] # NEW
    S4_arr[0,1,2,2] = S4_arr[2,2,1,0] #10 T48
    

    S4_arr[1,0,1,1] = S4_arr[1,1,0,1]
    S4_arr[2,0,2,2] = S4_arr[2,2,0,2] 
    S4_arr[1,0,1,2] = S4_arr[2,1,0,1] # NEW
    S4_arr[2,0,1,1] = S4_arr[1,1,0,2] 
    S4_arr[2,0,2,1] = S4_arr[1,2,0,2]
    S4_arr[2,2,0,1] = S4_arr[1,0,2,2] # NEW
    S4_arr[1,2,0,1] = S4_arr[1,0,2,1]
    S4_arr[2,1,0,2] = S4_arr[2,0,1,2] #10 T58

    S4_arr[0,1,0,1] = S4_arr[1,0,1,0] = S4_arr[1,0,0,1]
    S4_arr[0,2,0,2] = S4_arr[2,0,2,0] = S4_arr[2,0,0,2] #4 T62

    S4_arr[0,2,0,1] = S4_arr[0,1,0,2] = S4_arr[2,0,1,0] = S4_arr[1,0,2,0] #NEW
    S4_arr[2,0,0,1] = S4_arr[1,0,0,2]
    S4_arr[0,2,1,0] = S4_arr[0,1,2,0]
    S4_arr[2,1,0,0] = S4_arr[0,0,1,2] = S4_arr[0,0,2,1] = S4_arr[1,2,0,0] 
    S4_arr[0,0,1,1] = S4_arr[1,1,0,0]
    S4_arr[0,0,2,2] = S4_arr[2,2,0,0] #11 T 73


    S4_arr[0,0,1,0] = S4_arr[0,1,0,0]
    S4_arr[0,0,2,0] = S4_arr[0,2,0,0]
    S4_arr[0,0,0,1] = S4_arr[1,0,0,0]
    S4_arr[0,0,0,2] = S4_arr[2,0,0,0] #4 T77

    S4_arr[1,1,1,1] += S4_Au
    S4_arr[2,2,2,2] += S4_Bu


    S4_arr[3,3,3,3] += solv_cons

    return S4_arr

