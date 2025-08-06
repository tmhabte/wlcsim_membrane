# import numpy as np
from expl_bind_binding_calc import *
# before- each polymer-based sf had an identical sf integral, 
#   only with different binding state correlation average. 
#   this meant that could calculate all nth order sfs identically, then just
#   multiply by the appropriate binding average to get al sfs (AA, AB, etc)

# NOW- multiple differnt forms of the strucutre factor integral and binding correlation
#   cannot just have a single sf integral for n-order strucutre factor

# def calc_sisjs(s_bind_A, s_bind_B):
#     sig_0 = (1-s_bind_A)*(1-s_bind_B)
#     sig_A = s_bind_A * (1-s_bind_B)
#     sig_B = s_bind_B * (1 - s_bind_A)
#     sig_AB = s_bind_A * s_bind_B   
#     sisj_arr = [sig_0, sig_A, sig_B, sig_AB]
#     return sisj_arr

# def calc_mon_mat_2(s_bind_A, s_bind_B, competitive):
#     nm = len(s_bind_A)
   
#     sisj_arr =  calc_sisjs(s_bind_A, s_bind_B) #[sig_0, sig_A, sig_B, sig_AB]

#     if competitive:
#         # explicit competitive binding- sig_AB not considered
#         sig_inds = [0,1,2] # O, gamma1, gamma2
#     else:
#         sig_inds = [0,1,2,3] # O, gamma1, gamma2, gamma1&gamma2

#     M2_arr = np.zeros((len(sig_inds), len(sig_inds)), dtype= "object")
#     for a1, a2 in product(sig_inds, repeat=2):
#         # print([a1, a2, a3])
#         # M2_arr[a1][a2] = np.einsum("i,j", sisj_arr[a1],  sisj_arr[a2])

#         #calculate reduced monomer tensor 
#         sisj_tens = np.zeros(nm)

#         sisj_tens[0] = np.sum(sisj_arr[a1] * sisj_arr[a2])
#         conv = signal.convolve(sisj_arr[a1], sisj_arr[a2][::-1])
#         sisj_tens[1:] = conv[:nm-1][::-1] + conv[:nm-1:-1][::-1]
#         M2_arr[a1][a2] = sisj_tens

#     return M2_arr

def calc_sf2(psol, corrs, k):
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
    solv_cons = psol.solv_cons
    phi_p = psol.phi_p
    phi_A = psol.phi_A
    phi_B = psol.phi_B

    sA, sB = corrs
    sAsA = np.outer(sA, sA)
    sBsB = np.outer(sB, sB)
    sAsB = np.outer(sA, sB)
    
    x_p = (1/6)*N_P*b_P**2*k**2
    x_A = (1/6)*N_A*b_A**2*k**2
    x_B = (1/6)*N_B*b_B**2*k**2
    x_del = (1/6)*(N_P/(M-1))*b_P**2*k**2
    grid = np.indices((M,M))
    j1 = grid[0]
    j2 = grid[1]

    
    S_PP = ((2/x_p**2)*(x_p * np.exp(-x_p) - 1))[0]
    
    S_AuAu = ((2/x_A**2)*(x_A * np.exp(-x_A) - 1))[0]

    S_BuBu = ((2/x_B**2)*(x_B * np.exp(-x_B) - 1))[0]


    S_AA = 0
    C = np.zeros((M,M))
    # diagonal
    index = (j1 == j2)
    integral =  (2/x_A**2)*(x_A * np.exp(-x_A) - 1)
    corr = sA
    C[np.where((index) != 0)] += corr * integral

    #off diagonal
    index = (j2 != j1)
    delta = np.abs(j1 - j2)
    integral = (1/x_A)**2 * (1- np.exp(-x_A))**2 * np.exp(-x_del*delta)
    corr = sAsA
    C[np.where((index) != 0)] += corr[np.where((index) != 0)] * integral[np.where((index) != 0)]

    S_AA = np.sum(C)


    S_BB = 0
    C = np.zeros((M,M))
    # diagonal
    index = (j1 == j2)
    integral =  (2/x_B**2)*(x_B * np.exp(-x_B) - 1)
    corr = sB
    C[np.where((index) != 0)] += corr * integral

    #off diagonal
    index = (j2 != j1)
    delta = np.abs(j1 - j2)
    integral = (1/x_B)**2 * (1- np.exp(-x_B))**2 * np.exp(-x_del*delta)
    corr = sBsB
    C[np.where((index) != 0)] += corr[np.where((index) != 0)] * integral[np.where((index) != 0)]

    S_BB = np.sum(C)


    S_AB = 0
    C = np.zeros((M,M))
    #off diagonal
    index = (j2 != j1)
    delta = np.abs(j1 - j2)
    integral = (1/x_B)*(1/x_A) * (1- np.exp(-x_B)) * (1- np.exp(-x_A)) * np.exp(-x_del*delta)
    corr = sAsB
    C[np.where((index) != 0)] += corr[np.where((index) != 0)] * integral[np.where((index) != 0)]
    S_AB = np.sum(C)


    S_AP = 0
    C = np.zeros((M))
    j1_arr = np.arange(1, M+1)
    integral = (1/x_p)*(1/x_A) * (1- np.exp(-x_A)) * \
          (2 - np.exp(-x_del*(j1_arr-1)) - np.exp(-x_p + x_del*(j1_arr-1)))
    corr = sA
    C = corr * integral
    S_AP = np.sum(C)


    S_BP = 0
    C = np.zeros((M))
    j1_arr = np.arange(1, M+1)
    integral = (1/x_p)*(1/x_B) * (1- np.exp(-x_B)) * \
          (2 - np.exp(-x_del*(j1_arr-1)) - np.exp(-x_p + x_del*(j1_arr-1)))
    corr = sB
    C = corr * integral
    S_BP = np.sum(C)

    S_ss = solv_cons

    # P A_bound B_bound A_unbound B_unbound S
    # constants: 
    # AP: n_p N_A N_P / V_sys = N_A phi_p
    # AA: n_p N_A N_A / V_sys = n_p N_A N_A N_P / V_sys N_P = phi_p N_A^2 / N_P
    # AuAu: N_A N_A / V_sys N_A , but V_sys goes to exp[log(\bar{z}_p/V_sys)] term?
    S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0, 0, 0], \
          [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P, S_AB*(phi_p*N_A*N_B)/N_P, 0, 0, 0],\
          [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P, 0, 0, 0],\
          [0, 0, 0, S_AuAu*N_A, 0, 0],\
          [0, 0, 0, 0, S_BuBu*N_B, 0],\
          [0, 0, 0, 0, 0, S_ss]]
    return S2
    # delta = j1 - j2

    