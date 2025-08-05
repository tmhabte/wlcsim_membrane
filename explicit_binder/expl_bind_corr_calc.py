import numpy as np

# before- each polymer-based sf had an identical sf integral, 
#   only with different binding state correlation average. 
#   this meant that could calculate all nth order sfs identically, then just
#   multiply by the appropriate binding average to get al sfs (AA, AB, etc)

# NOW- multiple differnt forms of the strucutre factor integral and binding correlation
#   cannot just have a single sf integral for n-order strucutre factor

def calc_sf2(psol, corrs, k, competitive):
    v_P = psol.v_p
    N_P = psol.N_p
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
    phi_A_b = psol.phi_A_bound
    phi_A_u = psol.phi_A_unbound
    phi_B_b = psol.phi_B_bound
    phi_B_u = psol.phi_B_unbound


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

    
    S_PP = (2/x_p**2)*(x_p * np.exp(-x_p) - 1)

    S_AuAu = (2/x_A**2)*(x_A * np.exp(-x_A) - 1)

    S_BuBu = (2/x_B**2)*(x_B * np.exp(-x_B) - 1)


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
    C[np.where((index) != 0)] += corr * integral[np.where((index) != 0)]

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
    C[np.where((index) != 0)] += corr * integral[np.where((index) != 0)]

    S_BB = np.sum(C)


    S_AB = 0
    C = np.zeros((M,M))
    #off diagonal
    index = (j2 != j1)
    delta = np.abs(j1 - j2)
    integral = (1/x_B)*(1/x_A) * (1- np.exp(-x_B)) * (1- np.exp(-x_A)) * np.exp(-x_del*delta)
    corr = sAsB
    C[np.where((index) != 0)] += corr * integral[np.where((index) != 0)]
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
    # constants: in limit of N_A = N_B = 0, should be equivalent to OABS binder theory
    S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_P, S_BP*phi_p*N_P, 0, 0, 0],
          [S_AP*phi_p*N_P, S_AA*phi_p*N_P, S_AB*phi_p*N_P, 0, 0, 0],
          [S_BP*phi_p*N_P, S_AB*phi_p*N_P, S_BB*phi_p*N_P, 0, 0, 0],
          [0, 0, 0, S_AuAu*phi_A_u*N_A,0 ,0],
          [0, 0, 0, 0, S_BuBu*phi_B_u*N_B, 0],
          [0, 0, 0, 0, 0, S_ss]]
    return S2
    # delta = j1 - j2

    