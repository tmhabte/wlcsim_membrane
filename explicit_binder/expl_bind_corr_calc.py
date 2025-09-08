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

    phi_Ab = psol.phi_Ab
    phi_Au = psol.phi_Au
    phi_Bb = psol.phi_Bb
    phi_Bu = psol.phi_Bu

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

    
    S_PP = ((2/x_p**2)*(x_p + np.exp(-x_p) - 1))[0]
    
    S_AuAu = ((2/x_A**2)*(x_A + np.exp(-x_A) - 1))[0]

    S_BuBu = ((2/x_B**2)*(x_B + np.exp(-x_B) - 1))[0]


    S_AA = 0
    C = np.zeros((M,M))
    # diagonal
    index = (j1 == j2)
    integral =  ((2/x_A**2)*(x_A + np.exp(-x_A) - 1))[0]
    corr = sA
    C[np.where((index) != 0)] += corr * integral
#     print("removed diag")
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
    integral =  (2/x_B**2)*(x_B + np.exp(-x_B) - 1)
    corr = sB
    C[np.where((index) != 0)] += corr * integral
#     print("removed diag")
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
    # constants: np from z_P; Ns from sf definiton, V_sys form E density
    # assume v_P = v_A = v_B
    # AP: n_p N_A N_P / V_sys = N_A phi_p
    # AA: n_p N_A N_A / V_sys = n_p N_A N_A N_P / V_sys N_P = phi_p N_A^2 / N_P

    # AuAu: N_A N_A / V_sys N_A , but V_sys goes to exp[log(\bar{z}_p/V_sys)] term?
    # AuAu alt: n_A N_A N_A / V_sys = phi_A*N_A
#     S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0, 0, 0], \
#           [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P, S_AB*(phi_p*N_A*N_B)/N_P, 0, 0, 0],\
#           [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P, 0, 0, 0],\
#           [0, 0, 0, S_AuAu*N_A, 0, 0],\
#           [0, 0, 0, 0, S_BuBu*N_B, 0],\
#           [0, 0, 0, 0, 0, S_ss]]

    # P A_tot B_tot S  OLD unbound weighting
    # S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
    #       [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P + S_AuAu*N_A, S_AB*(phi_p*N_A*N_B)/N_P, 0],\
    #       [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P + S_BuBu*N_B, 0],\
    #       [0, 0, 0, S_ss]]
    # return S2  

#     # "intuitive" but largely correct AuAu, BuBu sfs- get a factor of phi from saddle point eqn
#     S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
#           [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P + S_AuAu*phi_A*N_A, S_AB*(phi_p*N_A*N_B)/N_P, 0],\
#           [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P + S_BuBu*phi_B*N_B, 0],\
#           [0, 0, 0, S_ss]]
#     return S2


    # # folowing algebra AuAu, BuBu sfs- get a factor of phi from saddle point eqn
    # # also have N_A^2 for unbound sfs. NOT THAT GOOD
    # S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
    #       [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P + S_AuAu*phi_A*N_A**2, S_AB*(phi_p*N_A*N_B)/N_P, 0],\
    #       [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P + S_BuBu*phi_B*N_B**2, 0],\
    #       [0, 0, 0, S_ss]]
    # return S2

    # applying saddle point result to bound and unbound sfs- NOT ALGEBRAICALLY FOUNDED
    S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
          [S_AP*phi_p*N_A, S_AA*phi_Ab*N_A + S_AuAu*phi_Au*N_A, S_AB*phi_Ab*N_B, 0],\
          [S_BP*phi_p*N_B, S_AB*phi_Ab*N_B, S_BB*phi_Bb*N_B + S_BuBu*phi_Bu*N_B, 0],\
          [0, 0, 0, S_ss]]
    return S2


#     print("IGNORNING UNBOUND POLY")
#     S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
#           [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P, S_AB*(phi_p*N_A*N_B)/N_P, 0],\
#           [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P , 0],\
#           [0, 0, 0, S_ss]]
    
#     print("IGNORNING UNBOUND POLY, IDENTICAL PREFACTORS. NOT USED IN PLOTS, can basically just do NA=NP=NB")
#     S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_P, S_BP*phi_p*N_P, 0,], \
#           [S_AP*phi_p*N_P, S_AA*phi_p*N_P, S_AB*phi_p*N_P, 0],\
#           [S_BP*phi_p*N_P, S_AB*phi_p*N_P, S_BB*phi_p*N_P, 0],\
#           [0, 0, 0, S_ss]]

    return S2
    return S2, S_AuAu*N_A, S_BuBu*N_B
    # delta = j1 - j2


# define a set of integral functions (e.g. S_AAA^(3,1)), and create a function that, when given appropriate
# integral functions for (3,1), (3,2), and (3,3) and the k and b and corr identities, returns the sf3

import numpy as np

# def S_AAA31_new(k_alp, k_bet, b, N_A, tol=1e-10):
#     """
#     j3 = j2 = j1
#     Triple integral:
#     I = \int_0^N dn3 \int_0^n3 dn2 \int_0^n2 dn1 
#         exp[-x1 (n3-n2) - x2 (n2-n1)]
#     """
#     x1 = (b**2/6) * k_alp**2
#     x2 = (b**2/6) * k_bet**2   
#     # Case 1: both zero
#     if np.isclose(x1, 0, atol=tol) and np.isclose(x2, 0, atol=tol):
#         return N_A**3 / 6.0 
    
#     # Case 2: x1 = 0
#     if np.isclose(x1, 0, atol=tol):
#         return (N_A/x2 - (1 - np.exp(-x2*N_A))/x2**2) * N_A
    
#     # Case 3: x2 = 0
#     if np.isclose(x2, 0, atol=tol):
#         return (N_A/x1 - (1 - np.exp(-x1*N_A))/x1**2) * N_A
    
#     # Case 4: x1 = x2
#     if np.isclose(x1, x2, atol=tol):
#         x = x1
#         term1 = N_A/x - (1 - np.exp(-x*N_A))/x**2
#         term2 = (N_A*np.exp(-x*N_A))/x
#         return (term1 - term2) / x
    
#     # General case
#     termA = N_A/x1 - (1 - np.exp(-x1*N_A))/x1**2
#     termB = ((1 - np.exp(-x2*N_A))/x2 - (1 - np.exp(-x1*N_A))/x1) / (x1 - x2)
#     return (termA - termB) / x2

def S_AAA31(k_alp, k_bet, b, N_A):
    """
    j3 = j2 = j1
    Triple integral:
    I = \int_0^N dn3 \int_0^n3 dn2 \int_0^n2 dn1 
        exp[-x1 (n3-n2) - x2 (n2-n1)]
    """
    x1 = (b**2/6) * k_alp**2
    x2 = (b**2/6) * k_bet**2

    # Handle x1 ≈ x2 with a tolerance
    if np.isclose(x1, x2, atol=1e-12):
        return (N_A/x1**2 
                + (N_A*x1**2*np.exp(-x1*N_A) - 2*x1*(1 - np.exp(-x1*N_A)))/x1**4)
    elif np.isclose(x2, 0, atol=1e-12):
        return (2-2*np.exp(-x1*N_A) - 2*x1*N_A + N_A**2*x1**2)/ (2*x1**3)
    elif np.isclose(x1, 0, atol=1e-12):
        return (2-2*np.exp(-x2*N_A) - 2*x2*N_A + N_A**2*x2**2)/ (2*x2**3)
    else:
        return (N_A/(x1*x2) 
                + (1 - np.exp(-x1*N_A))/(x1**2*(x1 - x2)) 
                - (1 - np.exp(-x2*N_A))/(x2**2*(x1 - x2)))

import numpy as np

def S_AAA32_laplace(k2, k3, bA, bP, N_A, N_P, M, j3, j1):
    # FROM ANDY- laplace calculation, only for k2 != k3
    assert k2 != k3
    x1 = (bA**2/6) * k2**2 * N_A
    x3 = (bA**2/6) * k3**2 * N_A
    delJ3 = (bP**2/6) * (N_P / (M-1)) * k3**2 * (j3-j1)
    return 2\
    * np.exp(-delJ3) * ( (1 / (x1*x3)) \
    - (np.exp(-x1) / (x1 * (x3 - x1))) \
    - (np.exp(-x3) / (x3 * (x1 - x3)))) \
    * (1 / x3) * (1 - np.exp(-x3)) 


def S_AAA32(k2, k3, bA, bP, N_A, N_P, M, j3, j1, tol = 1e-10):
    """WRA
    Compute
    I = 2 * \int_0^{N_A} dn3 \int_0^{N_A} dn2 \int_0^{n2} dn1 exp[-bA^2/6*k3^2*n1 - bA^2/6*k2^2*(n2-n1)
                                                     - X_del*k3^2 - bA^2/6*k3^2*n3]
    X_del = C = (1/6)*(N_P/(M-1))*bP^2 * (j3-j1).
    Branches:
      1) x2 == x3
      2) x2 == 0
      3) x3 == 0 and delJ3 == 0 (Mathematica simplified result)
      4) general case
    """
    x2 = (bA**2/6) * k2**2
    x3 = (bA**2/6) * k3**2
    delJ3 = (bP**2/6) * (N_P / (M-1)) * k3**2 * (j3-j1)
    
    #--- Case 1: x2 == x3 ---
    if np.isclose(x2, x3, atol=tol):
        num = (
            2.0
            * np.exp(-delJ3 - 2.0 * N_A * x3)
            * ( -1.0 + np.exp(N_A * x3) )
            * ( -1.0 + np.exp(N_A * x3) - N_A * x3 )
        )
        denom = x3**3
        return num / denom

    # --- Case 2: x2 == 0 ---
    if np.isclose(x2, 0.0, atol=tol):
        num = (
            np.exp(-delJ3 - 2.0 * N_A * x3)
            * ( -1.0 + np.exp(N_A * x3) )
            * ( 2.0 + 2.0 * np.exp(N_A * x3) * (-1.0 + N_A * x3) )
        )
        denom = x3**3
        return num / denom

    # --- Case 3: x3 == 0 ---
    if np.isclose(x3, 0.0, atol=tol): #and np.isclose(delJ3, 0.0, atol=tol):
        if np.isclose(x2, 0.0, atol=tol):
            # Limit x2 -> 0 using series expansion: (-1 + exp(-N_A x2) + N_A x2)/x2^2 -> N_A^2
            return N_A**2
        # Mathematica simplified numerator for x3=0, delJ3=0
        return np.ones_like(delJ3)*2.0 * N_A * (-1.0 + np.exp(-N_A * x2) + N_A * x2) / (x2**2)

    # --- General case ---
    num = (
        2*np.exp(-delJ3 - N_A * (x2 + 3.0 * x3))
        * ( -1.0 + np.exp(N_A * x3) )
        * ( np.exp(N_A * (x2 + x3)) * x2
            - np.exp(2.0 * N_A * x3) * x3
            + np.exp(N_A * (x2 + 2.0 * x3)) * (-x2 + x3) )
    )
    denom = x2 * (x3**2) * (-x2 + x3)
    
    return num / denom

def S_AAA33(k1, k2, k3, bA, bP, N_A, N_P, M, j1, j2, j3):
    """ 
    Compute the triple integral:
    I = \int_0^N_A dn1 \int_0^N_A dn2 \int_0^N_A dn3 exp[ ... ]
    """
    a1 = (bA**2 / 6.0) * k1**2
    a2 = (bA**2 / 6.0) * k2**2
    a3 = (bA**2 / 6.0) * k3**2
    
    # Propagator prefactor
    const = np.exp(
        - (N_P / (6.0*(M-1))) * bP**2 * (k1**2 * (j2 - j1) + k3**2 * (j3 - j2))
    )
    
    def f(a):
        return (1 - np.exp(-a * N_A)) / a if a > 1e-14 else N_A
    
    return const * f(a1) * f(a2) * f(a3)

import numpy as np

def S_AAP31(k_alpha, k_beta, bA, N_A):
    """
    Compute the AAP31 case 1 integral:
    I = ∫_0^N_A dn3 ∫_0^n3 dn2 exp[-(1/6)bA^2 k_alpha^2 (n3-n2) - (1/6)bA^2 k_beta^2 n2]
    """
    a_alpha = (bA**2 / 6.0) * k_alpha**2
    a_beta  = (bA**2 / 6.0) * k_beta**2
    
    def f(a):
        if np.isclose(a, 0.0, atol=1e-14):
            return N_A
        return (1 - np.exp(-a * N_A)) / a
    
    if np.isclose(a_alpha, a_beta, atol=1e-14):
        # limit case: a_alpha = a_beta
        return 0.5 * f(a_alpha) * N_A
    else:
        return (f(a_beta) - f(a_alpha)) / (a_alpha - a_beta)

import numpy as np

# def S_AAP32(k2, k3, bA, bP, N_A, N_P, n_i):
def S_AAP32(k2, k3, bA, bP, N_A, N_P, M, j3, j1):
     
      
    """
    Compute the AAP32 (case 2) integral:
    
    I = (2){ PartA + PartB } 
    with structure given in the problem.
    """
    a2 = (bA**2 / 6.0) * k2**2
    a3 = (bA**2 / 6.0) * k3**2
    c3 = (bP**2 / 6.0) * k3**2
    delJ3 = (bP**2/6) * (N_P / (M-1)) * k3**2 * (j3-j1)

    # G(a2,a3,N_A): n2,n1 contribution
    if np.isclose(a2, a3, atol=1e-14):
        G = (1.0 / a2**2) * (1 - np.exp(-a2 * N_A) * (1 + a2 * N_A))
    else:
        G = ((1 - np.exp(-a2 * N_A)) / a2 - (1 - np.exp(-a3 * N_A)) / a3) / (a3 - a2)
    
    # n3 contribution: part A + part B
    n3_factor = np.exp(-delJ3)

    # if np.isclose(c3, 0.0, atol=1e-14):
    #     n3_factor = (N_P - n_i) + n_i  # just N_P
    # else:
        # partA = (1 - np.exp(-c3 * (N_P - n_i))) / c3
        # partB = (1 - np.exp(-c3 * n_i)) / c3
        # n3_factor = partA + partB
    
    return 2.0 * G * n3_factor
import numpy as np

def S_AAP33(k1, k2, k3, bA, bB, bP, N_A, N_P, M, j1, j2, j3):
    """
    Compute the AAP33 integral as defined:
    Sum of two n3-ranged integrals over n1,n2.
    """
    a1 = (bA**2 / 6.0) * k1**2
    a2 = (bB**2 / 6.0) * k2**2
    # c3 = (bP**2 / 6.0) * k3**2
    
    # B = (N_P / (6.0 * (M - 1))) * bP**2 * k1**2 * (j2 - j1)
    
    def f(a, L):
        if np.isclose(a, 0.0, atol=1e-14):
            return L
        return (1 - np.exp(-a * L)) / a
    
    n1n2_factor = f(a1, N_A) * f(a2, N_A)
    
    n3_factor = np.exp(
        - (N_P / (6.0*(M-1))) * bP**2 * (k1**2 * (j2 - j1) + k3**2 * (j3 - j2))
    )
    # if np.isclose(c3, 0.0, atol=1e-14):
    #     n3_factor = N_P
    # else:
    #     partA = (1 - np.exp(-c3 * (N_P - n_i))) / c3
    #     partB = (1 - np.exp(-c3 * n_i)) / c3
    #     n3_factor = partA + partB
    
    return n1n2_factor * n3_factor

import numpy as np

def S_APA32(k1, k3, bA1, bA3, bP, N_A, N, M, j3, j1):
    """
    Compute the APA32 integral:
    I = ∫_0^{N_A} dn1 ∫_0^{N_A} dn3 exp[-a1*n1 - B - a3*n3]
    """
    a1 = (bA1**2 / 6.0) * k1**2
    a3 = (bA3**2 / 6.0) * k3**2
    B  = (N / (6.0 * (M - 1))) * bP**2 * k3**2 * (j3 - j1)
    
    def f(a, L):
        if np.isclose(a, 0.0, atol=1e-14):
            return L
        return (1 - np.exp(-a * L)) / a
    
    return np.exp(-B) * f(a1, N_A) * f(a3, N_A)
def S_APA33(k1, k2, k3, bA, bP, N_A, N_P, M, j1, j2, j3):
    """
    Compute the AAP33 integral as defined:
    Sum of two n3-ranged integrals over n1,n2.
    """
    a1 = (bA**2 / 6.0) * k1**2
    a2 = (bA**2 / 6.0) * k3**2
    # c3 = (bP**2 / 6.0) * k3**2
    
    # B = (N_P / (6.0 * (M - 1))) * bP**2 * k1**2 * (j2 - j1)
    
    def f(a, L):
        if np.isclose(a, 0.0, atol=1e-14):
            return L
        return (1 - np.exp(-a * L)) / a
    
    n1n2_factor = f(a1, N_A) * f(a2, N_A)
    
    n3_factor = np.exp(
        - (N_P / (6.0*(M-1))) * bP**2 * (k1**2 * (j2 - j1) + k3**2 * (j3 - j2))
    )
    # if np.isclose(c3, 0.0, atol=1e-14):
    #     n3_factor = N_P
    # else:
    #     partA = (1 - np.exp(-c3 * (N_P - n_i))) / c3
    #     partB = (1 - np.exp(-c3 * n_i)) / c3
    #     n3_factor = partA + partB
    
    return n1n2_factor * n3_factor


def S_APP31(k3, bA, N_A):
    """
    compute
    I = \int_0^{N_A} dn3 exp[- (1/6) bA^2 k3^2 * n3]
    """
    a3 = (bA**2 / 6.0) * k3**2
    
    if np.isclose(a3, 0.0, atol=1e-14):
        return N_A
    return (1 - np.exp(-a3 * N_A)) / a3

import numpy as np

def S_APP32(k1, k3, bA, bP, N_A, N_P, M, j3, j1):
    """    
    I = ∫_0^{N_A} dn1 ∫_{n_i}^{N_P} dn3 exp[-(1/6)bA^2 k1^2 n1 - (1/6)bP^2 k3^2 (n3 - n_i)]
      + ∫_0^{N_A} dn1 ∫_0^{n_i}  dn3 exp[-(1/6)bA^2 k1^2 n1 - (1/6)bP^2 k3^2 (n_i - n3)]
    """
    a1 = (bA**2 / 6.0) * k1**2
    a3 = (bP**2 / 6.0) * k3**2
    B  = (N_P / (6.0 * (M - 1))) * bP**2 * k3**2 * (j3 - j1)

    # n1 integral
    if np.isclose(a1, 0.0, atol=1e-14):
        F1 = N_A
    else:
        F1 = (1 - np.exp(-a1 * N_A)) / a1

    # n3 integrals
    # if np.isclose(a3, 0.0, atol=1e-14):
    #     G = (N_P - n_i) + n_i  # just N_P
    # else:
    #     G1 = (1 - np.exp(-a3 * (N_P - n_i))) / a3
    #     G2 = (1 - np.exp(-a3 * n_i)) / a3
    #     G = G1 + G2

    return F1 * np.exp(-B)#G

def S_APP33(k1, k2, k3, bA, bP, N_A, N_P, M, j1, j2, j3):
    a1 = (bA**2 / 6.0) * k1**2
    B  = (N_P / (6.0 * (M - 1))) * bP**2 * k3**2 * (j3 - j2)
    C  = (N_P / (6.0 * (M - 1))) * bP**2 * k1**2 * (j2 - j1)

    # n1 integral
    if np.isclose(a1, 0.0, atol=1e-14):
        F1 = N_A
    else:
        F1 = (1 - np.exp(-a1 * N_A)) / a1


    return F1 * np.exp(-B)  * np.exp(-C)#G


def calc_sf3(
    sA, sB, k1, k2, k12,
    b_A=1.0, b_B=1.0, b_P=1.0,
    N_A=100, N_B=100, N_P=1000, M=50
):
    """
    Compute third-order structure factors for a single (k1,k2,k12).
    Returns S3_arr with axes [species1, species2, species3] where
      0 -> P, 1 -> A, 2 -> B
    """

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

    S3_arr = np.zeros((3, 3, 3), dtype=float)

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
        S3_arr[0,0,0] += np.sum(S_AAA31(kA, kB, b_P, N_P))

        # AAA -> [1,1,1]
        S3_arr[1,1,1] += np.sum(sA * S_AAA31(kA, kB, b_A, N_A))
        I = S_AAA32(kA, kB, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr1 = sAsAsA / sA[ordered_js[0]]   # careful: ordered_js[0] indexes into sA
        S3_arr[1,1,1] += masked_sum(corr1, I, index1)
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
        I = S_AAP33(kA, kB, -kA-kB, b_A, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[1,1,0] += masked_sum(sAsAsP, I, index2)

        # APA -> [1,0,1]
        S3_arr[1,0,1] += np.sum(sA * S_AAP31(kA, kB, b_A, N_A))
        I = S_APA32(kA, kB, b_A, b_A, b_P, N_A, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sAsAsP
        S3_arr[1,0,1] += masked_sum(corr, I, index1)
        # handling As on different monomers
        S3_arr[1,1,0] += masked_sum(corr, I, index1)
        I = S_APA33(kA, kB, -kA-kB, b_A, b_P, N_A, N_P, M,
                    ordered_js[0], ordered_js[1], ordered_js[2])
        S3_arr[1,0,1] += masked_sum(sAsAsP, I, index2)

        # BBB -> [2,2,2]
        S3_arr[2,2,2] += np.sum(sB * S_AAA31(kA, kB, b_B, N_B))
        I = S_AAA32(kA, kB, b_B, b_P, N_B, N_P, M, ordered_js[0], ordered_js[-1])
        corr = sBsBsB / sB[ordered_js[0]]
        S3_arr[2,2,2] += masked_sum(corr, I, index1)
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
        S3_arr[2,0,2] += masked_sum(corr, I, index1)
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
        S3_arr[2,0,1] += masked_sum(corr, I, index1)
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

    return S3_arr