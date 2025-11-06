from expl_bind_beaker_binding_calc import *



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

# def S_AAP31(k_alpha, k_beta, bA, N_A):
#     """
#     I = int_0^N_A dn3 int_0^n3 dn2 exp[-(1/6)bA^2 k_alpha^2 (n3-n2) - (1/6)bA^2 k_beta^2 n2]
#     """
#     a_alpha = (bA**2 / 6.0) * k_alpha**2
#     a_beta  = (bA**2 / 6.0) * k_beta**2
    
#     def f(a):
#         if np.isclose(a, 0.0, atol=1e-14):
#             return N_A
#         return (1 - np.exp(-a * N_A)) / a
    
#     if np.isclose(a_alpha, a_beta, atol=1e-14):
#         # limit case: a_alpha = a_beta
#         # print("31:", 0.5 * f(a_alpha) * N_A)
#         return 0.5 * f(a_alpha) * N_A
#     else:
#         # print((f(a_beta) - f(a_alpha)) / (a_alpha - a_beta))
#         return (f(a_beta) - f(a_alpha)) / (a_alpha - a_beta)


def S_AAP31(k_alpha, k_beta, bA, N_A):
    """
    just a 2 point corr
    I = int_0^N_A dn3 int_0^n3 dn2 exp[-(1/6)bA^2 k_alpha^2 (n3-n2) - (1/6)bA^2 k_beta^2 n2]
    """
    a_alpha = (bA**2 / 6.0) * k_alpha**2
    # a_beta  = (bA**2 / 6.0) * k_beta**2
    
    # def f(a):
    #     if np.isclose(a, 0.0, atol=1e-14):
    #         return N_A
    #     return (1 - np.exp(-a * N_A)) / a
    if np.isclose(a_alpha, 0.0, atol=1e-12):
        return N_A**2 / 2.0
    return (-1.0 + np.exp(-a_alpha * N_A) + a_alpha * N_A) / (a_alpha**2)

    # else:
    #     # print((f(a_beta) - f(a_alpha)) / (a_alpha - a_beta))
    #     return (f(a_beta) - f(a_alpha)) / (a_alpha - a_beta)
# import numpy as np

# def S_AAP32(k2, k3, bA, bP, N_A, N_P, n_i):
# def S_AAP32(k2, k3, bA, bP, N_A, N_P, M, j3, j1):
     
      
#     """
#     Compute the AAP32 (case 2) integral:
    
#     I = (2){ PartA + PartB } 
#     with structure given in the problem.
#     """
#     a2 = (bA**2 / 6.0) * k2**2
#     a3 = (bA**2 / 6.0) * k3**2
#     c3 = (bP**2 / 6.0) * k3**2
#     delJ3 = (bP**2/6) * (N_P / (M-1)) * k3**2 * (j3-j1)

#     # G(a2,a3,N_A): n2,n1 contribution
#     if np.isclose(a2, a3, atol=1e-14):
#         G = (1.0 / a2**2) * (1 - np.exp(-a2 * N_A) * (1 + a2 * N_A))
#     else:
#         G = ((1 - np.exp(-a2 * N_A)) / a2 - (1 - np.exp(-a3 * N_A)) / a3) / (a3 - a2)
    
#     # n3 contribution: part A + part B
#     n3_factor = np.exp(-delJ3)

#     # if np.isclose(c3, 0.0, atol=1e-14):
#     #     n3_factor = (N_P - n_i) + n_i  # just N_P
#     # else:
#         # partA = (1 - np.exp(-c3 * (N_P - n_i))) / c3
#         # partB = (1 - np.exp(-c3 * n_i)) / c3
#         # n3_factor = partA + partB
#     print("32:", 2.0 * G * n3_factor)
#     return 2.0 * G * n3_factor


import numpy as np

def S_AAP32(k2, k3, bA, bP, N_A, N_P, M, j3, j1):
    a2 = (bA**2 / 6.0) * k2**2
    a3 = (bA**2 / 6.0) * k3**2
    c3 = (bP**2 / 6.0) * k3**2
    delJ3 = (bP**2 / 6.0) * (N_P / (M - 1)) * k3**2 * (j3 - j1)

    # ----- G(a2,a3,N_A) -----
    if np.isclose(a2, 0.0, atol=1e-14) and np.isclose(a3, 0.0, atol=1e-14):
        # Case 1: both zero
        G = 0.5 * N_A**2

    elif np.isclose(a2, 0.0, atol=1e-14):
        # Case 2: a2 -> 0, a3 ≠ 0
        G = (N_A - (1 - np.exp(-a3 * N_A)) / a3) / a3

    elif np.isclose(a3, 0.0, atol=1e-14):
        # Case 3: a3 -> 0, a2 ≠ 0
        G = (N_A - (1 - np.exp(-a2 * N_A)) / a2) / a2

    elif np.isclose(a2, a3, atol=1e-14):
        # Case 4: a2 ≈ a3 ≠ 0
        G = (1.0 / a2**2) * (1 - np.exp(-a2 * N_A) * (1 + a2 * N_A))

    else:
        # Case 5: general
        G = ((1 - np.exp(-a2 * N_A)) / a2 - (1 - np.exp(-a3 * N_A)) / a3) / (a3 - a2)

    # ----- n3 contribution -----
    n3_factor = np.exp(-delJ3)
    # print(2.0 * G * n3_factor)
    return 2.0 * G * n3_factor


def S_AAP33(k1, k2, k3, bA, bB, bP, N_A, N_P, M, j1, j2, j3):

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
    # print("33:", n1n2_factor * n3_factor)
    return n1n2_factor * n3_factor

import numpy as np

def S_APA32(k1, k3, bA1, bA3, bP, N_A, N, M, j3, j1):

    a1 = (bA1**2 / 6.0) * k1**2
    a3 = (bA3**2 / 6.0) * k3**2
    B  = (N / (6.0 * (M - 1))) * bP**2 * k3**2 * (j3 - j1)
    
    def f(a, L):
        if np.isclose(a, 0.0, atol=1e-14):
            return L
        return (1 - np.exp(-a * L)) / a
    
    return np.exp(-B) * f(a1, N_A) * f(a3, N_A)
def S_APA33(k1, k2, k3, bA, bP, N_A, N_P, M, j1, j2, j3):

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