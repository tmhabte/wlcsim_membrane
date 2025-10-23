import numpy as np
from scipy import signal
import scipy as sp

#from itertools import permutations as perms
from itertools import product

DATA_TYPE = np.float64

class Polymer_soln:
    def __init__(self, n_bind, v_int, e_m, phi_p, poly_marks, mu1_arr, mu2_arr, v_s, v_m, N, b):
        self.n_bind = n_bind
        self.v_int = v_int
        self.e_m = e_m
        self.phi_p = phi_p
        self.phi_s = 1 - phi_p
        self.alpha =  (N / (phi_p*v_s)) * (1 - (v_m * phi_p))  # alpha = (phi_s * N) / phi_p, then apply incompressibilty
        self.poly_marks = poly_marks
        self.mu1_arr = mu1_arr
        self.mu2_arr = mu2_arr
        self.v_s = v_s
        self.v_m = v_m
        self.N = N
        self.M = len(poly_marks[0])
        self.N_m = N / self.M
        self.b = b

# def def_chrom(n_bind, v_int, e_m, rho_c, rho_s, poly_marks, mu_max, mu_min, del_mu, v_s, v_m, chrom_type = "test"):
#     # fraction of nucleosomes with 0,1,2 marks per protein type, calculated form marks1, marks2: 
#     [marks_1, marks_2] = poly_marks # becomes probabilty of a given mark for averaged polymer
#     M = len(marks_1)
# #     [marks_1.astype(DATA_TYPE), marks_2.astype(DATA_TYPE)] = poly_marks
#     # f_om = np.array([(np.array(marks_1)==0).sum(),(np.array(marks_1)==1).sum(),(np.array(marks_1)==2).sum(), \
#     #                     (np.array(marks_2)==0).sum(),(np.array(marks_2)==1).sum(),(np.array(marks_2)==2).sum()])/len(marks_1)
    
#     if chrom_type == "DNA":
# #         l_p = 53 # 53 nm bare DNA
#         l_p = 20 # 20 nm chromosomal DNA
#         bp_p_b = 45 # base pairs per bond
#         nm_p_bp = 0.34 # nanometetrs per base pair
#         b = l_p * 2 #kuhn length

#         N = (len(marks_1)-1) * bp_p_b * nm_p_bp * (1/b)
#         N_m = N/(len(marks_1)-1)
    
#     elif chrom_type == "test":
#         b = 1
#         N_m = 1000
#         N = N_m * len(marks_1)
        
#     elif chrom_type == "diblock":
#         b = 1
#         N_m = 1000/100
#         print("N_m changed from 1000 to 10")
#         N = N_m * len(marks_1)
        
#     # r_int = 0.1#1#3 #nm
#     # Vol_int = (4/3) * np.pi * r_int**3
#     # Vol_int = 3.75e-2
#     # Vol_int = 8.75e-1
#     # Vol_int = 1.0e-2 # N = 500 appropriate volume
#     Vol_int = np.nan #1.0e-1 # N = 500 appropriate volume * 10

#     # PREVIOUS
#     # v_m = N_m * 1 # cross_secional area = 1
#     print("inf func N = ", N)
#     print("in func rho_c = ", rho_c)
#     alpha = (N / (rho_c*v_s)) * (1 - (v_m * rho_c))  # alpha = (rho_s * N) / rho_p, then apply incompressibilty

#     # # SIMPLIFIED
#     # # v_m = N_m * 1 # cross_secional area = 1

#     # phi_s = 1 - rho_c
#     # alpha = (N / rho_c) * phi_s#/v_s
#     # # return [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b]
#     return [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b]
