import numpy as np
from scipy import signal
import scipy as sp

from itertools import permutations as perms
from itertools import product

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