""" 
COMPETITIVE BINDING MODEL where both mark profiles are considered, but still only 2 max proteins  bound per nucleosome

RESTRUCUTRED so that minimizing matrix creation

Contains all functions needed to generate spinodal diagrams for a chromosome solution with two reader proteins
USES np.float16 for certain functions to reduce memory usage

Inlcudes:
Average binding state
Structure factor (2D and 1D)
Gamma 2
Spinodal

"""
import numpy as np
from scipy import signal
import scipy as sp

# DATA_TYPE = np.float16
# DATA_TYPE = np.float32
DATA_TYPE = np.float64

def def_chrom(n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, chrom_type = "test"):
    # fraction of nucleosomes with 0,1,2 marks per protein type, calculated form marks1, marks2: 
    [marks_1, marks_2] = poly_marks
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

    return [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b]

    
def find_zero_crossings(matrix):
    zero_crossings = np.where(np.diff(np.signbit(matrix)))
    zero_crossings_vert = np.where(np.diff(np.signbit(matrix), axis=0))
    crossings = np.array(list(zip(zero_crossings[0], zero_crossings[1]))+ list(zip(zero_crossings_vert[0], zero_crossings_vert[1])))
    return crossings

def eval_f_bind_competitive_2(n_b_arr, n_m_pair, e_m_arr,\
                        j_int_g1g1, j_int_g1g2, j_int_g2g2):
    r"""
    MODEL WHERE 2 DISTINCT MARK PROFILES.v still max 2 proteins bound
    eval_f_bind - Evaluate the binding free energy
    
    Parameters
    ----------
    n_b : integer
        Number of reader proteins bound to the nucleosome
    n_m : integer
        Number of marked tails
    e_me3 : float
        Favorable binding free energy to tri-methylated tails
        
    Returns
    -------
    f_bind : float
        Binding free energy
    
    """
    # Calculate the cases for the binding free energy
    f_bind = 0
    n_m1 = n_m_pair[0]
    n_m2 = n_m_pair[1]
    
    if n_m1 == n_m2:
        n_m = n_m1
        
        if n_b_arr==[0,0]:
            f_bind = 0
        elif n_b_arr==[1,0]:
            if n_m == 0:
                f_bind = -np.log(2)
            elif n_m == 1:
                f_bind = -np.log(np.exp(-e_m_arr[0]) + 1)
            elif n_m == 2:
                f_bind = -np.log(2) + e_m_arr[0]
        elif n_b_arr==[0,1]:
            if n_m == 0:
                f_bind = -np.log(2)
            elif n_m == 1:
                f_bind = -np.log(np.exp(-e_m_arr[1]) + 1)
            elif n_m == 2:
                f_bind = -np.log(2) + e_m_arr[1]
        elif n_b_arr==[1,1]:
            if n_m == 0:
                f_bind = -np.log(2)+j_int_g1g2
            elif n_m == 1:
                f_bind = -np.log(np.exp(-e_m_arr[0]) + np.exp(-e_m_arr[1]))+j_int_g1g2
            elif n_m == 2:
                f_bind = -np.log(2) + e_m_arr[0] + e_m_arr[1]+j_int_g1g2
        elif n_b_arr==[2,0]:
            if n_m == 0:
                f_bind = 0+j_int_g1g1
            elif n_m == 1:
                f_bind = e_m_arr[0]+j_int_g1g1
            elif n_m == 2:
                f_bind = 2*e_m_arr[0]+j_int_g1g1
        elif n_b_arr==[0,2]:
            if n_m == 0:
                f_bind = 0+j_int_g2g2
            elif n_m == 1:
                f_bind = e_m_arr[1]+j_int_g2g2
            elif n_m == 2:
                f_bind = 2*e_m_arr[1]+j_int_g2g2
    
    elif n_m1 == 1 and n_m2 == 0:
        if n_b_arr==[0,0]:
            f_bind = 0
        elif n_b_arr==[1,0]:
            f_bind = -np.log(np.exp(-e_m_arr[0]) + 1)
        elif n_b_arr==[0,1]:
            f_bind = -np.log(2)
        elif n_b_arr==[1,1]:
            f_bind = -np.log(np.exp(-e_m_arr[0]) + 1) +j_int_g1g2
        elif n_b_arr==[2,0]:
            f_bind = e_m_arr[0]+j_int_g1g1
        elif n_b_arr==[0,2]:
            f_bind = 0+j_int_g2g2
    elif n_m1 == 0 and n_m2 == 1:
        if n_b_arr==[0,0]:
            f_bind = 0
        elif n_b_arr==[0,1]:
            f_bind = -np.log(np.exp(-e_m_arr[1]) + 1)
        elif n_b_arr==[1,0]:
            f_bind = -np.log(2)
        elif n_b_arr==[1,1]:
            f_bind = -np.log(np.exp(-e_m_arr[1]) + 1) +j_int_g1g2
        elif n_b_arr==[0,2]:
            f_bind = e_m_arr[1]+j_int_g2g2
        elif n_b_arr==[2,0]:
            f_bind = 0+j_int_g1g1
            
    elif n_m1 == 2 and n_m2 == 0:
        if n_b_arr==[0,0]:
            f_bind = 0
        elif n_b_arr==[1,0]:
            f_bind = -np.log(2) + e_m_arr[0]
        elif n_b_arr==[0,1]:
            f_bind = -np.log(2)
        elif n_b_arr==[1,1]:
            f_bind = -np.log(2) + e_m_arr[0] +j_int_g1g2
        elif n_b_arr==[2,0]:
            f_bind = 2*e_m_arr[0]+j_int_g1g1
        elif n_b_arr==[0,2]:
            f_bind = 0+j_int_g2g2
    elif n_m1 == 0 and n_m2 == 2:
        if n_b_arr==[0,0]:
            f_bind = 0
        elif n_b_arr==[0,1]:
            f_bind = -np.log(2) + e_m_arr[1]
        elif n_b_arr==[1,0]:
            f_bind = -np.log(2)
        elif n_b_arr==[1,1]:
            f_bind = -np.log(2) + e_m_arr[1] +j_int_g1g2
        elif n_b_arr==[0,2]:
            f_bind = 2*e_m_arr[1]+j_int_g2g2
        elif n_b_arr==[2,0]:
            f_bind = 0+j_int_g1g1
            
    elif n_m1 == 2 and n_m2 == 1:
        if n_b_arr==[0,0]:
            f_bind = 0
        elif n_b_arr==[1,0]:
            f_bind = -np.log(2) + e_m_arr[0]
        elif n_b_arr==[0,1]:
            f_bind = -np.log(np.exp(-e_m_arr[1]) + 1)
        elif n_b_arr==[1,1]:
            f_bind = -np.log(np.exp(-e_m_arr[0]) + np.exp(-e_m_arr[0] + -e_m_arr[1])) +j_int_g1g2
        elif n_b_arr==[2,0]:
            f_bind = 2*e_m_arr[0]+j_int_g1g1
        elif n_b_arr==[0,2]:
            f_bind =  e_m_arr[1]+j_int_g2g2
    elif n_m1 == 1 and n_m2 == 2:
        if n_b_arr==[0,0]:
            f_bind = 0
        elif n_b_arr==[0,1]:
            f_bind = -np.log(2) + e_m_arr[1]
        elif n_b_arr==[1,0]:
            f_bind = -np.log(np.exp(-e_m_arr[0]) + 1)
        elif n_b_arr==[1,1]:
            f_bind = -np.log(np.exp(-e_m_arr[1]) + np.exp(-e_m_arr[1] + -e_m_arr[0])) +j_int_g1g2
        elif n_b_arr==[0,2]:
            f_bind = 2*e_m_arr[1]+j_int_g2g2
        elif n_b_arr==[2,0]:
            f_bind =  e_m_arr[0]+j_int_g1g1
    
    return f_bind

def eval_f_bind_competitive_arr_2(n_b_arr, n_m_arr_1, n_m_arr_2, e_m_arr,\
                        j_int_g1g1, j_int_g1g2, j_int_g2g2):
    r"""
    eval_f_bind - Evaluate the binding free energy
    
    Parameters
    ----------
    n_b : integer
        Number of reader proteins bound to the nucleosome
    n_m : integer
        Number of marked tails
    e_me3 : float
        Favorable binding free energy to tri-methylated tails
        
    Returns
    -------
    f_bind : float
        Binding free energy
    
    """
    # Calculate the cases for the binding free energy
    n_m_pair_arr = np.column_stack((n_m_arr_1,n_m_arr_2))
    
    f_bind_arr = np.zeros(len(n_m_arr_1))
    
    for i,n_m_pair in enumerate(n_m_pair_arr):
        f_bind = 0
        n_m1 = n_m_pair[0]
        n_m2 = n_m_pair[1]

        if n_m1 == n_m2:
            n_m = n_m1

            if n_b_arr==[0,0]:
                f_bind = 0
            elif n_b_arr==[1,0]:
                if n_m == 0:
                    f_bind = -np.log(2)
                elif n_m == 1:
                    f_bind = -np.log(np.exp(-e_m_arr[0]) + 1)
                elif n_m == 2:
                    f_bind = -np.log(2) + e_m_arr[0]
            elif n_b_arr==[0,1]:
                if n_m == 0:
                    f_bind = -np.log(2)
                elif n_m == 1:
                    f_bind = -np.log(np.exp(-e_m_arr[1]) + 1)
                elif n_m == 2:
                    f_bind = -np.log(2) + e_m_arr[1]
            elif n_b_arr==[1,1]:
                if n_m == 0:
                    f_bind = -np.log(2)+j_int_g1g2
                elif n_m == 1:
                    f_bind = -np.log(np.exp(-e_m_arr[0]) + np.exp(-e_m_arr[1]))+j_int_g1g2
                elif n_m == 2:
                    f_bind = -np.log(2) + e_m_arr[0] + e_m_arr[1]+j_int_g1g2
            elif n_b_arr==[2,0]:
                if n_m == 0:
                    f_bind = 0+j_int_g1g1
                elif n_m == 1:
                    f_bind = e_m_arr[0]+j_int_g1g1
                elif n_m == 2:
                    f_bind = 2*e_m_arr[0]+j_int_g1g1
            elif n_b_arr==[0,2]:
                if n_m == 0:
                    f_bind = 0+j_int_g2g2
                elif n_m == 1:
                    f_bind = e_m_arr[1]+j_int_g2g2
                elif n_m == 2:
                    f_bind = 2*e_m_arr[1]+j_int_g2g2

        elif n_m1 == 1 and n_m2 == 0:
            if n_b_arr==[0,0]:
                f_bind = 0
            elif n_b_arr==[1,0]:
                f_bind = -np.log(np.exp(-e_m_arr[0]) + 1)
            elif n_b_arr==[0,1]:
                f_bind = -np.log(2)
            elif n_b_arr==[1,1]:
                f_bind = -np.log(np.exp(-e_m_arr[0]) + 1) +j_int_g1g2
            elif n_b_arr==[2,0]:
                f_bind = e_m_arr[0]+j_int_g1g1
            elif n_b_arr==[0,2]:
                f_bind = 0+j_int_g2g2
        elif n_m1 == 0 and n_m2 == 1:
            if n_b_arr==[0,0]:
                f_bind = 0
            elif n_b_arr==[0,1]:
                f_bind = -np.log(np.exp(-e_m_arr[1]) + 1)
            elif n_b_arr==[1,0]:
                f_bind = -np.log(2)
            elif n_b_arr==[1,1]:
                f_bind = -np.log(np.exp(-e_m_arr[1]) + 1) +j_int_g1g2
            elif n_b_arr==[0,2]:
                f_bind = e_m_arr[1]+j_int_g2g2
            elif n_b_arr==[2,0]:
                f_bind = 0+j_int_g1g1

        elif n_m1 == 2 and n_m2 == 0:
            if n_b_arr==[0,0]:
                f_bind = 0
            elif n_b_arr==[1,0]:
                f_bind = -np.log(2) + e_m_arr[0]
            elif n_b_arr==[0,1]:
                f_bind = -np.log(2)
            elif n_b_arr==[1,1]:
                f_bind = -np.log(2) + e_m_arr[0] +j_int_g1g2
            elif n_b_arr==[2,0]:
                f_bind = 2*e_m_arr[0]+j_int_g1g1
            elif n_b_arr==[0,2]:
                f_bind = 0+j_int_g2g2
        elif n_m1 == 0 and n_m2 == 2:
            if n_b_arr==[0,0]:
                f_bind = 0
            elif n_b_arr==[0,1]:
                f_bind = -np.log(2) + e_m_arr[1]
            elif n_b_arr==[1,0]:
                f_bind = -np.log(2)
            elif n_b_arr==[1,1]:
                f_bind = -np.log(2) + e_m_arr[1] +j_int_g1g2
            elif n_b_arr==[0,2]:
                f_bind = 2*e_m_arr[1]+j_int_g2g2
            elif n_b_arr==[2,0]:
                f_bind = 0+j_int_g1g1

        elif n_m1 == 2 and n_m2 == 1:
            if n_b_arr==[0,0]:
                f_bind = 0
            elif n_b_arr==[1,0]:
                f_bind = -np.log(2) + e_m_arr[0]
            elif n_b_arr==[0,1]:
                f_bind = -np.log(np.exp(-e_m_arr[1]) + 1)
            elif n_b_arr==[1,1]:
                f_bind = -np.log(np.exp(-e_m_arr[0]) + np.exp(-e_m_arr[0] + -e_m_arr[1])) +j_int_g1g2
            elif n_b_arr==[2,0]:
                f_bind = 2*e_m_arr[0]+j_int_g1g1
            elif n_b_arr==[0,2]:
                f_bind =  e_m_arr[1]+j_int_g2g2
        elif n_m1 == 1 and n_m2 == 2:
            if n_b_arr==[0,0]:
                f_bind = 0
            elif n_b_arr==[0,1]:
                f_bind = -np.log(2) + e_m_arr[1]
            elif n_b_arr==[1,0]:
                f_bind = -np.log(np.exp(-e_m_arr[0]) + 1)
            elif n_b_arr==[1,1]:
                f_bind = -np.log(np.exp(-e_m_arr[1]) + np.exp(-e_m_arr[1] + -e_m_arr[0])) +j_int_g1g2
            elif n_b_arr==[0,2]:
                f_bind = 2*e_m_arr[1]+j_int_g2g2
            elif n_b_arr==[2,0]:
                f_bind =  e_m_arr[0]+j_int_g1g1

        f_bind_arr[i] = f_bind
    return f_bind_arr


def calc_saddle_point_E(f_bars, phi_c, v_int, mu, poly_marks, e_m, f_binds):
    # For two protein types, calculates the part of saddle point free energy that changes when the 
    # average binding state changes
    # f_gammas: list of the two average binding fractions, one for each protein type

    
    # calc mean-field protein-protein interaction
#     print("f_bars:", f_bars)
    mf_pp_int = 0.5*phi_c**2
    for g1 in range(len(f_bars)):
        for g2 in range(len(f_bars)):
            mf_pp_int += v_int[g1][g2]*f_bars[g1]*f_bars[g2]
#     print(" mean-field protein-protein interaction: ", mf_pp_int)
    # calc binding partition function

    phi_bind_arr = phi_c * np.array(f_bars)
#             erg_int = np.matmul(v_int, phi_bind_arr)
    erg_int = np.matmul(phi_bind_arr, v_int)
    
    coef1 = -erg_int[0] + mu[0] 
    coef2 = -erg_int[1] + mu[1]    
    
#     f_bind_g1_s1 = eval_f_bind_vec(1, poly_marks[0], e_m[0], v_int[0,0])
#     f_bind_g1_s2 = eval_f_bind_vec(2, poly_marks[0], e_m[0], v_int[0,0])
#     f_bind_g2_s1 = eval_f_bind_vec(1, poly_marks[1], e_m[1], v_int[1,1])
#     f_bind_g2_s2 = eval_f_bind_vec(2, poly_marks[1], e_m[1], v_int[1,1])
    
#     gam1_part = np.sum(np.log(1 + np.exp(coef1 - f_bind_g1_s1) + np.exp(2*coef1 - f_bind_g1_s2)))
#     gam2_part = np.sum(np.log(1 + np.exp(coef2 - f_bind_g2_s1) + np.exp(2*coef2 - f_bind_g2_s2)))
    
#     bind_part = phi_c*(gam1_part + gam2_part)    
    [P10_fbind, P01_fbind, P20_fbind, P02_fbind, P11_fbind] = f_binds
    P00= 1
    P10= np.exp(mu[0] - erg_int[0] \
                - P10_fbind)
    P01= np.exp(mu[1] - erg_int[1] \
                - P01_fbind)
    P20= np.exp(2*mu[0] - 2*erg_int[0] \
                - P20_fbind)
    P02= np.exp(2*mu[1] - 2*erg_int[1] \
                - P02_fbind)
    P11= np.exp(mu[0] - erg_int[0] + mu[1] - erg_int[1] \
                - P11_fbind)

    q = P00 + P10 + P01 + P20 + P02 + P11    
    
    bind_part = phi_c * np.sum(q)

    return -1 * (mf_pp_int + bind_part)

def calc_binding_states(chrom):
    # calculate f_gam and s_bind for each protein type, at each mu given
    # KEY RETURNS: f_gam_soln_arr, list of matrices of self-consistent f_gammas (for protein 1 and 2) at each mu1, mu2
    #              s_bind_soln_arr, list of matrices of s_binds at each mu1, mu2
    #TODO: 
    #  - deal with interpolating to find more precise zero crossing
        
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    [marks_1, marks_2] = poly_marks
    
    sig_to_frac_dic={}
    sig_to_int_dic={}
    i=0
    for om1 in range(3):
        for om2 in range(3):
            key = repr([om1,om2])
            frac = np.sum((marks_1 == om1)*1 * (marks_2 == om2)*1)/len(marks_1) #fraction of nucleosomes with sigma = [sig_1, sig_2]

            sig_to_frac_dic[key] = frac
            sig_to_int_dic[key] = i
            i+=1
            
    mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    f_gam_arr = np.arange(-0.001,2.002,0.001)

    f_gam_soln_arr = np.zeros((n_bind, len(mu1_array), len(mu2_array)))

    s_bind_soln_arr = np.zeros((n_bind*len(sig_to_frac_dic), len(mu1_array), len(mu2_array)))
    
    f_gam_soln_arr_max = np.zeros((n_bind, len(mu1_array), len(mu2_array)))
    f_gam_soln_arr_min = np.zeros((n_bind, len(mu1_array), len(mu2_array)))

    multi_soln_mus = np.zeros((len(mu1_array), len(mu2_array)))
            
    
    P10_fbind = eval_f_bind_competitive_arr_2([1,0] , poly_marks[0], poly_marks[1], e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1])
    P01_fbind = eval_f_bind_competitive_arr_2([0,1] , poly_marks[0], poly_marks[1], e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1])
    P20_fbind = eval_f_bind_competitive_arr_2([2,0] , poly_marks[0], poly_marks[1], e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1])
    P02_fbind = eval_f_bind_competitive_arr_2([0,2] , poly_marks[0], poly_marks[1], e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1])
    P11_fbind = eval_f_bind_competitive_arr_2([1,1] , poly_marks[0], poly_marks[1], e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1])
    f_binds = [P10_fbind, P01_fbind, P20_fbind, P02_fbind, P11_fbind]
    
    for k, mu1 in enumerate(mu1_array):
        for l, mu2 in enumerate(mu2_array):
            mu = [mu1, mu2]

    #         print(mu)

            # 1) generate right hand side (RHS) of f_gamma1 and f_gamma2 as 2d matrices ( [f1, f2] )


            phi_bind_arr = np.zeros(len(f_gam_arr))
            phi_bind_arr = phi_c * f_gam_arr

            RHS = np.zeros(len(f_gam_arr))
            RHS_arr = np.zeros((n_bind, len(f_gam_arr),len(f_gam_arr)))

            f1,f2 = np.meshgrid(f_gam_arr,f_gam_arr)
            combined_matrix = np.dstack((f1*phi_c, f2*phi_c))#.tolist()
            #erg_int = np.einsum('kl,ijk->jil', v_int, combined_matrix)  #WORKS this is the mat mul of each fgamma pair
            erg_int = np.einsum('ijk,kl->jil',combined_matrix, v_int)

            erg_ints = np.split(erg_int, n_bind, axis=2)
            for i in range(len(erg_ints)):
                erg_ints[i] = np.squeeze(erg_ints[i])

    #         for mark in range(n_bind): # for each reader protein/ mark type
            for om_1 in range(3): 
                for om_2 in range(3): 
                ## 00, 01, 10, 11, 20, 02

                    om = [om_1, om_2]
                    P00= 1
                    P10= np.exp(mu[0] - erg_ints[0] \
                                - eval_f_bind_competitive_2([1,0] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))
                    P01= np.exp(mu[1] - erg_ints[1] \
                                - eval_f_bind_competitive_2([0,1] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))
                    P20= np.exp(2*mu[0] - 2*erg_ints[0] \
                                - eval_f_bind_competitive_2([2,0] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))
                    P02= np.exp(2*mu[1] - 2*erg_ints[1] \
                                - eval_f_bind_competitive_2([0,2] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))
                    P11= np.exp(mu[0] - erg_ints[0] + mu[1] - erg_ints[1] \
                                - eval_f_bind_competitive_2([1,1] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))

                    q = P00 + P10 + P01 + P20 + P02 + P11
                    sg1_avg_raw = 0*P00 + 1*P10 + 0*P01 + 2*P20 + 0*P02 + 1*P11
                    sg2_avg_raw = 0*P00 + 0*P10 + 1*P01 + 0*P20 + 2*P02 + 1*P11
        #             q = 1 + np.exp(mu[mark] - f_bind_1 - erg_ints[mark]) + np.exp(2 * mu[mark] - f_bind_2 - 2 * erg_ints[mark])

                    key = repr(om)
                    frac = sig_to_frac_dic[key]
    #                 frac = np.sum((marks_1 == om1)*1 * (marks_2 == om2)*1)/len(marks_1) #fraction of nucleosomes with sigma = [sig_1, sig_2]

                    mark = 0
                    RHS = RHS_arr[mark] 

                    RHS += frac * sg1_avg_raw / q 
                    RHS_arr[mark] = RHS

                    mark = 1
                    RHS = RHS_arr[mark] 

                    RHS += frac * sg2_avg_raw / q 
                    RHS_arr[mark] = RHS           


            # 2) self-consistently solve for fgamma1 and fgamma2, finding where the difference (fgamma-RHS) crosses zero




            X, Y = np.meshgrid(f_gam_arr, f_gam_arr)

            crs1 = find_zero_crossings(RHS_arr[0]-Y)
            crs2 = find_zero_crossings(RHS_arr[1]-X)


            # 3) find the intersection of the sets of fgamma_n solutions, such that both f_gammas are self consistent


            aset = set([tuple(x) for x in crs1])
            bset = set([tuple(x) for x in crs2])
            inds = np.array([x for x in aset & bset])+1

            f_gam_solns = np.zeros(len(inds), dtype = "object")#f_gam_arr[inds[0]]
            min_soln = [-1]*n_bind
            min_E_soln = [None]*n_bind
            min_E = None

            for i in range(len(inds)):
    #             # TODO implement interpolation for each solution, only in vicinity (essentially redo entire analysis)
    #             # for each point, get an array of all neighboring points (x,y) and at those points RHS - Y, RHS - X (z)

                #original solution
                soln = f_gam_arr[inds[i]]           
    #             f_gam_solns[i] = soln

                # choosing solution with min free energy

                if len(inds) == 1:
                    min_E_soln = soln
                else:
                    E_soln = calc_saddle_point_E(soln, phi_c, v_int, mu, poly_marks, e_m, f_binds)
                    if min_E == None:
                        min_E_soln = soln
                        min_E = E_soln

                    else:
                        if E_soln < min_E:
                            min_E_soln = soln
                            min_E = E_soln
            multi_soln = False
            if len(f_gam_solns) > 1:
                multi_soln_mus[k,l] = 1

            f_gam_solns = [min_E_soln] # selecting minimum E solution


            # 4) for each f_gamma solution pair, calculate each individual s_bind (omega = 0-2)


            sbind = np.zeros((len(f_gam_solns), n_bind*len(sig_to_frac_dic)))

            for j, f_gam_soln in enumerate(f_gam_solns):
                phi_bind_arr = phi_c * np.array(f_gam_soln)
    #             erg_int = np.matmul(v_int, phi_bind_arr)
                erg_int = np.matmul(phi_bind_arr, v_int)

                calc_f_gam = 0

                for om1 in range(3):                 
                    for om2 in range(3): 
                        ## 00, 01, 10, 11, 20, 02

                        om = [om1, om2]
                        P00= 1
                        P10= np.exp(mu[0] - erg_int[0] \
                                    - eval_f_bind_competitive_2([1,0] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))
                        P01= np.exp(mu[1] - erg_int[1] \
                                    - eval_f_bind_competitive_2([0,1] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))
                        P20= np.exp(2*mu[0] - 2*erg_int[0] \
                                    - eval_f_bind_competitive_2([2,0] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))
                        P02= np.exp(2*mu[1] - 2*erg_int[1] \
                                    - eval_f_bind_competitive_2([0,2] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))
                        P11= np.exp(mu[0] - erg_int[0] + mu[1] - erg_int[1] \
                                    - eval_f_bind_competitive_2([1,1] , om, e_m, v_int[0, 0],v_int[0, 1],v_int[1, 1]))

                        q = P00 + P10 + P01 + P20 + P02 + P11
                        sg1_avg_raw = 0*P00 + 1*P10 + 0*P01 + 2*P20 + 0*P02 + 1*P11
                        sg2_avg_raw = 0*P00 + 0*P10 + 1*P01 + 0*P20 + 2*P02 + 1*P11
    #                     print("sg1_avg_raw", sg1_avg_raw)
    #                     print("P10", P10)
                        key = repr(om)
                        place = sig_to_int_dic[key]

                        mark = 0
                        sbind[j, place+mark*9] =  sg1_avg_raw / q 
    #                     print(sg1_avg_raw / q )
                        mark = 1
                        sbind[j, place+mark*9] =  sg2_avg_raw / q 

                        key = repr(om)
                        frac = sig_to_frac_dic[key]
                        calc_f_gam += frac * sg1_avg_raw / q 


    #             print("self-consistency test")
    #             mark = 0
    #             print(calc_f_gam)
    #             print(f_gam_solns[j][mark])


            # 4) store results in array
            # TODO: figure out how to handle mwhen multiple solutions to f_gam
            # current method is to choose minimum solution

            if multi_soln:
                raise Exception("not implemented")
                for f_gam_soln in f_gam_solns:
                    for mark in range(n_bind):
                        f_gam_soln_arr_max[mark][k][l] = ""
            else:
                for mark in range(n_bind): 
                    f_gam_soln_arr[mark][k][l] = f_gam_solns[0][mark]
                    for om1 in range(3):
                        for om2 in range(3):
                            om = [om1,om2]
                            key = repr(om)
                            place = sig_to_int_dic[key]
                            s_bind_soln_arr[place+mark*9][k][l] = sbind[0,place+mark*9]



    return f_gam_soln_arr, s_bind_soln_arr

def calc_sf2_chromo_shlk(chrom, M2s, k_vec = np.logspace(-3, -1, 30)):
    # calculates sf2 using rank 1 monomer correlation tensor 
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    M2_AA, M2_AB, M2_BA, M2_BB, M1_cgam0, M1_cgam1, M2_cc = M2s

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
        
    return S2_AA_arr*N**2, S2_AB_arr*N**2, S2_BA_arr*N**2, S2_BB_arr*N**2, S2_cgam0_arr*N**2, S2_cgam1_arr*N**2, S2_cc_arr*N**2

def inc_gamma(a, x): 
    # got rid of a = 0 condition!
    return sp.special.gamma(a)*sp.special.gammaincc(a, x)

def erf(n):
    return sp.special.erf(n)

def calc_sf2_nuclear(chrom, M2s, k_vec = np.logspace(-3, -1, 30)):
    # calculates sf2 using rank 1 monomer correlation tensor 
    # incorporates shift in scaling due to looping behavior in nucleus
    
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    M2_AA, M2_AB, M2_BA, M2_BB, M1_cgam0, M1_cgam1, M2_cc = M2s

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
    
    bp_per_nuc = 190 # bp per nucleosome
    nm_per_bp = 0.34 
    lower_cutoff = 28.89 * 1000 # bp, s1*, from Rubenstein
    upper_cutoff = 399.12 * 1000 # bp, s2*

    lc_kuhn = lower_cutoff * (nm_per_bp) * (1/b) # converted to nm, then kuhn lengths
    uc_kuhn = upper_cutoff * (nm_per_bp) * (1/b) 

    for i, k in enumerate(k_vec):
        C = np.zeros(M)
        k = np.linalg.norm(k)
        x_m = (1/6) * N_m * b**2 * k**2
        delta = np.arange(0, M, 1)
        
        
        # j1 = j2, s1 > s2
        
        
        index = 0#(j1 == j2)
        constant = 1
        debye = (2/(x_m**2)) * (x_m + np.exp(-x_m) - 1) 

        C[0] += debye
        

        # if j1 - j2 < lower_cutoff
        
        
        index = (np.arange(0, M, 1) * bp_per_nuc <= lower_cutoff) * (np.arange(0, M, 1) != 0) 
        constant = np.exp(-x_m*delta)
        integral = (1/(x_m**2)) * (np.exp(x_m) + np.exp(-x_m) - 2) 

        C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral        
        
        
        # intermediate case, 2/3 scaling
        
        
        index = (np.arange(0, M, 1) * bp_per_nuc > lower_cutoff) * (np.arange(0, M, 1) * bp_per_nuc <= upper_cutoff)
        dlta = delta[index]

        x_m = (1/6) * N_m**(2/3) * b**2 * k**2 * (lc_kuhn)**(1/3)
        

        gammas = -6*inc_gamma(2, dlta**(2/3)*x_m) + 3*inc_gamma(2, (dlta-1)**(2/3)*x_m) \
                + 3*inc_gamma(2, (dlta+1)**(2/3)*x_m)
        erfs = np.pi**0.5 * (dlta - 1)*x_m**(3/2)*erf((dlta-1)**(1/3)*x_m**0.5) \
                - 2*np.pi**0.5 * (dlta)*x_m**(3/2)*erf((dlta)**(1/3)*x_m**0.5) \
                + np.pi**0.5 * (dlta)*x_m**(3/2)*erf((dlta+1)**(1/3)*x_m**0.5) \
                + np.pi**0.5 * x_m**(3/2)*erf((dlta+1)**(1/3)*x_m**0.5)
        neiths = -2*(dlta)**(2/3)*x_m*np.exp(-dlta**(2/3)*x_m) - 2*np.exp(-dlta**(2/3)*x_m)\
                + (dlta - 1)**(2/3) * x_m * np.exp(-(dlta-1)**(2/3)*x_m) \
                + (dlta + 1)**(2/3) * x_m * np.exp(-(dlta+1)**(2/3)*x_m) \
                + np.exp(-(dlta-1)**(2/3)*x_m) + np.exp(-(dlta+1)**(2/3)*x_m)


        integral = 3/(4*x_m**3) * (gammas + erfs + neiths)
#         integral = ((np.exp(x_m)-1)**2 *  np.exp(-(delta**(2/3)+1)*x_m)  ) /x_m**2   # rough scaling implementation

        num_leading_zeros = np.sum((np.arange(0, M, 1) * bp_per_nuc <= lower_cutoff))
        num_trailing_zeros = np.sum((np.arange(0, M, 1) * bp_per_nuc > upper_cutoff))
        
        integral = np.pad(integral, (num_leading_zeros,num_trailing_zeros))
    
        C[np.where((index) != 0)] += integral[np.where((index) != 0)]
        
        
        # if j1 - j2 > upper cutoff -> 1 scaling
        
        
        index = (np.arange(0, M, 1) * bp_per_nuc > upper_cutoff)
        x_m = (1/6) * N_m * b**2 * k**2 * (lc_kuhn/uc_kuhn)**(1/3)
        constant = np.exp(-x_m*delta)
        integral = (1/(x_m**2)) * (np.exp(x_m) + np.exp(-x_m) - 2) 

        C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral   
        

        S2_AA_arr[i] = np.sum((1/M**2) * C * M2_AA)
        S2_AB_arr[i] = np.sum((1/M**2) * C * M2_AB)
        S2_BA_arr[i] = np.sum((1/M**2) * C * M2_BA)
        S2_BB_arr[i] = np.sum((1/M**2) * C * M2_BB)
        
        S2_cgam0_arr[i] = np.sum((1/M**2) * C * M1_cgam0)
        S2_cgam1_arr[i] = np.sum((1/M**2) * C * M1_cgam1)
        S2_cc_arr[i] = np.sum((1/M**2) * C * M2_cc)
        
    return S2_AA_arr*N**2, S2_AB_arr*N**2, S2_BA_arr*N**2, S2_BB_arr*N**2, S2_cgam0_arr*N**2, S2_cgam1_arr*N**2, S2_cc_arr*N**2


import concurrent.futures


def eval_and_reduce_cc(len_marks_1):
    res = np.zeros(len_marks_1)
    res[0] = len_marks_1
    res[1:] = np.arange(2, 2*(len_marks_1), 2)[::-1]
    return res

def eval_and_reduce_cgam(s_bnd, pairs_ind, gam_ind):
#     [marks_1, marks_2] = poly_marks
    
    s_bnd = s_bnd.astype(DATA_TYPE)
    if gam_ind == 0:
        M = len(pairs_ind)
        
        sisj_tens = np.zeros(M, dtype = DATA_TYPE)

        s_bnd_vec = s_bnd[pairs_ind] 
        
#         sisj_tens[0] = np.sum(s_bnd_vec)
#         for i in range(M-1):
#             sisj_tens[-(i+1)] = np.sum(s_bnd_vec[:(i+1)]) + np.sum(s_bnd_vec[-(i+1):])
        
        forward_cumsum = np.cumsum(s_bnd_vec)
        backward_cumsum = np.cumsum(s_bnd_vec[::-1])#[::-1]

        sisj_tens[0] = forward_cumsum[-1]
        sisj_tens[1:] = (forward_cumsum[:-1] + backward_cumsum[:-1])[::-1]
                
    elif gam_ind == 1:
        M = len(pairs_ind)
        
        sisj_tens = np.zeros(M, dtype = DATA_TYPE)

        s_bnd_vec = s_bnd[pairs_ind+9] 
        
#         sisj_tens[0] = np.sum(s_bnd_vec)
#         for i in range(M-1):
#             sisj_tens[-(i+1)] = np.sum(s_bnd_vec[:(i+1)]) + np.sum(s_bnd_vec[-(i+1):])
            
        forward_cumsum = np.cumsum(s_bnd_vec)
        backward_cumsum = np.cumsum(s_bnd_vec[::-1])#[::-1]

        sisj_tens[0] = forward_cumsum[-1]
        sisj_tens[1:] = (forward_cumsum[:-1] + backward_cumsum[:-1])[::-1]
          
    return sisj_tens
        
        
def eval_and_reduce_sisj_bind_simp(chrom, s_bnd, gam1_ind, gam2_ind, pairs_ind):
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

#     [marks_1, marks_2] = poly_marks


    s_bnd_A = s_bnd[pairs_ind+9*gam1_ind]
    s_bnd_B = s_bnd[pairs_ind+9*gam2_ind]
    
    M = len(pairs_ind)
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

def calc_sf_mats(chrom, f_gam_soln_arr, s_bind_soln_arr, k_vec = np.logspace(-3, -1, 30) ):
    # returns rank 3 tensor of mu1, mu2 , k, each value is S2 matrix    
    print(" I USING DATA TYPE " + str(DATA_TYPE))
            
#     start_time = time.time()
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom
    [marks_1, marks_2] = poly_marks
    len_marks_1 = len(marks_1)
    
    sig_to_frac_dic={}
    sig_to_int_dic={}
    i=0
    for om1 in range(3):
        for om2 in range(3):
            key = repr([om1,om2])
            frac = np.sum((marks_1 == om1)*1 * (marks_2 == om2)*1)/len(marks_1) #fraction of nucleosomes with sigma = [sig_1, sig_2]

            sig_to_frac_dic[key] = frac
            sig_to_int_dic[key] = i
            i+=1

    pairs = np.column_stack(poly_marks)

    mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    
    sf_mat = np.zeros((len(mu1_array[:]), len(mu2_array[:]), len(k_vec)), dtype = "object")
    
    
    pairs = np.column_stack(poly_marks)
    pairs_ind = np.zeros(len(marks_1), dtype = int) # for each nucleosome (2 mark values), the corresponding index in s_bnd

    for i,sig_pair in enumerate(pairs):
        key = repr(list(sig_pair))
        ind = sig_to_int_dic[key]
    #     print(type(ind))
        pairs_ind[i] = ind 
    
    for i, mu1 in enumerate(mu1_array[:]):
        for j, mu2 in enumerate(mu2_array[:]):
            
            mu = [mu1, mu2]
            f1 = f_gam_soln_arr[0][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]
            f2 = f_gam_soln_arr[1][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]
            f_bars = [f1, f2]
            
            s_bnd = np.zeros(n_bind*9)
            for ib in range(n_bind*9):
                s_bnd[ib] = s_bind_soln_arr[ib][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]
            
                
            cc_red = eval_and_reduce_cc(len_marks_1)

            s_cgam0_red = eval_and_reduce_cgam(s_bnd, pairs_ind, 0)

            s_cgam1_red = eval_and_reduce_cgam(s_bnd, pairs_ind, 1)

            sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd, 0, 0, pairs_ind)
            
            sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd, 0, 1, pairs_ind)
        
            sisj_BA_red = sisj_AB_red
          
            sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd, 1, 1, pairs_ind)
            
            M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]

            
            for ik, k in enumerate(k_vec):
#                 g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = phi_c * np.array(calc_sf2_chromo_shlk(chrom, M2s, [k]))
                g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = phi_c * np.array(calc_sf2_nuclear(chrom, M2s, [k]))
                
                ss = 1-phi_c
                S2_mat = 1/N**2 * np.array([[cc[0], 0, cg1[0], cg2[0]],\
                                [0, ss*N**2, 0, 0], \
                                [cg1[0], 0, g1g1[0], g1g2[0]],\
                                [cg2[0], 0, g2g1[0], g2g2[0]]])
                sf_mat[i][j][ik] = S2_mat
    return sf_mat
