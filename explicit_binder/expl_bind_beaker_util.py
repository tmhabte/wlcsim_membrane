import numpy as np

class Polymer_soln:
    def __init__(self, n_bind, v_int, e_m, phi_p, phi_au, phi_bu, poly_marks,\
                  v_s, v_p, v_A, v_B, N_P, N_A, N_B, b_P, b_A, b_B, chi_AB, bs_per_M):
        self.n_bind = n_bind
        self.v_int = v_int
        self.e_m = e_m
        self.phi_p = phi_p 
        # self.V_p = V_p # raw volumes of initial beakers
        self.phi_au = phi_au
        # self.V_A = V_A
        self.phi_bu = phi_bu
        # self.V_B = V_B
        self.poly_marks = poly_marks
        self.v_s = v_s
        self.N_P = N_P
        self.v_p = v_p # monomer volume
        self.b_P = b_P
        self.N_A = N_A
        self.v_A = v_A # monomer volume
        self.b_A = b_A
        self.N_B = N_B
        self.v_B = v_B # monomer volume
        self.b_B = b_B
        self.M = len(poly_marks[0])
        self.chi_AB = chi_AB
        self.bs_per_M = bs_per_M # binding sites per averaged monomer M



#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------
# OLD
# class Polymer_soln:
#     def __init__(self, n_bind, v_int, e_m, phi_p_i, phi_a_i, phi_b_i, V_p, V_A, V_B, poly_marks,\
#                   v_s, v_p, v_A, v_B, N_P, N_A, N_B, b_P, b_A, b_B, chi_AB, bs_per_M):
#         self.n_bind = n_bind
#         self.v_int = v_int
#         self.e_m = e_m
#         self.phi_p_i = phi_p_i # volume fractionsof initila beakers
#         self.V_p = V_p # raw volumes of initial beakers
#         self.phi_a_i = phi_a_i
#         self.V_A = V_A
#         self.phi_b_i = phi_b_i
#         self.V_B = V_B
#         self.poly_marks = poly_marks
#         self.v_s = v_s
#         self.N_P = N_P
#         self.v_p = v_p # monomer volume
#         self.b_P = b_P
#         self.N_A = N_A
#         self.v_A = v_A # monomer volume
#         self.b_A = b_A
#         self.N_B = N_B
#         self.v_B = v_B # monomer volume
#         self.b_B = b_B
#         self.M = len(poly_marks[0])
#         self.chi_AB = chi_AB
#         self.bs_per_M = bs_per_M # binding sites per averaged monomer M

