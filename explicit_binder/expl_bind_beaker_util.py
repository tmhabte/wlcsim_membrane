import numpy as np

class Polymer_soln:
    def __init__(self, n_bind, v_int, e_m, phi_p_i, phi_a_i, phi_b_i, V_p, V_A, V_B, poly_marks,\
                  v_s, v_p, v_A, v_B, N_P, N_A, N_B, b_P, b_A, b_B, chi_AB, bs_per_M):
        self.n_bind = n_bind
        self.v_int = v_int
        self.e_m = e_m
        self.phi_p_i = phi_p_i # volume fractionsof initila beakers
        self.V_p = V_p # raw volumes of initial beakers
        self.phi_a_i = phi_a_i
        self.V_A = V_A
        self.phi_b_i = phi_b_i
        self.V_B = V_B
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

        # self.N_m = N / self.M
        # self.b = b


    # v_P = psol.v_p
    # N_P = psol.N_p
    # b_P = psol.b_P
    # v_A = psol.v_A
    # N_A = psol.N_A
    # b_A = psol.b_A
    # v_B = psol.v_B
    # N_B = psol.N_B
    # b_B = psol.b_B
    # M = psol.M
    # solv_cons = psol.solv_cons
    # phi_p = psol.phi_p
    # phi_A_b = psol.phi_A_bound
    # phi_A_u = psol.phi_A_unbound
    # phi_B_b = psol.phi_B_bound
    # phi_B_u = psol.phi_B_unbound