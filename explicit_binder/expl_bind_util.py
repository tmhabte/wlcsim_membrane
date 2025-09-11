import numpy as np

class Polymer_soln:
    def __init__(self, n_bind, v_int, e_m, phi_p, phi_s, poly_marks,\
                  v_s, v_p, v_A, v_B, N_P, N_A, N_B, b_P, b_A, b_B, chi_AB):
        self.n_bind = n_bind
        self.v_int = v_int
        self.e_m = e_m
        self.phi_p = phi_p
        # self.phi_Ab = phi_Ab
        # self.phi_Au = phi_Au
        # self.phi_Bb = phi_Bb
        # self.phi_Bu = phi_Bu
        # self.phi_A = phi_Ab + phi_Au
        # self.phi_B = phi_Bb + phi_Bu
        self.phi_s = phi_s
        self.solv_cons =  self.phi_s  # alpha = (phi_s * N) / phi_p, then apply incompressibilty
        self.poly_marks = poly_marks
        # self.mu1_arr = mu1_arr
        # self.mu2_arr = mu2_arr
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