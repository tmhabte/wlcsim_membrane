# import numpy as np
from expl_bind_corr_calc import *
def gamma2_chis(psol, s_bnd_A, s_bnd_B, K, chis):
    # chrom object contains polymer parameters
    # s_bind arrays of appropriate mu1, mu2
    # polymer-solv chi

    # phi_p = psol.phi_p
    # N = psol.N
    # v_int = psol.v_int

    chi_AP, chi_BP, chi_AB, chi_PS, chi_AS, chi_BS = chis
    # [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom

    # print("vm=A=1 edits - M = N")

    # M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)
    corrs = [s_bnd_A, s_bnd_B]
    S2_mat = calc_sf2(psol, corrs, np.array([K]))
    # print(S2_mat)
    # S2_mat[-1,-1] *= (N / M)

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)
    
    # #gam-gam
    # S2_inv[0,1] += chi_AP
    # S2_inv[0,2] += chi_BP
    # S2_inv[0,3] += chi_AP
    # S2_inv[0,4] += chi_BP
    # S2_inv[0,5] += chi_PS

    # S2_inv[1,0] += chi_AP
    # S2_inv[1,2] += chi_AB
    # S2_inv[1,4] += chi_AB
    # S2_inv[1,5] += chi_AS    

    # S2_inv[2,0] += chi_BP
    # S2_inv[2,1] += chi_AB
    # S2_inv[2,3] += chi_AB
    # S2_inv[2,5] += chi_BS   

    # S2_inv[3,0] += chi_AP
    # S2_inv[3,2] += chi_AB
    # S2_inv[3,4] += chi_AB
    # S2_inv[3,5] += chi_AS 

    # S2_inv[4,0] += chi_BP
    # S2_inv[4,1] += chi_AB
    # S2_inv[4,3] += chi_AB
    # S2_inv[4,5] += chi_BS  


    # S2_inv[5,0] += chi_PS
    # S2_inv[5,1] += chi_AS
    # S2_inv[5,2] += chi_BS
    # S2_inv[5,3] += chi_AS
    # S2_inv[5,4] += chi_BS

    # T = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [-1,-1,-1,-1,-1]]) # S = - (P+Ab+Bb+Au+Bu)    

    # S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
    #       [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P + S_AuAu*N_A, S_AB*(phi_p*N_A*N_B)/N_P, 0],\
    #       [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P + S_BuBu*N_B, 0],\
    #       [0, 0, 0, S_ss]]
    #gam-gam
    S2_inv[0,1] += chi_AP
    S2_inv[0,2] += chi_BP
    S2_inv[0,3] += chi_PS

    S2_inv[1,0] += chi_AP
    S2_inv[1,2] += chi_AB
    S2_inv[1,3] += chi_AS   

    S2_inv[2,0] += chi_BP
    S2_inv[2,1] += chi_AB
    S2_inv[2,3] += chi_BS   

    S2_inv[3,0] += chi_PS
    S2_inv[3,1] += chi_AS
    S2_inv[3,2] += chi_BS


    T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]) # S = - (P+A+B)    

    G2 = np.einsum("ij, im, jn -> mn", S2_inv, T, T) # \Delta_{unred} = T \Delta_{red}    
    return G2