# import numpy as np
from expl_bind_corr_calc import *

def gamma2_chis(psol, s_bnd_A, s_bnd_B, phius, K):
    # chrom object contains polymer parameters
    # s_bind arrays of appropriate mu1, mu2
    # polymer-solv chi

    chi_AB = psol.chi_AB
    chi_AP, chi_BP, chi_PS, chi_AS, chi_BS = np.zeros(5) # SET BY 

    # print("vm=A=1 edits - M = N")

    # M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)
    corrs = [s_bnd_A, s_bnd_B]
    S2_mat = calc_sf2(psol, corrs, phius, K)
    # print(S2_mat)
    # S2_mat[-1,-1] *= (N / M)

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)

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

def sf2_inv_zeroq(psol, corrs, phius, K1):
    return 0 # not necessary b/c s2 is invertable at q = 0 (unbound sfs!)

def sf2_inv_raw(psol, corrs, phius, K1):
    # [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    # N = psol.N
    # phi_p = psol.phi_p
    if np.linalg.norm(K1) < 1e-5:
        S2_mat_k0 = calc_sf2(psol, corrs, phius, 0.0001)
        # S2_mat_k1[-1,-1] *= (N / M)    
        S2_inv = np.linalg.inv(S2_mat_k0)
        return S2_inv

    S2_mat_k1 = calc_sf2(psol, corrs, phius, K1)
    # S2_mat_k1[-1,-1] *= (N / M)    
    S2_inv = np.linalg.inv(S2_mat_k1)
    return S2_inv

def gamma3(psol, s_bnd_A, s_bnd_B, phius, Ks):
    K1, K2, K3 = Ks
    
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise Exception('Qs must add up to zero')

    
    # [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    # phi_p = psol.phi_p
    corrs = [s_bnd_A, s_bnd_B]
    # N = psol.N

    #calc sf2
    S2_inv_red = sf2_inv_raw(psol, corrs, phius, K1)
    S2_inv_red_2 = sf2_inv_raw(psol, corrs, phius, K2)
    S2_inv_red_3 = sf2_inv_raw(psol, corrs, phius, K3)
    # print("S2_inv: ", S2_inv_red, S2_inv_red_2, S2_inv_red_3)


    # print("vm=A=1")
    s3 = calc_sf3(psol, corrs, phius, K1, K2, K3)
    # s3[3,3,3] *=  N/M
    # print("s3:", s3)
    T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]) # S = - (O+A+B)    
    #      
    G3 = np.einsum("ijk,il,jm,kn-> lmn", -s3, S2_inv_red, S2_inv_red_2, S2_inv_red_3)

    G3_red = np.einsum("ijk, im, jn, ko -> mno", G3, T, T, T) # only in terms of P, A, B       
    return G3_red



def gamma4(psol, s_bnd_A, s_bnd_B, phius, Ks):
    K1, K2, K3, K4 = Ks
    if np.linalg.norm(K1+K2+K3+K4) >= 1e-10:
        raise Exception('Qs must add up to zero')    
    K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)

    # [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    phi_p = psol.phi_p
    # N = psol.N
    corrs = [s_bnd_A, s_bnd_B]

    # print("vm=a=1")
    # (psol, corrs, phius, k1, k2, k12)
    s4 = calc_sf4(psol, corrs, phius, K1, K2, K3, K4) 
    s3_12 = calc_sf3(psol, corrs, phius, K1, K2, -K1-K2)
    s3_14 = calc_sf3(psol, corrs, phius, K1, K4, -K1-K4)
    s3_23 = calc_sf3(psol, corrs, phius, K2, K3, -K2-K3)
    s3_24 = calc_sf3(psol, corrs, phius, K2, K4, -K2-K4)
    s3_34 = calc_sf3(psol, corrs, phius, K3, K4, -K3-K4)
    s3_13 = calc_sf3(psol, corrs, phius, K1, K3, -K1-K3)

    
    S2_inv_red = sf2_inv_raw(psol, corrs, phius, K)
    S2_inv_red_12 = sf2_inv_raw(psol, corrs, phius, K12)
    S2_inv_red_13 = sf2_inv_raw(psol, corrs, phius, K13)
    S2_inv_red_14 = sf2_inv_raw(psol, corrs, phius, K14)

    S2_inv_red_2 = sf2_inv_raw(psol, corrs, phius, K2)
    S2_inv_red_3 = sf2_inv_raw(psol, corrs, phius, K3)
    S2_inv_red_4 = sf2_inv_raw(psol, corrs, phius, K4)

    part1 = np.einsum("ijkl, im, jn, ko, lp-> mnop", s4, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) 

    part2 = 0
    # edited index so that alphas match correctly b/w s3s and s2s
    part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 += np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) #s3(k1,k3) {abc} @ s2inv(k1){a} @ s2inv(k3){b} @ s2inv(k1+k3){c}
    part2 += np.einsum("abc, def, cf, ai, dk, el, bj -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    
    G4 = (part1 - part2)

    T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]) # S = - (O+A+B)    
    
    G4_red = np.einsum("ijkl, im, jn, ko, lp -> mnop", G4, T, T, T, T) # only in terms of P, A, B

    return G4_red
# def gamma2_chis(psol, s_bnd_A, s_bnd_B, K, chis):
#     # chrom object contains polymer parameters
#     # s_bind arrays of appropriate mu1, mu2
#     # polymer-solv chi

#     # phi_p = psol.phi_p
#     # N = psol.N
#     # v_int = psol.v_int

#     chi_AP, chi_BP, chi_AB, chi_PS, chi_AS, chi_BS = chis
#     # [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom

#     # print("vm=A=1 edits - M = N")

#     # M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)
#     corrs = [s_bnd_A, s_bnd_B]
#     S2_mat = calc_sf2(psol, corrs, np.array([K]))
#     # print(S2_mat)
#     # S2_mat[-1,-1] *= (N / M)

#     #invert, calc g2
#     S2_inv = np.linalg.inv(S2_mat)
    
#     # #gam-gam
#     # S2_inv[0,1] += chi_AP
#     # S2_inv[0,2] += chi_BP
#     # S2_inv[0,3] += chi_AP
#     # S2_inv[0,4] += chi_BP
#     # S2_inv[0,5] += chi_PS

#     # S2_inv[1,0] += chi_AP
#     # S2_inv[1,2] += chi_AB
#     # S2_inv[1,4] += chi_AB
#     # S2_inv[1,5] += chi_AS    

#     # S2_inv[2,0] += chi_BP
#     # S2_inv[2,1] += chi_AB
#     # S2_inv[2,3] += chi_AB
#     # S2_inv[2,5] += chi_BS   

#     # S2_inv[3,0] += chi_AP
#     # S2_inv[3,2] += chi_AB
#     # S2_inv[3,4] += chi_AB
#     # S2_inv[3,5] += chi_AS 

#     # S2_inv[4,0] += chi_BP
#     # S2_inv[4,1] += chi_AB
#     # S2_inv[4,3] += chi_AB
#     # S2_inv[4,5] += chi_BS  


#     # S2_inv[5,0] += chi_PS
#     # S2_inv[5,1] += chi_AS
#     # S2_inv[5,2] += chi_BS
#     # S2_inv[5,3] += chi_AS
#     # S2_inv[5,4] += chi_BS

#     # T = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [-1,-1,-1,-1,-1]]) # S = - (P+Ab+Bb+Au+Bu)    

#     # S2 = [[S_PP*phi_p*N_P, S_AP*phi_p*N_A, S_BP*phi_p*N_B, 0,], \
#     #       [S_AP*phi_p*N_A, S_AA*(phi_p*N_A**2)/N_P + S_AuAu*N_A, S_AB*(phi_p*N_A*N_B)/N_P, 0],\
#     #       [S_BP*phi_p*N_B, S_AB*(phi_p*N_A*N_B)/N_P, S_BB*(phi_p*N_B**2)/N_P + S_BuBu*N_B, 0],\
#     #       [0, 0, 0, S_ss]]
#     #gam-gam
#     S2_inv[0,1] += chi_AP
#     S2_inv[0,2] += chi_BP
#     S2_inv[0,3] += chi_PS

#     S2_inv[1,0] += chi_AP
#     S2_inv[1,2] += chi_AB
#     S2_inv[1,3] += chi_AS   

#     S2_inv[2,0] += chi_BP
#     S2_inv[2,1] += chi_AB
#     S2_inv[2,3] += chi_BS   

#     S2_inv[3,0] += chi_PS
#     S2_inv[3,1] += chi_AS
#     S2_inv[3,2] += chi_BS


#     T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]) # S = - (P+A+B)    

#     G2 = np.einsum("ij, im, jn -> mn", S2_inv, T, T) # \Delta_{unred} = T \Delta_{red}    
#     return G2