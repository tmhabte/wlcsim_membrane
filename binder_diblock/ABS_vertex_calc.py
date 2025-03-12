from PABS_util import *
from PABS_binding_calc import *
from PABS_corr_calc import *

def gamma2(chrom, s_bnd_A, s_bnd_B, K, chi, vol_terms):
    # chrom object contains polymer parameters
    # s_bind arrays of appropriate mu1, mu2
    # polymer-solv chi
    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
        
    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)
    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]

    #calc sf2
    g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  (phi_p / (N * A))* np.array(calc_sf2_chromo_shlk(chrom, M2s, [K]))
    # print("s2s div by N")

    ss = (phi_s / v_s ) * v_s**2                  
    S2_mat =  np.array([[ g1g1[0], g1g2[0], 0],\
                        [g2g1[0], g2g2[0], 0], \
                        [0, 0, ss]])

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)

    S2_inv[0,1] += chi
    S2_inv[1,0] += chi

    T = np.array([[1, 0], [0, 1], [-1,-1]]) # \Delta_{unred} = T \Delta_{red} 
    print("simple transformation matrix")

    # FA = np.sum(s_bnd_A)/len(s_bnd_A)
    # FB = 1-FA#np.sum(s_bnd_B)/len(s_bnd_B)
    # T = np.array([[phi_p, -FA], [phi_p, FB], [0,v_s**0.5]]) # Shifan solvent effects \Delta_{unred} = T \Delta_{red} 
    # print("solvent effects transformation matrix")

    G2_red = np.einsum("ij, im, jn -> mn", S2_inv, T, T) # only in terms of A, B   
    # G2_red[0,1]  += chi
    # G2_red[1,0]  += chi
    return G2_red

def gamma2_alt_T(chrom, s_bnd_A, s_bnd_B, K, chi, vol_terms):
    # chrom object contains polymer parameters
    # s_bind arrays of appropriate mu1, mu2
    # polymer-solv chi
    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
        
    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)
    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]

    #calc sf2
    g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  (phi_p / (N * A))* np.array(calc_sf2_chromo_shlk(chrom, M2s, [K]))
    # print("s2s div by N")
    # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  (1 / (N**2))* np.array(calc_sf2_chromo_shlk(chrom, M2s, [K]))
    # print("s2s div by N2- shifan definition")

    ss = (phi_s / v_s ) * v_s**2                  
    S2_mat =  np.array([[ g1g1[0], g1g2[0], 0],\
                        [g2g1[0], g2g2[0], 0], \
                        [0, 0, ss]])

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)

    S2_inv[0,1] += chi
    S2_inv[1,0] += chi

    # S2_inv[0:2, 0:2] *= 
    # T = np.array([[1, 0], [0, 1], [-1,-1]]) # \Delta_{unred} = T \Delta_{red} 

    FA = np.sum(s_bnd_A)/len(s_bnd_A)
    FB = 1-FA#np.sum(s_bnd_B)/len(s_bnd_B)
    # T = np.array([[phi_p, -FA], [phi_p, FB], [0,v_s**0.5]]) # Shifan solvent effects \Delta_{unred} = T \Delta_{red} 
    T = np.array([[-phi_p, -FA], [phi_p, -FB], [0,v_s**0.5]])    
    print("solvent effects transformation matrix")

    G2_red = np.einsum("ij, im, jn -> mn", S2_inv, T, T) # only in terms of A, B   
    # G2_red[0,1]  += chi
    # G2_red[1,0]  += chi
    return G2_red
def gamma2_shifan(chrom, s_bnd_A, s_bnd_B, K, chi, vol_terms):
    # chrom object contains polymer parameters
    # s_bind arrays of appropriate mu1, mu2
    # polymer-solv chi
    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
        
    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)
    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]

    #calc sf2
    g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  (1 / N**2) * np.array(calc_sf2_chromo_shlk(chrom, M2s, [K]))
    # print("s2s div by N")

    ss = (phi_s / v_s ) * v_s**2                  
    S2_mat =  np.array([[g1g1[0], g1g2[0]],\
                        [g2g1[0], g2g2[0]]])

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)

    # S2_inv[0,1] += chi
    # S2_inv[1,0] += chi

    # T = np.array([[1, 0], [0, 1], [-1,-1]]) # \Delta_{unred} = T \Delta_{red} 

    FA = np.sum(s_bnd_A)/len(s_bnd_A)
    FB = 1-FA#np.sum(s_bnd_B)/len(s_bnd_B)

    G2_11 = -2 * chi * phi_p**2 + (phi_p / N) * ( S2_inv[0,0] - 2 * S2_inv[0,1] + S2_inv[1,1])
    G2_12 = (1/N) * ( FA*(S2_inv[0,0] - S2_inv[0,1]) \
                     - FB*(S2_inv[1,1] - S2_inv[0,1]) ) + (chi*phi_p * (1-2*FA))
    G2_22 = (1/(N*phi_p)) * (FA**2 * S2_inv[0,0] + 2 * FA * FB * S2_inv[0,1] + FB**2 * S2_inv[1,1]) \
        + 2*chi * FA * FB + (b*A)/(phi_s * v_s)
    
    G2 = np.array([[G2_11, G2_12], [G2_12, G2_22]])

    # T = np.array([[phi_p, -FA], [phi_p, FB], [0,v_s**0.5]]) # Shifan solvent effects \Delta_{unred} = T \Delta_{red} 
    # print("solvent effects transformation matrix")

    # G2_red = np.einsum("ij, im, jn -> mn", S2_inv, T, T) # only in terms of A, B   
    # # G2_red[0,1]  += chi
    # # G2_red[1,0]  += chi
    return G2

def sf2_inv_zeroq_alt(chrom, rho_p, s_bnd_A, s_bnd_B, vol_terms):
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    # fa = calc_fa(s_bnd_A, s_bnd_B)
    # fb = calc_fb(s_bnd_A, s_bnd_B)

    # s2 = np.ones((2,2),dtype='complex')

    # return s2/(N) # = N * s2 / (N**2), where extra N factor comes from inverting the (s2 / N)
    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p

    s2 = np.zeros((3,3),dtype='complex')
    s2[0,0] = 1/(N*A*phi_p)
    s2[0,1] = 1/(N*A*phi_p)
    s2[1,0] = 1/(N*A*phi_p)
    s2[1,1] = 1/(N*A*phi_p)
    # s2[2,2] = v_s/phi_s
    s2[2,2] = 1/(phi_s*v_s)

    return s2

def sf2_inv_raw_alt(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B, vol_terms):
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    if np.linalg.norm(K1) < 1e-5:
        return sf2_inv_zeroq_alt(chrom, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p

    g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = (phi_p / (N * A))* np.array(calc_sf2_chromo_shlk(chrom, M2s, [K1]))
    # print("s2s div by N")

    ss = (phi_s / v_s ) * v_s**2        

    S2_mat =  np.array([[ g1g1[0], g1g2[0], 0],\
                        [g2g1[0], g2g2[0], 0], \
                        [0, 0, ss]])

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)

    return S2_inv

def gamma3_alt(chrom, s_bnd_A, s_bnd_B, Ks, vol_terms):
    K1, K2, K3 = Ks
    
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise Exception('Qs must add up to zero')

    
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom

    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p

    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)    
    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]

    #calc sf2\
    S2_inv_red = sf2_inv_raw_alt(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    S2_inv_red_2 = sf2_inv_raw_alt(chrom, M2s, K2, rho_p, s_bnd_A, s_bnd_B, vol_terms)
 
    S2_inv_red_3 = sf2_inv_raw_alt(chrom, M2s, K3, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)



    s3 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K1], [K2])[1:4,1:4,1:4]

    ss = (phi_s / v_s ) * v_s**2     
    s3[2,2,2] = ss

    # print("s3: ", s3)
    # print("s2_inv_1: ", S2_inv_red)
    # print("s2_inv_2: ", S2_inv_red_2)
    # print("s2_inv_3: ", S2_inv_red_3)

    # T = np.array([[1, 0], [0, 1], [-1,-1]]) # \Delta_{unred} = T \Delta_{red}

    FA = np.sum(s_bnd_A)/len(s_bnd_A)
    FB = 1-FA#np.sum(s_bnd_B)/len(s_bnd_B)
    # T = np.array([[phi_p, -FA], [phi_p, FB], [0,v_s**0.5]]) # Shifan solvent effects \Delta_{unred} = T \Delta_{red}
    T = np.array([[-phi_p, -FA], [phi_p, -FB], [0,v_s**0.5]])    
    
    print("solvent effects transformation matrix")

    G3 = np.einsum("ijk,il,jm,kn-> lmn", -s3, S2_inv_red, S2_inv_red_2, S2_inv_red_3)

    G3_red = np.einsum("ijk, im, jn, ko -> mno", G3, T, T, T) # only in terms of A, B        
    return G3_red

def gamma3(chrom, s_bnd_A, s_bnd_B, Ks, vol_terms):
    K1, K2, K3 = Ks
    
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise Exception('Qs must add up to zero')

    
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom

    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p

    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)    
    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]

    #calc sf2\
    S2_inv_red = sf2_inv_raw_alt(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    S2_inv_red_2 = sf2_inv_raw_alt(chrom, M2s, K2, rho_p, s_bnd_A, s_bnd_B, vol_terms)
 
    S2_inv_red_3 = sf2_inv_raw_alt(chrom, M2s, K3, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)



    s3 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K1], [K2])[1:4,1:4,1:4]

    ss = (phi_s / v_s ) * v_s**2     
    s3[2,2,2] = ss

    # print("s3: ", s3)
    # print("s2_inv_1: ", S2_inv_red)
    # print("s2_inv_2: ", S2_inv_red_2)
    # print("s2_inv_3: ", S2_inv_red_3)

    T = np.array([[1, 0], [0, 1], [-1,-1]]) # \Delta_{unred} = T \Delta_{red}
    print("simple transofrmation matrix")
    # FA = np.sum(s_bnd_A)/len(s_bnd_A)
    # FB = 1-FA#np.sum(s_bnd_B)/len(s_bnd_B)
    # T = np.array([[phi_p, -FA], [phi_p, FB], [0,v_s**0.5]]) # WRONG unred} = T \Delta_{red}
    # T = np.array([[-phi_p, -FA], [phi_p, -FB], [0,v_s**0.5]])    
    # print("solvent effects transformation matrix")

    G3 = np.einsum("ijk,il,jm,kn-> lmn", -s3, S2_inv_red, S2_inv_red_2, S2_inv_red_3)

    G3_red = np.einsum("ijk, im, jn, ko -> mno", G3, T, T, T) # only in terms of A, B        
    return G3_red

def gamma4_alt(chrom, s_bnd_A, s_bnd_B, Ks, vol_terms):
    K1, K2, K3, K4 = Ks
    if np.linalg.norm(K1+K2+K3+K4) >= 1e-10:
        raise Exception('Qs must add up to zero')    
    K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)

    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom

    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p

    M4 = calc_mon_mat_4(s_bnd_A, s_bnd_B)
    
    s4 = (phi_p / (N * A))* calc_sf4(chrom, M4, [K1], [K2], [K3], vol_terms)[1:4,1:4,1:4,1:4] 

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)

    s3_12 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K1], [K2], vol_terms)[1:4,1:4,1:4]
    s3_13 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K1], [K3], vol_terms)[1:4,1:4,1:4]
    s3_14 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K1], [K4], vol_terms)[1:4,1:4,1:4]
    s3_23 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K2], [K3], vol_terms)[1:4,1:4,1:4]
    s3_24 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K2], [K4], vol_terms)[1:4,1:4,1:4]
    s3_34 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K3], [K4], vol_terms)[1:4,1:4,1:4] 

    ss = (phi_s / v_s ) * v_s**2        
    s4[2,2,2,2] = ss
    s3_12[2,2,2] = ss
    s3_13[2,2,2] = ss
    s3_14[2,2,2] = ss
    s3_23[2,2,2] = ss
    s3_24[2,2,2] = ss
    s3_34[2,2,2] = ss

    # print("s3, s4 div by N")

    # rho_p = rho_c
    # n_p = np.nan 
    
    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)    
    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]
    
    S2_inv_red = sf2_inv_raw_alt(chrom, M2s, K, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_12 = sf2_inv_raw_alt(chrom, M2s, K12, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_13 = sf2_inv_raw_alt(chrom, M2s, K13, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_14 = sf2_inv_raw_alt(chrom, M2s, K14, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    S2_inv_red_2 = sf2_inv_raw_alt(chrom, M2s, K2, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_3 = sf2_inv_raw_alt(chrom, M2s, K3, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_4 = sf2_inv_raw_alt(chrom, M2s, K4, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    # print("Sfs:")
    # print("S4:", s4)
    # print("s3s:")
    # print("s31:",s3_12)
    # print("s32:",s3_13)
    # print("s33:",s3_14)
    # # print(s3_23)
    # # print(s3_24)
    # # print(s3_34)

    # print("s2s:")
    # print("s2inv:", S2_inv_red)
    # print("s21inv:", S2_inv_red_12) #OUTLIER
    # print("s22inv:", S2_inv_red_13)
    # print("s23inv:", S2_inv_red_14) #OUTLIER
    # print(S2_inv_red_2)
    # print(S2_inv_red_3)
    # print(S2_inv_red_4)

    part1 = np.einsum("ijkl, im, jn, ko, lp-> mnop", s4, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) 

    # print("part1:", part1)

    # print("part1 with sfs changed bakc:",  np.einsum("ijkl, im, jn, ko, lp-> mnop", s4, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 = 0

    # edited index so that alphas match correctly b/w s3s and s2s
    part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 += np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) #s3(k1,k3) {abc} @ s2inv(k1){a} @ s2inv(k3){b} @ s2inv(k1+k3){c}
    part2 += np.einsum("abc, def, cf, ai, dk, el, bj -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    
    G4 = (part1 - part2)

    # T = np.array([[1, 0], [0, 1], [-1,-1]]) # \Delta_{unred} = T \Delta_{red} 

    FA = np.sum(s_bnd_A)/len(s_bnd_A)
    FB = 1-FA#np.sum(s_bnd_B)/len(s_bnd_B)
    # T = np.array([[phi_p, -FA], [phi_p, FB], [0,v_s**0.5]]) # Shifan solvent effects \Delta_{unred} = T \Delta_{red} 
    T = np.array([[-phi_p, -FA], [phi_p, -FB], [0,v_s**0.5]])    
    
    print("solvent effects transformation matrix")

    G4_red = np.einsum("ijkl, im, jn, ko, lp -> mnop", G4, T, T, T, T) # only in terms of A, B

    return G4_red

def gamma4(chrom, s_bnd_A, s_bnd_B, Ks, vol_terms):
    K1, K2, K3, K4 = Ks
    if np.linalg.norm(K1+K2+K3+K4) >= 1e-10:
        raise Exception('Qs must add up to zero')    
    K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)

    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom

    phi_p, A, v_s = vol_terms
    phi_s = 1 - phi_p

    M4 = calc_mon_mat_4(s_bnd_A, s_bnd_B)
    
    s4 = (phi_p / (N * A))* calc_sf4(chrom, M4, [K1], [K2], [K3], vol_terms)[1:4,1:4,1:4,1:4] 

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)

    s3_12 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K1], [K2], vol_terms)[1:4,1:4,1:4]
    s3_13 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K1], [K3], vol_terms)[1:4,1:4,1:4]
    s3_14 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K1], [K4], vol_terms)[1:4,1:4,1:4]
    s3_23 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K2], [K3], vol_terms)[1:4,1:4,1:4]
    s3_24 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K2], [K4], vol_terms)[1:4,1:4,1:4]
    s3_34 = (phi_p / (N * A))* calc_sf3(chrom, M3, [K3], [K4], vol_terms)[1:4,1:4,1:4] 

    ss = (phi_s / v_s ) * v_s**2        
    s4[2,2,2,2] = ss
    s3_12[2,2,2] = ss
    s3_13[2,2,2] = ss
    s3_14[2,2,2] = ss
    s3_23[2,2,2] = ss
    s3_24[2,2,2] = ss
    s3_34[2,2,2] = ss

    
    # calc m2s
    cc_red = eval_and_reduce_cc(M)
    s_cgam0_red = eval_and_reduce_cgam(s_bnd_A)
    s_cgam1_red = eval_and_reduce_cgam(s_bnd_B)
    sisj_AA_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_A)
    sisj_AB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_A, s_bnd_B)
    sisj_BA_red = sisj_AB_red
    sisj_BB_red = eval_and_reduce_sisj_bind_simp(chrom, s_bnd_B, s_bnd_B)    
    M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]
    
    S2_inv_red = sf2_inv_raw_alt(chrom, M2s, K, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_12 = sf2_inv_raw_alt(chrom, M2s, K12, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_13 = sf2_inv_raw_alt(chrom, M2s, K13, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_14 = sf2_inv_raw_alt(chrom, M2s, K14, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    S2_inv_red_2 = sf2_inv_raw_alt(chrom, M2s, K2, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_3 = sf2_inv_raw_alt(chrom, M2s, K3, rho_p, s_bnd_A, s_bnd_B, vol_terms)
    S2_inv_red_4 = sf2_inv_raw_alt(chrom, M2s, K4, rho_p, s_bnd_A, s_bnd_B, vol_terms)

    part1 = np.einsum("ijkl, im, jn, ko, lp-> mnop", s4, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) 

    # print("part1:", part1)

    # print("part1 with sfs changed bakc:",  np.einsum("ijkl, im, jn, ko, lp-> mnop", s4, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 = 0

    # edited index so that alphas match correctly b/w s3s and s2s
    part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 += np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) #s3(k1,k3) {abc} @ s2inv(k1){a} @ s2inv(k3){b} @ s2inv(k1+k3){c}
    part2 += np.einsum("abc, def, cf, ai, dk, el, bj -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    
    G4 = (part1 - part2)

    T = np.array([[1, 0], [0, 1], [-1,-1]]) # \Delta_{unred} = T \Delta_{red} 
    print("simple transformation matrix")
    # FA = np.sum(s_bnd_A)/len(s_bnd_A)
    # FB = 1-FA#np.sum(s_bnd_B)/len(s_bnd_B)
    # T = np.array([[phi_p, -FA], [phi_p, FB], [0,v_s**0.5]]) # Shifan solvent effects \Delta_{unred} = T \Delta_{red} 
    # T = np.array([[-phi_p, -FA], [phi_p, -FB], [0,v_s**0.5]])    
    
    # print("solvent effects transformation matrix")

    G4_red = np.einsum("ijkl, im, jn, ko, lp -> mnop", G4, T, T, T, T) # only in terms of A, B

    return G4_red