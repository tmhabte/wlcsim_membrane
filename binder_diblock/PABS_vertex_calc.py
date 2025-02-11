from PABS_util import *
from PABS_binding_calc import *
from PABS_corr_calc import *


def gamma2(chrom, s_bnd_A, s_bnd_B, K, chi):
    # chrom object contains polymer parameters
    # s_bind arrays of appropriate mu1, mu2
    # polymer-solv chi

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
    g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  np.array(calc_sf2_chromo_shlk(chrom, M2s, [K]))
    
    ss = alpha
                
    S2_mat = (rho_p / M) *  np.array([[cc[0], cg1[0], cg2[0], 0],\
                    [cg1[0], g1g1[0], g1g2[0], 0],\
                    [cg2[0], g2g1[0], g2g2[0], 0],\
                    [0, 0, 0, ss]])

    #invert, calc g2
    S2_inv = np.linalg.inv(S2_mat)

    # en_fac = 1e-1# energetic prefactor
    G2 = np.array([[S2_inv[0,0] - 2*chi + S2_inv[3,3], S2_inv[0,1], S2_inv[0, 2]],\
       [S2_inv[1,0], S2_inv[1,1] + v_int[0,0]*Vol_int, S2_inv[1,2] + v_int[0,1]*Vol_int],\
       [S2_inv[2,0], S2_inv[2,1] + v_int[1,0]*Vol_int, S2_inv[2,2] + v_int[1,1]*Vol_int]])
    
    # en_fac = 1e-1# energetic prefactor
    # G2 = np.array([[S2_inv[0,0] - 2*chi + S2_inv[3,3], S2_inv[0,1], S2_inv[0, 2]],\
    #    [S2_inv[1,0], S2_inv[1,1] + v_int[0,0]*Vol_int*(en_fac), S2_inv[1,2] + v_int[0,1]*Vol_int*(en_fac)],\
    #    [S2_inv[2,0], S2_inv[2,1] + v_int[1,0]*Vol_int*(en_fac), S2_inv[2,2] + v_int[1,1]*Vol_int*(en_fac)]])
    # print("ADDED 1/N to vol_int terms!") didnt change anytihng    
    # print("multiplied v_int by rho_p")
    # print("multiplied v_int by M/rho_p")
    # print("multiplied v_int by small num: ", en_fac)

    return G2

def calc_fa(phia, phib):
    nm = len(phia)
    phiu = 1 - phia - phib
    ind = 0
    # for i in range(nm):
    #     if phia[i] > phib[i]:
    #         ind += 1

    for i in range(nm):
        if phia[i] > (phib[i] + phiu[i]):
            ind += 1

    fa = ind / nm
    # print("edited fa")
    return fa
def calc_fb(phia, phib):
    nm = len(phia)
    phiu = 1 - phia - phib
    
    ind = 0
    # for i in range(nm):
    #     if phib[i] > phia[i]:
    #         ind += 1
    for i in range(nm):
        if phib[i] > (phia[i] + phiu[i]):
            ind += 1

    fb = ind / nm
    
    return fb

def sf2_inv_zeroq(chrom, rho_p, s_bnd_A, s_bnd_B):
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    fa = calc_fa(s_bnd_A, s_bnd_B)
    fb = calc_fb(s_bnd_A, s_bnd_B)
    s2 = np.zeros((4,4),dtype='complex')
    # s2[0,0] += 1/9
    # s2[1,0] += (1/fa) / 9
    # s2[0,1] += (1/fa) / 9
    # s2[2,0] += (1/fb) / 9
    # s2[0,2] += (1/fb) / 9
    # s2[1,1] += (1/ fa**2) / 9
    # s2[1,2] += (1 / (fa * fb)) / 9
    # s2[2,1] += (1 / (fa * fb)) / 9
    # s2[2,2] += (1/fb**2) / 9
    # s2[3,3] += (N**2 / alpha)

    # C = 1 / (1 + fa**2 + fb**2 + 2*fa + 2*fb + 2*fa*fb)
    # s2[0,0] += C
    # s2[1,0] += C
    # s2[0,1] += C
    # s2[2,0] += C
    # s2[0,2] += C
    # s2[1,1] += C
    # s2[1,2] += C
    # s2[2,1] += C
    # s2[2,2] += C
    # s2[3,3] += (N**2 / alpha)

    # s2 *= (M / (rho_p*N**2)) 

    # AB "alt" analysis analog
    C = 1 / (1 + fa**2 + fb**2 + 2*fa + 2*fb + 2*fa*fb)
    s2[0,0] += C
    s2[1,0] += C
    s2[0,1] += C
    s2[2,0] += C
    s2[0,2] += C
    s2[1,1] += C
    s2[1,2] += C
    s2[2,1] += C
    s2[2,2] += C
    s2[3,3] += (N**1 / alpha)

    s2 *= (M / (rho_p*N**1)) 
    print("alt s2_0qinv")
    return s2    

# def sf2_inv(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B):
#     [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
#     if np.linalg.norm(K1) < 1e-5:
#         # g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  rho_p * ((N_m**2 * M)/n_p) * np.array(s2wlc_zeroq(chrom))
#         return sf2_inv_zeroq(chrom, rho_p, s_bnd_A, s_bnd_B)

#     g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  np.array(calc_sf2_chromo_shlk(chrom, M2s, [K1]))
#     ss = alpha#
#     S2_mat_k1 = (rho_c / M) * np.array([[cc[0], 0, cg1[0], cg2[0]],\
#                     [0, ss, 0, 0], \
#                     [cg1[0], 0, g1g1[0], g1g2[0]],\
#                     [cg2[0], 0, g2g1[0], g2g2[0]]])
#     S2_inv = np.linalg.inv(S2_mat_k1)
#     S2_inv_red = np.array([[S2_inv[0,0] + S2_inv[1,1], S2_inv[0,2], S2_inv[0, 3]],\
#        [S2_inv[2,0], S2_inv[2,2], S2_inv[2,3] ],\
#        [S2_inv[3,0], S2_inv[3,2] , S2_inv[3,3]]])  
#     return S2_inv_red

def sf2_inv_raw(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B):
    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    if np.linalg.norm(K1) < 1e-5:
        return sf2_inv_zeroq(chrom, rho_p, s_bnd_A, s_bnd_B)

    g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc =  np.array(calc_sf2_chromo_shlk(chrom, M2s, [K1]))
    ss = alpha#
                
    S2_mat_k1 = (rho_p / M) *  np.array([[cc[0], cg1[0], cg2[0], 0],\
                    [cg1[0], g1g1[0], g1g2[0], 0],\
                    [cg2[0], g2g1[0], g2g2[0], 0],\
                    [0, 0, 0, ss]])
    S2_inv = np.linalg.inv(S2_mat_k1)
    return S2_inv

def gamma3(chrom, s_bnd_A, s_bnd_B, Ks):
    K1, K2, K3 = Ks
    
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise Exception('Qs must add up to zero')

    
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

    #calc sf2\
    S2_inv_red = sf2_inv_raw(chrom, M2s, K1, rho_p, s_bnd_A, s_bnd_B)

    S2_inv_red_2 = sf2_inv_raw(chrom, M2s, K2, rho_p, s_bnd_A, s_bnd_B)
 
    S2_inv_red_3 = sf2_inv_raw(chrom, M2s, K3, rho_p, s_bnd_A, s_bnd_B)

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)



    s3 = ( rho_p/(M) ) * calc_sf3(chrom, M3, [K1], [K2])
            
    T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,0,0]])# \Delta_{unred} = T \Delta_{red}           
        
    G3 = np.einsum("ijk,il,jm,kn-> lmn", -s3, S2_inv_red, S2_inv_red_2, S2_inv_red_3)

    G3_red = np.einsum("ijk, im, jn, ko -> mno", G3, T, T, T) # only in terms of P, A, B       
    return G3_red

def gamma4(chrom, s_bnd_A, s_bnd_B, Ks):
    K1, K2, K3, K4 = Ks
    if np.linalg.norm(K1+K2+K3+K4) >= 1e-10:
        raise Exception('Qs must add up to zero')    
    K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)

    [n_bind, v_int, Vol_int, e_m, rho_p, rho_s, poly_marks, M, mu_max, mu_min, del_mu, alpha, N, N_m, b] = chrom
    
    M4 = calc_mon_mat_4(s_bnd_A, s_bnd_B)
    
    s4 = ( rho_p/(M) ) * calc_sf4(chrom, M4, [K1], [K2], [K3]) 

    M3 = calc_mon_mat_3(s_bnd_A, s_bnd_B)

    s3_12 = ( rho_p/(M) ) * calc_sf3(chrom, M3, [K1], [K2])
    s3_13 = ( rho_p/(M) ) * calc_sf3(chrom, M3, [K1], [K3])
    s3_14 = ( rho_p/(M) ) * calc_sf3(chrom, M3, [K1], [K4])
    s3_23 = ( rho_p/(M) ) * calc_sf3(chrom, M3, [K2], [K3])
    s3_24 = ( rho_p/(M) ) * calc_sf3(chrom, M3, [K2], [K4])
    s3_34 = ( rho_p/(M) ) * calc_sf3(chrom, M3, [K3], [K4])
    
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
    
    S2_inv_red = sf2_inv_raw(chrom, M2s, K, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_12 = sf2_inv_raw(chrom, M2s, K12, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_13 = sf2_inv_raw(chrom, M2s, K13, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_14 = sf2_inv_raw(chrom, M2s, K14, rho_p, s_bnd_A, s_bnd_B)

    S2_inv_red_2 = sf2_inv_raw(chrom, M2s, K2, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_3 = sf2_inv_raw(chrom, M2s, K3, rho_p, s_bnd_A, s_bnd_B)
    S2_inv_red_4 = sf2_inv_raw(chrom, M2s, K4, rho_p, s_bnd_A, s_bnd_B)

    part1 = np.einsum("ijkl, im, jn, ko, lp-> mnop", s4, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) 


    part2 = 0

    # ###################PRIOR PART 2############################

    # edited index so that alphas match correctly b/w s3s and s2s
    part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    part2 += np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) #s3(k1,k3) {abc} @ s2inv(k1){a} @ s2inv(k3){b} @ s2inv(k1+k3){c}
    part2 += np.einsum("abc, def, cf, ai, dk, el, bj -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    
    # ###################PRIOR PART 2############################^^^^^^^^^^^^^


    ####################### NEW PART 2 #############################

    # # edited index so that alphas match correctly b/w s3s and s2s
    # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    # part2 += np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4) #s3(k1,k3) {abc} @ s2inv(k1){a} @ s2inv(k3){b} @ s2inv(k1+k3){c}
    # part2 += np.einsum("abc, def, cf, ai, dk, el, bj -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)

    # print("part1:", part1)
    # print("part2:", part2)
    G4 = (part1 - part2)
    T = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,0,0]]) # \Delta_{unred} = T \Delta_{red} 

    G4_red = np.einsum("ijkl, im, jn, ko, lp -> mnop", G4, T, T, T, T) # only in terms of P, A, B

    return G4_red

    # print("Sfs:")
    # print("S4:", s4)
    # print("s3s:")
    # print(s3_12)
    # print(s3_13)
    # print(s3_14)
    # print(s3_23)
    # print(s3_24)
    # print(s3_34)

    # print("s2s:")
    # print(S2_inv_red)
    # print(S2_inv_red_12) #OUTLIER
    # print(S2_inv_red_13)
    # print(S2_inv_red_14) #OUTLIER
    # print(S2_inv_red_2)
    # print(S2_inv_red_3)
    # print(S2_inv_red_4)

    # # ORIGINAL
    # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    # part2 += np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_14, s3_23, S2_inv_red_14, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)

    # print("g4 part2 raw NEW:")
    # print(part2)

    # print("non zero part of part 2:", np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4))

    # print("zero part of p2:", np.einsum("abc, def, cf, ai, bj, dk, el -> ijkl" ,s3_12, s3_34, S2_inv_red_12, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4))

    # print("-------------------------------------------------------------------------------------------------")
    # print("swithced index equivalency test")
    # # part2_A = np.einsum("abc, def, cf, ai, dk, bj, el -> ijkl" ,s3_13, s3_24, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)
    # # part2_B = np.einsum("abc, def, cf, dk, ai, el, bj -> ijkl" ,s3_24, s3_13, S2_inv_red_13, S2_inv_red, S2_inv_red_2, S2_inv_red_3, S2_inv_red_4)

    # print("PART A:", part2_A)
    # print("PART B:", part2_B)
    # print(part2_A == part2_B)
    # print("-------------------------------------------------------------------------------------------------")


          
        
    # G3 = np.einsum("ijk,il,jm,kn-> lmn", -s3, S2_inv_raw, S2_inv_raw_2, S2_inv_raw_3)

