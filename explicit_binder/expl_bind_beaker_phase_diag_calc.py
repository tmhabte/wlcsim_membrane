from expl_bind_beaker_vertex_calc import *
from scipy.spatial import cKDTree

n_bind = 2 #types of proteins/marks
# e_m = np.array([1.52, 1.52]) #binding energy FOR F_BIND_ALT
e_m = np.array([1.52, 1.52]) #binding energy FOR F_BIND_ALT
# e_m = np.array([2, 2]) #binding energy FOR F_BIND_ALT


v_int =  np.array([[-4, 4], [4, -4]])
# phi_s = 
M = 50
nm = M
pa_vec = np.arange(0, nm, 1) / (nm-1)
pb_vec = 1-pa_vec
poly_marks = [pa_vec, pb_vec]


v_s = 1
v_p = 1
v_A = v_p
v_B = v_p
N_P = 5000
N_A = 1
N_B = N_A
b_P = 1
b_A = b_P
b_B = b_P

V = 1
V_p = 4*V
V_A = V_B = V 
# phi_p_i = 0.9
phi_p_i = 0.7
# phi_p_i = 0.5
del_phi_a_i = 0.1
phi_a_i = np.arange(0,1.01,del_phi_a_i)
phi_b_i = np.arange(0,1.01,del_phi_a_i)
print("del_phi_a_i: ", del_phi_a_i)

res_factor = 10
print("res_factor: ", res_factor)
phi_a_i_fine = np.arange(0.1, 1 + 0.1 / res_factor, 0.1 / res_factor)
phi_b_i_fine = np.arange(0.1, 1 + 0.1 / res_factor, 0.1 / res_factor)

# chi_AB_phi_p_N = 20
# chi_AB = 0.009
chi_AB = 0.00857 * 1000 # point particle limit

psol = Polymer_soln(n_bind, v_int, e_m, phi_p_i, phi_a_i, phi_b_i, V_p, V_A, V_B, poly_marks,\
                  v_s, v_p, v_A, v_B, N_P, N_A, N_B, b_P, b_A, b_B, chi_AB)

# psol = Polymer_soln(n_bind, v_int, e_m, phi_p, phi_s, \
#                     poly_marks, v_s, v_p, v_A, v_B, N_P, N_A, N_B,
#                     b_P, b_A, b_B, chi_AB)



phi_p_f, phi_a_f, phi_b_f, phi_s, phi_Au_mat, phi_Ab_mat, \
phi_Bu_mat, phi_Bb_mat, mu_A_mat, mu_B_mat, fA_mat, fB_mat, f0_mat, sA_mat, sB_mat = calc_mu_phi_bind(psol,)


ID_number = chi_AB + phi_p_i + N_A + N_P + V_p + V_A + e_m[0] + M + res_factor*10


params_dic = {"M":M, "e_m":e_m, "v_s=v_pv_A=v_B": v_s,\
              "N_P":N_P, "N_A = N_B":N_A, "b_P = b_A = b_B":b_P,\
              "Beaker V_P":V_p, "beaker V_A, V_B":V_A, "phi_p_i":phi_p_i, \
              "phi_a_i=phi_b_i":phi_a_i, "chi_AB": chi_AB,\
             "psol": psol, "phi_p_f":phi_p_f, "del_phi_a_i":del_phi_a_i, \
                "res_factor" : res_factor, "ID_number": ID_number}

print("N_P: ", N_P)
print("N_A=N_B: ", N_A)
print("phi_p_i: ", phi_p_i)
print("phi_p_f: ", phi_p_f)
print("chi_AB_phi_p_N: ", chi_AB)


np.save("params_dic_ID="+str(ID_number), params_dic) 
print("param dic saved")


klog_min = -2.5
klog_max = -1.69 # = np.log10(1/N_A)
klog_num = 10

k_vec = np.logspace(klog_min, klog_max, klog_num) / b_P


#-------------------------------------------------------------------------------------------------
# Coarse spinodal analysis
#--------------------------------------------------------------------------------------------------
chi_AP = 0
chi_BP = 0
# chi_AB = 69.5 / (phi_p*N_P) 
chi_PS = 0
chi_AS = 0
chi_BS = 0

# chis = [chi_AP, chi_BP, chi_AB, chi_PS, chi_AS, chi_BS]

min_eigval_arr = np.zeros((len(phi_a_i[:]), len(phi_b_i[:]), len(k_vec)))

min_eigval_arr_allk_DENS = np.zeros((len(phi_a_i[:]), len(phi_b_i[:])))
min_eigval_arr_allk_ps = np.zeros((len(phi_a_i[:]), len(phi_b_i[:])))

# min_eigvec_arr = np.zeros((len(phi_Au_arr[:]), len(phi_Bu_arr[:]), len(k_vec), 5))
# min_eigvec_arr_allk_DENS = np.zeros((len(phi_Au_arr[:]), len(phi_Bu_arr[:]), 5))
min_eigvec_arr = np.zeros((len(phi_a_i[:]), len(phi_b_i[:]), len(k_vec), 3))
min_eigvec_arr_allk_DENS = np.zeros((len(phi_a_i[:]), len(phi_b_i[:]), 3))
k_star_arr_DENS= np.zeros((len(phi_a_i[:]), len(phi_b_i[:]))) 

cond_num_arr = np.zeros((len(phi_a_i[:]), len(phi_b_i[:]), len(k_vec)))
max_cond_arr = np.zeros((len(phi_a_i[:]), len(phi_b_i[:])))

for i in range(len(phi_a_i)):
    for j in range(len(phi_b_i)):
#         if mu1 == mu2:
#             continue
        # mu = [mu1, mu2]
        # print("mu: ", mu)
        s_bnd_A = sA_mat[i,j]
        s_bnd_B = sB_mat[i,j]

        phi_Au = phi_Au_mat[i,j]
        phi_Bu = phi_Bu_mat[i,j]
        phi_sf = phi_s[i,j]

        phis = [phi_p_f, phi_Au, phi_Bu, phi_sf]
        # if i == 0 and j == len(phi_Au_arr)-1:
        #     print("SA: ", s_bnd_A)
        #     print("SB: ", s_bnd_B)
        for ik, k in enumerate(k_vec):

            # print(k)

            # s_bnd_A = s_bind_A_ALL[i, j]
            # s_bnd_B = s_bind_B_ALL[i, j]

            # M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)
            # S2_mat = (phi_p / N) * calc_sf2(psol, M2s, [k], competitive)
            # cond_num_arr[i][j][ik] = np.linalg.cond(S2_mat)

            G2 = gamma2_chis(psol, s_bnd_A, s_bnd_B, phis, k)

            # print(G2)
            # s_bnd_A = s_bind_A_ar[i, j]
            # s_bnd_B = s_bind_B_ar[i, j]

            # G2 = gamma2(chrom, s_bnd_A, s_bnd_B, k, chi, competitive)

            
            val, vec = np.linalg.eigh(G2)
            # print(val)
            vec = vec.T
#                 print(vec)
#                 print(vec.T)
#                 print(val)
            min_val = np.min(val)            
            min_eigval_arr[i][j][ik] = min_val
            min_eigvec_arr[i][j][ik] = vec[np.where(val == min_val)]


        # all-k analysis
        # min eigvals, eigvecs at given mu1,mu2 for each k        
        min_vals = min_eigval_arr[i][j][:][np.nonzero(min_eigval_arr[i][j][:])] 
        min_vecs = min_eigvec_arr[i][j][:][np.nonzero(min_eigval_arr[i][j][:])]

        # minimum eigenvalue at given mu1,mu2 across all ks
        minval_allk = np.min(min_vals) 
        
        # store this eigenvalue and corresponding eigenvector
        min_eigval_arr_allk_DENS[i][j] = minval_allk 
        if len(np.where(min_vals==minval_allk)[0]) == 1:
            min_eigvec_arr_allk_DENS[i][j] = min_vecs[np.where(min_vals==minval_allk)]
            k_star = k_vec[np.where(min_vals==minval_allk)]
        else: #if mulitple minimum eigenvalues, just choose the first one (0) or last (-1)
            min_eigvec_arr_allk_DENS[i][j] = min_vecs[np.where(min_vals==minval_allk)][-1,:]
            k_star = k_vec[np.where(min_vals==minval_allk)][-1]

        k_star_arr_DENS[i][j] = k_star

        # max_cond_num = np.max(cond_num_arr[i][j][:])#[np.nonzero(min_eigval_arr[i][j][:])] 
        # max_cond_arr[i][j] = max_cond_num
# setting all non-decomposed/ separated systems to 0
# 
print("coarse stability analysis finished")


poly_fluc = min_eigvec_arr_allk_DENS[:,:,0]
poly_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

prot1_fluc = min_eigvec_arr_allk_DENS[:,:,1]
prot1_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

prot2_fluc = min_eigvec_arr_allk_DENS[:,:,2]
prot2_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

k_star_arr_DENS[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = -1 # unphysical value, to indicate outside of spinodal

spinodal = np.copy(k_star_arr_DENS)
spinodal[np.where(spinodal == k_vec[0])] = 0 # macro
spinodal[np.where(spinodal > k_vec[0])] = 1 # micro

k_star_arr_DENS[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = -1 # unphysical value, to indicate outside of spinodal


#-------------------------------------------------------------------------------------------------
# border identification
#------------------------------------------------------------------------------------------------


mat_coarse = spinodal[1:, 1:]
# res_factor = 2

# phi_a_i_fine = np.arange(0.1, 1 + 0.1 / res_factor, 0.1 / res_factor)
# phi_b_i_fine = np.arange(0.1, 1 + 0.1 / res_factor, 0.1 / res_factor)
mat_fine = np.zeros((len(phi_a_i_fine), len(phi_b_i_fine)))

x_min, x_max = 0.1, 1.0
y_min, y_max = 0.1, 1.0

n, m = mat_coarse.shape[0], mat_fine.shape[0]

x_coarse = np.linspace(x_min, x_max, n)
y_coarse = np.linspace(y_min, y_max, n)
x_fine = np.linspace(x_min, x_max, m)
y_fine = np.linspace(y_min, y_max, m)

Xc, Yc = np.meshgrid(x_coarse, y_coarse)
Xf, Yf = np.meshgrid(x_fine, y_fine)

# --- Find boundaries between 1 and -1 ---
edges_x = np.abs(np.diff(mat_coarse, axis=1)) > 0
edges_y = np.abs(np.diff(mat_coarse, axis=0)) > 0

# --- Compute midpoints of edges instead of corners ---
# horizontal boundaries (between cols)
bx = (Xc[:, :-1][edges_x] + Xc[:, 1:][edges_x]) / 2
by = (Yc[:, :-1][edges_x] + Yc[:, 1:][edges_x]) / 2

# vertical boundaries (between rows)
bx2 = (Xc[:-1, :][edges_y] + Xc[1:, :][edges_y]) / 2
by2 = (Yc[:-1, :][edges_y] + Yc[1:, :][edges_y]) / 2

# Combine boundary coordinates
boundary_coords = np.column_stack([
    np.concatenate([bx, bx2]),
    np.concatenate([by, by2])
])

# look at fine points near boundary
fine_points = np.column_stack([Xf.ravel(), Yf.ravel()])

tree = cKDTree(boundary_coords)
dist_thresh = (x_max - x_min) / n * 0.75  # half–one coarse cell width works best

idxs = tree.query_ball_point(fine_points, r=dist_thresh)
near_boundary = np.array([len(i) > 0 for i in idxs]).reshape(m, m)

mat_fine[near_boundary] = 1

# Recalculate all quantities wrt this fine-grained diagram

psol_fine = Polymer_soln(n_bind, v_int, e_m, phi_p_i, phi_a_i_fine, phi_b_i_fine, V_p, V_A, V_B, poly_marks,\
                  v_s, v_p, v_A, v_B, N_P, N_A, N_B, b_P, b_A, b_B, chi_AB)


phi_p_f, phi_a_f, phi_b_f, phi_s, phi_Au_mat, phi_Ab_mat, \
phi_Bu_mat, phi_Bb_mat, mu_A_mat, mu_B_mat, fA_mat, fB_mat, f0_mat, sA_mat, sB_mat = calc_mu_phi_bind(psol_fine,)

# TO-DONE! :D : want to have only one point within boundary, but at a very high resolution.
# ANSWER: in phase calc traversal, can just end a iteration when one phase value is calculated

# TO-DONE: recalculate spinodal and k_stars for fine matrix

phi_a_i = phi_a_i_fine
phi_b_i = phi_b_i_fine


chi_AP = 0
chi_BP = 0
# chi_AB = 69.5 / (phi_p*N_P) 
chi_PS = 0
chi_AS = 0
chi_BS = 0


# chis = [chi_AP, chi_BP, chi_AB, chi_PS, chi_AS, chi_BS]

min_eigval_arr = np.zeros((len(phi_a_i[:]), len(phi_b_i[:]), len(k_vec)))

min_eigval_arr_allk_DENS = np.zeros((len(phi_a_i[:]), len(phi_b_i[:])))
min_eigval_arr_allk_ps = np.zeros((len(phi_a_i[:]), len(phi_b_i[:])))

# min_eigvec_arr = np.zeros((len(phi_Au_arr[:]), len(phi_Bu_arr[:]), len(k_vec), 5))
# min_eigvec_arr_allk_DENS = np.zeros((len(phi_Au_arr[:]), len(phi_Bu_arr[:]), 5))
min_eigvec_arr = np.zeros((len(phi_a_i[:]), len(phi_b_i[:]), len(k_vec), 3))
min_eigvec_arr_allk_DENS = np.zeros((len(phi_a_i[:]), len(phi_b_i[:]), 3))
k_star_arr_DENS= np.zeros((len(phi_a_i[:]), len(phi_b_i[:]))) 

cond_num_arr = np.zeros((len(phi_a_i[:]), len(phi_b_i[:]), len(k_vec)))
max_cond_arr = np.zeros((len(phi_a_i[:]), len(phi_b_i[:])))

for i in range(len(phi_a_i)):
    for j in range(len(phi_b_i)):
        if mat_fine[i,j] == 0: # not near boundary
            continue
#         if mu1 == mu2:
#             continue
        # mu = [mu1, mu2]
        # print("mu: ", mu)
        s_bnd_A = sA_mat[i,j]
        s_bnd_B = sB_mat[i,j]

        phi_Au = phi_Au_mat[i,j]
        phi_Bu = phi_Bu_mat[i,j]
        phi_sf = phi_s[i,j]

        phis = [phi_p_f, phi_Au, phi_Bu, phi_sf]
        # if i == 0 and j == len(phi_Au_arr)-1:
        #     print("SA: ", s_bnd_A)
        #     print("SB: ", s_bnd_B)
        for ik, k in enumerate(k_vec):

            # print(k)

            # s_bnd_A = s_bind_A_ALL[i, j]
            # s_bnd_B = s_bind_B_ALL[i, j]

            # M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)
            # S2_mat = (phi_p / N) * calc_sf2(psol, M2s, [k], competitive)
            # cond_num_arr[i][j][ik] = np.linalg.cond(S2_mat)

            G2 = gamma2_chis(psol, s_bnd_A, s_bnd_B, phis, k)

            # print(G2)
            # s_bnd_A = s_bind_A_ar[i, j]
            # s_bnd_B = s_bind_B_ar[i, j]

            # G2 = gamma2(chrom, s_bnd_A, s_bnd_B, k, chi, competitive)

            
            val, vec = np.linalg.eigh(G2)
            # print(val)
            vec = vec.T
#                 print(vec)
#                 print(vec.T)
#                 print(val)
            min_val = np.min(val)            
            min_eigval_arr[i][j][ik] = min_val
            min_eigvec_arr[i][j][ik] = vec[np.where(val == min_val)]


        # all-k analysis
        # min eigvals, eigvecs at given mu1,mu2 for each k        
        min_vals = min_eigval_arr[i][j][:][np.nonzero(min_eigval_arr[i][j][:])] 
        min_vecs = min_eigvec_arr[i][j][:][np.nonzero(min_eigval_arr[i][j][:])]

        # minimum eigenvalue at given mu1,mu2 across all ks
        minval_allk = np.min(min_vals) 
        
        # store this eigenvalue and corresponding eigenvector
        min_eigval_arr_allk_DENS[i][j] = minval_allk 
        if len(np.where(min_vals==minval_allk)[0]) == 1:
            min_eigvec_arr_allk_DENS[i][j] = min_vecs[np.where(min_vals==minval_allk)]
            k_star = k_vec[np.where(min_vals==minval_allk)]
        else: #if mulitple minimum eigenvalues, just choose the first one (0) or last (-1)
            min_eigvec_arr_allk_DENS[i][j] = min_vecs[np.where(min_vals==minval_allk)][-1,:]
            k_star = k_vec[np.where(min_vals==minval_allk)][-1]

        k_star_arr_DENS[i][j] = k_star

        # max_cond_num = np.max(cond_num_arr[i][j][:])#[np.nonzero(min_eigval_arr[i][j][:])] 
        # max_cond_arr[i][j] = max_cond_num
# setting all non-decomposed/ separated systems to 0
# 
print("fine stability analysis finished")


poly_fluc = min_eigvec_arr_allk_DENS[:,:,0]
poly_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

prot1_fluc = min_eigvec_arr_allk_DENS[:,:,1]
prot1_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

prot2_fluc = min_eigvec_arr_allk_DENS[:,:,2]
prot2_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

k_star_arr_DENS[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = -1 # unphysical value, to indicate outside of spinodal

spinodal = np.copy(k_star_arr_DENS)
spinodal[np.where(spinodal == k_vec[0])] = 0 # macro
spinodal[np.where(spinodal > k_vec[0])] = 1 # micro

k_star_arr_DENS[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = -1 # unphysical value, to indicate outside of spinodal

#-------------------------------------------------------------------------------------------------
# BOUNDARY phase calculation
#------------------------------------------------------------------------------------------------
phases = np.zeros((len(phi_a_i_fine), len(phi_b_i_fine))) - 1 
minF_arr = np.zeros((len(phi_a_i_fine), len(phi_b_i_fine)))

for i in range(len(phi_a_i_fine)):
    real_phase_calc = False # only want a single phase calc in the boundary
    for j in range(len(phi_b_i_fine)):
        if mat_fine[i,j] == 0: # not near boundary
            continue
        if real_phase_calc == True:
            continue

        # if i == 0 or j == 0: # was for getting rid of phi_a_i =0, but no
            # continue

        # if i != mu1_i:
        #     continue
        # if j != mu2_j:
        #     continue

        # print("mu: ", mu1, mu2)
        q_star = k_star_arr_DENS[i,j]
        vec_star = min_eigvec_arr_allk_DENS[i,j]
        
        phi_sf = phi_s[i,j]
        phi_Au = phi_Au_mat[i,j]
        phi_Bu = phi_Bu_mat[i,j]
        # phi_Ab = phi_Ab_mat[i,j]
        # phi_Bb = phi_Bu_mat[i,j]
 
        phis = [phi_p_f, phi_Au, phi_Bu, phi_sf]
                        
        if q_star == -1:
            # print("out of spinodal; no q_star")
            phases[i,j] = 0 # disordered phase- outside of spinodal
        elif q_star == k_vec[0]:
        # elif q_star != k_vec[-1]:
            # print("macrophase sep")
            phases[i,j] = 1 # macrophase sep

        else: #microphse sep
        
            s_bnd_A = sA_mat[i,j,:]#s_bind_A_ALL[i,j,:]
            s_bnd_B = sB_mat[i,j,:]#s_bind_B_ALL[i,j,:]
    
            lam_q = q_star*np.array([1, 0, 0])
            lam_q = np.linalg.norm(lam_q)
    
            cyl_q1 = q_star*np.array([1, 0, 0])
            # cyl_q1 = np.linalg.norm(cyl_q1)
            cyl_q2 = 0.5*q_star*np.array([-1, np.sqrt(3), 0])
            # cyl_q2 = np.linalg.norm(cyl_q2)
            cyl_q3 = 0.5*q_star*np.array([-1, -np.sqrt(3), 0])
            # cyl_q3 = np.linalg.norm(cyl_q3)
            cyl_qs = np.array([cyl_q1, cyl_q2, cyl_q3])
            
            bcc_q1 = 2**(-0.5)*q_star*np.array([1,1,0])
            bcc_q2 = 2**(-0.5)*q_star*np.array([-1,1,0])
            bcc_q3 = 2**(-0.5)*q_star*np.array([0,1,1])
            bcc_q4 = 2**(-0.5)*q_star*np.array([0,1,-1])
            bcc_q5 = 2**(-0.5)*q_star*np.array([1,0,1])
            bcc_q6 = 2**(-0.5)*q_star*np.array([1,0,-1])
            
            
            lam_g3 = 0
            G3 = gamma3(psol, s_bnd_A, s_bnd_B, phis, cyl_qs) # all g3s are eqivlaent
            cyl_g3 = (1/6)  * (1/(3*np.sqrt(3))) * 12 * G3#
            bcc_g3 = (4/(3*np.sqrt(6))) * G3
            
            G4_00 = gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([lam_q, -lam_q, lam_q, -lam_q]))
            lam_g4 = (1/24) * (6) * (1) * G4_00#gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q, -lam_q, lam_q, -lam_q]))        
            
            cyl_g4 = (1/24) * (1/9) *(18*G4_00 + \
              72*gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([cyl_q1, -cyl_q1, cyl_q2, -cyl_q2])))
            
            bcc_g4 = (1/24)* (G4_00 \
                     + 8*gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([bcc_q1, -bcc_q1, bcc_q3, -bcc_q3])) \
                     + 2*gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([bcc_q1, -bcc_q1, bcc_q2, -bcc_q2])) \
                     + 4*gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([bcc_q1, -bcc_q3, bcc_q2, -bcc_q4])) )
    
            lam_g2 = (1/2) * 2 * (1) * gamma2_chis(psol, s_bnd_A, s_bnd_B, phis, q_star)
            cyl_g2 = lam_g2
            bcc_g2 = lam_g2

            # # # PHYSICALLY BOUNDED AMPLITUDE
            # # del_phi = phi - phi_0. phi goes from 0 to phi_p. therefore del_phi(max) = phi_p-phi_0 ; del_phi(min) = - phi_0
            # # need to use a sigmoid (goes 0 to 1) then del_phi = phi_p * sigmoid(amps) - phi_0 ; phi_0 = [f_O, f_A, f_B]
            # # raise Exception("NNED to implement phi right----- look below :D")
            # phi_0 = phi_p * np.array([fo_mat[0][0], fa_mat[0][0], fb_mat[0][0]])
            # initial = [0, 0, 0] # poly, A, B
            # res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", lam_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
            #                                    +  np.einsum("ijkl,i,j,k,l ->", -lam_g4, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
            #                                                 phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
            # lamF_q = res.fun
            
            # res = 0#sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", cyl_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
            #                                    #  + np.einsum("ijk,i,j,k ->", cyl_g3, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
            #                                    # +  np.einsum("ijkl,i,j,k,l ->", -cyl_g4,  phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
            #                                    #              phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
            # cylF_q = 1000000000#res.fun

            # res = 0#sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", bcc_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
            #                                    #  + np.einsum("ijk,i,j,k ->", bcc_g3, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
            #                                    # +  np.einsum("ijkl,i,j,k,l ->", -bcc_g4,  phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
            #                                    #              phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
            # bccF_q = 1000000000#res.fun          



            # print("made it to crit fluc")            
            # CRITICAL FLUCTIATION
            # define a lambda scalar value for a given critica lfluctutaion, then use roots equation to find cirtical points, and eval F at each
            lam_lambda_2 = np.einsum("ij,i,j ->", lam_g2, vec_star, vec_star)
            lam_lambda_4 = np.einsum("ijkl,i,j,k,l ->", -lam_g4, vec_star, vec_star, vec_star, vec_star)
    
            pos_l_amps = np.real(np.roots([4*lam_lambda_4,0,2*lam_lambda_2,0]))        
            lamF = min(np.real( lam_lambda_2 * pos_l_amps**2 + lam_lambda_4 * pos_l_amps**4))
    
            
            cyl_lambda_2 = np.einsum("ij,i,j ->", cyl_g2, vec_star, vec_star)
            cyl_lambda_3 = np.einsum("ijk,i,j,k->", cyl_g3, vec_star, vec_star, vec_star)
            cyl_lambda_4 = np.einsum("ijkl,i,j,k,l ->", -cyl_g4, vec_star, vec_star, vec_star, vec_star)
    
            pos_c_amps = np.real(np.roots([4*cyl_lambda_4,3*cyl_lambda_3,2*cyl_lambda_2,0]))        
            cylF = min(np.real( cyl_lambda_2 * pos_c_amps**2 + cyl_lambda_3*pos_c_amps**3 + cyl_lambda_4 * pos_c_amps**4))
    
            
            bcc_lambda_2 = np.einsum("ij,i,j ->", bcc_g2, vec_star, vec_star)
            bcc_lambda_3 = np.einsum("ijk,i,j,k->", bcc_g3, vec_star, vec_star, vec_star)
            bcc_lambda_4 = np.einsum("ijkl,i,j,k,l ->", -bcc_g4, vec_star, vec_star, vec_star, vec_star)
    
            pos_b_amps = np.real(np.roots([4*bcc_lambda_4,3*bcc_lambda_3,2*bcc_lambda_2,0]))        
            bccF = min(np.real( bcc_lambda_2 * pos_b_amps**2 + bcc_lambda_3*pos_b_amps**3 + bcc_lambda_4 * pos_b_amps**4))        


            

            # # # BASINHOPPING UNRESTRICTED works at low chi
        
            # initial = [0, 0, 0] # poly, A, B
            # res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", lam_g2, amps, amps) \
            #                                    +  np.einsum("ijkl,i,j,k,l ->", -lam_g4, amps, amps, amps, amps)), initial)
            # lamF_q = res.fun
            # res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", cyl_g2, amps, amps) \
            #                                     + np.einsum("ijk,i,j,k ->", cyl_g3, amps, amps, amps) \
            #                                    +  np.einsum("ijkl,i,j,k,l ->", -cyl_g4, amps, amps, amps, amps)), initial)
            # cylF_q = res.fun
    
            # res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", bcc_g2, amps, amps) \
            #                                     + np.einsum("ijk,i,j,k ->", bcc_g3, amps, amps, amps) \
            #                                    +  np.einsum("ijkl,i,j,k,l ->", -bcc_g4, amps, amps, amps, amps)), initial)
            # bccF_q = res.fun            


            
            minF = np.min([lamF, cylF, bccF, 0])
            minF_arr[i,j] = minF
            # print("--------------------------------------------------------------------")
            print("energies:")
            print([lamF, cylF, bccF, 0])
            # print("--------------------------------------------------------------------")

            # # print([lamF])
            # # print([lamF, cylF])#, bccF])
            if minF == 0:
                # raise Exception("phase sep not stable in spinodal??")
                phases[i,j] = -2
            elif minF == lamF:
                phases[i,j] = 2
            elif minF == cylF:
                phases[i,j] = 3
            elif minF == bccF:
                phases[i,j] = 4
            real_phase_calc = True


        np.save("expl_bind_beaker_towers_minF_boundary_ID="+str(ID_number), minF_arr)
        np.save("expl_bind_beaker_towers_phases_boundary_ID="+str(ID_number), phases)
            
# #-------------------------------------------------------------------------------------------------
# # FULL phase calculation
# #------------------------------------------------------------------------------------------------
# phases = np.zeros((len(phi_a_i), len(phi_b_i))) - 1 
# minF_arr = np.zeros((len(phi_a_i), len(phi_b_i)))

# for i in range(len(phi_a_i)):
#     for j in range(len(phi_b_i)):
#          if i == 0 or j == 0:
#              continue

#         # if i != mu1_i:
#         #     continue
#         # if j != mu2_j:
#         #     continue

#         # print("mu: ", mu1, mu2)
#         q_star = k_star_arr_DENS[i,j]
#         vec_star = min_eigvec_arr_allk_DENS[i,j]
        
#         phi_sf = phi_s[i,j]
#         phi_Au = phi_Au_mat[i,j]
#         phi_Bu = phi_Bu_mat[i,j]
#         # phi_Ab = phi_Ab_mat[i,j]
#         # phi_Bb = phi_Bu_mat[i,j]
 
#         phis = [phi_p_f, phi_Au, phi_Bu, phi_sf]
                
#         # TODO- define phis
        
#         if q_star == -1:
#             print("out of spinodal; no q_star")
#             phases[i,j] = 0 # disordered phase- outside of spinodal
#         elif q_star == k_vec[0]:
#         # elif q_star != k_vec[-1]:
#             print("macrophase sep")
#             phases[i,j] = 1 # macrophase sep

#         else: #microphse sep
        
#             s_bnd_A = sA_mat[i,j,:]#s_bind_A_ALL[i,j,:]
#             s_bnd_B = sB_mat[i,j,:]#s_bind_B_ALL[i,j,:]
    
#             lam_q = q_star*np.array([1, 0, 0])
#             lam_q = np.linalg.norm(lam_q)
    
#             cyl_q1 = q_star*np.array([1, 0, 0])
#             # cyl_q1 = np.linalg.norm(cyl_q1)
#             cyl_q2 = 0.5*q_star*np.array([-1, np.sqrt(3), 0])
#             # cyl_q2 = np.linalg.norm(cyl_q2)
#             cyl_q3 = 0.5*q_star*np.array([-1, -np.sqrt(3), 0])
#             # cyl_q3 = np.linalg.norm(cyl_q3)
#             cyl_qs = np.array([cyl_q1, cyl_q2, cyl_q3])
            
#             bcc_q1 = 2**(-0.5)*q_star*np.array([1,1,0])
#             bcc_q2 = 2**(-0.5)*q_star*np.array([-1,1,0])
#             bcc_q3 = 2**(-0.5)*q_star*np.array([0,1,1])
#             bcc_q4 = 2**(-0.5)*q_star*np.array([0,1,-1])
#             bcc_q5 = 2**(-0.5)*q_star*np.array([1,0,1])
#             bcc_q6 = 2**(-0.5)*q_star*np.array([1,0,-1])
            
            
#             lam_g3 = 0
#             G3 = gamma3(psol, s_bnd_A, s_bnd_B, phis, cyl_qs) # all g3s are eqivlaent
#             cyl_g3 = (1/6)  * (1/(3*np.sqrt(3))) * 12 * G3#
#             bcc_g3 = (4/(3*np.sqrt(6))) * G3
            
#             G4_00 = gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([lam_q, -lam_q, lam_q, -lam_q]))
#             lam_g4 = (1/24) * (6) * (1) * G4_00#gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q, -lam_q, lam_q, -lam_q]))        
            
#             cyl_g4 = (1/24) * (1/9) *(18*G4_00 + \
#               72*gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([cyl_q1, -cyl_q1, cyl_q2, -cyl_q2])))
            
#             bcc_g4 = (1/24)* (G4_00 \
#                      + 8*gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([bcc_q1, -bcc_q1, bcc_q3, -bcc_q3])) \
#                      + 2*gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([bcc_q1, -bcc_q1, bcc_q2, -bcc_q2])) \
#                      + 4*gamma4(psol, s_bnd_A, s_bnd_B, phis, np.array([bcc_q1, -bcc_q3, bcc_q2, -bcc_q4])) )
    
#             lam_g2 = (1/2) * 2 * (1) * gamma2_chis(psol, s_bnd_A, s_bnd_B, phis, q_star)
#             cyl_g2 = lam_g2
#             bcc_g2 = lam_g2

#             # # # PHYSICALLY BOUNDED AMPLITUDE
#             # # del_phi = phi - phi_0. phi goes from 0 to phi_p. therefore del_phi(max) = phi_p-phi_0 ; del_phi(min) = - phi_0
#             # # need to use a sigmoid (goes 0 to 1) then del_phi = phi_p * sigmoid(amps) - phi_0 ; phi_0 = [f_O, f_A, f_B]
#             # # raise Exception("NNED to implement phi right----- look below :D")
#             # phi_0 = phi_p * np.array([fo_mat[0][0], fa_mat[0][0], fb_mat[0][0]])
#             # initial = [0, 0, 0] # poly, A, B
#             # res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", lam_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
#             #                                    +  np.einsum("ijkl,i,j,k,l ->", -lam_g4, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
#             #                                                 phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
#             # lamF_q = res.fun
            
#             # res = 0#sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", cyl_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
#             #                                    #  + np.einsum("ijk,i,j,k ->", cyl_g3, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
#             #                                    # +  np.einsum("ijkl,i,j,k,l ->", -cyl_g4,  phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
#             #                                    #              phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
#             # cylF_q = 1000000000#res.fun

#             # res = 0#sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", bcc_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
#             #                                    #  + np.einsum("ijk,i,j,k ->", bcc_g3, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
#             #                                    # +  np.einsum("ijkl,i,j,k,l ->", -bcc_g4,  phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
#             #                                    #              phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
#             # bccF_q = 1000000000#res.fun          




#             print("made it to crit fluc")            
#             # CRITICAL FLUCTIATION
#             # define a lambda scalar value for a given critica lfluctutaion, then use roots equation to find cirtical points, and eval F at each
#             lam_lambda_2 = np.einsum("ij,i,j ->", lam_g2, vec_star, vec_star)
#             lam_lambda_4 = np.einsum("ijkl,i,j,k,l ->", -lam_g4, vec_star, vec_star, vec_star, vec_star)
    
#             pos_l_amps = np.real(np.roots([4*lam_lambda_4,0,2*lam_lambda_2,0]))        
#             lamF = min(np.real( lam_lambda_2 * pos_l_amps**2 + lam_lambda_4 * pos_l_amps**4))
    
            
#             cyl_lambda_2 = np.einsum("ij,i,j ->", cyl_g2, vec_star, vec_star)
#             cyl_lambda_3 = np.einsum("ijk,i,j,k->", cyl_g3, vec_star, vec_star, vec_star)
#             cyl_lambda_4 = np.einsum("ijkl,i,j,k,l ->", -cyl_g4, vec_star, vec_star, vec_star, vec_star)
    
#             pos_c_amps = np.real(np.roots([4*cyl_lambda_4,3*cyl_lambda_3,2*cyl_lambda_2,0]))        
#             cylF = min(np.real( cyl_lambda_2 * pos_c_amps**2 + cyl_lambda_3*pos_c_amps**3 + cyl_lambda_4 * pos_c_amps**4))
    
            
#             bcc_lambda_2 = np.einsum("ij,i,j ->", bcc_g2, vec_star, vec_star)
#             bcc_lambda_3 = np.einsum("ijk,i,j,k->", bcc_g3, vec_star, vec_star, vec_star)
#             bcc_lambda_4 = np.einsum("ijkl,i,j,k,l ->", -bcc_g4, vec_star, vec_star, vec_star, vec_star)
    
#             pos_b_amps = np.real(np.roots([4*bcc_lambda_4,3*bcc_lambda_3,2*bcc_lambda_2,0]))        
#             bccF = min(np.real( bcc_lambda_2 * pos_b_amps**2 + bcc_lambda_3*pos_b_amps**3 + bcc_lambda_4 * pos_b_amps**4))        


            

#             # # # BASINHOPPING UNRESTRICTED works at low chi
        
#             # initial = [0, 0, 0] # poly, A, B
#             # res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", lam_g2, amps, amps) \
#             #                                    +  np.einsum("ijkl,i,j,k,l ->", -lam_g4, amps, amps, amps, amps)), initial)
#             # lamF_q = res.fun
#             # res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", cyl_g2, amps, amps) \
#             #                                     + np.einsum("ijk,i,j,k ->", cyl_g3, amps, amps, amps) \
#             #                                    +  np.einsum("ijkl,i,j,k,l ->", -cyl_g4, amps, amps, amps, amps)), initial)
#             # cylF_q = res.fun
    
#             # res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", bcc_g2, amps, amps) \
#             #                                     + np.einsum("ijk,i,j,k ->", bcc_g3, amps, amps, amps) \
#             #                                    +  np.einsum("ijkl,i,j,k,l ->", -bcc_g4, amps, amps, amps, amps)), initial)
#             # bccF_q = res.fun            


            
#             minF = np.min([lamF, cylF, bccF, 0])
#             minF_arr[i,j] = minF
#             # print("--------------------------------------------------------------------")
#             # print("energies:")
#             print([lamF, cylF, bccF, 0])
#             # print("--------------------------------------------------------------------")

#             # # print([lamF])
#             # # print([lamF, cylF])#, bccF])
#             if minF == 0:
#                 # raise Exception("phase sep not stable in spinodal??")
#                 phases[i,j] = -2
#             elif minF == lamF:
#                 phases[i,j] = 2
#             elif minF == cylF:
#                 phases[i,j] = 3
#             elif minF == bccF:
#                 phases[i,j] = 4

#         np.save("expl_bind_beaker_towers_minF_ID="+str(ID_number), minF_arr)
#         np.save("expl_bind_beaker_towers_phases_ID="+str(ID_number), phases)
            