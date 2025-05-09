import numpy as np
from OABS_vertex_calc import *
import time
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
from matplotlib.colors import LogNorm

start = time.time()

n_bind = 2 #types of proteins/marks
e_m = np.array([1.52, 1.52]) #binding energy FOR F_BIND_ALT
v_int =  np.array([[-4, 4], [4, -4]])
phi_p = 0.7
M = 50
nm = M
pa_vec = np.arange(0, nm, 1) / (nm-1)
pb_vec = 1-pa_vec
poly_marks = [pa_vec, pb_vec]

mu_max_1 = -2.6#8#0.1 #10
mu_min_1 = -3#-9
del_mu_1 = .01#.5 #0.25

mu_max_2 = -2.6#8#0.1 #10
mu_min_2 = -3#-9
del_mu_2 = .01#.5 #0.25

mu1_arr = np.arange(mu_min_1, mu_max_1, del_mu_1)
mu2_arr = np.arange(mu_min_2, mu_max_2, del_mu_2)
v_s = 1
v_m = 1
N = 5000
b = 1

chi_AB = 50 / (phi_p*N) #(v_int[0,1] - 0.5*(v_int[0,0] + v_int[1,1]))*Vol_int/100
chi_AS = 00 / N #834 / N
chis = [chi_AB, chi_AS]

print("phys bound, N = %s, chi_AB = %s, phi_p = %s" % (N, chi_AB, phi_p))

psol = Polymer_soln(n_bind, v_int, e_m, phi_p, poly_marks, mu1_arr, mu2_arr, v_s, v_m, N, b)

competitive = True

klog_min = -2.5
klog_max = -.1
klog_num = 30
k_vec = np.logspace(klog_min, klog_max, klog_num) / b
s_bind_A_ALL, s_bind_B_ALL = calc_binding_states(psol)

fa_mat = np.zeros((len(mu1_arr), len(mu2_arr))) - 1
fb_mat = np.zeros((len(mu1_arr), len(mu2_arr))) - 1
fab_mat = np.zeros((len(mu1_arr), len(mu2_arr))) - 1
fo_mat = np.zeros((len(mu1_arr), len(mu2_arr))) - 1

for i, mu1 in enumerate(mu1_arr[:]):
    for j, mu2 in enumerate(mu2_arr[:]):
        s_Abnd_ar = s_bind_A_ALL[i,j]
        s_Bbnd_ar = s_bind_B_ALL[i,j]
        f_a, f_b, f_ab, f_o = calc_fas(s_Abnd_ar, s_Bbnd_ar)
        fa_mat[i,j] = f_a
        fb_mat[i,j] = f_b
        fab_mat[i,j] = f_ab
        fo_mat[i,j] = f_o

print(np.round((time.time() - start)/(60),4), "mins elapsed")

# k* stability analysis NUMBER DENSITY THEORY
assert competitive == True
# chi = 0#5*N/N

min_eigval_arr = np.zeros((len(mu1_arr[:]), len(mu2_arr[:]), len(k_vec)))

min_eigval_arr_allk_DENS = np.zeros((len(mu1_arr[:]), len(mu2_arr[:])))
min_eigval_arr_allk_ps = np.zeros((len(mu1_arr[:]), len(mu2_arr[:])))

min_eigvec_arr = np.zeros((len(mu1_arr[:]), len(mu2_arr[:]), len(k_vec), 3))
min_eigvec_arr_allk_DENS = np.zeros((len(mu1_arr[:]), len(mu2_arr[:]), 3))

k_star_arr_DENS= np.zeros((len(mu1_arr[:]), len(mu2_arr[:]))) 

cond_num_arr = np.zeros((len(mu1_arr[:]), len(mu2_arr[:]), len(k_vec)))
max_cond_arr = np.zeros((len(mu1_arr[:]), len(mu2_arr[:])))


for i, mu1 in enumerate(mu1_arr):
    for j, mu2 in enumerate(mu2_arr):
#         if mu1 == mu2:
#             continue
        mu = [mu1, mu2]
        # print("mu: ", mu)

        for ik, k in enumerate(k_vec):

            s_bnd_A = s_bind_A_ALL[i, j]
            s_bnd_B = s_bind_B_ALL[i, j]

            M2s = calc_mon_mat_2(s_bnd_A, s_bnd_B, competitive)
            S2_mat = (phi_p / N) * calc_sf2(psol, M2s, [k], competitive)
            cond_num_arr[i][j][ik] = np.linalg.cond(S2_mat)

            G2 = gamma2_chis(psol, s_bnd_A, s_bnd_B, k, chis, competitive)
            
            val, vec = np.linalg.eigh(G2)
            vec = vec.T

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
        min_eigvec_arr_allk_DENS[i][j] = min_vecs[np.where(min_vals==minval_allk)]

        k_star = k_vec[np.where(min_vals==minval_allk)]
        k_star_arr_DENS[i][j] = k_star

        max_cond_num = np.max(cond_num_arr[i][j][:])#[np.nonzero(min_eigval_arr[i][j][:])] 
        max_cond_arr[i][j] = max_cond_num
# setting all non-decomposed/ separated systems to 0 
# poly_fluc = min_eigvec_arr_allk_DENS[:,:,0]
# poly_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

# prot1_fluc = min_eigvec_arr_allk_DENS[:,:,1]
# prot1_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

# prot2_fluc = min_eigvec_arr_allk_DENS[:,:,2]
# prot2_fluc[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = 0

k_star_arr_DENS[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = -1 # unphysical value, to indicate outside of spinodal
# np.save("OABS_k_star_arr_DENS", k_star_arr_DENS)
# np.save("OABS_min_eigval_arr_allk_DENS", min_eigval_arr_allk_DENS)


k_star_arr_DENS[np.where(np.sign(min_eigval_arr_allk_DENS) == 1) ] = -1 # unphysical value, to indicate outside of spinodal

phases = np.zeros((len(mu1_arr), len(mu2_arr))) - 1 
minF_arr = np.zeros((len(mu1_arr), len(mu2_arr))) 

print("entering phase calc")

def sigmoid(z):
    return 1/(1 + np.exp(-z))


for i, mu1 in enumerate(mu1_arr):
    for j, mu2 in enumerate(mu2_arr):
        # if i != mu1_i:
        #     continue
        # if j != mu2_j:
        #     continue

        print("mu: ", mu1, mu2)
        q_star = k_star_arr_DENS[i,j]
        vec_star = min_eigvec_arr_allk_DENS[i,j]
        
        if q_star == -1:
            print("out of spinodal; no q_star")
            phases[i,j] = 0 # disordered phase- outside of spinodal
        elif q_star == k_vec[0]:
        # elif q_star != k_vec[-1]:
            print("macrophase sep")
            phases[i,j] = 1 # macrophase sep

        else: #microphse sep
        
            s_bnd_A = s_bind_A_ALL[i,j,:]
            s_bnd_B = s_bind_B_ALL[i,j,:]
            
            lam_q = q_star*np.array([1, 0, 0])
    
            cyl_q1 = q_star*np.array([1, 0, 0])
            cyl_q2 = 0.5*q_star*np.array([-1, np.sqrt(3), 0])
            cyl_q3 = 0.5*q_star*np.array([-1, -np.sqrt(3), 0])
            cyl_qs = np.array([cyl_q1, cyl_q2, cyl_q3])
            
            bcc_q1 = 2**(-0.5)*q_star*np.array([1,1,0])
            bcc_q2 = 2**(-0.5)*q_star*np.array([-1,1,0])
            bcc_q3 = 2**(-0.5)*q_star*np.array([0,1,1])
            bcc_q4 = 2**(-0.5)*q_star*np.array([0,1,-1])
            bcc_q5 = 2**(-0.5)*q_star*np.array([1,0,1])
            bcc_q6 = 2**(-0.5)*q_star*np.array([1,0,-1])
            
            
            lam_g3 = 0
            G3 = gamma3(psol, s_bnd_A, s_bnd_B, cyl_qs, competitive) # all g3s are eqivlaent
            cyl_g3 = (1/6)  * (1/(3*np.sqrt(3))) * 12 * G3#
            bcc_g3 = (4/(3*np.sqrt(6))) * G3
            
            G4_00 = gamma4(psol, s_bnd_A, s_bnd_B, np.array([lam_q, -lam_q, lam_q, -lam_q]), competitive)
            lam_g4 = (1/24) * (6) * (1) * G4_00#gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q, -lam_q, lam_q, -lam_q]))        
            
            cyl_g4 = (1/24) * (1/9) *(18*G4_00 + \
              72*gamma4(psol, s_bnd_A, s_bnd_B, np.array([cyl_q1, -cyl_q1, cyl_q2, -cyl_q2]), competitive))
            
            bcc_g4 = (1/24)* (G4_00 \
                     + 8*gamma4(psol, s_bnd_A, s_bnd_B, np.array([bcc_q1, -bcc_q1, bcc_q3, -bcc_q3]), competitive) \
                     + 2*gamma4(psol, s_bnd_A, s_bnd_B, np.array([bcc_q1, -bcc_q1, bcc_q2, -bcc_q2]), competitive) \
                     + 4*gamma4(psol, s_bnd_A, s_bnd_B, np.array([bcc_q1, -bcc_q3, bcc_q2, -bcc_q4]), competitive) )
    
            # lam_g2 = (1/2) * 2 * (1) * gamma2(chrom, s_bnd_A, s_bnd_B, q_star, chi, competitive)
            lam_g2 = (1/2) * 2 * (1) * gamma2_chis(psol, s_bnd_A, s_bnd_B, q_star, chis, competitive)
            cyl_g2 = lam_g2
            bcc_g2 = lam_g2

            # # PHYSICALLY BOUNDED AMPLITUDE
            # del_phi = phi - phi_0. phi goes from 0 to phi_p. therefore del_phi(max) = phi_p-phi_0 ; del_phi(min) = - phi_0
            # need to use a sigmoid (goes 0 to 1) then del_phi = phi_p * sigmoid(amps) - phi_0 ; phi_0 = [f_O, f_A, f_B]
            
            phi_0 = phi_p * np.array([fo_mat[0][0], fa_mat[0][0], fb_mat[0][0]])
            initial = [0, 0, 0] # poly, A, B
            res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", lam_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
                                               +  np.einsum("ijkl,i,j,k,l ->", -lam_g4, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
                                                            phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
            lamF = res.fun
            res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", cyl_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
                                                + np.einsum("ijk,i,j,k ->", cyl_g3, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
                                               +  np.einsum("ijkl,i,j,k,l ->", -cyl_g4,  phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
                                                            phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
            cylF = res.fun

            res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", bcc_g2, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
                                                + np.einsum("ijk,i,j,k ->", bcc_g3, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0) \
                                               +  np.einsum("ijkl,i,j,k,l ->", -bcc_g4,  phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0, \
                                                            phi_p*sigmoid(amps)-phi_0, phi_p*sigmoid(amps)-phi_0)), initial)
            bccF = res.fun  

            minF = np.min([lamF, cylF, bccF, 0])
            minF_arr[i,j] = minF
            print("energies:")
            print([lamF, cylF, bccF, 0])
            print("--------------------------------------------------------------------")

            # print([lamF])
            # print([lamF, cylF])#, bccF])
            if minF == 0:
                # raise Exception("phase sep not stable in spinodal??")
                phases[i,j] = -2
            elif minF == lamF:
                phases[i,j] = 2
            elif minF == cylF:
                phases[i,j] = 3
            elif minF == bccF:
                phases[i,j] = 4

# np.save("OAB,AB,S_phases_arr_zoomed", phases)
# np.save("OAB,AB,S_min_F_arr_zoomed", minF_arr)

np.save("OABS_phases_arr_physbound_chiABphipNeq"+str(int(chi_AB*phi_p*N))+"N="+str(int(N)), phases)
np.save("OABS_min_F_arr_physbound_chiABphipNeq"+str(int(chi_AB*phi_p*N))+"N="+str(int(N)), minF_arr)

