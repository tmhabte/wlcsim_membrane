import multiprocessing
import os
import numpy as np
from OABS_vertex_calc import *

def worker1(psol, chi_AB, result):

    competitive = True
    b = psol.b
    N = psol.N
    phi_p = psol.phi_p

    klog_min = -2.5
    klog_max = -.1
    klog_num = 30
    k_vec = np.logspace(klog_min, klog_max, klog_num) / b
    s_bind_A_ALL, s_bind_B_ALL = calc_binding_states(psol)

    mu1_arr = psol.mu1_arr
    mu2_arr = psol.mu2_arr

    assert competitive == True
    # chi = 0#5*N/N

    min_eigval_arr = np.zeros((len(mu1_arr[:]), len(mu2_arr[:]), len(k_vec)))

    # min_eigval_arr_allk_DENS = np.zeros((len(mu1_arr[:]), len(mu2_arr[:])))

    min_eigvec_arr = np.zeros((len(mu1_arr[:]), len(mu2_arr[:]), len(k_vec), 3))
    min_eigvec_arr_allk_DENS = np.zeros((len(mu1_arr[:]), len(mu2_arr[:]), 3))

    # k_star_arr_DENS= np.zeros((len(mu1_arr[:]), len(mu2_arr[:])))
    cond_num_arr = np.zeros((len(mu1_arr[:]), len(mu2_arr[:]), len(k_vec)))
    max_cond_arr = np.zeros((len(mu1_arr[:]), len(mu2_arr[:])))

    k_star_arr_DENS = np.zeros(len(mu2_arr[:])) #result
    # k_star_arr_DENS = result
    min_eigval_arr_allk_DENS = np.zeros(len(mu2_arr[:]))

    chi_AB = chi_AB#13 / (phi_p*N) #(v_int[0,1] - 0.5*(v_int[0,0] + v_int[1,1]))*Vol_int/100
    chi_AS = 00 / N #834 / N
    chis = [chi_AB, chi_AS]
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
            min_eigval_arr_allk_DENS[j] = minval_allk
            min_eigvec_arr_allk_DENS[i][j] = min_vecs[np.where(min_vals==minval_allk)]

            k_star = k_vec[np.where(min_vals==minval_allk)]
            k_star_arr_DENS[j] = k_star

            # max_cond_num = np.max(cond_num_arr[i][j][:])#[np.nonzero(min_eigval_arr[i][j][:])]
            # max_cond_arr[i][j] = max_cond_num

    # indices where minimum eigenvalue is positive - solution stable
    sign_index = (np.where(np.sign(min_eigval_arr_allk_DENS) == 1))[0]

    for k in range(len(k_star_arr_DENS)):
        if k in sign_index:
            k_star_arr_DENS[k] = -1

    phases = result
    # phases = np.zeros((len(mu1_arr), len(mu2_arr))) - 1
    minF_arr = np.zeros((len(mu1_arr), len(mu2_arr)))

    print("entering phase calc")

    for i, mu1 in enumerate(mu1_arr):
        for j, mu2 in enumerate(mu2_arr):
            # if i != mu1_i:
            #     continue
            # if j != mu2_j:
            #     continue
            print("--------------------------------------------------------------------")
            print("mu: ", mu1, mu2)
            q_star = k_star_arr_DENS[j]
            vec_star = min_eigvec_arr_allk_DENS[i,j]

            if q_star == -1:
                print("out of spinodal; no q_star")
                phases[j] = 0 # disordered phase- outside of spinodal
            elif q_star == k_vec[0]:
            # elif q_star != k_vec[-1]:
                print("macrophase sep")
                phases[j] = 1 # macrophase sep

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


                # # BASINHOPPING UNRESTRICTED

                initial = [0, 0, 0] # poly, A, B
                res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", lam_g2, amps, amps) \
                                                +  np.einsum("ijkl,i,j,k,l ->", -lam_g4, amps, amps, amps, amps)), initial)
                lamF_q = res.fun
                res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", cyl_g2, amps, amps) \
                                                    + np.einsum("ijk,i,j,k ->", cyl_g3, amps, amps, amps) \
                                                +  np.einsum("ijkl,i,j,k,l ->", -cyl_g4, amps, amps, amps, amps)), initial)
                cylF_q = res.fun

                res = sp.optimize.basinhopping(lambda amps: np.real(np.einsum("ij,i,j ->", bcc_g2, amps, amps) \
                                                    + np.einsum("ijk,i,j,k ->", bcc_g3, amps, amps, amps) \
                                                +  np.einsum("ijkl,i,j,k,l ->", -bcc_g4, amps, amps, amps, amps)), initial)
                bccF_q = res.fun

                minF = np.min([lamF_q, cylF_q, bccF_q, 0])
                minF_arr[i,j] = minF
                print("energies:")
                print([lamF_q, cylF_q, bccF_q, 0])
                print("--------------------------------------------------------------------")

                # print([lamF])
                # print([lamF, cylF])#, bccF])
                if minF == 0:
                    # raise Exception("phase sep not stable in spinodal??")
                    phases[j] = -2
                elif minF == lamF_q:
                    phases[j] = 2
                elif minF == cylF_q:
                    phases[j] = 3
                elif minF == bccF_q:
                    phases[j] = 4

if __name__ == "__main__": 
    # printing main program process id
    print("ID of main process: {}".format(os.getpid()))
    n_bind = 2 #types of proteins/marks
    e_m = np.array([1.52, 1.52]) #binding energy FOR F_BIND_ALT
    v_int =  np.array([[-4, 4], [4, -4]])
    phi_p = 0.7
    M = 50
    nm = M
    pa_vec = np.arange(0, nm, 1) / (nm-1)
    pb_vec = 1-pa_vec
    poly_marks = [pa_vec, pb_vec]

    v_s = 1
    v_m = 1
    N = 5000
    b = 1

    chi_AB = 13 / (phi_p*N)
    competitive = True

    # #TESTING
    # mu_max_1 = -2.889#-2.8299#8#0.1 #10
    # mu_min_1 = -2.9#-9
    # del_mu_1 = .01#.5 #0.25
    # mu_max_2 = -2.9099#8#0.1 #10
    # mu_min_2 = -2.91#-9
    # del_mu_2 = .01#.5 #0.25

    # tipzoom
    mu_max_1 = -2.849#-2.8299#8#0.1 #10
    mu_min_1 = -2.95#-9
    del_mu_1 = .001#.5 #0.25
    mu_max_2 = mu_max_1 
    mu_min_2 = mu_min_1 
    del_mu_2 = del_mu_1  

    num_proc = 2 

    psol_arr = np.zeros(num_proc, dtype = "object")

    mu1_arr = np.arange(mu_min_1, mu_max_1, del_mu_1)
    mu1_splits = np.array_split(mu1_arr, num_proc)
    mu2_arr_sub = np.arange(mu_min_2, mu_max_2, del_mu_2) # same for all poly solns

    result_arr = np.zeros(num_proc, dtype = "object")
    proc_arr = np.zeros(num_proc, dtype = "object")
    for i in range(num_proc):
        # #intiialze appropriate psol object
        mu1_arr_sub = mu1_splits[i]
        psol = Polymer_soln(n_bind, v_int, e_m, phi_p, poly_marks, mu1_arr_sub, mu2_arr_sub, v_s, v_m, N, b)
        psol_arr[i] = psol
        
        #initialize corresponding results array
        result_i = multiprocessing.Array('d', len(mu2_arr_sub))
        result_arr[i] = result_i

        #start process
        proc = multiprocessing.Process(target=worker1, args=(psol, chi_AB, result_i))
        proc.start()
        proc_arr[i] = proc
    
    for i in range(num_proc):
        # wait until processes are finished
        proc = proc_arr[i]
        proc.join()

    #combine results
    phases_master = np.zeros((num_proc, len(mu2_arr_sub)))
    for i in range(num_proc):
        result = np.array(result_arr[i])
        phases_master[i,:] = result
    # np.save("multipros_phases_test", phases_master)
    np.save("OABS_phases_arr_tipzoom_chiABphipNeq"+str(int(chi_AB*phi_p*N))+"N="+str(int(N)), phases_master)
