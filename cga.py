import numpy as np
import random
import pandas as pd
from scipy import stats
from itertools import product
from scipy import optimize

import sys
sys.path.append("./vertex_subroutines")
from GAMcalc import *  # Code for calculating vertex functions
import propagator  # propagator object used to store pre-calculated values
import wignerD as wd # wigner D object used to store pre-calculated values

pset=propagator.prop_set(nlam=10) # nlam sepcifies number of angular eigenvalues
wigset = wd.wigner_d_vals()

def r2(N):
    return N - 0.5*(1-np.exp(-2*N))

def wlc_cga_vect(N, n_p, n_b, f_a, num_snapshots):
    #num_snapshots = 1000
    #n_p = 15
    #n_b = 200

    n_b_calc = n_b
    l_0 = (N*2)/n_b#.01 # length_kuhn = (10 l_k) = (20 l_p) = (200 l_0) ### length_kuhn = (1 l_k) = (2 l_p) = (200 l_0) #
    l_p = 1
    length_kuhn = n_b *l_0 / (l_p*2)
    kappa = l_p/l_0
    all_snaps_vect_copoly = np.zeros(num_snapshots, dtype=object)

    #f_a = 0.5

    axes_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin_1 = np.array([0, 0, 0])
    ###
    ###
    ###
    ###trying to vectorize snaps, so that sfs can be calcd faster
    output = np.zeros([n_p*n_b*num_snapshots, 3])
    r1 = np.array([0, 0, 0])
    output[::n_b] = r1
    
    phi = 2*np.pi*np.random.rand(n_p*num_snapshots)
    theta = np.arccos(stats.uniform(-1, 2).rvs(n_p*num_snapshots))
    u2 = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T
    r2 = u2 * l_0
    output[1::n_b] = r2
    for bead in range(n_b-2):
        r = np.random.rand(n_p*num_snapshots)
        rho = (1/kappa)*np.log(np.exp(-kappa)+r*(np.exp(kappa)-np.exp(-kappa)))
        theta = np.arccos(rho)
        phi = 2*np.pi*np.random.rand(n_p*num_snapshots)

        z_prime = output[bead+1::n_b] - output[bead::n_b] #previous bond vector
        z_prime = z_prime/np.linalg.norm(z_prime, axis = -1)[:, np.newaxis] #normalize each row in matrix - could just /l0
        #modified gram-schmidt
        x_prime = np.random.randn(n_p*num_snapshots, 3)
        x_prime -= np.sum(x_prime*z_prime, axis=1)[:, None] * z_prime #np.sum is row-wise dot product
        x_prime = x_prime/np.linalg.norm(x_prime, axis = -1)[:, np.newaxis]

        y_prime = np.cross(z_prime, x_prime)

        r_prime = (l_0 * np.array([np.sin(theta)*np.cos(phi), 0+np.sin(theta)*np.sin(phi), np.cos(theta)])).T # u_bead where (phi = 0)
        #even though x and y axes are random, ensuring randomness by including phi rotation.

        #convert r_prime from x'y'z' to xyz 
        
        # o1 + m1.p1 = o2 + m2.p2
        origin_2 = output[bead+1::n_b]
        axes_2 = np.stack((x_prime, y_prime, z_prime), axis=-1) #axis = -1 transposes
        r_bead = origin_2 + np.einsum('ipq,iq->ip',axes_2,r_prime) #element-wise dot product
        output[bead+2::n_b] = r_bead
    
    ## gen bead identities
    bead_identity = np.zeros(n_p*n_b*num_snapshots)
    for i in range(n_p*num_snapshots):
        bead_identity[i*n_b:int(i*n_b + n_b*f_a)] = np.ones(int(n_b*f_a))
    ##
    bead_identity = np.array([[i] for i in bead_identity])

    return np.append(output, bead_identity, axis=1)

def calc_elastic_constants(DEL):
    # EPS_B, DEL_GAM, EPS_PAR, EPS_PERP, ETA
    params = np.loadtxt('dssWLCparams', usecols=range(6))
    if (DEL < params[0][0] or DEL > params[-1][0]):
        raise Exception("DEL must be b/w 0.01, 10")
    
    row = 0
    while (DEL > params[row][0]):
        row += 1
    interp_params = np.zeros(5)
    for i in range(1, 6):
        slope = (params[row][i] - params[row-1][i])/ (params[row][0] - params[row-1][0])
        interp_params[i-1] = slope*(DEL - params[row][0]) + params[row][i]
    return interp_params

def dsswlc_cga(N, n_p, n_b, f_a, num_snapshots, DEL):
    ## EDIT SO THAT N_B IS SOMETHING YOU CALCULATE given DEL
    if (num_snapshots != 1):
        raise Exception("only one shapshot- increase n_p")
        
#     n_b_calc = n_b
#     l_0 = (N*2) / n_b#.01 # length_kuhn = (10 l_k) = (20 l_p) = (200 l_0) ### length_kuhn = (1 l_k) = (2 l_p) = (200 l_0) #
    l_p = 1
#     length_kuhn = n_b * l_0 / (l_p*2)
#     kappa = l_p/l_0
    #all_snaps_vect_copoly = np.zeros(num_snapshots, dtype=object)

    n_b = int( (N*2)/DEL ) #N in kuhn lenghts, DEL in persistence lengths
    
    if (n_b == 0):
        raise Exception("DEL too big for N")
        
    EPS_B, GAM, EPS_PAR, EPS_PERP, ETA = calc_elastic_constants(DEL)

#     EPS_B = EPS_B/DEL
#     EPS_PAR = EPS_PAR/(DEL * l_p**2)
#     EPS_PERP = EPS_PERP/(DEL*l_p**2)
#     GAM = GAM*l_p*DEL
#     ETA = ETA/l_p

    axes_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin_1 = np.array([0, 0, 0])

#     for snap in range(num_snapshots):
    r_output = np.zeros([n_p*n_b, 3])
    u_output = np.zeros([n_p*n_b, 3])
    r1 = np.array([0, 0, 0])
    r_output[::n_b] = r1
    u_output[::n_b] = r1

    phi = 2*np.pi*np.random.rand(n_p)
    theta = np.arccos(stats.uniform(-1, 2).rvs(n_p))
    u2 = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T
    r2 = u2 * DEL#* l_0
    r_output[1::n_b] = r2
    u_output[1::n_b] = u2
    #for poly in range(n_p):
    for bead in range(n_b-2):
        phi = 2*np.pi*np.random.rand(n_p)

        # pull r parallel
        r_par = np.random.normal(GAM, (EPS_PAR/DEL)**-0.5, n_p)

        # pull r perp
        A = (EPS_B*ETA**2)/DEL
        B = (EPS_B*ETA)/DEL
        V0 = (EPS_B)/DEL
        #A_eff = A - (1*(B**2)*(1/np.tanh(V0)))/V0 + 1*(B**2)/(V0**2) 
        A_eff = ((EPS_B/DEL)-(1*EPS_B/DEL)*(1/np.tanh(EPS_B/DEL))+1)*ETA**2
        r_perp = np.random.normal(0, (A_eff + EPS_PERP/DEL)**-0.5, n_p)

        # combine to find next r
        r_prime = DEL*np.array([r_perp*np.cos(phi), r_perp*np.sin(phi), r_par]).T  ######## SHOULDNT MULTPLY L0????? #############

        # pull u_i+1
        #v = np.array([np.zeros(n_p), B*r_perp, -V0*np.ones(n_p)])# + -V0*np.array([0, 0, 1])
        v_mag = np.sqrt(B**2 * r_perp**2 + V0**2)
        r = np.random.rand(n_p)
        rho = (1 / v_mag) * np.log(np.exp(-v_mag)+r*(np.exp(v_mag)-np.exp(-v_mag)))
        theta_1 = np.arccos(rho) #angle between ui+1 and v
        #theta_2 = np.arccos(np.dot(v, np.array([0,0,1]))/v_mag) #angle between v and ui
        theta_2 = np.arccos((EPS_B/DEL)/v_mag)
        theta = theta_1+theta_2
        u_prime = (np.array([np.sin(theta)*np.cos(phi), 0+np.sin(theta)*np.sin(phi), np.cos(theta)])).T

        # convert all from "ui frame of ref" to global frame of ref
        z_prime = u_output[bead+1::n_b]
        #z_prime = z_prime/np.linalg.norm(z_prime, axis = -1)[:, np.newaxis] to normalize- SHOULDNT NEED
        x_prime = np.random.randn(n_p, 3)
        x_prime -= np.sum(x_prime*z_prime, axis=1)[:, None] * z_prime #np.sum is row-wise dot product
        x_prime = x_prime/np.linalg.norm(x_prime, axis = -1)[:, np.newaxis]

        y_prime = np.cross(z_prime, x_prime)

        origin_2 = r_output[bead+1::n_b]
        axes_2 = np.stack((x_prime, y_prime, z_prime), axis=-1) #axis = -1 transposes
        r_bead = origin_2 + np.einsum('ipq,iq->ip',axes_2,r_prime) #element-wise dot product
        u_bead = np.einsum('ipq,iq->ip',axes_2,u_prime)
        r_output[bead+2::n_b] = r_bead
        u_output[bead+2::n_b] = u_bead

    ## gen bead identities
    bead_identity = np.zeros(n_p*n_b)
    for i in range(n_p):
        bead_identity[i*n_b:int(i*n_b + n_b*f_a)] = np.ones(int(n_b*f_a))
    ##
    bead_identity = np.array([[i] for i in bead_identity])

    r_output = np.append(r_output, bead_identity, axis=1)
    #all_snaps_vect_copoly[snap] = r_output
    return r_output

def spin(n_p, n_b, all_snaps_vect_copoly, N, FA):
    CHI = 0
    K0 = 1/np.sqrt(r2(N))
    
    KS = optimize.fmin(lambda K: np.real(gam2(n_p, n_b, all_snaps_vect_copoly, N, np.array([K, 0, 0]), CHI, FA)), K0, disp=False)
    
    return KS


def get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, k_vec, FA, N):
    
    if (np.linalg.norm(k_vec) < 1e-5):
        s2 = np.zeros((2,2),dtype='complex')

        FB = 1.0-FA
        s2[0][0] = FA*FA
        s2[1][1] = FB*FB
        s2[0][1] = FA*FB
        s2[1][0] = FB*FA

        return s2*N**2
    
    num_snapshots = int(len(all_snaps_vect_copoly)/(n_p * n_b))
    i_snap_f = num_snapshots-1
    i_snap_0 = 0
    n_b_calc = n_b
    
    r_i = all_snaps_vect_copoly[:, 0:3]/2
    sigma_i = all_snaps_vect_copoly[:, 3] #(A)
    sigma_j = 1-sigma_i          #(B)
    
    s_mat1 = sigma_i*(np.exp(1j * ((k_vec[0] * r_i[:, 0]) + (k_vec[1] * r_i[:, 1]) + (k_vec[2] * r_i[:,2])))) 
    s_mat2 = sigma_j*(np.exp(1j * ((k_vec[0] * r_i[:, 0]) + (k_vec[1] * r_i[:, 1]) + (k_vec[2] * r_i[:,2]))))   

    s_mat1_neg = sigma_i*(np.exp(1j * ((-k_vec[0] * r_i[:, 0]) + (-k_vec[1] * r_i[:, 1]) + (-k_vec[2] * r_i[:, 2]))))   
    s_mat2_neg = sigma_j*(np.exp(1j * ((-k_vec[0] * r_i[:, 0]) + (-k_vec[1] * r_i[:, 1]) + (-k_vec[2] * r_i[:, 2]))))
   
    polys1 = np.array(np.split(s_mat1, n_p*num_snapshots))
    polys1_neg = np.array(np.split(s_mat1_neg, n_p*num_snapshots))
    polys2 = np.array(np.split(s_mat2, n_p*num_snapshots))
    polys2_neg = np.array(np.split(s_mat2_neg, n_p*num_snapshots))
    
    polys1sum = np.sum(polys1, axis = -1)
    polys1_negsum = np.sum(polys1_neg, axis = -1)
    polys2sum = np.sum(polys2, axis = -1)
    polys2_negsum = np.sum(polys2_neg, axis = -1)
    
    sums_AA = polys1sum * polys1_negsum
    sums_AA = sums_AA / (n_b_calc ** 2 * (num_snapshots) * n_p)
    s2_sim_AA_newer = np.sum(sums_AA)
    
    sums_AB = polys1sum * polys2_negsum
    sums_AB = sums_AB / (n_b_calc ** 2 * (num_snapshots) * n_p)
    s2_sim_AB_newer = np.sum(sums_AB)
    
    sums_BB = polys2sum * polys2_negsum
    sums_BB = sums_BB / (n_b_calc ** 2 * (num_snapshots) * n_p)
    s2_sim_BB_newer = np.sum(sums_BB)
    
    s2_sim_test_newer = np.array([[s2_sim_AA_newer, s2_sim_AB_newer], [s2_sim_AB_newer, s2_sim_BB_newer]])
    
    
    return s2_sim_test_newer * (N**2)

   

def invert_sf2(s2_matrix, K, N):
    if (np.linalg.norm(K) < 1e-5):
        s2 = np.ones((2,2),dtype='complex')
        return s2/(N**2)
    s2inv_cga = np.zeros((2,2),dtype=object)

    [s2aa, s2ab], [s2ba, s2bb] = s2_matrix
    det = s2aa*s2bb - s2ab*s2ba

    s2inv_cga[0,0] = s2bb/det
    s2inv_cga[0,1] = -s2ab/det
    s2inv_cga[1,0] = -s2ba/det
    s2inv_cga[1,1] = s2aa/det
    return s2inv_cga#/(N**2)


def get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, k_vecs, FA, N):
    num_snapshots = int(len(all_snaps_vect_copoly)/(n_p * n_b))
    k1_vec, k2_vec, k3_vec = k_vecs
    
    if np.linalg.norm(k1_vec+k2_vec+k3_vec) >= 1e-10:
        raise ValueError('Qs must add up to zero')
    if np.linalg.norm(k3_vec) < 1e-5:
        s3 = np.zeros((2,2,2),dtype=type(1+1j))

        FB = 1.0-FA
        s2 = get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, k1_vec, FA, N)
        s3[0][0][0] = s2[0][0]*FA*N
        s3[0][0][1] = s2[0][0]*FB*N
        s3[0][1][0] = s2[0][1]*FA*N
        s3[0][1][1] = s2[0][1]*FB*N
        s3[1][0][0] = s2[1][0]*FA*N
        s3[1][0][1] = s2[1][0]*FB*N
        s3[1][1][0] = s2[1][1]*FA*N
        s3[1][1][1] = s2[1][1]*FB*N
        return s3
    
    i_snap_f = num_snapshots-1
    i_snap_0 = 0
    n_b_calc = n_b
    
    s3_matrix = np.zeros((2, 2, 2), dtype = object)
    s3_sim_AAA = np.zeros(1,  dtype = type(1 + 1j))
    s3_sim_AAB = np.zeros(1,  dtype = type(1 + 1j))
    s3_sim_ABB = np.zeros(1,  dtype = type(1 + 1j))
    s3_sim_BBB = np.zeros(1,  dtype = type(1 + 1j))
    
    r_i = all_snaps_vect_copoly[:, 0:3]/2
    sigma_i = all_snaps_vect_copoly[:, 3] #(A)
    sigma_j = 1-sigma_i          #(B)

    
#     s_mat1i = sigma_i*(np.exp(1j * (np.outer(k1_vec[0], r_i[:, 0]) + np.outer(k1_vec[1], r_i[:, 1]) + np.outer(k1_vec[2], r_i[:, 2]))))   
#     s_mat2i = sigma_i*(np.exp(1j * (np.outer(k2_vec[0], r_i[:, 0]) + np.outer(k2_vec[1], r_i[:, 1]) + np.outer(k2_vec[2], r_i[:, 2]))))
#     s_mat3i = sigma_i*(np.exp(1j * (np.outer(k3_vec[0], r_i[:, 0]) + np.outer(k3_vec[1], r_i[:, 1]) + np.outer(k3_vec[2], r_i[:, 2]))))

#     s_mat1j = sigma_j*(np.exp(1j * (np.outer(k1_vec[0], r_i[:, 0]) + np.outer(k1_vec[1], r_i[:, 1]) + np.outer(k1_vec[2], r_i[:, 2]))))   
#     s_mat2j = sigma_j*(np.exp(1j * (np.outer(k2_vec[0], r_i[:, 0]) + np.outer(k2_vec[1], r_i[:, 1]) + np.outer(k2_vec[2], r_i[:, 2]))))
#     s_mat3j = sigma_j*(np.exp(1j * (np.outer(k3_vec[0], r_i[:, 0]) + np.outer(k3_vec[1], r_i[:, 1]) + np.outer(k3_vec[2], r_i[:, 2]))))
    
#     polys1i = np.array(np.split(s_mat1i.T, n_p*num_snapshots))
#     polys1j = np.array(np.split(s_mat1j.T, n_p*num_snapshots))
#     polys2i = np.array(np.split(s_mat2i.T, n_p*num_snapshots))
#     polys2j = np.array(np.split(s_mat2j.T, n_p*num_snapshots))
#     polys3i = np.array(np.split(s_mat3i.T, n_p*num_snapshots))
#     polys3j = np.array(np.split(s_mat3j.T, n_p*num_snapshots))
    
    s_mat1i = sigma_i*(np.exp(1j * ((k1_vec[0] * r_i[:, 0]) + (k1_vec[1] * r_i[:, 1]) + (k1_vec[2] * r_i[:, 2]))))   
    s_mat2i = sigma_i*(np.exp(1j * ((k2_vec[0] * r_i[:, 0]) + (k2_vec[1] * r_i[:, 1]) + (k2_vec[2] * r_i[:, 2]))))
    s_mat3i = sigma_i*(np.exp(1j * ((k3_vec[0] * r_i[:, 0]) + (k3_vec[1] * r_i[:, 1]) + (k3_vec[2] * r_i[:, 2]))))

    s_mat1j = sigma_j*(np.exp(1j * ((k1_vec[0] * r_i[:, 0]) + (k1_vec[1] * r_i[:, 1]) + (k1_vec[2] * r_i[:, 2]))))   
    s_mat2j = sigma_j*(np.exp(1j * ((k2_vec[0] * r_i[:, 0]) + (k2_vec[1] * r_i[:, 1]) + (k2_vec[2] * r_i[:, 2]))))
    s_mat3j = sigma_j*(np.exp(1j * ((k3_vec[0] * r_i[:, 0]) + (k3_vec[1] * r_i[:, 1]) + (k3_vec[2] * r_i[:, 2]))))
    
    polys1i = np.array(np.split(s_mat1i, n_p*num_snapshots))
    polys1j = np.array(np.split(s_mat1j, n_p*num_snapshots))
    polys2i = np.array(np.split(s_mat2i, n_p*num_snapshots))
    polys2j = np.array(np.split(s_mat2j, n_p*num_snapshots))
    polys3i = np.array(np.split(s_mat3i, n_p*num_snapshots))
    polys3j = np.array(np.split(s_mat3j, n_p*num_snapshots))
    
#     sums_AAA = np.sum(polys1i, axis = 1) * np.sum(polys2i, axis = 1) * np.sum(polys3i, axis = 1)
#     sums_AAA = sums_AAA / (n_b_calc ** 3 * (num_snapshots) * n_p)
#     s3_sim_AAA = np.sum(sums_AAA)
    
#     sums_AAB = np.sum(polys1i, axis = 1) * np.sum(polys2i, axis = 1) * np.sum(polys3j, axis = 1)
#     sums_AAB = sums_AAB / (n_b_calc ** 3 * (num_snapshots) * n_p)
#     s3_sim_AAB = np.sum(sums_AAB)
    
#     sums_ABB = np.sum(polys1i, axis = 1) * np.sum(polys2j, axis = 1) * np.sum(polys3j, axis = 1)
#     sums_ABB = sums_ABB / (n_b_calc ** 3 * (num_snapshots) * n_p)
#     s3_sim_ABB = np.sum(sums_ABB)
    
#     sums_BBB = np.sum(polys1j, axis = 1) * np.sum(polys2j, axis = 1) * np.sum(polys3j, axis = 1)
#     sums_BBB = sums_BBB / (n_b_calc ** 3 * (num_snapshots) * n_p)
#     s3_sim_BBB = np.sum(sums_BBB)
    
    polys1_alph = [polys1i, polys1j]
    polys2_alph = [polys2i, polys2j]
    polys3_alph = [polys3i, polys3j]
    
    for a1, a2, a3 in product([0,1], repeat=3):
        if [a1, a2, a3] in [[1,0,0], [0,1,1]]:
            continue
        polys1_cur = polys1_alph[a1]
        polys2_cur = polys2_alph[a2]
        polys3_cur = polys3_alph[a3]
        s3_matrix[a1][a2][a3] = np.sum(((np.sum(polys1_cur, axis = -1) * np.sum(polys2_cur, axis = -1) * np.sum(polys3_cur, axis = -1)) / (n_b_calc ** 3 * (num_snapshots) * n_p)))
        # use Leibler's identities
    s3_matrix[1][0][0] = s3_matrix[0][1][0]
    s3_matrix[0][1][1] = s3_matrix[1][0][1]
    
#     for i_snap in range(num_snapshots):
#         r_snap = all_snaps_vect_copoly[i_snap]
#         for i_p in range(n_p):
#             start_col = i_snap*n_p*n_b+n_b*i_p
            
#             s3_sim_AAA += np.sum(s_mat1i[:, start_col:start_col+n_b], axis = 1) * np.sum(s_mat2i[:, start_col:start_col+n_b], axis = 1) * np.sum(s_mat3i[:, start_col:start_col+n_b], axis = 1) / (n_b_calc ** 3 * (i_snap_f - i_snap_0 + 1) * n_p)
#             s3_sim_AAB += np.sum(s_mat1i[:, start_col:start_col+n_b], axis = 1) * np.sum(s_mat2i[:, start_col:start_col+n_b], axis = 1) * np.sum(s_mat3j[:, start_col:start_col+n_b], axis = 1) / (n_b_calc ** 3 * (i_snap_f - i_snap_0 + 1) * n_p)
#             s3_sim_ABB += np.sum(s_mat1i[:, start_col:start_col+n_b], axis = 1) * np.sum(s_mat2j[:, start_col:start_col+n_b], axis = 1) * np.sum(s_mat3j[:, start_col:start_col+n_b], axis = 1) / (n_b_calc ** 3 * (i_snap_f - i_snap_0 + 1) * n_p)
#             s3_sim_BBB += np.sum(s_mat1j[:, start_col:start_col+n_b], axis = 1) * np.sum(s_mat2j[:, start_col:start_col+n_b], axis = 1) * np.sum(s_mat3j[:, start_col:start_col+n_b], axis = 1) / (n_b_calc ** 3 * (i_snap_f - i_snap_0 + 1) * n_p)

#     s3_matrix[0][0][0] = s3_sim_AAA
#     s3_matrix[0][0][1] = s3_matrix[0][1][0] = s3_matrix[1][0][0] = s3_sim_AAB
#     s3_matrix[0][1][1] = s3_matrix[1][0][1] = s3_matrix[1][1][0] = s3_sim_ABB
#     s3_matrix[1][1][1] = s3_sim_BBB
    return s3_matrix * (N ** 3)

def get_sf4_vect(n_p, n_b, all_snaps_vect_copoly, k_vecs, N):
    num_snapshots = int(len(all_snaps_vect_copoly)/(n_p * n_b))
    k1_vec, k2_vec, k3_vec, k4_vec = k_vecs
    i_snap_f = num_snapshots-1
    i_snap_0 = 0
    n_b_calc = n_b
    s4_matrix = np.zeros((2, 2, 2, 2), dtype = object)
    
#     s4_sim_AAAA = 0#np.zeros(1,  dtype = type(1 + 1j))
#     s4_sim_AAAB = 0#np.zeros(1,  dtype = type(1 + 1j))
#     s4_sim_AABB = 0#np.zeros(1,  dtype = type(1 + 1j))
#     s4_sim_ABBB = 0#np.zeros(1,  dtype = type(1 + 1j))
#     s4_sim_BBBB = 0#np.zeros(1,  dtype = type(1 + 1j))

    r_i = all_snaps_vect_copoly[:, 0:3]/2
    sigma_i = all_snaps_vect_copoly[:, 3] #(A)
    sigma_j = 1-sigma_i          #(B)

    s_mat1i = sigma_i*(np.exp(1j * ((k1_vec[0] * r_i[:, 0]) + (k1_vec[1] * r_i[:, 1]) + (k1_vec[2] * r_i[:, 2]))))   
    s_mat2i = sigma_i*(np.exp(1j * ((k2_vec[0] * r_i[:, 0]) + (k2_vec[1] * r_i[:, 1]) + (k2_vec[2] * r_i[:, 2]))))
    s_mat3i = sigma_i*(np.exp(1j * ((k3_vec[0] * r_i[:, 0]) + (k3_vec[1] * r_i[:, 1]) + (k3_vec[2] * r_i[:, 2]))))
    s_mat4i = sigma_i*(np.exp(1j * ((k4_vec[0] * r_i[:, 0]) + (k4_vec[1] * r_i[:, 1]) + (k4_vec[2] * r_i[:, 2]))))

    s_mat1j = sigma_j*(np.exp(1j * ((k1_vec[0] * r_i[:, 0]) + (k1_vec[1] * r_i[:, 1]) + (k1_vec[2] * r_i[:, 2]))))   
    s_mat2j = sigma_j*(np.exp(1j * ((k2_vec[0] * r_i[:, 0]) + (k2_vec[1] * r_i[:, 1]) + (k2_vec[2] * r_i[:, 2]))))
    s_mat3j = sigma_j*(np.exp(1j * ((k3_vec[0] * r_i[:, 0]) + (k3_vec[1] * r_i[:, 1]) + (k3_vec[2] * r_i[:, 2]))))
    s_mat4j = sigma_j*(np.exp(1j * ((k4_vec[0] * r_i[:, 0]) + (k4_vec[1] * r_i[:, 1]) + (k4_vec[2] * r_i[:, 2]))))
    
    polys1i = np.array(np.split(s_mat1i, n_p*num_snapshots))
    polys1j = np.array(np.split(s_mat1j, n_p*num_snapshots))
    polys2i = np.array(np.split(s_mat2i, n_p*num_snapshots))
    polys2j = np.array(np.split(s_mat2j, n_p*num_snapshots))
    polys3i = np.array(np.split(s_mat3i, n_p*num_snapshots))
    polys3j = np.array(np.split(s_mat3j, n_p*num_snapshots))
    polys4i = np.array(np.split(s_mat4i, n_p*num_snapshots))
    polys4j = np.array(np.split(s_mat4j, n_p*num_snapshots))
    
    ###########################################
    polys1_alph = [polys1i, polys1j]
    polys2_alph = [polys2i, polys2j]
    polys3_alph = [polys3i, polys3j]
    polys4_alph = [polys4i, polys4j]

    for a1, a2, a3, a4 in product([0,1], repeat=4):
        if [a1,a2,a3,a4] in [[0,0,1,0], [0,1,0,0], [1,0,0,0],
                             [0,0,1,1], [0,1,1,0], [0,1,0,1]]:
            continue
        polys1_cur = polys1_alph[a1]
        polys2_cur = polys2_alph[a2]
        polys3_cur = polys3_alph[a3]
        polys4_cur = polys4_alph[a4]
        s4_matrix[a1][a2][a3][a4] = np.sum(((np.sum(polys1_cur, axis = -1) * np.sum(polys2_cur, axis = -1) * np.sum(polys3_cur, axis = -1) * np.sum(polys4_cur, axis = -1)) / (n_b_calc ** 4 * (num_snapshots) * n_p)))

    # use Leibler's identities
    s4_matrix[0][0][1][0], s4_matrix[0][1][0][0], s4_matrix[1][0][0][0] = [s4_matrix[0][0][0][1]]*3
    s4_matrix[0][0][1][1] = s4_matrix[1][1][0][0]
    s4_matrix[0][1][1][0] = s4_matrix[1][0][0][1]
    s4_matrix[0][1][0][1] = s4_matrix[1][0][1][0]
    ######################################################
    
#     sums_AAAA = np.sum(polys1i, axis = 1) * np.sum(polys2i, axis = 1) * np.sum(polys3i, axis = 1) * np.sum(polys4i, axis = 1)
#     sums_AAAA = sums_AAAA / (n_b_calc ** 4 * (num_snapshots) * n_p)
#     s4_sim_AAAA = np.sum(sums_AAAA)
    
#     sums_AAAB = np.sum(polys1i, axis = 1) * np.sum(polys2i, axis = 1) * np.sum(polys3i, axis = 1) * np.sum(polys4j, axis = 1)
#     sums_AAAB = sums_AAAB / (n_b_calc ** 4 * (num_snapshots) * n_p)
#     s4_sim_AAAB = np.sum(sums_AAAB)
    
#     sums_AABB = np.sum(polys1i, axis = 1) * np.sum(polys2i, axis = 1) * np.sum(polys3j, axis = 1) * np.sum(polys4j, axis = 1)
#     sums_AABB = sums_AABB / (n_b_calc ** 4 * (num_snapshots) * n_p)
#     s4_sim_AABB = np.sum(sums_AABB)
    
#     sums_ABBB = np.sum(polys1i, axis = 1) * np.sum(polys2j, axis = 1) * np.sum(polys3j, axis = 1) * np.sum(polys4j, axis = 1)
#     sums_ABBB = sums_ABBB / (n_b_calc ** 4 * (num_snapshots) * n_p)
#     s4_sim_ABBB = np.sum(sums_ABBB)
    
#     sums_BBBB = np.sum(polys1j, axis = 1) * np.sum(polys2j, axis = 1) * np.sum(polys3j, axis = 1) * np.sum(polys4j, axis = 1)
#     sums_BBBB = sums_BBBB / (n_b_calc ** 4 * (num_snapshots) * n_p)
#     s4_sim_BBBB = np.sum(sums_BBBB)
    
#     s4_matrix[0][0][0][0] = s4_sim_AAAA
#     s4_matrix[0][0][0][1] = s4_matrix[0][0][1][0] = s4_matrix[0][1][0][0] = s4_matrix[1][0][0][0] = s4_sim_AAAB
#     s4_matrix[0][0][1][1] = s4_matrix[0][1][0][1] = s4_matrix[1][0][0][1] = s4_matrix[1][0][1][0] = s4_matrix[1][1][0][0] = s4_matrix[0][1][1][0] = s4_sim_AABB
#     s4_matrix[0][1][1][1] = s4_matrix[1][1][0][1] = s4_matrix[1][0][1][1] = s4_matrix[1][1][1][0] = s4_sim_ABBB
#     s4_matrix[1][1][1][1] = s4_sim_BBBB
    return s4_matrix * (N ** 4)



def gam2(n_p, n_b, all_snaps_vect_copoly, N, K, CHI, FA):
    s2inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K, FA, N), K, N)
    D = [1,-1]    # sign indicator
    G = 0
    for I0, I1 in product([0,1], repeat=2):
        G += s2inv[I0, I1]*D[I0]*D[I1]
    #print(G) 
    return (-2*CHI + N*G)#*N
    

def gam3_vertex(n_p, n_b, all_snaps_vect_copoly, N, Ks, FA): #FOR TESTING
    K1, K2, K3 = Ks
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise('Qs must add up to zero')
        
    if not (abs(np.linalg.norm(K1)-np.linalg.norm(K2)) < 1e-5 and abs(np.linalg.norm(K2)-np.linalg.norm(K3)) < 1e-5):
        raise('Qs must have same length')
        
    s3 = s3wlc(pset, N, FA, Ks)
    #s3 = get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, Ks, FA, N)
    
    #s2inv = s2inverse(pset, N, FA, np.linalg.norm(K1))
    s2inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1, FA, N), K1, N)
    
    val = 0
    for I0, I1, I2 in product([0,1], repeat=3):
        val -= s3[I0][I1][I2]* (s2inv[I0][0] - s2inv[I0][1])* (s2inv[I1][0] - s2inv[I1][1])* (s2inv[I2][0] - s2inv[I2][1])

    return val*(N**2)

def gam3(n_p, n_b, all_snaps_vect_copoly, N, Ks, FA):
    K1, K2, K3 = Ks
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise('Qs must add up to zero')
        
    if not (abs(np.linalg.norm(K1)-np.linalg.norm(K2)) < 1e-5 and abs(np.linalg.norm(K2)-np.linalg.norm(K3)) < 1e-5):
        raise('Qs must have same length')
    
    s3 = get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, Ks, FA, N)
    
    s2inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1, FA, N), K1, N)
    
    val = 0
    for I0, I1, I2 in product([0,1], repeat=3):
        val -= (s3[I0][I1][I2]* (s2inv[I0][0] - s2inv[I0][1])* (s2inv[I1][0] - s2inv[I1][1])* (s2inv[I2][0] - s2inv[I2][1]))

    return val*(N**2)

def gam4(n_p, n_b, all_snaps_vect_copoly, N, Ks, FA):
    K1, K2, K3, K4 = Ks
    if not (abs(np.linalg.norm(K1)-np.linalg.norm(K2)) < 1e-5
            and abs(np.linalg.norm(K2)-np.linalg.norm(K3)) < 1e-5
            and abs(np.linalg.norm(K3)-np.linalg.norm(K4)) < 1e-5):
        print(K1, K2, K3, K4)
        raise('Qs must have same length')
    
    #K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)
    
    
    
    
    #print("False is goodie bro!")
    s4 = get_sf4_vect(n_p, n_b, all_snaps_vect_copoly, Ks, N)
    #s4 = s4wlc(pset, wigset, N, FA, Ks)
    #print("s4: ", s4)
    s31 = get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K1, K2, -K1-K2]), FA, N)
    #print("s31: ", s31)
    s32 = get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K1, K3, -K1-K3]), FA, N)
    #print("s32: ", s32)
    s33 = get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K1, K4, -K1-K4]), FA, N)
    #print("s33: ", s33)
    
    
    
#     s2inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1, FA, N), K1, N)
#     #print("s2inv ", s2inv == s2inv)
#     s21inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K12,0,0]), FA, N), np.array([K12,0,0]), N)
#     #print("s21inv ", s21inv == s21inv)
#     s22inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K13,0,0]), FA, N), np.array([K13,0,0]), N)
#     #print("s22inv ", s22inv == s22inv)
#     s23inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K14,0,0]), FA, N), np.array([K14,0,0]), N)
#     #print("s23inv ", s23inv == s23inv)

    ####
    ####
    ####
    ####
    
    #newgam4 - replace np.array([K12, 0, 0]) with K1 + K2
    
    s2inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1, FA, N), K1, N)
    #print("s2inv: ", s2inv)
    s21inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1 + K2, FA, N), K1 + K2, N)
    #print("s21inv: ", s21inv)
    s22inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1 + K3, FA, N), K1 + K3, N)
    #print("s22inv: ", s22inv)
    s23inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1 + K4, FA, N), K1 + K4, N)
    #print("s23inv: ", s23inv)

    ####
    ####
    ####
    ####
    
    G4 = np.zeros((2,2,2,2),dtype=type(1+1j))
    for a1, a2, a3, a4 in product([0,1], repeat=4):
        for I0, I1 in product([0,1], repeat=2):
            G4[a1][a2][a3][a4] += \
                     s31[a1][a2][I0]*s31[a3][a4][I1]*s21inv[I0][I1] + \
                     s32[a1][a4][I0]*s32[a2][a3][I1]*s22inv[I0][I1] + \
                     s33[a1][a3][I0]*s33[a2][a4][I1]*s23inv[I0][I1]
    G4 =  G4 - s4
    #G4 is roughly accurate
    #print(G4)
#     print("s2inv: ", s2inv)
    #s2inv = s2inverse(pset, N, FA, K1)
    
    val = 0
    for I0, I1, I2, I3 in product([0,1], repeat=4):
#         print(I0,I1,I2,I3, ":")
#         print((G4[I0][I1][I2][I3]))
#         print(np.real(s2inv[I0][0]) - np.real(s2inv[I0][1]))
#         print(np.real(s2inv[I1][0]) - np.real(s2inv[I1][1]))
#         print(np.real(s2inv[I2][0]) - np.real(s2inv[I2][1]))
#         print(np.real(s2inv[I3][0]) - np.real(s2inv[I3][1]))
#         print("prod--")
#         print(np.real(G4[I0][I1][I2][I3]) *                (np.real(s2inv[I0][0]) - np.real(s2inv[I0][1]))*                (np.real(s2inv[I1][0]) - np.real(s2inv[I1][1]))*                (np.real(s2inv[I2][0]) - np.real(s2inv[I2][1]))*                (np.real(s2inv[I3][0]) - np.real(s2inv[I3][1])))
        
        val += G4[I0][I1][I2][I3] *                ((s2inv[I0][0]) - (s2inv[I0][1]))*                ((s2inv[I1][0]) - (s2inv[I1][1]))*                ((s2inv[I2][0]) - (s2inv[I2][1]))*                ((s2inv[I3][0]) - (s2inv[I3][1]))
        #print(val)
                
    return val*(N**3)







































###################################################
################# OLD FUNCTIONS ###################
###################################################



































def wlc_cga(n_p, n_b, f_a, num_snapshots):
    #num_snapshots = 1000
    #n_p = 15
    #n_b = 200
    n_b_calc = n_b
    l_0 = .01 # length_kuhn = (10 l_k) = (20 l_p) = (200 l_0) ### length_kuhn = (1 l_k) = (2 l_p) = (200 l_0) #
    l_p = 1
    length_kuhn = n_b *l_0 / (l_p*2)
    kappa = l_p/l_0
    all_snaps_vect_copoly = np.zeros(num_snapshots, dtype=object)

    #f_a = 0.5

    axes_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin_1 = np.array([0, 0, 0])

    for snap in range(num_snapshots):
        output = np.zeros([n_p*n_b, 3])
        r1 = np.array([0, 0, 0])
        output[::n_b] = r1

        phi = 2*np.pi*np.random.rand(n_p)
        theta = np.arccos(stats.uniform(-1, 2).rvs(n_p))
        u2 = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T
        r2 = u2 * l_0
        output[1::n_b] = r2

        for bead in range(n_b-2):
            r = np.random.rand(n_p)
            rho = (1/kappa)*np.log(np.exp(-kappa)+r*(np.exp(kappa)-np.exp(-kappa)))
            theta = np.arccos(rho)
            phi = 2*np.pi*np.random.rand(n_p)

            z_prime = output[bead+1::n_b] - output[bead::n_b] #previous bond vector
            z_prime = z_prime/np.linalg.norm(z_prime, axis = -1)[:, np.newaxis] #normalize each row in matrix - could just /l0

            x_prime = np.random.randn(n_p, 3)
            x_prime -= np.sum(x_prime*z_prime, axis=1)[:, None] * z_prime #np.sum is row-wise dot product
            x_prime = x_prime/np.linalg.norm(x_prime, axis = -1)[:, np.newaxis]

            y_prime = np.cross(z_prime, x_prime)

            r_prime = (l_0 * np.array([np.sin(theta)*np.cos(phi), 0+np.sin(theta)*np.sin(phi), np.cos(theta)])).T # u_bead where (phi = 0)
            #even though x and y axes are random, ensuring randomness by including phi rotation.

            #convert r_prime from x'y'z' to xyz 

            origin_2 = output[bead+1::n_b]
            axes_2 = np.stack((x_prime, y_prime, z_prime), axis=-1) #axis = -1 transposes
            r_bead = origin_2 + np.einsum('ipq,iq->ip',axes_2,r_prime) #element-wise dot product
            output[bead+2::n_b] = r_bead

        ## gen bead identities
        bead_identity = np.zeros(n_p*n_b)
        for i in range(n_p):
            bead_identity[i*n_b:int(i*n_b + n_b*f_a)] = np.ones(int(n_b*f_a))
        ##
        bead_identity = np.array([[i] for i in bead_identity])

        output = np.append(output, bead_identity, axis=1)
        all_snaps_vect_copoly[snap] = output
    return all_snaps_vect_copoly


def get_sf2_vect_old(n_p, n_b, all_snaps_vect_copoly, k_vec, FA, N):
    if (np.linalg.norm(k_vec) < 1e-5):
        s2 = np.zeros((2,2),dtype='complex')

        FB = 1.0-FA
        s2[0][0] = FA*FA
        s2[1][1] = FB*FB
        s2[0][1] = FA*FB
        s2[1][0] = FB*FA

        return s2*N**2
    
    num_snapshots = int(len(all_snaps_vect_copoly)/(n_p * n_b))
    i_snap_f = num_snapshots-1
    i_snap_0 = 0
    n_b_calc = n_b
    
    ###
    ###
    ###
    ### vect sf2 calc
    r_i = all_snaps_vect_copoly[:, 0:3]/2
    sigma_i = all_snaps_vect_copoly[:, 3] #(A)
    sigma_j = 1-sigma_i          #(B)
    
    s_mat1 = sigma_i*(np.exp(1j * (np.outer(k_vec[0], r_i[:, 0]) + np.outer(k_vec[1], r_i[:, 1]) + np.outer(k_vec[2], r_i[:,2])))) 
    s_mat2 = sigma_j*(np.exp(1j * (np.outer(k_vec[0], r_i[:, 0]) + np.outer(k_vec[1], r_i[:, 1]) + np.outer(k_vec[2], r_i[:,2]))))   

    s_mat1_neg = sigma_i*(np.exp(1j * (np.outer(-k_vec[0], r_i[:, 0]) + np.outer(-k_vec[1], r_i[:, 1]) + np.outer(-k_vec[2], r_i[:, 2]))))   
    s_mat2_neg = sigma_j*(np.exp(1j * (np.outer(-k_vec[0], r_i[:, 0]) + np.outer(-k_vec[1], r_i[:, 1]) + np.outer(-k_vec[2], r_i[:, 2]))))
    
    polys1 = np.array(np.split(s_mat1.T, n_p*num_snapshots))
    polys1_neg = np.array(np.split(s_mat1_neg.T, n_p*num_snapshots))
    polys2 = np.array(np.split(s_mat2.T, n_p*num_snapshots))
    polys2_neg = np.array(np.split(s_mat2_neg.T, n_p*num_snapshots))
    
    sums_AA = np.sum(polys1, axis = 1) * np.sum(polys1_neg, axis = 1)
    sums_AA = sums_AA / (n_b_calc ** 2 * (num_snapshots) * n_p)
    s2_sim_AA_new = np.sum(sums_AA)
    
    sums_AB = np.sum(polys1, axis = 1) * np.sum(polys2_neg, axis = 1)
    sums_AB = sums_AB / (n_b_calc ** 2 * (num_snapshots) * n_p)
    s2_sim_AB_new = np.sum(sums_AB)
    
    sums_BB = np.sum(polys2, axis = 1) * np.sum(polys2_neg, axis = 1)
    sums_BB = sums_BB / (n_b_calc ** 2 * (num_snapshots) * n_p)
    s2_sim_BB_new = np.sum(sums_BB)
    
    s2_sim_test = np.array([[s2_sim_AA_new, s2_sim_AB_new], [s2_sim_AB_new, s2_sim_BB_new]])
    return s2_sim_test * (N**2)

def get_sf2(n_p, n_b, all_snaps_vect_copoly, k_vec):
    
            
    num_snapshots = len(all_snaps_vect_copoly)
    i_snap_f = num_snapshots-1
    i_snap_0 = 0
    n_b_calc = n_b
    s2_matrix = np.zeros((2, 2), dtype = object)
    s2_sim_AA = np.zeros(1,  dtype = type(1 + 1j))
    s2_sim_AB = np.zeros(1,  dtype = type(1 + 1j))
    s2_sim_BB = np.zeros(1,  dtype = type(1 + 1j))

    for i_snap in range(num_snapshots):
        r_snap = all_snaps_vect_copoly[i_snap]
        for i_p in range(n_p):
            i_0 = n_b * i_p
            i_f = i_0 + n_b_calc
            #u_i = u_snap[i_0:i_f, :]
            r_i = r_snap[i_0:i_f, 0:3]/2
            sigma_i = r_snap[i_0:i_f, 3] #(A)
            sigma_j = 1-sigma_i          #(B)
            
            s_mat1 = sigma_i*(np.exp(1j * (np.outer(k_vec[0], r_i[:, 0]) + np.outer(k_vec[1], r_i[:, 1]) + np.outer(k_vec[2], r_i[:,2])))) 
            s_mat2 = sigma_j*(np.exp(1j * (np.outer(k_vec[0], r_i[:, 0]) + np.outer(k_vec[1], r_i[:, 1]) + np.outer(k_vec[2], r_i[:,2]))))   

            s_mat1_neg = sigma_i*(np.exp(1j * (np.outer(-k_vec[0], r_i[:, 0]) + np.outer(-k_vec[1], r_i[:, 1]) + np.outer(-k_vec[2], r_i[:, 2]))))         
            s_mat2_neg = sigma_j*(np.exp(1j * (np.outer(-k_vec[0], r_i[:, 0]) + np.outer(-k_vec[1], r_i[:, 1]) + np.outer(-k_vec[2], r_i[:, 2]))))        

            s2_sim_AA += np.sum(s_mat1, axis = 1) * np.sum(s_mat1_neg, axis = 1) / (n_b_calc ** 2 * (i_snap_f - i_snap_0 + 1) * n_p)
            s2_sim_AB += np.sum(s_mat1, axis = 1) * np.sum(s_mat2_neg, axis = 1) / (n_b_calc ** 2 * (i_snap_f - i_snap_0 + 1) * n_p)
            s2_sim_BB += np.sum(s_mat2, axis = 1) * np.sum(s_mat2_neg, axis = 1) / (n_b_calc ** 2 * (i_snap_f - i_snap_0 + 1) * n_p)
    return np.array([[s2_sim_AA, s2_sim_AB], [s2_sim_AB, s2_sim_BB]]) * (N ** 2)


def get_sf3(n_p, n_b, all_snaps_vect_copoly, k_vecs):
    num_snapshots = int(len(all_snaps_vect_copoly))
    k1_vec, k2_vec, k3_vec = k_vecs
    
    i_snap_f = num_snapshots-1
    i_snap_0 = 0
    n_b_calc = n_b
    
    s3_matrix = np.zeros((2, 2, 2), dtype = object)
    s3_sim_AAA = np.zeros(1,  dtype = type(1 + 1j))
    s3_sim_AAB = np.zeros(1,  dtype = type(1 + 1j))
    s3_sim_ABB = np.zeros(1,  dtype = type(1 + 1j))
    s3_sim_BBB = np.zeros(1,  dtype = type(1 + 1j))

    for i_snap in range(num_snapshots):
        r_snap = all_snaps_vect_copoly[i_snap]
        for i_p in range(n_p):
            i_0 = n_b * i_p
            i_f = i_0 + n_b_calc
            #u_i = u_snap[i_0:i_f, :]
            r_i = r_snap[i_0:i_f, 0:3]/2
            sigma_i = r_snap[i_0:i_f, 3] #(A)
            sigma_j = 1-sigma_i          #(B)

            s_mat1i = sigma_i*(np.exp(1j * (np.outer(k1_vec[0], r_i[:, 0]) + np.outer(k1_vec[1], r_i[:, 1]) + np.outer(k1_vec[2], r_i[:, 2]))))   
            s_mat2i = sigma_i*(np.exp(1j * (np.outer(k2_vec[0], r_i[:, 0]) + np.outer(k2_vec[1], r_i[:, 1]) + np.outer(k2_vec[2], r_i[:, 2]))))
            s_mat3i = sigma_i*(np.exp(1j * (np.outer(k3_vec[0], r_i[:, 0]) + np.outer(k3_vec[1], r_i[:, 1]) + np.outer(k3_vec[2], r_i[:, 2]))))

            s_mat1j = sigma_j*(np.exp(1j * (np.outer(k1_vec[0], r_i[:, 0]) + np.outer(k1_vec[1], r_i[:, 1]) + np.outer(k1_vec[2], r_i[:, 2]))))   
            s_mat2j = sigma_j*(np.exp(1j * (np.outer(k2_vec[0], r_i[:, 0]) + np.outer(k2_vec[1], r_i[:, 1]) + np.outer(k2_vec[2], r_i[:, 2]))))
            s_mat3j = sigma_j*(np.exp(1j * (np.outer(k3_vec[0], r_i[:, 0]) + np.outer(k3_vec[1], r_i[:, 1]) + np.outer(k3_vec[2], r_i[:, 2]))))

            s3_sim_AAA += np.sum(s_mat1i, axis = 1) * np.sum(s_mat2i, axis = 1) * np.sum(s_mat3i, axis = 1) / (n_b_calc ** 3 * (i_snap_f - i_snap_0 + 1) * n_p)
            s3_sim_AAB += np.sum(s_mat1i, axis = 1) * np.sum(s_mat2i, axis = 1) * np.sum(s_mat3j, axis = 1) / (n_b_calc ** 3 * (i_snap_f - i_snap_0 + 1) * n_p)
            s3_sim_ABB += np.sum(s_mat1i, axis = 1) * np.sum(s_mat2j, axis = 1) * np.sum(s_mat3j, axis = 1) / (n_b_calc ** 3 * (i_snap_f - i_snap_0 + 1) * n_p)
            s3_sim_BBB += np.sum(s_mat1j, axis = 1) * np.sum(s_mat2j, axis = 1) * np.sum(s_mat3j, axis = 1) / (n_b_calc ** 3 * (i_snap_f - i_snap_0 + 1) * n_p)

    s3_matrix[0][0][0] = s3_sim_AAA
    s3_matrix[0][0][1] = s3_matrix[0][1][0] = s3_matrix[1][0][0] = s3_sim_AAB
    s3_matrix[0][1][1] = s3_matrix[1][0][1] = s3_matrix[1][1][0] = s3_sim_ABB
    s3_matrix[1][1][1] = s3_sim_BBB
    return s3_matrix

def get_sf4(n_p, n_b, all_snaps_vect_copoly, k_vecs):
    num_snapshots = int(len(all_snaps_vect_copoly))
    k1_vec, k2_vec, k3_vec, k4_vec = k_vecs
    i_snap_f = num_snapshots-1
    i_snap_0 = 0
    n_b_calc = n_b
    s4_matrix = np.zeros((2, 2, 2, 2), dtype = object)
    
    #cant these all jst be zero
    s4_sim_AAAA = 0#np.zeros(1,  dtype = type(1 + 1j))
    s4_sim_AAAB = 0#np.zeros(1,  dtype = type(1 + 1j))
    s4_sim_AABB = 0#np.zeros(1,  dtype = type(1 + 1j))
    s4_sim_ABBB = 0#np.zeros(1,  dtype = type(1 + 1j))
    s4_sim_BBBB = 0#np.zeros(1,  dtype = type(1 + 1j))

    for i_snap in range(num_snapshots):
        r_snap = all_snaps_vect_copoly[i_snap]
        for i_p in range(n_p):
            i_0 = n_b * i_p
            i_f = i_0 + n_b_calc
            #u_i = u_snap[i_0:i_f, :]
            r_i = r_snap[i_0:i_f, 0:3]/2
            sigma_i = r_snap[i_0:i_f, 3] #(A)
            sigma_j = 1-sigma_i          #(B)

            s_mat1i = sigma_i*(np.exp(1j * (np.outer(k1_vec[0], r_i[:, 0]) + np.outer(k1_vec[1], r_i[:, 1]) + np.outer(k1_vec[2], r_i[:, 2]))))   
            s_mat2i = sigma_i*(np.exp(1j * (np.outer(k2_vec[0], r_i[:, 0]) + np.outer(k2_vec[1], r_i[:, 1]) + np.outer(k2_vec[2], r_i[:, 2]))))
            s_mat3i = sigma_i*(np.exp(1j * (np.outer(k3_vec[0], r_i[:, 0]) + np.outer(k3_vec[1], r_i[:, 1]) + np.outer(k3_vec[2], r_i[:, 2]))))
            s_mat4i = sigma_i*(np.exp(1j * (np.outer(k4_vec[0], r_i[:, 0]) + np.outer(k4_vec[1], r_i[:, 1]) + np.outer(k4_vec[2], r_i[:, 2]))))

            s_mat1j = sigma_j*(np.exp(1j * (np.outer(k1_vec[0], r_i[:, 0]) + np.outer(k1_vec[1], r_i[:, 1]) + np.outer(k1_vec[2], r_i[:, 2]))))   
            s_mat2j = sigma_j*(np.exp(1j * (np.outer(k2_vec[0], r_i[:, 0]) + np.outer(k2_vec[1], r_i[:, 1]) + np.outer(k2_vec[2], r_i[:, 2]))))
            s_mat3j = sigma_j*(np.exp(1j * (np.outer(k3_vec[0], r_i[:, 0]) + np.outer(k3_vec[1], r_i[:, 1]) + np.outer(k3_vec[2], r_i[:, 2]))))
            s_mat4j = sigma_j*(np.exp(1j * (np.outer(k4_vec[0], r_i[:, 0]) + np.outer(k4_vec[1], r_i[:, 1]) + np.outer(k4_vec[2], r_i[:, 2]))))

            s4_sim_AAAA += np.sum(s_mat1i, axis = 1) * np.sum(s_mat2i, axis = 1) * np.sum(s_mat3i, axis = 1) * np.sum(s_mat4i, axis = 1) / (n_b_calc ** 4 * (i_snap_f - i_snap_0 + 1) * n_p)
            s4_sim_AAAB += np.sum(s_mat1i, axis = 1) * np.sum(s_mat2i, axis = 1) * np.sum(s_mat3i, axis = 1) * np.sum(s_mat4j, axis = 1) / (n_b_calc ** 4 * (i_snap_f - i_snap_0 + 1) * n_p)
            s4_sim_AABB += np.sum(s_mat1i, axis = 1) * np.sum(s_mat2i, axis = 1) * np.sum(s_mat3j, axis = 1) * np.sum(s_mat4j, axis = 1) / (n_b_calc ** 4 * (i_snap_f - i_snap_0 + 1) * n_p)
            s4_sim_ABBB += np.sum(s_mat1i, axis = 1) * np.sum(s_mat2j, axis = 1) * np.sum(s_mat3j, axis = 1) * np.sum(s_mat4j, axis = 1) / (n_b_calc ** 4 * (i_snap_f - i_snap_0 + 1) * n_p)
            s4_sim_BBBB += np.sum(s_mat1j, axis = 1) * np.sum(s_mat2j, axis = 1) * np.sum(s_mat3j, axis = 1) * np.sum(s_mat4j, axis = 1) / (n_b_calc ** 4 * (i_snap_f - i_snap_0 + 1) * n_p)

    s4_matrix[0][0][0][0] = s4_sim_AAAA
    s4_matrix[0][0][0][1] = s4_matrix[0][0][1][0] = s4_matrix[0][1][0][0] = s4_matrix[1][0][0][0] = s4_sim_AAAB
    s4_matrix[0][0][1][1] = s4_matrix[0][1][0][1] = s4_matrix[1][0][0][1] = s4_matrix[1][0][1][0] = s4_matrix[1][1][0][0] = s4_matrix[0][1][1][0] = s4_sim_AABB
    s4_matrix[0][1][1][1] = s4_matrix[1][1][0][1] = s4_matrix[1][0][1][1] = s4_matrix[1][1][1][0] = s4_sim_ABBB
    s4_matrix[1][1][1][1] = s4_sim_BBBB
    return s4_matrix

def assumeone_gam4(n_p, n_b, all_snaps_vect_copoly, N, Ks, FA):
    K1, K2, K3, K4 = Ks
    if not (abs(np.linalg.norm(K1)-np.linalg.norm(K2)) < 1e-5
            and abs(np.linalg.norm(K2)-np.linalg.norm(K3)) < 1e-5
            and abs(np.linalg.norm(K3)-np.linalg.norm(K4)) < 1e-5):
        print(K1, K2, K3, K4)
        raise('Qs must have same length')
    
    #K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)
    
    
    
    
    #print("False is goodie bro!")
    s4 = get_sf4_vect(n_p, n_b, all_snaps_vect_copoly, Ks, N)
    #s4 = s4wlc(pset, wigset, N, FA, Ks)
    #print("s4: ", s4)
    s31 = get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K1, K2, -K1-K2]), FA, N)
    #print("s31: ", s31)
    s32 = get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K1, K3, -K1-K3]), FA, N)
    #print("s32: ", s32)
    s33 = get_sf3_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K1, K4, -K1-K4]), FA, N)
    #print("s33: ", s33)
    
    
    
    s2inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1, FA, N), K1, N)
    #print("s2inv ", s2inv == s2inv)
    s21inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K12,0,0]), FA, N), np.array([K12,0,0]), N)
    #print("s21inv ", s21inv == s21inv)
    s22inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K13,0,0]), FA, N), np.array([K13,0,0]), N)
    #print("s22inv ", s22inv == s22inv)
    s23inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, np.array([K14,0,0]), FA, N), np.array([K14,0,0]), N)
    #print("s23inv ", s23inv == s23inv)

#     ####
#     ####
#     ####
#     ####
    
#     #newgam4 - replace np.array([K12, 0, 0]) with K1 + K2
    
#     s2inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1, FA, N), K1, N)
#     #print("s2inv: ", s2inv)
#     s21inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1 + K2, FA, N), K1 + K2, N)
#     #print("s21inv: ", s21inv)
#     s22inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1 + K3, FA, N), K1 + K3, N)
#     #print("s22inv: ", s22inv)
#     s23inv = invert_sf2(get_sf2_vect(n_p, n_b, all_snaps_vect_copoly, K1 + K4, FA, N), K1 + K4, N)
#     #print("s23inv: ", s23inv)

#     ####
#     ####
#     ####
#     ####
    
    G4 = np.zeros((2,2,2,2),dtype=type(1+1j))
    for a1, a2, a3, a4 in product([0,1], repeat=4):
        for I0, I1 in product([0,1], repeat=2):
            G4[a1][a2][a3][a4] += \
                     s31[a1][a2][I0]*s31[a3][a4][I1]*s21inv[I0][I1] + \
                     s32[a1][a4][I0]*s32[a2][a3][I1]*s22inv[I0][I1] + \
                     s33[a1][a3][I0]*s33[a2][a4][I1]*s23inv[I0][I1]
    G4 =  G4 - s4
    #G4 is roughly accurate
    #print(G4)
#     print("s2inv: ", s2inv)
    #s2inv = s2inverse(pset, N, FA, K1)
    
    val = 0
    for I0, I1, I2, I3 in product([0,1], repeat=4):
#         print(I0,I1,I2,I3, ":")
#         print((G4[I0][I1][I2][I3]))
#         print(np.real(s2inv[I0][0]) - np.real(s2inv[I0][1]))
#         print(np.real(s2inv[I1][0]) - np.real(s2inv[I1][1]))
#         print(np.real(s2inv[I2][0]) - np.real(s2inv[I2][1]))
#         print(np.real(s2inv[I3][0]) - np.real(s2inv[I3][1]))
#         print("prod--")
#         print(np.real(G4[I0][I1][I2][I3]) *                (np.real(s2inv[I0][0]) - np.real(s2inv[I0][1]))*                (np.real(s2inv[I1][0]) - np.real(s2inv[I1][1]))*                (np.real(s2inv[I2][0]) - np.real(s2inv[I2][1]))*                (np.real(s2inv[I3][0]) - np.real(s2inv[I3][1])))
        
        val += G4[I0][I1][I2][I3] *                ((s2inv[I0][0]) - (s2inv[I0][1]))*                ((s2inv[I1][0]) - (s2inv[I1][1]))*                ((s2inv[I2][0]) - (s2inv[I2][1]))*                ((s2inv[I3][0]) - (s2inv[I3][1]))
        #print(val)
                
    return val*(N**3)
