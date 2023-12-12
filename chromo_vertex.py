""" 
Contains all functions needed to generate spinodal diagrams for a chromosome solution with two reader proteins

Inlcudes:
Average binding state
Strucuture factor (2D and 1D)
Gamma 2
Spinodal

"""
import numpy as np

def def_chrom(n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu):
    # fraction of nucleosomes with 0,1,2 marks per protein type, calculated form marks1, marks2: 
    [marks_1, marks_2] = poly_marks
    f_om = np.array([(np.array(marks_1)==0).sum(),(np.array(marks_1)==1).sum(),(np.array(marks_1)==2).sum(), \
                        (np.array(marks_2)==0).sum(),(np.array(marks_2)==1).sum(),(np.array(marks_2)==2).sum()])/len(marks_1)
    
#     l_p = 53 # 53 nm
#     bp_p_b = 45 # base pairs per bond
#     nm_p_bp = 0.34 # nanometetrs per base pair
#     b = l_p * 2 #kuhn length

#     N = (len(marks_1)-1) * bp_p_b * nm_p_bp * (1/b)
#     N_m = N/(len(marks_1)-1)
    
    b = 1
    N_m = 1000
    N = N_m * len(marks_1)
    
    return [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b]

    
def find_zero_crossings(matrix):
    zero_crossings = np.where(np.diff(np.signbit(matrix)))
    zero_crossings_vert = np.where(np.diff(np.signbit(matrix), axis=0))
    crossings = np.array(list(zip(zero_crossings[0], zero_crossings[1]))+ list(zip(zero_crossings_vert[0], zero_crossings_vert[1])))
    return crossings

def eval_f_bind(n_b, n_m, e_m, j_int):
    r"""
    eval_f_bind - Evaluate the binding free energy
    
    Parameters
    ----------
    n_b : integer
        Number of reader proteins bound to the nucleosome
    n_m : integer
        Number of marked tails
    e_me3 : float
        Favorable binding free energy to tri-methylated tails
        
    Returns
    -------
    f_bind : float
        Binding free energy
    
    """
    # Calculate the cases for the binding free energy
    f_bind = 0

    if n_b == 0:
        f_bind = 0
    elif n_b == 1:
        if n_m == 0:
            f_bind = -np.log(2)
        elif n_m == 1:
            f_bind = -np.log(1. + np.exp(-e_m))
        elif n_m == 2:
            f_bind = e_m - np.log(2)
    elif n_b == 2:
        if n_m == 0:
            f_bind = j_int
        elif n_m == 1:
            f_bind = e_m + j_int
        elif n_m == 2:
            f_bind = 2 * e_m + j_int
    
    return f_bind

def eval_f_bind_vec(n_b, n_m_arr, e_m, j_int):
    r"""
    eval_f_bind - Evaluate the binding free energy
    
    Parameters
    ----------
    n_b : integer
        array of Number of reader proteins bound to the nucleosome
    n_m_arr : arr of integer
        Number of marked tails
    e_me3 : float
        Favorable binding free energy to tri-methylated tails
        
    Returns
    -------
    f_bind : float
        Binding free energy
    
    """
    # Calculate the cases for the binding free energy
    f_bind_arr = np.zeros(len(n_m_arr))
    for i,n_m in enumerate(n_m_arr):
        f_bind = 0
        if n_b == 0:
            f_bind = 0
        elif n_b == 1:
            if n_m == 0:
                f_bind = -np.log(2)
            elif n_m == 1:
                f_bind = -np.log(1. + np.exp(-e_m))
            elif n_m == 2:
                f_bind = e_m - np.log(2)
        elif n_b == 2:
            if n_m == 0:
                f_bind = j_int
            elif n_m == 1:
                f_bind = e_m + j_int
            elif n_m == 2:
                f_bind = 2 * e_m + j_int
        f_bind_arr[i] = f_bind
    return f_bind_arr

def calc_binding_states(chrom):
    # calculate f_gam and s_bind for each protein type, at each mu given
    # KEY RETURNS: f_gam_soln_arr, list of matrices of self-consistent f_gammas (for protein 1 and 2) at each mu1, mu2
    #              s_bind_soln_arr, list of matrices of s_binds at each mu1, mu2
    #TODO: 
    #  - deal with multiple solutions (currently, just choosing minimum soln)
    #  - deal with interpolating to find more precise zero crossing

    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom
    
    mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    f_gam_arr = np.arange(-0.01,2.02,0.01)

    f_gam_soln_arr = np.zeros((n_bind, len(mu1_array), len(mu2_array)))

    s_bind_soln_arr = np.zeros((n_bind*3, len(mu1_array), len(mu2_array)))

    f_gam_soln_arr_max = np.zeros((n_bind, len(mu1_array), len(mu2_array)))
    f_gam_soln_arr_min = np.zeros((n_bind, len(mu1_array), len(mu2_array)))

    multi_soln_mus = np.zeros((len(mu1_array), len(mu2_array)))
    for k, mu1 in enumerate(mu1_array):
        for l, mu2 in enumerate(mu2_array):
            mu = [mu1, mu2]


            # 1) generate right hand side (RHS) of f_gamma1 and f_gamma2 as 2d matrices ( [f1, f2] )


            phi_bind_arr = np.zeros(len(f_gam_arr))
            phi_bind_arr = phi_c * f_gam_arr

            RHS = np.zeros(len(f_gam_arr))
            RHS_arr = np.zeros((n_bind, len(f_gam_arr),len(f_gam_arr)))

            f1,f2 = np.meshgrid(f_gam_arr,f_gam_arr)
            combined_matrix = np.dstack((f1*phi_c, f2*phi_c))#.tolist()
    #         erg_int = np.einsum('kl,ijk->jil', v_int, combined_matrix)  #WORKS this is the mat mul of each fgamma pair
            erg_int = np.einsum('ijk,kl->jil', combined_matrix, v_int)  # this is the mat mul of each fgamma pair

            erg_ints = np.split(erg_int, n_bind, axis=2)
            for i in range(len(erg_ints)):
                erg_ints[i] = np.squeeze(erg_ints[i])

            for mark in range(n_bind): # for each reader protein/ mark type
                for om in range(3): 
                    f_bind_1 = eval_f_bind(1, om, e_m[mark], v_int[mark, mark])
                    f_bind_2 = eval_f_bind(2, om, e_m[mark], v_int[mark, mark])
                    q = 1 + np.exp(mu[mark] - f_bind_1 - erg_ints[mark]) + np.exp(2 * mu[mark] - f_bind_2 - 2 * erg_ints[mark])

                    RHS = RHS_arr[mark] 

                    RHS += f_om[om+(mark*3)] * (np.exp(mu[mark] - f_bind_1 - erg_ints[mark]) + 
                                                 2 * np.exp(2 * mu[mark] - f_bind_2 - 2 * erg_ints[mark])) / q 
                    RHS_arr[mark] = RHS




            # 2) self-consistently solve for fgamma1 and fgamma2, finding where the difference (fgamma-RHS) crosses zero



            X, Y = np.meshgrid(f_gam_arr, f_gam_arr)

            crs1 = find_zero_crossings(RHS_arr[0]-Y)
            crs2 = find_zero_crossings(RHS_arr[1]-X)



            # 3) find the intersection of the sets of fgamma_n solutions, such that both f_gammas are self consistent



            aset = set([tuple(x) for x in crs1])
            bset = set([tuple(x) for x in crs2])
            inds = np.array([x for x in aset & bset])+1

            f_gam_solns = np.zeros(len(inds), dtype = "object")#f_gam_arr[inds[0]]
            min_soln = [-1]*n_bind

            for i in range(len(inds)):
                soln = f_gam_arr[inds[i]]           
                f_gam_solns[i] = soln
                #choosing only one solution- minimum one
                if min_soln[0] == -1:
                    min_soln = soln
                else:
    #                 if soln[0] <= min_soln[0] and soln[1] <= min_soln[1]:
    #                     min_soln = soln
    #                 if soln[0] + soln[1] < min_soln[0] + min_soln[1]:# and soln[1] <= min_soln[1]:
    #                     min_soln = soln

    #                 # extremes
                    if soln[1] < min_soln[1]:
                        min_soln = soln
                    elif soln[1] == min_soln[1] and soln[0] < min_soln[0]:
                        min_soln = soln

            multi_soln = False
            if len(f_gam_solns) > 1:
                multi_soln_mus[k,l] = 1
    #             multi_soln = True

            f_gam_solns = [min_soln] #overwriting all solutions with just minimum one



            # 4) for each f_gamma solution pair, calculate each individual s_bind (omega = 0-2)



            sbind = np.zeros((len(f_gam_solns), n_bind*3))

            for j, f_gam_soln in enumerate(f_gam_solns):
                phi_bind_arr = phi_c * np.array(f_gam_soln)
    #             erg_int = np.matmul(v_int, phi_bind_arr)
                erg_int = np.matmul(phi_bind_arr, v_int)

                ind = 0
                for mark in range(n_bind): # for each reader protein/ mark type
                    for om in range(3): # for each possible number of marked tails on nucl
                        f_bind_1 = eval_f_bind(1, om, e_m[mark], v_int[mark, mark])
                        f_bind_2 = eval_f_bind(2, om, e_m[mark], v_int[mark, mark])
                        q = 1 + np.exp(mu[mark] - f_bind_1 - erg_int[mark]) + np.exp(2 * mu[mark] - f_bind_2 - 2 * erg_int[mark])

                        sbind[j, om+mark*3] =  (np.exp(mu[mark] - f_bind_1 - erg_int[mark]) + 
                                                     2 * np.exp(2 * mu[mark] - f_bind_2 - 2 * erg_int[mark])) / q 
                        ind+=1    
                    ind0 = mark * 3 
                    f_gam_orig = f_gam_solns[j][mark]
                    f_gam_calc = np.sum(f_om[ind0:(ind0 + 3)] * sbind[j][ind0:(ind0 + 3)])
                    if f_gam_calc + 0.02 < f_gam_orig or f_gam_calc - 0.02 > f_gam_orig:
                        print("FAILED self-consistency")
                        print("mu: ", mu)
                        print(np.sum(f_om[ind0:(ind0 + 3)] * sbind[j][ind0:(ind0 + 3)]))
                        print(f_gam_solns[j][mark])
                        raise Excpetion("failed self-consistency")

            # 4) store results in array

            if multi_soln:
                raise Exception("not implemented")
                for f_gam_soln in f_gam_solns:
                    for mark in range(n_bind):
                        f_gam_soln_arr_max[mark][k][l] = ""
            else:
                for mark in range(n_bind): 
                    f_gam_soln_arr[mark][k][l] = f_gam_solns[0][mark]
                    for om in range(3):
                        s_bind_soln_arr[om+mark*3][k][l] = sbind[0,om+mark*3]
    return f_gam_soln_arr, s_bind_soln_arr

# need to produce rank 2 tensor that, given two protein types, 
# find avg of product of s for index ij
def eval_sisj_bind(chrom, f_bars, mu, gam1_ind, gam2_ind):
    '''
    poly marks (array of arrays) contains an array of marks per nucleosome for each protein type
    gam1_ind (int) is the index of first protein
    gam2_ind (int) is the index of second protein
    '''
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    n_bind = len(mu)
    # evaluate eqn 85 for each index ij
 
    phi_bind = phi_c * np.array(f_bars)

    erg_int = np.matmul(phi_bind, v_int) #sum over gamma 2 of int and phi and f

    coef1 = -erg_int[gam1_ind] + mu[gam1_ind] 
    coef2 = -erg_int[gam2_ind] + mu[gam2_ind]
    
#     coef1 = -erg_int[gam1_ind] + mu[gam2_ind] 
#     coef2 = -erg_int[gam2_ind] + mu[gam1_ind]
    
    #make vector form of f_bind
    f_bind_g1_s1 = eval_f_bind_vec(1, poly_marks[gam1_ind], e_m[gam1_ind], v_int[gam1_ind,gam1_ind])
    f_bind_g1_s2 = eval_f_bind_vec(2, poly_marks[gam1_ind], e_m[gam1_ind], v_int[gam1_ind,gam1_ind])
    f_bind_g2_s1 = eval_f_bind_vec(1, poly_marks[gam2_ind], e_m[gam2_ind], v_int[gam2_ind,gam2_ind])
    f_bind_g2_s2 = eval_f_bind_vec(2, poly_marks[gam2_ind], e_m[gam2_ind], v_int[gam2_ind,gam2_ind])
    
    #combine coef and f_bind to create 4 dif energies needed
#     - each should be rank 2 tensors
    
    exp_g1_s1 = np.exp(coef1 - f_bind_g1_s1)  #energy at each nucleosome if one gamma 1 protein bound
    exp_g1_s2 = np.exp(2*coef1 - f_bind_g1_s2)
    exp_g2_s1 = np.exp(coef2 - f_bind_g2_s1)
    exp_g2_s2 = np.exp(2*coef2 - f_bind_g2_s2)
    
    exp_11 = np.outer(exp_g1_s1, exp_g2_s1) #getting combined probability at each nucleosome pair
    exp_12 = np.outer(exp_g1_s1, exp_g2_s2)
    exp_21 = np.outer(exp_g1_s2, exp_g2_s1)
    exp_22 = np.outer(exp_g1_s2, exp_g2_s2)
    
    #  (0,0)                (1,0)                                      (0,1)
    q = 1 + np.outer(exp_g1_s1, np.ones(len(exp_g1_s1))) + np.outer(exp_g2_s1, np.ones(len(exp_g1_s1))).T\
    + np.outer(exp_g1_s2,np.ones(len(exp_g1_s1))) + np.outer(exp_g2_s2, np.ones(len(exp_g1_s1))).T\
    + (exp_11 + exp_12 + exp_21 + exp_22) 
    #calculate average matrix (eq 85)
    sisj_bind = (exp_11 + 2*exp_12 + 2*exp_21 + 4*exp_22) / q
    return sisj_bind 



def eval_sisj_bind_shlk(chrom, f_bars, mu, gam1_ind, gam2_ind):
    '''
    poly marks (array of arrays) contains an array of marks per nucleosome for each protein type
    gam1_ind (int) is the index of first protein
    gam2_ind (int) is the index of second protein
    
    returns rank 1 tensor to describe sisj of monomers separated by DEL. 
    '''
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    n_bind = len(mu)
    # evaluate eqn 85 for each index ij
 
    phi_bind = phi_c * np.array(f_bars)

    erg_int = np.matmul(phi_bind, v_int) #sum over gamma 2 of int and phi and f
    
    coef1 = -erg_int[gam1_ind] + mu[gam1_ind] 
    coef2 = -erg_int[gam2_ind] + mu[gam2_ind]
    
    
    #make vector form of f_bind
    f_bind_g1_s1 = eval_f_bind_vec(1, poly_marks[gam1_ind], e_m[gam1_ind], v_int[gam1_ind,gam1_ind])
    f_bind_g1_s2 = eval_f_bind_vec(2, poly_marks[gam1_ind], e_m[gam1_ind], v_int[gam1_ind,gam1_ind])
    f_bind_g2_s1 = eval_f_bind_vec(1, poly_marks[gam2_ind], e_m[gam2_ind], v_int[gam2_ind,gam2_ind])
    f_bind_g2_s2 = eval_f_bind_vec(2, poly_marks[gam2_ind], e_m[gam2_ind], v_int[gam2_ind,gam2_ind])
    
    #combine coef and f_bind to create 4 dif energies needed
#     - each should be rank 2 tensors
    
    exp_g1_s1 = np.exp(coef1 - f_bind_g1_s1)  #energy at each nucleosome if one gamma 1 protein bound
    exp_g1_s2 = np.exp(2*coef1 - f_bind_g1_s2)
    exp_g2_s1 = np.exp(coef2 - f_bind_g2_s1)
    exp_g2_s2 = np.exp(2*coef2 - f_bind_g2_s2)

    sisj_bind_numerator = 0
    q = 1 + np.outer(exp_g1_s1, np.ones(len(exp_g1_s1))) + np.outer(exp_g2_s1, np.ones(len(exp_g1_s1))).T\
    + np.outer(exp_g1_s2,np.ones(len(exp_g1_s1))) + np.outer(exp_g2_s2, np.ones(len(exp_g1_s1))).T
    
    exp_11 = np.outer(exp_g1_s1, exp_g2_s1) #getting combined probability at each nucleosome pair
    sisj_bind_numerator += exp_11
    q += exp_11
    del exp_11
    
    exp_12 = np.outer(exp_g1_s1, exp_g2_s2)
    sisj_bind_numerator += 2*exp_12
    q += exp_12
    del exp_12   
    
    exp_21 = np.outer(exp_g1_s2, exp_g2_s1)
    sisj_bind_numerator += 2*exp_21
    q += exp_21
    del exp_21
    
    exp_22 = np.outer(exp_g1_s2, exp_g2_s2)
    sisj_bind_numerator += 4*exp_22
    q += exp_22
    del exp_22
    
    sisj_bind = sisj_bind_numerator / q

    return sisj_bind

def reduce_sisj_bind(sisj_bind):
    # reduce rank 2 tensor to rank 1
    M = np.shape(sisj_bind)[0]
    sisj_tens = np.zeros(M)
    
    ind = np.arange(0,M,1)
    ind_mesh = np.meshgrid(ind, ind)
    dist = np.abs(ind_mesh[0] - ind_mesh[1])
    for d in range(M):
#         print(d)
        sisj_tens[d] = np.sum(sisj_bind[np.where(dist==d)])
    return sisj_tens

def calc_sf2_chromo(chrom, M2s, k_vec = np.logspace(-3, 2, 50)):
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    M2_AA, M2_AB, M2_BA, M2_BB, M1_cgam0, M1_cgam1, M2_cc = M2s
#     print(k_vec[0])
    M = np.shape(M2_AA)[0]
    nk = len(k_vec)
    N = M*N_m
        
    grid = np.indices((M, M))
    j1 = grid[0]
    j2 = grid[1]

    S2_AA_arr = np.zeros(nk)
    S2_AB_arr = np.zeros(nk)
    S2_BA_arr = np.zeros(nk)
    S2_BB_arr = np.zeros(nk)
    
    S2_cgam0_arr = np.zeros(nk)
    S2_cgam1_arr = np.zeros(nk)
    S2_cc_arr = np.zeros(nk)

    for i, k in enumerate(k_vec):
        C = np.zeros((M, M))
        k = np.linalg.norm(k)
        x_m = (1/6) * N_m * b**2 * k**2

        #j1 = j2, s1 > s2
        index = (j1 == j2)
        constant = 1
        debye = (2/(x_m**2)) * (x_m + np.exp(-x_m) - 1) 
        
        C[np.where((index) != 0)] += debye
        
        #j1 > j2, s1 s2 any
        index = (j1 > j2)
        constant = np.exp(-x_m*(j1-j2))
        integral = (1/(x_m**2)) * (np.exp(x_m) + np.exp(-x_m) - 2) #for off-diagonal terms

        C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
        
        #j2 > j1, s1 s2 any
        index = (j2 > j1)
        constant = np.exp(-x_m*(j2-j1))
#         integral is the same

        C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral   
#         print(C/M**2)
        S2_AA_arr[i] = np.sum((1/M**2) * C * M2_AA)
        S2_AB_arr[i] = np.sum((1/M**2) * C * M2_AB)
        S2_BA_arr[i] = np.sum((1/M**2) * C * M2_BA)
        S2_BB_arr[i] = np.sum((1/M**2) * C * M2_BB)
        
        S2_cgam0_arr[i] = np.sum((1/M**2) * C * M1_cgam0)
        S2_cgam1_arr[i] = np.sum((1/M**2) * C * M1_cgam1)
        S2_cc_arr[i] = np.sum((1/M**2) * C * M2_cc)

    return S2_AA_arr*N**2, S2_AB_arr*N**2, S2_BA_arr*N**2, S2_BB_arr*N**2, S2_cgam0_arr*N**2, S2_cgam1_arr*N**2, S2_cc_arr*N**2


def calc_sf2_chromo_shlk(chrom, M2s, k_vec = np.logspace(-3, -1, 30)):
    # calculates sf2 using rank 1 monomer correlation tensor 
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    M2_AA, M2_AB, M2_BA, M2_BB, M1_cgam0, M1_cgam1, M2_cc = M2s

    M = np.shape(M2_AA)[0]
    nk = len(k_vec)
    N = M*N_m
    
    S2_AA_arr = np.zeros(nk)
    S2_AB_arr = np.zeros(nk)
    S2_BA_arr = np.zeros(nk)
    S2_BB_arr = np.zeros(nk)
    
    S2_cgam0_arr = np.zeros(nk)
    S2_cgam1_arr = np.zeros(nk)
    S2_cc_arr = np.zeros(nk)


    for i, k in enumerate(k_vec):
        C = np.zeros(M)
        k = np.linalg.norm(k)
        x_m = (1/6) * N_m * b**2 * k**2

        #j1 = j2, s1 > s2
        index = 0#(j1 == j2)
        constant = 1
        debye = (2/(x_m**2)) * (x_m + np.exp(-x_m) - 1) 

    #     C[np.where((index) != 0)] += debye
        C[0] += debye

        #j1 > j2, s1 s2 any
        index = np.arange(0, M, 1)#(j1 > j2) #index = del!
        constant = np.exp(-x_m*index)
        integral = (1/(x_m**2)) * (np.exp(x_m) + np.exp(-x_m) - 2) #for off-diagonal terms

        C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral

        S2_AA_arr[i] = np.sum((1/M**2) * C * M2_AA)
        S2_AB_arr[i] = np.sum((1/M**2) * C * M2_AB)
        S2_BA_arr[i] = np.sum((1/M**2) * C * M2_BA)
        S2_BB_arr[i] = np.sum((1/M**2) * C * M2_BB)
        
        S2_cgam0_arr[i] = np.sum((1/M**2) * C * M1_cgam0)
        S2_cgam1_arr[i] = np.sum((1/M**2) * C * M1_cgam1)
        S2_cc_arr[i] = np.sum((1/M**2) * C * M2_cc)
        
    return S2_AA_arr*N**2, S2_AB_arr*N**2, S2_BA_arr*N**2, S2_BB_arr*N**2, S2_cgam0_arr*N**2, S2_cgam1_arr*N**2, S2_cc_arr*N**2

def calc_sf_mats(chrom, f_gam_soln_arr, s_bind_soln_arr):
    # returns rank 3 tensor of mu1, mu2 , k, each value is S2 matrix    
    [n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom
    [marks_1, marks_2] = poly_marks
    k_vec = np.logspace(-3, -1, 30)
    
    mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    
    min_eigval_arr = np.zeros((len(mu1_array[:]), len(mu2_array[:]), len(k_vec)))
    sf_mat = np.zeros((len(mu1_array[:]), len(mu2_array[:]), len(k_vec)), dtype = "object")

    for i, mu1 in enumerate(mu1_array[:]):
        for j, mu2 in enumerate(mu2_array[:]):
            
            mu = [mu1, mu2]
            f1 = f_gam_soln_arr[0][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]
            f2 = f_gam_soln_arr[1][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]
            f_bars = [f1, f2]

            s_bnd = np.zeros(6)
            for ib in range(n_bind*3):
                s_bnd[ib] = s_bind_soln_arr[ib][np.where(mu1_array == mu[0]), np.where(mu2_array== mu[1])][0][0]

            #TODO confirm these indices are correct
            indices_0 = np.tile(marks_1, (len(marks_1),1))#.T
            indices_1 = np.tile(marks_2, (len(marks_2),1)).T + 3

            cc = np.ones((len(marks_1), len(marks_1)))
            cc_red = reduce_sisj_bind(cc)

            s_cgam0 = s_bnd[indices_0]
            s_cgam0_red = reduce_sisj_bind(s_cgam0)

            s_cgam1 = s_bnd[indices_1]
            s_cgam1_red = reduce_sisj_bind(s_cgam1)

        
            sisj_AA_shlk = eval_sisj_bind_shlk(chrom, f_bars, mu, 0, 0)
            sisj_AA_red = reduce_sisj_bind(sisj_AA_shlk)

            sisj_AB_shlk = eval_sisj_bind_shlk(chrom, f_bars, mu, 0, 1)
            sisj_AB_red = reduce_sisj_bind(sisj_AB_shlk)

            sisj_BA_shlk = eval_sisj_bind_shlk(chrom, f_bars, mu, 1, 0)
            sisj_BA_red = reduce_sisj_bind(sisj_BA_shlk)

            sisj_BB_shlk = eval_sisj_bind_shlk(chrom, f_bars, mu, 1, 1)
            sisj_BB_red = reduce_sisj_bind(sisj_BB_shlk)
            
            M2s = [sisj_AA_red,sisj_AB_red,sisj_BA_red,sisj_BB_red, s_cgam0_red, s_cgam1_red, cc_red]
            
#             sisj_AA = eval_sisj_bind(chrom, f_bars, mu, 0, 0)
#             sisj_AB = eval_sisj_bind(chrom, f_bars, mu, 0, 1)
#             sisj_BA = eval_sisj_bind(chrom, f_bars, mu, 1, 0)
#             sisj_BB = eval_sisj_bind(chrom, f_bars, mu, 1, 1)
#             M2s = [sisj_AA_shlk,sisj_AB_shlk,sisj_BA_shlk,sisj_BB_shlk, s_cgam0, s_cgam1, cc]
#             M2s = [sisj_AA, sisj_AB, sisj_BA, sisj_BB, s_cgam0, s_cgam1, cc] # 2d cross corr

            for ik, k in enumerate(k_vec):
                g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = phi_c * np.array(calc_sf2_chromo_shlk(chrom, M2s, [k]))

#                 g1g1, g1g2, g2g1, g2g2, cg1, cg2, cc = phi_c * np.array(calc_sf2_chromo(chrom, M2s, [k])) # 2d cross corr
                
                ss = 1-phi_c
                S2_mat = 1/N**2 * np.array([[cc[0], 0, cg1[0], cg2[0]],\
                                [0, ss*N**2, 0, 0], \
                                [cg1[0], 0, g1g1[0], g1g2[0]],\
                                [cg2[0], 0, g2g1[0], g2g2[0]]])
                sf_mat[i][j][ik] = S2_mat
    return sf_mat
        