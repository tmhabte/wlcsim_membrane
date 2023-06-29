import numpy as np
from scipy import optimize
from itertools import product

class Mix:
    def __init__(self, poly_mat, dens, M_arr, N_m, b):
        self.poly_mat = poly_mat
        self.dens = dens
        self.M_arr = M_arr
        self.N_m = N_m
        self.b = b
        self.n_p = len(dens)
        self.M_max = max(M_arr)
        if len(dens) != len(M_arr):
            raise Exception("dens and M_arr not same length")

        N = 0
        for i in range(len(M_arr)):
            m = M_arr[i]
            phi = dens[i]
            N += (m * N_m * phi)
        self.N = N
        self.M_ave = N/N_m
        
        # find poly with largest M
        # pad all other polys with monomer type 2
        # then repackage as poly_mat_padded
        M_max = max(self.M_arr)
        if self.n_p != 1:
            poly_mat_padded = []
            for i, poly in enumerate(poly_mat):
                M_p = M_arr[i]
                if M_p < M_max:
                    new_poly = poly + [2]*(M_max - M_p)
                    poly_mat_padded.append(new_poly)
                else:
                    poly_mat_padded.append(poly)
        else: 
            poly_mat_padded = poly_mat
        self.poly_mat_padded = poly_mat_padded
        
        #need to calculate each monomer matrix explicitly to handle length polydispersity correctly
        self.M2s = [self.calc_monomer_matrix([0,0]), self.calc_monomer_matrix([0,1])\
                    , self.calc_monomer_matrix([1,0]), self.calc_monomer_matrix([1,1])]
        self.M2_inds = ["AA", "AB", "BA", "BB"]
        
        self.M3s = [self.calc_monomer_matrix_3([0,0,0]), self.calc_monomer_matrix_3([0,0,1]),\
                   self.calc_monomer_matrix_3([0,1,0]), self.calc_monomer_matrix_3([1,0,0]), \
                   self.calc_monomer_matrix_3([0,1,1]), self.calc_monomer_matrix_3([1,0,1]), \
                   self.calc_monomer_matrix_3([1,1,0]), self.calc_monomer_matrix_3([1,1,1])]
        self.M3_inds = ["AAA", "AAB", "ABA", "BAA", "ABB", "BAB",  "BBA", "BBB"]
        
        self.M4s =[self.calc_monomer_matrix_4([0,0,0,0]), self.calc_monomer_matrix_4([0,0,0,1]),\
                   self.calc_monomer_matrix_4([0,0,1,0]), self.calc_monomer_matrix_4([0,1,0,0]),\
                   self.calc_monomer_matrix_4([1,0,0,0]), self.calc_monomer_matrix_4([0,0,1,1]),\
                   self.calc_monomer_matrix_4([1,1,0,0]), self.calc_monomer_matrix_4([1,0,0,1]),\
                   self.calc_monomer_matrix_4([0,1,1,0]), self.calc_monomer_matrix_4([1,0,1,0]),\
                   self.calc_monomer_matrix_4([0,1,0,1]), self.calc_monomer_matrix_4([1,1,1,0]),\
                   self.calc_monomer_matrix_4([1,1,0,1]), self.calc_monomer_matrix_4([1,0,1,1]),\
                   self.calc_monomer_matrix_4([0,1,1,1]), self.calc_monomer_matrix_4([1,1,1,1]),]
        self.M4_inds = ["AAAA, AAAB, AABA, ABAA, BAAA, AABB, BBAA, BAAB, ABBA, BABA, ABAB, BBBA, BBAB, BABB, ABBB, BBBB"]
        
        self.q_star = self.spinodal_gaus()#poly_mat, dens, N_m, b, M, M_arr)

    def calc_monomer_matrix(self, alphas):
        # calculates the alpha1 alpha2 monomer identity cross correlation matrix
        # assumes all polymers consist of monomers with the same length N_m

        #polymat - each row is a polymer
        #dens is an array where each entry is rel vol frac of correponding polymer
        #lenght_arr - M of each corresponding polymer in poly_mat
#         poly_mat = np.array(poly_mat)

        epsilon = 0.00001
        if not ((np.sum(self.dens) + epsilon > 1) and (np.sum(self.dens) - epsilon < 1)):
            raise Exception("polymer volumer fractions do not sum to one")
        if len(self.M_arr) == 1: #single poly
            n_p = 1
            M = len(self.poly_mat)

            alph1 = np.array([alphas[0]]*M)#np.zeros(M)
            alph2 = np.array([alphas[1]]*M)#np.zeros(M)
            sig1 = 1*(self.poly_mat == alph1)
            sig2 = 1*(self.poly_mat == alph2)
            M2 = np.outer(sig1, sig2)
            return M2

        if alphas[0] == 0:
            alph1 =np.zeros((self.n_p, self.M_max))
        elif alphas[0] == 1:
            alph1 =np.ones((self.n_p, self.M_max))

        if alphas[1] == 0:
            alph2 =np.zeros((self.n_p, self.M_max))
        elif alphas[1] == 1:
            alph2 =np.ones((self.n_p, self.M_max))                

        #extend dens into n_pxM matrix
        poly_weights = (np.ones((self.n_p, self.M_max)).T * self.dens).T

        #multiply sigams by density of each polymer
        sigma1 = 1*((self.poly_mat_padded == alph1))#.sum(axis = 0) #sigma. could multiply each
        sigma2 = 1*((self.poly_mat_padded == alph2))#.sum(axis = 0)

        #need to do each row outer product with corresponding row, get n_p MxM matrices, then sum the results
        prods = np.einsum('bi,bo->bio', sigma1*poly_weights, sigma2) # performing row wise cross product (each poly contribution)
        M2 = np.sum(prods, axis = 0)#           ^^^^ averaging each contribution
        return M2
    
    def calc_monomer_matrix_3(self, alphas):
        # calculates the alph1 aph2 alph3 monomer identity cross correlation matrix
        # assumes all polymers consist of monomers with the same length N_m

        # polymat - each row is a polymer
        # dens is an array where each entry is rel vol frac of correponding polymer

        if self.n_p == 1: # single poly
            n_p = 1
            M = len(self.poly_mat)
            
            alph1 = np.array([alphas[0]]*M) #np.zeros(M)
            alph2 = np.array([alphas[1]]*M) #np.zeros(M)
            alph3 = np.array([alphas[2]]*M) #np.zeros(M)
            sig1 = 1*(self.poly_mat == alph1)
            sig2 = 1*(self.poly_mat == alph2)
            sig3 = 1*(self.poly_mat == alph3)
            M3_AAA = np.einsum('i,j,k',sig1,sig2,sig3)
            return M3_AAA

        if alphas[0] == 0:
            alph1 =np.zeros((self.n_p, self.M_max))
        elif alphas[0] == 1:
            alph1 =np.ones((self.n_p, self.M_max))

        if alphas[1] == 0:
            alph2 =np.zeros((self.n_p, self.M_max))
        elif alphas[1] == 1:
            alph2 =np.ones((self.n_p, self.M_max))     

        if alphas[2] == 0:
            alph3 =np.zeros((self.n_p, self.M_max))
        elif alphas[2] == 1:
            alph3 =np.ones((self.n_p, self.M_max)) 

        #extend dens into n_pxM matrix
        poly_weights = (np.ones((self.n_p, self.M_max)).T * self.dens).T

        #multiply sigams by density of each polymer
        sigma1 = 1*((self.poly_mat_padded == alph1))#.sum(axis = 0) #sigma. could multiply each
        sigma2 = 1*((self.poly_mat_padded == alph2))#.sum(axis = 0)
        sigma3 = 1*((self.poly_mat_padded == alph3))

        #need to do each row outer product with corresponding row, get n_p MxMxM matrices, then sum the results
        prods = np.einsum('bi,bo,bn->bion', sigma1*poly_weights, sigma2, sigma3) # performing row wise cross product (each poly contribution)
        M3 = np.sum(prods, axis = 0)#           ^^^^ averaging each contribution
        return M3

    def calc_monomer_matrix_4(self, alphas):
        # calculates the 4pnt monomer identity cross correlation matrix
        # assumes all polymers have monomer of same length N_m

        if self.n_p == 1: # single poly
            n_p = 1
            M = len(self.poly_mat)
            alph1 = np.array([alphas[0]]*self.M_max) #np.zeros(M)
            alph2 = np.array([alphas[1]]*self.M_max) #np.zeros(M)
            alph3 = np.array([alphas[2]]*self.M_max) #np.zeros(M)
            alph4 = np.array([alphas[3]]*self.M_max) #np.zeros(M)

            sig1 = 1*(self.poly_mat == alph1)
            sig2 = 1*(self.poly_mat == alph2)
            sig3 = 1*(self.poly_mat == alph3)
            sig4 = 1*(self.poly_mat == alph4)

            M4 = np.einsum('i,j,k,l',sig1,sig2,sig3,sig4)
            return M4

        if alphas[0] == 0:
            alph1 =np.zeros((self.n_p, self.M_max))
        elif alphas[0] == 1:
            alph1 =np.ones((self.n_p, self.M_max))

        if alphas[1] == 0:
            alph2 =np.zeros((self.n_p, self.M_max))
        elif alphas[1] == 1:
            alph2 =np.ones((self.n_p, self.M_max))     

        if alphas[2] == 0:
            alph3 =np.zeros((self.n_p, self.M_max))
        elif alphas[2] == 1:
            alph3 =np.ones((self.n_p, self.M_max)) 

        if alphas[3] == 0:
            alph4 =np.zeros((self.n_p, self.M_max))
        elif alphas[3] == 1:
            alph4 =np.ones((self.n_p, self.M_max)) 

        #extend dens into n_pxM matrix
        poly_weights = (np.ones((self.n_p, self.M_max)).T * self.dens).T

        #multiply sigams by density of each polymer
        sigma1 = 1*((self.poly_mat_padded == alph1))#.sum(axis = 0) #sigma. could multiply each
        sigma2 = 1*((self.poly_mat_padded == alph2))#.sum(axis = 0)
        sigma3 = 1*((self.poly_mat_padded == alph3))
        sigma4 = 1*((self.poly_mat_padded == alph4))

        #need to do each row outer product with corresponding row, get n_p MxMxM matrices, then sum the results
        prods = np.einsum('bi,bo,bn,bm->bionm', sigma1*poly_weights, sigma2, sigma3, sigma4) # performing row wise cross product (each poly contribution)
        M4 = np.sum(prods, axis = 0) #           ^^^^ averaging each contribution
        return M4

    def spinodal_gaus(self):
        chi = 0
        K0 = 1/np.sqrt(self.N*(self.b**2))

        KS = optimize.fmin(lambda K: np.real(self.gamma2_E(K, chi)), K0,\
                          disp=False)

        return KS

    def gamma2_E(self, k, chi):
        s2_inv = self.calc_sf2_inv(k)

        D = [1,-1]    # sign indicator
        G = 0
        for I0, I1 in product([0,1], repeat=2):
            G += s2_inv[I0, I1]*D[I0]*D[I1]

        return -2*chi + self.N*G

    def calc_sf2_inv(self, k_vec):
        if np.linalg.norm(k_vec[0]) < 1e-5:
            s2 = np.ones((2,2),dtype='complex')
            return s2/self.N**2 # s2[0][0]/(N**2), s2[0][1]/(N**2), s2[1][1]/(N**2)
        (S2_AA_arr, S2_AB_arr, S2_BA_arr, S2_BB_arr) = self.calc_sf2(k_vec)
        det = S2_AA_arr * S2_BB_arr - (S2_AB_arr*S2_BA_arr)
        S2_AA_inv = S2_BB_arr * (1/det)
        S2_AB_inv = -S2_AB_arr * (1/det)
        S2_BA_inv = -S2_BA_arr * (1/det)
        S2_BB_inv = S2_AA_arr * (1/det)

        s2inv = np.zeros((2,2))
        s2inv[0][0] = S2_AA_inv[0]
        s2inv[0][1] = S2_AB_inv[0]
        s2inv[1][0] = S2_BA_inv[0]
        s2inv[1][1] = S2_BB_inv[0]
        return s2inv#(S2_AA_inv, S2_AB_inv, S2_BB_inv)

    def calc_sf2(self, k_vec):
        M2_AA, M2_AB, M2_BA, M2_BB = self.M2s
        nk = len(k_vec)

        grid = np.indices((self.M_max, self.M_max))
        j1 = grid[0]
        j2 = grid[1]

        S2_AA_arr = np.zeros(nk)
        S2_AB_arr = np.zeros(nk)
        S2_BA_arr = np.zeros(nk)
        S2_BB_arr = np.zeros(nk)
        for i, k in enumerate(k_vec):
            C = np.zeros((self.M_max, self.M_max))
            k = np.linalg.norm(k)
            x_m = (1/6) * self.N_m * self.b**2 * k**2

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
            S2_AA_arr[i] = np.sum((1/self.M_max**2) * C * M2_AA)
            S2_AB_arr[i] = np.sum((1/self.M_max**2) * C * M2_AB)
            S2_BA_arr[i] = np.sum((1/self.M_max**2) * C * M2_BA)
            S2_BB_arr[i] = np.sum((1/self.M_max**2) * C * M2_BB)
        return S2_AA_arr*self.N**2, S2_AB_arr*self.N**2, S2_BA_arr*self.N**2, S2_BB_arr*self.N**2

def calc_sf2(mix, k_vec = np.logspace(-2, 2, 50)):
    N = mix.N
    M = mix.M_ave
    M_max = mix.M_max
    N_m = mix.N_m
    b = mix.b
    M2_AA, M2_AB, M2_BA, M2_BB = mix.M2s
    nk = len(k_vec)
    
    grid = np.indices((M_max, M_max))
    j1 = grid[0]
    j2 = grid[1]

    S2_AA_arr = np.zeros(nk)
    S2_AB_arr = np.zeros(nk)
    S2_BA_arr = np.zeros(nk)
    S2_BB_arr = np.zeros(nk)
    for i, k in enumerate(k_vec):
        C = np.zeros((M_max, M_max))
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
    return S2_AA_arr*N**2, S2_AB_arr*N**2, S2_BA_arr*N**2, S2_BB_arr*N**2

def calc_sf2_inv(mix, k_vec = np.logspace(-2, 2, 50)):
    N = mix.N
    M = mix.M_max
    N_m = mix.N_m
    b = mix.b
    
    if np.linalg.norm(k_vec[0]) < 1e-5:
        s2 = np.ones((2,2),dtype='complex')
        return s2/N**2 # s2[0][0]/(N**2), s2[0][1]/(N**2), s2[1][1]/(N**2)
    (S2_AA_arr, S2_AB_arr, S2_BA_arr, S2_BB_arr) = calc_sf2(mix, k_vec)
    det = S2_AA_arr * S2_BB_arr - (S2_AB_arr*S2_BA_arr)
    S2_AA_inv = S2_BB_arr * (1/det)
    S2_AB_inv = -S2_AB_arr * (1/det)
    S2_BA_inv = -S2_BA_arr * (1/det)
    S2_BB_inv = S2_AA_arr * (1/det)
    
    s2inv = np.zeros((2,2))
    s2inv[0][0] = S2_AA_inv[0]
    s2inv[0][1] = S2_AB_inv[0]
    s2inv[1][0] = S2_BA_inv[0]
    s2inv[1][1] = S2_BB_inv[0]
    return s2inv#(S2_AA_inv, S2_AB_inv, S2_BB_inv)

# def find_kstar(poly_mat, dens, N_m, b, M, k_vec = np.logspace(-2, 2, 50), M_arr=[]):
#     (S2_AA_arr, S2_AB_arr, S2_BA_arr, S2_BB_arr)= calc_sf2_inv(poly_mat, dens, N_m, b, M, k_vec, M_arr)
#     G2 = 0.5*(S2_AA_inv - S2_AB_inv - S2_BA_inv + S2_BB_inv) # chi = 0

#     # eigvalues,eigvectors = np.linalg.eigh(G2)
#     eigvalues_lst = G2
#     min_eig = np.min(eigvalues_lst[~np.isnan(eigvalues_lst)])

#     k_star = k_vec[np.where(eigvalues_lst==min_eig)]#[0]][0] 
#     return k_star

def calc_sf3(mix, k_vec, k_vec_2, plotting = False):
    # for a gaussian chain of M monomers, each of length N_m
    N = mix.N
    M = mix.M_ave
    M_max = mix.M_max
    N_m = mix.N_m
    b = mix.b
    M3_AAA, M3_AAB, M3_ABA, M3_BAA, M3_ABB, M3_BAB,  M3_BBA, M3_BBB = mix.M3s
    nk = len(k_vec)

    grid = np.indices((M_max,M_max,M_max))
    j1 = grid[0]
    j2 = grid[1]
    j3 = grid[2]
    
    S3_AAA_arr =  np.zeros(nk)
    S3_BAA_arr = np.zeros(nk)
    S3_BBA_arr = np.zeros(nk)
    S3_BBB_arr = np.zeros(nk)
    
    S3_ABA_arr = np.zeros(nk)
    S3_BAB_arr = np.zeros(nk)
    
    S3_AAB_arr = np.zeros(nk)
    S3_ABB_arr = np.zeros(nk) 
    
    for i, k_1 in enumerate(k_vec):
        k_2 = k_vec_2[i]
        k_12 = k_1 + k_2

        # CASE 1; kA = k1 + k2, kB = k_1; S3 > S2 > S1 and S1 > S2 > S3
        case1 = [[k_12, k_1], [j3, j2, j1]]

        # CASE 2; kA = k2, kB = k1 + k2; S2 > S1 > S3 and S3 > S1 > S2
        case2 = [[k_2, k_12], [j2, j1, j3]]
        
        # CASE 3; kA = k2, kB = -k1; S2 > S3 > S1 and S1 > S3 > S2
        case3 = [[-k_2, k_1], [j2, j3, j1]] # SWITCHED negatives from -k_1
        
        case_arr = [case1, case2, case3]#, case1deg, case2deg, case3deg]
        # need to consider degenerate cases. flipping each element in array, then appending to original case array
        case_arr = np.vstack((case_arr, [[np.flipud(el) for el in cse] for cse in case_arr]))
        
#        for each case and sub case, add to a matrix C(j1, j2, j3) which contains the contribution to the overall S3
#        then sum over all indices. Need to keep track of js so that aproiate multiplications with cross corr matrix M3        
        C = np.zeros((M_max,M_max,M_max))

        for cse in case_arr:
            kA, kB = cse[0]
            ordered_js = cse[1]
            
            xm_A = (1/6) * N_m * b**2 * np.linalg.norm(kA)**2
            xm_B = (1/6) * N_m * b**2 * np.linalg.norm(kB)**2
            
            C = calc_case_s3(C, xm_A, xm_B, ordered_js)
    
        S3_AAA_arr[i] += np.sum((1/M**3) * M3_AAA * C)*(N**3)
        S3_BAA_arr[i] += np.sum((1/M**3) * M3_BAA * C)*(N**3)
        S3_BBA_arr[i] += np.sum((1/M**3) * M3_BBA * C)*(N**3)
        S3_BBB_arr[i] += np.sum((1/M**3) * M3_BBB * C)*(N**3)
        
        S3_ABA_arr[i] += np.sum((1/M**3) * M3_ABA * C)*(N**3)
        S3_BAB_arr[i] += np.sum((1/M**3) * M3_BAB * C)*(N**3)
        
        S3_AAB_arr[i] += np.sum((1/M**3) * M3_AAB * C)*(N**3)
        S3_ABB_arr[i] += np.sum((1/M**3) * M3_ABB * C)*(N**3)
        
    s3 = np.zeros((2,2,2)) 
    s3[0][0][0] = S3_AAA_arr[0]
    s3[1][0][0] = S3_BAA_arr[0]
    s3[0][1][0] = S3_ABA_arr[0]
    s3[0][0][1] = S3_AAB_arr[0]
    s3[0][1][1] = S3_ABB_arr[0]
    s3[1][0][1] = S3_BAB_arr[0]
    s3[1][1][0] = S3_BBA_arr[0]
    s3[1][1][1] = S3_BBB_arr[0]
    
    if plotting: # matrix only contains single value, for calculating gamma functions
        return S3_AAA_arr, S3_AAB_arr, S3_ABA_arr, S3_BAA_arr, S3_ABB_arr, S3_BAB_arr,  S3_BBA_arr, S3_BBB_arr
    
    return s3

def calc_case_s3(C, xm_A, xm_B, ordered_js):

    jmax, jmid, jmin = ordered_js
    
    cylindrical = False
    epsilon = 0.0000001
    if xm_A + epsilon > xm_B and xm_A - epsilon < xm_B:
        cylindrical = True
    
    xm_A_eq_0 = False
    if xm_A < 1e-5:
        xm_A_eq_0 = True
        
    xm_B_eq_0 = False
    if xm_B < 1e-5:
        xm_B_eq_0 = True

    #for each sub case, looking at the degenerate case where 1 and 2 are switched
    constant = np.exp(-xm_A*(jmax - jmid)) * np.exp(-xm_B*(jmid - jmin)) 

    # sub case 1; jmax > jmid > jmin, {s1, s2, s3} any 
    index = (jmax > jmid) * (jmid > jmin)
    
    if cylindrical == True:
        integral = (1 / xm_A**2) * 2 * (-1 + np.cosh(xm_A))
    elif xm_B_eq_0:
        integral = (2*(-1+np.cosh(xm_A)))/ (xm_A**2)
    elif xm_A_eq_0:
        integral = (2*(-1+np.cosh(xm_B)))/ (xm_B**2)
    else:
        integral = (-2 / (xm_A * (xm_A - xm_B) * xm_B)) \
        * (-np.sinh(xm_A) + np.sinh(xm_A - xm_B) + np.sinh(xm_B))

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    # sub case 2; jmax = jmid > jmin, s3 > s2, {s1} any
    index = (jmax == jmid) * (jmid > jmin)
    
    if cylindrical == True:
        integral = (1 / xm_A**3) *( (2 + xm_A) * (-1 + np.cosh(xm_A)) - (xm_A * np.sinh(xm_A)) )
    elif xm_B_eq_0:
        integral = (-1 + xm_A + np.cosh(xm_A) - np.sinh(xm_A))/ (xm_A**2)
    elif xm_A_eq_0:
        integral = (np.exp(-xm_B)*(-1 + np.exp(xm_B))*(1+np.exp(xm_B)*(-1 + xm_B))) / (xm_B**3)   
    else:
        integral = ((-1 + np.exp(xm_B))/(xm_A * (xm_A - xm_B)*xm_B**2)) \
        * (xm_A + (-1 + np.exp(-xm_A))*xm_B - xm_A*np.cosh(xm_B) + xm_A*np.sinh(xm_B))

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral 

    # BONUS sub case 4; jmax > jmid = jmin, s2 > s1, {s3} any 
    index = (jmax > jmid) * (jmid == jmin)
    
    if cylindrical == True:
        integral = (1 / xm_A**3) *( (2 + xm_A) * (-1 + np.cosh(xm_A)) - (xm_A * np.sinh(xm_A)) )
    elif xm_B_eq_0:
        integral = ((-2+xm_A)*(-1+np.cosh(xm_A))+ (xm_A*np.sinh(xm_A)))/ (xm_A**3)
    elif xm_A_eq_0:
        integral = (-1+xm_B+np.cosh(xm_B) - np.sinh(xm_B))/ (xm_B**2)
    else:
        integral = (((-1 + np.exp(xm_A))*(np.exp(-xm_A - xm_B)))/(xm_B * (xm_A - xm_B)*xm_A**2)) \
        * (-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B) * (xm_A -xm_B) + np.exp(xm_B)*xm_B)

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral 

    # sub case 3; jmax = jmid = jmin, s3 > s2 > s1
    index = (jmax == jmid) * (jmid == jmin)

    if cylindrical == True:
        integral = (1 / xm_A**3) * (np.exp(-xm_A) * (2 + np.exp(xm_A)*(-2 + xm_A) + xm_A))
    elif xm_B_eq_0:
        integral = (2-2*np.exp(-xm_A) - 2*xm_A + xm_A**2)/ (2*xm_A**3)
    elif xm_A_eq_0:
        integral = (2-2*np.exp(-xm_B) - 2*xm_B + xm_B**2)/ (2*xm_B**3)
    else:
        integral = (1 / (xm_A**2 * xm_B - xm_A * xm_B**2))\
        * ( xm_A + (((-1 + np.exp(-xm_B)) * xm_A)/(xm_B)) - xm_B + ((xm_B - np.exp(-xm_A)*xm_B)/(xm_A)) )

    C[np.where(index != 0)] += 1\
                                    * constant[np.where(index != 0)]\
                                    * integral
    return C

def calc_sf4(mix, k_vec, k_vec_2, k_vec_3, plotting = False):
    N = mix.N
    M = mix.M_ave
    M_max = mix.M_max
    N_m = mix.N_m
    b = mix.b
    M4_AAAA, M4_AAAB, M4_AABA, M4_ABAA,\
    M4_BAAA, M4_AABB, M4_BBAA, M4_BAAB,\
    M4_ABBA, M4_BABA, M4_ABAB, M4_BBBA,\
    M4_BBAB, M4_BABB, M4_ABBB, M4_BBBB = mix.M4s
    nk = len(k_vec)

    grid = np.indices((M_max,M_max,M_max,M_max))
    j1 = grid[0]
    j2 = grid[1]
    j3 = grid[2]
    j4 = grid[3]
    
    S4_AAAA_arr = np.zeros(nk)
    
    S4_BAAA_arr = np.zeros(nk)
    S4_ABAA_arr = np.zeros(nk)
    S4_AABA_arr = np.zeros(nk)
    S4_AAAB_arr = np.zeros(nk)
    
    S4_AABB_arr = np.zeros(nk)
    S4_BBAA_arr = np.zeros(nk)
    S4_ABBA_arr = np.zeros(nk)
    S4_BAAB_arr = np.zeros(nk)
    S4_ABAB_arr = np.zeros(nk)
    S4_BABA_arr = np.zeros(nk)

    S4_ABBB_arr = np.zeros(nk)
    S4_BABB_arr = np.zeros(nk)
    S4_BBAB_arr = np.zeros(nk)
    S4_BBBA_arr = np.zeros(nk)
    
    S4_BBBB_arr = np.zeros(nk)
    
    for i, k1 in enumerate(k_vec):
        k2 = k_vec_2[i]
        k3 = k_vec_3[i]
        k12 = k1 + k2
        k13 = k1 + k3
        k23 = k2 + k3
        k123 = k1 + k2 + k3
        
        # CASE 1; kA = k1 + k2 + k3; kB = k_1 + k_2; kC = k_1  S4 > S3 > S2 > S1 (and reverse). All cases on wlcstat
        case1 = [[k123, k12, k1], [j4, j3, j2, j1]]
        case2 = [[k123, k12, k2], [j4, j3, j1, j2]]
        case3 = [[k123, k13, k1], [j4, j2, j3, j1]]
        case4 = [[k123, k23, k2], [j4, j1, j3, j2]]
        case5 = [[k123, k13, k3], [j4, j2, j1, j3]]
        case6 = [[k123, k23, k3], [j4, j1, j2, j3]]
        case7 = [[-k3, k12, k1], [j3, j4, j2, j1]]
        case8 = [[-k3, k12, k2], [j3, j4, j1, j2]]
        case9 = [[-k2, k13, k1], [j2, j4, j3, j1]]
        case10 = [[-k1, k23, k2], [j1, j4, j3, j2]]
        case11 = [[-k2, k13, k3], [j2, j4, j1, j3]]
        case12 = [[-k1, k23, k3], [j1, j4, j2, j3]]
        
#         case1_deg = [[k1, k12, k123], [j1, j2, j3, j4]]
#         case2_deg = [[k2, k12, k123], [j2, j1, j3, j4]]
#         case3_deg = [[k1, k13, k123], [j1, j3, j2, j4]]
#         case4_deg = [[k2, k23, k123], [j2, j3, j1, j4]]
#         case5_deg = [[k3, k13, k123], [j3, j1, j2, j4]]
#         case6_deg = [[k3, k23, k123], [j3, j2, j1, j4]]
#         case7_deg = [[k1, k12, -k3], [j1, j2, j4, j3]]
#         case8_deg = [[k2, k12, -k3], [j2, j1, j4, j3]]
#         case9_deg = [[k1, k13, -k2], [j1, j3, j4, j2]]
#         case10_deg = [[k2, k23, -k1], [j2, j3, j4, j1]]
#         case11_deg = [[k3, k13, -k2], [j3, j1, j4, j2]]
#         case12_deg = [[k3, k23, -k1], [j3, j2, j4, j1]]



        case_arr = [case1, case2, case3, case4, case5, case6, \
                   case7, case8, case9, case10, case11, case12, ]#\
#                     case1_deg, case2_deg, case3_deg, case4_deg, case5_deg, case6_deg, \
#                    case7_deg, case8_deg, case9_deg, case10_deg, case11_deg, case12_deg]
#         print("FASTER??") nope
        # need to consider degenerate cases. flipping each element in array, then appending to original case array
        case_arr = np.vstack((case_arr, [[np.flipud(el) for el in cse] for cse in case_arr]))
        
#        for each case and sub case, add to a matrix C(j1, j2, j3, j4) which contains the contribution to the overall S4
#        then sum over all indices. Need to keep track of js so that aproiate multiplications with cross corr matrix M4 
        C = np.zeros((M_max,M_max,M_max,M_max))
        for cse in case_arr:
            kA, kB, kC = cse[0]
            ordered_js = cse[1]
            
            xm_A = (1/6) * N_m * b**2 * np.linalg.norm(kA)**2
            xm_B = (1/6) * N_m * b**2 * np.linalg.norm(kB)**2
            xm_C = (1/6) * N_m * b**2 * np.linalg.norm(kC)**2
            
            C = calc_case_s4(C, xm_A, xm_B, xm_C, ordered_js)
            
        S4_AAAA_arr[i] += np.sum((1/M**4) * M4_AAAA * C)*(N**4)
        
        S4_BAAA_arr[i] += np.sum((1/M**4) * M4_BAAA * C)*(N**4)
        S4_ABAA_arr[i] += np.sum((1/M**4) * M4_ABAA * C)*(N**4)
        S4_AABA_arr[i] += np.sum((1/M**4) * M4_AABA * C)*(N**4)
        S4_AAAB_arr[i] += np.sum((1/M**4) * M4_AAAB * C)*(N**4)

        S4_AABB_arr[i] += np.sum((1/M**4) * M4_AABB * C)*(N**4)        
        S4_BBAA_arr[i] += np.sum((1/M**4) * M4_BBAA * C)*(N**4)
        S4_ABBA_arr[i] += np.sum((1/M**4) * M4_ABBA * C)*(N**4)
        S4_BAAB_arr[i] += np.sum((1/M**4) * M4_BAAB * C)*(N**4)
        S4_ABAB_arr[i] += np.sum((1/M**4) * M4_ABAB * C)*(N**4)
        S4_BABA_arr[i] += np.sum((1/M**4) * M4_BABA * C)*(N**4)

        S4_ABBB_arr[i] += np.sum((1/M**4) * M4_ABBB * C)*(N**4)
        S4_BABB_arr[i] += np.sum((1/M**4) * M4_BABB * C)*(N**4)
        S4_BBAB_arr[i] += np.sum((1/M**4) * M4_BBAB * C)*(N**4)
        S4_BBBA_arr[i] += np.sum((1/M**4) * M4_BBBA * C)*(N**4)

        S4_BBBB_arr[i] += np.sum((1/M**4) * M4_BBBB * C)*(N**4)
        
        
    s4 = np.zeros((2, 2, 2, 2))
    
    s4[0][0][0][0] = S4_AAAA_arr[0]
    s4[0][0][0][1] = S4_AAAB_arr[0]
    s4[0][0][1][0] = S4_AABA_arr[0]
    s4[0][1][0][0] = S4_ABAA_arr[0]
    s4[1][0][0][0] = S4_BAAA_arr[0]
    s4[0][0][1][1] = S4_AABB_arr[0]
    s4[1][1][0][0] = S4_BBAA_arr[0]
    s4[1][0][0][1] = S4_BAAB_arr[0]
    s4[0][1][1][0] = S4_ABBA_arr[0]
    s4[1][0][1][0] = S4_BABA_arr[0]
    s4[0][1][0][1] = S4_ABAB_arr[0]
    s4[1][1][1][0] = S4_BBBA_arr[0]
    s4[1][1][0][1] = S4_BBAB_arr[0]
    s4[1][0][1][1] = S4_BABB_arr[0]
    s4[0][1][1][1] = S4_ABBB_arr[0]
    s4[1][1][1][1] = S4_BBBB_arr[0]
    
    if plotting: # matrix only contains single value, for calculating gamma functions
#         raise Exception("need to fix return value")
        return S4_AAAA_arr, S4_AAAB_arr, S4_AABA_arr, S4_ABAA_arr, S4_BAAA_arr, S4_AABB_arr, S4_BBAA_arr, S4_BAAB_arr, S4_ABBA_arr, S4_BABA_arr, S4_ABAB_arr, S4_BBBA_arr, S4_BBAB_arr, S4_BABB_arr, S4_ABBB_arr, S4_BBBB_arr 
    
    return s4 

def calc_case_s4(C, xm_A, xm_B, xm_C, ordered_js):

    jmax, jupp, jlow, jmin = ordered_js
    
    xmA_eq_xmB = False
    xmA_eq_xmC = False
    xmB_eq_xmC = False
    epsilon = 0.0000001
    if xm_A + epsilon > xm_B and xm_A - epsilon < xm_B:
        xmA_eq_xmB = True
    if xm_A + epsilon > xm_C and xm_A - epsilon < xm_C:
        xmA_eq_xmC = True
    if xm_B + epsilon > xm_C and xm_B - epsilon < xm_C:
        xmB_eq_xmC = True
        
    xm_B_eq_0 = False
    if xm_B < 1e-5:
        xm_B_eq_0 = True

    #for each sub case, looking at the degenerate case where 1 and 2 are switched
    constant = np.exp(-xm_A*(jmax - jupp)- xm_B*(jupp - jlow) - xm_C*(jlow - jmin))

    # sub case 1; jmax > jupp > jlow > jmin, {s1234} any
    index = (jmax > jupp) * (jupp > jlow) * (jlow > jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = 2*(-1 + np.cosh(xm_A)) / xm_A**2
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = (16 / (xm_A**2 * xm_C**2) )* \
                    np.sinh(xm_A / 2)**2 *np.sinh(xm_C / 2)**2
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (16 / xm_A**4 )* \
                    np.sinh(xm_A / 2)**4
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = (16*np.sinh(xm_A / 2) * np.sinh((xm_A - xm_B)/2) * np.sinh((xm_B - xm_C)/2) * np.sinh(xm_C/2))\
                    / (xm_A * (xm_A - xm_B) * (xm_B - xm_C) * xm_C)
    elif xmA_eq_xmB:
        integral = -(2 / (xm_A*xm_C*(xm_A - xm_C))) * \
                    (-np.sinh(xm_A) + np.sinh(xm_A - xm_C) + np.sinh(xm_C))
    elif xmB_eq_xmC:
        integral = -(2 / (xm_A*xm_B*(xm_A - xm_B))) * \
                    (-np.sinh(xm_A) + np.sinh(xm_A - xm_B) + np.sinh(xm_B))
    elif xmA_eq_xmC:
        integral = (16 / (xm_A**2 * (xm_A - xm_B)**2) )* \
                    np.sinh(xm_A / 2)**2 *np.sinh((xm_A - xm_B) / 2)**2

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    # sub case 2; jmax = jupp > jlow > jmin
    index = (jmax == jupp) * (jupp > jlow) * (jlow > jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (np.exp(-xm_A) * (-1 + np.exp(xm_A)) * (-1 + np.exp(xm_A) - xm_A))\
        / (xm_A**3)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = ((np.exp(-xm_A - xm_C)) * (-1+np.exp(xm_C))**2 * (1+np.exp(xm_A) *(-1 + xm_A)))\
        /(xm_A**2*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = ((np.exp(-2*xm_A)) * (-1+np.exp(xm_A))**2 * (1+np.exp(xm_A) *(-1 + xm_A)))\
        /(xm_A**2*xm_A**2)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((np.exp(-xm_A-xm_B-xm_C)) * (-1+np.exp(xm_C)) * (-np.exp(xm_B)+np.exp(xm_C))\
                    * (-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B))\
                    / (xm_A*xm_B*xm_C*(xm_A - xm_B) * (xm_C - xm_B))
    elif xmA_eq_xmB:
        integral = ((np.exp(-xm_A - xm_C)) * (-1+np.exp(xm_C)) * (-np.exp(xm_A)+np.exp(xm_C)) * (-1+np.exp(xm_A) - xm_A))\
        /(xm_A**2*xm_C*(xm_C - xm_A))
    elif xmB_eq_xmC:
        integral = ((-1+np.exp(xm_B))*(xm_A - xm_A*np.exp(-xm_B) + (-1+np.exp(-xm_A))*xm_B))/(xm_B**2*xm_A*(xm_A - xm_B))
    elif xmA_eq_xmC:
        integral = ( (np.exp(-2*xm_A-xm_B))*(-1+np.exp(xm_A))*(np.exp(xm_A) - np.exp(xm_B))\
                    *(-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B)) / (xm_B*xm_A**2*(xm_A-xm_B)**2)

    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    


    # sub case 3; jmax > jupp = jlow > jmin
    index = (jmax > jupp) * (jupp == jlow) * (jlow > jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (-1 + np.cosh(xm_A)) / (xm_A**2)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = ((np.exp(-xm_A - xm_C)) * (-1+np.exp(xm_A)) * (-1+np.exp(xm_C)) * (-np.exp(xm_A)*xm_A + np.exp(xm_A+xm_C)*(xm_A-xm_C) + np.exp(xm_C)*xm_C))\
        /(xm_A**2*xm_C**2*(xm_A - xm_C))
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (np.exp(-xm_A) * (-1+np.exp(xm_A))**2 * (-1 + np.exp(xm_A) - xm_A))\
        /(xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((1-np.exp(-xm_A))*(-1+np.exp(xm_C))*( ( (1-np.exp(xm_A-xm_B))/(xm_A-xm_B) ) + ( (-1+np.exp(xm_A-xm_C))/(xm_A-xm_C) ) ))/(xm_A*xm_C*(xm_B - xm_C))
    elif xmA_eq_xmB:
        integral = ((np.exp(-xm_A - xm_C)) * (-1+np.exp(xm_A)) * (-1+np.exp(xm_C)) * (np.exp(xm_A)+np.exp(xm_C)*(-1-xm_A+xm_C)))\
        /(xm_A*xm_C*(xm_A - xm_C)**2)
    elif xmB_eq_xmC:
        integral = ((np.exp(-xm_A - xm_B)) * (-1+np.exp(xm_A)) * (-1+np.exp(xm_B)) * (np.exp(xm_B)+np.exp(xm_A)*(-1+xm_A-xm_B)))\
        /(xm_A*xm_B*(xm_A - xm_B)**2)
    elif xmA_eq_xmC:
        integral = ((np.exp(-xm_A - xm_B)) * (-1+np.exp(xm_A))**2 * (np.exp(xm_A)+np.exp(xm_B)*(-1-xm_A+xm_B)))/(xm_A**2 * (xm_A - xm_B)**2)
  
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    


    # sub case 4; jmax > jupp > jlow = jmin
    index = (jmax > jupp) * (jupp > jlow) * (jlow == jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = ((2+xm_A)*(-1+np.cosh(xm_A)) - xm_A*np.sinh(xm_A))/(xm_A**3)
    
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = (np.exp(-xm_A - xm_C) * (-1+np.exp(xm_A))**2 * (1 + np.exp(xm_C)*(-1 + xm_C)))\
        /(xm_A**2*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (np.exp(-2*xm_A) * (-1+np.exp(xm_A))**2 * (1 + np.exp(xm_A)*(-1 + xm_A)))\
        /(xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((np.exp(-xm_A-xm_B-xm_C)) * (-1+np.exp(xm_A)) * (-np.exp(xm_B)+np.exp(xm_A))\
                    * (-np.exp(xm_B)*xm_B + np.exp(xm_C + xm_B)*(xm_B-xm_C) + np.exp(xm_C)*xm_C))\
                    / (xm_A*xm_B*xm_C*(xm_A - xm_B) * (xm_B - xm_C))
    elif xmA_eq_xmB:
        integral = ((np.exp(-xm_A-xm_C)) * (-1+np.exp(xm_A)) * (-np.exp(xm_A)*xm_A + np.exp(xm_A+xm_C)*(xm_A-xm_C)+np.exp(xm_C)*xm_C)) / (xm_A**2*xm_C*(xm_A - xm_C))
    elif xmB_eq_xmC:
        integral = ((np.exp(-xm_A - xm_B)) * (-1+np.exp(xm_A)) * (-np.exp(xm_A)+np.exp(xm_B)) * (-1+np.exp(xm_B) - xm_B))\
        /(xm_A*xm_B**2*(xm_B - xm_A))
    elif xmA_eq_xmC:
        integral = ( (np.exp(-2*xm_A-xm_B))*(-1+np.exp(xm_A))*(np.exp(xm_A) - np.exp(xm_B))\
                    *(-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B)) / (xm_B*xm_A**2*(xm_A-xm_B)**2)
  
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    
    # sub case 5; jmax == jupp == jlow > jmin
    index = (jmax == jupp) * (jupp == jlow) * (jlow > jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (np.exp(-xm_A) * (-1 + np.exp(xm_A)) * (-2 + 2*np.exp(xm_A) - 2*xm_A - xm_A**2))\
        / (2*xm_A**4)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = ((-1+np.exp(xm_C))*( -xm_C + (   (xm_A*(-1+np.exp(-xm_C) + xm_C))  / (xm_C)  ) +  ( (xm_C-np.exp(-xm_A)*xm_C)  / (xm_A)  )     )) / (xm_A*(xm_A-xm_C)*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (4-4*np.cosh(xm_A) + 2*xm_A*np.sinh(xm_A)) / (xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((-1+np.exp(xm_C)) * (  ( (np.exp(-xm_B))/((xm_A-xm_B)*xm_B)   )  +  ( (xm_B - xm_C)/(xm_A*xm_B*xm_C)  )  +   (   (np.exp(-xm_C))/(xm_C*(xm_C-xm_A))    )  +    (    (np.exp(-xm_A)*(-xm_B+xm_C)) / ( xm_A*(xm_A-xm_B)*(xm_A-xm_C))  )   ))/((xm_B-xm_C)*xm_C)
    elif xmA_eq_xmB:
        integral = ((-1+np.exp(xm_C))* (  (-np.exp(-xm_C) * xm_A**2)     +    ((xm_A-xm_C)**2)    +     (np.exp(-xm_A)*xm_C*(xm_A**2 - xm_A*(-2+xm_C)-xm_C)) ))/(xm_A**2*xm_C**2*(xm_A-xm_C)**2)
    elif xmB_eq_xmC:
        integral = -((-1+np.exp(xm_B))* (  (np.exp(-xm_A) * xm_B**2)     +    -((xm_A-xm_B)**2)    +     (np.exp(-xm_B)*xm_A*(xm_A*(1+xm_B) - xm_B*(2+xm_B))) ))/(xm_B**3*xm_A*(xm_A-xm_B)**2)
    elif xmA_eq_xmC:
        integral = ((-1+np.exp(xm_A))* (  (-np.exp(-xm_B) * xm_A**2)     +    ((xm_A-xm_B)**2)    +     (np.exp(-xm_A)*xm_B*(xm_A**2 - xm_A*(-2+xm_B)-xm_B)) ))/(xm_A**3*xm_B*(xm_A-xm_B)**2)

        
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    # sub case 6; jmax > jupp == jlow = jmin
    index = (jmax > jupp) * (jupp == jlow) * (jlow == jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = ((-1 + np.exp(-xm_A)) * (2 - 2*np.exp(xm_A) + 2*xm_A + xm_A**2))\
        / (2*xm_A**4)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = (np.exp(-xm_A - xm_C)) * (-1 + np.exp(xm_A)) * ( (np.exp(xm_A)*xm_A**2)  +   (np.exp(xm_A+xm_C)*(xm_A-xm_C)*(xm_A*(-1+xm_C) -xm_C) )   -    (np.exp(xm_C)*xm_C**2) ) / (xm_A**3*(xm_A-xm_C)*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (4-4*np.cosh(xm_A) + 2*xm_A*np.sinh(xm_A)) / (xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ( (np.exp(-xm_A - xm_B - xm_C)) * (-1+np.exp(xm_A)) * \
                   (          (-np.exp(xm_A+xm_B)*xm_A*(xm_A-xm_B)*xm_B)      +         (np.exp(xm_A+xm_C)*xm_A*(xm_A-xm_C)*xm_C)     +       (np.exp(xm_B+xm_C)*(xm_B-xm_C)*(np.exp(xm_A)*(xm_A-xm_B)*(xm_A-xm_C) -xm_B*xm_C) )     ))\
                    / (xm_A**2 * (xm_A - xm_B) * xm_B * (xm_A-xm_C) * (xm_B - xm_C) * xm_C)
    elif xmA_eq_xmB:
        integral = ( (np.exp(-xm_A - xm_C)) * (-1 + np.exp(xm_A)) * ((-np.exp(xm_A)*xm_A**2) +   (np.exp(xm_A+xm_C)*(xm_A-xm_C)**2)    +    -(np.exp(xm_C)*xm_C*(-xm_A**2+xm_A*(-2+xm_C)+xm_C)) ))/ (xm_A**3*(xm_A-xm_C)**2*xm_C)
    elif xmB_eq_xmC:
        integral = ( (np.exp(-xm_A - xm_B)) * (-1 + np.exp(xm_A)) * ((-np.exp(xm_B)*xm_B**2) +   (np.exp(xm_A+xm_B)*(xm_A-xm_B)**2)    +    -(np.exp(xm_A)*xm_A*(xm_A*(1+xm_B)-xm_B*(2+xm_B))) ))/ (xm_A**2*(xm_A-xm_B)**2*xm_B**2)
    elif xmA_eq_xmC:
        integral = ( (np.exp(-xm_A - xm_B)) * (-1 + np.exp(xm_A)) * ((-np.exp(xm_A)*xm_A**2) +   (np.exp(xm_A+xm_B)*(xm_A-xm_B)**2)    +    -(np.exp(xm_B)*xm_B*(-xm_A**2+xm_A*(-2+xm_B)+xm_B)) ))/ (xm_A**3*(xm_A-xm_B)**2*xm_B)
    
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
    
    # sub case 7; jmax == jupp == jlow = jmin
    index = (jmax == jupp) * (jupp == jlow) * (jlow == jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (np.exp(-xm_A) * (6 + 2*(np.exp(xm_A)*(-3+xm_A)) + 4*xm_A + xm_A**2) ) / (2*xm_A**4)
    
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = (  (2*(-1 + np.exp(-xm_A))*xm_C**3) + (2*xm_A*xm_C**3)  + (-xm_A**2*xm_C**3) + (xm_A**3*(2-2*np.exp(-xm_C)-2*xm_C+xm_C**2)) ) / (2*xm_A**3*(xm_A-xm_C)*xm_C**3)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = (np.exp(-xm_A) / (2*xm_A**4)) * ( (-2*(3+xm_A)) + (np.exp(xm_A) * (6-4*xm_A + xm_A**2)))

    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ( (np.exp(-xm_A)) / (xm_A**2 * (xm_A-xm_B) * (xm_A - xm_C)) )      +      ( (np.exp(-xm_B)) / (xm_B**2 * (xm_B-xm_A) * (xm_B - xm_C)) )     +      ( (np.exp(-xm_C)) / (xm_C**2 * (xm_C-xm_A) * (xm_C - xm_B)) )      +        -(  (xm_B*xm_C + xm_A*(xm_B+xm_C-xm_B*xm_C))   /    (xm_A**2*xm_B**2*xm_C**2)  )
    elif xmA_eq_xmB:
        integral = (  (-xm_A**3) + (np.exp(-xm_C)*xm_A**3) + (xm_A*(xm_A-xm_C)**2*xm_C) + ((3*xm_A-2*xm_C)*xm_C**2) + ( np.exp(-xm_A)*xm_C**2 *(2*xm_C + xm_A*(-3-xm_A+xm_C))) ) / (xm_A**3 * xm_C**2 * (xm_A - xm_C)**2)
    elif xmB_eq_xmC:
        integral = ( ((xm_A - xm_B)**2 * (xm_A*(-2 + xm_B) - xm_B))     +     (np.exp(-xm_A) * xm_B**3)     +    (np.exp(-xm_B)*xm_A**2 * (xm_A*(2+xm_B) - xm_B*(3 + xm_B))  )      ) / (xm_A**2 * (xm_A - xm_B)**2 * xm_B**3)
    elif xmA_eq_xmC:
        integral =  (  (-xm_A**3) + (np.exp(-xm_B)*xm_A**3) + (xm_A*(xm_A-xm_B)**2*xm_B) + ((3*xm_A-2*xm_B)*xm_B**2) + ( np.exp(-xm_A)*xm_B**2 *(2*xm_B + xm_A*(-3-xm_A+xm_B))) ) / (xm_A**3 * xm_B**2 * (xm_A - xm_B)**2)
 
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral #* 2
    # sub case 8; jmax == jupp > jlow = jmin
    index = (jmax == jupp) * (jupp > jlow) * (jlow == jmin)

    if xmA_eq_xmB and xmB_eq_xmC:
        integral = (np.exp(-xm_A) * (1 + -np.exp(xm_A) +xm_A)**2)\
        / (xm_A**4)
        
    elif xm_B_eq_0 and (not xmA_eq_xmC): #fABCBzero
        integral = ((np.exp(-xm_A - xm_C)) * (1+np.exp(xm_A) *(-1 + xm_A))*(1+np.exp(xm_C)*(-1+xm_C)))\
        /(xm_A**2*xm_C**2)
    elif xm_B_eq_0 and xmA_eq_xmC: #fABABzero
        integral = ( np.exp(-2*xm_A) * (1+np.exp(xm_A)*(-1 + xm_A))**2)\
        /(xm_A**4)
        
    elif (not xmA_eq_xmB) and (not xmB_eq_xmC) and (not xmA_eq_xmC):
        integral = ((np.exp(-xm_A-xm_B-xm_C)) * (-np.exp(xm_A)*xm_A + np.exp(xm_A + xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B) *(-np.exp(xm_B)*xm_B + np.exp(xm_B + xm_C)*(xm_B-xm_C) + np.exp(xm_C)*xm_C) )\
                    / (xm_A*xm_B**2*xm_C*(xm_A - xm_B) * (xm_B - xm_C))
    elif xmA_eq_xmB:
        integral = ( (np.exp(-xm_A - xm_C)) * (-1 + np.exp(xm_A)-xm_A) * ( (-np.exp(xm_A)*xm_A) +   (np.exp(xm_A+xm_C)*(xm_A-xm_C))    +  (np.exp(xm_C)*xm_C) ))/ (xm_A**3*(xm_A-xm_C)*xm_C)
    elif xmB_eq_xmC:
        integral = ( (-1+np.exp(xm_B) - xm_B) * (xm_A-np.exp(-xm_B)*xm_A + (-1+np.exp(-xm_A))*xm_B) )/ (xm_A*(xm_A-xm_B)*xm_B**3)
    elif xmA_eq_xmC:
        integral = ( (np.exp(-2*xm_A-xm_B)) * (-np.exp(xm_A)*xm_A + np.exp(xm_A+xm_B)*(xm_A-xm_B) + np.exp(xm_B)*xm_B)**2) / (xm_A**2*xm_B**2*(xm_A-xm_B)**2)
  
    C[np.where((index) != 0)] += constant[np.where(index != 0)] \
                                    * integral
 
    return C

def gamma2_E(mix, k, chi):
# like an ensemble-averaged G2, using gaussian chain 
    s2_inv = calc_sf2_inv(mix, [k])
    N = mix.N
            
    D = [1,-1]    # sign indicator
    G = 0
    for I0, I1 in product([0,1], repeat=2):
        G += s2_inv[I0, I1]*D[I0]*D[I1]
        
    return -2*chi + N*G


def gamma3_E(mix, Ks):
    K1, K2, K3 = Ks
    N = mix.N
    if np.linalg.norm(K1+K2+K3) >= 1e-10:
        raise Exception('Qs must add up to zero')
        
#     if not (abs(np.linalg.norm(K1)-np.linalg.norm(K2)) < 1e-5 \
#         and abs(np.linalg.norm(K2)-np.linalg.norm(K3)) < 1e-5):
#         raise Exception('Qs must have same length')
    
    s2inv = calc_sf2_inv(mix, [K1])
    s2inv_2 = calc_sf2_inv(mix, [K2])
    s2inv_3 = calc_sf2_inv(mix, [K3])
    
    s3 = calc_sf3(mix, [K1], [K2])

    val = 0
    for I0, I1, I2 in product([0,1], repeat=3):
        val -= s3[I0][I1][I2]* (s2inv[I0][0] - s2inv[I0][1])*\
                               (s2inv_2[I1][0] - s2inv_2[I1][1])*\
                               (s2inv_3[I2][0] - s2inv_3[I2][1])
    
    return val*(N**2) 

def gamma4_E(mix, Ks):
    K1, K2, K3, K4 = Ks
    if np.linalg.norm(K1+K2+K3+K4) >= 1e-10:
        raise Exception('Qs must add up to zero')

    N = mix.N
    
    K = np.linalg.norm(K1)
    K12 = np.linalg.norm(K1+K2)
    K13 = np.linalg.norm(K1+K3)
    K14 = np.linalg.norm(K1+K4)

    s4 = calc_sf4(mix, [K1], [K2], [K3]) 
    s31 = calc_sf3(mix, [K1], [K2])
    s32 = calc_sf3(mix, [K1], [K3])
    s33 = calc_sf3(mix,  [K1], [K4])
    s2inv = calc_sf2_inv(mix, [K])
    s21inv = calc_sf2_inv(mix, [K12])
    s22inv = calc_sf2_inv(mix, [K13])
    s23inv = calc_sf2_inv(mix, [K14])

    s2inv_2 = calc_sf2_inv(mix, [K2])
    s2inv_3 = calc_sf2_inv(mix, [K3])
    s2inv_4 = calc_sf2_inv(mix, [K4])

    G4 = np.zeros((2,2,2,2),dtype=type(1+1j))
    for a1, a2, a3, a4 in product([0,1], repeat=4):
        for I0, I1 in product([0,1], repeat=2):
            G4[a1][a2][a3][a4] += \
                s31[a1][a2][I0]*s31[a3][a4][I1]*s21inv[I0][I1] + \
                s32[a1][a4][I0]*s32[a2][a3][I1]*s22inv[I0][I1] + \
                s33[a1][a3][I0]*s33[a2][a4][I1]*s23inv[I0][I1]
    G4 -= s4
    
    val = 0
    for I0, I1, I2, I3 in product([0,1], repeat=4):
        val += G4[I0][I1][I2][I3] *\
                (s2inv[I0][0] - s2inv[I0][1])*\
                (s2inv_2[I1][0] - s2inv_2[I1][1])*\
                (s2inv_3[I2][0] - s2inv_3[I2][1])*\
                (s2inv_4[I3][0] - s2inv_4[I3][1])
                
    return val*(N**3)

def poly_mat_gen(poly_type, M, n_p = 1, FA = 0.5):
    # use this at beginning of free energy calculation to create the desired poly_mat
    # could extend to adaptivly change M and N_m to achieve FA
    if poly_type == "random":
        if FA != 0.5:
            raise Exception("for random, can only choose randomly (<fa> will be 0.5)")
#         n_mixs = 1
        rand_polys = np.random.choice(2,(n_p, M))
        rand_dens = np.random.dirichlet(np.ones(n_p),size=(1))[0]
        return rand_polys, rand_dens
    elif poly_type == "diblock":
        if n_p != 1:
            raise Exception("for diblock, can only do monodisperse- np = 1")
        if int(M*FA) != np.round(M*FA, 3):
            raise Exception("M*FA must be whole number")
        
        poly_mat = [0]*int(M*FA) + [1]*int(M*np.round((1-FA), 3))
        dens = [1.]
        return np.array(poly_mat), dens
    else:
        raise Exception("incorrect poly_type")
        
# def find_phase(poly_mat, dens, N_m, b, M, chi_array):
#     # returns an array of the phase identiifed (as a string) at each chi value in chi array
#     # for labeling dataset to then train mlp
#     res_arr = np.array([])
    
#     q_star = spinodal_gaus(poly_mat, dens, N_m, b, M)
#     q_star = q_star[0]
    
#     if q_star <= 0.01:
#         #only disorderd or macrophase separation possible
#         for CHI in chi_array:
#             G2 = gamma2_E(poly_mat, dens, N_m, b, M, q_star, CHI) 
#             if G2 < 0:
#                 phase_name = "macro"
#             elif G2 >= 0:
#                 phase_name = "dis"
#             res_arr = np.append(res_arr, [CHI, phase_name])
#         return res_arr
    
    
#     lam_q = q_star*np.array([1, 0, 0])
    
#     cyl_q1 = q_star*np.array([1, 0, 0])
#     cyl_q2 = 0.5*q_star*np.array([-1, np.sqrt(3), 0])
#     cyl_q3 = 0.5*q_star*np.array([-1, -np.sqrt(3), 0])
#     cyl_qs = np.array([cyl_q1, cyl_q2, cyl_q3])
    
#     bcc_q1 = 2**(-0.5)*q_star*np.array([1,1,0])
#     bcc_q2 = 2**(-0.5)*q_star*np.array([-1,1,0])
#     bcc_q3 = 2**(-0.5)*q_star*np.array([0,1,1])
#     bcc_q4 = 2**(-0.5)*q_star*np.array([0,1,-1])
#     bcc_q5 = 2**(-0.5)*q_star*np.array([1,0,1])
#     bcc_q6 = 2**(-0.5)*q_star*np.array([1,0,-1])
    
#     sq_6 = (1/np.sqrt(6)) * q_star
#     gyr_q1 = sq_6*np.array([-1, 2, 1])
#     gyr_q2 = sq_6*np.array([2, 1, -1])
#     gyr_q3 = sq_6*np.array([1, -1, 2])
#     gyr_q4 = sq_6*np.array([2, -1, -1])
#     gyr_q5 = sq_6*np.array([-1, 2, -1])
#     gyr_q6 = sq_6*np.array([-1, -1, 2])
    
#     gyr_q7 = sq_6*np.array([2, 1, 1])
#     gyr_q8 = sq_6*np.array([1, 2, 1])
#     gyr_q9 = sq_6*np.array([1, 1, 2])
#     gyr_q10 = sq_6*np.array([2, -1, 1])
#     gyr_q11 = sq_6*np.array([1, 2, -1])
#     gyr_q12 = sq_6*np.array([-1, 1, 2])

#     sq_cyl_q1 = q_star * np.array([1,0,0])
#     sq_cyl_q2 = q_star * np.array([0,1,0])
    
#     sim_cub_q1 = q_star * np.array([1,0,0])
#     sim_cub_q2 = q_star * np.array([0,1,0])
#     sim_cub_q3 = q_star * np.array([0,0,1])
    
#     fcc_q1 = 3**(-0.5)*q_star*np.array([1,1,1])
#     fcc_q2 = 3**(-0.5)*q_star*np.array([1,1,-1])
#     fcc_q3 = 3**(-0.5)*q_star*np.array([1,-1,1])
#     fcc_q4 = 3**(-0.5)*q_star*np.array([-1,1,1])
    
    
#     G3 = gamma3_E(poly_mat, dens, N_m, b, M, cyl_qs) # all g3s are eqivlaent
#     lam_g3 = 0
#     cyl_g3 = -(1/6) * (1/(3*np.sqrt(3))) * 12 * G3
#     bcc_g3 = -(4/(3*np.sqrt(6))) * G3 #* gamma3_E(poly_mat, dens, N_m, b, M, np.array([bcc_q6, bcc_q3, -bcc_q1]))
#     gyr_g3 = (1/6)  * (1/(12*np.sqrt(12))) * 48  * G3 #* gamma3_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q11, -gyr_q3]))
#     sq_cyl_g3 = 0
#     sim_cub_g3 = 0
#     fcc_g3 = 0
    
#     G4_00 = gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q, -lam_q, lam_q, -lam_q]))
#     lam_g4 = (1/24) * (6) * (1) * G4_00#gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q, -lam_q, lam_q, -lam_q]))
#     cyl_g4 = (1/12)* (G4_00 + \
#               4*gamma4_E(poly_mat, dens, N_m, b, M, np.array([cyl_q1, -cyl_q1, cyl_q2, -cyl_q2])))
#     bcc_g4 = (1/24)* (G4_00 \
#                      + 8*gamma4_E(poly_mat, dens, N_m, b, M, np.array([bcc_q1, -bcc_q1, bcc_q3, -bcc_q3])) \
#                      + 2*gamma4_E(poly_mat, dens, N_m, b, M, np.array([bcc_q1, -bcc_q1, bcc_q2, -bcc_q2])) \
#                      + 4*gamma4_E(poly_mat, dens, N_m, b, M, np.array([bcc_q1, -bcc_q3, bcc_q2, -bcc_q4])) )
#     gyr_g4 = (1/24)* (1/(12*12)) * (72*G4_00 + \
#                        288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q8, -gyr_q8])) + \
#                        288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q10, -gyr_q10])) + \
#                        288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q11, -gyr_q11])) + \
#                        144*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q4, -gyr_q4])) + \
#                        576*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q12, -gyr_q12])) + \
#                        -288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, gyr_q4, -gyr_q10, -gyr_q5])) + \
#                        144*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q2, gyr_q4, -gyr_q10])) + \
#                        -288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, -gyr_q5, -gyr_q7, gyr_q2])))
    
#     G4_90deg = gamma4_E(poly_mat, dens, N_m, b, M, np.array([sq_cyl_q1 , -sq_cyl_q1 , sq_cyl_q2 , -sq_cyl_q2 ]))
#     sq_cyl_g4 = (1/24) * (1/4) * (12*G4_00 + \
#                                   24*G4_90deg)
#     sim_cub_g4 = (1/24) * (1/9) * (18*G4_00+ \
#                                    72*G4_90deg)
#     fcc_g4 = (1/24) * (1/16) * (24*G4_00\
#                                 + 144*gamma4_E(poly_mat, dens, N_m, b, M, np.array([fcc_q1, -fcc_q1, fcc_q2, -fcc_q2]))\
#                                 - 48*gamma4_E(poly_mat, dens, N_m, b, M, np.array([-fcc_q1, fcc_q2, fcc_q3, fcc_q4])))
    
#     for CHI in chi_array:
#         lam_g2 = (1/2) * 2 * (1) * gamma2_E(poly_mat, dens, N_m, b, M, q_star, CHI)                
#         cyl_g2 = lam_g2#(1/2) * 6 * (1/3) * gamma2_E(poly_mat, dens, N_m, b, M, q_star, CHI)
#         bcc_g2 = lam_g2#(1/2) * 12 * (1/6) * gamma2_E(poly_mat, dens, N_m, b, M, q_star, CHI)
#         gyr_g2 = lam_g2
#         sq_cyl_g2 = lam_g2#(1/2) * (1/2) * 4 * gamma2_E(poly_mat, dens, N_m, b, M, q_star, CHI)
#         sim_cub_g2 = lam_g2
#         fcc_g2 = lam_g2
        
#         amp_l1 = optimize.fmin(lambda amps: np.real(amps**2 * lam_g2 + amps**3 * lam_g3 + amps**4 * lam_g4), \
#                               1, disp=False)
#         amp_c1 = optimize.fmin(lambda amps: np.real(amps**2 * cyl_g2 + amps**3 * cyl_g3 + amps**4 * cyl_g4), \
#                               1, disp=False)
#         amp_bcc1 = optimize.fmin(lambda amps: np.real(amps**2 * bcc_g2 + amps**3 * bcc_g3 + amps**4 * bcc_g4), \
#                               1, disp=False)
#         amp_g1 = optimize.fmin(lambda amps: np.real(amps**2 * gyr_g2 + amps**3 * gyr_g3 + amps**4 * gyr_g4), \
#                               1, disp=False)
#         amp_sq_c1 = optimize.fmin(lambda amps: np.real(amps**2 * sq_cyl_g2 + amps**3 * sq_cyl_g3 + amps**4 * sq_cyl_g4), \
#                               1, disp=False)
#         amp_sim_cub1 = optimize.fmin(lambda amps: np.real(amps**2 * sim_cub_g2 + amps**3 * sim_cub_g3 + amps**4 * sim_cub_g4), \
#                               1, disp=False)
#         amp_fcc1 = optimize.fmin(lambda amps: np.real(amps**2 * fcc_g2 + amps**3 * fcc_g3 + amps**4 * fcc_g4), \
#                               1, disp=False)

        
#         lamF = amp_l1**2 * lam_g2 + amp_l1**3 * lam_g3 + amp_l1**4 * lam_g4 
#         cylF = amp_c1**2 * cyl_g2 + amp_c1**3 * cyl_g3 + amp_c1**4 * cyl_g4 
#         bccF = amp_bcc1**2 * bcc_g2 + amp_bcc1**3 * bcc_g3 + amp_bcc1**4 * bcc_g4
#         gyrF = amp_g1**2 * gyr_g2 + amp_g1**3 * gyr_g3 + amp_g1**4 * gyr_g4
#         sq_cylF = amp_sq_c1**2 * sq_cyl_g2 + amp_sq_c1**3 * sq_cyl_g3 + amp_sq_c1**4 * sq_cyl_g4 
#         sim_cubF = amp_sim_cub1**2 * sim_cub_g2 + amp_sim_cub1**3 * sim_cub_g3 + amp_sim_cub1**4 * sim_cub_g4 
#         fccF = amp_fcc1**2 * fcc_g2 + amp_fcc1**3 * fcc_g3 + amp_fcc1**4 * fcc_g4
        
        
#         minF = min([lamF, cylF, bccF, sq_cylF, gyrF, sim_cubF, fccF])

#         if minF > 0:
#             phase_name = "dis"
#         elif minF == lamF:
#             phase_name = "lam" 
#         elif minF == cylF:
#             phase_name = "cyl"
#         elif minF == bccF:
#             phase_name = "bcc"
#         elif minF == gyrF:
#             phase_name = "gyr"
#         elif minF == sq_cylF:
#             phase_name = "sqcyl"
#         elif minF == sim_cubF:
#             phase_name = "simcub"
#         elif minF == fccF:
#             phase_name = "fcc"
#         else:
#             raise Exception("error in min F phase assignment")
        
#         res_arr = np.append(res_arr, [CHI, phase_name])
#     return res_arr

# def find_phase_2wvmd(poly_mat, dens, N_m, b, M, chi_array):
#     # returns an array of the phase identiifed (as a string) at each chi value in chi array
#     # for labeling dataset to then train mlp
#     res_arr = np.array([])
    
#     q_star = spinodal_gaus(poly_mat, dens, N_m, b, M)
#     q_star = q_star[0]
    
#     if q_star <= 0.01:
#         #only disorderd or macrophase separation possible
#         for CHI in chi_array:
#             G2 = gamma2_E(poly_mat, dens, N_m, b, M, q_star, CHI) 
#             if G2 < 0:
#                 phase_name = "macro"
#             elif G2 >= 0:
#                 phase_name = "dis"
#             res_arr = np.append(res_arr, [CHI, phase_name])
#         return res_arr
    
#     lam_q = q_star*np.array([1, 0, 0])
    
#     lam_q_2 = q_star*np.array([2, 0, 0])
    
#     cyl_q1 = q_star*np.array([1, 0, 0])
#     cyl_q2 = 0.5*q_star*np.array([-1, np.sqrt(3), 0])
#     cyl_q3 = 0.5*q_star*np.array([-1, -np.sqrt(3), 0])
#     cyl_qs = np.array([cyl_q1, cyl_q2, cyl_q3])
    
#     cyl_q1_2 = q_star*np.array([0, np.sqrt(3), 0])
#     cyl_q2_2 = 0.5*q_star*np.array([3, -np.sqrt(3), 0])
#     cyl_q3_2 = 0.5*q_star*np.array([-3, -np.sqrt(3), 0])
#     cyl_qs_2 = np.array([cyl_q1_2, cyl_q2_2, cyl_q3_2])
    
#     sq_6 = (1/np.sqrt(6)) * q_star
#     gyr_q1 = sq_6*np.array([-1, 2, 1])
#     gyr_q2 = sq_6*np.array([2, 1, -1])
#     gyr_q3 = sq_6*np.array([1, -1, 2])
#     gyr_q4 = sq_6*np.array([2, -1, -1])
#     gyr_q5 = sq_6*np.array([-1, 2, -1])
#     gyr_q6 = sq_6*np.array([-1, -1, 2])
    
#     gyr_q7 = sq_6*np.array([2, 1, 1])
#     gyr_q8 = sq_6*np.array([1, 2, 1])
#     gyr_q9 = sq_6*np.array([1, 1, 2])
#     gyr_q10 = sq_6*np.array([2, -1, 1])
#     gyr_q11 = sq_6*np.array([1, 2, -1])
#     gyr_q12 = sq_6*np.array([-1, 1, 2])
    
#     gyr_q1_2 = sq_6*np.array([2, 2, 0])
#     gyr_q2_2 = sq_6*np.array([2, 0, 2])
#     gyr_q3_2 = sq_6*np.array([0, 2, 2])
#     gyr_q4_2 = sq_6*np.array([-2, 2, 0])
#     gyr_q5_2 = sq_6*np.array([-2, 0, 2])
#     gyr_q6_2 = sq_6*np.array([0, -2, 2])
    
#     bcc_q1 = 2**(-0.5)*q_star*np.array([1,1,0])
#     bcc_q2 = 2**(-0.5)*q_star*np.array([-1,1,0])
#     bcc_q3 = 2**(-0.5)*q_star*np.array([0,1,1])
#     bcc_q4 = 2**(-0.5)*q_star*np.array([0,1,-1])
#     bcc_q5 = 2**(-0.5)*q_star*np.array([1,0,1])
#     bcc_q6 = 2**(-0.5)*q_star*np.array([1,0,-1])

#     sq_cyl_q1 = q_star * np.array([1,0,0])
#     sq_cyl_q2 = q_star * np.array([0,1,0])
    
#     sq_cyl_q1_2 = q_star * np.array([2,0,0])
#     sq_cyl_q2_2 = q_star * np.array([0,2,0])
    
#     sim_cub_q1 = q_star * np.array([1,0,0])
#     sim_cub_q2 = q_star * np.array([0,1,0])
#     sim_cub_q3 = q_star * np.array([0,0,1])
    
#     sim_cub_q1_2 = q_star * np.array([2,0,0])
#     sim_cub_q2_2 = q_star * np.array([0,2,0])
#     sim_cub_q3_2 = q_star * np.array([0,0,2])
    
#     fcc_q1 = 3**(-0.5)*q_star*np.array([1,1,1])
#     fcc_q2 = 3**(-0.5)*q_star*np.array([1,1,-1])
#     fcc_q3 = 3**(-0.5)*q_star*np.array([1,-1,1])
#     fcc_q4 = 3**(-0.5)*q_star*np.array([-1,1,1])
    
#     G3 = gamma3_E(poly_mat, dens, N_m, b, M, cyl_qs) # all g3s are eqivlaent
#     G3_211 = gamma3_E(poly_mat, dens, N_m, b, M, np.array([lam_q, lam_q, -lam_q_2])) 

#     lam_g3 = 0
#     lam_g3_2 = 0
#     lam_g3_mix = -(1/6) * 6 * G3_211
    
#     cyl_g3 = (1/6)  * (1/(3*np.sqrt(3))) * 12 * G3#gamma3_E(poly_mat, dens, N_m, b, M, cyl_qs)
#     cyl_g3_2 = -(1/6)  * (1/(3*np.sqrt(3))) * 12 * gamma3_E(poly_mat, dens, N_m, b, M, cyl_qs_2)
#     cyl_g3_mix = -(1/6) * (1/(3*np.sqrt(3))) * 36 * gamma3_E(poly_mat, dens, N_m, b, M, np.array([-cyl_q2_2, -cyl_q2, cyl_q1]))

#     gyr_g3 = (1/6)  * (1/(12*np.sqrt(12))) * 48 * G3#gamma3_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q11, -gyr_q3]))
#     gyr_g3_2 = (1/6) * (1/(6*np.sqrt(6))) * 48 * gamma3_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1_2, -gyr_q3_2, gyr_q5_2]))

#     gyr_g3_mix = -(1/6) * (1/(12*np.sqrt(6))) * 72 * gamma3_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q4, -gyr_q3_2]))

#     bcc_g3 = (4/(3*np.sqrt(6))) * G3#gamma3_E(poly_mat, dens, N_m, b, M, np.array([bcc_q6, bcc_q3, -bcc_q1]))

#     sq_cyl_g3 = 0
#     sq_cyl_g3_2 = 0
#     sq_cyl_g3_mix = -(1/6) * (1/(2*np.sqrt(2))) * 12 * G3_211
    
#     sim_cub_g3 = 0
#     sim_cub_g3_2 = 0
#     sim_cub_g3_mix = -(1/6) * (1/(3*np.sqrt(3))) * 18 * G3_211
    
#     fcc_g3 = 0

#     G4_00 = gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q, -lam_q, lam_q, -lam_q]))
#     G4_2_00 = gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q_2, -lam_q_2, lam_q_2, -lam_q_2]))
#     G4_mix = gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q, -lam_q, lam_q_2, -lam_q_2]))
    
#     lam_g4 = (1/24) * (6) * G4_00#gamma4_E(poly_mat, dens, N_m, b, M, np.array([lam_q, -lam_q, lam_q, -lam_q]))
#     lam_g4_2 = (1/24) * (6)  * G4_2_00
#     lam_g4_mix = (1/24) * 24  * G4_mix
    
#     cyl_g4 = (1/24) * (1/9) *(18*G4_00 + \
#               72*gamma4_E(poly_mat, dens, N_m, b, M, np.array([cyl_q1, -cyl_q1, cyl_q2, -cyl_q2])))
#     cyl_g4_2 = (1/24) * (1/9) * (18*gamma4_E(poly_mat, dens, N_m, b, M, np.array([cyl_q1_2 , -cyl_q1_2 , cyl_q1_2 , -cyl_q1_2 ])) + \
#               72*gamma4_E(poly_mat, dens, N_m, b, M, np.array([cyl_q1_2, -cyl_q1_2, cyl_q2_2, -cyl_q2_2]))) 

#     cyl_g4_mix1 = (1/3) * (2*gamma4_E(poly_mat, dens, N_m, b, M, np.array([-cyl_q3_2, cyl_q3, -cyl_q2, -cyl_q2_2])) + \
#                            3*gamma4_E(poly_mat, dens, N_m, b, M, np.array([cyl_q2_2, -cyl_q2_2, -cyl_q3, cyl_q3])) + \
#                            2*gamma4_E(poly_mat, dens, N_m, b, M, np.array([cyl_q1, -cyl_q1, -cyl_q3_2, cyl_q3_2])))
#     cyl_g4_mix2 = (1/3) * gamma4_E(poly_mat, dens, N_m, b, M, np.array([cyl_q2_2, cyl_q2, cyl_q2, cyl_q3]))

    
#     gyr_g4 = (1/24)* (1/(12*12)) * (72*G4_00 + \
#                        288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q8, -gyr_q8])) + \
#                        288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q10, -gyr_q10])) + \
#                        288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q11, -gyr_q11])) + \
#                        144*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q4, -gyr_q4])) + \
#                        576*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q12, -gyr_q12])) + \
#                        -288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, gyr_q4, -gyr_q10, -gyr_q5])) + \
#                        144*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q2, gyr_q4, -gyr_q10])) + \
#                        -288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, -gyr_q5, -gyr_q7, gyr_q2])))
    
#     gyr_g4_2 = (1/24) * (1/36) * (36*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1_2, -gyr_q1_2, gyr_q1_2, -gyr_q1_2])) + \
#                        288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1_2, -gyr_q1_2, gyr_q2_2, -gyr_q2_2])) + \
#                        72*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1_2, -gyr_q1_2, gyr_q4_2, -gyr_q4_2])) + \
#                        144*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1_2, -gyr_q2_2, gyr_q5_2, -gyr_q4_2])))

#     gyr_g4_mix1 = (1/24) * (1/(6*12)) * (576*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, -gyr_q1, gyr_q4_2, -gyr_q4_2])) + \
#                            576*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, -gyr_q1, gyr_q1_2, -gyr_q1_2])) + \
#                            -576*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, -gyr_q4_2, gyr_q11, -gyr_q1_2])) + \
#                            288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, -gyr_q4_2, gyr_q1, -gyr_q3_2])) + \
#                            -576*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, -gyr_q4_2, -gyr_q6_2, -gyr_q11])) + \
#                            288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q6_2, -gyr_q6_2])) + \
#                            288*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q7, gyr_q3_2, -gyr_q3_2])))
    
#     gyr_g4_mix2 = (1/3) * (2*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, -gyr_q4_2, -gyr_q2, -gyr_q6])) + \
#                            -1*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q7, -gyr_q8, gyr_q3_2, -gyr_q9])) + \
#                            -1*gamma4_E(poly_mat, dens, N_m, b, M, np.array([gyr_q1, gyr_q2, -gyr_q9, gyr_q6_2])))

#     bcc_g4 = (1/24)* (G4_00 \
#                      + 8*gamma4_E(poly_mat, dens, N_m, b, M, np.array([bcc_q1, -bcc_q1, bcc_q3, -bcc_q3])) \
#                      + 2*gamma4_E(poly_mat, dens, N_m, b, M, np.array([bcc_q1, -bcc_q1, bcc_q2, -bcc_q2])) \
#                      + 4*gamma4_E(poly_mat, dens, N_m, b, M, np.array([bcc_q1, -bcc_q3, bcc_q2, -bcc_q4])) )
    
#     G4_90deg = gamma4_E(poly_mat, dens, N_m, b, M, np.array([sq_cyl_q1 , -sq_cyl_q1 , sq_cyl_q2 , -sq_cyl_q2 ]))
#     G4_2_90deg = gamma4_E(poly_mat, dens, N_m, b, M, np.array([sq_cyl_q1_2 , -sq_cyl_q1_2 , sq_cyl_q2_2 , -sq_cyl_q2_2 ]))
#     G4_mix_90deg = gamma4_E(poly_mat, dens, N_m, b, M, np.array([sq_cyl_q1, -sq_cyl_q2_2, -sq_cyl_q1, sq_cyl_q2_2]))
    
#     sq_cyl_g4 = (1/24) * (1/4) * (12*G4_00 + \
#                                   24*G4_90deg)
#     sq_cyl_g4_2 = (1/24) * (1/4) * (12*G4_2_00 + \
#                                    24*G4_2_90deg)
#     sq_cyl_g4_mix = (1/24) * (1/4) * (48 * G4_mix + 48 * G4_mix_90deg)
    
#     sim_cub_g4 = (1/24) * (1/9) * (18*G4_00+ \
#                                    72*G4_90deg)
#     sim_cub_g4_2 = (1/24) * (1/9) * (18*G4_2_00+ \
#                                    72*G4_2_90deg)
#     sim_cub_g4_mix = (1/24) * (1/9) * (72*G4_mix + 144 * G4_mix_90deg)
    
#     fcc_g4 = (1/24) * (1/16) * (24*G4_00\
#                                 + 144*gamma4_E(poly_mat, dens, N_m, b, M, np.array([fcc_q1, -fcc_q1, fcc_q2, -fcc_q2]))\
#                                 - 48*gamma4_E(poly_mat, dens, N_m, b, M, np.array([-fcc_q1, fcc_q2, fcc_q3, fcc_q4])))
    
#     for CHI in chi_array:
#         lam_g2 = (1/2) * 2 * (1) * gamma2_E(poly_mat, dens, N_m, b, M, q_star, CHI)       
#         lam_g2_2 = (1/2) * 2 * (1) * gamma2_E(poly_mat, dens, N_m, b, M, 2*q_star, CHI)         
        
#         cyl_g2 = lam_g2
#         cyl_g2_2 = (1/2) * (1/3) * 6  * gamma2_E(poly_mat, dens, N_m, b, M, np.sqrt(3)*q_star, CHI)    

#         gyr_g2 = lam_g2
#         gyr_g2_2 = (1/2) * 12 * (1/6) * gamma2_E(poly_mat, dens, N_m, b, M, np.sqrt(4/3)*q_star, CHI)    
        
#         bcc_g2 = lam_g2
        
#         sq_cyl_g2 = lam_g2#(1/2) * (1/2) * 4 * gamma2_E(poly_mat, dens, N_m, b, M, q_star, CHI)
#         sq_cyl_g2_2 = lam_g2_2
        
#         sim_cub_g2 = lam_g2
#         sim_cub_g2_2 = lam_g2_2
        
#         fcc_g2 = lam_g2
        
#         # when doing phase minimization, should always have the gamma 3 be negative.
#         if cyl_g3>0:
#             cyl_g3*= -1
#         if cyl_g3_2>0:
#             cyl_g3_2*= -1
#         if cyl_g3_mix>0:
#             cyl_g3_mix*= -1
            
#         if gyr_g3>0:
#             gyr_g3*= -1
#         if gyr_g3_2>0:
#             gyr_g3_2*= -1
#         if gyr_g3_mix>0:
#             gyr_g3_mix*= -1
            
#         if lam_g3_mix>0:
#             lam_g3_mix*= -1

#         if sim_cub_g3_mix>0:
#             sim_cub_g3_mix*= -1
            
#         if sq_cyl_g3_mix>0:
#             sq_cyl_g3_mix*= -1
        
#         if bcc_g3>0:
#             bcc_g3*= -1

#         if fcc_g3>0:
#             fcc_g3*= -1

# #         if FA >= 0.5:
# #             initial = [-1, -1] 
# #             in_bcc = -1
# #         else:
# #             initial = [1,1]
# #             in_bcc = 1
            
#         initial = [0, 0] 
#         in_bcc = 0


#         amp_l1, amp_l2 = optimize.fmin(lambda amps: np.real(amps[0]**2 * lam_g2 + amps[0]**3 * lam_g3 + amps[0]**4 * lam_g4 + \
#                                                         amps[1]**2 * lam_g2_2 + amps[1]**3 * lam_g3_2 + amps[1]**4 * lam_g4_2 + \
#                                                         amps[0]**2 * amps[1] * lam_g3_mix + amps[0]**2 * amps[1]**2 * lam_g4_mix), \
#                               initial, disp=False)

        
#         amp_c1, amp_c2 = optimize.fmin(lambda amps: np.real(amps[0]**2 * cyl_g2 + amps[0]**3 * cyl_g3 + amps[0]**4 * cyl_g4 + \
#                                                  amps[1]**2 * cyl_g2_2 + amps[1]**3 * cyl_g3_2 + amps[1]**4 * cyl_g4_2 + \
#                                                  amps[0]**2 * amps[1] * cyl_g3_mix + amps[0]**2 * amps[1]**2 * cyl_g4_mix1 + \
#                                                  amps[0]**3 * amps[1] * cyl_g4_mix2), \
#                               initial, disp=False)
        
#         amp_g1, amp_g2 = optimize.fmin(lambda amps: np.real(amps[0]**2 * gyr_g2 + amps[0]**3 * gyr_g3 + amps[0]**4 * gyr_g4 + \
#                                                  amps[1]**2 * gyr_g2_2 + amps[1]**3 * gyr_g3_2 + amps[1]**4 * gyr_g4_2 + \
#                                                  amps[0]**2 * amps[1] * gyr_g3_mix + amps[0]**2 * amps[1]**2 * gyr_g4_mix1 + \
#                                                  amps[0]**3 * amps[1] * gyr_g4_mix2), \
#                               initial, disp=False)
        
#         amp_bcc1 = optimize.fmin(lambda amps: np.real(amps**2 * bcc_g2 + amps**3 * bcc_g3 + amps**4 * bcc_g4), \
#                               in_bcc, disp=False)
        
#         amp_sq_c1, amp_sq_c2 = optimize.fmin(lambda amps: np.real(amps[0]**2 * sq_cyl_g2 + amps[0]**3 * sq_cyl_g3 + amps[0]**4 * sq_cyl_g4 + \
#                                                         amps[1]**2 * sq_cyl_g2_2 + amps[1]**3 * sq_cyl_g3_2 + amps[1]**4 * sq_cyl_g4_2 + \
#                                                         amps[0]**2 * amps[1] * sq_cyl_g3_mix + amps[0]**2 * amps[1]**2 * sq_cyl_g4_mix), \
#                               initial, disp=False)
        
#         amp_sim_c1, amp_sim_c2 = optimize.fmin(lambda amps: np.real(amps[0]**2 * sim_cub_g2 + amps[0]**3 * sim_cub_g3 + amps[0]**4 * sim_cub_g4 + \
#                                                         amps[1]**2 * sim_cub_g2_2 + amps[1]**3 * sim_cub_g3_2 + amps[1]**4 * sim_cub_g4_2 + \
#                                                         amps[0]**2 * amps[1] * sim_cub_g3_mix + amps[0]**2 * amps[1]**2 * sim_cub_g4_mix), \
#                               initial, disp=False)
        
#         amp_fcc1 = optimize.fmin(lambda amps: np.real(amps**2 * fcc_g2 + amps**3 * fcc_g3 + amps**4 * fcc_g4), \
#                               1, disp=False)
        
#         lamF = amp_l1**2 * lam_g2 + amp_l1**3 * lam_g3 + amp_l1**4 * lam_g4 + \
#                 amp_l2**2 * lam_g2_2 + amp_l2**3 * lam_g3_2 + amp_l2**4 * lam_g4_2 +\
#                 amp_l1**2 * amp_l2 * lam_g3_mix + amp_l1**2 * amp_l2**2 * lam_g4_mix
        
#         cylF = amp_c1**2 * cyl_g2 + amp_c1**3 * cyl_g3 + amp_c1**4 * cyl_g4 +\
#                 amp_c2**2 * cyl_g2_2 + amp_c2**3 * cyl_g3_2 + amp_c2**4 * cyl_g4_2 + \
#                 amp_c1**2 * amp_c2 * cyl_g3_mix + amp_c1**2 * amp_c2**2 * cyl_g4_mix1 +\
#                 amp_c1**3 * amp_c2 * cyl_g4_mix2
        
#         gyrF = amp_g1**2 * gyr_g2 + amp_g1**3 * gyr_g3 + amp_g1**4 * gyr_g4 +\
#                 amp_g2**2 * gyr_g2_2 + amp_g2**3 * gyr_g3_2 + amp_g2**4 * gyr_g4_2 + \
#                 amp_g1**2 * amp_g2 * gyr_g3_mix + amp_g1**2 * amp_g2**2 * gyr_g4_mix1 +\
#                 amp_g1**3 * amp_g2 * gyr_g4_mix2
        
#         bccF = amp_bcc1**2 * bcc_g2 + amp_bcc1**3 * bcc_g3 + amp_bcc1**4 * bcc_g4

#         sq_cylF = amp_sq_c1**2 * sq_cyl_g2 + amp_sq_c1**3 * sq_cyl_g3 + amp_sq_c1**4 * sq_cyl_g4 + \
#                 amp_sq_c2**2 * sq_cyl_g2_2 + amp_sq_c2**3 * sq_cyl_g3_2 + amp_sq_c2**4 * sq_cyl_g4_2 +\
#                 amp_sq_c1**2 * amp_sq_c2 * sq_cyl_g3_mix + amp_sq_c1**2 * amp_sq_c2**2 * sq_cyl_g4_mix
        
#         sim_cubF = amp_sim_c1**2 * sim_cub_g2 + amp_sim_c1**3 * sim_cub_g3 + amp_sim_c1**4 * sim_cub_g4 + \
#                 amp_sim_c2**2 * sim_cub_g2_2 + amp_sim_c2**3 * sim_cub_g3_2 + amp_sim_c2**4 * sim_cub_g4_2 +\
#                 amp_sim_c1**2 * amp_sim_c2 * sim_cub_g3_mix + amp_sim_c1**2 * amp_sim_c2**2 * sim_cub_g4_mix

#         fccF = amp_fcc1**2 * fcc_g2 + amp_fcc1**3 * fcc_g3 + amp_fcc1**4 * fcc_g4

# #         num_iters = 100
# #         lam = basinhopping(lambda amps: np.real(amps[0]**2 * lam_g2 + amps[0]**3 * lam_g3 + amps[0]**4 * lam_g4 + \
# #                                                         amps[1]**2 * lam_g2_2 + amps[1]**3 * lam_g3_2 + amps[1]**4 * lam_g4_2 + \
# #                                                         amps[0]**2 * amps[1] * lam_g3_mix + amps[0]**2 * amps[1]**2 * lam_g4_mix), \
# #                               initial, disp=False, niter = num_iters)


# #         cyl = basinhopping(lambda amps: np.real(amps[0]**2 * cyl_g2 + amps[0]**3 * cyl_g3 + amps[0]**4 * cyl_g4 + \
# #                                                  amps[1]**2 * cyl_g2_2 + amps[1]**3 * cyl_g3_2 + amps[1]**4 * cyl_g4_2 + \
# #                                                  amps[0]**2 * amps[1] * cyl_g3_mix + amps[0]**2 * amps[1]**2 * cyl_g4_mix1 + \
# #                                                  amps[0]**3 * amps[1] * cyl_g4_mix2), \
# #                               initial, disp=False, niter = num_iters)
        
# #         gyr = basinhopping(lambda amps: np.real(amps[0]**2 * gyr_g2 + amps[0]**3 * gyr_g3 + amps[0]**4 * gyr_g4 + \
# #                                                  amps[1]**2 * gyr_g2_2 + amps[1]**3 * gyr_g3_2 + amps[1]**4 * gyr_g4_2 + \
# #                                                  amps[0]**2 * amps[1] * gyr_g3_mix + amps[0]**2 * amps[1]**2 * gyr_g4_mix1 + \
# #                                                  amps[0]**3 * amps[1] * gyr_g4_mix2), \
# #                               initial, disp=False, niter = num_iters)
        
# #         bcc = basinhopping(lambda amps: np.real(amps**2 * bcc_g2 + amps**3 * bcc_g3 + amps**4 * bcc_g4), \
# #                               in_bcc, disp=False, niter = num_iters)
        
# #         lamF = lam.fun
        
# #         cylF = cyl.fun
        
# #         gyrF = gyr.fun
        
# #         bccF = bcc.fun

#         minF = min([lamF, cylF, gyrF, bccF, sq_cylF, sim_cubF, fccF])

#         if minF > 0:
#             phase_name = "dis"
#         elif minF == lamF:
#             phase_name = "lam" 
#         elif minF == cylF:
#             phase_name = "cyl"
#         elif minF == bccF:
#             phase_name = "bcc"
#         elif minF == gyrF:
#             phase_name = "gyr"
#         elif minF == sq_cylF:
#             phase_name = "sqcyl"
#         elif minF == sim_cubF:
#             phase_name = "simcub"
#         elif minF == fccF:
#             phase_name = "fcc"
#         else:
#             raise Exception("error in min F phase assignment")
        
#         res_arr = np.append(res_arr, [CHI, phase_name])
#     return res_arr