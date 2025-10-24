from expl_bind_beaker_util import *

def binding_state_calc(p_markA, p_markB, phi_Au, phi_Bu, e_A, e_B):
    p_AmAb = p_markA * phi_Au*np.exp(e_A) #A marked A bound
    p_BmAb = p_markB * phi_Au #B marked A bound
    p_AmBb = p_markA * phi_Bu #A marked B bound
    p_BmBb = p_markB * phi_Bu*np.exp(e_B) #B marked B bound
    
    q = 1 + p_AmAb + p_BmAb + p_AmBb + p_BmBb
    s_Aj = (1*p_AmAb + 1*p_BmAb) / q
    s_Bj = (1*p_AmBb + 1*p_BmBb) / q
    return s_Aj, s_Bj

def calc_fas(s_bnd_A, s_bnd_B):
    # [sig_0, sig_A, sig_B, sig_AB] =  calc_sisjs(s_bnd_A, s_bnd_B) #[sig_0, sig_A, sig_B, sig_AB]
    sig_A = s_bnd_A
    sig_B = s_bnd_B
    sig_0 = 1 - s_bnd_A - s_bnd_B

    f_a = np.sum(sig_A) / (np.sum(np.ones(len(s_bnd_A))))
    f_b = np.sum(sig_B) / (np.sum(np.ones(len(s_bnd_A))))
    # f_ab = np.sum(sig_AB) / (np.sum(np.ones(len(s_bnd_A))))
    f_o = np.sum(sig_0) / (np.sum(np.ones(len(s_bnd_A))))
    return [f_a, f_b, f_o]


def calc_mu_phi_bind(psol, ):
    # for each (phi_a_i, phi_b_i) pair going to:
    # 1) calculate corresponging final volume fractions (phi_p_f scalar; phi_a_f,phi_b_f matrix for each possible pair)
    # 2) calculate resulting phi_Au, phi_BU using newton raphson
    # 3) calculate phi_Ab, phi_Bb, binding profiles, and chemical potentials

    phi_p_i = psol.phi_p_i # volume fractionsof initila beakers
    V_p = psol.V_p # raw volumes of initial beakers
    phi_a_i = psol.phi_a_i
    V_A = psol.V_A
    phi_b_i = psol.phi_b_i
    V_B = psol.V_B

    v_P = psol.v_p
    N_P = psol.N_P
    b_P = psol.b_P
    v_A = psol.v_A
    N_A = psol.N_A
    b_A = psol.b_A
    v_B = psol.v_B
    N_B = psol.N_B
    b_B = psol.b_B
    M = psol.M
    p_markA, p_markB = psol.poly_marks
    chi_AB = psol.chi_AB
    e_A, e_B = psol.e_m

    phi_p_f = (V_p * phi_p_i) / (V_p + V_A + V_B)

    phi_a_f = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    phi_b_f = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    phi_s = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    phi_tot = np.zeros((len(phi_a_i),len(phi_b_i))) - 1 # POST Newton-raphson
    
    phi_Au_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    phi_Ab_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    phi_Bu_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    phi_Bb_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    mu_A_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    mu_B_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    fA_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    fB_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    f0_mat = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
    sA_mat = np.zeros((len(phi_a_i),len(phi_b_i), M)) - 1
    sB_mat = np.zeros((len(phi_a_i),len(phi_b_i), M)) - 1

    chi_AB = chi_AB / (N_P * phi_p_f)

    for i in range(len(phi_a_i)):
        for j in range(len(phi_b_i)):
            phiai = phi_a_i[i]
            phibi = phi_b_i[j]

            phiaf = (V_A * phiai) / (V_p + V_A + V_B)
            phibf = (V_B * phibi) / (V_p + V_A + V_B)
            phisf = 1 - phi_p_f - phiaf - phibf
            phi_a_f[i,j] = phiaf
            phi_b_f[i,j] = phibf
            phi_s[i,j] = phisf

            phi_Au, phi_Bu = find_unbound(psol, phi_p_f, phiaf, phibf)
            # phi_Ab = phiaf - phi_Au
            # phi_Bb = phibf - phi_Bu
            # print("phiAu, Bu: ", phi_Au, phi_Bu)
            s_Aj, s_Bj  = binding_state_calc(p_markA, p_markB, phi_Au, phi_Bu, e_A, e_B)
            fA, fB, f0 = calc_fas(s_Aj, s_Bj)


            phi_Ab = ((N_A*v_A)/ (N_P*v_P)) * phi_p_f * np.sum(s_Aj)
            phi_Bb = ((N_B*v_B)/ (N_P*v_P)) * phi_p_f * np.sum(s_Bj)
            # print("phi_p_f", phi_p_f)
            # print("phiaf", phiaf)
            # print("phibf", phibf)
            # print("phis", phi_s)

            phi_tot_orig = phi_p_f + phiaf + phibf + phisf
            phi_tot_calc = phi_p_f + phi_Ab + phi_Au + phi_Bb + phi_Bu + phisf
            # print("--"*20)
            # print("phi_tot_orig:", phi_tot_orig)
            # print("phi_tot_calc:", phi_tot_calc)
            # print("--"*3)
            # print("phiA_tot: ", phiaf)
            # print("phiAu + phiAb: ", phi_Au + phi_Ab)
            # print("--"*3)
            # print("phiB_tot: ", phibf)
            # print("phiBu + phiBb: ", phi_Bu + phi_Bb)

            # ensure that sum of phiau, phiab, phibu, phibb, phip, phis ~= 1
            if not np.isclose(phi_tot_calc, phi_tot_orig, rtol = 0.1):
                print("phi_tot_orig:", phi_tot_orig)
                print("phi_tot_calc:", phi_tot_calc)
                raise ValueError("phi_totals should match b/w calcs")

            mu_A = np.log(phi_Au) + v_A*N_A*chi_AB*(phi_Bb+phi_Bu)
            mu_B = np.log(phi_Bu) + v_B*N_B*chi_AB*(phi_Ab+phi_Au)

            # store results in matrices, and return!!
            phi_Au_mat[i,j] = phi_Au
            phi_Bu_mat[i,j] = phi_Bu
            phi_Ab_mat[i,j] = phi_Ab
            phi_Bb_mat[i,j] = phi_Bb
            mu_A_mat[i,j] = mu_A
            mu_B_mat[i,j] = mu_B 
            fA_mat[i,j] = fA
            fB_mat[i,j] = fB
            f0_mat[i,j] = f0 
            sA_mat[i,j] = s_Aj
            sB_mat[i,j] = s_Bj

    return phi_p_f, phi_a_f, phi_b_f, phi_s, phi_Au_mat, phi_Ab_mat, \
    phi_Bu_mat, phi_Bb_mat, mu_A_mat, mu_B_mat, fA_mat, fB_mat, f0_mat, sA_mat, sB_mat


def residuals(x,y, sA, sB, phiA_tot, phiB_tot, C_A, C_B):
    D = 1.0 + x*sA + y*sB 
    F1 = (phiA_tot - x) - C_A * np.sum( x * sA / D )
    F2 = (phiB_tot - y) - C_B * np.sum( y * sB / D )
    return F1, F2, D

def jacobian(x,y,sA, sB, C_A, C_B, D):
    # D is vector of denominators for each j
    D2 = D**2
    # partials for F1
    dF1_dx = -1.0 - C_A * np.sum( sA * (1.0 + y * sB) / D2 )
    dF1_dy = - C_A * np.sum( - x * sA * sB / D2 )   # equals -C_A * sum( x*sA*sB / D2 ) times (-1)
    # partials for F2
    dF2_dx = - C_B * np.sum( - y * sB * sA / D2 )
    dF2_dy = -1.0 - C_B * np.sum( sB * (1.0 + x * sA) / D2 )
    return np.array([[dF1_dx, dF1_dy],
                    [dF2_dx, dF2_dy]])


def find_unbound(psol, phi_p, phiA_tot, phiB_tot):
    v_P = psol.v_p
    N_P = psol.N_P
    b_P = psol.b_P
    v_A = psol.v_A
    N_A = psol.N_A
    b_A = psol.b_A
    v_B = psol.v_B
    N_B = psol.N_B
    b_B = psol.b_B
    M = psol.M
    # phi_S = psol.solv_cons
    # phi_P = psol.phi_p
    pA, pB = psol.poly_marks
    chi_AB = psol.chi_AB
    epsA, epsB = psol.e_m


    C_A = (N_A*v_A)/(N_P*v_P)*phi_p 
    C_B = (N_B*v_B)/(N_P*v_P)*phi_p

    # pA = pa_vec
    # pB = pb_vec 
    # epsA = e_m[0]
    # epsB = e_m[1]

    sA = pA * np.exp(epsA) + pB
    sB = pB * np.exp(epsB) + pA


    # initial guess
    x = max(1e-8, phiA_tot*0.5)
    y = max(1e-8, phiB_tot*0.5)

    for k in range(60):
        F1,F2,D = residuals(x,y, sA, sB, phiA_tot, phiB_tot, C_A, C_B)
        res_norm = np.sqrt(F1**2 + F2**2)
        if res_norm < 1e-12:
            break
        J = jacobian(x,y,sA, sB, C_A, C_B, D)
        # solve J * delta = -F
        delta = np.linalg.solve(J, -np.array([F1,F2]))
        x_new = x + delta[0]
        y_new = y + delta[1]
        # enforce non-negativity
        if x_new < 0: x_new = 1e-12
        if y_new < 0: y_new = 1e-12
        x, y = x_new, y_new

    # print("solution:", x, y, "residual norm", res_norm, "iters", k)
    phi_Au, phi_Bu = x,y
    return phi_Au, phi_Bu


        #     # calculate final volume fractions
        # phi_p_f = (v_p * phi_p_i) / (v_p + v_a + v_b)

        # phi_a_f = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
        # phi_b_f = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
        # phi_s = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
        # phi_tot = np.zeros((len(phi_a_i),len(phi_b_i))) - 1
        # for i in range(len(phi_a_i)):
        #     for j in range(len(phi_b_i)):
        #         phiai = phi_a_i[i]
        #         phibi = phi_b_i[j]

        #         phiaf = (v_a * phiai) / (v_p + v_a + v_b)
        #         phibf = (v_b * phibi) / (v_p + v_a + v_b)
        #         phi_a_f[i,j] = phiaf
        #         phi_b_f[i,j] = phibf
        #         phi_s[i,j] = 1 - phi_p_f - phiaf - phibf
        
        # self.phi_p_f = phi_p_f
        # self.phi_