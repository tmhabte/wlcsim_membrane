from expl_bind_binding_calc import *
from expl_bind_s3_integrals import *
import numpy as np
from scipy.special import exprel

def _f1_stable(x, N):
    """Stable eval of (1 - exp(-xN))/x."""
    if abs(x) < 1e-12:
        return N - 0.5*x*N**2 + (x**2)*N**3/6.0 - (x**3)*N**4/24.0
    return np.exp(-x*N) * N * exprel(-x*N)

def _f2_stable(x, N):
    """Stable eval of (1 - exp(-xN)(1+xN))/x^2."""
    if abs(x) < 1e-8:
        # higher-order series expansion
        return (N**2)/2.0 - (x*N**3)/3.0 + (x**2*N**4)/8.0 - (x**3*N**5)/30.0 \
               + (x**4*N**6)/144.0 - (x**5*N**7)/840.0
    else:
        return (1.0 - np.exp(-x*N)*(1.0 + x*N)) / (x**2)
    

def S_AAAA41(k1, k2, k3, k4, bA, N_A, tol=1e-12):#(N, x1, x2, x3, tol=1e-12):
    """
    Evaluate I(N; x1,x2,x3) = nested 4-fold integral described above.
    Handles degenerate and small-x cases stably.
    """
    # all zeros -> N^4/24

    q3 = k4
    q2 = k3 + k4
    q1 = k2 + k3 + k4

    x1 = (bA**2 / 6.0) * (np.asarray(q1)**2)
    x2 = (bA**2 / 6.0) * (np.asarray(q2)**2)
    x3 = (bA**2 / 6.0) * (np.asarray(q3)**2)

    N = N_A
    if abs(x1) < tol and abs(x2) < tol and abs(x3) < tol:
        return N**4 / 24.0

    xs = [x1, x2, x3]

    # if any two are (near-)equal, handle pair limit explicitly:
    if np.isclose(x1, x2, rtol=0, atol=tol) and not np.isclose(x1, x3, rtol=0, atol=tol):
        # x1==x2 != x3  -> use derivative-limit formula
        a = x1; b = x3
        def H(x): return N * _f1_stable(x, N) - _f2_stable(x, N)
        # derivative dH/da numeric-stable analytic:
        # d/dx [N*F1 - F2] = N*dF1/dx - dF2/dx ; implement via small-diff if necessary
        # here use finite difference with adaptive step:
        h = max(abs(a)*1e-6, 1e-8)
        Hm = H(a - h); Hp = H(a + h)
        dH = (Hp - Hm) / (2*h)
        return ( (H(b) - H(a)) - (b - a)*dH ) / ((b - a)**2)

    if np.isclose(x1, x3, rtol=0, atol=tol) and not np.isclose(x1, x2, rtol=0, atol=tol):
        a = x1; b = x2
        def H(x): return N * _f1_stable(x, N) - _f2_stable(x, N)
        h = max(abs(a)*1e-6, 1e-8)
        Hm = H(a - h); Hp = H(a + h)
        dH = (Hp - Hm) / (2*h)
        return ( (H(b) - H(a)) - (b - a)*dH ) / ((b - a)**2)

    if np.isclose(x2, x3, rtol=0, atol=tol) and not np.isclose(x1, x2, rtol=0, atol=tol):
        a = x2; b = x1
        def H(x): return N * _f1_stable(x, N) - _f2_stable(x, N)
        h = max(abs(a)*1e-6, 1e-8)
        Hm = H(a - h); Hp = H(a + h)
        dH = (Hp - Hm) / (2*h)
        return ( (H(b) - H(a)) - (b - a)*dH ) / ((b - a)**2)

    # triple-equal handled above; now general distinct case
    Hvals = [N * _f1_stable(x, N) - _f2_stable(x, N) for x in xs]
    val = 0.0
    for i in range(3):
        xi = xs[i]
        denom = 1.0
        for j in range(3):
            if i==j: continue
            denom *= (xs[j] - xi)
        val += Hvals[i] / denom
    return float(val)
    

# # OLD-NOTE HIGH TOLERANCE
# def S_AAAA41(k1, k2, k3, k4, bA, N_A, tol=1e-2):
#     """
#     Case (4,1) AAAA: all four A's on same binder.
#     Stable evaluation using _f1_stable/_f2_stable.
#     """
#     q3 = k4
#     q2 = k3 + k4
#     q1 = k2 + k3 + k4

#     x1 = (bA**2 / 6.0) * (np.asarray(q1)**2)
#     x2 = (bA**2 / 6.0) * (np.asarray(q2)**2)
#     x3 = (bA**2 / 6.0) * (np.asarray(q3)**2)
#     N = N_A

#     # all ~0 → volume of 4-simplex
#     if abs(x1) < tol and abs(x2) < tol and abs(x3) < tol:
#         return N**4 / 24.0

#     # triple equal case
#     if np.isclose(x1, x2, atol=tol) and np.isclose(x1, x3, atol=tol):
#         x = x1
#         if abs(x) < tol:
#             return N**4 / 24.0
#         # series-safe limit
#         return (N**4/24.0
#                 - (x*N**5)/60.0
#                 + (x**2*N**6)/360.0
#                 - (x**3*N**7)/2520.0)

#     # general case: partial fraction style, but using stable f1,f2
#     xs = [x1, x2, x3]
#     val = 0.0
#     for i, xi in enumerate(xs):
#         denom = 1.0
#         for j, xj in enumerate(xs):
#             if i == j:
#                 continue
#             denom *= (xj - xi)
#         Ai = 1.0 / denom
#         f1 = _f1_stable(xi, N)
#         f2 = _f2_stable(xi, N)
#         val += Ai * (N * f1 - f2)

#     return float(val)



def S_AAAA42(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_trip, j_iso, tol=1e-12):
    """
    I^{(4,2)}: three points (k1,k2,k3) on same binder at j_trip, isolated point k4 at j_iso.
    Mapping convention (for the triple block alone, s-ordered 1<=2<=3):
      use S_AAA31 with (k_alpha = k3, k_beta = k2 + k3)
    Isolated binder single factor uses x_iso = (bA^2/6)*k4^2
    Backbone prefactor D = (bP^2/6)*(N_P/(M-1)) multiplies k4^2 * |j_iso - j_trip|
    """
    # triple block: use canonical 3-point mapping (for triple indices 1..3)
    # S_AAA31 expects (k_alpha, k_beta, bA, N_A) as you provided earlier:
    k_alpha = k3
    k_beta = k2 + k3
    # call your S_AAA31 implementation (we assume it exists in the namespace)
    I_trip = S_AAA31(k_alpha, k_beta, bA, N_A)

    # isolated single
    x_iso = (bA**2 / 6.0) * (k4**2)
    single_iso = (1.0 - np.exp(-x_iso * N_A)) / x_iso if not np.isclose(x_iso, 0.0, atol=tol) else N_A

    # backbone separation exponential (natural/minimal mapping: use isolated k)
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k4**2) * abs(j_iso - j_trip))

    return backbone * I_trip * single_iso

def S_AAAA43(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_pair, j3, j4, tol=1e-12):
    """
    I^{(4,3)}: pair (k1,k2) on binder at j_pair, singles at j3 (k3) and j4 (k4).
    Canonical mapping for q_pair (if the pair are the earliest two points in ordering):
      q_pair = k2 + k3 + k4  -> x_pair = (bA^2/6) * |q_pair|^2
    single factors use x3=(bA^2/6)*k3^2, x4=(bA^2/6)*k4^2
    Backbone prefactor D uses bP and N_P as usual.
    """
    # x for pair (canonical earliest-pair mapping)
    q_pair = k2 + k3 + k4
    x_pair = (bA**2 / 6.0) * (q_pair**2)

    # 2-point nested integral (pair)
    if np.isclose(x_pair, 0.0, atol=tol):
        pair_val = N_A**2 / 2.0
    else:
        pair_val = (-1.0 + np.exp(-x_pair * N_A) + x_pair * N_A) / (x_pair**2)

    # singles
    x3 = (bA**2 / 6.0) * (k3**2)
    x4 = (bA**2 / 6.0) * (k4**2)
    single3 = (1.0 - np.exp(-x3 * N_A)) / x3 if not np.isclose(x3, 0.0, atol=tol) else N_A
    single4 = (1.0 - np.exp(-x4 * N_A)) / x4 if not np.isclose(x4, 0.0, atol=tol) else N_A

    # backbone prefactor: connect pair to singles (natural mapping uses each single's k)
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k3**2) * abs(j_pair - j3) - D * (k4**2) * abs(j_pair - j4))

    return backbone * pair_val * single3 * single4

def S_AAAA44(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j1, j2, j3, j4, tol=1e-12):
    """
    I^{(4,4)}: all four points on different binders.
    Single-point factors times backbone prefactor (natural nearest-neighbor mapping).
    If you use different phase bookkeeping, replace the k^2's in the backbone factor
    with your desired q_bb^2 combinations.
    """
    xs = [(bA**2 / 6.0) * (k**2) for k in (k1, k2, k3, k4)]
    def single(x):
        return (1.0 - np.exp(-x * N_A)) / x if not np.isclose(x, 0.0, atol=tol) else N_A

    prod = 1.0
    for x in xs:
        prod *= single(x)

    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k1**2) * abs(j2 - j1)
                      - D * (k2**2) * abs(j3 - j2)
                      - D * (k3**2) * abs(j4 - j3))

    return backbone * prod


# def S_AAAP41(k1, k2, k3, k4, bA, bP, N_A, N_P, M, tol=1e-12):
#     """
#     Case (4,1) for AAAP: A1,A2,A3 and P all attached to the same backbone monomer.
#     The P point is anchored (no backbone integral). The remaining integral is the
#     triple-nested integral over the three A contour variables:
#         I = ∫_0^{N_A} dn3 ∫_0^{n3} dn2 ∫_0^{n2} dn1 exp[-x2 (n3-n2) - x1 (n2-n1)]
#     Canonical mapping (for A1,A2,A3): q2 = k3, q1 = k2 + k3.
#     k1 (the leftmost A) does not enter the triple-block exponent directly under this
#     s-ordering — its contribution shows up when summing permutations externally.
#     """
#     # build x's (scalars)
#     q2 = k3
#     q1 = k2 + k3
#     x1 = (bA**2 / 6.0) * (q1**2)
#     x2 = (bA**2 / 6.0) * (q2**2)
#     N = N_A

#     # If both x1,x2 are ~0 -> integral = volume of 3-simplex = N^3/6
#     if abs(x1) < tol and abs(x2) < tol:
#         return N**3 / 6.0

#     # Stable evaluation of the triple nested integral:
#     # Use the standard closed form (same algebraic structure as your S_AAA31)
#     # Let f(x) = (1 - exp(-x N)) / x ; g(x) = (1 - exp(-x N)(1 + x N)) / x^2
#     f1 = _f1_stable(x1, N)
#     f2 = _f1_stable(x2, N)
#     g1 = _f2_stable(x1, N)
#     g2 = _f2_stable(x2, N)

#     if abs(x1) < 1e-2 and abs(x2) < 1e-2:
#         # Series expansion for whole triple integral
#         # print("return 2")
#         # print(N**3/6.0 - (N**4/24.0)*(x1 + x2) + (N**5/120.0)*(x1**2 + 3*x1*x2 + x2**2))
#         return N**3/6.0 - (N**4/24.0)*(x1 + x2) + (N**5/120.0)*(x1**2 + 3*x1*x2 + x2**2)
        
#     # analytic expression (equivalent to the S_AAA31 closed form)
#     # I = (1/(x1 - x2)) * [ (f2 - f1) / x? ... ]
#     # A stable and compact form is:
#     if np.isclose(x1, x2, atol=tol):
#         # pair limit x1 -> x2
#         # limit gives: (N / x1**2) + (N*x1**2*exp(-x1*N) - 2*x1*(1 - exp(-x1*N)))/x1**4
#         if abs(x1) < tol:
#             return N**3 / 6.0
#         num = (N * x1**2 * np.exp(-x1 * N) - 2.0 * x1 * (1.0 - np.exp(-x1 * N)))
#         # print("return 3")
#         # print((N / x1**2) + num / (x1**4))
#         return (N / x1**2) + num / (x1**4)
#     else:
#         # general distinct case using stable f & g
#         # print("GENERAL- WHERE THE ISSUE IS")
#         # print((N / (x1 * x2)
#         #         + (f1) / (x1**2 * (x1 - x2))
#         #         - (f2) / (x2**2 * (x1 - x2))))
#         # print("ALT")
#         # print(N/(x1*x2) + ( (f1/x1**2) - (f2/x2**2) ) / (x1 - x2))
#         return (N / (x1 * x2)
#                 + (f1) / (x1**2 * (x1 - x2))
#                 - (f2) / (x2**2 * (x1 - x2)))

def S_AAAP41(k1, k2, k3, k4, bA, bP, N_A, N_P, M, tol=1e-12):
    """
    I^(4,1) for AAAP: three A contour integrals with the binder-attachment anchored (n1=0).
    Canonical mapping:
       x_alpha = (bA^2/6) * k4^2
       x_beta  = (bA^2/6) * (k3 + k4)^2
       x_delta = (bA^2/6) * (k2 + k3 + k4)^2

    Returns numerical value with stable handling of small / nearly-equal x's.
    """
    # build x's
    x_alpha = (bA**2 / 6.0) * (np.asarray(k4)**2)
    x_beta  = (bA**2 / 6.0) * (np.asarray(k3 + k4)**2)
    x_delta = (bA**2 / 6.0) * (np.asarray(k2 + k3 + k4)**2)

    xs = [float(x_alpha), float(x_beta), float(x_delta)]
    N = float(N_A)

    # all approximately zero -> N^3 / 6 (volume of 3-simplex)
    if all(abs(x) < tol for x in xs):
        return N**3 / 6.0

    # helper H(x) = F1(x)  (for this anchored-1 case the partial-frac numerator is F1)
    def H(x):
        return _f1_stable(x, N)

    # handle pairwise-equality / degeneracies explicitly
    # If two are equal (within tol) use an analytic limit; we implement the exact algebraic limit below.
    def pair_limit(a, b, c):
        # assume a == b (repeated), c distinct. We want final value:
        # I = H(a)/((a-c)*(a-b=0)) + H(b)/... -> compute limit a->b analytically.
        # Derived closed form for a==b:
        # I = (H(c) - H(a) - (c - a)*H'(a))/((c-a)^2)
        # We'll compute H'(a) numerically via small central difference for stability.
        h = max(abs(a) * 1e-6, 1e-8)
        Hp = H(a + h); Hm = H(a - h)
        Hp_arr = (Hp - Hm) / (2*h)
        return (H(c) - H(a) - (c - a) * Hp_arr) / ((c - a) ** 2)

    # check pairwise equals
    xa, xb, xc = xs
    # three equal handled above
    if np.isclose(xa, xb, atol=tol) and not np.isclose(xa, xc, atol=tol):
        return pair_limit(xa, xb, xc)
    if np.isclose(xa, xc, atol=tol) and not np.isclose(xa, xb, atol=tol):
        return pair_limit(xa, xc, xb)
    if np.isclose(xb, xc, atol=tol) and not np.isclose(xb, xa, atol=tol):
        return pair_limit(xb, xc, xa)

    # general distinct case: partial fraction sum
    Hvals = [H(x) for x in xs]
    val = 0.0
    for i in range(3):
        denom = 1.0
        xi = xs[i]
        for j in range(3):
            if i == j:
                continue
            denom *= (xi - xs[j])
        val += Hvals[i] / denom
    return float(val)

def S_AAAP42(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_trip, j_iso, tol=1e-12):
    """
    Case (4,2): triple-A block at j_trip (A1,A2,A3), isolated P at j_iso.
    Returns: triple-block integral * backbone exponential factor.
    """
    # compute triple block using same routine as S_AAAP41 (s-ordered)
    I_trip = S_AAAP41(k1, k2, k3, k4, bA, bP, N_A, N_P, M, tol=tol)
    # backbone prefactor: D * k4^2 * |j_iso - j_trip|
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k4**2) * abs(j_iso - j_trip))
    return backbone * I_trip

def S_AAAP43(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_pair, j_a3, j_p, tol=1e-12):
    """
    Case (4,3): A-pair at j_pair (k1,k2), single A at j_a3 (k3), P at j_p (k4).
    Returns: pair nested integral * single A * backbone prefactors.
    """
    # x for pair: canonical mapping (if pair are earliest two)
    q_pair = k2 + k3 + k4   # canonical for earliest pair in full ordering; change if your ordering differs
    x_pair = (bA**2 / 6.0) * (q_pair**2)

    # pair nested integral (2-point)
    if np.isclose(x_pair, 0.0, atol=tol):
        pair_val = N_A**2 / 2.0
    else:
        pair_val = (-1.0 + np.exp(-x_pair * N_A) + x_pair * N_A) / (x_pair**2)

    # single A (third A)
    xA3 = (bA**2 / 6.0) * (k3**2)
    singleA3 = _f1_stable(xA3, N_A)

    # backbone prefactors using D and minimal mapping: connect pair to P and to A3
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k4**2) * abs(j_pair - j_p) - D * (k3**2) * abs(j_pair - j_a3))

    return backbone * pair_val * singleA3

def S_AAAP44(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j1, j2, j3, j4, tol=1e-12):
    """
    Case (4,4): all points on distinct binders/monomers: factorized singles times backbone prefactor.
    Here j1,j2,j3 are the three A-binding backbone indices and j4 is the P backbone index.
    """
    xs = [(bA**2 / 6.0) * (k**2) for k in (k1, k2, k3)]
    prod = 1.0
    for x in xs:
        prod *= _f1_stable(x, N_A)

    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    # nearest-neighbor mapping for backbone separations (minimal choice)
    backbone = np.exp(- D * (k1**2) * abs(j2 - j1)
                      - D * (k2**2) * abs(j3 - j2)
                      - D * (k3**2) * abs(j4 - j3))

    return backbone * prod


import numpy as np

# # stable helpers (repeated here for convenience)
# def _f1_stable(x, L):
#     if abs(x) < 1e-12:
#         return L - 0.5*x*L**2 + (x**2)*L**3/6.0
#     return (1.0 - np.exp(-x*L)) / x

# def _f2_stable(x, L):
#     if abs(x) < 1e-10:
#         return (L**2)/2.0 - (x*L**3)/6.0 + (x**2 * L**4)/24.0
#     return (1.0 - np.exp(-x*L)*(1.0 + x*L)) / (x**2)


# -------------------------------
# I^(4,1): all four attached to same backbone monomer
# -------------------------------
def S_AAPP41(k1, k2, k3, k4, bA, bP, N_A, N_P, M, tol=1e-12):
    """
    s-ordered integral for topology (4,1): all attached to same backbone monomer.
    Canonical mapping for two A points (positions 1 and 2):
      q_alpha = k2 + k3 + k4
      q_beta  = k3 + k4
    Integral: I = ∫_0^{N_A} dn2 ∫_0^{n2} dn1 exp[-x_alpha (n2-n1) - x_beta n1]
    """
    q_alpha = k2 + k3 + k4
    q_beta  = k3 + k4
    x_alpha = (bA**2 / 6.0) * (q_alpha**2)
    x_beta  = (bA**2 / 6.0) * (q_beta**2)
    N = N_A

    # small x limits
    if abs(x_alpha) < tol and abs(x_beta) < tol:
        return N**2 / 2.0  # integral of triangle area

    # equal-case
    if np.isclose(x_alpha, x_beta, atol=tol):
        if abs(x_alpha) < tol:
            return N**2 / 2.0
        # limit expression (stable)
        return ( (1.0 - np.exp(-x_alpha*N) - x_alpha*N*np.exp(-x_alpha*N)) / (x_alpha**2) )

    # general distinct case (stable using _f1_stable):
    f_alpha = _f1_stable(x_alpha, N)
    f_beta  = _f1_stable(x_beta, N)

    return (f_beta - f_alpha) / (x_alpha - x_beta)


# -------------------------------
# I^(4,2): 3+1 partition (triple A+P at j_trip, isolated P at j_iso)
# -------------------------------
def S_AAPP42(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_trip, j_iso, tol=1e-12):
    """
    topology: A1,A2,P1 attached at j_trip; P2 attached at j_iso (isolated)
    s-ordered triple A-block integral times backbone exponential for isolated P2.
    Uses same canonical mapping for the A-block as in S_AAPP41.
    """
    # triple A-block reduces to two-A nested integral (A positions 1,2)
    I_Ablock = S_AAPP41(k1, k2, k3, k4, bA, bP, N_A, N_P, M, tol=tol)

    # backbone prefactor D * k4^2 * |j_iso - j_trip|
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k4**2) * abs(j_iso - j_trip))

    return backbone * I_Ablock


# -------------------------------
# I^(4,3): 2 + 1 + 1 partition
# Two main variants handled in one function by specifying which pair is paired.
# -------------------------------
def S_AAPP43_pairA(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_pair, j_p1, j_p2, tol=1e-12):
    """
    Variant where the pair is the two A points (A1,A2) attached at j_pair,
    P1 attached at j_p1, P2 attached at j_p2 (singles).
    Canonical q_pair = k2 + k3 + k4
    Returns pair_nested_integral * backbone exponential connecting pair->P1 and pair->P2.
    """
    q_pair = k2 + k3 + k4
    x_pair = (bA**2 / 6.0) * (q_pair**2)
    N = N_A

    # pair nested integral (A pair)
    if np.isclose(x_pair, 0.0, atol=tol):
        pair_val = N**2 / 2.0
    else:
        # (-1 + exp(-xN) + xN) / x^2
        pair_val = (-1.0 + np.exp(-x_pair * N) + x_pair * N) / (x_pair**2)

    # backbone prefactors (minimal mapping): connect pair site to P1 and P2 using their k's
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k3**2) * abs(j_pair - j_p1) - D * (k4**2) * abs(j_pair - j_p2))

    return backbone * pair_val


def S_AAPP43_pairP(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_pairP, j_a1, j_a2, tol=1e-12):
    """
    Variant where the pair is the two P points (P1,P2) on the same backbone index j_pairP,
    and the two A points are singles attached at j_a1 and j_a2.
    The pair integral is over P contour (N_P, b_P).
    Canonical q_pairP mapping: sum of right-of-pair ks (depends on ordering) — use k3,k4 as P ks.
    Here we assume q_pairP = k3 (segment between P points), and pair decay y=(bP^2/6)*k3^2.
    """
    # nested P integral
    y1 = (bP**2 / 6.0) * (k3**2)  # canonical choice; adjust if your bookkeeping differs
    if np.isclose(y1, 0.0, atol=tol):
        pairP_val = N_P**2 / 2.0
    else:
        pairP_val = (-1.0 + np.exp(-y1 * N_P) + y1 * N_P) / (y1**2)

    # singles: A1 and A2 single integrals
    xA1 = (bA**2 / 6.0) * (k1**2)
    xA2 = (bA**2 / 6.0) * (k2**2)
    singleA1 = _f1_stable(xA1, N_A)
    singleA2 = _f1_stable(xA2, N_A)

    # backbone prefactors connecting P-pair site to each A single
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k1**2) * abs(j_pairP - j_a1) - D * (k2**2) * abs(j_pairP - j_a2))

    return backbone * pairP_val * singleA1 * singleA2

def S_AAPP43(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_pairP, j_a1, j_a2, tol=1e-12):
    return S_AAPP43_pairP(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_pairP, j_a1, j_a2, tol=1e-12)\
    +  S_AAPP43_pairA(k1, k2, k3, k4, bA, bP, N_A, N_P, M, j_pairP, j_a1, j_a2, tol=1e-12)
# -------------------------------
# I^(4,4): all four on distinct monomers (fully factorized)
# -------------------------------
def S_AAPP44(k1, k2, k3, k4, bA, bP, N_A, N_P, M, jA1, jA2, jP1, jP2, tol=1e-12):
    """
    Fully factorized case: A1 at jA1, A2 at jA2, P1 at jP1, P2 at jP2 (all distinct).
    Returns product of single-A integrals times backbone prefactors (minimal nearest-neighbor).
    Note: mapping of backbone prefactor k's to segments is minimal/nearest-neighbour and may need
    adjustment if your bookkeeping assigns different q_bb.
    """
    xA1 = (bA**2 / 6.0) * (k1**2)
    xA2 = (bA**2 / 6.0) * (k2**2)
    sA1 = _f1_stable(xA1, N_A)
    sA2 = _f1_stable(xA2, N_A)

    # minimal backbone prefactor: connect along the chain in index order jA1 -> jP1 -> jA2 -> jP2 (example)
    # YOU SHOULD ADJUST this mapping to your actual backbone order; here we use a simple nearest-neighbor
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k1**2) * abs(jP1 - jA1)
                      - D * (k3**2) * abs(jA2 - jP1)
                      - D * (k4**2) * abs(jP2 - jA2))

    return backbone * sA1 * sA2


import numpy as np

# def _f1_stable(x, L):
#     """Stable evaluation of (1 - exp(-x L)) / x for scalar x."""
#     if abs(x) < 1e-12:
#         # series to order x^2
#         return L - 0.5*x*L**2 + (x**2)*L**3/6.0
#     return (1.0 - np.exp(-x * L)) / x

def S_APPP41(kA, kP1, kP2, kP3, bA, bP, N_A, N_P, M, tol=1e-12):
    """
    Case (4,1) for S_{APPP}: all points attached to the same backbone monomer j0.
    Only the A binder contour is free.
    Inputs:
      kA, kP1, kP2, kP3 : scalar wavevector magnitudes (or scalars representing |k|).
      bA, bP            : Kuhn lengths for A- and P-chains (bP unused here but kept for API consistency).
      N_A, N_P, M       : contour lengths and polymer parameters (N_P,M unused here).
      j0                : backbone monomer index where all attach (unused but kept for API consistency).
    Returns:
      I = ∫_0^{N_A} dn exp[- (bA^2/6) * kA^2 * n ]  (computed stably)
    """
    xA = (bA**2 / 6.0) * (kA**2)
    return _f1_stable(xA, N_A)

import numpy as np

def S_APPP42(kA, kP1, kP2, kP3, bA, bP, N_A, N_P, M, jA, jP):
    """
    Case (4,2): 3 P coincide, A separate.
    Integral over A + exponential P factor.
    """
    kPsum = kP1+ kP2+ kP3
    aA = (bA**2 / 6.0) * kA**2
    deltaJP = jP - jA
    expP = np.exp(-(N_P / (6.0*(M-1))) * bP**2 * kPsum**2 * deltaJP)

    # A integral
    if np.isclose(aA, 0.0, atol=1e-14):
        FA = N_A
    else:
        FA = (1.0 - np.exp(-aA * N_A)) / aA

    return FA * expP


def S_APPP43(kA, kP1, kP2, kP3, bA, bP, N_A, N_P, M, j1, j2, j3):
    """
    Case (4,3): (A,P) pair + (P,P) pair.
    """
    aA = (bA**2 / 6.0) * kA**2
    if np.isclose(aA, 0.0, atol=1e-14):
        FA = N_A
    else:
        FA = (1.0 - np.exp(-aA * N_A)) / aA

    delJ1 = (N_P/(6.0*(M-1))) * bP**2 * kP1**2 * (j2 - j1)
    delJ2 = (N_P/(6.0*(M-1))) * bP**2 * kP2**2 * (j3 - j2)

    return FA * np.exp(-(delJ1 + delJ2))


def S_APPP44(kA, kP1, kP2, kP3, bA, bP, N_A, N_P, M, j1, j2, j3, j4):
    """
    Case (4,4): all distinct indices.
    """
    aA = (bA**2 / 6.0) * kA**2
    if np.isclose(aA, 0.0, atol=1e-14):
        FA = N_A
    else:
        FA = (1.0 - np.exp(-aA * N_A)) / aA

    delJ = (N_P/(6.0*(M-1))) * bP**2 * (
        kP1**2 * (j2 - j1) + kP2**2 * (j3 - j2) + kP3**2 * (j4 - j3)
    )
    return FA * np.exp(-delJ)


def S_AAPA42_AAPtriple_Aisolated(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_trip,j_iso,tol=1e-12):
    # two-A nested block (A1,A2 with anchored P3 inside triple)
    a_alpha = (bA**2 / 6.0) * ( (k2 + k3)**2 )
    a_beta  = (bA**2 / 6.0) * ( k3**2 )
    # use same stable routine as S_AAP31 for two-A nested integral
    def twoA_nested(a_alpha,a_beta,N):
        if np.isclose(a_alpha,a_beta,atol=tol):
            f = _f1_stable(a_alpha,N)
            return 0.5 * f * N
        f_alpha = _f1_stable(a_alpha,N)
        f_beta  = _f1_stable(a_beta,N)
        return (f_beta - f_alpha) / (a_alpha - a_beta)

    I_twoA = twoA_nested(a_alpha,a_beta,N_A)
    singleA4 = _f1_stable((bA**2 / 6.0) * (k4**2), N_A)
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k4**2) * abs(j_iso - j_trip))
    return backbone * I_twoA * singleA4


def S_AAPA43(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pair,j_p,j_a4,tol=1e-12):
    # pair integral (A1,A2)
    q_pair = k2 + k3 + k4  # canonical mapping when pair are earliest
    x_pair = (bA**2 / 6.0) * (q_pair**2)
    if np.isclose(x_pair,0.0,atol=tol):
        pair_val = N_A**2 / 2.0
    else:
        pair_val = (-1.0 + np.exp(-x_pair * N_A) + x_pair * N_A) / (x_pair**2)

    singleA4 = _f1_stable((bA**2 / 6.0) * (k4**2), N_A)

    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k3**2) * abs(j_pair - j_p) - D * (k4**2) * abs(j_pair - j_a4))

    return backbone * pair_val * singleA4


def S_APPA42(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_trip,j_iso,tol=1e-12):
    """
    triple = A1,P2,P3 at j_trip; isolated A4 at j_iso.
    - triple A factor (A1 single): x_t uses q_t = k2 + k3
    - isolated A4 single factor uses k4
    - backbone prefactor uses k4^2 * |j_iso - j_trip|
    """
    x_t = (bA**2 / 6.0) * ((k2 + k3)**2)  # q_t canonical
    single_trip = _f1_stable(x_t, N_A)

    x4 = (bA**2 / 6.0) * (k4**2)
    single4 = _f1_stable(x4, N_A)

    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k4**2) * abs(j_iso - j_trip))

    return backbone * single_trip * single4

def S_APPA43_pairP(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pairP,j_a1,j_a4,tol=1e-12):
    # nested P pair
    # choose canonical q for P-segment (e.g. qP = k3 if P3 sits right of P2)
    y = (bP**2 / 6.0) * (k3**2)
    if np.isclose(y, 0.0, atol=tol):
        pair_val = N_P**2 / 2.0
    else:
        pair_val = (-1.0 + np.exp(-y * N_P) + y * N_P) / (y**2)

    # singles A
    s1 = _f1_stable((bA**2/6.0)*(k1**2), N_A)
    s4 = _f1_stable((bA**2/6.0)*(k4**2), N_A)

    # backbone prefactors linking pair site to each A
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k1**2) * abs(j_pairP - j_a1) - D * (k4**2) * abs(j_pairP - j_a4))

    return backbone * pair_val * s1 * s4

def S_APPA43(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pairP,j_a1,j_a4,tol=1e-12):
    return S_APPA43_pairP(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pairP,j_a1,j_a4)

def S_APPA44(k1,k2,k3,k4,bA,bP,N_A,N_P,M,jA1,jP2,jP3,jA4,tol=1e-12):
    # singles (A1 and A4)
    s1 = _f1_stable((bA**2/6.0)*(k1**2), N_A)
    s4 = _f1_stable((bA**2/6.0)*(k4**2), N_A)

    # backbone prefactors connecting along the backbone (minimal mapping example)
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k1**2) * abs(jP2 - jA1)
                      - D * (k2**2) * abs(jP3 - jP2)
                      - D * (k3**2) * abs(jA4 - jP3))
    return backbone * s1 * s4


def S_PAAP42(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_trip,j_iso,tol=1e-12):
    I_block = S_AAPP41(k1,k2,k3,k4,bA,bP,N_A,N_P,M,tol=tol)
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    return I_block * np.exp(- D * (k4**2) * abs(j_iso - j_trip))

def S_PAAP43_pairA(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pair,j_p1,j_p4,tol=1e-12):
    # pair integral over A (A2,A3)
    q_pair = k3 + k4   # canonical when A2 is left of A3 and P4 to right
    x = (bA**2 / 6.0) * (q_pair**2)
    if np.isclose(x, 0.0, atol=tol):
        pair = N_A**2 / 2.0
    else:
        pair = (-1.0 + np.exp(-x * N_A) + x * N_A) / (x**2)

    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k1**2) * abs(j_pair - j_p1) - D * (k4**2) * abs(j_pair - j_p4))
    return backbone * pair

def S_PAAP43_pairP(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pairP,j_a2,j_a3,tol=1e-12):
    # nested P pair: choose canonical q for P-segment (e.g. qP=k4 if P4 to right)
    y = (bP**2 / 6.0) * (k4**2)   # adapt if your mapping differs
    if np.isclose(y, 0.0, atol=tol):
        pairP = N_P**2 / 2.0
    else:
        pairP = (-1.0 + np.exp(-y * N_P) + y * N_P) / (y**2)

    singleA2 = _f1_stable((bA**2 / 6.0) * (k2**2), N_A)
    singleA3 = _f1_stable((bA**2 / 6.0) * (k3**2), N_A)

    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    backbone = np.exp(- D * (k2**2) * abs(j_pairP - j_a2) - D * (k3**2) * abs(j_pairP - j_a3))

    return backbone * pairP * singleA2 * singleA3

def S_PAAP43(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pair,j_p1,j_p4,tol=1e-12):
    return S_PAAP43_pairA(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pair,j_p1,j_p4) \
    + S_PAAP43_pairP(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_pair,j_p1,j_p4)

def S_PAAP44(k1,k2,k3,k4,bA,bP,N_A,N_P,M,jP1,jA2,jA3,jP4,tol=1e-12):
    s2 = _f1_stable((bA**2/6.0)*(k2**2), N_A)
    s3 = _f1_stable((bA**2/6.0)*(k3**2), N_A)

    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    # example backbone mapping: jP1 -> jA2 -> jA3 -> jP4
    backbone = np.exp(- D * (k1**2) * abs(jA2 - jP1)
                      - D * (k2**2) * abs(jA3 - jA2)
                      - D * (k4**2) * abs(jP4 - jA3))
    # note: we used k2^2 for middle seg in example; adapt as per your q_bb convention
    return backbone * s2 * s3



def S_PAPP42_tripleP_isolatedA(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_trip,j_iso,tol=1e-12):
    xA = (bA**2 / 6.0) * (k2**2)
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    return _f1_stable(xA, N_A) * np.exp(- D * (k2**2) * abs(j_iso - j_trip))

def S_PAPP42_triple_with_A_isolatedP(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_trip,j_iso,tol=1e-12):
    xA = (bA**2 / 6.0) * (k2**2)
    D = (bP**2 / 6.0) * (N_P / (M - 1.0))
    return _f1_stable(xA, N_A) * np.exp(- D * (k4**2) * abs(j_iso - j_trip))

def S_PAPP42(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_trip,j_iso): 
    return S_PAPP42_tripleP_isolatedA(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_trip,j_iso) \
    + S_PAPP42_triple_with_A_isolatedP(k1,k2,k3,k4,bA,bP,N_A,N_P,M,j_trip,j_iso)
