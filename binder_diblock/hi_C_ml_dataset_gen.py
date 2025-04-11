import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

product_only = True # [sigma_i*sigma_j]
product_only = False # [sigma_i*sigma_j, sigma_i, sigma_j]

bp_per_nuc = 200


resolution = 1e6 #1mb
Hi_c_raw_obs = pd.read_csv(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_1mb.RAWobserved", sep="\t", header=None)
Hi_c_raw_obs.columns = ["i","j",r"$M_{ij}$"]
Hi_c_KRnorm = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_1mb.KRnorm")
Hi_c_KRexpected = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_1mb.KRexpected")

# resolution = 500e3 #500kb
# Hi_c_raw_obs = pd.read_csv(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_500kb.RAWobserved", sep="\t", header=None)
# Hi_c_raw_obs.columns = ["i","j",r"$M_{ij}$"]
# Hi_c_KRnorm = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_500kb.KRnorm")
# Hi_c_KRexpected = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_500kb.KRexpected")

# resolution = 100e3 #100kb
# Hi_c_raw_obs = pd.read_csv(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_100kb.RAWobserved", sep="\t", header=None)
# Hi_c_raw_obs.columns = ["i","j",r"$M_{ij}$"]
# Hi_c_KRnorm = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_100kb.KRnorm")
# Hi_c_KRexpected = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_100kb.KRexpected")

# resolution = 50e3 #50kb
# Hi_c_raw_obs = pd.read_csv(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_50kb.RAWobserved", sep="\t", header=None)
# Hi_c_raw_obs.columns = ["i","j",r"$M_{ij}$"]
# Hi_c_KRnorm = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_50kb.KRnorm")
# Hi_c_KRexpected = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_50kb.KRexpected")

# resolution = 5e3 #5kb
# Hi_c_raw_obs = pd.read_csv(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_5kb.RAWobserved", sep="\t", header=None)
# Hi_c_raw_obs.columns = ["i","j",r"$M_{ij}$"]
# Hi_c_KRnorm = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_5kb.KRnorm")
# Hi_c_KRexpected = np.loadtxt(r"C:\Users\tmhab\wlcsim_membrane\config_binding\Hi-C_data_chr_16\chr16_5kb.KRexpected")

binding_resolution = resolution
nuc_per_bin = int(binding_resolution / (bp_per_nuc)) # nuc per bin.    [resolution] = bp per bin, bp_per_nuc = 200


assert(binding_resolution == resolution)


# bigwig file from quinn bottom up paper, then converted to bedgraph using bigWigToBedGraph from UCSC command line tool on WSL
h3k9me3_data = pd.read_csv('output.bedGraph', sep="\t", header=None)
h3k9me3_data.columns = ["chromosome", "start", "end", "value"]


# isolate chromosome 16
chr16_h3k9me3_data = h3k9me3_data[h3k9me3_data["chromosome"] == "chr16"] #.groupby("chromosome")
chr16_h3k9me3_data.sort_values("start", inplace=True)


# convert matrix of ranges into 1d array
# 1. Determine the size of the array
array_size = chr16_h3k9me3_data['end'].max()

# 2. Initialize the 1D array with zeros
h3k9me3_signal_bp = np.zeros(array_size)

# 3. Use numpy's vectorized approach to fill the array
# For each row, we create a range from start to end and assign the value

# Create a mask for each range and use broadcasting
starts = chr16_h3k9me3_data['start'].values
ends = chr16_h3k9me3_data['end'].values
values = chr16_h3k9me3_data['value'].values

# Create an array of indices from all ranges using np.concatenate and np.arange
indices = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])

# Repeat the values according to the length of each interval
repeated_values = np.repeat(values, ends - starts)

# Assign the values to the appropriate positions in result_array
np.add.at(h3k9me3_signal_bp, indices, repeated_values)


# integrate (sum) over nucleosomes
trimmed_size = (h3k9me3_signal_bp.size // bp_per_nuc) * bp_per_nuc
# print(trimmed_size)
reshaped_arr = h3k9me3_signal_bp[:trimmed_size].reshape(-1, bp_per_nuc)

# 2. Sum along the rows (axis 1)
h3k9me3_signal_nuc = np.sum(reshaped_arr, axis=1)

num_nucs = len(h3k9me3_signal_nuc)

one_mark_cutoff = 220
two_mark_cutoff = one_mark_cutoff*2
# print("# NO mark nucs: ", np.sum(h3k9me3_signal_nuc<one_mark_cutoff))
# print("# one mark nucs: ", np.sum(1*(h3k9me3_signal_nuc>=one_mark_cutoff) * 1*(h3k9me3_signal_nuc<two_mark_cutoff)))
# print("# two mark nucs: ", np.sum(h3k9me3_signal_nuc>=two_mark_cutoff))

marks_1 = np.zeros(num_nucs)

marks_1[np.where(h3k9me3_signal_nuc<one_mark_cutoff)] = 0
marks_1[np.where(1*(h3k9me3_signal_nuc>=one_mark_cutoff) * 1*(h3k9me3_signal_nuc<two_mark_cutoff))] = 1
marks_1[np.where(h3k9me3_signal_nuc>=two_mark_cutoff)] = 2
# np.sum(h3k9me3_signal_nuc>=two_mark_cutoff)

def eval_tmat(mu_a, mu_b, pa1 = 1, pa2 = 1, ea = 0, eb = 0, j_aa = 0, j_bb = 0, j_ab = 0, f_ref = 0):
    r"""
    eval_tmat - Evaluate the transfer matrix or the nucleosome
    T: considereing all the possible binding state combinations of the nucleosome to the left and the right
        , all the contributions to the binding state partition function
    Parameters
    ----------
    mu : float
        HP1 chemical potential
    nm1 : int
        Number of methylated tails in the left-side nucleosome
    nm2 : int
        Number of methylated tails in the right-side nucleosome
    nu : int
        Indicator for nucleosomes within the interaction length
    j_int : float
        Strength of the HP1 interactions
    
    Returns
    -------
    tmat : 3x3 float array
        Transfer matrix for the nucleosome    
    
    """
    
    v1 = np.array([1, np.sqrt(pa1) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa1) * np.exp(mu_b / 2), np.sqrt(1 - pa1) * np.exp(mu_a / 2), np.sqrt(1 - pa1) * np.exp(mu_b / 2 - eb / 2)])
    v2 = np.array([1, np.sqrt(pa2) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa2) * np.exp(mu_b / 2), np.sqrt(1 - pa2) * np.exp(mu_a / 2), np.sqrt(1 - pa2) * np.exp(mu_b / 2 - eb / 2)])
    tmat = np.outer(v1, v2) * np.exp(f_ref)
    # T: tmat is all possible combinations of (un-normalized) probability of adjacent nucleosomes binding state (boltzmann weightings)
    #   essentially joint probability matrix
    
    # Add the interaction terms
    # T: all the matrix elements with a 0 have at max one protein bound, therefore no interaction
    # T: QUESTION: tmat[1,1]] means left nucleosome and right nucleosome have A bound. why assume interaction between them then?
    # ANSWER: tmat is bbetween two nucleosomes total: left-side and right-side. propogating from left-side to right-side
    tmat[1, 1] *= np.exp(-j_aa)
    tmat[1, 2] *= np.exp(-j_ab)
    tmat[1, 3] *= np.exp(-j_aa)
    tmat[1, 4] *= np.exp(-j_ab)
    tmat[2, 1] *= np.exp(-j_ab)
    tmat[2, 2] *= np.exp(-j_bb)
    tmat[2, 3] *= np.exp(-j_ab)
    tmat[2, 4] *= np.exp(-j_bb)
    tmat[3, 1] *= np.exp(-j_aa)
    tmat[3, 2] *= np.exp(-j_ab)
    tmat[3, 3] *= np.exp(-j_aa)
    tmat[3, 4] *= np.exp(-j_ab)
    tmat[4, 1] *= np.exp(-j_ab)
    tmat[4, 2] *= np.exp(-j_bb)
    tmat[4, 3] *= np.exp(-j_ab)
    tmat[4, 4] *= np.exp(-j_bb)
                    
    return tmat

def eval_tend(mu_a, mu_b, pa = 1, ea = 0, eb = 0, f_ref = 0):
    r"""
    eval_tmat - Evaluate the transfer matrix or the nucleosome
    T: only have one interaction if nucleosome is first or last genomic position
    Parameters
    ----------
    mu : float
        HP1 chemical potential
    nm1 : int
        Number of methylated tails in the left-side nucleosome
    nm2 : int
        Number of methylated tails in the right-side nucleosome
    nu : int
        Indicator for nucleosomes within the interaction length
    j_int : float
        Strength of the HP1 interactions
    
    Returns
    -------
    tmat : 3x3 float array
        Transfer matrix for the nucleosome    
    
    """

    v1 = np.array([1, np.sqrt(pa) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa) * np.exp(mu_b / 2), np.sqrt(1 - pa) * np.exp(mu_a / 2), np.sqrt(1 - pa) * np.exp(mu_b / 2 - eb / 2)])    
    
    tend = v1 * np.exp(f_ref)
                    
    return tend

def eval_dtdmu(mu_a, mu_b, pa1 = 1, pa2 = 1, ea = 0, eb = 0, j_aa = 0, j_bb = 0, j_ab = 0, f_ref = 0):
    r"""
    eval_tmat - Evaluate the transfer matrix or the nucleosome
    
    Parameters
    ----------
    mu : float
        HP1 chemical potential
    nm1 : int
        Number of methylated tails in the left-side nucleosome
    nm2 : int
        Number of methylated tails in the right-side nucleosome
    nu : int
        Indicator for nucleosomes within the interaction length
    j_int : float
        Strength of the HP1 interactions
    
    Returns
    -------
    tmat : 3x3 float array
        Transfer matrix for the nucleosome    
    
    """

    
    v1 = np.array([1, np.sqrt(pa1) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa1) * np.exp(mu_b / 2), np.sqrt(1 - pa1) * np.exp(mu_a / 2), np.sqrt(1 - pa1) * np.exp(mu_b / 2 - eb / 2)])
    # change in transfer matrix as change mu_a, mu_b wrt left nucleosome
    dv1da = np.array([0, 0.5 * np.sqrt(pa1) * np.exp(mu_a / 2 - ea / 2), 0, 0.5 * np.sqrt(1 - pa1) * np.exp(mu_a / 2), 0])
    dv1db = np.array([0, 0, 0.5 * np.sqrt(pa1) * np.exp(mu_b / 2), 0, 0.5 * np.sqrt(1 - pa1) * np.exp(mu_b / 2 - eb / 2)])
    
    v2 = np.array([1, np.sqrt(pa2) * np.exp(mu_a / 2 - ea / 2), np.sqrt(pa2) * np.exp(mu_b / 2), np.sqrt(1 - pa2) * np.exp(mu_a / 2), np.sqrt(1 - pa2) * np.exp(mu_b / 2 - eb / 2)])
    dv2da = np.array([0, 0.5 * np.sqrt(pa2) * np.exp(mu_a / 2 - ea / 2), 0, 0.5 * np.sqrt(1 - pa2) * np.exp(mu_a / 2), 0])
    dv2db = np.array([0, 0, 0.5 * np.sqrt(pa2) * np.exp(mu_b / 2), 0, 0.5 * np.sqrt(1 - pa2) * np.exp(mu_b / 2 - eb / 2)])
    
    dtda1 = np.outer(dv1da, v2) * np.exp(f_ref)
    dtdb1 = np.outer(dv1db, v2) * np.exp(f_ref)
    dtda2 = np.outer(v1, dv2da) * np.exp(f_ref)
    dtdb2 = np.outer(v1, dv2db) * np.exp(f_ref)
    
    # Add the interaction terms
    
    dtda1[1, 1] *= np.exp(-j_aa)
    dtda1[1, 2] *= np.exp(-j_ab)
    dtda1[1, 3] *= np.exp(-j_aa)
    dtda1[1, 4] *= np.exp(-j_ab)
    dtda1[2, 1] *= np.exp(-j_ab)
    dtda1[2, 2] *= np.exp(-j_bb)
    dtda1[2, 3] *= np.exp(-j_ab)
    dtda1[2, 4] *= np.exp(-j_bb)
    dtda1[3, 1] *= np.exp(-j_aa)
    dtda1[3, 2] *= np.exp(-j_ab)
    dtda1[3, 3] *= np.exp(-j_aa)
    dtda1[3, 4] *= np.exp(-j_ab)
    dtda1[4, 1] *= np.exp(-j_ab)
    dtda1[4, 2] *= np.exp(-j_bb)
    dtda1[4, 3] *= np.exp(-j_ab)
    dtda1[4, 4] *= np.exp(-j_bb)
    
    dtdb1[1, 1] *= np.exp(-j_aa)
    dtdb1[1, 2] *= np.exp(-j_ab)
    dtdb1[1, 3] *= np.exp(-j_aa)
    dtdb1[1, 4] *= np.exp(-j_ab)
    dtdb1[2, 1] *= np.exp(-j_ab)
    dtdb1[2, 2] *= np.exp(-j_bb)
    dtdb1[2, 3] *= np.exp(-j_ab)
    dtdb1[2, 4] *= np.exp(-j_bb)
    dtdb1[3, 1] *= np.exp(-j_aa)
    dtdb1[3, 2] *= np.exp(-j_ab)
    dtdb1[3, 3] *= np.exp(-j_aa)
    dtdb1[3, 4] *= np.exp(-j_ab)
    dtdb1[4, 1] *= np.exp(-j_ab)
    dtdb1[4, 2] *= np.exp(-j_bb)
    dtdb1[4, 3] *= np.exp(-j_ab)
    dtdb1[4, 4] *= np.exp(-j_bb)
                    
    dtda2[1, 1] *= np.exp(-j_aa)
    dtda2[1, 2] *= np.exp(-j_ab)
    dtda2[1, 3] *= np.exp(-j_aa)
    dtda2[1, 4] *= np.exp(-j_ab)
    dtda2[2, 1] *= np.exp(-j_ab)
    dtda2[2, 2] *= np.exp(-j_bb)
    dtda2[2, 3] *= np.exp(-j_ab)
    dtda2[2, 4] *= np.exp(-j_bb)
    dtda2[3, 1] *= np.exp(-j_aa)
    dtda2[3, 2] *= np.exp(-j_ab)
    dtda2[3, 3] *= np.exp(-j_aa)
    dtda2[3, 4] *= np.exp(-j_ab)
    dtda2[4, 1] *= np.exp(-j_ab)
    dtda2[4, 2] *= np.exp(-j_bb)
    dtda2[4, 3] *= np.exp(-j_ab)
    dtda2[4, 4] *= np.exp(-j_bb)
    
    dtdb2[1, 1] *= np.exp(-j_aa)
    dtdb2[1, 2] *= np.exp(-j_ab)
    dtdb2[1, 3] *= np.exp(-j_aa)
    dtdb2[1, 4] *= np.exp(-j_ab)
    dtdb2[2, 1] *= np.exp(-j_ab)
    dtdb2[2, 2] *= np.exp(-j_bb)
    dtdb2[2, 3] *= np.exp(-j_ab)
    dtdb2[2, 4] *= np.exp(-j_bb)
    dtdb2[3, 1] *= np.exp(-j_aa)
    dtdb2[3, 2] *= np.exp(-j_ab)
    dtdb2[3, 3] *= np.exp(-j_aa)
    dtdb2[3, 4] *= np.exp(-j_ab)
    dtdb2[4, 1] *= np.exp(-j_ab)
    dtdb2[4, 2] *= np.exp(-j_bb)
    dtdb2[4, 3] *= np.exp(-j_ab)
    dtdb2[4, 4] *= np.exp(-j_bb)
        
    return dtda1, dtda2, dtdb1, dtdb2
    
def eval_dtenddmu(mu_a, mu_b, pa = 1, ea = 0, eb = 0, f_ref = 0):
    r"""
    eval_tmat - Evaluate the transfer matrix or the nucleosome
    
    Parameters
    ----------
    mu : float
        HP1 chemical potential
    nm1 : int
        Number of methylated tails in the left-side nucleosome
    nm2 : int
        Number of methylated tails in the right-side nucleosome
    nu : int
        Indicator for nucleosomes within the interaction length
    j_int : float
        Strength of the HP1 interactions
    
    Returns
    -------
    tmat : 3x3 float array
        Transfer matrix for the nucleosome    
    
    """

    
    dv1da = np.array([0, 0.5 * np.sqrt(pa) * np.exp(mu_a / 2 - ea / 2), 0, 0.5 * np.sqrt(1 - pa) * np.exp(mu_a / 2), 0])
    dv1db = np.array([0, 0, 0.5 * np.sqrt(pa) * np.exp(mu_b / 2), 0, 0.5 * np.sqrt(1 - pa) * np.exp(mu_b / 2 - eb / 2)])
    
    dtendda = dv1da * np.exp(f_ref)
    dtenddb = dv1db * np.exp(f_ref)
                    
    return dtendda, dtenddb
    
def eval_phi(pa_vec, mu_a = 0, mu_b = 0, ea = 0, eb = 0, j_aa = 0, j_bb = 0, j_ab = 0, f_ref = 0):
    
    nm = len(pa_vec)
    phiva = np.zeros((nm, 5))
    phivb = np.zeros((nm, 5))
    phia = np.zeros((nm))
    phib = np.zeros((nm))
    
    # Evaluate binding for the first bead
    
    pa2 = pa_vec[0]
    tend = eval_tend(mu_a, mu_b, pa2, ea, eb, f_ref)
    dtendda, dtenddb = eval_dtenddmu(mu_a, mu_b, pa2, ea, eb, f_ref)

    q_vec = tend
    phiva[0, :] = dtendda
    phivb[0, :] = dtenddb
    for j in range(1, nm):
        phiva[j, :] = tend
        phivb[j, :] = tend
    
    # Evaluate binding for the intermediate beads
    
    for i in range(0, nm - 1):

        # update mark probabilty of left and right nucleosome
        pa1 = pa2
        pa2 = pa_vec[i + 1]
        
        tmat = eval_tmat(mu_a, mu_b, pa1, pa2, ea, eb, j_aa, j_bb, j_ab, f_ref)
        dtda1, dtda2, dtdb1, dtdb2 = eval_dtdmu(mu_a, mu_b, pa1, pa2, ea, eb, j_aa, j_bb, j_ab, f_ref)
        
        q_vec = np.matmul(q_vec, tmat)

        for j in range(0, nm):
            if j == i:
                phiva[j, :] = np.matmul(phiva[j, :], tmat) + np.matmul(phiva[i + 1, :], dtda1)
                phivb[j, :] = np.matmul(phivb[j, :], tmat) + np.matmul(phivb[i + 1, :], dtdb1)
            elif j == (i + 1):
                # only condiser neighbor in one direction- whole point of transfer matrix method
                phiva[j, :] = np.matmul(phiva[j, :], dtda2)
                phivb[j, :] = np.matmul(phivb[j, :], dtdb2)
            else:
                phiva[j, :] = np.matmul(phiva[j, :], tmat)
                phivb[j, :] = np.matmul(phivb[j, :], tmat)
    
    # Evaluate binding for the last bead

    pa1 = pa2
    tend = eval_tend(mu_a, mu_b, pa1, ea, eb, f_ref)
    dtendda, dtenddb = eval_dtenddmu(mu_a, mu_b, pa1, ea, eb, f_ref)

    # calculate average binding fractions
    q = np.matmul(q_vec, tend) #part func
    print(q)
    phia[nm - 1] = (np.matmul(q_vec, dtendda) + np.matmul(phiva[nm - 1, :], tend)) / q
    phib[nm - 1] = (np.matmul(q_vec, dtenddb) + np.matmul(phivb[nm - 1, :], tend)) / q
    for j in range(0, nm - 1):
        phia[j] = np.matmul(phiva[j, :], tend) / q
        phib[j] = np.matmul(phivb[j, :], tend) / q
    
    return phia, phib
    

def calc_binding_states(chrom):
    # evaluate average binding state for each nucleosome at each mu1,mu2
    
    [n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

    [pa_vec, marks_2] = poly_marks

    mu1_array = np.arange(mu_min, mu_max, del_mu)#[-5]
    mu2_array = np.arange(mu_min, mu_max, del_mu)#[-5]

    s_bind_1_soln_arr = np.zeros((len(mu1_array), len(mu2_array), M))
    s_bind_2_soln_arr = np.zeros((len(mu1_array), len(mu2_array), M))
    
    f_ref = np.min(np.array([v_int[0,0], v_int[1,1], v_int[0,1], e_m[0] / 2,  e_m[1] / 2]))

    for i, mu1 in enumerate(mu1_array):
        for j, mu2 in enumerate(mu2_array):
            s_bind_ave_a, s_bind_ave_b = eval_phi(pa_vec, mu1, mu2, e_m[0], e_m[1], v_int[0,0], v_int[1,1], v_int[0,1], f_ref)
            s_bind_1_soln_arr[i,j,:] = s_bind_ave_a
            s_bind_2_soln_arr[i,j,:] = s_bind_ave_b
    
    return s_bind_1_soln_arr, s_bind_2_soln_arr


# AVERAGED
mlp_data_mark_1 = marks_1[:(len(marks_1)//nuc_per_bin)*nuc_per_bin].reshape(-1,nuc_per_bin)#.astype(sigma_dt)
mark_1_avgd = np.mean(mlp_data_mark_1, axis=1)

sig = mark_1_avgd/2 # should be from 0 to 1; percent marked#[:10]
mu_hp1 = -5
mu_prc1 = -1000
ea = -1.52
eb = 0#-1.52

j_aa = -4
j_bb = 0
j_ab = 0

f_ref = 0#np.min(np.array([j_aa, j_bb, j_ab, ea, eb]))
s_bind_hp1_avgd, phib = eval_phi(sig, mu_hp1, mu_prc1, ea, eb, j_aa, j_bb, j_ab, f_ref)

Hi_c_Kr_obs = Hi_c_raw_obs[r"$M_{ij}$"].values / (Hi_c_KRnorm[(Hi_c_raw_obs["i"].values // resolution).astype(int)]*Hi_c_KRnorm[(Hi_c_raw_obs["j"].values // resolution).astype(int)])
Hi_c_raw_obs[r"$M_{ij}^{KR}$"] = Hi_c_Kr_obs

# observed/expected correction
i_j_diff = Hi_c_raw_obs["i"].values - Hi_c_raw_obs["j"].values
Kr_indices = (i_j_diff // resolution).astype(int)
Hi_c_Kr_OE = Hi_c_Kr_obs / Hi_c_KRexpected[Kr_indices]
Hi_c_raw_obs[r"$(O/E)^{KR}$"] = Hi_c_Kr_OE

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron 
import itertools

assert(binding_resolution == resolution)
nuc_per_bin = int(resolution / (bp_per_nuc)) # nuc per bin.    [resolution] = bp per bin, bp_per_nuc = 200

# generate copy of Hi-c data where genomic position turned into {bin index precursor}
Hi_c_raw_obs_div = Hi_c_raw_obs.copy()
Hi_c_raw_obs_div = Hi_c_raw_obs_div.dropna()
Hi_c_raw_obs_div["i"] = (Hi_c_raw_obs_div["i"].values//resolution).astype(int)
Hi_c_raw_obs_div["j"] = (Hi_c_raw_obs_div["j"].values//resolution).astype(int)

#cutoff last bin so that mark data and hi-c data are same size
cutoff_bin = len(marks_1)//nuc_per_bin
Hi_c_raw_obs_div = Hi_c_raw_obs_div[Hi_c_raw_obs_div["i"] != cutoff_bin]
Hi_c_raw_obs_div = Hi_c_raw_obs_div[Hi_c_raw_obs_div["j"] != cutoff_bin]

i_combs = Hi_c_raw_obs_div["i"].values
j_combs = Hi_c_raw_obs_div["j"].values
Hi_C = Hi_c_raw_obs_div[r"$(O/E)^{KR}$"].values

#Turn Hi_C data into binary classification; 1 = contact, 0 = no
s_ij = np.zeros(len(Hi_C))
s_ij[Hi_C > np.mean(Hi_C)] = 1 # Hi_C into binary classification, based on MEDIAN

# # first divide mark data into bins
# mlp_data_mark_1 = marks_1[:(len(marks_1)//nuc_per_bin)*nuc_per_bin].reshape(-1,nuc_per_bin)#.astype(sigma_dt)

# # get average mark signal for each bin
# sig_i_avgd_1 = np.mean(sig_i, axis=1)
# sig_j_avgd_1 = np.mean(sig_j, axis=1)

num_marks = 1

sig_mark_i = mark_1_avgd[i_combs]
sig_mark_j = mark_1_avgd[j_combs]

sig_bind_i = s_bind_hp1_avgd[i_combs] 
sig_bind_j = s_bind_hp1_avgd[j_combs]

N_y = 100
# vector = "horizontal"
vector = "diagonal"
df = Hi_c_raw_obs_div

# (partially) VECTORIZED
if product_only:

    num_bins = len(mlp_data_mark_1)
    n_dp = (num_bins - (N_y*2) + 1) # number of valid datapoints
    set_size = (N_y*2 + 1)
    dp_dim = set_size**2 # length of each datapoint vector
    x_arr_bind = np.zeros((n_dp, dp_dim))
    x_arr_mark = np.zeros((n_dp, dp_dim))
    y_arr_vect = np.zeros((n_dp, N_y + 1))

    adjust = 0 
    for k in range(num_bins):
        if (k - N_y <0) or (k + N_y > num_bins): # ensure at valid diag point
            adjust += 1
            continue
        k_valid = k - adjust
        
        # X VAL- mark/bind products
        nuc_set = np.arange(k_valid-N_y, k_valid+N_y+1) # all nucleosome indices to consider
        
        sig_inds = np.array([p for p in itertools.product(nuc_set, repeat=2)]) # all possible pair permutation
        i_inds = sig_inds[:,0]
        j_inds = sig_inds[:,1]
        
        D_i = mark_1_avgd[i_inds]
        D_j = mark_1_avgd[j_inds]
        D_mark = D_i * D_j
        x_arr_mark[k_valid, :] = D_mark
        
        D_i = s_bind_hp1_avgd[i_inds]
        D_j = s_bind_hp1_avgd[j_inds]
        D_bind = D_i * D_j # x
        x_arr_bind[k_valid, :] = D_bind

        if vector == "diagonal":
            y_indices = np.arange(N_y + 1)
            y_i = k_valid + y_indices
            y_j = k_valid - y_indices
            
            # Use Pandas indexing to get values for both cases at once
            y_vals = df.set_index(['i', 'j'])[r"$(O/E)^{KR}$"]
            
            # Try to retrieve values for (y_i, y_j)
            y = y_vals.reindex(zip(y_i, y_j)).values
            
            # If values are missing (NaN), use (y_j, y_i)
            mask = np.isnan(y)
            y[mask] = y_vals.reindex(zip(y_j[mask], y_i[mask])).values
        
        if vector == "horizontal":
            y_indices = np.arange(N_y + 1)
            y_i = k_valid + np.zeros_like(y_indices)
            y_j = k_valid + y_indices
            
            # Use Pandas indexing to get values for both cases at once
            y_vals = df.set_index(['i', 'j'])[r"$(O/E)^{KR}$"]
            
            # Retrieve values for (y_i, y_j)
            y = y_vals.reindex(zip(y_i, y_j)).values
            # print(y)
            # print(y_arr[k_valid,:])
        
        y_arr_vect[k_valid,:] = y    


    # REMOVE NANS
    y_arr_vect_clean = y_arr_vect[~np.isnan(y_arr_vect).any(axis=1)]
    x_arr_mark_clean = x_arr_mark[~np.isnan(y_arr_vect).any(axis=1)]
    x_arr_bind_clean = x_arr_bind[~np.isnan(y_arr_vect).any(axis=1)]

    np.save("y_arr_Ny=%s_res=%s_product_only=%" % (N_y, resolution, product_only), y_arr_vect_clean)
    np.save("x_arr_mark_Ny=%s_res=%s_product_only=%" % (N_y, resolution, product_only), x_arr_mark_clean)
    np.save("x_arr_bind_Ny=%s_res=%s_product_only=%" % (N_y, resolution, product_only), x_arr_bind_clean)

else:
    num_bins = len(mlp_data_mark_1)
    n_dp = (num_bins - (N_y*2) + 1) # number of valid datapoints
    set_size = (N_y*2 + 1)
    dp_dim = set_size**2 # length of each datapoint vector
    x_arr_bind = np.zeros((n_dp, dp_dim,3)) # 3 becasue [sigma_i*sigma_j, sigma_i, sigma_j]
    x_arr_mark = np.zeros((n_dp, dp_dim,3))
    y_arr_vect = np.zeros((n_dp, N_y + 1))

    adjust = 0 
    for k in range(num_bins):
        if (k - N_y <0) or (k + N_y > num_bins): # ensure at valid diag point
            adjust += 1
            continue
        k_valid = k - adjust
        
        # X VAL- mark/bind products
        nuc_set = np.arange(k_valid-N_y, k_valid+N_y+1) # all nucleosome indices to consider
        
        sig_inds = np.array([p for p in itertools.product(nuc_set, repeat=2)]) # all possible pair permutation
        i_inds = sig_inds[:,0]
        j_inds = sig_inds[:,1]
        
        D_i = mark_1_avgd[i_inds]
        D_j = mark_1_avgd[j_inds]
        D_mark = D_i * D_j
        x_arr_mark[k_valid, :, 0] = D_mark
        x_arr_mark[k_valid, :, 1] = D_i
        x_arr_mark[k_valid, :, 2] = D_j
        
        D_i = s_bind_hp1_avgd[i_inds]
        D_j = s_bind_hp1_avgd[j_inds]
        D_bind = D_i * D_j # x
        x_arr_bind[k_valid, :, 0] = D_bind
        x_arr_bind[k_valid, :, 1] = D_i
        x_arr_bind[k_valid, :, 2] = D_j

        if vector == "diagonal":
            y_indices = np.arange(N_y + 1)
            y_i = k_valid + y_indices
            y_j = k_valid - y_indices
            
            # Use Pandas indexing to get values for both cases at once
            y_vals = df.set_index(['i', 'j'])[r"$(O/E)^{KR}$"]
            
            # Try to retrieve values for (y_i, y_j)
            y = y_vals.reindex(zip(y_i, y_j)).values
            
            # If values are missing (NaN), use (y_j, y_i)
            mask = np.isnan(y)
            y[mask] = y_vals.reindex(zip(y_j[mask], y_i[mask])).values
        
        if vector == "horizontal":
            y_indices = np.arange(N_y + 1)
            y_i = k_valid + np.zeros_like(y_indices)
            y_j = k_valid + y_indices
            
            # Use Pandas indexing to get values for both cases at once
            y_vals = df.set_index(['i', 'j'])[r"$(O/E)^{KR}$"]
            
            # Retrieve values for (y_i, y_j)
            y = y_vals.reindex(zip(y_i, y_j)).values
            # print(y)
            # print(y_arr[k_valid,:])

        y_arr_vect[k_valid,:] = y    

    # REMOVE NANS
    y_arr_vect_clean = y_arr_vect[~np.isnan(y_arr_vect).any(axis=1)]
    x_arr_mark_clean = x_arr_mark[~np.isnan(y_arr_vect).any(axis=1)]
    x_arr_bind_clean = x_arr_bind[~np.isnan(y_arr_vect).any(axis=1)]

    np.save("y_arr_Ny=%s_res=%s_product_only=%" % (N_y, resolution, product_only), y_arr_vect_clean)
    np.save("x_arr_mark_Ny=%s_res=%s_product_only=%" % (N_y, resolution, product_only), x_arr_mark_clean)
    np.save("x_arr_bind_Ny=%s_res=%s_product_only=%" % (N_y, resolution, product_only), x_arr_bind_clean)