import numpy as np
from chromo_vertex_low_mem import *

mu_max = 0 #10
mu_min = -10
del_mu = 1


n_bind = 2 #types of proteins/marks


chi = None
e_m = np.array([-1.5, -1.5]) #binding energy

v_int =  np.array([[-3.92, 3], [3, -3.92]])
# v_int = np.array([[-3.92,0],[0,-3.92]]) # protein-protein interaction param
# v_int = np.array([[0,3.92],[3.92,0]]) # protein-protein interaction param
# v_int = np.array([[0,0],[0,0]]) # protein-protein interaction param


phi_c = 0.6# avg amount of chromosome
# phi_c = 1.# avg amount of chromosome


chrom_type = "DNA"
marks_1 = np.loadtxt(r"H3K9me3_ENCFF651ZTT_Chr_22_trimmed.txt").astype(np.uint8)
marks_2 = np.loadtxt(r"H3K27me3_ENCFF470ECE_Chr_22_trimmed.txt").astype(np.uint8)


#chrom_type = "test" #"DNA"  # test if 6 nucleosomes, DNA if using joes chromosome
#marks_1 = [0,2,1,0,2,1] # marks for protein 1
#marks_2 = [1,1,1,0,2,2] # marks for protein 2

poly_marks = [marks_1, marks_2]

chrom = def_chrom(n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, chrom_type)

[n_bind, v_int, chi, e_m, phi_c, poly_marks, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

klog_min = -2.5
klog_max = -1
klog_num = 30
k_vec = np.logspace(klog_min, klog_max, klog_num) / (b)

ID = np.sum(e_m) + np.sum(v_int) + phi_c + mu_max + mu_min + del_mu + klog_min + klog_max + klog_num//5
ID = np.round(ID, 5)
np.save(r"ID=%s_settings" % ID, [chrom, [klog_min, klog_max, klog_num]])
print("saved settings file")

f_gam, s_bind = calc_binding_states(chrom)


s2_mat_shlk = calc_sf_mats(chrom, f_gam, s_bind, k_vec)


np.save(r"ID=%s_chrom_s2_mats_v_int=[[" + str(v_int[0,0]) + "," + str(v_int[0,1]) + "],["  + str(v_int[1,0]) + ","  + str(v_int[1,1]) + "]],"\
       + chrom_type + ",mu_max=" + str(mu_max) % ID, s2_mat_shlk)