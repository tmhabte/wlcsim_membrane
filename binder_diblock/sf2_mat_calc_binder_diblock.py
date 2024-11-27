import numpy as np
#from chromo_vertex_tower import *
#from chromo_vertex_nuclear_density_corrected import *
#from chromo_vertex_nuclear_competitive_2_density import *
# from ALT_F_BIND_chromo_vertex_nuclear_competitive_2_density import *
from binder_diblock_vertex_competitive import *

import time

start = time.time()

mu_max = 8#0.1 #10
mu_min = -8#-9
del_mu = .1 #0.25

# mu_max = 10
# mu_min = -10
# del_mu = 10 #0.25

klog_min = -2.5
klog_max = -1
klog_num = 30


n_bind = 2 #types of proteins/marks


chi = None
#e_m = np.array([-1.52, -1.52]) #binding energy
e_m = np.array([1.52, 1.52]) #binding energy FOR F_BIND_ALT

#v_int =  np.array([[-3.92, 3], [3, -3.92]])
v_int =  np.array([[-4, 0], [0, -4]])
#v_int =  np.array([[0,4], [4,0]]) 


rho_c = (3e7)  /  ((4/3) * np.pi*  (5)**3 * (1000/1)**3) # nucleosomes per nm^3

avo = 6.02e23
water_molmas = 18
rho_s = avo * (1/water_molmas) * (1000  *(1000/1) * (1/1e9)**3)#  num / nm^3, pure water

#phi_c = 0.4# avg amount of chromosome

# chrom_type = "DNA"
#marks_1 = np.loadtxt(r"H3K9me3_ENCFF651ZTT_Chr_22_trimmed.txt").astype(np.uint8)
#marks_2 = np.loadtxt(r"H3K27me3_ENCFF470ECE_Chr_22_trimmed.txt").astype(np.uint8)

# marks_1 = np.loadtxt(r"HNCFF683HCZ_H3K9me3_methyl.txt").astype(np.uint8)
# marks_2 = np.loadtxt(r"ENCFF919DOR_H3K27me3_methyl.txt").astype(np.uint8)

#chrom_type = "test" #"DNA"  # test if 6 nucleosomes, DNA if using joes chromosome
#marks_1 = [0,2,1,0,2,1] # marks for protein 1
#marks_2 = [1,1,1,0,2,2] # marks for protein 2

chrom_type = "diblock"

nm = 500
pa_vec = np.arange(0, nm, 1) / (nm-1)
pb_vec = 1-pa_vec

poly_marks = [pa_vec, pb_vec]


chrom = def_chrom(n_bind, v_int, e_m, rho_c, rho_s, poly_marks, mu_max, mu_min, del_mu, chrom_type)

[n_bind, v_int, Vol_int, e_m, rho_c, rho_s, poly_marks, M, mu_max, mu_min, del_mu, f_om, N, N_m, b] = chrom

k_vec = np.logspace(klog_min, klog_max, klog_num) / (b)

ID = 10000
ID += 0 + np.sum(e_m) + np.sum(v_int) + rho_c  + mu_max + mu_min + del_mu + klog_min + klog_max + klog_num//5 # competitive 2 ALT BINDING

ID = np.round(ID, 5)
np.save(r"ID=%s_settings" % ID, np.array([chrom, [klog_min, klog_max, klog_num]], dtype = "object")  )
# np.save(r"ID=%s_settings" % ID, chrom)

print("saved settings file")

s_bind_A, s_bind_B = calc_binding_states(chrom)

np.save(r"ID=%s_s_bind_A" % ID, s_bind_A)
np.save(r"ID=%s_s_bind_B" % ID, s_bind_B)

print("saved density maps")

s2_mat_shlk = calc_sf_mats(chrom, s_bind_A, s_bind_B, k_vec)


np.save(r"ID=%s_chrom_s2_mats_v_int=[[" % ID  + str(v_int[0,0]) + "," + str(v_int[0,1]) + "],["  + str(v_int[1,0]) + ","  + str(v_int[1,1]) + "]],"\
       + chrom_type + ",mu_max=" + str(mu_max), s2_mat_shlk)

print(np.round((time.time() - start)/(60*60),4), "hours elapsed")


