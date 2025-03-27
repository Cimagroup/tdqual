import numpy as np
import gudhi

import scipy.spatial.distance as dist
import itertools
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
import matplotlib as mpl

import tdqual.topological_data_quality_0 as tdqual
import time
import pickle 

d = {}
for dim in [10,20,30,40,50]:
    print("Computing dim: ", dim)
    for data_size in [1000,10000,20000,50000]:
        for subset_prop in [0.1,0.2,0.5,0.8]:
            Z=np.random.random((data_size,dim))
            X_indices = np.random.choice(len(Z), replace=False, size=int(data_size*subset_prop))
            X=Z[X_indices]
            start_time = time.time()
            # Sort indices of points in Z, so that 
            X_compl = np.ones(Z.shape[0], dtype="bool")
            X_compl[X_indices] = False
            Z = np.vstack((Z[X_indices], Z[X_compl]))
            X_indices = range(len(X_indices))
            X = Z[X_indices]
            filt_X, filt_Z, matching = tdqual.compute_Mf_0(X,Z)
            t = time.time()-start_time
            #fig, ax = plt.subplots(figsize=(7,2.5))
            #tdqual.plot_matching_0(filt_X, filt_Z, matching, ax)
            #plt.tight_layout()
            d["Dim_"+str(dim)+"_"+"size_"+str(data_size)+"_"+"Prop_"+str(subset_prop)] = t
            print("Dim:",dim," Size:",data_size,"Prop:",subset_prop," Time:",t)

with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(d, f)       