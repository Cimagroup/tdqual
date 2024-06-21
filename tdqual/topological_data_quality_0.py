import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.spatial.distance as dist
from scipy.sparse.csgraph import minimum_spanning_tree

def compute_Mf_0(X, Z, indices_X):
    filtration_list_X, pairs_arr_X = mst_edge_filtration(X) # MST(X)
    filtration_list_Z, pairs_arr_Z = mst_edge_filtration(Z) # MST(Z)
    TMT_X_pairs = compute_tmt_pairs(filtration_list_X, pairs_arr_X)
    TMT_Z_pairs = compute_tmt_pairs(filtration_list_Z, pairs_arr_Z)
    indices_X_Z = np.max(TMT_Z_pairs, axis=1)<X.shape[0]
    TMT_X_Z_pairs = TMT_Z_pairs[indices_X_Z]
    indices_X_Z = np.nonzero(indices_X_Z)[0]
    FX = get_inclusion_matrix(TMT_X_pairs, TMT_X_Z_pairs) # Associated matrix
    matchingX = get_inclusion_matrix_pivots(FX, Z.shape[0]) # Matching in TMT_X_Z
    matching =[indices_X_Z[i] for i in matchingX] # Matching in all TMT_Z
    return filtration_list_X, filtration_list_Z, matching

def read_csr_matrix(cs_matrix):
    """Function to read output from minimum_spanning_tree and prepare it as a list of 
    filtration values (in order) together with an array with the corresponding pairs.
    """ 
    filtration_list = []
    edges = []
    entry_idx = 0
    for i, cummul_num_entries in enumerate(cs_matrix.indptr[1:]):
        while entry_idx < cummul_num_entries:
            edges.append((i, cs_matrix.indices[entry_idx]))
            filtration_list.append(cs_matrix.data[entry_idx])
            entry_idx+=1
    # Sort filtration values and pairs
    edges_arr = np.array(edges)
    np.argsort(filtration_list)
    sort_idx = np.argsort(filtration_list)
    filtration_list = np.array(filtration_list)[sort_idx].tolist()
    edges_arr = edges_arr[sort_idx]
    return filtration_list, edges_arr

def mst_edge_filtration(points):
    """Returns the edges and filtration values for the Euclidean minimum spanning tree
    of a given point sample. 
    This is a wrapper for the scipy minimum_spanning_tree function.
    """ 
    mst = minimum_spanning_tree(dist.squareform(dist.pdist(points)))
    # We now read the compressed sparse row matrix
    return read_csr_matrix(mst)

def compute_tmt_pairs(filtration_list, edges_arr, tolerance=10e-8):
    # Get proper merge tree pairs 
    E_b = []
    C = np.array(list(range(edges_arr.shape[0]+1)))
    tmt_pairs_list = []
    for i, (b, edge) in enumerate(zip(filtration_list, edges_arr)):
        E_b.append(edge)
        # If the next filtration value is very close, we continue addin edges to E_b
        if i < len(filtration_list)-1:
            if np.abs(b - filtration_list[i+1]) < tolerance:
                continue
        
        # We iterate over E_b, adding triplets to tmt_pairs_list following steps (i)-(v) from the article
        while(len(E_b)>0):
            # (i) we take the edge [i,j] from E_b such that min{C[i],C[j]} is smallest
            E_b_C_min = [np.min(C[edge]) for edge in E_b]
            idx = np.argmin(E_b_C_min)
            edge = E_b[idx]
            # (ii) 
            M, m = np.max(C[edge]), np.min(C[edge])
            assert m < M
            # (iii) 
            tmt_pairs_list.append([M,m])
            # (iv)
            C[C==M]=m
            # (v)
            del(E_b[idx])
        # end when E_b is empty
    # end computing tmt
    tmt_pairs_arr = np.array(tmt_pairs_list)
    return tmt_pairs_arr


def add_columns_mod_2(col1, col2):
    """ Given two lists of integers, which are sparse representations of a pair of vectors in Z mod 2, this funciton adds them and 
    returns the result in the same input format.
    """
    diff_1 = set(col1).difference(set(col2))
    diff_2 = set(col2).difference(set(col1))
    result = diff_1.union(diff_2)
    return list(result)


def get_inclusion_matrix(pairs_arr_X, pairs_arr_Z, subset_indices=[]):
    """ Given two pairs of arrays with the vertex merge pairs, this function returns the associated inclusion matrix. 
    From the point of view of minimum spanning trees, the output matrix columns can be interpreted as the minimum paths that are needed to 
    go through in MST(Z) in order to connect the endpoints from an edge in MST(X)
    """
    # If subset indices are not specified, we assume that the indices of vertices from S correspond to the first #X vertices from Z
    if (len(subset_indices)==0):
        subset_indices = list(range(pairs_arr_X.shape[0]+1))
    pivot2column = [-1] + np.argsort(np.max(pairs_arr_Z, axis=1)).tolist()
    inclusion_matrix = []
    for col_X in pairs_arr_X:
        col_X = [subset_indices[i] for i in col_X]
        col_M = []
        while(len(col_X)>0):
            piv = np.max(col_X)
            col_M.append(pivot2column[piv])
            col_X = add_columns_mod_2(col_X, pairs_arr_Z[pivot2column[piv]])
        # end reducing column X
        col_M.sort()
        inclusion_matrix.append(col_M)
    return inclusion_matrix

def get_inclusion_matrix_pivots(matrix_list, num_rows):
    """ Returns the pivots of a matrix given in list format"""
    pivots = []
    pivot2column = np.ones(num_rows, dtype="int")*-1
    for i, column in enumerate(matrix_list):
        reduce_column = list(column)
        piv = np.max(reduce_column)
        while(pivot2column[piv]>-1):
            reduce_column = add_columns_mod_2(reduce_column, matrix_list[pivot2column[piv]])
            piv = np.max(reduce_column)
            # we assume that columns are never reduced to the 0 column
        pivots.append(piv)
        pivot2column[piv] = i
    # end getting pivots
    return pivots  


def plot_matching_0(filt_X, filt_Z, matching, ax):
    """ Given two zero dimensional barcodes as well as a block function between them, this function plots the associated diagram"""
    # Plot matching barcode
    for i, Z_end in enumerate(filt_Z):
        if i in matching:
            X_end = filt_X[matching.index(i)]
            ax.add_patch(mpl.patches.Rectangle([0, i-0.2], Z_end, 0.4, color="navy", zorder=2))
            ax.add_patch(mpl.patches.Rectangle([Z_end*0.9, i-0.2], X_end-Z_end, 0.4, color="orange", zorder=1.9))
        else:
            ax.add_patch(mpl.patches.Rectangle([0, i-0.2], Z_end, 0.4, color="aquamarine", zorder=2))

    MAX_PLOT_RAD = max(np.max(filt_X), np.max(filt_Z))*1.1
    ax.set_xlim([-0.1*MAX_PLOT_RAD, MAX_PLOT_RAD*1.1])
    ax.set_ylim([-0.1*len(filt_X), len(filt_Z)])
    ax.set_frame_on(False)
    ax.set_yticks([])
#end plot_matching_0

def plot_density_matrix(filt_X, filt_Z, matching, ax, nbins=5):
    endpoints_Z = np.array(filt_Z)[matching]
    differences = np.array([[a, b] for (a,b) in zip(filt_X, endpoints_Z)])
    ax.hist2d(differences[:,0], differences[:,1], bins=(nbins, nbins), cmap=plt.cm.jet)
    ax.set_xlabel('Ends in X')
    ax.set_ylabel('Ends in Z')

def plot_density_matrix_percentage(filt_X, filt_Z, matching, ax, nbins=5):
    ends_X = np.array(filt_X)
    max_end = np.max(ends_X)
    ends_diff = ends_X - np.array(filt_Z)[matching]
    Diag_diff = np.vstack((ends_X, ends_diff)).transpose()
    hist = np.histogram2d(Diag_diff[:,1], Diag_diff[:,0], bins=nbins, range=[[0,max_end], [0,max_end]])[0]
    sum_cols = np.sum(hist, axis=0)
    sum_cols = np.maximum(sum_cols, 1)
    hist = np.divide(hist, sum_cols)
    hist=hist[-1::-1]
    ax.imshow(hist, extent=(0, max_end, 0, max_end))
    ax.set_xlabel('Differences of the bars (percent)')
    ax.set_ylabel('Length of the bars Z')

def compute_matching_diagram(filt_X, filt_Z, matching, _tol=1e-5):
    pairs = []
    for i, a in enumerate(filt_X):
        b = filt_Z[matching[i]]
        pairs.append((a, b))
    # end for 
    multiplicities = [] 
    pairs_copy = list(pairs)
    old_pair = pairs_copy.pop()
    multiplicities.append(1)
    # First compute matched pairs and their multiplicities
    while len(pairs_copy)>0:
        pair = pairs_copy.pop()
        if np.all(np.abs(np.array(pair)-np.array(old_pair)) < _tol):
            multiplicities[-1]+=1
        else:
            multiplicities.append(1)
            old_pair = pair
        # end checking if pair similar
    # end while
    # Next compute right infinity points and their multiplicities 
    unmatched_idx = [j for j in range(len(filt_Z)) if j not in matching]
    if len(unmatched_idx)>0:
        b = filt_Z[unmatched_idx.pop()]
        pairs.append((np.inf, b))
        multiplicities.append(1)
        while len(unmatched_idx)>0:
            prev_b = b
            b = filt_Z[unmatched_idx.pop()]
            if (np.abs(b-prev_b)<_tol):
                multiplicities[-1]+=1
            else:
                pairs.append((np.inf, b))
                multiplicities.append(1)
            # end checking same endpoint
        # iterate over unmatched intervals
    # only if some intervals are unmatched
    return np.array(pairs), multiplicities


def plot_matching_diagram(pairs, ax, colorpt="black", marker="o", size=15, hmax=None):
    fin_pairs = pairs[pairs[:,0]<np.inf]
    if hmax is None:
        max_x = max(np.max(fin_pairs), np.max(pairs[:,1]))
        lim_x = max_x*1.3
        infty_x = max_x*1.1
    else:
        max_x, lim_x, infty_x = hmax, hmax*1.3, hmax*1.1
    # end if-else loop
    ax.plot([0, lim_x], [0, lim_x], c="gray", linewidth=1, zorder=1)
    ax.plot([infty_x, infty_x], [0, lim_x], c="blue", linewidth=1, zorder=1)
    ax.scatter(fin_pairs[:,0], fin_pairs[:,1], color=colorpt, s=size, marker=marker, zorder=2)
    inf_points = pairs[pairs[:,0]==np.inf]
    ax.scatter(np.ones(len(inf_points))*infty_x, inf_points[:,1], color=colorpt, s=size, marker=marker, zorder=2)
    ax.text(infty_x*1.02, infty_x*1.1, "∞", fontsize=15, color="blue")
    ax.scatter([infty_x], [infty_x], color=colorpt, s=size, marker=marker, zorder=2)
    # Adjust spines 
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_position("zero")
    # adjust margins 
    ax.set_xlim([0,lim_x])
    ax.set_ylim([0,lim_x])

def plot_density_matching_diagram(D_f_rep, coker_f, savepath, nbins=5, cmap="jet", show_colorbar=False):
    ### First, create figure with two axes usin gridspec 
    gs = mpl.gridspec.GridSpec(1,2, width_ratios=[nbins,1])
    fig = plt.figure(figsize=(4,4))
    ax = [fig.add_subplot(gs[0,i]) for i in range(2)]
    ### Next, let us read the information from the matching diagram and plot it
    fin_D_f = D_f_rep[D_f_rep[:,0]<np.inf]
    lim_x = 1.2*max(np.max(fin_D_f), np.max(D_f_rep[:,1]))
    hist_fin_D_f = np.histogram2d(fin_D_f[:,1], fin_D_f[:,0], bins=nbins, range=[[0,lim_x], [0,lim_x]])[0]
    hist_coker_f = np.histogram2d(np.zeros(len(coker_f)), coker_f, bins=[1,nbins], range=[[0,lim_x*0.1], [0,lim_x]])[0]
    vmax = max(np.max(hist_fin_D_f), np.max(hist_coker_f))
    ax[0].imshow(hist_fin_D_f, cmap=cmap, vmin=0, vmax=vmax, origin="lower", extent=(0,lim_x,0,lim_x))
    image = ax[1].imshow(hist_coker_f.transpose(), cmap=cmap, vmin=0, vmax=vmax, origin="lower", extent=(0,lim_x/nbins,0,lim_x))
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    # Draw labels
    ax[0].set_xlabel('Ends in X')
    ax[0].set_ylabel('Ends in Z')
    ax[1].set_ylabel('Coker(f)')
    # Colorbar
    if show_colorbar:
        plt.colorbar(mappable=image, ax=ax, orientation="horizontal", location="top")
    ### Save figure
    plt.savefig(savepath)

def plot_matching_0(filt_X, filt_Z, matching, ax):
    """ Given two zero dimensional barcode endpoint lists as well as a block function between them, this function plots the associated diagram"""
    # Plot matching barcode
    for i, Z_end in enumerate(filt_Z):
        if i in matching:
            X_end = filt_X[matching.index(i)]
            ax.add_patch(mpl.patches.Rectangle([0, i-0.2], Z_end, 0.4, color="navy", zorder=2))
            ax.add_patch(mpl.patches.Rectangle([Z_end*0.9, i-0.2], X_end-Z_end, 0.4, color="orange", zorder=1.9))
        else:
            ax.add_patch(mpl.patches.Rectangle([0, i-0.2], Z_end, 0.4, color="aquamarine", zorder=2))

    MAX_PLOT_RAD = max(np.max(filt_X), np.max(filt_Z))*1.1
    ax.set_xlim([-0.1*MAX_PLOT_RAD, MAX_PLOT_RAD*1.1])
    ax.set_ylim([-0.1*len(filt_Z), len(filt_Z)])
    ax.set_frame_on(False)
    ax.set_yticks([])
#end plot_matching_0

### Random Circle Creation 
def sampled_circle(r, R, n, RandGen):
    assert r<=R
    radii = RandGen.uniform(r,R,n)
    angles = RandGen.uniform(0,2*np.pi,n)
    return np.vstack((np.cos(angles)*radii, np.sin(angles)*radii)).transpose()
