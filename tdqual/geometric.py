import numpy as np
import matplotlib as mpl

import itertools

import scipy.spatial.distance as dist

# We consider functions computing the block function in dimension zero and 
# relating it to cycles
def compute_components(edgelist, num_points):
    components = np.array(range(num_points))
    for edge in edgelist:
        max_idx = np.max(components[edge])
        min_idx = np.min(components[edge])
        indices = np.nonzero(components == components[max_idx])[0]
        components[indices]=np.ones(len(indices))*components[min_idx]
    
    return components

def plot_geometric_matching(a, b, idx_X, Z, filt_X, filt_Z, edges_X, edges_Z, ax, _tol=1e-5, labelsize=10):
    X = Z[idx_X]
    # Obtain indices of bars that are approximately equal to a and b, these go from (a_idx - a_shift) to a_idx. (same for b_idx)
    a_idx = np.max(np.nonzero(np.array(filt_X) < a + _tol))
    a_shift = np.sum(np.array(filt_X)[:a_idx+1] > a - _tol)
    b_idx = np.max(np.nonzero(np.array(filt_Z) < b + _tol))
    b_shift = np.sum(np.array(filt_Z)[:b_idx+1] > b - _tol)
    pair_ab = [a_idx, b_idx]
    shift_ab = [a_shift, b_shift]
    num_points = Z.shape[0]
    for idx in range(3):
        ax[idx].scatter(X[:,0], X[:,1], color=mpl.colormaps["RdBu"](0.3/1.3), s=60, marker="o", zorder=2)
        ax[idx].scatter(Z[:,0], Z[:,1], color=mpl.colormaps["RdBu"](1/1.3), s=40, marker="x", zorder=1)
        # Plot edges that came before a, b
        bool_smaller = dist.pdist(X)<=a-_tol
        edgelist = np.array([[i,j] for (i,j) in itertools.product(idx_X, idx_X) if i < j])[bool_smaller].tolist()
        bool_smaller = dist.pdist(Z)<=b-_tol
        edgelist += np.array([[i,j] for (i,j) in itertools.product(range(num_points), range(num_points)) if i < j])[bool_smaller].tolist()
        for edge in edgelist:
            ax[idx].plot(Z[edge][:,0], Z[edge][:,1], c="black", zorder=0.5)
        # Remove axis 
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        # Draw node labels
        for i in range(Z.shape[0]):
            ax[idx].text(Z[i,0]+0.05, Z[i,1], f"{i}", fontsize=labelsize)
        # end for labels 
    # end for plots
    # Plot edges from a 
    bool_smaller = dist.pdist(X)<a+_tol
    edgelist = np.array([[i,j] for (i,j) in itertools.product(idx_X, idx_X) if i < j])[bool_smaller].tolist()
    for edge in edgelist:
        ax[0].plot(Z[edge][:,0], Z[edge][:,1], c="black", zorder=0.5)
    # # Plot edges from b
    bool_smaller = dist.pdist(Z) <b +_tol
    edgelist = np.array([[i,j] for (i,j) in itertools.product(range(num_points), range(num_points)) if i < j])[bool_smaller].tolist()
    for edge in edgelist:
        ax[2].plot(Z[edge][:,0], Z[edge][:,1], c="black", zorder=0.5)
    # Now, plot cycle graph of components 
    ax[3].set_xticks([])
    ax[3].set_yticks(list(range(num_points)))
    components_mat = []
    for idx in range(3):
        edgelist = edges_X[:pair_ab[0]-shift_ab[0]*int(idx!=0)+1].tolist()
        edgelist += edges_Z[:pair_ab[1]-shift_ab[1]*int(idx!=2)+1].tolist()
        components = compute_components(edgelist, num_points)
        components_mat.append(components)
    
    components_mat = np.array(components_mat)
    for idx in range(3):
        u_components = np.unique(components_mat[idx]).tolist()
        points = np.array([np.ones(len(u_components))*idx, u_components]).transpose()
        ax[3].scatter(points[:,0], points[:,1], c="black", zorder=2)
        if idx==1:
            for comp in u_components:
                col_idx = components_mat[1].tolist().index(comp)
                left_comp = components_mat[0, col_idx]
                right_comp = components_mat[2, col_idx]
                ax[3].plot([0,1,2],[left_comp, comp, right_comp], c="black")
    
    
    # Adjust frames a bit more far appart
    for idx in range(4):
        xlim = ax[idx].get_xlim()
        xlength = xlim[1]-xlim[0]
        xlim = (xlim[0]-xlength*0.1, xlim[1]+xlength*0.1)
        ylim = ax[idx].get_ylim()
        ylength = ylim[1]-ylim[0]
        ylim = (ylim[0]-ylength*0.1, ylim[1]+ylength*0.1)
        ax[idx].set_xlim(xlim)
        ax[idx].set_ylim(ylim)
        if idx < 3:
            ax[idx].set_aspect("equal")
    
    # Write titles 
    ax[0].set_title(f"{a:.2f}+, {b:.2f}-")
    ax[1].set_title(f"{a:.2f}-, {b:.2f}-")
    ax[2].set_title(f"{a:.2f}-, {b:.2f}+")
    ax[3].set_title(f"G({a:.2f},{b:.2f})")