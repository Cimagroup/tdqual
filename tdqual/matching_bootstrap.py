import numpy as np
from scipy.stats import bootstrap
import tdqual.topological_data_quality_0 as tdqual


def bootstrap_subsamples(X, Z, size_subsample_X, size_subsample_compl, nb_times=80):
    # Assume that the first X.shape[0] points from Z are those from X
    sub = []
    # construct a list of nb_times x nb_points
    for sub_idx in range(nb_times):
        X_indices = np.random.choice(range(X.shape[0]), size_subsample_X)
        compl_indices = np.random.choice(range(X.shape[0]+1, Z.shape[0]), size_subsample_compl)
        X_subsample = Z[X_indices]
        C_subsample = Z[compl_indices]
        Z_subsample = np.vstack((X_subsample,C_subsample))
        # Append subsample 
        sub.append((X_subsample, Z_subsample))
    # range over all subsamples
    return sub

def bootstrap_subsamples_pairs(X, Z, size_subsamples, nb_times=80):
    sub = []
    # construct a list of nb_times x nb_points
    for sub_idx in range(nb_times):
        X_idx = np.random.choice(range(X.shape[0]), size_subsamples)
        X_sub = X[X_idx]
        Z_idx = np.random.choice(range(Z.shape[0]), size_subsamples)
        Z_sub = Z[Z_idx]
        # Append subsample 
        sub.append((X_sub, Z_sub))
    # range over all subsamples
    return sub


######################################
# L1 norm
######################################

def process_pairs_L1(pairs_subsamples):
    """
    Replaces the pipeline. Loops through your custom data samples, 
    extracts finite diagram points using your function, and converts them to landscapes.
    """
    finite_L1_weights = []
    inf_L1_weights = []
    
    for (X,Z) in pairs_subsamples:
        filt_X, filt_Z, matching = filt_X, filt_Z, matching = tdqual.compute_Mf_0(X, Z)
        
        # Call your previous function to split the points
        finite_diag, inf_diag = tdqual.compute_matching_diagram_separated(filt_X, filt_Z, matching)
        
        finite_L1_weights.append(np.sum(np.abs(finite_diag[:,0]-finite_diag[:,1])))
        inf_L1_weights.append(np.sum(np.abs(inf_diag[:,1])))
    # return 
    return finite_L1_weights, inf_L1_weights
    
######################################
# Landscapes
######################################


def plot_CI_landscape(landscapes, color, label, ax, landscape_resolution):
    """Processes and plots the bootstrapped confidence intervals for the landscapes."""
    rng = np.random.default_rng()
    res = bootstrap(
        (np.transpose(landscapes),),
        np.std,
        method="basic",
        axis=-1,
        confidence_level=0.95,
        random_state=rng,
    )
    ci_l, ci_u = res.confidence_interval
    ax.fill_between(np.arange(0, landscape_resolution, 1), ci_l, ci_u, alpha=0.3, color=color, label=label)

def process_pairs_landscapes(pairs_subsamples, landscape_transformer):
    """
    Replaces the pipeline. Loops through your custom data samples, 
    extracts finite diagram points using your function, and converts them to landscapes.
    """
    finite_landscapes = []
    inf_landscapes = []
    
    for (X,Z) in pairs_subsamples:
        filt_X, filt_Z, matching = filt_X, filt_Z, matching = tdqual.compute_Mf_0(X, Z)
        
        # Call your previous function to split the points
        finite_diag, inf_diag = tdqual.compute_matching_diagram_separated(filt_X, filt_Z, matching)

        # We need to transpose the coordinates of the finite diagram in order for the GUDHI landscape to work properly
        finite_diag = finite_diag[:,[1,0]]
        # GUDHI's Landscape transformer expects a list of diagrams (each a numpy array)
        # If finite_pts is empty, we pass an empty array of shape (0, 2)
        finite_diag = finite_diag if len(finite_diag) > 0 else np.empty((0, 2))
        inf_diag = inf_diag if len(inf_diag) > 0 else np.empty((0, 2))
        finite_landscapes.append(finite_diag)
        inf_landscapes.append(inf_diag)
        
    # Fit/transform the landscapes globally for this activity matrix
    # (In a global setup, you'd want to fit on ALL activities combined first 
    # to lock the internal min/max grid layout, just like pipe.fit did)
    return landscape_transformer.fit_transform(finite_landscapes), landscape_transformer.fit_transform(inf_landscapes)