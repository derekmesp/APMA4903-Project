import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
import scipy.linalg as la
from ot import emd2
import multiprocessing as mp
import time

def data_to_graph_laplacian(X, corr_method='pearson'):
    """
    Compute a graph Laplacian from a data matrix using pairwise correlations.

    This function constructs a weighted, undirected graph where edge weights 
    represent pairwise correlations between variables (e.g., genes or features). 
    The correlation matrix is converted to a nonnegative adjacency matrix, 
    and the corresponding random-walk normalized graph Laplacian is computed.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input data matrix of shape `(n_samples, n_features)` or `(n_features, n_samples)`.
        If the number of rows is less than the number of columns, the matrix is 
        transposed automatically to ensure that correlations are computed between variables.
    corr_method : {'pearson', 'spearman', 'kendall'}, optional, default='pearson'
        Method used to compute pairwise correlations.

    Returns
    -------
    adj : np.ndarray
        The nonnegative adjacency matrix of shape `(n_features, n_features)`, 
        derived from the pairwise correlations. Values are scaled to `[0, 1]` and 
        diagonal entries are set to zero.
    L : np.ndarray
        The random-walk normalized graph Laplacian matrix of shape `(n_features, n_features)`, 
        computed as `L = I - D^{-1} A`, where `D` is the degree matrix.
    
    if X.shape[0] < X.shape[1]:
        X = X.T  
    """
    corr_mat = pd.DataFrame(X).T.corr(method=corr_method).values

    adj = (1 + corr_mat) / 2
    np.fill_diagonal(adj, 0)
    adj = np.maximum(adj, 0)

    rowsum = np.sum(adj, axis=1)
    D_inv = np.diag(1 / np.maximum(rowsum, 1e-8))
    L = np.eye(adj.shape[0]) - np.dot(D_inv, adj)

    print("Graph Laplacian computed: shape =", L.shape)
    return adj, L


def diffuse_for_tau(L, tau_range):
    """
    Compute diffusion kernels over a range of diffusion times (τ) from a graph Laplacian.

    This function applies the heat diffusion process on a graph for multiple 
    diffusion scales. For each τ in `tau_range`, it computes the matrix exponential 
    of `-τL`, which represents the diffusion kernel that describes how information 
    (or heat) propagates through the graph over time τ.

    Parameters
    ----------
    L : np.ndarray
        Graph Laplacian matrix of shape `(n, n)`. Typically obtained from 
        `data_to_graph_laplacian()`. Must be square and symmetric positive semidefinite.
    tau_range : array-like of float
        Sequence of diffusion time scales (τ values) over which to compute the 
        diffusion kernels. Larger τ values correspond to more global, smoothed diffusion.

    Returns
    -------
    diffused_list : list of np.ndarray
        List of diffusion matrices `exp(-τL)` for each τ in `tau_range`.  
        Each matrix has shape `(n, n)` and is contiguous in memory for efficient computation.
    """
    diffused_list = []
    for tau in tau_range:
        diffused = la.expm(-tau * L)
        diffused_list.append(np.ascontiguousarray(diffused))
    return diffused_list


def orc_edge(args): 
    """
    Compute the Ollivier-Ricci curvature (edge cost) between two nodes at a given diffusion scale.

    This function evaluates the Wasserstein (Earth Movers) distance between the diffusion 
    distributions of two nodes `u` and `v` on a graph, at a specific diffusion time `τ`. 
    The resulting value represents the transport cost used in computing the 
    Ollivier-Ricci curvature (ORC) along the edge `(u, v)`.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        
        - **u** : int  
          Index of the first node.
        - **v** : int  
          Index of the second node.
        - **ti** : int  
          Index into `diffusion_list` specifying which diffusion time (τ) to use.
        - **diffusion_list** : list of np.ndarray  
          List of diffusion kernels (e.g., outputs from `diffuse_for_tau()`), 
          where each element is a matrix `exp(-τL)` of shape `(n, n)`.
        - **d0** : np.ndarray  
          Ground distance matrix of shape `(n, n)` representing the pairwise 
          base distances between all nodes.

    Returns
    -------
    w : float
        The Earth Mover’s Distance (Wasserstein distance) between the diffusion 
        distributions of nodes `u` and `v` at diffusion time `τ_ti`.  
        This value is typically used as the transport cost in computing 
        the Ollivier-Ricci curvature along edge `(u, v)`.
    """
    u, v, ti, diffusion_list, d0 = args
    m_u = diffusion_list[ti][u, :]
    m_v = diffusion_list[ti][v, :]
    w = emd2(m_u, m_v, d0)
    return w


def compute_orco(adj, dist_mat, diffusion_list, tau_range, n_proc=4):
    """
    Compute multi-scale Ollivier-Ricci Curvature (ORCO) across all graph edges.

    This function estimates the Ollivier-Ricci curvature between all pairs of 
    connected nodes (edges) in a weighted, undirected graph, across multiple 
    diffusion time scales `τ`. For each τ, the curvature is computed as a 
    function of the Wasserstein distance between local diffusion distributions 
    of node pairs.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix of the graph, shape `(n, n)`. Should be symmetric and 
        nonnegative, typically output from `data_to_graph_laplacian()`.
    dist_mat : np.ndarray
        Pairwise ground distance matrix between nodes, shape `(n, n)`.  
        Used by the optimal transport solver (`emd2`) to define the cost between 
        diffusion distributions.
    diffusion_list : list of np.ndarray
        List of diffusion kernel matrices, each of shape `(n, n)`, computed from 
        the graph Laplacian (e.g., using `diffuse_for_tau()`).
    tau_range : array-like of float
        Sequence of diffusion time scales (τ values) for which to compute 
        ORC values. Each τ corresponds to one diffusion kernel in `diffusion_list`.
    n_proc : int, optional, default=4
        Number of parallel worker processes used for computing edge-wise 
        Earth Mover’s Distances (via multiprocessing).

    Returns
    -------
    lv : np.ndarray
        Array of row indices of the upper-triangular nonzero edges in `adj`.
    rv : np.ndarray
        Array of column indices of the corresponding edges.
    edge_orc_all : list of np.ndarray
        List of Ollivier-Ricci curvature values for each τ in `tau_range`.  
        Each element has shape `(n_edges,)`, corresponding to the curvature 
        of each edge `(lv[i], rv[i])`.
    """
    lv, rv = sp.triu(sp.csr_matrix(adj)).nonzero()
    d0 = dist_mat
    n_edges = len(lv)
    edge_w_all = []
    edge_orc_all = []

    for ti, tau in enumerate(tau_range):
        print(f"  Tau = {tau:.3f} ({ti+1}/{len(tau_range)})")
        args_list = [(lv[i], rv[i], ti, diffusion_list, d0) for i in range(n_edges)]
        with mp.Pool(n_proc) as pool:
            edge_w = pool.map(orc_edge, args_list)
        edge_orc = 1 - (np.array(edge_w) / d0[lv, rv])
        edge_w_all.append(edge_w)
        edge_orc_all.append(edge_orc)
    return lv, rv, edge_orc_all


def run_orco_from_adata(adata, layer=None, corr_method='pearson', tau_range=None, n_proc=4):
    """
    Run multi-scale Ollivier-Ricci Curvature (ORCO) analysis directly on an AnnData object.

    This function performs a full ORCO computation pipeline on a single-cell or 
    bulk gene expression dataset stored in an AnnData object. It constructs a 
    correlation-based graph, computes the corresponding Laplacian, derives 
    diffusion kernels over a range of diffusion scales, and estimates 
    edge-wise Ollivier-Ricci curvature values across all scales.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix from the `anndata` package. Expression data are taken 
        from `adata.X` or a specified `layer`.
    layer : str or None, optional, default=None
        If provided, specifies which layer of `adata.layers` to use as the input matrix.
        If `None`, the main data matrix `adata.X` is used.
    corr_method : {'pearson', 'spearman', 'kendall'}, optional, default='pearson'
        Method used to compute pairwise correlations between features (e.g., genes).
    tau_range : array-like of float, optional
        Sequence of diffusion time scales for multi-scale curvature computation.
        If `None`, defaults to `np.logspace(-2, 2, 51)` (logarithmically spaced).
    n_proc : int, optional, default=4
        Number of parallel processes used for curvature computation.

    Returns
    -------
    adata : AnnData
        The input AnnData object with additional entries stored in `adata.uns`:
        
        - **"orco_tau"** : array  
          The list of diffusion time scales used.
        - **"orco_lv"**, **"orco_rv"** : arrays  
          The indices of the upper-triangular edges in the graph.
        - **"orco_curvature"** : list of np.ndarray  
          Ollivier-Ricci curvature values for each τ in `tau_range`.
        
        The graph structure and diffusion geometry can be used for downstream 
        visualization or manifold-based analysis.
    """
    if tau_range is None:
        tau_range = np.logspace(-2, 2, 51)

    X = adata.layers[layer] if layer else adata.X
    X = np.array(X.todense()) if sp.issparse(X) else np.array(X)

    adj, L = data_to_graph_laplacian(X, corr_method=corr_method)
    dist_mat = cg.shortest_path(sp.csr_matrix(1 / np.maximum(adj, 1e-8)))
    diffusion_list = diffuse_for_tau(L, tau_range)

    lv, rv, edge_orc_all = compute_orco(adj, dist_mat, diffusion_list, tau_range, n_proc=n_proc)

    adata.uns["orco_tau"] = tau_range
    adata.uns["orco_lv"] = lv
    adata.uns["orco_rv"] = rv
    adata.uns["orco_curvature"] = edge_orc_all
    return adata
