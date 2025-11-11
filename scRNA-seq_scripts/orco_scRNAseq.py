import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
import scipy.linalg as la
from ot import emd2
import multiprocessing as mp
import time

def data_to_graph_laplacian(X, corr_method='pearson'):
    if X.shape[0] < X.shape[1]:
        X = X.T  

    print("Computing correlation matrix...")
    corr_mat = pd.DataFrame(X).T.corr(method=corr_method).values

    # Similarity and weighted adjacency
    adj = (1 + corr_mat) / 2
    np.fill_diagonal(adj, 0)
    adj = np.maximum(adj, 0)

    # Graph Laplacian
    rowsum = np.sum(adj, axis=1)
    D_inv = np.diag(1 / np.maximum(rowsum, 1e-8))
    L = np.eye(adj.shape[0]) - np.dot(D_inv, adj)

    print("Graph Laplacian computed: shape =", L.shape)
    return adj, L


def diffuse_for_tau(L, tau_range):
    print("Computing diffusions for tau range...")
    diffused_list = []
    for tau in tau_range:
        diffused = la.expm(-tau * L)
        diffused_list.append(np.ascontiguousarray(diffused))
    print("Completed diffusion over", len(tau_range), "tau values.")
    return diffused_list


def orc_edge(args):
    u, v, ti, diffusion_list, d0 = args
    m_u = diffusion_list[ti][u, :]
    m_v = diffusion_list[ti][v, :]
    w = emd2(m_u, m_v, d0)
    return w


def compute_orco(adj, dist_mat, diffusion_list, tau_range, n_proc=4):
    lv, rv = sp.triu(sp.csr_matrix(adj)).nonzero()
    d0 = dist_mat
    n_edges = len(lv)
    edge_w_all = []
    edge_orc_all = []

    print("Starting ORCO computation on", n_edges, "edges...")
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
    if tau_range is None:
        tau_range = np.logspace(-2, 2, 51)

    print("Extracting expression matrix...")
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
    print("✅ ORCO computation complete and stored in adata.uns['orco_curvature']")
    return adata
