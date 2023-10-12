"""Data processing utilities."""
import os
import copy
import random
from typing import Optional, Union, List

import scipy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import scanpy as sc
from anndata import AnnData

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
import torch_cluster


# set all seeds together
def seed_all(seed):
    if not seed:
        seed = 33

    print(f"Using random seed:{seed}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



def edge_to_A(edge_index, num_nodes):
    r"""
    Transform PyG edge_index to adjcency matrix
    """

    A = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)

    A[edge_index[0], edge_index[1]] = 1
    A[edge_index[1], edge_index[0]] = 1

    return A

def edge_sim(adata, edge_index, sim_mode):
    r"""
    Generate the similarity-edge for ST graph of slice data.

    Parameters Statements
    ----------
    adata
        AnnData object
    edge_index
        edge set of constructed spatial graph (from k-NN or radius)
    sim_mode
        Similarity function employed to generate edge_weight (cos, euclidean)

    Returns
    -------
    The similarity-based edge_weight of spatial graph for graph model input
    """
    if edge_index.device != 'cpu':
        edge_index = edge_index.to('cpu')

    num_nodes = len(adata)
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)
    A[edge_index[0], edge_index[1]] = 1
    A[edge_index[1], edge_index[0]] = 1

    if sim_mode == 'cos':
        sim_matrix = cosine_similarity(adata.X, adata.X) * A.numpy()
        edge_weight = sim_matrix[edge_index[0], edge_index[1]]

    elif sim_mode == 'euclidean':
        eud_global = euclidean_distances(adata.X, adata.X)
        eud_matrix = (1 - eud_global / np.max(eud_global)) * A.numpy()
        edge_weight = eud_matrix[edge_index[0], edge_index[1]]

    return edge_weight


# construct consistent pipeline for benchmark datasets
def scanpy_workflow(adata: AnnData,
                    filter_cell: Optional[bool] = False,
                    scale: Optional[bool] = True,
                    min_gene: Optional[int] = 200,
                    min_cell: Optional[int] = 30,
                    call_hvg: Optional[bool] = True,
                    n_top_genes: Optional[Union[int, List]] = 2500,
                    batch_key: Optional[str] = None,
                    n_comps: Optional[int] = 50,
                    viz: Optional[bool] = False,
                    resolution: Optional[float] = 0.8
                    ) -> AnnData:
    r"""
    Scanpy workflow using Seurat HVG

    Parameters
    ----------
    adata
        adata
    filter_cell
        whether to filter cells and genes
    min_gene
        min number of genes per cell
    min_cell
        min number of cells per gene
    call_hvg
        whether to call highly variable genes (only support seurat_v3 method)
    n_top_genes
        n top genes or gene list
    n_comps
        n PCA components
    viz
        whether to run visualize steps
    resolution
        resolution for leiden clustering (used when viz=True)

    Return
    ----------
    anndata object
    """
    if 'counts' not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()

    if filter_cell:
        print("Filter genes and cells.")
        sc.pp.filter_cells(adata, min_genes=min_gene)
        sc.pp.filter_genes(adata, min_cells=min_cell)

    if call_hvg:
        if isinstance(n_top_genes, int):
            if adata.n_vars > n_top_genes:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3", batch_key=batch_key)
            else:
                adata.var['highly_variable'] = True
                print("All genes are highly variable.")
        elif isinstance(n_top_genes, list):
            adata.var['highly_variable'] = False
            n_top_genes = list(set(adata.var.index).intersection(set(n_top_genes)))
            adata.var.loc[n_top_genes, 'highly_variable'] = True
    else:
        print("Skip calling highly variable genes.")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")

    if viz:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_comps)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
    return adata


def Cal_Spatial_Net(adata: AnnData,
                    rad_cutoff: Optional[Union[None, int]] = None,
                    k_cutoff: Optional[Union[None, int]] = 6,
                    model: Optional[str] = 'KNN',
                    return_data: Optional[bool] = False,
                    verbose: Optional[bool] = True,
                    sim_mode: Optional[Union[None, str]] = None
                    ) -> None:
    r"""
    Construct the spatial neighbor graph for ST data.

    Parameters Statements
    ----------
    adata
        AnnData object
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff.
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    sim_mode
        Mode to construct edge_weight

    Returns
    -------
    The spatial graphs are saved in adata.uns['(model)_Spatial_Graph']
    """

    assert (model in ['Radius', 'KNN']), "Please choose the right model('Radius' or 'KNN') for spatial graph construction"
    if verbose:
        print(f'Choosing {model} method to calculate spatial neighbor graph ...')

    if model == 'KNN':
        edge_index = torch_cluster.knn_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                             k=k_cutoff, loop=True, num_workers=8)
        edge_index = to_undirected(edge_index, num_nodes=adata.shape[0])  # ensure the graph is undirected
    elif model == 'Radius':
        if rad_cutoff is None:
            rad_cutoff = 2 * (adata.obsm['spatial'][:, 1].max() - adata.obsm['spatial'][:, 1].min()) / np.sqrt(
                adata.shape[0])
        edge_index = torch_cluster.radius_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                                r=rad_cutoff, loop=True, num_workers=8)

    graph_df = pd.DataFrame(edge_index.numpy().T, columns=['Cell1', 'Cell2'])
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df['Cell1'] = graph_df['Cell1'].map(id_cell_trans)
    graph_df['Cell2'] = graph_df['Cell2'].map(id_cell_trans)
    if sim_mode:
        edge_weight = edge_sim(adata, edge_index, sim_mode)
        graph_df['weight'] = edge_weight
    adata.uns['Spatial_Net'] = graph_df
    
    if verbose:
        print(f'The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{graph_df.shape[0] / adata.n_obs} neighbors per cell on average.')

    if return_data:
        return adata
    
def Transfer_pyg_Data(adata:AnnData,
                      feature:Optional[str]='PCA',
                      weight:Optional[bool] = False
    ) -> Data:
    r"""
    Transfer an adata with spatial info into PyG dataset (only for test)
    
    Parameters:
    ----------
    adata
        Anndata object
    feature
        use which data to build graph
        - PCA (default)
        
    Note:
    ----------
    Only support 'Spatial_Net' which store in `adata.uns` yet
    """
    adata = adata.copy()
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    
    # build Adjacent Matrix
    G = scipy.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + scipy.sparse.eye(G.shape[0])

    edgeList = np.nonzero(G)
    
    # select feature
    assert feature.lower() in ['hvg','pca','raw']
    if feature.lower() == 'raw':
        if type(adata.X) == np.ndarray:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
        return data
    elif feature.lower() in ['pca','hvg']:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        if feature.lower() == 'hvg':
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))
        sc.pp.scale(adata, max_value=10)
        print('Use PCA to format graph')
        sc.tl.pca(adata, svd_solver='arpack')
        data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['X_pca'].copy()))
        return data, adata.varm['PCs']
    
def transform2dict(anchor: AnnData,
                   positive: AnnData,
                   negative: AnnData,
                   feature:Optional[str]='raw'):
    """
    transform a anndata to training data (triplet)
    """
    anchor_data = Transfer_pyg_Data(anchor, feature=feature)
    positive_data = Transfer_pyg_Data(positive, feature=feature)
    negative_data = Transfer_pyg_Data(negative, feature=feature)
    
    new_data = dict()
    new_data["edge_index_1"] = anchor_data.edge_index
    new_data["edge_index_2"] = positive_data.edge_index
    new_data["edge_index_3"] = negative_data.edge_index

    new_data["features_1"] = anchor_data.x
    new_data["features_2"] = positive_data.x
    new_data["features_3"] = negative_data.x
    
    return new_data

def generate_single(datalist: List,
                    path: Optional[str] = './dataset/single/'
                    ):
    '''
    generate single graph dataset for GATE pre-training
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'path not exists, create path {path}')
        
    for i, adata in enumerate(datalist):
        single_graph = Transfer_pyg_Data(adata, feature='raw')
        single_data = dict()
        single_data["edge_index"] = single_graph.edge_index
        single_data["features"] = single_graph.x
        torch.save(single_data, path+str(i)+'.pt')
    return 'single graph data generation accomplished!'
    

def generate_contrastive(datalist: List,
                         path: Optional[str] = './dataset/train/',
                         size: Optional[Union[str, int]] = 'minimum',
                         **kwargs
                         ):
    '''
    generate triplet dataset for contrastive learning
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'path not exists, create path {path}')
    
    if size == 'balance':
        sample_times = int(len(sum(datalist, []))/len(datalist))
        sample_times = int(sample_times * (sample_times - 1) / 2) * (len(datalist)-1)
    elif size == 'maximum':    
        sample_times = max([len(element) for element in datalist])
        sample_times = int(sample_times * (len(sum(datalist, [])) - sample_times) / 2)
    elif size == 'minimum':
        sample_times = min([len(element) for element in datalist])
        sample_times = int(sample_times * (len(sum(datalist, [])) - sample_times) / 2)
    elif isinstance(size, int):
        sample_times = size
    else:
        raise ValueError('choose the right size mode for generation')
        
    print(f'-------------------sample times:{sample_times}----------------------')
    
    for i in range(len(datalist)):
        extract_list = copy.deepcopy(datalist)
        extract = extract_list.pop(i)
        assert len(extract) > 1
        negatives = sum(extract_list, [])
        for j in range(sample_times):
            anchor, positive, *_ = random.sample(extract, 2)
            negative, *_ = random.sample(negatives, 1)
            data = transform2dict(anchor, positive, negative, **kwargs)
            torch.save(data, path+str((i*sample_times+j))+'.pt')

