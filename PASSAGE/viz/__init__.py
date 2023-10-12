'''
visualization modules of PASSAGE
'''

import torch
from pynvml import *
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

import umap as umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from .utils import map, calculate_purity
from ..model import Transfer_pyg_Data, get_free_gpu
from .color import *


gpu_index = get_free_gpu()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')


def view_pattern(model:torch.nn.Module,
                 adata:AnnData,
                 **kwargs
                 ):
    '''
    view phenotype-associated spatial signatures for given spatial slice
    '''
    graph_data = Transfer_pyg_Data(adata, feature='raw')
    data = dict()
    data["edge_index"] = graph_data.edge_index
    data["features"] = graph_data.x
    if device != 'cpu':
        for key in data:
            data[key] = data[key].to(device)
    abstract_features, _ = model.GATE(data['features'], data['edge_index'])
    abstract_features = model.MLP(abstract_features)
    attention_score, _ = model.attention(abstract_features)
    if device == 'cpu':
        attention_score = attention_score.detach().numpy().reshape(-1)
    else:
        attention_score = attention_score.cpu().detach().numpy().reshape(-1)
    series = pd.Series(attention_score)
    series.index = adata.obs.index

    attention_score = map(attention_score, 0, 1)
    adata1 = adata[np.argwhere(attention_score >= 0.5).reshape(-1)]
    adata2 = adata[np.argwhere(attention_score < 0.5).reshape(-1)]
    score1 = calculate_purity(adata1)
    score2 = calculate_purity(adata2)

    if score1 > score2:
        adata.obs['attention'] = map(series, 1, 0)
    else:
        adata.obs['attention'] = map(series, 0, 1)

    sc.pl.spatial(adata, size=1, color='attention', **kwargs)

def generate_representation(model:torch.nn.Module,
                            adata_list:List[AnnData]):
    '''
    generate the embedding of spatial slices trained by PASSAGE
    '''
    embed = np.zeros((len(adata_list), model.attention_pool_size))
    for i, adata in enumerate(adata_list):
        graph_data = Transfer_pyg_Data(adata, feature='raw')
        data = dict()
        data["edge_index"] = graph_data.edge_index
        data["features"] = graph_data.x
        if device != 'cpu':
            for key in data:
                data[key] = data[key].to(device)
        abstract_features, _ = model.GATE(data['features'], data['edge_index'])
        abstract_features = model.MLP(abstract_features)
        _, representation = model.attention(abstract_features)
        if device == 'cpu':
            representation = representation.detach().numpy().reshape(-1)
        else:
            representation = representation.cpu().detach().numpy().reshape(-1)
        embed[i] = representation
    return embed

def view_embedding(model:torch.nn.Module,
                   adata_list:List[AnnData],
                   labels:List[int],
                   title:Optional[str]='UMAP of trained embedding',
                   save:Optional[Union[None, str]]=None,
                   **kwargs
                   ):
    '''
    view distribution of trained embeddings of spatial slices
    '''
    embed = generate_representation(model, adata_list)
    df = pd.DataFrame(embed)
    df['class'] = labels
    reducer = umap.UMAP(n_neighbors = 8, min_dist=1)
    scaled_values = StandardScaler().fit_transform(df.iloc[:,:-1])
    umap_embedding = reducer.fit_transform(scaled_values)
    df = pd.DataFrame({'x': umap_embedding[:, 0], 'y': umap_embedding[:, 1],
                       'class': df['class']})
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='x', y='y', hue='class', palette=ggsci_npg, data=df, s=100)
    plt.legend(title='Class')
    plt.title(title)
    if save:
        plt.savefig(save)
    plt.show()

def classification(model:torch.nn.Module,
                   train_adatas:List[AnnData],
                   labels:List[int],
                   test_adatas:List[AnnData],
                   n_neighbors:Optional[int]=5):
    '''
    employ KNN for classification
    '''
    train_embed = generate_representation(model, train_adatas)
    pred_embed = generate_representation(model, test_adatas)
    train_df = pd.DataFrame(train_embed)
    pred_df = pd.DataFrame(pred_embed)
    train_df['class'] = labels

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.iloc[:,:-1])
    test_scaled = scaler.transform(pred_df)

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(train_scaled, train_df.iloc[:,-1])
    y_pred = knn.predict(test_scaled)

    return y_pred


