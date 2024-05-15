#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 05:13:07 2021

@author: zhongfeisheng
"""


import dgl.backend as F
import numpy as np
import os
import torch
import pickle
from dgl import save_graphs, load_graphs
from multiprocessing import Pool
from rdkit import Chem
from math import log2
from dgllife.utils.io import pmap
from fea_mole import  construct_bigraph_from_mol_int,featurize_atoms
from torch.nn.utils.rnn import pad_sequence


__all__ = ['InteractionDataset']
#torch.multiprocessing.set_sharing_strategy('file_system')
def compute_squared_EDM_method(X):
  # 获得矩阵都行和列，因为是行向量，因此一共有n个向量
  n,m = X.shape
  G_diag = torch.diagonal(X,dim1=-2,dim2=-1).unsqueeze(-2) # 取出一个batch中的对角线元素
  # 因为是行向量，n是向量个数,沿y轴复制n倍，x轴复制一倍
  H = G_diag.repeat(n,1)
  return (H + H.permute([1,0]) - 2*X).sqrt()    

def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

class InteractionDataset(object):
    """

    This is a general class for loading molecular data from :class:`pandas.DataFrame`.

    In data pre-processing, we construct a binary mask indicating the existence of labels.

    All molecules are converted into DGLGraphs. After the first-time construction, the
    DGLGraphs can be saved for reloading so that we do not need to reconstruct them every time.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe including smiles and labels. Can be loaded by pandas.read_csv(file_path).
        One column includes smiles and some other columns include labels.
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES string into a DGLGraph.
    node_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    edge_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    smiles_column: str
        Column name for smiles in ``df``.
    cache_file_path: str
        Path to store the preprocessed DGLGraphs. For example, this can be ``'dglgraph.bin'``.
    task_names : list of str or None
        Columns in the data frame corresponding to real-valued labels. If None, we assume
        all columns except the smiles_column are labels. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to False.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. It only comes
        into effect when :attr:`n_jobs` is greater than 1. Default to 1000.
    init_mask : bool
        Whether to initialize a binary mask indicating the existence of labels. Default to True.
    n_jobs : int
        The maximum number of concurrently running jobs for graph construction and featurization,
        using joblib backend. Default to 1.
    """
    def __init__(self,  data_paths, load=True, n_jobs=10,log_every=1000, init_mask=True
                 ):
        self.data_paths=data_paths

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the datapoint for all tasks
        Tensor of dtype float32 and shape (T), optional
            Binary masks indicating the existence of labels for all tasks. This is only
            generated when ``init_mask`` is True in the initialization.
        """
        sdf = self.data_paths[item]
        [mol, g, G, pos, dist, angle, edge_index, idx_i, idx_j, idx_k] = pickle.load(open(sdf, 'rb'))
        g.ndata['hv'] = convert_to_single_emb(g.ndata['hv'])
        G = torch.tensor(G)
        # return mol, g, G, pos, dist, angle, torch.from_numpy(np.array(edge_index)), idx_i, idx_j, idx_k
        return mol, g, G, pos, dist, angle, edge_index, idx_i, idx_j, idx_k

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.data_paths)


class InteractionDataset_moleculenet(object):
    def __init__(self,  data_paths, load=True, n_jobs=10,log_every=1000, init_mask=True
                 ):
        self.data_paths=data_paths

    def __getitem__(self, item):
        sdf = self.data_paths[item]
        [mol, g, tasks] = pickle.load(open(sdf, 'rb'))
        g.ndata['hv'] = convert_to_single_emb(g.ndata['hv'])
        return g, tasks

    def __len__(self):
        return len(self.data_paths)