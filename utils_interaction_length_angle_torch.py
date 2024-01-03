# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import numpy as np
import random
import torch
import os

from dgllife.utils import one_hot_encoding
from torch.nn.utils.rnn import pad_sequence


def set_random_seed(seed=0):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def chirality(atom):
    """Get Chirality information for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list of 3 boolean values
        The 3 boolean values separately indicate whether the atom
        has a chiral tag R, whether the atom has a chiral tag S and
        whether the atom is a possible chiral center.
    """
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]
def pad(array, shape,):
    """Pad a 2-dimensional array with zeros.
    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.
    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = torch.zeros(shape)
    for i in range(len(array)):
        padded_array[i] = array[i]
    return padded_array
def pad_array(array, shape,):
    """Pad a 2-dimensional array with zeros.
    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.
    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = torch.zeros(shape[0],shape[1])

    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array
def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    mol,g,G, pos, dist, angle, edge_index, idx_i, idx_j, idx_k = map(list, zip(*data))
    bg = dgl.batch(g)
    max_atoms=max([m.shape[1] for m in G])   #
    G= [pad_array(r,shape=(max_atoms,max_atoms)) for r in G]  #
    G=torch.stack(G)
    batch_dist = torch.cat(dist)
    batch_angle = torch.cat(angle)
    num_steps_list2 = torch.tensor([0] + [max_atoms for i in range(len(mol) -1)])
    num_steps_list2 = torch.cumsum(num_steps_list2, dim=0)
    repeats = torch.tensor([len(i.GetBonds())*2 for i in mol])
    batch_idx_repeated_offsets = torch.repeat_interleave(num_steps_list2, repeats)
    batch_edge_index = torch.cat([index for i,index in enumerate(edge_index)], dim=1) + batch_idx_repeated_offsets
    repeats_i = torch.tensor([len(i) for i in idx_i])
    batch_idx_repeated_offsets_i = torch.repeat_interleave(num_steps_list2, repeats_i)
    batch_idx_i = torch.cat([index for i,index in enumerate(idx_i)], dim=0) + batch_idx_repeated_offsets_i
    repeats_j = torch.tensor([len(i) for i in idx_j])
    batch_idx_repeated_offsets_j = torch.repeat_interleave(num_steps_list2, repeats_j)
    batch_idx_j = torch.cat([index for i,index in enumerate(idx_j)], dim=0) + batch_idx_repeated_offsets_j
    repeats_k = torch.tensor([len(i) for i in idx_k])
    batch_idx_repeated_offsets_k = torch.repeat_interleave(num_steps_list2, repeats_k)
    batch_idx_k = torch.cat([index for i,index in enumerate(idx_k)], dim=0) + batch_idx_repeated_offsets_k

    pos = pad_sequence(pos, batch_first=True, padding_value=-0.)
    return mol, bg, G, pos, batch_dist, batch_angle, batch_edge_index, batch_idx_i, batch_idx_j, batch_idx_k
