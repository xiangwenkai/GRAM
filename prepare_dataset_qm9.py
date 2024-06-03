#!/usr/bin/env python3
import torch
from torch_sparse import SparseTensor
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import rdkit 
import pandas as pd
import os
from rdkit.Chem.rdchem import ChiralType
from rdkit import Chem
from fea_mole import construct_bigraph_from_mol_int,featurize_atoms

from rdkit.Chem import GetAdjacencyMatrix 
from scipy.sparse import coo_matrix  # COO
from tqdm import tqdm
import random
import dgl
import torch.nn as nn


def xyztodat(pos, edge_index, num_nodes):
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # |sin_angle| * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)

    return dist, angle, idx_i, idx_j, idx_k


def get_coor(G):
    from scipy import linalg as lg

    eig,arr=lg.eig(G)
    print("eig is {}".format(np.round(eig,2)))
    print("arr is {}".format(np.round(arr,2)))

    top_k=3
    top_k_idx=eig.argsort()[::-1][0:top_k]
    eig_=eig[top_k_idx]
    eig_=np.float32(eig_)
    arr=np.array(np.float32(arr[:,top_k_idx]))
    z=np.array(np.diag(eig_)**0.5)
    print("eig_sort is {}".format(np.round(eig,2)))
    print("arr_sort is {}".format(np.round(arr,2)))
    k=np.dot(arr,z)
    return k

if __name__ == "__main__":
    random.seed(42)

    atom_distance_dict = {}
    path = "data/qm9/gdb9.sdf"
    path_12_task = "data/qm9/qm9.csv"
    pd_reader = pd.read_csv(path_12_task)

    save_dir = "data/qm9/processed/p2"
    mols_suppl = Chem.SDMolSupplier(path, removeHs=True)

    n = len(mols_suppl)
    for i in tqdm(range(n)):
        mol = mols_suppl[i]
        if mol is None or len(mol.GetAtoms()) <= 1:
            pass
        else:
            # conformation
            signs = []
            g = construct_bigraph_from_mol_int(mol, featurize_atoms)
            d = AllChem.Get3DDistanceMatrix(mol)
            d = d**2
            sum_all=d.mean()
            sum_r=d.mean(0,keepdims=True)
            sum_c=d.mean(1,keepdims=True)
            # Gram
            G=(sum_all-sum_r-sum_c+d)*(-0.5)
            name = mol.GetProp('_Name')

            # dist and bond
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

            A = GetAdjacencyMatrix(mol)   
            coo_A = coo_matrix(A)
            edge_index = [coo_A.row,coo_A.col]
            num_nodes = mol.GetNumAtoms()
            dist, angle, idx_i, idx_j, idx_k = xyztodat(pos, torch.tensor(edge_index,dtype=torch.long), num_nodes)

            # pos_3 = pos - pos[0]

            # tasks
            row = pd_reader[pd_reader['mol_id'] == name]
            mu = row.mu.values[0]
            alpha = row.alpha.values[0]
            HOMO = row.homo.values[0]
            LUMO = row.lumo.values[0]
            gap = row.gap.values[0]
            R2 = row.r2.values[0]
            ZPVE = row.zpve.values[0]
            U0 = row.u0.values[0]
            U = row.u298.values[0]
            H = row.h298.values[0]
            G_qm9 = row.g298.values[0]
            Cv = row.cv.values[0]

            save_file = os.path.join(save_dir, 'all', name)
            with open(save_file,"wb") as file:
                pickle.dump((mol,g,G,pos,mu,alpha,HOMO,LUMO,gap,R2,ZPVE,U0,U,H,G_qm9,Cv,dist, angle, torch.tensor(edge_index,dtype=torch.long), idx_i, idx_j, idx_k), file)

