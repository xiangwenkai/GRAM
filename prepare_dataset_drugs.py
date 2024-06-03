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
from rdkit.Chem import GetAdjacencyMatrix  # 构建分子邻接矩阵
from scipy.sparse import coo_matrix  #转换成COO格式
from tqdm import tqdm
import random
import json
random.seed(42)


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

    eig,arr=lg.eig(G)  # 打印特征值eig、特征向量arr
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
    # change to where you untarred the rdkit folder
    base_path = "./data/geom_drugs"
    drugs_file = os.path.join(base_path, "rdkit_folder/summary_drugs.json")

    save_dir = "./data/geom_drugs/processed"

    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    mol_paths = []
    for smiles, sub_dic in drugs_summ.items():
        pickle_path = os.path.join(base_path, "rdkit_folder",
                                   sub_dic.get("pickle_path", ""))
        if os.path.isfile(pickle_path):
            mol_paths.append(pickle_path)

    n = len(mol_paths)
    for i in tqdm(range(n)):
        rand = random.random()
        mol_path = mol_paths[i]
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)

        # set the keys of the new dictionary to
        # be SMILES strings
        lowestenergy = float("inf")
        for conf in dic['conformers']:
            if conf['totalenergy'] < lowestenergy:
                lowestenergy = conf['totalenergy']
                mol = conf['rd_mol']
                name = 'gdb' + str(i)

        mol = Chem.RemoveHs(mol)
        num_nodes = mol.GetNumAtoms()
        if num_nodes > 40:
            continue

        # conformations
        signs = []
        g = construct_bigraph_from_mol_int(mol, featurize_atoms)
        d = AllChem.Get3DDistanceMatrix(mol)
        d = d ** 2
        sum_all = d.mean()
        sum_r = d.mean(0, keepdims=True)
        sum_c = d.mean(1, keepdims=True)
        
        # Gram matrix
        G = (sum_all - sum_r - sum_c + d) * (-0.5)

        # dist and bond
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

        # 2d position
        smi = Chem.MolToSmiles(mol)
        tem = Chem.MolFromSmiles(smi)
        AllChem.Compute2DCoords(tem)
        pos_2d = torch.tensor(tem.GetConformer().GetPositions(), dtype=torch.float)

        A = GetAdjacencyMatrix(mol)  # 创建邻接矩阵
        coo_A = coo_matrix(A)
        edge_index = [coo_A.row, coo_A.col]

        dist, angle, idx_i, idx_j, idx_k = xyztodat(pos, torch.tensor(edge_index, dtype=torch.long), num_nodes)

        if rand < 0.8:
            save_file = os.path.join(save_dir, 'train', str(name))
        elif rand < 0.9:
            save_file = os.path.join(save_dir, 'val', str(name))
        else:
            save_file = os.path.join(save_dir, 'test', str(name))
        with open(save_file, "wb") as file:
            pickle.dump((mol, g, G, pos, dist, angle, torch.tensor(edge_index, dtype=torch.long),
                         idx_i, idx_j, idx_k), file)


