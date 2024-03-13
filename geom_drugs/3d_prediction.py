import pandas as pd
import torch
from model import Graphormer
from dgl import backend
import torch.nn as nn
import datetime
# import bcolors
import os
import numpy as np
import glob
from sklearn.metrics import r2_score  # noqa
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from dgllife.utils import EarlyStopping, Meter, RandomSplitter
from torch.utils.data import DataLoader
from interaction_dataset_bond_length_angle import InteractionDataset
from utils_interaction_length_angle_torch import set_random_seed, collate_molgraphs
import time
from rdkit.Geometry import Point3D
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD
import random
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from copy import deepcopy
random.seed(11)


def run_an_eval_epoch(model, val_loader):
    model.eval()
    all_pre_G, all_label_G, all_pre_length, all_label_length, all_pre_angle, all_label_angle = [], [], [], [], [], []
    train_loss_G, train_loss_length, train_loss_angle = 0.0, 0.0, 0.0
    batch_id = 0
    for batch_id, batch_data in enumerate(val_loader):
        bg, b_G, b_dist, pos, b_angle, b_edge_index, b_idx_i, b_idx_j, b_idx_k = batch_data
        bg = bg.to(device)
        atom_feats, bond_feats = bg.ndata.pop('hv'), bg.edata.pop('he')
        # atom_feats = atom_feats + torch.from_numpy(np.random.normal(mu, sigma, atom_feats.shape)).to(device)
        atom_feats = backend.pad_packed_tensor(atom_feats.float(), bg.batch_num_nodes(), 0)
        attn_bias = torch.zeros([atom_feats.shape[0], atom_feats.shape[1], atom_feats.shape[1]],
                                dtype=torch.float).to(device)
        b_G = b_G.to(device)

        k, G_prediction = model(bg, atom_feats, attn_bias, bond_feats)
        loss_G = (loss_fn(G_prediction[b_G != 0.].flatten(), b_G[b_G != 0.].flatten())).mean().item()
        train_loss_G = train_loss_G + loss_G
        all_pre_G.extend(G_prediction.flatten().detach().cpu().numpy().reshape(-1).tolist())
        all_label_G.extend(b_G.flatten().detach().cpu().numpy().reshape(-1).tolist())
    all_pre_G = np.array(all_pre_G)
    all_label_G = np.array(all_label_G)
    train_loss_G = train_loss_G / (batch_id + 1)
    return all_pre_G, all_label_G, train_loss_G


def pearson_r2_score(y_true, y_pred):
    """Computes Pearson R^2 (square of Pearson correlation).
    Parameters
    ----------
    y: np.ndarray
      ground truth array
    y_pred: np.ndarray
      predicted array
    Returns
    -------
    float
      The Pearson-R^2 score.
    """
    return r2_score(y_true, y_pred)


def rmse_score(y_true, y_pred):
    """Computes RMS error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_score(y_true, y_pred):
    """Computes MAE."""
    return mean_absolute_error(y_true, y_pred)


def get_all_metric(y_true, y_pred):
    y_true_ = y_true[y_true != 0]
    y_pred_ = y_pred[y_true != 0]
    return pearson_r2_score(y_true_, y_pred_), rmse_score(y_true_, y_pred_), mae_score(y_true_, y_pred_)

test_sdf_paths = r"data\fastsmcg\processed\*"
test_sdfs = glob.glob(test_sdf_paths)
test_set = InteractionDataset(test_sdfs, load=True, n_jobs=10)
num_samples = len(test_sdfs)
print("train samples: {}\n".format(num_samples))

bsz = 1
test_loader = DataLoader(dataset=test_set, batch_size=bsz, shuffle=False, num_workers=1, collate_fn=collate_molgraphs)

mu = 0.0
sigma = 0.2

loss_fn = nn.MSELoss(reduction='none')
Blr_loss = nn.SmoothL1Loss()
Bar_loss = nn.SmoothL1Loss()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Graphormer(
    n_layers=6,
    num_heads=8,
    hidden_dim=512,
    sp_num_heads=8,
    dropout_rate=0.1,
    intput_dropout_rate=0.1,
    ffn_dim=512,
    attention_dropout_rate=0.1
)

PATH = r"models\pretraining_checkpoints_best.pt"

model.load_state_dict(torch.load(PATH))
model.to(device)

if __name__ == '__main__':
    model.eval()
    t0 = time.time()
    tc = 0
    for batch_id, batch_data in enumerate(test_loader):
        mol, bg, b_G, b_dist, pos, b_angle, b_edge_index, b_idx_i, b_idx_j, b_idx_k = batch_data
        bg = bg.to(device)
        b_G = b_G.to(device)
        b_dist = b_dist.to(device)
        b_angle = b_angle.to(device)
        b_edge_index.to(device)
        b_idx_i.to(device)
        b_idx_j.to(device)
        b_idx_k.to(device)

        atom_feats, bond_feats = bg.ndata.pop('hv'), bg.edata.pop('he')
        atom_feats = atom_feats + torch.from_numpy(np.random.normal(mu, sigma, atom_feats.shape)).to(device)
        atom_feats = backend.pad_packed_tensor(atom_feats.float(), bg.batch_num_nodes(), 0)
        attn_bias = torch.zeros([atom_feats.shape[0], atom_feats.shape[1], atom_feats.shape[1]],
                                dtype=torch.float).to(device)
        k, G_prediction = model.forward(bg, atom_feats, attn_bias, bond_feats, perturb=None)  # 1024,9,17

        pdb_block = Chem.MolToPDBBlock(mol[0])
        with open(r'pdb_compare\real{}.pdb'.format(tc), 'w') as f:
            f.write(pdb_block)
        conf = mol[0].GetConformer()
        for i in range(mol[0].GetNumAtoms()):
            x, y, z = k[i].detach().cpu().numpy().tolist()
            conf.SetAtomPosition(i, Point3D(x, y, z))
        # MMFFOptimizeMolecule(mol[0])
        pdb_block = Chem.MolToPDBBlock(mol[0])
        with open(r'pdb_compare\pre{}.pdb'.format(tc), 'w') as f:
            f.write(pdb_block)
        tc += 1

    rmsds = []
    for i in range(tc):
        try:
            mol1 = mda.Universe(
                r"pdb_compare\real{}.pdb".format(i))
            mol2 = mda.Universe(
                r"pdb_compare\pre{}.pdb".format(i))
        except:
            continue

        # init
        rmsd_analysis = RMSD(mol1, mol2)

        # RMSD analysis
        rmsd_analysis.run()
        # print(f"RMSD: {rmsd_analysis.rmsd[0, 2]:.2f}")
        rmsds.append(rmsd_analysis.rmsd[0, 2])

    print("average rmsd: {}".format(np.mean(rmsds)))


