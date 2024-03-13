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
import random
random.seed(42)


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


val_sdf_paths = "/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/data/processed/p2/val/*"
test_sdf_paths = "/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/data/processed/p2/test/*"

val_sdfs = glob.glob(val_sdf_paths)
test_sdfs = glob.glob(test_sdf_paths)
val_set = InteractionDataset(val_sdfs, load=True, n_jobs=10)
test_set = InteractionDataset(test_sdfs, load=True, n_jobs=10)


bsz = 12
val_loader = DataLoader(dataset=val_set, batch_size=bsz, shuffle=False, num_workers=1, collate_fn=collate_molgraphs)
test_loader = DataLoader(dataset=test_set, batch_size=bsz, shuffle=False, num_workers=1, collate_fn=collate_molgraphs)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

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

model_type = 'gram'
if model_type == 'gram':
    PATH = "/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/model/checkpoints_best.pt"
elif model_type == 'rd':
    PATH = "/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/model_rd/checkpoints_best.pt"
elif model_type == 'coor':
    PATH = "/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/model_coordinate/checkpoints_best.pt"
model.load_state_dict(torch.load(PATH))


model.to(device)

if __name__ == '__main__':
    model.eval()
    all_pre_G, all_label_G, all_pre_length, all_label_length, all_pre_angle, all_label_angle, all_pre_d, all_label_d = [], [], [], [], [], [], [], []
    for batch_id, batch_data in enumerate(test_loader):
        # if batch_id < 1:
        #     continue
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
        atom_feats = backend.pad_packed_tensor(atom_feats.float(), bg.batch_num_nodes(), 0)
        attn_bias = torch.zeros([atom_feats.shape[0], atom_feats.shape[1], atom_feats.shape[1]],
                                dtype=torch.float).to(device)
        try:
            k, G_prediction = model.forward(bg, atom_feats, attn_bias, bond_feats, perturb=None)  # 1024,9,17
        except:
            continue
        max_atom = atom_feats.shape[1]
        # diff_coordinate = [x.to(device)-k[i*max_atom: i*max_atom+x.shape[0]] for i, x in enumerate(pos)]
        # loss_coor = sum([torch.mean(torch.abs(diff)) for diff in diff_coordinate])/bsz

        d = [AllChem.Get3DDistanceMatrix(mol_) for mol_ in mol]
        d = np.concatenate([x.flatten() for x in d])

        d_i = []
        d_j = []
        for i, x in enumerate(pos):
            start = max_atom * i
            dl = x.shape[0]
            for j in range(start, start+dl):
                for t in range(start, start+dl):
                    d_i.append(j)
                    d_j.append(t)

        k_i = k[b_edge_index[0]]
        k_j = k[b_edge_index[1]]
        length_prediction = (k_i - k_j).pow(2).sum(dim=-1).sqrt()

        k_di = k[d_i]
        k_dj = k[d_j]
        dlength_prediction = (k_di - k_dj).sum(dim=-1)

        k_i_a = k[b_idx_i]
        k_j_a = k[b_idx_j]
        k_k_a = k[b_idx_k]
        pos_ji = k_i_a - k_j_a
        pos_jk = k_k_a - k_j_a
        a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
        b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # |sin_angle| * |pos_ji| * |pos_jk|
        eps = 1e-15
        a_diff = (a == 0) * eps
        a = a + a_diff
        angle_prediction = torch.atan2(b, a)

        all_pre_G.extend(G_prediction.flatten().detach().cpu().numpy().reshape(-1).tolist())
        all_label_G.extend(b_G.flatten().detach().cpu().numpy().reshape(-1).tolist())

        all_pre_length.extend(length_prediction[b_dist != 0.].flatten().detach().cpu().numpy().reshape(-1).tolist())
        all_label_length.extend(b_dist[b_dist != 0.].flatten().detach().cpu().numpy().reshape(-1).tolist())

        all_pre_angle.extend(angle_prediction[b_angle != 0.].flatten().detach().cpu().numpy().reshape(-1).tolist())
        all_label_angle.extend(b_angle[b_angle != 0.].flatten().detach().cpu().numpy().reshape(-1).tolist())

        all_pre_d.extend(dlength_prediction.detach().cpu().numpy().reshape(-1).tolist())
        all_label_d.extend(list(d))

    all_label_length = np.array(all_label_length)
    all_pre_length = np.array(all_pre_length)
    all_label_angle = np.array(all_label_angle)
    all_pre_angle = np.array(all_pre_angle)
    all_label_d = np.array(all_label_d)
    all_pre_d = np.array(all_pre_d)
    all_label_G = np.array(all_label_G)
    all_pre_G = np.array(all_pre_G)

    all_label_length_ = all_label_length[all_label_length != 0]
    all_pre_length_ = all_pre_length[all_label_length != 0]
    all_label_angle_ = all_label_angle[all_label_angle != 0]
    all_pre_angle_ = all_pre_angle[all_label_angle != 0]
    all_label_d_ = all_label_d[all_label_d != 0]
    all_pre_d_ = all_pre_d[all_label_d != 0]
    all_label_G_ = all_label_G[all_label_G != 0]
    all_pre_G_ = all_pre_G[all_label_G != 0]

    length_mae = mae_score(all_label_length_, all_pre_length_)
    angle_mae = mae_score(all_label_angle_, all_pre_angle_)
    d_mae = mae_score(all_label_d_, all_pre_d_)

    G_r2 = r2_score(all_label_G_, all_pre_G_)
    G_rmse = rmse_score(all_label_G_, all_pre_G_)
    G_mae = mae_score(all_label_G_, all_pre_G_)
    print("length_mae: {}; angle_mae: {}; d_mae: {}; G_r2: {}; G_rmse: {}, G_mae: {}".format(length_mae, angle_mae, d_mae, G_r2, G_rmse, G_mae))






