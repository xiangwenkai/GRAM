import torch
from model import Graphormer
from dgl import backend
import torch.nn as nn
import datetime
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
import random
import argparse
random.seed(42)


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

        G_prediction = model(bg, atom_feats, attn_bias, bond_feats)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-GTM')
    parser.add_argument('--path', type=str, default='./data/geom_drugs', help='Dataset path')
    parser.add_argument('--device', type=str, default='cuda', help='Which gpu to use if any (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=16, help='Input batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mu', type=float, default=0.0, help='Mean value of noise')
    parser.add_argument('--sigma', type=float, default=0.2, help='Std value of noise')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of model')
    parser.add_argument('--n_attention', type=int, default=8, help='Number of attention')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Drop out rate')
    args = parser.parse_args()

    data_path = args.path
    device = args.device
    bsz = args.batch_size
    mu = args.mu
    sigma = args.sigma

    metric_name = ['epoch', 'val_r2', 'val_rmse', 'val_mae', 'test_r2', 'test_rmse', 'test_mae', 'loss']
    out = open("./out/graphormer_geom_pretrain.txt", "a+")
    out.write(",".join(metric_name) + "\n")

    train_sdf_paths = data_path + '/train/*'
    val_sdf_paths = data_path + '/val/*'
    test_sdf_paths = data_path + '/test/*'

    train_sdfs = glob.glob(train_sdf_paths)
    val_sdfs = glob.glob(val_sdf_paths)
    test_sdfs = glob.glob(test_sdf_paths)

    num_samples = len(train_sdfs)
    print("train samples: {}\n".format(num_samples))

    train_set = InteractionDataset(train_sdfs, load=True, n_jobs=10)
    val_set = InteractionDataset(val_sdfs, load=True, n_jobs=10)
    test_set = InteractionDataset(test_sdfs, load=True, n_jobs=10)

    train_loader = DataLoader(dataset=train_set, batch_size=bsz, shuffle=False, num_workers=1,
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set, batch_size=bsz, shuffle=False, num_workers=1, collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=test_set, batch_size=bsz, shuffle=False, num_workers=1,
                             collate_fn=collate_molgraphs)

    loss_fn = nn.MSELoss(reduction='none')
    # loss_fn = nn.L1Loss(reduction='none')
    Blr_loss = nn.SmoothL1Loss()
    Bar_loss = nn.SmoothL1Loss()

    model = Graphormer(
        n_layers=args.n_layers,
        num_heads=args.n_attention,
        hidden_dim=args.hidden_size,
        sp_num_heads=args.n_attention,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.dropout_rate,
        ffn_dim=args.hidden_size,
        attention_dropout_rate=args.dropout_rate
    )

    # PATH = "./models/checkpoints_1.pt"
    # model.load_state_dict(torch.load(PATH))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=0.000000001)

    stopper = EarlyStopping(mode='lower', filename='./out/gemo_graphormer_pretrain.pth', patience=80)


    epoch = 0
    while True:
        model.train()
        train_loss_G = 0.0
        t0 = time.time()
        tc = 0
        for batch_id, batch_data in enumerate(train_loader):
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
            try:
                k, G_prediction = model.forward(bg, atom_feats, attn_bias, bond_feats, perturb=None)  # 1024,9,17
            except:
                continue
            max_atom = atom_feats.shape[0]
            diff_coordinate = [x.to(device)-k[i*max_atom: i*max_atom+x.shape[0]] for i, x in enumerate(pos)]
            loss_coor = sum([torch.mean(torch.abs(diff)) for diff in diff_coordinate])/bsz

            k_i = k[b_edge_index[0]]
            k_j = k[b_edge_index[1]]
            length_prediction = (k_i - k_j).pow(2).sum(dim=-1)

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

            loss_G = (loss_fn(G_prediction[b_G != 0.].flatten(), b_G[b_G != 0.].flatten())).mean()
            b_dist_pow = b_dist.pow(2)
            loss_length = (
                Blr_loss(length_prediction[b_dist_pow != 0.].flatten(), b_dist_pow[b_dist_pow != 0.].flatten()))
            loss_angle = (Bar_loss(angle_prediction[b_angle != 0.].flatten(), b_angle[b_angle != 0.].flatten()))
            loss = loss_G + 10 * loss_length + 10 * loss_angle

            # loss = (loss_fn(G_prediction[b_G != 0].flatten(), b_G[b_G != 0].flatten())).mean()
            if batch_id % 500 == 0:
                print("epoch:{}  iter: {}/{} --- loss: {}\n".format(epoch, batch_id, int(num_samples/bsz), train_loss_G/(batch_id + 1)))
            train_loss_G = train_loss_G + loss.item()
            optimizer.zero_grad()
            loss.backward()
            # with torch.autograd.detect_anomaly():
            #     loss_angle.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=5.)
            optimizer.step()

        train_time = time.time() - t0
        epoch += 1
        if epoch % 1 == 0:
            torch.save(model.state_dict(), './models/checkpoints_{}_{}.pt'.format(epoch, time.strftime("%Y%m%d%H", time.localtime())))
        train_loss_G = train_loss_G / (batch_id + 1)
        val_pre_G, val_label_G, val_loss_G = run_an_eval_epoch(model, val_loader)
        test_pre_G, test_label_G, test_loss_G = run_an_eval_epoch(model, test_loader)

        all_metric = get_all_metric(val_label_G, val_pre_G) + get_all_metric(test_label_G, test_pre_G)
        all_metric = [round(i, 3) for i in all_metric]
        print(all_metric)
        all_metric = [str(epoch)] + [str(i) for i in all_metric] + [str(train_loss_G)] + [str(val_loss_G)] + [
            str(test_loss_G)]
        val_score = float(all_metric[3])
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.3f}, best validation {} {:.3f}, epoch train time: {:.1f}'.format(
            epoch, 1000, 'rmse',
            val_score, 'rmse', stopper.best_score, train_time))

        out.write(",".join(all_metric) + "\n")
        out.flush()

        if early_stop:
            break

