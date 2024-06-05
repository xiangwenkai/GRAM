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
from interaction_dataset_bond_length_angle_qm9 import InteractionDataset
from utils_interaction_length_angle_torch_qm9 import set_random_seed, collate_molgraphs
import time
import argparse
from torch.nn.utils.rnn import pad_sequence


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def run_an_eval_epoch(model, val_loader):
    model.eval()
    all_pre_G, all_label_G, all_pre_length, all_label_length, all_pre_angle, all_label_angle = [], [], [], [], [], []
    train_loss_G, train_loss_length, train_loss_angle = 0.0, 0.0, 0.0
    for batch_id, batch_data in enumerate(val_loader):
        mol, bg, b_G, pos, b_dist, b_angle, b_edge_index, b_idx_i, b_idx_j, b_idx_k = batch_data
        bg = bg.to(device)
        atom_feats, bond_feats = bg.ndata.pop('hv'), bg.edata.pop('he')
        # atom_feats = atom_feats + torch.from_numpy(np.random.normal(mu, sigma, atom_feats.shape)).to(device)
        atom_feats = backend.pad_packed_tensor(atom_feats.float(), bg.batch_num_nodes(), 0)
        attn_bias = torch.zeros([atom_feats.shape[0], atom_feats.shape[1], atom_feats.shape[1]],
                                dtype=torch.float).to(device)

        b_G = b_G.to(device)
        # pos = pos.to(device)

        k, G_prediction = model.forward(bg, atom_feats, attn_bias, bond_feats, perturb=None)

        loss_G = (loss_fn(G_prediction.flatten(), b_G.flatten())).mean().item()
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
    parser.add_argument('--path', type=str, default='./data/qm9', help='Dataset path')
    parser.add_argument('--device', type=str, default='cuda', help='Which gpu to use if any (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mu', type=float, default=0.0, help='Mean value of noise')
    parser.add_argument('--sigma', type=float, default=0.2, help='Std value of noise')
    parser.add_argument('--n_layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of model')
    parser.add_argument('--n_attention', type=int, default=8, help='Number of attention')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Drop out rate')
    args = parser.parse_args()

    set_random_seed(123)

    data_path = args.path
    device = args.device
    bsz = args.batch_size
    mu = args.mu
    sigma = args.sigma

    metric_name = ['epoch', 'val_r2', 'val_rmse', 'val_mae', 'test_r2', 'test_rmse', 'test_mae', 'loss']
    out = open("out/qm9_graphormer_pretrain.txt", "w")
    out.write(",".join(metric_name) + "\n")
    all_sdf_paths = data_path + "/processed/*"
    all_sdfs = glob.glob(all_sdf_paths)
    dataset = InteractionDataset(all_sdfs, load=True, n_jobs=1)
    # 999 888
    train_set, val_set, test_set = RandomSplitter.train_val_test_split(
        dataset, frac_train=0.8, frac_val=0.1,
        frac_test=0.1, random_state=42)
    train_loader = DataLoader(dataset=train_set, batch_size=bsz, shuffle=True, num_workers=1,
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set, batch_size=bsz, shuffle=False, num_workers=1, collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=test_set, batch_size=bsz, shuffle=False, num_workers=1,
                             collate_fn=collate_molgraphs)

    loss_fn = nn.MSELoss(reduction='none')
    # loss_fn = nn.L1Loss(reduction='none')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

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

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=0.000000001)
    stopper = EarlyStopping(mode='lower', filename='out/qm9_graphormer_pretrain.pth', patience=80)

    epoch = 0
    while True:
        model.train()
        train_loss_G = 0.0
        t0 = time.time()
        for batch_id, batch_data in enumerate(train_loader):
            mol, bg, b_G, pos, b_dist, b_angle, b_edge_index, b_idx_i, b_idx_j, b_idx_k = batch_data
            bg = bg.to(device)
            b_G = b_G.to(device)

            pos = pos.to(device)

            atom_feats, bond_feats = bg.ndata.pop('hv'), bg.edata.pop('he')
            atom_feats = atom_feats + torch.from_numpy(np.random.normal(mu, sigma, atom_feats.shape)).to(device)
            atom_feats = backend.pad_packed_tensor(atom_feats.float(), bg.batch_num_nodes(), 0)
            attn_bias = torch.zeros([atom_feats.shape[0], atom_feats.shape[1], atom_feats.shape[1]],
                                    dtype=torch.float).to(device)

            k, G_prediction = model.forward(bg, atom_feats, attn_bias, bond_feats, perturb=None)  # 1024,9,17

            loss = (loss_fn(G_prediction[b_G != 0].flatten(), b_G[b_G != 0].flatten())).mean()

            train_loss_G = train_loss_G + loss.item()
            if batch_id % 50 == 0:
                print(loss)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
        train_time = time.time() - t0

        epoch += 1
        train_loss_G = train_loss_G / (batch_id + 1)
        val_pre_G, val_label_G, val_loss_G = run_an_eval_epoch(model, val_loader)
        test_pre_G, test_label_G, test_loss_G = run_an_eval_epoch(model, test_loader)
        #  print(get_all_metric(test_label, test_pre))
        #  all_metric=get_all_metric(train_label, train_pre)+get_all_metric(val_label, val_pre)+get_all_metric(test_label,test_pre)+train_loss+val_loss+test_loss
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
        torch.save(model.state_dict(), r'models\checkpoints_{}_{}.pt'.format(epoch, time.strftime("%Y%m%d%H", time.localtime())))
        if early_stop:
            break


