import torch
from model import Graphormer, Graphormer_finetune, Graphormer_finetune_regression, Graphormer_all
from dgl import backend
import torch.nn as nn
import datetime
# import bcolors
import os
import numpy as np
from dgl.batch import unbatch
import glob
from sklearn.metrics import r2_score  # noqa
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from dgllife.utils import EarlyStopping, Meter, RandomSplitter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from interaction_dataset_bond_length_angle import InteractionDataset_moleculenet
from utils_interaction_length_angle_torch import set_random_seed, collate_molgraphs_moleculenet
import time
import random
import pickle
from torch.cuda.amp import GradScaler
random.seed(42)


def get_all_metric(y_true, y_pred):
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    return RMSE, MAE


def run_an_eval_epoch(model, model_3D, val_loader):
    model.eval()
    if pretrain_feature:
        model_3D.eval()
    all_pre = []
    all_label = []
    train_loss = 0.0
    for batch_id, batch_data in enumerate(val_loader):
        bg, tasks = batch_data
        bg = bg.to(device)
        tasks = tasks.to(torch.float32).to(device)

        atom_feats, bond_feats = bg.ndata.pop('hv'), bg.edata.pop('he')
        atom_feats = backend.pad_packed_tensor(atom_feats.float(), bg.batch_num_nodes(), 0)

        attn_bias = torch.zeros([atom_feats.shape[0], atom_feats.shape[1] + 1, atom_feats.shape[1] + 1],
                                dtype=torch.float).to(device)
        if pretrain_feature:
            attn_bias_pre = torch.zeros([atom_feats.shape[0], atom_feats.shape[1], atom_feats.shape[1]],
                                        dtype=torch.float).to(device)
            atom_3D_feats = model_3D.get_feature(bg, atom_feats, attn_bias_pre, bond_feats)
        else:
            atom_3D_feats = None
        b_prediction = model(bg, atom_feats, attn_bias, bond_feats, atom_3D_feats)

        loss = loss_fn(b_prediction, tasks).mean().item()
        train_loss = (train_loss + loss)

        all_pre.extend(b_prediction.detach().cpu().numpy().tolist())
        all_label.extend(tasks.detach().cpu().numpy().tolist())

        train_loss = (train_loss + loss)

    train_loss = train_loss / (batch_id + 1)
    return all_pre, all_label, train_loss


name = 'lipophilicity'
pretrain_feature = False
if pretrain_feature:
    pretrain_flag = ''
else:
    pretrain_flag = 'no3d'

metric_name = ['epoch', 'val_rmse', 'val_mae', 'test_rmse', 'test_mae', 'train_loss', 'val_loss', 'test_loss']
out = open("./out/graphormer_{}_finetune{}.txt".format(name, pretrain_flag), "a+")
out.write(",".join(metric_name)+"\n")

train_sdf_paths = "./data/{}/train/*".format(name)
val_sdf_paths = "./data/{}/val/*".format(name)
test_sdf_paths = "./data/{}/test/*".format(name)

train_sdfs = glob.glob(train_sdf_paths)
val_sdfs = glob.glob(val_sdf_paths)
test_sdfs = glob.glob(test_sdf_paths)
train_set = InteractionDataset_moleculenet(train_sdfs, load=True, n_jobs=1)
val_set = InteractionDataset_moleculenet(val_sdfs, load=True, n_jobs=1)
test_set = InteractionDataset_moleculenet(test_sdfs, load=True, n_jobs=1)

num_samples = len(train_sdfs)
# print("train samples: {}\n".format(num_samples))

bsz = 16
train_loader = DataLoader(dataset=train_set, batch_size=bsz, shuffle=True, num_workers=1, collate_fn=collate_molgraphs_moleculenet)
val_loader = DataLoader(dataset=val_set, batch_size=bsz, shuffle=False, num_workers=1, collate_fn=collate_molgraphs_moleculenet)
test_loader = DataLoader(dataset=test_set, batch_size=bsz, shuffle=False, num_workers=1, collate_fn=collate_molgraphs_moleculenet)

mu = 0.0
sigma = 0.2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

if pretrain_feature:
    model_pretrain = Graphormer(
        n_layers=6,
        num_heads=8,
        hidden_dim=512,
        sp_num_heads=8,
        dropout_rate=0.1,
        intput_dropout_rate=0.1,
        ffn_dim=512,
        attention_dropout_rate=0.1
    )

    PATH = "./models/pretraining_checkpoints_best.pt"

    model_pretrain.load_state_dict(torch.load(PATH))

    model_pretrain.to(device)
else:
    model_pretrain = None

model_finetune = Graphormer_finetune(
    n_layers=5,
    num_heads=8,
    hidden_dim=512,
    sp_num_heads=8,
    dropout_rate=0.1,
    intput_dropout_rate=0.1,
    ffn_dim=512,
    attention_dropout_rate=0.1,
    n_tasks=1,
)

model_finetune.to(device)

loss_fn = nn.MSELoss(reduction='none')

optimizer = torch.optim.Adam(model_finetune.parameters(), lr=0.0001,
                             weight_decay=0.00000001)
stopper = EarlyStopping(mode='lower', filename='./out/graphormer_{}_finetune.pth'.format(name), patience=80)
scaler = GradScaler()

if __name__ == '__main__':
    epoch = 0
    while True:
        model_finetune.train()
        train_loss = 0.0
        t0 = time.time()
        for batch_id, batch_data in enumerate(train_loader):
            # if batch_id < 1:
            #     continue
            bg, tasks = batch_data
            bg = bg.to(device)
            tasks = tasks.to(torch.float32).to(device)
            atom_feats, bond_feats = bg.ndata.pop('hv'), bg.edata.pop('he')
            atom_feats = atom_feats + torch.from_numpy(np.random.normal(mu, sigma, atom_feats.shape)).to(device)
            atom_feats = backend.pad_packed_tensor(atom_feats.float(), bg.batch_num_nodes(), 0)

            attn_bias = torch.zeros([atom_feats.shape[0], atom_feats.shape[1] + 1, atom_feats.shape[1] + 1],
                                    dtype=torch.float).to(device)
            if pretrain_feature:
                attn_bias_pre = torch.zeros([atom_feats.shape[0], atom_feats.shape[1], atom_feats.shape[1]],
                                            dtype=torch.float).to(device)
                with torch.no_grad():
                    atom_3D_feats = model_pretrain.get_feature(bg, atom_feats, attn_bias_pre, bond_feats)
            else:
                atom_3D_feats = None
            tasks_prediction = model_finetune(bg, atom_feats, attn_bias, bond_feats, atom_3D_feats)

            loss = loss_fn(tasks_prediction, tasks).mean().float()

            train_loss += loss.item()
            if batch_id % 10 == 0:
                print(f"Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_time = time.time() - t0
        epoch += 1
        train_pre, train_label, train_loss = run_an_eval_epoch(model_finetune, model_pretrain, train_loader)
        val_pre, val_label, val_loss = run_an_eval_epoch(model_finetune, model_pretrain, val_loader)
        test_pre, test_label, test_loss = run_an_eval_epoch(model_finetune, model_pretrain, test_loader)
        train_rmse, train_mae = get_all_metric(train_label, train_pre)
        val_rmse, val_mae = get_all_metric(val_label, val_pre)
        test_rmse, test_mae = get_all_metric(test_label, test_pre)
        all_metric = [str(epoch)] + [str(val_rmse)] + [str(val_mae)] + [str(test_rmse)] + [str(test_mae)] + [str(train_loss)] + [str(val_loss)] + [str(test_loss)]
        print("epoch: {}; train rmse: {}; val rmse: {}; test rmse: {}".format(epoch, round(train_rmse, 3), round(val_rmse, 3), round(test_rmse, 3)))
        val_score = np.array(val_rmse).mean()
        out.write(",".join(all_metric) + "\n")
        out.flush()

        early_stop = stopper.step(val_score, model_finetune)
        if epoch > 10 and val_score==stopper.best_score and pretrain_feature:
            torch.save(model_finetune.state_dict(), './models/model_{}/checkpoints_{}.pt'.format(name, epoch))

        print('epoch {:d}/{:d}, validation {} {:.3f}, best validation {} {:.3f}, epoch train time: {:.1f}'.format(
            epoch, 1000, 'rmse',
            val_score, 'rmse', stopper.best_score, train_time))

        if early_stop:
            break

