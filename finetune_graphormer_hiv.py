import torch
from model import Graphormer, Graphormer_finetune
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
    roc_auc = roc_auc_score(y_true, y_pred)
    return roc_auc


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
        tasks = tasks.long().to(device)

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

        pred_ = F.softmax(b_prediction, dim=-1).detach().cpu()[:, 0]
        try:
            loss = loss_fn(b_prediction.squeeze(), tasks.flatten()).mean().item()
        except:
            loss = loss_fn(b_prediction.squeeze().unsqueeze(), tasks.flatten()).mean().item()
        train_loss = (train_loss + loss)

        all_pre.extend(pred_.numpy())
        all_label.extend(tasks.flatten().numpy())

        train_loss = (train_loss + loss)

    train_loss = train_loss / (batch_id + 1)
    return all_pre, all_label, train_loss


name = 'hiv'
pretrain_feature = True
if pretrain_feature:
    pretrain_flag = ''
else:
    pretrain_flag = 'no3d'

metric_name = ['epoch', 'val_score', 'test_score', 'train_loss', 'val_loss', 'test_loss']
out = open("./out/graphormer_{}_finetune{}.txt".format(name, pretrain_flag), "a+")
out.write(",".join(metric_name)+"\n")

# log = open("/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/log.txt", "w+")
log = open("log_{}.txt".format(name), "w+")

train_sdf_paths = "./data/{}/train/*".format(name)
val_sdf_paths = "./data/{}/val/*".format(name)
test_sdf_paths = "./data/{}/test/*".format(name)

train_sdfs = glob.glob(train_sdf_paths)
val_sdfs = glob.glob(val_sdf_paths)
test_sdfs = glob.glob(test_sdf_paths)
train_set = InteractionDataset_moleculenet(train_sdfs, load=True, n_jobs=10)
val_set = InteractionDataset_moleculenet(val_sdfs, load=True, n_jobs=10)
test_set = InteractionDataset_moleculenet(test_sdfs, load=True, n_jobs=10)

num_samples = len(train_sdfs)
print("train samples: {}\n".format(num_samples))

bsz = 16
train_loader = DataLoader(dataset=train_set, batch_size=bsz, shuffle=True, num_workers=10, collate_fn=collate_molgraphs_moleculenet)
val_loader = DataLoader(dataset=val_set, batch_size=bsz, shuffle=False, num_workers=10, collate_fn=collate_molgraphs_moleculenet)
test_loader = DataLoader(dataset=test_set, batch_size=bsz, shuffle=False, num_workers=10, collate_fn=collate_molgraphs_moleculenet)

mu = 0.0
sigma = 0.2

loss_fn = nn.MSELoss(reduction='none')

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
    n_tasks=2,
)

model_finetune.to(device)

positive, negative = 0, 0
for data in train_set:
    g,task = data
    if task == 1:
        positive = positive + 1
    else:
        negative = negative + 1
for data in test_set:
    g,task = data
    if task == 1:
        positive = positive + 1
    else:
        negative = negative + 1
for data in val_set:
    g,task = data
    if task == 1:
        positive = positive + 1
    else:
        negative = negative + 1
weight = [(positive+negative)/negative, (positive+negative)/positive]

loss_fn = nn.CrossEntropyLoss(torch.Tensor(weight).to(device), reduction='mean')

optimizer = torch.optim.Adam(model_finetune.parameters(), lr=0.0001,
                             weight_decay=0.000000001)
stopper = EarlyStopping(mode='higher', filename='./out/graphormer_{}_finetune.pth'.format(name), patience=80)
# stopper = EarlyStopping(mode='higher', filename='models/graphormer_{}_finetune.pth'.format(name), patience=80)
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
            tasks = tasks.to(device)
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

            #pred_ = F.softmax(tasks_prediction, dim=-1)[:, 0]
            loss = loss_fn(tasks_prediction.squeeze(), tasks.flatten()).mean().float()

            train_loss += loss.item()
            if batch_id % 20 == 0:
                # print(loss)
                log.write("epoch:{}  iter: {}/{} --- loss: {}\n".format(epoch, batch_id, int(num_samples/bsz), train_loss/(batch_id + 1)))
                log.flush()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_time = time.time() - t0
        epoch += 1
        val_pre, val_label, val_loss = run_an_eval_epoch(model_finetune, model_pretrain, val_loader)
        test_pre, test_label, test_loss = run_an_eval_epoch(model_finetune, model_pretrain, test_loader)
        val_roc_auc = get_all_metric(val_label, val_pre)
        test_roc_auc = get_all_metric(test_label, test_pre)
        all_metric = [str(epoch)] + [str(val_roc_auc)] + [str(test_roc_auc)] + [
            str(train_loss)] + [str(val_loss)] + [str(test_loss)]
        val_score = val_roc_auc
        out.write(",".join(all_metric) + "\n")
        out.flush()

        early_stop = stopper.step(val_score, model_finetune)
        if epoch > 1 and val_score==stopper.best_score and pretrain_feature:
             torch.save(model_finetune.state_dict(), './models/model_{}/checkpoints_{}.pt'.format(name, epoch))
        print('epoch {:d}/{:d}, validation {} {:.3f}, best validation {} {:.3f}, epoch train time: {:.1f}'.format(
            epoch, 1000, 'auc',
            val_score, 'auc', stopper.best_score, train_time))

        if early_stop:
            break

