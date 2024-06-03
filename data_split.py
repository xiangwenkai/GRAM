import subprocess
import os
import random
import pickle
import torch

for split in ['train', 'val', 'test']:
    path = './data/processed/{}'.format(split)
    files = os.listdir(path)
    ids = [x.split('_')[0] for x in files]
    ids = list(set(ids))
    name_dict = {}
    for i in ids:
        name_dict[i] = random.random()

    for name in files:
        name_id = name.split('_')[0]
        [mol, g, G, pos, dist, angle, edge_index, idx_i, idx_j, idx_k] = pickle.load(open(path+'/{}'.format(name), 'rb'))

        if name_dict[name_id] <= 0.8:
            destination_file = "./data/processed/train/{}".format(name)
        elif name_dict[name_id] <= 0.9:
            destination_file = "./data/processed/val/{}".format(name)
        else:
            destination_file = "./data/processed/test/{}".format(name)
        with open(destination_file, "wb") as f:
            pickle.dump((mol, g, G, pos, dist, angle, edge_index, idx_i, idx_j, idx_k), f)



