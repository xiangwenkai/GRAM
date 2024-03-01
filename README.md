# Gram Matrix: An Efficient Representation of Molecular Conformation and Learning Objective for Molecular Pre-training

![gram预训练模型总结](https://github.com/xiangwenkai/GRAM/assets/93317912/e8b3a482-c3ac-4003-8543-8c656087953c)


## Environment Setup

### windows:
```shell
conda create -n dgl python==3.8
conda activate dgl
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 
pip install https://data.dgl.ai/wheels/cu117/dgl-1.1.1%2Bcu117-cp38-cp38-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_sparse-0.6.17%2Bpt20cu117-cp38-cp38-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_scatter-2.1.1%2Bpt20cu117-cp38-cp38-win_amd64.whl
pip install dgllife
pip install scikit-learn
pip install rdkit
```

### Linux：
```shell
conda create -n dgl python==3.8
conda activate dgl
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install  dgl -f https://data.dgl.ai/wheels/cu113/repo.html or pip install https://data.dgl.ai/wheels/cu113/dgl-1.1.1%2Bcu113-cp38-cp38-manylinux1_x86_64.whl
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.16%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install rdkit
pip install dgllife
```

## Usage
### training
```shell
python train.py
```
### inference
```shell
python prepare_custom_data.py
python inference.py
```
