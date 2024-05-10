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
pip install --upgrade MDAnalysis
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
pip install --upgrade MDAnalysis
```

## Usage
### Prepare dataset(GEOM)  
**step1** download data  
Download rdkit_folder.tar.gz from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF  
**step2** decompression the data  
```
tar -zcvf rdkit_folder.tar.gz geom_drugs/data/
```  
**step3** processing the dataset  
```
python geom_drugs/prepare_dataset_drugs.py
```

### Prepare dataset(fastsmcg)  
source: https://github.com/wangzhehyd/fastsmcg/tree/main  
all processed files are avaliable at ./geom_drugs/data/fastsmcg/processed  

### training
```shell
python geom_drugs/graphormer_geom_pretrain.py
```
### inference
1.Prepare dataset
You can refer to the steps in "Prepare dataset" to prepare your dataset

2.3D structure prediction 
You should change the file path prepared first (example: the fastsmcg dataset path "./geom_drugs/data/fastsmcg/processed")  
Then run:  
```shell
python geom_drugs/3d_prediction.py
```
The RMSD will be printed when it finished.

