# Gram Matrix: An Efficient Representation of Molecular Conformation and Learning Objective for Molecular Pre-training

![gram预训练模型总结](https://github.com/xiangwenkai/GRAM/assets/93317912/e8b3a482-c3ac-4003-8543-8c656087953c)


## Environment Setup
You can follow the command to create the conda environment:   
```
conda create --name GRAM --file requirements.txt
```
Or you can install the packages step by step. The following installation commands have been tested to be OK(2024/5/30):
```
conda create -n GRAM python==3.9
conda activate GRAM
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c dglteam/label/th21_cu121 dgl
conda install pydantic
pip install rdkit
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install scikit-learn
pip install dgllife
pip install MDAnalysis
```


# Usage  
## Prepare dataset  
### 1.GEOM dataset  
**step1** download data  
Download rdkit_folder.tar.gz from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF  


**step2** decompression the data  
```
tar -zcvf rdkit_folder.tar.gz ./data/geom_drugs
```


**step3** processing the dataset  
```
python prepare_dataset_drugs.py
```  

### 2.Fastsmcg dataset  
Source: https://github.com/wangzhehyd/fastsmcg/tree/main  
(dataset-1)  
All processed files are avaliable at "./data/fastsmcg/processed.zip", you need to unzip it first.    

### 3.Moleculenet dataset  
Source: https://moleculenet.org/datasets-1  
The raw dataset are alredy downloaded in data/moleculenet/

## Pre-Training
```shell
python graphormer_geom_pretrain.py
```  
## Pre-Training model for Conformer prediction
**step 1** Prepare dataset  
You can refer to the steps in "Prepare dataset" to prepare your dataset  

**step 2** 3D structure prediction  
You should change the file path prepared first (example: the fastsmcg dataset path "./data/fastsmcg/processed")  
Then run:  
```shell
python 3d_prediction.py --path ./data/fastsmcg/processed/* --device cuda --batch_size 16
```  
The RMSD will be printed when it finished.  

## property prediction model  
### 1.Prepare data
```
python prepare_moleculenet.py  
```
### 2.Training (take Sider dataset for example)  
```
python finetune_graphormer_sider.py
```

The trained models are avaliable at https://drive.google.com/drive/folders/1S9IzlOthWOiC5E9jxLgZALka7HdgNO90?usp=sharing



