# PRiDAN : Person Re-identification from Drones with Adaptive Weights and Expanded Neighbourhood"
This repository contains the code for the paper "PRiDAN : Person Re-identification from Drones with Adaptive Weights and Expanded Neighbourhood"

The majority of the code is based on [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch), [Person-reID-triplet-loss](https://github.com/layumi/Person-reID-triplet-loss)

## Overview
This repository contains following resources
- Two traditional person re-id datasets: Market-1501 and DukeMTMC-reID
- An aerial person re-id dataset: PRAI-1581
- PRiDAN implementation

## Market-1501
You can download the Market-1501 dataset from [Google Drive](https://drive.google.com/file/d/1_KwUvfhI-6iqNj2ZUBDJYBEcWYK7gv0L/view?usp=sharing)

## DukeMTMC-reID
You can download the DukeMTMC-reID dataset from [Google Drive](https://drive.google.com/file/d/1_iqu_Q0GtKU7e3r1VjhpcbNfGffADxdU/view?usp=sharing)
## PRAI-1581
This is the link to the original creator of this dataset [link](https://github.com/stormyoung/PRAI-1581)

It contains 39461 images for 1580 IDs which are divided into:
- Training: 19523 images for 781 IDs 
- Testing: 19938 images for 799 IDs 
  - Query: 4680 images
  - Gallery: 15258 images
  
You can download the PRAI-1581 dataset from [Google Drive](https://drive.google.com/file/d/168UcmbW1twnq7F8BB_FtvPzCKiCLAMIp/view?usp=sharing).
I have converted the dataset into Market-1501 format, which contains the following directories:
```bash 
├── PRAI-1581/
│   ├── bounding_box_test/          /* Files for testing (images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── query/                      /* Files for testing (query images)
│   ├── partitions.pkl 
│   ├── readme.txt
```
or simply run the following code to fetch PRAI-1581 dataset and unzip it
```bash 
!wget -q https://www.dropbox.com/s/hc9tg34vxmb82pz/PRAI-1581_train_test_query-Market-1501_format.zip?dl=0
!unzip -q PRAI-1581_train_test_query-Market-1501_format.zip?dl=0
```
# Implementation
Clone the repository to get started
```bash 
!git clone https://github.com/Chatchanan-V/PRiDAN.git
cd PRiDAN
```
### Preparing dataset
run the following to prepare the dataset 
```bash 
!python prepare_dataset.py
```
The script creates new folder named pytorch in PRAI-1581 folder:
```bash 
├── PRAI-1581/
│   ├── bounding_box_test/        
│   ├── bounding_box_train/     
│   ├── pytorch/  
│   │   ├── gallery/   
│   │   ├── multi-query/   
│   │   ├── query/   
│   │   ├── train/   
│   │   ├── train_all/   
│   │   ├── val/   
│   ├── query/                     
│   ├── partitions.pkl 
│   ├── readme.txt
```

### Training
Install and import torch_snippets, torch_summary, and other required libraries.
```bash 
!pip install -q torch_snippets
import torch_snippets
from torch_snippets import *
import os 
from shutil import copyfile
device = 'cuda' if torch.cuda.is_available() else 'cpu'

!pip install torch_summary
from torchsummary import summary
```

create new folder named model for storing files related to the training phase
```bash
os.mkdir('model')
```

Train the model
```bash
!python train.py --gpu_ids 0 --name PCB --PCB --train_all --batchsize 16 --margin 0.3 --lr 0.01 --alpha 0.0 --data_dir ../PRAI-1581/pytorch
```

You may need to change the batchsize if GPU memory is not enough

I use n = 3 for the number of negative samples used in adaptive weight scheme and p = 8 for the number of parts used in PCB method, as it has been shown to achieve the optimal result.

### Testing


### Re-ranking result
![Compare_3_ranking_list_2](https://user-images.githubusercontent.com/94464876/149622359-cec64dd0-8ca9-4ede-bf54-268aedb89d3a.png)

### Visualization

## Reference

