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
## Implementation
## Preparing dataset

