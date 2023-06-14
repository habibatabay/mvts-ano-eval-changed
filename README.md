
# Semi-Supervised Skeleton-Based Anomaly Detection in Electronic-Exam Videos a

## Original Authors:
A. Garg, W. Zhang, J. Samaran, R. Savitha and C. -S. Foo, "An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2021.3105827.
https://ieeexplore.ieee.org/document/9525836 Arxiv: https://arxiv.org/abs/2109.11428

### Related Publication:
Agh Atabay, Habibollah and Hassanpour, Hamid, Semi-Supervised Skeleton-Based Anomaly Detection in Electronic-Exam Videos. Unpublished 


## Used Algorithms

### Univar Auto-encoder
Channel-wise auto-encoders for each channel. 

### Autoencoder
Hawkins, Simon et al. "Outlier detection using replicator neural networks." DaWaK, 2002.

### LSTM-ED
Malhotra, Pankaj et al. "LSTM-based encoder-decoder for multi-sensor anomaly detection." ICML, 2016.

### TCN-ED
Based on the TCN benchmarked by Bai, Shaojie et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arxiv, 2018

### LSTM VAE
Based on Park, Daehyung, Yuuna Hoshi, and Charles C. Kemp. "A multimodal anomaly detector for robot-assisted feeding using an LSTM-based variational autoencoder." IEEE Robotics and Automation Letters 3.3 (2018): 1544-1551.

### Omni-anomaly
Su, Ya, et al. "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.

### MSCRED
Zhang, Chuxu, et al. "A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data." Proceedings of the AAAI Conference on Artificial Intelligence. 2019.

## Training schemes:
### Global
Training on the first 30% of all videos and testing on the collected dataset
### Video-specific
Training on the first 30% of each video and testing on the remaining part of the same video
### Combined
Combination of two methods by averaging the scores
## Data sets and how to get them

### edBB
Located in /data/edBB
### My collected dataset: the preliminary version
Located in /data/MyDataset 
### My collected dataset: the complete version
Located in /data/CombinedDataset 

Skeleton features are extracted by Tensorflow Movnet and COCO structure

Data is 2D joint coordinates of skeletons extracted from e-exam videos

## Scripts:
### my_experiments_mvts.py
Contains my codes to train and test the methods
### my_data_functions.py
Contains my codes to read data from files
### my_extract_features.py
Contains my codes to extract features from the original features

The "tested_algorithms" branch is the updated one


