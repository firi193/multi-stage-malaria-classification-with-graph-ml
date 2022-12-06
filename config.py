# -*- coding: utf-8 -*-


import os
#import torchvision.transforms as transforms

source_path = r'/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/train'
target_path = r'/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/test'


class_num = 6

# resnet
source_batch_size = 30
target_batch_size = 10

# GCN_model
features_dim_num = 2048
GCN_hidderlayer_dim_num = 512

features_dim = 1024

epoches = 500
k = 10

#test and train transformation code