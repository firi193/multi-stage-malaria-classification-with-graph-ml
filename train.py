
from tensorflow.keras.preprocessing.image import ImageDataGenerator



import numpy as np
import os
import sys
sys.path.insert(0,'/multi-stage-malaria-graph-ml')
#import a scheduler too
import utils_fun
import config
import wholeneuralnet_s

from sklearn.cluster import KMeans
from sklearn import preprocessing
import math

datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

def generate_batch(source_data, target_data):
  #this iterates on the batches of the source_data and target_data
    source_iter = iter(source_data)
    target_iter = iter(target_data)
    #the source images to be put in an array?
    source_x1 = []
    #source labels
    source_labels1 = []
    #why a second one?
    source_x2 = []
    source_labels2 = []
    #target training images
    target_x = []
    target_labels = []
    #the input images and labels in the next batch?
    inputs, labels = next(source_iter)
    #append the inputs in this array
    source_x1.append(inputs)
    for label in labels:
        source_labels1.append(label.item())
    inputs, labels = next(source_iter)
    #then for the next batch, append in the second source and label list
    source_x2.append(inputs)
    for label in labels:
        source_labels2.append(label.item())
    #do the same for the target batch
    inputs, labels = next(target_iter)
    target_x.append(inputs)
    for label in labels:
        target_labels.append(label.item())
    #concatenate along axis-0 and reassign
    source_x1 = np.concatenate(source_x1, 0)
    source_x2 = np.concatenate(source_x2, 0)
    target_x = np.concatenate(target_x, 0)
    #change them into tensors??
    # source_labels1 = torch.tensor(source_labels1)
    # source_labels2 = torch.tensor(source_labels2)
    # target_labels = torch.tensor(target_labels)
    #what's the purpose of the code below??
    source_labels = (source_labels1.numpy() == source_labels2.numpy()).astype('float32')
    return source_x1, source_x2, target_x, source_labels1, source_labels2, target_labels, source_labels





source_data_loader = datagen.flow_from_directory('/multi-stage-malaria-graph-ml/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/train', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode='categorical', 
                                       batch_size=config.source_batch_size)
# load and iterate validation dataset
target_data_loader = datagen.flow_from_directory('/multi-stage-malaria-graph-ml/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/test', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode='categorical', 
                                      batch_size=config.target_batch_size)

test_data_loader = datagen.flow_from_directory('/multi-stage-malaria-graph-ml/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/test', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode='categorical', 
                                      batch_size=config.target_batch_size)

source_x1, source_x2, target_x, source_labels1, source_labels2, target_labels, source_labels = \
                generate_batch(source_data_loader, target_data_loader)

model()