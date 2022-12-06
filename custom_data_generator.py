# -*- coding: utf-8 -*-



import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator

from keras.preprocessing.image import ImageDataGenerator

imageGen = ImageDataGenerator()

source_augmentor=ImageDataGenerator(rotation_range = 5, shear_range = 0.02,zoom_range = 0.02, samplewise_center=True, samplewise_std_normalization= True)

class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""
    def __init__(self, x, y, datagen, batch_size=30, shuffle=True):
        self.datagen = datagen#will augment the images
        self.batch_size = batch_size#should I be defining this in my code
        datagen.fit(x)#augment the training data
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(sampling_strategy={0:1500, 1:1500, 2:2600, 3:2500, 4:1300, 5:2000}), batch_size=self.batch_size, keep_sparse=True)
        #what does the above code do
        self._shape = (self.steps_per_epoch * batch_size, *x.shape[1:])
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(4352)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

# train_ds = imageGen.flow_from_directory(
#     '/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/train',
#     color_mode="rgb",
#     # labels='inferred',
#     class_mode='categorical',
#     batch_size=30,
#     target_size=(224, 224))

malaria_images_path = '/multi-stage-malaria-graph-ml/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/train' # Dataset contains folders Parasitized and Uninfected
Train_0_path = malaria_images_path + '/gametocyte/'
Train_1_path = malaria_images_path + '/leukocyte/'
Train_2_path = malaria_images_path + '/red_blood_cell/'
Train_3_path = malaria_images_path + '/ring/'
Train_4_path = malaria_images_path + '/schizont/'
Train_5_path = malaria_images_path + '/trophozoite/'

last = Train_4_path.split("/")[-2]
last

import os
p0_folder_T = os.listdir(Train_0_path)
p1_folder_T = os.listdir(Train_1_path)
p2_folder_T = os.listdir(Train_2_path)
p3_folder_T = os.listdir(Train_3_path)
p4_folder_T = os.listdir(Train_4_path)
p5_folder_T = os.listdir(Train_5_path)

dictt={'gametocyte':0, 'leukocyte':1, 'red_blood_cell':2, 'ring':3, 'schizont':4, 'trophozoite':5}

X = []
Y = []
dim = (224, 224)

def load_images(path_folder,train_path):
  for image in path_folder:
      try:
          image = image_utils.load_img(train_path + image, color_mode="rgb", target_size=dim)
          #image = cv2.resize(image, dim)
          image = image_utils.img_to_array(image)
          X.append(image)
          Y.append(dictt[train_path.split("/")[-2]])
      except:
          continue

def load_all_images():
  load_images(p0_folder_T,Train_0_path)
  load_images(p1_folder_T,Train_1_path)
  load_images(p2_folder_T,Train_2_path)
  load_images(p3_folder_T,Train_3_path)
  load_images(p4_folder_T,Train_4_path)
  load_images(p5_folder_T,Train_5_path)

# load_all_images()



# X=np.array(X)
# Y=np.array(Y)

# X=np.load('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/input_data.npy')
# Y=np.load('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/input_label.npy')

# Y.shape

# input_data=np.save('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/input_data.npy',X)
# input_label=np.save('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/input_label.npy',Y)

# source_balancer = BalancedDataGenerator(X, Y, source_augmentor, batch_size=30)