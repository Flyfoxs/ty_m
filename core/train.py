from __future__ import division
import numpy as np
import os
import glob

from random import *
from PIL import Image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop

pd.read_csv('./input/train_face_value_label.csv', names=['name','label'], skiprows=1).label.value_counts()

sample = pd.read_csv('./input/train_face_value_label.csv', names=['name','label'], skiprows=1)#.head()
sample['label_cat'] = sample.label.astype('category').cat.codes.astype(int)
sample.label = sample.label.astype('str')
sample.head()

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
for train_index, valid_index in folds.split(sample, sample.label_cat.values):
    train_df = sample.iloc[train_index]
    valid_df = sample.iloc[valid_index]
    break

train_df.dtypes

"""
This script goes along my blog post:
Keras InceptionResetV2 (https://jkjung-avt.github.io/keras-inceptionresnetv2/)
"""


from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


DATASET_PATH  = './input/train_data'
IMAGE_SIZE    = (299, 299)
NUM_CLASSES   = 9
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 20
WEIGHTS_FINAL = 'model-inception_resnet_v2-final.h5'




train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
#                                                   target_size=IMAGE_SIZE,
#                                                   interpolation='bicubic',
#                                                   class_mode='categorical',
#                                                   shuffle=True,
#                                                   batch_size=BATCH_SIZE)


train_batches = train_datagen.flow_from_dataframe(train_df, directory=DATASET_PATH, x_col='name',
                    y_col='label', target_size=IMAGE_SIZE,
                    color_mode='rgb', classes=None,
                    class_mode='categorical', batch_size=32,
                    shuffle=True, seed=None, save_to_dir=None,
                    save_prefix='', save_format='png', subset=None,
                    interpolation='nearest', drop_duplicates=True)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
#                     train_batches                              target_size=IMAGE_SIZE,
#                                                   interpolation='bicubic',
#                                                   class_mode='categorical',
#                                                   shuffle=False,
#                                                   batch_size=BATCH_SIZE)

valid_batches = valid_datagen.flow_from_dataframe(valid_df, directory=DATASET_PATH, x_col='name',
                    y_col='label', target_size=IMAGE_SIZE,
                    color_mode='rgb', classes=None,
                    class_mode='categorical', batch_size=32,
                    shuffle=True, seed=None, save_to_dir=None,
                    save_prefix='', save_format='png', subset=None,
                    interpolation='nearest', drop_duplicates=True)

# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')

# build our classifier model based on pre-trained InceptionResNetV2:
# 1. we don't include the top (fully connected) layers of InceptionResNetV2
# 2. we add a DropOut layer followed by a Dense (fully connected)
#    layer which generates softmax class score for each class
# 3. we compile the final model using an Adam optimizer, with a
#    low learning rate (since we are 'fine-tuning')
net = InceptionResNetV2(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
#print(net_final.summary())

# train the model
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)

# save trained weights
net_final.save(WEIGHTS_FINAL)


"""
nohup python -u  core/train.py &

"""