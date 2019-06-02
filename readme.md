
```python
import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from bokeh.palettes import Category10


from tqdm import tqdm, tqdm_notebook

file_folder = globals()['_dh'][0]
wk_dir = os.path.dirname(file_folder)
os.chdir(wk_dir)

%matplotlib inline



from glob import glob
import json

import matplotlib.pyplot as plt
import seaborn as sns


!hostname
```

    ai-prd-01



```python
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
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
```

    Using TensorFlow backend.


# 查看每种货币的样本数量是否倾斜


```python
pd.read_csv('./input/train_face_value_label.csv', names=['name','label'], skiprows=1).label.value_counts()

sample = pd.read_csv('./input/train_face_value_label.csv', names=['name','label'], skiprows=1)#.head()
sample['label_cat'] = sample.label.astype('category').cat.codes.astype(int)
sample.label = sample.label.astype('str')
sample.head()
sample.label_cat.value_counts().sort_index()
```




    0    4233
    1    4373
    2    4407
    3    4424
    4    4411
    5    4413
    6    4283
    7    4408
    8    4668
    Name: label_cat, dtype: int64



# 每种样本显示4张图片,看看图片质量


```python
import cv2
for label in range(9):
    row_num = 2
    fig,ax = plt.subplots(row_num,2, figsize=(18,4*row_num))
    sub_sample = sample.loc[sample.label_cat==label].sample(row_num * 2).reset_index()
    for i, file in sub_sample.name.iteritems():
        
        with open(f'./input/train_data/{file}' ,'rb') as f:
            img = Image.open(f)
            ax[i%row_num][i//row_num].title.set_text(file)
            ax[i%row_num][i//row_num].imshow(img)
             
            img = cv2.imread(f'./input/train_data/{file}')
            R, B, G = img[:,:,0].sum().sum(), img[:,:,1].sum().sum(), img[:,:,2].sum().sum()
            total = img.sum().sum().sum()
            R, B, G = round(R/total,4), round(B/total,4), round(G/total,4)
            print(file, img.shape, i%row_num, R, B, G)
    #fig.show()
    plt.show()

```

 

![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_1.png)
 
 



![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_3.png)

 



![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_5.png)

 



![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_7.png)

 



![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_9.png)

![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_11.png)

 



![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_13.png)

 
![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_15.png)

 
![image](https://github.com/Flyfoxs/ty_m/raw/master/img/output_5_17.png)


# EDA 结果
* 可以看出数据分布十分均匀, 图片数据也十分干净
* 应该不需要所有的数据投入训练, 可以成倍加快训练速度
* 为了增加数据的鲁棒性,增加了数据的左右切割,以及数据小角度旋转


```python
from __future__ import division

import os

import pandas as pd

pd.read_csv('./input/train_face_value_label.csv', names=['name','label'], skiprows=1).label.value_counts()

sample = pd.read_csv('./input/train_face_value_label.csv', names=['name','label'], skiprows=1)#.head()
sample['label_cat'] = sample.label.astype('category').cat.codes.astype(int)
sample.label = sample.label.astype('str')
sample.head()

from sklearn.model_selection import StratifiedKFold
#按照原始比例随机5折,20%保留验证 (其实traning和验证数据都不需要这么多,可以成倍提高性能)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
for train_index, valid_index in folds.split(sample, sample.label_cat.values):
    train_df = sample.iloc[train_index]
    valid_df = sample.iloc[valid_index]
    break

train_df.dtypes

"""
This script goes along my blog post:
Keras InceptionResetV2 (https://jkjung-avt.github.io/keras-inceptionresnetv2/)
https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
"""

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


#Traning 数据, 最多选择2度, 因为货币都是比较宽的长方形,所以对图片宽度随机裁剪70%, 高度随机高度裁剪10%, 
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=2,
                                   width_shift_range=0.7,
                                   height_shift_range=0.1,
                                   zoom_range=[0.9, 1.2],
                                   fill_mode='nearest'
                                   )



train_batches = train_datagen.flow_from_dataframe(train_df, directory=DATASET_PATH, x_col='name',
                    y_col='label', target_size=IMAGE_SIZE,
                    color_mode='rgb', classes=None,
                    class_mode='categorical', batch_size=32,
                    shuffle=True, seed=None, save_to_dir=None,
                    save_prefix='', save_format='png', subset=None,
                    interpolation='nearest', drop_duplicates=True)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_batches = valid_datagen.flow_from_dataframe(valid_df, directory=DATASET_PATH, x_col='name',
                    y_col='label', target_size=IMAGE_SIZE,
                    color_mode='rgb', classes=None,
                    class_mode='categorical', batch_size=32,
                    shuffle=True, seed=None, save_to_dir=None,
                    save_prefix='', save_format='png', subset=None,
                    interpolation='nearest', drop_duplicates=True)


test_dir = "./input/public_test_data/"
testdf = pd.DataFrame({'name':os.listdir(test_dir)})

test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=test_dir,
    x_col="name",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=IMAGE_SIZE )


def gen_sub(model, testdf, sn=0):
    """
    输入模型,得到提交文件
    """
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    print(f'{test_generator.n}, {test_generator.batch_size}, {STEP_SIZE_TEST}')
    pred = model.predict_generator(test_generator,
                                   steps=STEP_SIZE_TEST,
                                   verbose=1)
    df = pd.read_csv('./input/train_face_value_label.csv')
    print(df.columns)
    label_list = ['0.1', '0.2', '0.5', '1', '10', '100', '2', '5', '50']

    res = round(pd.DataFrame(pred, columns=label_list, index=testdf.name), 2)
    res['label'] = res.idxmax(axis=1)
    res[['label']].to_csv(f'./output/res_inception_res_{sn}.csv')

```

    Found 31692 images belonging to 9 classes.
    Found 7928 images belonging to 9 classes.
    Found 20000 images.



```python

def train_raw():
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
    # 每次迭代,都生成一个文件,可以上线测试,不用等太久就可以得到提交文件.如果性能稳定,到后面每次生成的文件会是一模一样.
    for i in range(2):
        net_final.fit_generator(train_batches,
                                steps_per_epoch = train_batches.samples // BATCH_SIZE//10,
                                validation_data = valid_batches,
                                validation_steps = valid_batches.samples // BATCH_SIZE//10,
                                epochs = 1)

        gen_sub(net_final, testdf, sn=i)

        WEIGHTS_FINAL = f'./output/model-inception_resnet_v{i}-27.h5'

        # save trained weights
        net_final.save(WEIGHTS_FINAL)
        print(f'weight save to {WEIGHTS_FINAL}')

```

# 开始训练模型,并生成提交文件


```python
#生成了2个提交文件,可以提交在线测试(在模型收敛的情况下,2个文件大概率是一模一样)
train_raw()
```

    ****************
    Class #0 = 0.1
    Class #1 = 0.2
    Class #2 = 0.5
    Class #3 = 1.0
    Class #4 = 10.0
    Class #5 = 100.0
    Class #6 = 2.0
    Class #7 = 5.0
    Class #8 = 50.0
    ****************
    Epoch 1/1
    396/396 [==============================] - 953s 2s/step - loss: 0.5867 - acc: 0.8205 - val_loss: 0.0167 - val_acc: 0.9991
    20000, 32, 625
    625/625 [==============================] - 1129s 2s/step
    Index(['name', ' label'], dtype='object')
    weight save to ./output/model-inception_resnet_v0-27.h5
    Epoch 1/1
    396/396 [==============================] - 1027s 3s/step - loss: 0.0288 - acc: 0.9957 - val_loss: 0.0118 - val_acc: 0.9991
    20000, 32, 625
    625/625 [==============================] - 1055s 2s/step
    Index(['name', ' label'], dtype='object')
    weight save to ./output/model-inception_resnet_v1-27.h5


