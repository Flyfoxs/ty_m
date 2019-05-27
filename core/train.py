from __future__ import division

import os

import pandas as pd

pd.read_csv('./input/train_face_value_label.csv', names=['name','label'], skiprows=1).label.value_counts()

sample = pd.read_csv('./input/train_face_value_label.csv', names=['name','label'], skiprows=1)#.head()
sample['label_cat'] = sample.label.astype('category').cat.codes.astype(int)
sample.label = sample.label.astype('str')
sample.head()

from sklearn.model_selection import StratifiedKFold

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

def train_exist(mode_path='./output/model-inception_resnet_v1-25.h5'):
    from tensorflow.keras.models import load_model
    model = load_model(mode_path)
    gen_sub(model, testdf, sn=999)

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
    for i in range(10):
        net_final.fit_generator(train_batches,
                                steps_per_epoch = train_batches.samples // BATCH_SIZE//10,
                                validation_data = valid_batches,
                                validation_steps = valid_batches.samples // BATCH_SIZE,
                                epochs = 1)

        gen_sub(net_final, testdf, sn=i)

        WEIGHTS_FINAL = f'./output/model-inception_resnet_v{i}-27.h5'

        # save trained weights
        net_final.save(WEIGHTS_FINAL)
        print(f'weight save to {WEIGHTS_FINAL}')


if __name__ == '__main__':
    import fire
    fire.Fire()




"""
nohup python -u  core/train.py > check_27.log 2>&1 &
"""
