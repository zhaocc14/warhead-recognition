import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from loguru import logger
import sys
import scipy.io as scio
import os
from numpy.random import randint
from random import shuffle
'''
大型数据集
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

TrainDataPath = "..\\HRRP_Data\\train\\"
TestDataPath = "..\\HRRP_Data\\val\\"
EvalDataPath = '..\\HRRP_Data\\evaluate\\'


class RangeDopplerSequence(keras.utils.Sequence):
    def __init__(self, x_set, batch_size, Augment=False):
        self.x = x_set
        self.batch_size = batch_size
        self.Augment = Augment

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        # batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        output_x = []
        output_y = []
        for file_name in batch_x:
            mat_tmp = scio.loadmat(file_name)
            if self.Augment:
                index_begin = np.random.randint(29-13)
            else:
                index_begin = 0
            tmp = mat_tmp['HRRPmap'][index_begin:index_begin+13, :]
            tmp = tmp/np.max(tmp)
            output_x.append((tmp.reshape(13, 256, 1)))
            if mat_tmp['label'] == 0:
                output_y.append([1, 0, 0])
            elif mat_tmp['label'] == 1:
                output_y.append([0, 1, 0])
            else:
                output_y.append([0, 0, 1])
        output_x = np.array(output_x)
        output_y = np.array(output_y)

        return output_x, output_y


def getTimeString(t):
    YY = str(t[0])[2:]
    MM = str(t[1]) if t[1] >= 10 else '0' + str(t[1])
    DD = str(t[2]) if t[2] >= 10 else '0' + str(t[2])
    hh = str(t[3]) if t[3] >= 10 else '0' + str(t[3])
    mm = str(t[4]) if t[4] >= 10 else '0' + str(t[4])
    return YY + MM + DD + hh + mm


class EvalTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = model.evaluate_generator(
            generator=RangeDopplerSequence(eval_list, 40))
        logs.update({'eval_loss': loss, 'eval_acc': acc})
        super().on_epoch_end(epoch, logs)
        print('\n eval loss:{:2f},eval acc:{:.2f}'.format(loss, acc))


if __name__ == '__main__':
    timestamp = time.localtime()

    len_video = 13
    res_r = 256
    res_c = 256

    logger.add('log.txt',
               format='{time:YYYY-MM-DD at hh:mm:ss} | {level} | {message}')

    initialiser = 'glorot_uniform'
    reg_lambda = 0.001
    HRRP_input = keras.layers.Input(shape=(len_video, res_r, 1),
                                     dtype='float32',
                                     name='HRRP_input')
    # normalizayion_input = keras.layers.Input(shape=(len_video,1),dtype='float32',name='normalization_input')

    pl = keras.layers.Conv2D(
        32,
        (3, 5),
        strides=(1, 2),
        # activation='relu',
        data_format='channels_last',
        padding='same',
        kernel_initializer=initialiser,
        kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
        name='Conv1')(HRRP_input)
    k = 32
    for i_block in range(5):
        resblock_1_1 = keras.layers.BatchNormalization(name=str(i_block) +
                                                       'res_BN1')(pl)
        resblock_1_2 = keras.layers.Activation('relu',
                                               name=str(i_block) +
                                               'res_Relu1')(resblock_1_1)
        resblock_1_3 = keras.layers.Conv2D(
            k, (3, 3),
            strides=(1, 1),
            # activation='relu',
            data_format='channels_last',
            padding='same',
            kernel_initializer=initialiser,
            kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
            name=str(i_block) + 'res_Conv1')(resblock_1_2)
        resblock_2_1 = keras.layers.BatchNormalization(name=str(i_block) +
                                                       'res_BN2')(resblock_1_3)
        resblock_2_2 = keras.layers.Activation('relu',
                                               name=str(i_block) +
                                               'res_Relu2')(resblock_2_1)
        resblock_2_3 = keras.layers.Conv2D(
            k, (3, 3),
            strides=(1, 1),
            # activation='relu',
            data_format='channels_last',
            padding='same',
            kernel_initializer=initialiser,
            kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
            name=str(i_block) + 'res_Conv2')(resblock_2_2)
        resblock_add = keras.layers.Add(name=str(i_block) +
                                        'res_Add')([pl, resblock_2_3])
        if i_block%2==0:
            pl = keras.layers.MaxPool2D((2, 2),
                                        strides=(2, 2),
                                        name=str(i_block) + 'Pool')(resblock_add)
        else :
            k *= 2
            pl = keras.layers.Conv2D(
                k,(3,3),
                strides=(1,2),
                data_format='channels_last',
                padding='same',
                kernel_initializer=initialiser,
                kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                name=str(i_block) + 'ConvPool'
            )(resblock_add)


    flatten = keras.layers.Flatten()(pl)
    dp = keras.layers.Dropout(0.5)(flatten)
    dense = keras.layers.Dense(64, activation='relu')(dp)
    output = keras.layers.Dense(3, activation='softmax')(dense)

    model = keras.models.Model(inputs=HRRP_input, outputs=output)
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  patience=8,
                                                  mode='auto')
    stop_val = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=20,
                                             verbose=0,
                                             mode='auto')
    modelname = 'model_HRRP' + getTimeString(timestamp)
    check_point = keras.callbacks.ModelCheckpoint('model\\' + modelname +
                                                  '.h5',
                                                  monitor='val_acc',
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  verbose=1)
    tensorboard = EvalTensorBoard(log_dir='.\\logs\\log' +
                                              getTimeString(timestamp),
                                              batch_size=30,
                                              write_images=True)

    train_list = [
        TrainDataPath + name for name in os.listdir(TrainDataPath)
        if name.endswith('.mat')
    ]
    test_list = [
        TestDataPath + name for name in os.listdir(TestDataPath)
        if name.endswith('.mat')
    ]
    eval_list = [
        EvalDataPath + name for name in os.listdir(EvalDataPath)
        if name.endswith('.mat')
    ]
	
    shuffle(train_list)
    shuffle(test_list)
    shuffle(eval_list)


    history = model.fit_generator(
        generator=RangeDopplerSequence(train_list, 30, True),
        epochs=150,
        validation_data=RangeDopplerSequence(test_list, 30),
        callbacks=[
            reduce_lr, stop_val, tensorboard, check_point
        ])


    logger.info(modelname + ' is create by ' +
                sys.argv[0])
    logger.info('HRRP, 3 classes')
    logger.info('-' * 49)
