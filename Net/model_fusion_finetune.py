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
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers.core import Lambda
'''
大型数据集
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

HRRPTrainDataPath = "..\\HRRP_Data\\train\\"
HRRPValDataPath = "..\\HRRP_Data\\val\\"
HRRPEvalDataPath = '..\\HRRP_Data\\evaluate\\'

mDTrainDataPath = "..\\mD_Data\\OMP_Data\\train\\"
mDValDataPath = "..\\mD_Data\\OMP_Data\\val\\"
mDEvalDataPath = '..\\mD_Data\\OMP_Data\\evaluate\\'

HRRP_PATH_DICT = {
    'train':HRRPTrainDataPath,
    'val':HRRPValDataPath,
    'eval':HRRPEvalDataPath
}

mD_PATH_DICT = {
    'train':mDTrainDataPath,
    'val':mDValDataPath,
    'eval':mDEvalDataPath
}


train_list = [
    name for name in os.listdir(HRRPTrainDataPath)
    if name.endswith('.mat')
]
val_list = [
    name for name in os.listdir(HRRPValDataPath)
    if name.endswith('.mat')
]
eval_list = [
    name for name in os.listdir(HRRPEvalDataPath)
    if name.endswith('.mat')
]


shuffle(train_list)
shuffle(val_list)
shuffle(eval_list)




class MyDataLoader(keras.utils.Sequence):
    def __init__(self, x_set, batch_size, phase, Augment=False):
        self.x = x_set
        self.batch_size = batch_size
        self.Augment = Augment
        self.phase = phase

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        output_x1 = []
        output_x2 = []
        output_y = []
        for file_name in batch_x:
            mat_tmp_HRRP = scio.loadmat(HRRP_PATH_DICT[self.phase]+file_name)
            mat_tmp_mD = scio.loadmat(mD_PATH_DICT[self.phase]+file_name)
            if self.Augment:
                index_begin = np.random.randint(29-13)
            else:
                index_begin = 0
            tmp_HRRP = mat_tmp_HRRP['HRRPmap'][index_begin:index_begin+13, :]
            tmp_mD = mat_tmp_mD['mDmap'][index_begin:index_begin+13, :]
            tmp_HRRP = tmp_HRRP/np.max(tmp_HRRP)
            tmp_mD = tmp_mD/np.max(tmp_mD)
            output_x1.append(tmp_HRRP.reshape(13, 256, 1))
            output_x2.append(tmp_mD.reshape(13, 256, 1))
            if mat_tmp_HRRP['label'] == 0:
                output_y.append([1, 0])
            else:
                output_y.append([0, 1])
        output_x1 = np.array(output_x1)
        output_x2 = np.array(output_x2)
        output_y = np.array(output_y)

        return ({'HRRP_input': output_x1, 'mD_input': output_x2}, {'output': output_y})
        # return output_x, output_y


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
            generator=MyDataLoader(eval_list, 20,'eval', False))
        logs.update({'eval_loss': loss, 'eval_acc': acc})
        super().on_epoch_end(epoch, logs)
        print('\n eval loss:{:2f},eval acc:{:.2f}'.format(loss, acc))


def createModel():
    HRRP_model = tf.keras.models.load_model('model\\model1912031004.h5')
    mD_model = tf.keras.models.load_model('model\\model1912022034.h5')

    HRRP_model._name = 'model-HRRP'
    mD_model._name = 'model-mD'

    # len_video = 13
    # res_r = 256
    # res_c = 256
    HRRP_input = keras.layers.Input(shape=(len_video, res_r, 1),
                                        dtype='float32',
                                        name='HRRP_input')
    mD_input = keras.layers.Input(shape=(len_video, res_r, 1),
                                        dtype='float32',
                                        name='mD_input')
    output_HRRP = HRRP_model(HRRP_input)
    output_mD = mD_model(mD_input)
    output = WeightedAddLayer(name='output')([output_HRRP,output_mD])
    model = keras.models.Model(inputs=[HRRP_input,mD_input], outputs=output)
    return model


class WeightedAddLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedAddLayer, self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, ),  
                                      initializer='uniform',
                                      trainable=True)      
        super(WeightedAddLayer, self).build(input_shape) 
 
    def call(self, x):
        A,B = x
        # A /= 2
        # B /= 2
        # return keras.layers.Add()([A,B])
        return A * self.kernel + B*(1-self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[0]



len_video = 13
res_r = 256
res_c = 256


if __name__ == '__main__':
    timestamp = time.localtime()



    logger.add('log.txt',
               format='{time:YYYY-MM-DD at hh:mm:ss} | {level} | {message}')

    initialiser = 'glorot_uniform'
    reg_lambda = 0.001
    
    model = createModel()
    
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
    modelname = 'model' + getTimeString(timestamp)
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



    bs=20
    # for i_epoch in range(10):
    history = model.fit_generator(
        generator=MyDataLoader(train_list, bs,'train', True),
        epochs=150,
        validation_data=MyDataLoader(val_list, bs,'val', False),
        callbacks=[
            reduce_lr, stop_val, tensorboard, check_point
        ])


    logger.info('model' + getTimeString(timestamp) + ' is create by ' +
                sys.argv[0])
    logger.info('fusion')
    logger.info('-' * 49)
