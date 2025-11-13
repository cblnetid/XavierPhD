# -*- coding: utf-8 -*-
import warnings



import tensorflow as tf

# Verificar la versión de TensorFlow
print("TensorFlow version:", tf.__version__)

# Listar dispositivos disponibles
print("\nDispositivos disponibles:")
print(tf.config.list_physical_devices())

# Verificar si hay GPU disponible
print("\nGPU disponible:", tf.test.is_gpu_available())
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Configurar TensorFlow para usar la GPU eficientemente
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Permitir crecimiento de memoria en lugar de asignación total
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Configurar computación distribuida si hay múltiples GPUs
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print('Número de dispositivos: {}'.format(strategy.num_replicas_in_sync),flush=True)
        else:
            strategy = tf.distribute.get_strategy()
    except RuntimeError as e:
        print(e,flush=True)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print('cargando modulos', flush=True)

from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10, cifar100
from keras.optimizers import Adam#, sgd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from arch.vgg import vgg19
from arch.wrn import wrn
import getopt
from sys import argv

batch_size = 64
num_classes = 10
epochs = 1
data_augmentation = False
lcc_norm = 2
lambda_conv = float("inf")
lambda_dense = float("inf")
lambda_bn = float("inf")
drop_conv = 0.2
drop_dense = 0.5
sd_conv=0.001
sd_dense=0.001
batchnorm = False
model_path = "modelo_keras"
valid = True
width=10
depth=16
arch="vgg"
log_path = "/home/clirlab/xavier/CovLip/keras/logs/"
subsample = 1

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
arch='vgg'
if valid:
    x_test = x_train[40000:]
    y_test = y_train[40000:]
    x_train = x_train[0:40000]
    y_train = y_train[0:40000]

if subsample < 1:
    train_size = int(x_train.shape[0] * subsample)
    x_train = x_train[0:train_size]
    y_train = y_train[0:train_size]

print ('Ejemplos cargados',flush=True)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples',flush=True)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

in_chan = x_train.shape[3]
in_dim = x_train.shape[1]

def lr_schedule_vgg(epoch):
    if epoch >= 120:
        return 0.000001
    elif epoch >= 100:
        return 0.00001
    else:
        return 0.0001

def lr_schedule_wrn(epoch):
    if epoch >= 160:
        return 0.0008
    elif epoch >= 120:
        return 0.004
    elif epoch >= 60:
        return 0.02
    else:
        return 0.1

if arch == "vgg":
    model = vgg19(
        in_chan,
        in_dim,
        num_classes,
        bn=batchnorm,
        drop_rate_conv=drop_conv,
        drop_rate_dense=drop_dense,
        lcc_norm=lcc_norm,
        lambda_conv=lambda_conv,
        lambda_dense=lambda_dense,
        lambda_bn=lambda_bn,
        sd_conv=sd_conv,
        sd_dense=sd_dense
    )

    epochs = 140
    lr_scheduler = LearningRateScheduler(lr_schedule_vgg)
    opt = Adam(amsgrad=True)

elif arch == "wrn":
    model = wrn(
        in_chan,
        in_dim,
        num_classes,
        width,
        depth,
        drop_rate_conv=drop_conv,
        lcc_norm=lcc_norm,
        lambda_conv=lambda_conv,
        lambda_dense=lambda_dense,
        lambda_bn=lambda_bn,
        sd_conv=sd_conv,
        sd_dense=sd_dense
    )
    epochs = 160
    batch_size = 64
    lr_scheduler = LearningRateScheduler(lr_schedule_wrn)
    opt = sgd(momentum=0.9, nesterov=True)
else:
    raise Exception("Unknown architecture")

epochs=10
batch_size = 256
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 128
x_test /= 128
x_train -= 1
x_test -= 1

datagen = ImageDataGenerator(
    width_shift_range=0.125,
    height_shift_range=0.125,
    fill_mode='nearest',
    horizontal_flip=True)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    workers=4,
                    callbacks=[lr_scheduler],
                    steps_per_epoch=x_train.shape[0] / batch_size)


model.save('/home/clirlab/xavier/CovLip/keras/modelo_attack/')
scores = model.evaluate(x_test, y_test, verbose=1)

print ('loss=%f' % scores[0],flush=True)
print ('accuracy=%f' % scores[1])

