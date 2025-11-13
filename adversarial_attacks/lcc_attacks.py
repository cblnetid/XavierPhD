    # -*- coding: utf-8 -*-
import warnings
import tensorflow as tf
from sklearn.metrics import roc_curve, auc as sk_auc
import numpy as np
#from sklearn.metrics import roc_curve, auc
from torchmetrics import Accuracy, F1Score, ConfusionMatrix,MetricCollection
from torchmetrics.classification import MulticlassROC, MulticlassAUROC
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
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


valid = True
arch="vgg"
subsample = 1

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
arch='vgg'
if valid:
    x_test = x_train[48000:]
    y_test = y_train[48000:]
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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 128
x_test /= 128
x_train -= 1
x_test -= 1


print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")


from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
#from cleverhans.tf2.attacks.deepfool import deepfool
from sklearn.metrics import roc_curve, auc

model = tf.keras.models.load_model('/home/clirlab/xavier/CovLip/keras/modelo_attack/')
scores = model.evaluate(x_test, y_test, verbose=1)
predict= model.predict(x_test)

import torch
import torch.nn.functional as F

print('Sin ataques', flush=True)
print('loss=%f' % scores[0], flush=True)
print('accuracy=%f' % scores[1])

roc = MulticlassROC(num_classes=10, thresholds=None,average='micro')
auc= MulticlassAUROC(num_classes=10, thresholds=None,average='macro')

# Convertir a tensores de PyTorch
predict_tensor = torch.from_numpy(predict)
y_test_tensor = torch.from_numpy(y_test)

# Para MulticlassROC, necesitamos que target sea de forma (N,) no (N, C)
y_test_labels = torch.argmax(y_test_tensor, dim=1)

fpr, tpr, thresholds = roc(predict_tensor, y_test_labels)
area = auc(predict_tensor, y_test_labels)
plt.plot(fpr, tpr,label=f" clean| {area:.2f}")

print('Iniciando los ataques adversarios', flush=True)

#-----------------------------------------------------------------------------------------
print('Iniciando los ataques FGSM', flush=True)

x_adv_fgsm = fast_gradient_method(model, x_test, eps=0.05, norm=np.inf)
_, acc_fgsm = model.evaluate(x_adv_fgsm, y_test, verbose=0)
print(f"Precisión bajo FGSM (ε=0.05): {acc_fgsm * 100:.2f}%")
predict= model.predict(x_adv_fgsm)

# Convertir a tensores de PyTorch
predict_tensor = torch.from_numpy(predict)

fpr, tpr, thresholds = roc(predict_tensor, y_test_labels)
area = auc(predict_tensor, y_test_labels)

plt.plot(fpr, tpr,label=f" fgsm| {area:.2f}")
#----------------------------------------------------------------------------------------
print('Iniciando los ataques PGD', flush=True)

x_adv_pgd = projected_gradient_descent(model, x_test, eps=0.1, eps_iter=0.01, nb_iter=40, norm=np.inf)
_, acc_pgd = model.evaluate(x_adv_pgd, y_test, verbose=0)
print(f"Precisión bajo PGD (40 iter): {acc_pgd * 100:.2f}%")
predict= model.predict(x_adv_pgd )

# Convertir a tensores de PyTorch
predict_tensor = torch.from_numpy(predict)
y_test_tensor = torch.from_numpy(y_test)

# Para MulticlassROC, necesitamos que target sea de forma (N,) no (N, C)
y_test_labels = torch.argmax(y_test_tensor, dim=1)


fpr, tpr, thresholds = roc(predict_tensor, y_test_labels)
area = auc(predict_tensor, y_test_labels)

plt.plot(fpr, tpr,label=f" pgd| {area:.2f}")
#-----------------------------------------------------------------------------------------------
print('Iniciando los ataques Gaussian', flush=True)

noise = np.random.normal(0, 0.05, size=x_test.shape)
x_adv_gauss = np.clip(x_test + noise, 0, 1)
_, acc_gauss = model.evaluate(x_adv_gauss, y_test, verbose=0)
print(f"Precisión bajo ruido gaussiano (σ=0.05): {acc_gauss * 100:.2f}%")
predict= model.predict(x_adv_gauss)

# Convertir a tensores de PyTorch
predict_tensor = torch.from_numpy(predict)
y_test_tensor = torch.from_numpy(y_test)

# Para MulticlassROC, necesitamos que target sea de forma (N,) no (N, C)
y_test_labels = torch.argmax(y_test_tensor, dim=1)

fpr, tpr, thresholds = roc(predict_tensor, y_test_labels)
area = auc(predict_tensor, y_test_labels)

plt.plot(fpr, tpr,label=f" gaussian| {area:.2f}")

#-------------------------------------------------------------------------------

plt.legend(
fontsize=12,                # Tamaño de fuente
title='Attack | AUC',           # Título opcional
title_fontsize=12,         # Tamaño del título
#loc='upper right',         # Posición
frameon=True,              # Marco alrededor
fancybox=True,             # Marco con esquinas redondeadas
#shadow=True                # Sombra
)
plt.title('Adversarial Attacks ROC - LCC')

plt.savefig('full_ROC_matrix.pdf', format="pdf",bbox_inches='tight')
plt.savefig('full_ROC_matrix.png', bbox_inches='tight')
plt.close()

