  # -*- coding: utf-8 -*-

import sys
sys.path.append('/home/clirlab/xavier/CovLip/clip/CLIP-main')

import importlib
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import utils.configuration as cf
from utils.datasets import get_data_set
import models
import train

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

importlib.reload(cf)  # Recarga los cambios
importlib.reload(train)
importlib.reload(models)
from torchvision.models import efficientnet_v2_s, densenet121,mobilenet_v3_small,inception_v3, resnet18


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device= ',device,flush=True )
if device=='cuda':
    print (torch.cuda.get_device_name(0))


# -----------------------------------------------------------------------------------
# Set up variable and data for an example
# -----------------------------------------------------------------------------------
# specify the path of your data
data_file = "/home/clirlab/xavier/CovLip/MNIST/raw/"
epocs=2
# load up configuration from examples
#conf = cf.plain_example(data_file, use_cuda=True, download=True, epochs=epocs)
conf = cf.clip_example(data_file, use_cuda=True, download=True, epochs=epocs)

# get train, validation and test loader
#train_loader, valid_loader, test_loader = get_data_set(conf)

print(25*'-')
print('cargando las imagenes para entrenamiento')

# get train, validation and test loader
#train_loader, valid_loader, test_loader = get_data_set(conf)
batch_size_train = 128
batch_size_test = 128

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='./files/''/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

valid_loader=test_loader

'''
sizes = [784, 512, 128, 10]
model = models.fully_connected(sizes, conf.activation_function).to(conf.device)
best_model = train.best_model(models.fully_connected(sizes, conf.activation_function).to(conf.device), goal_acc = conf.goal_acc)
'''

#otros modelos de pytorch

'''
model = efficientnet_v2_s().to(device)
model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(device) #cambiar primera capa para imagenes de un canal
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet
'''
#---- Resnet 18

model=resnet18().to(device)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),padding=(3, 3), bias=False).to(device) #cambiar primera capa para imagenes de un canal
model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet


# -----------------------------------------------------------------------------------
# Initialize optimizer and lamda scheduler
# -----------------------------------------------------------------------------------
opt = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
lamda_scheduler = train.lamda_scheduler(conf, warmup = 5, warmup_lamda = 0.0, cooldown = 1)
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# initalize history
# -----------------------------------------------------------------------------------
tracked = ['train_loss', 'train_acc', 'train_lip_loss', 'val_loss', 'val_acc']
history = {key: [] for key in tracked}
# -----------------------------------------------------------------------------------
# cache for the lipschitz update
cache = {'counter':0}

print("Train model: {}".format(conf.model))
for i in range(conf.epochs):

    #print(25*"<>")
    #print(50*"|")
    print(25*"<>",flush=True)
    print('Epoch', i+1)

    # train_step
    train_data = train.train_step(conf, model, opt, train_loader, valid_loader, cache)

    # ------------------------------------------------------------------------
    # validation step
    val_data = train.validation_step(conf, model, valid_loader)
    stop=val_data['val_acc']
    
    if stop>=0.99:
      print('Se llegó al valor deseado',flush=True)
      lamda_scheduler(conf, train_data['train_acc'])
      #best_model(train_data['train_acc'], val_data['val_acc'], model=model)
      break
    
    # ------------------------------------------------------------------------
    # update history
    for key in tracked:
        if key in val_data:
            history[key].append(val_data[key])
        if key in train_data:
            history[key].append(train_data[key])

    # ------------------------------------------------------------------------
    lamda_scheduler(conf, train_data['train_acc'])

# -----------------------------------------------------------------------------------
# Test the model afterwards
# -----------------------------------------------------------------------------------
print('revisando conjunto de prueba con o sin ataque',flush=True)

#conf.attack.attack_iters=200
#test_data = train.test_step(conf, model, test_loader, attack=conf.attack)
#test_data = train.test_step(conf, best_model.best_model, test_loader, attack=conf.attack)
print('finalizado, guardando modelo',flush=True)

torch.save(model, '/home/clirlab/xavier/CovLip/clip/CLIP_MNIST_resnet.pth')
