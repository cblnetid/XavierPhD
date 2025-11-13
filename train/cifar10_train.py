# -*- coding: utf-8 -*-

print('inicia importar modulos',flush=True)

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
import time

print('modulos cargados',flush=True)
###########################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device= ',device,flush=True )
if device=='cuda':
    print (torch.cuda.get_device_name(0))
#########################################################################################


#%% LOAD DATA ---------------------------------------------------------------------------
print('load data',flush=True)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size_train = 1024
batch_size_test=2000

logi= int((50000/batch_size_train)/3)
print('logi= ', logi)
trainset = torchvision.datasets.CIFAR10(root='/home/clirlab/xavier/CovLip/cifar-10-python', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='/home/clirlab/xavier/CovLip/cifar-10-python', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=False, num_workers=0)

print('fin load data',flush=True)


#%% COVARIANCE LOSS

class CLIR(nn.Module):
    def __init__(self, num_classes):
        super(CLIR, self).__init__()
        self.num_classes = num_classes


    def forward(self, x, t):
        vc=torch.zeros(1).to(device)
        for j in range (self.num_classes): #para cada clase
          cla=x[t==j].to(device)
          ccov=torch.cov(torch.t(cla))
          ncla=torch.trace(ccov)
          vc=vc+ncla

        return vc/self.num_classes

#%% TESTING
def test():
  network.eval()
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in testloader:
          images, labels = data[0].to(device),data[1].to(device)
          # calculate outputs by running images through the network
          outputs = network(images)
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  accu=100 * correct / total
  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %',flush=True)

  return accu
  
#%% BUILD MODEL------------------------------------------------------------------------------------------
#aca vamos a cargar y modeificar el modelo para que pueda usarse en nuestra implementación
from torchvision.models import efficientnet_v2_s, densenet121,mobilenet_v3_small,inception_v3, resnet18, vgg19,convnext_tiny,mobilenet_v3_large
from torchvision.models import ResNet18_Weights, MobileNet_V3_Small_Weights,MobileNet_V3_Large_Weights, VGG19_Weights

'''
network = vgg19(VGG19_Weights.DEFAULT).to(device)
network.classifier[6] = torch.nn.Linear(in_features=4096, out_features=10, bias=True).to(device)

print(network)
'''

#network = Net().to(device)

network = efficientnet_v2_s().to(device)
#network.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(device) #cambiar primera capa para imagenes de un canal
network.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet

'''
#MobileNet ---

network=mobilenet_v3_large(MobileNet_V3_Large_Weights.DEFAULT).to(device)
#network.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(device) #cambiar primera capa para imagenes de un canal
network.classifier[3] = torch.nn.Linear(in_features=1280, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet
'''
#---- Resnet 18
'''
network=resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
#network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),padding=(3, 3), bias=False).to(device) #cambiar primera capa para imagenes de un canal
network.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet
'''
#print(network)
#-----------------------------------------------------------------------------------------------------------------------


#%% BEGIN TRIANING
print('inicia training vgg19 CLIR', flush=True)

#random_seed = 506
#torch.manual_seed(random_seed)

accus=[]
lamda= 1.5 #covarianza

si=True  #<---------- PARAMETRO
epochs=2
stop = 95.0
number_classes=10

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
myloss=CLIR(number_classes)

e=0
start_time = time.time()
varianz=[]
for epoch in range(epochs):  # loop over the dataset multiple times
    e=e+1
    network.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device),data[1].to(device)
        optimizer.zero_grad()
        outputs= network(inputs)
 
        if si:
          covi=myloss(outputs,labels)
          loss=criterion(outputs, labels)+lamda*covi
        else:
          loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % logi == 0 and i != 0:    # print every 32 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 32:.7f}',flush=True)
            running_loss = 0.0
    accu=test()
    accus.append(accu)
    if si:
      print("variance= ", covi.detach().cpu().numpy())
      varianz.append(covi.detach().cpu().numpy())

execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time))
print('Finished Training', flush=True)

name='/home/clirlab/xavier/CovLip/cifar-10-python/cifar10_final_test.pth'
torch.save(network,name)
