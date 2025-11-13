# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from sklearn import datasets
from sklearn.manifold import TSNE
import time

from torchmetrics import Accuracy, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device= ',device,flush=True )
if device=='cuda':
    print (torch.cuda.get_device_name(0))

def test():
  network.eval()
  test_loss = 0
  correct = 0
  # Listas para acumular todas las predicciones y targets
  all_preds = []
  all_targets = []
  #variables para torchmetrics
  accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
  f1=F1Score(task="multiclass", num_classes=10).to(device)
  confus= ConfusionMatrix(task="multiclass", num_classes=10).to(device)
  
  with torch.no_grad():
    for data, target in test_loader:
      data=data.to(device)
      target=target.to(device)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()

      accuracy.update(output,target)
      f1.update(output,target)
      confus.update(output, target)

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

    
  # Calcular métricas sobre todo el conjunto
  total_accuracy = accuracy.compute()
  f1score=f1.compute()
  cmat = confus.compute()
  print(f"Accuracy: {total_accuracy}")
  #print(f"F1: {f1score}")
  #print("Confusion: \n", cmat.detach().cpu().numpy())

  accuracy.reset()
  f1.reset()
  confus.reset()
  return total_accuracy,f1score,cmat.detach().cpu().numpy()
#--------------------------------------------------------------------------
epsi=0.01
batch_size_train = 1024
batch_size_test = 1024
learning_rate = 0.01
momentum = 0.3
log_interval = 10

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='/home/clirlab/xavier/CovLip/MNIST', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root='/home/clirlab/xavier/CovLip/MNIST', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.to(device)
example_targets.to(device)

print(example_data.shape)


class ClassDistancePenaltyLoss(nn.Module):
    def __init__(self, num_classes, norma):
        super(ClassDistancePenaltyLoss, self).__init__()
        self.num_classes = num_classes
        self.norma=norma

    def forward(self, x, t):
        vc=torch.zeros(1).to(device)
        medias=torch.zeros(self.num_classes,x.shape[1]).to(device)
        distancias=torch.zeros(self.num_classes,self.num_classes).to(device)

        for j in range (self.num_classes): #para cada clase

          cla=x[t==j].to(device) #extraigo la matriz con los puntos de esa clase
          ccov=torch.cov(torch.t(cla))

          if self.norma:
              ncla=torch.linalg.vector_norm(torch.diag(ccov,0),ord=1)
          else:
              ccov=torch.cov(torch.t(cla))
              ncla=torch.trace(ccov)
          vc=vc+ncla

        penalty=vc/self.num_classes
        return penalty


class fully_connected(nn.Module):
    def __init__(self, sizes, act_fun, mean = 0.1307, std = 0.3081):
        super(fully_connected, self).__init__()
        
        self.act_fn = act_fun
        self.mean = mean
        self.std = std
        
        layer_list = [nn.Flatten()]
        for i in range(len(sizes)-1):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(nn.Sigmoid())
            
        self.layers = nn.Sequential(*layer_list)
        
        
    def forward(self, x):
        x = (x - self.mean)/self.std
        return self.layers(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128,22 )
        self.fc3 = nn.Linear(22, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=False)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        #x = F.relu(x)
        x = self.relu(self.conv2(x))
        #x = F.relu(x)
        x = self.pool(x)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        y=x
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output,y

#aca vamos a cargar y modeificar el modelo para que pueda usarse en nuestra implementación
from torchvision.models import efficientnet_v2_s, densenet121,mobilenet_v3_small,inception_v3, resnet18

#RED PARA ENTRENAR CHEAP LIPSCHITZ
'''
sizes = [784, 512, 128, 10]
network = fully_connected(sizes, 'sigmoid').to(device)
'''

network = efficientnet_v2_s().to(device)
network.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(device) #cambiar primera capa para imagenes de un canal
network.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet

#MobileNet ---
'''
network=mobilenet_v3_small().to(device)
network.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(device) #cambiar primera capa para imagenes de un canal
network.classifier[3] = torch.nn.Linear(in_features=1024, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet
'''
#---- Resnet 18
'''
network=resnet18().to(device)
network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),padding=(3, 3), bias=False).to(device) #cambiar primera capa para imagenes de un canal
network.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet
'''
#print(network)
#-----------------------------------------------------------------------------------------------------------------------
print('Inicia',flush=True )
print(device,flush=True )
exes=2000 #numero de ejemplos que se van a utilizar para el entrenamiento

optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
n_epochs = 2
lam=0.5 #covarianza

si= True  #<------------------------------------ACA ESTA EL PARAMETRO-----------
stop = 0.999
myloss=ClassDistancePenaltyLoss(10,si)

f1r=[]
cov=[]
test_acc=[]
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

#test()
e=0
start_time = time.time()

for epoch in range(n_epochs):
  network.train()
  e=e+1

  for batch_idx, (data, target) in enumerate(train_loader):
      data=data.to(device)
      target=target.to(device)
      optimizer.zero_grad()
      output = network(data)
      loss = F.cross_entropy(output, target)

      if si:
        p1=myloss(output,target)
        loss=loss+(lam*p1)
      else:
        with torch.no_grad():
          p1=myloss(output,target)
      
      salir = loss.item()
      loss.backward()
      optimizer.step()
      '''
      if salir<0.000000000005:
        break
      '''
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch,1 * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()),flush=True )
        train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
  network.eval()
  acc,f1score,cmat=test()

  if acc>stop:
    break

execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time),flush=True )
torch.save(network, 'Mnist_CLIR_effi99_2.pth')
