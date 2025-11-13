# -*- coding: utf-8 -*-

print('inicia importar modulos')

import torchmetrics
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
import time

from torchmetrics import Accuracy, F1Score, ConfusionMatrix,MetricCollection
from torchmetrics.classification import MulticlassROC
from torchmetrics.classification import MulticlassConfusionMatrix


print('modulos cargados',flush=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device= ',device,flush=True )
if device=='cuda':
    print (torch.cuda.get_device_name(0))



#%% LOAD DATA

print('load data',flush=True)
DATA_DIM = 64 # height and width 64 si cabe en GPU, 224 para ViT

transform = transforms.Compose(
    [transforms.Resize((DATA_DIM, DATA_DIM)),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

batch_size_train = 1024
batch_size_test=2000

logi= int((8000/batch_size_train)/3)
print('logi= ', logi,flush=True)

dataset = torchvision.datasets.ImageFolder(root='/home/clirlab/xavier/CovLip/imagenet', transform=transform)

print('load data')
DATA_DIM = 64

transform = transforms.Compose(
    [transforms.Resize((DATA_DIM, DATA_DIM)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


TRAIN_TEST_SPLIT = [0.8, 0.2]

trainset, testset = torch.utils.data.random_split(
    dataset,
    TRAIN_TEST_SPLIT)
print(len(trainset))
print(len(testset))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=2, drop_last = True, prefetch_factor=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=False, num_workers=2, drop_last = True, prefetch_factor=2)
print('fin load data',flush=True)


#%% COVARIANCE LOSS

class CLIR(nn.Module):
    def __init__(self, num_classes,):
        super(CLIR, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, t):
        vc=torch.zeros(1).to(device)
        for j in range (self.num_classes): #para cada clase
          cla=x[t==j].to(device)
          ccov=torch.cov(torch.t(cla))
          ncla=torch.trace(ccov)
          vc=vc+ncla

        penalty1=vc/self.num_classes
        return penalty1

#%% TESTING
def test():
  network.eval()
  correct = 0
  total = 0

    #variables para torchmetrics
  accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
  f1=F1Score(task="multiclass", num_classes=10).to(device)
  confus= ConfusionMatrix(task="multiclass", num_classes=10).to(device)


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

          accuracy.update(outputs,labels)
          f1.update(outputs,labels)
          confus.update(outputs, labels)

  accu= correct / total
  #print(f'Accuracy of the network on the test images: {100 * correct / total} %',flush=True)

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

  return total_accuracy,f1score
#del dataiter

#%% BUILD MODEL------------------------------------------------------------------------------------------
  #aca vamos a cargar y modeificar el modelo para que pueda usarse en nuestra implementación

from torchvision.models import efficientnet_v2_s, densenet121,mobilenet_v3_small,inception_v3, resnet18, vgg16,convnext_tiny,mobilenet_v3_large, vit_b_16
from torchvision.models import ResNet18_Weights, MobileNet_V3_Small_Weights,MobileNet_V3_Large_Weights,EfficientNet_V2_S_Weights
from torchvision.models import EfficientNet_V2_S_Weights, DenseNet121_Weights, Inception_V3_Weights, ResNet18_Weights, VGG16_Weights
from torchvision.models import ConvNeXt_Tiny_Weights,MobileNet_V3_Large_Weights,ViT_B_16_Weights

#network = Net().to(device)

'''
network = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT).to(device)
#network.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(device) #cambiar primera capa para imagenes de un canal
network.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet
'''

#MobileNet ---
'''
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

#graph(network, False)
#-----------------------------------------------------------------------------------------------------------------------


##################        empiezan entrenamientos-------------------------------
#%% BEGIN TRIANING
#%% HYPERPARAMETERS AND TRAINING CONDITIONS
print('inicia training',flush=True)



'''
random_seed = 5900
torch.manual_seed(random_seed)
'''

si=False  #<---------- PARAMETRO


network = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT).to(device)
network.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10, bias=True).to(device) #cambiar la ultima capa para 10 clases en ves de 1000 de imagenet

num_classes = 10
roc = MulticlassROC(num_classes=num_classes, thresholds=None)

f1r=[]
cov=[]
test_acc=[]
accus=[]

lamda= 0.7 

metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)

epochs=2

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.00051)
myloss=CLIR(num_classes)
accu_a=0
e=0
start_time = time.time()

for epoch in range(epochs):  # loop over the dataset multiple times
    e=e+1
    network.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs= network(inputs)
        loss = criterion(outputs, labels)
        if si:
          covi=myloss(outputs,labels)
          loss=loss+lamda*covi
        else:
          with torch.no_grad():
            covi=myloss(outputs,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % logi == 0 and i != 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {(i + 1)*10:5d}] loss: {running_loss / 3:.7f}',flush=True)
            print ('Variance= ', covi.item(),flush=True)
            running_loss = 0.0
    
    #accu=test()
    acc,f1score=test()

    accus.append(acc)
    roc.update(outputs,labels)



    f1r.append(f1score.detach().cpu().numpy())
    test_acc.append(acc.detach().cpu().numpy())
    cov.append(covi.item())

    accu=0


execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time),flush=True)
print('Finished Training')

torch.save(network, 'renet18_clir.pth')
    




