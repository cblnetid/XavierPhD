import numpy as np
import torch
import torch.nn as nn
import network_bound
import my_config
import time


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.maxpool = nn.MaxPool2d(2) 
        '''
        self.fc1 = nn.Linear(d,dh) #aca puedo construir una capa simple
        self.fc2 = nn.Linear(d,1) #esta seria mi capa de salida
        '''
        #modelo de dos capas ocultas
        self.fc1 = nn.Linear(d,dh) #aca puedo construir una capa simple
        self.fc2 = nn.Linear(dh,dh) 
        self.fc3=nn.Linear(dh,1)#esta seria mi capa de salida
        
        
        self.relu = nn.ReLU(inplace=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        #x = self.relu(self.fc3(x))
        x = self.fc3(x) #solo para la senoidal
        return x


b1=np.load('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/b1.npy')
b2=np.load('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/b2.npy')
w1=np.load('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/w1.npy')
w2=np.load('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/w2.npy')
x = np.load('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/input.npy')
dh = np.load('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/dimo.npy')

#x=np.random.rand(x.shape[0])*3.14 # use for sin function

d=len(x)
# create network
net = MyNet()
#net.to(my_config.device)
relu = torch.nn.ReLU(inplace=False)

'''
#use for one hidden layer only
net.layers = [net.fc1, relu,
              net.fc2, relu]
'''
#use for tow hidden layers
b3=np.load('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/b3.npy')
w3=np.load('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/w3.npy')

net.layers = [net.fc1, relu,
              net.fc2, relu,
              net.fc3]#, relu]

net.fc1.weight = nn.Parameter(torch.Tensor(w1))
net.fc2.weight = nn.Parameter(torch.Tensor(w2))
net.fc1.bias = nn.Parameter(torch.Tensor(b1))
net.fc2.bias = nn.Parameter(torch.Tensor(b2))
net.fc3.weight = nn.Parameter(torch.Tensor(w3))
net.fc3.bias = nn.Parameter(torch.Tensor(b3))
# nominal input
x0 = torch.Tensor(x)
x0 = x0.to(my_config.device)
net.to(my_config.device) 

# input perturbation size and batch size
eps = 0.001
batch_size = 100

start_time = time.time()
# calculate global Lipschitz bound
layer_bounds = network_bound.global_bound(net, x0)
glob_bound = np.prod(layer_bounds)
print('GLOBAL LIPSCHITZ UPPER BOUND')
print('bound:', glob_bound)

# calculate local Lipschitz bound
bound = network_bound.local_bound(net, x0, eps, batch_size=batch_size)
print('\nLOCAL LIPSCHITZ UPPER BOUND')
print('epsilon:', eps)
print('bound:', bound)
#print(w1)
#print(net.layers)

execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time))

