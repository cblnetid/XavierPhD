import sys 
sys.path.append('/home/xavier/Dropbox/00 Doctorado 2022/Semestre 1/lipMIP-master/')
import torch 
from relu_nets import ReLUNet 
from hyperbox import Hyperbox 
from lipMIP import LipMIP
import numpy as np 
import torch.nn as nn 
import time

# load net
b1=np.load('b1.npy')
b2=np.load('b2.npy')
w1=np.load('w1.npy')
w2=np.load('w2.npy')
x = np.load('input.npy')

to=np.random.rand(x.shape[0])*3.14  #use for sin function


dim = len(x)
dimh=np.load('dimo.npy') #number of hidden seurons XOR

# Define an input domain and c_vector 
simple_domain = Hyperbox.build_unit_hypercube(dim)
simple_c_vector = torch.Tensor([1])

'''
#use for one hidden layer model
model = ReLUNet([dim, dimh, 1]) #
model.net[0].weight=nn.Parameter(torch.Tensor(w1))
model.net[0].bias=nn.Parameter(torch.Tensor(b1))
model.net[2].weight=nn.Parameter(torch.Tensor(w2))
model.net[2].bias=nn.Parameter(torch.Tensor(b2))

'''
#use for two hidden layers model

b3=np.load('b3.npy')
w3=np.load('w3.npy')
model = ReLUNet([dim, dimh, dimh, 1]) #
model.net[0].weight=nn.Parameter(torch.Tensor(w1))
model.net[0].bias=nn.Parameter(torch.Tensor(b1))
model.net[2].weight=nn.Parameter(torch.Tensor(w2))
model.net[2].bias=nn.Parameter(torch.Tensor(b2))
model.net[4].weight=nn.Parameter(torch.Tensor(w3))
model.net[4].bias=nn.Parameter(torch.Tensor(b3))


start_time = time.time()


simple_prob = LipMIP(model, simple_domain, simple_c_vector, verbose=True, num_threads=2)
simple_result = simple_prob.compute_max_lipschitz()

print(simple_result)  # A lot more information under the hood 
print('\n')

execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time))
