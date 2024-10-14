#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Thu Mar 30 08:10:02 2023

@author: xavier sierra
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device ='cpu'

def rad (x):
    r=x*(np.pi/180)
    return r
    
def lips (x,y):
    L=np.abs(np.sin(x)-np.sin(y))/np.abs(x-y)
    return L

dim = 50

X= np.arange(0.0,2*np.pi,0.001)
t=[]
for i in range (dim):
   t.append(X)
#print(t)
x=np.array(t).T
s=np.sin(x)

so=np.prod(s,axis=1)

#print('el valor de sin(s) es ',s,'\n')

xi=torch.tensor(np.full(dim,np.pi/2),dtype=torch.float32)
print(xi)

xo=torch.tensor(x,dtype=torch.float32)
to=torch.tensor([so],dtype=torch.float32)
#tensor([[0., 1., 1., 0.]])

print(xo.shape)
print(to.shape)

input_units = dim
hidden_units= dim
output_units = 1
epocas=15000

model = nn.Sequential(nn.Linear(input_units, hidden_units), \
                      nn.ReLU(), \
                      nn.Linear(hidden_units, hidden_units), \
                      nn.ReLU(), \
                      nn.Linear(hidden_units, output_units), \
                         )
                      
                      #nn.ReLU(), ) #quitar para senoidal regresion

loss_funct = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = model.to(device)
xo = xo.to(device)
to = to.to(device)
xi = xi.to(device)
losses = []

start_time = time.time()

for i in range(epocas):
    y_pred = model(xo)
    loss = loss_funct(y_pred, to.T)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    
    if i%1000 == 0:
        print(i, loss.item())

plt.plot(range(0,epocas), losses)
plt.show()

w1=model[0].weight.cpu().detach().numpy()
b1=model[0].bias.cpu().detach().numpy()
w2=model[2].weight.cpu().detach().numpy()
b2=model[2].bias.cpu().detach().numpy()

w3=model[4].weight.cpu().detach().numpy()
b3=model[4].bias.cpu().detach().numpy()

'''
w3=model[4].weight.cpu().detach().numpy()
b3=model[4].bias.cpu().detach().numpy()
'''
LG=np.linalg.norm(w1,2) * np.linalg.norm(w2,2) *np.linalg.norm(w3,2)

#LG=np.linalg.norm(w1,2) * np.linalg.norm(w2,2)*np.linalg.norm(w3,2)


print("Training Final Error= ", losses[epocas-1])
plt.plot(range(0,epocas), losses)

print('Final Global Lipschitz ',LG,'\n')
print(model(xi)) #valorar cuando es un vector [pi/2,...,pi/2] debe ser 1
#ys = model(xo).detach().numpy()
#plt.plot(ys,'o')
#plt.show()

execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time))

np.save('w1.npy',w1)
np.save('w2.npy',w2)
np.save('b1.npy',b1)
np.save('w3.npy',w3)
np.save('b3.npy',b3)
np.save('b2.npy',b2)
np.save('dimo.npy',np.array(hidden_units))
#print(dimo)


#inp=x[random.randrange(2**dim)]
np.save('input.npy',x[0])