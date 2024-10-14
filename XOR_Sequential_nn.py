import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import time


#### Función para convertir un número d a un código binarios de dim bits

def binario(d,dim): 
    binary_list = []
    while d > 0:
        binary_list.append(d % 2)
        d //= 2

    for i in range(dim-len(binary_list)):
        binary_list.append(0)
    # Reverse the order of the binary digits in the list
    binary_list.reverse()
    return binary_list

### Esta función genera un array de códigos binarios desde o hasta 2^dim
def inpu(dim):
    X=[]
    for k in range(2**dim):
        X.append(binario(k,dim))
    x=np.array(X)
    return x

### Esta función genera el array de salidas (target) 
def outpu(inpu):
    Y=[]
    for k in range(len(inpu)):
        if np.sum(inpu[k,:])%2==0:
            Y.append(0)
        else:
            Y.append(1)
    y=np.array([Y])
    return y

#torch.random.manual_seed(8)

#establecemos la dimension de entrada, las capas ocultas seran del doble
dim=12
dimo=dim*2

#se crean los ejemplos y sus targets
x_np=inpu(dim) 

if dim>10:
    np.random.shuffle(x_np)
    x_np = x_np[:1024,:]


t_np=outpu(x_np)



x=torch.tensor(x_np,dtype=torch.float32)
t=torch.tensor(t_np,dtype=torch.float32)
print (x.shape)
print (t.shape)

ejem=x.size(0) #extraemos el numero de ejemplos que se utilizaron

# se dimensiona la red
input_units = dim
hidden_units= dimo
output_units = 1

epocas=12000
'''
#Modelo de una capa de entrada, una oculta y una neurona de salida
model = nn.Sequential(nn.Linear(input_units, hidden_units), \
                      nn.ReLU(), \
                      nn.Linear(hidden_units, output_units), \
                      nn.ReLU(),    )
'''
#Modelo de una capa de entrada, dos ocultas y una neurona de salida
model = nn.Sequential(nn.Linear(input_units, hidden_units), \
                      nn.ReLU(), \
                      nn.Linear(hidden_units, hidden_units), \
                      nn.ReLU(), \
                      nn.Linear(hidden_units, output_units), \
                      nn.ReLU(), )
    
loss_funct = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

#pasamos los tensores a la GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device ='cpu' #solo lo usamos para forzar a cpu

model = model.to(device)
x = x.to(device)
t = t.to(device)
losses = []

start_time = time.time()

for i in range(epocas):
    y_pred = model(x)
   #naday y_pred=torch.reshape(y_pred,(4,1))

    loss = loss_funct(y_pred,t.T)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    
    if i%1000 == 0:
        print(i, loss.item())

print("Training Final Error= ", losses[epocas-1])
plt.plot(range(0,epocas), losses)
plt.show()
#print(model)
w1=model[0].weight.cpu().detach().numpy()
b1=model[0].bias.cpu().detach().numpy()
w2=model[2].weight.cpu().detach().numpy()
b2=model[2].bias.cpu().detach().numpy()

w3=model[4].weight.cpu().detach().numpy()
b3=model[4].bias.cpu().detach().numpy()

execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time))
LG=np.linalg.norm(w1,2) * np.linalg.norm(w2,2)*np.linalg.norm(w3,2)
print('Final Global Lipschitz ',LG,'\n')

'''
def pred(x):
    ys=model(x)
    return ys


def lips(f1,f2,x1,x2):
    ns=np.abs(f2-f1)
    ni=np.linalg.norm(x2-x1,2)
    print(ns,ni)
    l=ns/ni
    return l

lip=[]
k=0

print ('evaluar = ', 2**dim)
for i in range (2**dim):
    if i%10==0:
        print(i)
    for j in range (2**dim):
        if np.array_equal(x[i], x[j]):
            n=0
        else:
            f1=pred(x[i])
            f2=pred(x[j])
            f11=f1.cpu()
            f21=f2.cpu()
            x1=x[i].cpu()
            x2=x[j].cpu()
            lip.append(lips(f11.detach().numpy(),f21.detach().numpy(),x1,x2))


lipo=np.array(lip)
print(lipo)
print("Exact = ",np.max(lipo))
'''

np.save('w1.npy',w1)
np.save('w2.npy',w2)
np.save('b1.npy',b1)
np.save('w3.npy',w3)
np.save('b3.npy',b3)
np.save('b2.npy',b2)
np.save('dimo.npy',np.array(dimo))
#print(dimo)


inp=x[random.randrange(ejem)]
np.save('input.npy',inp.cpu())
#print(inp)
#print(pred(x))
#print(w1)
