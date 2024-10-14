# Este es el programa para una red neuronal para aproximar la funcion XOR del Dr. Brito
# se modificó para poder permitir que la red tenga la dimensión de entrada que se desee
# para ello se requiere construir un vector de entrada con los 1 y 0 de la compuerta
# y el vector de salidas, donde la XOR devuelve un calor 1 cuando la suma de los elementos
# es par

import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(4) # 3/2000 iter     ----
normad = 2 #np.inf  #VERIFICAR LA NORMA USADA
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
    binary_list.append(1)
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
    y=np.array(Y)
    return y

def sigmoid(x):
    return (1/(1+np.exp(-x)))

dim=2
epocas = 1

#aca estoy cambiando la entada por una funcion que me permita cambiar la dimension
#x=np.vstack(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]))
#t=np.array([0,1,1,0,1,0,0,1]).reshape(-1,1)
x=inpu(dim)
print(x)
t=outpu(x)
t=np.reshape(t,(-1,1))

w1=np.random.rand(dim+1,dim)
w2=np.random.rand(dim+1,1)

unos = np.ones(2**dim)
unos = np.reshape(unos, (2**dim,1))

alpha=0.5

loss=[]
for i in range(epocas):
    if i%10000==0:
        print(i)
    y1=sigmoid(np.dot(x,w1))
    y1=np.append(y1, unos,axis=1)
    ys=sigmoid(np.dot(y1,w2))
    E=(1/(2**dim))*np.sum((ys-t)**2)
    dEdw2=(2/(2**dim))*(np.dot(y1.T,(ys-t)*ys*(1-ys)))
    dEdw1=(2/(2**dim))*np.dot(x.T,np.dot((ys-t)*ys*(1-ys),w2.T)*y1*(1-y1))
    w2=w2-alpha*dEdw2
    w1=w1-alpha*dEdw1
    loss.append(E)

plt.plot(loss)
plt.title('gráfica del valor de la función de perdida')
plt.xlabel('épocas')
plt.ylabel('loss')
plt.grid()

"""La función de predicción es entonces"""
'''
def pred(x,w1,w2):
    y1=sigmoid(np.dot(x,w1))
    ys=sigmoid(np.dot(y1,w2))
    return ys

def pred2(x,w1,w2):
    y1=sigmoid(np.dot(x,w1))
    ys=sigmoid(np.dot(y1,w2))
    if ys>=0.5:
        return 1
    else:
        return 0

def lips(f1,f2,x1,x2):
    ns=np.abs(f2-f1)
    ni=np.linalg.norm(x2-x1,normad)
    l=ns/ni
   # print('l= ',l)
    return l

lip=[]
k=0
for i in range (2**dim):
    for j in range (2**dim):
        if np.array_equal(x[i], x[j]):
            n=0
        else:
            f1=pred(x[i],w1,w2)
            f2=pred(x[j],w1,w2)
            lip.append(lips(f1,f2,x[i],x[j]))

lipo=np.array(lip)
print("Exact = ",np.max(lipo))
print(lipo.shape)

b1=w1[:,dim]
w1=w1[:,0:dim]
b2=w2[dim]
w2=w2[0:dim]

print(w1)
LG=np.linalg.norm(np.matmul(w1,w2),normad)
print('Final Global Lipschitz ',LG,'\n')

np.save('w1.npy',w1)
np.save('w2.npy',w2)
np.save('b1.npy',b1)
np.save('b2.npy',b2)

inp=x[random.randrange(2**dim)]
np.save('input.npy',inp)
print(inp)
'''
plt.show()

