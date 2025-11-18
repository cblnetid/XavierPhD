import torchmetrics

from sklearn import datasets
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from sklearn import datasets
from sklearn.manifold import TSNE

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def generate_n_spirals(n_spirals=3, n_samples=1000, noise=0.1, a=0.5, b=0.3,
                      radial_sep=0.5, angular_sep=1.0, random_state=None):
    """
    Genera espirales con control independiente de separación radial y angular.

    Parámetros:
        radial_sep (float): Separación radial entre espirales (0-1)
        angular_sep (float): Separación angular entre espirales (>1 para más separación)
        otros parámetros igual que antes
    """
    if random_state is not None:
        np.random.seed(random_state)

    data_list = []
    label_list = []
    theta = np.linspace(0, 2 * np.pi * angular_sep, n_samples)

    for i in range(n_spirals):
        # Añadir desplazamiento radial proporcional al índice
        r = (a + radial_sep * i) + b * theta

        angle = 2 * np.pi * i / n_spirals  # ángulo de rotación
        x = r * np.cos(theta + angle)
        y = r * np.sin(theta + angle)

        x += np.random.normal(0, noise, n_samples)
        y += np.random.normal(0, noise, n_samples)

        data_list.append(np.column_stack([x, y]))
        label_list.append(np.full(n_samples, i))

    data = np.vstack(data_list)
    labels = np.concatenate(label_list)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    return torch.FloatTensor(data), torch.LongTensor(labels)

def plot_dataset(data, labels, title):
    """Función auxiliar para visualizar los datasets generados."""
    plt.figure(figsize=(8, 6))
    for i in range(np.unique(labels)):
        mask = labels == i
        plt.scatter(data[mask, 0], data[mask, 1], label=f'Clase {i}', alpha=0.6)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


n_classes=3

X_m, y_m= generate_n_spirals(n_spirals=n_classes, n_samples=500, noise=0.1, a=0.5, b=0.3,
                      radial_sep=2.5, angular_sep=0.8, random_state=None)

fig = plt.figure()
ax = fig.add_subplot()
scatter=  ax.scatter(
    x=X_m[:, 0],
    y=X_m[:, 1],
    c=y_m,
    cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.savefig('espirales.pdf', format="pdf", bbox_inches="tight")

plt.show()

class CLIR(nn.Module):
    def __init__(self, num_classes):
        super(CLIR, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, t):

        vc=torch.zeros(1).to(device)
        t=t.squeeze()

        for j in range (self.num_classes): #para cada clase

          cla=x[t==j].to(device) #extraigo la matriz con los puntos de esa clase
          ccov=torch.cov(torch.t(cla))
          ncla=torch.linalg.vector_norm(torch.diag(ccov,0),ord=1)
          vc=vc+ncla

        penalty1=vc/self.num_classes
        return penalty1

def test(net,x,t):
  initial_time = time.time()
  correct = 0
  total = 0
  with torch.no_grad():
      net.eval()
      x_t = torch.tensor(x, dtype=torch.float32).to(device)
      t_t = torch.tensor(t, dtype=torch.float32).to(device)

      t_pred = net(x_t)
      acc = metric(t_pred, t_t)

      print(f"Accuracy on all data: {acc}")

  return(acc)

#################33

def train(net, clir,lmax):
  n = train
  criterion = nn.CrossEntropyLoss()
  miloss=CLIR(n_classes)
  optimizer = optim.Adam(net.parameters(), lr=0.001)
  accu=0
  lamda=lmax
  beta=0.7
  accus=[]
  covs=[]
  for epoch in range(epos):
    if accu > 1.999:
      break
    running_loss = 0.0
    cc=0
    inputs, labels = X_m, y_m

    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    outputs = net(inputs)
    luss=miloss(outputs,labels)
    if clir:
      loss = criterion(outputs, labels)+lamda*luss
    else:
      loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    # print statistics
    if epoch == 0 or epoch % frames == 0:
      points.append(outputs.detach().cpu().numpy())
      accus.append(loss.detach().cpu().numpy())
      covs.append(luss.item())
    if epoch % 1000 == 0 or epoch == 0:
      accu = test(net,X_m,y_m)
      #print(accu)
      #lamda=accu
      if clir:
        print('epoch=',epoch,' accu=',
              accu.detach().cpu().numpy(),'lambda=', lamda,'cov=',luss.detach().cpu().numpy())
      else:
        print('epoch=',epoch,' accu=',
              accu.detach().cpu().numpy(),'lambda=', lamda,'cov=',luss.detach().cpu().numpy())
    running_loss = 0.0


  print('Finished Training')
  return accus,covs

model = nn.Sequential(
    nn.Linear(2, 512),  # Input layer with 2 features
    nn.ReLU(),          # Activation function
    nn.Linear(512, 1024), # Hidden layer
    nn.ReLU(),          # Activation function
    nn.Linear(1024, 1024), # Hidden layer
    nn.ReLU(),
    nn.Linear(1024, 512), # Hidden layer
    nn.ReLU(),
    nn.Linear(512, 256), # Hidden layer
    nn.ReLU(),
    nn.Linear(256, n_classes)  # Output layer with n_classes
)

metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes).to(device)
clir=True
epos=800
lmax=0.75
frames=10; #number of computed epochs before save points
points=[]
#myloss=ClassEigenPenaltyLoss(10)

## Main ##
net = model.to(device)
#print(net)
accus, covs=train(net, clir, lmax)

## Test ##
#net.load()
test(net,X_m,y_m)
print(np.asarray(points).shape)


inputs, labels = X_m, y_m
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
inputs = inputs.to(device)
labels = labels.to(device)

fig = plt.figure(figsize=(15, 5))
x_pred = net(inputs)
vals= x_pred.detach().cpu().numpy()

ax = fig.add_subplot(1,2,1)
scatter=  ax.scatter(
    x=vals[:, 0],
    y=vals[:, 1],
    c=y_m,
    cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')


# Primer subplot
ax1 = fig.add_subplot(1, 2, 2)
line1, = ax1.plot(accus, 'b-', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Segundo eje Y
ax2 = ax1.twinx()
line2, = ax2.plot(covs, 'r-', label='Variance')
ax2.set_ylabel('Variance', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Combinar las leyendas
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')
plt.show()

if clir:
  plt.savefig('spriral_clir.pdf', format="pdf", bbox_inches="tight")
else:
  plt.savefig('spriral_no_clir.pdf', format="pdf", bbox_inches="tight")

def animate_both(i):


    fig.clear()
    # Get the point from the points list at index i
    point = points[i]
    # Plot that point using the x and y coordinates
    ax = fig.add_subplot(1,2,1)
    scatter=  ax.scatter(
    x=points[i][:,0], # Changed line: Using list indexing to access the array and then NumPy slicing on the array.
    y=points[i][:,1],
    c=y_m,
    cmap='coolwarm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


    # Primer subplot (ajusta esto según tu necesidad real)
    ax1 = fig.add_subplot(1,2,2)
    line1, = ax1.plot(accus[0:i-1], 'b-',marker='+', label='Loss')  # Nota la coma después de line1
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim([0, m2])

    # Segundo eje Y
    ax2 = ax1.twinx()
    line2, = ax2.plot(covs[0:i-1], 'r-',marker='o', label='Variance')  # Nota la coma después de line2
    ax2.set_ylabel('Variance', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim([0, m1])
    # Combinar las leyendas
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')  # o cualquier otra ubicación




fig = plt.figure(figsize=(10, 5))
m1= np.array(covs).max()
m2 = np.array([arr.item() for arr in accus]).max()

ani = FuncAnimation(fig, animate_both, frames=len(points),
                    interval=500, repeat=False)
plt.close()


if clir:
# Save the animation as an animated GIF
  ani.save("spiral_train_CLIR.gif", dpi=300,writer=PillowWriter(fps=5))
else:
  ani.save("spiral_train_no_CLIR.gif", dpi=300,writer=PillowWriter(fps=5))

#ani.save("simple_animation_no_CLIR.gif", writer=FFMpegWriter(fps=10)) # Use FFMpegWriter for MP4
