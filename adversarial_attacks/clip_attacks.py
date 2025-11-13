import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchattacks

from sklearn import datasets
from sklearn.manifold import TSNE
import time

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, densenet121,mobilenet_v3_small,inception_v3, resnet18

import sys
sys.path.append('/home/clirlab/xavier/CovLip/clip/CLIP-main')
#---------- clase para FC del CLIP----------------------------------
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_model(conf):
    model = None
    if conf.model.lower() == "fc":
        model = fully_connected(conf)
    else:
        raise NameError("Modelname: {} does not exist!".format(conf.model))
    model = model.to(conf.device)
    return model


def get_activation_function(activation_function):
    af = None
    if activation_function == "ReLU":
        af = nn.ReLU
    elif activation_function == "sigmoid":
        af = nn.Sigmoid
    else:
        af = nn.ReLU
    return af

class fully_connected(nn.Module):
    def __init__(self, sizes, act_fun, mean = 0.0, std = 1.0):
        super(fully_connected, self).__init__()

        self.act_fn = get_activation_function(act_fun)
        self.mean = mean
        self.std = std

        layer_list = [Flatten()]
        for i in range(len(sizes)-1):
            layer_list.append(nn.Linear(sizes[i], sizes[i+1]))
            layer_list.append(self.act_fn())

        self.layers = nn.Sequential(*layer_list)


    def forward(self, x):
        x = (x - self.mean)/self.std
        return self.layers(x)

#------------------------------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device= ',device,flush=True )
if device=='cuda':
    print (torch.cuda.get_device_name(0))

print ("iniciando" , flush=True)
epsi=0.01
batch_size_train = 2048
batch_size_test = 1
learning_rate = 0.01
momentum = 0.3
log_interval = 10

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

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.to(device)
example_targets.to(device)

print(example_data.shape, flush=True)

def testa( model, device, test_loader, epsilon, attack ):
    #print('device= ',device)
    # Accuracy counter
    correct = 0
    #batch correct
    bok=0
    adv_examples = []
    accs=[]
    i=0
    # Loop over all examples in test set
    for data, target in test_loader:
        if i%500 == 0 and i !=0 :
          b_acc=bok/500
          accs.append(b_acc)
          print('i= ', i, flush=True)
          print(f"Batch Accuracy = {bok} / 500 = {b_acc}")
          bok=0
          #break
        i+=1

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Forward pass the data through the model
        output = model(data).to(device)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        if attack == "fgsm":
            atk = torchattacks.FGSM(model, 8/255)
        if attack ==  "jsma":
            atk = torchattacks.JSMA(model, theta=1.0, gamma=0.1)
        if attack == "cw":
            atk= torchattacks.CW(model, c=1, kappa=40, steps=50, lr=0.01) #kappa ca de 0 a 40 en el paper a mayor valor mas fuerte
        if attack == "df":
            atk = torchattacks.DeepFool(model, steps=50, overshoot=epsilon)
        if attack == "gaussian":
            atk = torchattacks.GN(model, std=epsilon)
        if attack == "clean":
            atk = torchattacks.VANILA(model)
        if attack == "pgd":
            atk = torchattacks.PGD(model, eps=epsilon, alpha=1/255, steps=10, random_start=True)
       

        atk.set_normalization_used(mean=[0.1307,], std=[0.3081,]) # If inputs were normalized, then
        perturbed_data = atk(data, target)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            bok+=1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}", flush=True)

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples,accs

    
lista =['clean','fgsm','pgd','jsma','df','cw','gaussian'] #lista completa
#lista =['clean']

# LOAD YOUR CLIP TRAINED MODEL HERE ----------------------------------------------------------
network = torch.load('/home/clirlab/xavier/CovLip/clip/CLIP_MNIST_effi.pth', weights_only=False)
network.eval()

for attack in lista:
    epsilons = [0.2]
    accuracies = []
    examples = []
    print('--------> iniciando ataque CLIP-------> ', attack, flush=True)
    # Run test for each epsilon
    for eps in epsilons:
        acc, ex,accse = testa(network, device, test_loader, eps,attack)
        accuracies.append(acc)
        examples.append(ex)

    print(accse, flush=True)
    texto='results_clip_effi_'+attack+'.txt'
    np.savetxt(texto, accse, fmt='%.3f')



