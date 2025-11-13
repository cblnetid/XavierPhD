import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchattacks
from torchmetrics import ROC
from torchmetrics.functional.classification import binary_auroc
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, F1Score, ConfusionMatrix,MetricCollection
from torchmetrics.classification import MulticlassROC, MulticlassAUROC
from torchmetrics.classification import MulticlassConfusionMatrix

import time
import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device= ', device, flush=True)
if device == 'cuda':
    print(torch.cuda.get_device_name(0))

print("iniciando", flush=True)
epsi = 0.01
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
    torchvision.datasets.MNIST(root='./files/', train=False, download=True,
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

roc = MulticlassROC(num_classes=10, thresholds=None,average='micro')
auc= MulticlassAUROC(num_classes=10, thresholds=None,average='macro')

from torchvision.models import efficientnet_v2_s, densenet121,mobilenet_v3_large,inception_v3, resnet18, vgg16,convnext_tiny
from torchvision.models import ResNet18_Weights,MobileNet_V3_Large_Weights,EfficientNet_V2_S_Weights


def testa(model, device, test_loader, epsilon, attack):
    correct = 0
    bok = 0
    adv_examples = []
    accs = []
    
    # Listas para almacenar las probabilidades y etiquetas reales para ROC
    all_probs = []
    all_labels = []
    
    i = 0
    
    for data, target in test_loader:
        if i % 500 == 0 and i != 0:
            b_acc = bok / 500
            accs.append(b_acc)
            print('i= ', i, flush=True)
            print(f"Batch Accuracy = {bok} / 500 = {b_acc}")
            bok = 0
        i += 1
        
        if i>2000:
            break
        
        data, target = data.to(device), target.to(device)
        '''
        # Forward pass original
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        '''
        # Si la predicción inicial es incorrecta, saltar
        #if init_pred.item() == target.item():
            

        # Generar ataque adversario
        if attack == "fgsm":
            atk = torchattacks.FGSM(model, 8/255)
        elif attack == "jsma":
            atk = torchattacks.JSMA(model, theta=1.0, gamma=0.1)
        elif attack == "cw":
            atk = torchattacks.CW(model, c=1, kappa=40, steps=50, lr=0.01)
        elif attack == "df":
            atk = torchattacks.DeepFool(model, steps=50, overshoot=epsilon)
        elif attack == "gaussian":
            atk = torchattacks.GN(model, std=epsilon)
        elif attack == "clean":
            atk = torchattacks.GN(model, std=0.0)
        elif attack == "pgd":
            atk = torchattacks.PGD(model, eps=epsilon, alpha=1/255, steps=10, random_start=True)
        elif attack == "sparse":
            atk = torchattacks.SparseFool(model, steps=10, lam=3, overshoot=0.02)

        atk.set_normalization_used(mean=[0.1307,], std=[0.3081,])
        perturbed_data = atk(data, target)
        
        # Clasificar la imagen perturbada
        output = model(perturbed_data)
        
        # Obtener probabilidades usando softmax
        probabilities = F.softmax(output, dim=1)
        final_pred = output.max(1, keepdim=True)[1]

        # Verificar si la predicción es correcta
        if final_pred.item() == target.item():
            correct += 1
            bok += 1

        # Almacenar outputs y etiquetas para ROC
        all_probs.append(output.detach().cpu().numpy()[0])
        all_labels.append(target.item())



    # Calcular accuracy final
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / 2000 = {final_acc}", flush=True)


    all_probs = torch.from_numpy(np.array(all_probs))
    all_labels = torch.from_numpy(np.array(all_labels))

    fpr, tpr, thresholds = roc(all_probs, all_labels)
    area = auc(all_probs,all_labels)
    return final_acc, adv_examples, accs, fpr, tpr, thresholds,area

# Lista de ataques a probar
lista = ['clean','gaussian','fgsm','jsma','cw','pgd','df'] #completa y en orden
#lista =['clean','fgsm', 'gaussian'] #quitar, solo lo puse para tener el dato sin ataque ya que no lo guardé

models= ['EfficientNet No CLIR', 'EfficientNet CLIR','MobileNet No CLIR', 'MobileNet CLIR', 'ResNet18 No CLIR', 'ResNet18 CLIR' ]
#models= ['EfficientNet No CLIR', 'EfficientNet CLIR' ]

# Diccionario para almacenar resultados
results = {}
k=0
for model in models:
    if model == 'EfficientNet No CLIR':
        #network = torch.load('/home/clirlab/xavier/CovLip/MNIST/Mnist_effi_no_CLIR_98.pth', weights_only=False)
        network = torch.load('/home/clirlab/xavier/CovLip/MNIST/Modelos_C/EfficientNet.pth', weights_only=False)

    if model == 'EfficientNet CLIR':
        #network = torch.load('/home/clirlab/xavier/CovLip/MNIST/Mnist_effi_CLIR_98.pth', weights_only=False)
        network = torch.load('/home/clirlab/xavier/CovLip/MNIST/Modelos_C/EfficientNet CLIR.pth', weights_only=False)

    if model == 'MobileNet No CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/MNIST/Mnist_mob_no_CLIR_98.pth', weights_only=False)
    if model == 'MobileNet CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/MNIST/Mnist_mob_CLIR_98.pth', weights_only=False)
    if model == 'ResNet18 No CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/MNIST/Mnist_res18_NO_CLIR_98.pth', weights_only=False)
    if model == 'ResNet18 CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/MNIST/Mnist_res18_CLIR_98.pth', weights_only=False)

    network.eval()

    for attack in lista:
        epsilons = 0.2
        accuracies = []
        examples = []
        auc_scores_list = []
        aurocs=[]
        fprs=[]
        tprs=[]
        now = datetime.datetime.now()

        print('--------> iniciando ataque_'+model+'-------> ', attack, flush=True)

        acc, ex, accse, fpr, tpr, thresholds,area  = testa(network, device, test_loader, epsilons, attack)

        #print ('fpr=',fpr)

        plt.plot(fpr, tpr,label=f"{attack} | {area:.2f}")


        fprs.append(fpr)
        tprs.append(tpr)

        aurocs.append(area)
        accuracies.append(acc)
        examples.append(ex)

        now = datetime.datetime.now()
        print(now, flush=True)


        texto='fprs_tprs/full_fprs_matrix_'+model[0:16]+'_'+attack+'.txt'
        np.savetxt(texto, fpr, fmt='%.3f')
        texto='fprs_tprs/full_tprs_matrix_'+model[0:16]+'_'+attack+'.txt'
        np.savetxt(texto, tpr, fmt='%.3f')
        '''
        # Guardar accuracy por batch
        texto = 'accuracies/full_results_matrix_'+model[0:16]+'_' + attack + '.txt'
        np.savetxt(texto, accse, fmt='%.3f')
        '''

        k=k+1;
    

    plt.legend(
    fontsize=12,                # Tamaño de fuente
    title='Attack | AUC',           # Título opcional
    title_fontsize=12,         # Tamaño del título
    #loc='upper right',         # Posición
    frameon=True,              # Marco alrededor
    fancybox=True,             # Marco con esquinas redondeadas
    #shadow=True                # Sombra
    )
    plt.title('Adversarial Attacks ROC - '+ model)
    #print('tipo final', type(tprs))

    plt.savefig('ROCS/full_ROC_matrix_'+model[0:18]+'.pdf', format="pdf",bbox_inches='tight')
    plt.close()
