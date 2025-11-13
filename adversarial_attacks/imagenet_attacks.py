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
import datetime

from torchmetrics import Accuracy, F1Score, ConfusionMatrix,MetricCollection
from torchmetrics.classification import MulticlassROC,MulticlassAUROC
from torchmetrics.classification import MulticlassConfusionMatrix

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device= ',device,flush=True )
if device=='cuda':
    print (torch.cuda.get_device_name(0))

print ("iniciando" , flush=True)

#%% LOAD DATA

print('load data')
DATA_DIM = 64 # height and width

transform = transforms.Compose(
    [transforms.Resize((DATA_DIM, DATA_DIM)),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

batch_size_train = 1024
batch_size_test=1

logi= int((10400/batch_size_train)/3)
print('logi= ', logi)



dataset = torchvision.datasets.ImageFolder(root='/home/clirlab/xavier/CovLip/imagenet', transform=transform)
print(len(dataset))
# print(dataset.imgs[0])
TRAIN_TEST_SPLIT = [0.5, 0.5]

trainset, testset = torch.utils.data.random_split(
    dataset,
    TRAIN_TEST_SPLIT) 
print(len(trainset))
print(len(testset))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=0, drop_last = True)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=False, num_workers=0, drop_last = True)

classes = ('penguin', 'Maltese', 'Panthera ', 'airliner',
           'dirigible', 'containership', 'soccer ball', 'sport car', 'trailer truck', 'orange')
print('fin load data')

roc = MulticlassROC(num_classes=10, thresholds=None,average='micro')
auc= MulticlassAUROC(num_classes=10, thresholds=None,average='macro')

#%% BUILD MODEL------------------------------------------------------------------------------------------
#aca vamos a cargar y modeificar el modelo para que pueda usarse en nuestra implementación
from torchvision.models import efficientnet_v2_s, densenet121,mobilenet_v3_large,inception_v3, resnet18, vgg16,convnext_tiny
from torchvision.models import ResNet18_Weights,MobileNet_V3_Large_Weights,EfficientNet_V2_S_Weights

#-----------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------

def testa( model, device, test_loader, epsilon, attack ):
    #print('device= ',device)
    # Accuracy counter
    correct = 0
    #batch correct
    bok=0
    adv_examples = []
    accs=[]
            # Listas para almacenar las probabilidades y etiquetas reales para ROC
    all_probs = []
    all_labels = []

    i=0

    # Loop over all examples in test set
    for data, target in test_loader:
        if (i%500 == 0 and i !=0): 
          b_acc=bok/1000
          accs.append(b_acc)
          print('i= ', i, flush=True)
          print(f"Batch Accuracy = {bok} / 1000 = {b_acc}")
          bok=0
          #break
        i+=1
        
        if i>2000:
            break
        
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)


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
            atk = torchattacks.GN(model, std=0.0)
        if attack == "pgd":
            atk = torchattacks.PGD(model, eps=epsilon, alpha=1/255, steps=10, random_start=True)
        if attack == "sparse":
            atk = torchattacks.SparseFool(model, steps=10, lam=3, overshoot=0.02) #lambda >= 1 va de 1,2,3,..,6 en el paper

        atk.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # If inputs were normalized, then
        perturbed_data = atk(data, target)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
                # Almacenar outputs y etiquetas para ROC
        all_probs.append(output.detach().cpu().numpy()[0])
        all_labels.append(target.item())

        if final_pred.item() == target.item():
            correct += 1
            bok+=1
         

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}", flush=True)

    #calculo de roc

    all_probs = torch.from_numpy(np.array(all_probs))
    all_labels = torch.from_numpy(np.array(all_labels))
    fpr, tpr, thresholds = roc(all_probs, all_labels)
    area = auc(all_probs,all_labels)

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples,accs,fpr, tpr, thresholds,area

    
lista = ['clean','gaussian','fgsm','jsma','cw','pgd','df'] #completa y en orden
#lista =['clean','gaussian'] #quitar, solo lo puse para tener el dato sin ataque ya que no lo guardé
models= ['EfficientNet No CLIR', 'EfficientNet CLIR','MobileNet No CLIR', 'MobileNet CLIR', 'ResNet18 No CLIR', 'ResNet18 CLIR' ]
#models= ['EfficientNet No CLIR', 'EfficientNet CLIR']


#network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Mobile_no_CLIR_90_CPU.pth', weights_only=False)
#network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Effi_No_CLIR_90.pth', weights_only=False)
#network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Effi_CLIR_90.pth', weights_only=False)
#network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Mob_No_CLIR_90.pth', weights_only=False)
#network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Mob_CLIR_90.pth', weights_only=False)
#network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_res_NO_Clir_90.pth', weights_only=False)
#network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_res_CLIR_90.pth', weights_only=False)

clir=True

for model in models:
    if model == 'cpu':
        network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Mobile_no_CLIR_90_CPU.pth', weights_only=False)

    if model == 'EfficientNet No CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Effi_No_CLIR_90.pth', weights_only=False)
    if model == 'EfficientNet CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Effi_CLIR_90.pth', weights_only=False)
    if model == 'MobileNet No CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Mob_No_CLIR_90.pth', weights_only=False)
    if model == 'MobileNet CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_Mob_CLIR_90.pth', weights_only=False)
    if model == 'ResNet18 No CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_res_NO_Clir_90.pth', weights_only=False)
    if model == 'ResNet18 CLIR':
        network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/image10_res_CLIR_90.pth', weights_only=False)
    

    #network = torch.load('/home/clirlab/xavier/CovLip/Imagenet_test/Image10_models/'+model, weights_only=False)
    network.eval()
    for attack in lista:
        epsilons = 0.2
        accuracies = []
        examples = []
        fprs=[]
        tprs=[]
        aurocs=[]

        now = datetime.datetime.now()
        print('--------> iniciando ataque '+model+'-------> ', attack, flush=True)
        print(now, flush=True)


        acc, ex,accse,fpr, tpr, thresholds, area = testa(network, device, testloader, epsilons,attack)
        plt.plot(fpr, tpr,label=f"{attack} | {area:.2f}")

        accuracies.append(acc)
        examples.append(ex)
        fprs.append(fpr)
        tprs.append(tpr)

        aurocs.append(area)
        print(accse, flush=True)
        now = datetime.datetime.now()
        print(now, flush=True)

        texto='fprs_tprs/full_fprs_'+str(clir)+'_CLIR_'+model[0:18]+'_'+attack+'.txt'
        np.savetxt(texto, fpr, fmt='%.3f')
        texto='fprs_tprs/full_tprs_'+str(clir)+'_CLIR_'+model[0:18]+'_'+attack+'.txt'
        np.savetxt(texto, tpr, fmt='%.3f')

    plt.legend(
        fontsize=12,                # Tamaño de fuente
        title='Attack | AUC',           # Título opcional
        title_fontsize=12,         # Tamaño del título
        #loc='upper right',         # Posición
        frameon=True,              # Marco alrededor
        fancybox=True,             # Marco con esquinas redondeadas
        #shadow=True                # Sombra
    )
    plt.title('Adversarial Attacks ROC - '+model)
    #print('tipo final', type(tprs))
    plt.savefig('ROCS/Full_ROC_'+model[0:18]+'.pdf', format="pdf",bbox_inches='tight')
    plt.close()



