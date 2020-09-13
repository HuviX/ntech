import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import pickle


data_dir = sys.argv[1]

model_name = 'inceptionv3'
num_classes = 2
batch_size = 128
num_epochs = 100
input_size = 299 #Необходимо, чтобы размер изображения по наименьшему краю был больше или равен 299.
feature_extract = True

def model_init():
    model = models.inception_v3(pretrained = True, aux_logits=False) #загружаем претрен 
    for param in model.parameters():
            param.requires_grad = False #делаем обучение только для последнего слоя.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_model(model, model_name, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    val_acc_history = []
    val_loss_history = []
    best_acc = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        for mode in ['train','val']:
            if mode == 'train':
                model.train()
            else:
                model.eval()
            ovr_loss = 0.0
            true_preds = 0
            for image, labels in dataloaders[mode]:
                image = image.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # Если валидация, то делам torch.no_grad()
                with torch.set_grad_enabled(mode == 'train'): #
                    outputs = model(image)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim = 1)
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                ovr_loss += loss.item() * image.size(0)
                true_preds += torch.sum(preds == labels.data)
            epoch_loss = ovr_loss/ len(dataloaders[mode].dataset)
            epoch_acc = true_preds.double() / len(dataloaders[mode].dataset)
            
            print(f"{mode} Loss: {epoch_loss}. Accuracy : {epoch_acc}")
            if mode == 'val':
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)

        name = f"models/{model_name}_{epoch}"
        torch.save(model.state_dict(), name)
    return model, val_acc_history, val_loss_history

def create_dataloader(data_dir):
    # Аугментации, чтобы модель была более устойчива к поворотам, отражениям и пр. 
    # Это также увиличивает размер тренировочной выборки 
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }
    image_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    train_size = int(0.8 * len(image_dataset)) #80 процентов в тренировочные данные, 20 в валидацию
    val_size = len(image_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(image_dataset, [train_size, val_size])

    dataloader_dict = {'train':torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                   'val':torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return dataloader_dict, device



def main():
    model = model_init()
    print("Model Loaded")
    print(model) #смотрим на архитектуру модели
    dataloader_dict, device = create_dataloader(data_dir=data_dir)
    model = model.to(device)
    params_to_update = model.parameters()

    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    
    optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()
    model, acc_hist, loss_hist = train_model(model, model_name, dataloader_dict, criterion, optimizer_ft, scheduler, num_epochs, device)
    with open('outfile', 'wb') as fp:
        pickle.dump(acc_hist, fp)
    with open('outfile2', 'wb') as fp:
        pickle.dump(loss_hist, fp)

    plt.plot(acc_hist)
    plt.savefig('acc_hist.png')
    plt.show()
    
    plt.plot(loss_hist)
    plt.savefig('loss_hist.png')
    plt.show()
    print("Done")
if __name__ == '__main__':
    main()

