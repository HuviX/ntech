import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import time
import os
import sys
import pickle
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', help='epochs', default = 10)
parser.add_argument('--path', help = 'train dataset path')
parser.add_argument('--batch', help = 'batch size', default = 64)
args = parser.parse_args()

num_epochs = int(args.n_epochs)
data_dir = args.path
batch_size = int(args.batch)

model_name = 'resnet34_'
num_classes = 2
input_size = 100 
feature_extract = True

def model_init():
    model = models.resnet34(pretrained = True) #загружаем предобученную модель
    for param in model.parameters():
            param.requires_grad = True #размораживаем обучение для всех слоёв.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # меняем последний слой
    return model

def train_model(model, model_name, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    val_acc_history = []
    val_loss_history = []
    writer = SummaryWriter('logs/') # в логах будет сохраняться информация об обучении и можно наблюдать за обучением в Tensorboard
    # tensorboard --logdir=logs/
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
                images = image.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # Если валидация, то делам torch.no_grad()
                with torch.set_grad_enabled(mode == 'train'): 
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim = 1)
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                ovr_loss += loss.item() * images.size(0)
                true_preds += torch.sum(preds == labels.data)

            epoch_loss = ovr_loss/ len(dataloaders[mode].dataset)
            epoch_acc = true_preds.double() / len(dataloaders[mode].dataset)
            writer.add_scalar(f"Loss/{mode}", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/{mode}", epoch_acc, epoch)
            print(f"{mode} Loss: {epoch_loss}. Accuracy : {epoch_acc}")
            #Смотрим на функцию потерь на валидации и передаём значение в scheduler.
            if mode == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)

        name = f"models/{model_name}_{epoch}"
        torch.save(model.state_dict(), name)
        writer.close()
    return model, val_acc_history, val_loss_history

def create_dataloader(data_dir):
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)), 
        transforms.RandomResizedCrop(input_size),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]) # нормализация данных
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

    dataloader_dict, device = create_dataloader(data_dir=data_dir) #создание объекта, который применит к входным данным необходимые преобразование и сможет итерироваться через них
    model = model.to(device) #перекладываем модель на GPU (если оно есть), чтобы ускорить обучение

    params_to_update = model.parameters() #необходимо получить список параметров модели, чтобы отследить, точно ли будет происходит обучение только последнего слоя
    print("Params to learn:") 
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t",name)
    #создание оптимизатора SGD с параметром LearningRate = 0.01 и momentum = 0.9
    optimizer= optim.SGD(params_to_update, lr=0.01, momentum=0.9)
    #Создание объекта класса ReduceLROnPlateau. Цель - следить за метрикой, которая передается, и если в течении 2х эпох не происходит уменьшение(т.к. mode = 'min'), 
    # то новое значение lr = lr * 0.1 (factor).
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    #Т.к. задача бинарной классификации, то можно минимизировать КроссЭнтропию с целью повышение качества модели (в качестве метрики качества выступает Accuracy)
    criterion = nn.CrossEntropyLoss()
    model, _, _ = train_model(model, model_name, dataloader_dict, criterion, optimizer, scheduler, num_epochs, device)
    print("Model Training Finished")

if __name__ == '__main__':
    main()
