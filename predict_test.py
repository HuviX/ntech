from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import json

"""
Script to evaluate model quality with known labels
"""
def model_init(num_classes = 2):
    model = models.resnet34(pretrained = False) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # меняем последний слой
    model.load_state_dict(torch.load("models/resnet34__19"))
    return model

def load_data(data_dir):
    data_transforms = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]) 
    ])
    image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)    
    dataloader_dict = {'test':torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=4)}   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return dataloader_dict, device

def main():
    
    path = sys.argv[1]
    dataloader_dict, device = load_data(path)
    model = model_init()
    model = model.to(device)
    model.eval()
    true_preds = 0
    #Зная лейблы для каждого изображения можно посчитать точность.
    for images, labels in dataloader_dict['test']:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, dim = 1)
            true_preds += torch.sum(preds == labels.data)
    print(true_preds.double() / len(dataloader_dict['test'].dataset)) 

if __name__ == '__main__':
    main() 

