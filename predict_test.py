import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import sys
import json
import argparse
from PIL import Image

def model_init(architecture, num_classes = 2):
    model = models.inception_v3(pretrained = True, aux_logits = True) if architecture == 'inception' else models.resnet34(pretrained = True) #загружаем претрен
    model_path = 'models/'+architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model_path = 'pretrained_models/'+str(architecture) + '_no_aug'
    print(f"Model Weights from {model_path}")
    model.load_state_dict(torch.load(model_path))
    return model

def load_data(data_dir, architecture):
    size = 299 if architecture == 'inception' else 100
    data_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])
    ])
    image = Image.open(data_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return dataloader_dict, device

def main():
    mapping = {0:'female', 1:'male'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', help='net architecture', default = 'resnet')
    parser.add_argument('--path', help = 'test dataset path')
    args = parser.parse_args()
    path = args.path
    architecture = args.net


    file_names = os.listdir(path)
    predictions = []
    #dataloader_dict, device = load_data(path, architecture)
    num_classes = 2
    model = model_init(architecture, num_classes)
    model = model.to(torch.device("cuda:0"))
    size = 299 if architecture == 'inception' else 100
    data_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])
    ])
    model.eval()   
    to_json = {}
    i = 0
    for name in file_names:
        image = Image.open(path + '/' + name)
        image = data_transforms(image)
        image = image.to(torch.device("cuda:0"))
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            _, preds = torch.max(output, dim = 1)
            predicted_class = preds.data.tolist()[0]
        predictions.append(mapping[predicted_class])
        i+=1
        print(f"{i}/{len(file_names)}")
        to_json[name] = mapping[predicted_class]
    #print(true_preds.double() / len(dataloader_dict['test'].dataset)) 
    
    with open("predictions.json", "w") as outfile:  
        json.dump(to_json, outfile)

if __name__ == '__main__':
    main() 

