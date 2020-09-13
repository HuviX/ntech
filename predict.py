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
    image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)    
    dataloader_dict = {'test':torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=4)}   
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

    file_names = np.hstack([os.listdir(path + '/female'), os.listdir(path + '/male')])
    predictions = []
    dataloader_dict, device = load_data(path, architecture)

    num_classes = 2
    model = model_init(architecture, num_classes)
    model = model.to(device)

    model.eval()    
    true_preds = 0
    i = 0
    for images, labels in dataloader_dict['test']:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, dim = 1)
            true_preds += torch.sum(preds == labels.data)
            predicted_class = preds.data.tolist()[0]
        predictions.append(mapping[predicted_class])
        i+=1
        print(f"{i}/{file_names.shape[0]}")
    print(true_preds.double() / len(dataloader_dict['test'].dataset)) 
    to_json = dict(zip(file_names, predictions))
    with open("predictions.json", "w") as outfile:  
        json.dump(to_json, outfile)
if __name__ == '__main__':
    main() 

