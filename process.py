import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import sys
import json
import argparse
from PIL import Image

num_classes = 2
size = 100

def model_init(num_classes = 2):
    model =  models.resnet34(pretrained = False) #Загрузка модели
    model_path = 'models/resnet34__19'
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) #меняем последний слой
    print(f"Model Weights from {model_path}")
    model.load_state_dict(torch.load(model_path)) #Обученная модель
    return model

def main():
    mapping = {0:'female', 1:'male'}
    path = sys.argv[1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Если есть возможность, то считать нужно на видеокарте.

    file_names = os.listdir(path)
    predictions = []
    #инициализация модели и её трансфер на GPU (если есть). 
    model = model_init(num_classes)
    model = model.to(device)
    model.eval()  
    #Создаём объект, который будет преобразовывать входные данные для подачи в сеть 
    data_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])
    ])
    to_json = {}
    i = 0
    #Итерация по всем файлам в папке, применение к ним преобразований и получение предсказаний от сети
    for name in file_names:
        image = Image.open(path + '/' + name)
        image = data_transforms(image)
        image = image.to(torch.device(device))
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            _, preds = torch.max(output, dim = 1)
            predicted_class = preds.data.tolist()[0]
        predictions.append(mapping[predicted_class])
        i+=1
        to_json[name] = mapping[predicted_class] 

    with open("process_results.json", "w") as outfile:  
        json.dump(to_json, outfile)
    print("Done")

if __name__ == '__main__':
    main() 

