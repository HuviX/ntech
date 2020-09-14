# ntech

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


# Описание
В данном репозиитории находятся решения для заданий от NtechLab.

## #1. Maximum Subarray
В первом задании было необходимо найти непрерывный подмассив в исходном массиве (списке), такой что сумма элементов подмассива максимальна. Для решения использовался
Kadane's Algorithm. Файл `max_subarray.py`.

## #2. Нейросеть для классификации пола по фотографии
В задании #2 необходимо обучить нейросеть определять по фотографии пол человека.

### Решение:
1. В качестве архитектуры была выбрана [ResNet34](https://arxiv.org/abs/1512.03385) обученная на ImageNet. В pytorch можно взять предобученную модель и поменять последний полносвязный слой таким образом, чтобы
получить необходимое число классов(Transfer Learning). В нашем случае, количество выходов из полносвязного слоя равно 2.

2. Было дано 100.000 фотографий. 50.000 фотографий лица мужчин и столько же фотографий лица женщин. В качесте тестовой выборки я сразу отложил по 3 тысячи фотогорафий каждого класса. Затем, от оставшихся данных 80% поместил в обучающую выборку и 20% в валидационную.

3. В качестве оптимизатора был выбран SGD с параметром темпа обучения(learning rate) 0.01 и momentum 0.9. Т.к. не было уверенности, что параметр темпа обучения оптимален, то был использован инструмент lr_scheduler.ReduceLROnPleateau. Принцип работы заключается в отслеживании ключевой метрики и изменения параметра learning rate.
Например, если оптимизируемая метрика не понижается в ходе обучения (при том, что цель - минимизация данной метрики), то lr изменяется в n раз. Обычно n = 0.1. В таком случае, оптимизатор будет медленнее двигаться к оптимуму(минимуму или максимуму), но при этом вероятность попасть в более удачный из них повышается.

4. В ходе обучения использовались аугментации (вертикальное отражение с вероятность 0.5 и повороты от -30 до 30 градусов).

### Запуск обучения и проверка качества нейросети:

Чтобы проверить работу, необходимо выполнить команду `python3 process.py \path\to\test\`. При этом в папке со скриптом создастся файл .json, который имеет структуру - 
`{'filename_i':'class_i'}`.

Для запуска обучения необходимо выполнить `python3 train34.py --path \path\to\train`. При этом можно задать дополнительные параметры, количество эпох и размер батча `python3 train34.py --path \path\to\train --n_epochs 10 --batch 128`. В ходе обучения в папку *logs* помешается информация об обучение, и можно наблюдать за ходом обучения в режиме реального времени. Для этого необходим tensroboard.
Если нет, то `pip install tensorboard`. И после запуска обучения выполнить `tensorboard --logdir=logs/ --bind_all`.

### Качество модели
Модель обучалась 20 эпох, размер батча 128. Ниже представлены графики из tensorboard, показывающие изменение метрик
![значение точности на обучающей и валид выборках](https://github.com/HuviX/ntech/blob/master/pics/resnet_accuracy.png)
![значение функции потерь (кроссэнтропия) на обучающей и валид выборках](https://github.com/HuviX/ntech/blob/master/pics/loss_resnet.png)

В результате обучения точность достигла 96.44% на обучающих данных и 96.23% на валидационных.

файлы `tfevents` представлены в папке `logs`.

На отложенных 6к фотографиях модель показывает точность 98%. (:open_mouth:)
![значение точности на тесте](https://github.com/HuviX/ntech/blob/master/pics/test_acc.png)

### P.S:
Также тестировалась сеть [InceptionV3](https://arxiv.org/abs/1512.00567). Но для обучения требуется больше вычислительных ресурсов, т.к. необходимое минимальное разрешение входных данных - 299 х 299. Это значительно влияет на скорость обучения. 

