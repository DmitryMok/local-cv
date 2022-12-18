# классы и методы распознавания объектов
import os
import torch
# import numpy as np
import yaml
import sys

class _Model:
    def __init__(self, weights, classes, description, menu_name):
        self.weights = weights
        self.classes = classes
        self.description = description
        self.menu_name = menu_name
        self.picts = []
        self.cmap = []


class DetectorYolo:
    # класс для загрузки конфигурации и весов моделей
    def __init__(self):
        # загрузка моделей
        self.loaded = False  # статус загрузки моделей
        self.wdir = '.\weights'  # папка с весами yolo
        self.models = []  # здесь будут загруженные модели

    # загружаем веса yolo из папки self.wdir = './weights'
    def loadWeights(self, config_file='weights.yaml', progress=False):
        self.config_file = config_file  #
        maxnm = 3   # максимальное количество загружаемых моделей
        try:
            self.loadConfig()
        except:
            print(f'Проверьте файл конфигурации {os.path.abspath(os.path.join(self.wdir, self.config_file))}')
            # sys.exit()
            return 'exit'
        # загружаем по очереди файлы с весами
        for i, mdl in enumerate(self.models):
            path = os.path.join(self.wdir, mdl.weights)
            if os.path.exists(path) and i <= maxnm:  # добавляем условие для загрузки не больше трех моделей
                print(f'Файл {mdl.weights} существует, загружаем')
                model = torch.hub.load('yolov5-master', 'custom',
                                       path=path,
                                       source='local')
                mdl.model = model
                # self.models.append(model)
                if progress:  progress(int(90/min(len(mdl.weights),maxnm)*i+10))
            else:
                print(f'Файл {mdl.weights} не существует, пропускаем')
                del self.models[i]
        self.loaded = True  # флаг, что модели загрузились
        print('Результат загрузки моделей:')
        print(f'из списка {[k["weights"] for k in self.config]} \nзагружены {[k.weights for k in self.models]}')
        return [k.menu_name for k in self.models]

    # загружаем параметры конфигурации из файла self.wdir = './weights/weights.yaml'
    def loadConfig(self):
        with open(os.path.join(self.wdir, self.config_file)) as fh:
            read_data = yaml.load(fh, Loader=yaml.FullLoader)
        # парсер конфигурационного файла
        for k in read_data:
            # print(k.keys())
            model = _Model(k['weights'], k['classes'], k['description'], k['menu_name'])
            # print(k['name'])
            if 'picts' in k.keys():
                model.picts = k['picts']
            else:
                model.picts = []
            if 'cmap' in k.keys():
                # model.cmap = k['cmap']
                for p in k['classes'].keys():
                    h = k['cmap'][p].lstrip('#')
                    model.cmap.append(tuple(int(h[i:i + 2], 16) for i in (4, 2, 0)))
                else: model.cmap = []
            self.models.append(model)   # массив моделей

        # self.weights = [k['name'] for k in read_data]
        self.config = read_data  # сохраняем, чтобы брать описание и т.п.

    # делаем распознавание (на вход изображение и индекс модели, возвращает массив с координатами рамок)
    def getDetect(self, image, model_index=0):
        res = self.models[model_index].model(image)
        # print(self.class_filter)
        # if len(self.class_filter[model_index]):
        #     res = res[[torch.isin(res[...,4], self.class_filter[model_index])]]
        # print('xywh1')
        # print('xywh', res.xywh[0])
        # print('xywh1')
        return res.xyxy[0].cpu().detach().numpy()
        # return res.xyxy[0]
