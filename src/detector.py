# классы и методы распознавания объектов
import os
import torch
import numpy as np

class DetectorYolo():

    # constructor
    def __init__(self):
        # загрузка моделей
        self.loaded = False     # статус загрузки моделей
        self.wdir = './weights' # папка с весами yolo
        self.models = []        # здесь будут загруженные модели

    # загружаем веса yolo из папки self.wdir = './weights'
    def load_weights(self, models_config):
        self.config_file = models_config 	# не используется, в будущем классы модели будут загружаться из файла (ToDo)
        # self.models_config = {'heridal_070922.pt':{0: 'person'}}    # только одна модель для тестов, чтобы быстрее загружалась
        self.models_config = {'heridal_070922.pt':{0: 'person'},
                              'land_full_270922.pt':{0:'man', 1:'car', 2:'car', 3:'bike'},
                              'mil_111022.pt':{0: 'arms', 1: 'arms', 2: 'transport', 3: 'equip', 4: 'man',
                                               5: 'plane', 6: 'warship', 7: 'boat', 8: 'plane'},
                              'VisBig_270922.pt':{0:'man', 1:'car', 2:'car', 3:'bike'}}

        self.class_filter = [[],[0,1,2],[0,1,2,3,4],[]]   # какие классы фильтруем (оставляем)
        # self.class_filter = [torch.tensor(f) for f in self.class_filter]
        self.weights = list(self.models_config.keys())
        self.class_names = [value for key, value in self.models_config.items()]

        # Проверяем наличие файлов с весами и загружаем
        for i, wfile in enumerate(self.weights):
            path = os.path.join(self.wdir,self.weights[i])
            if os.path.exists(path):
                print(f'Файл {wfile} существует, загружаем')
                model = torch.hub.load('yolov5-master', 'custom',
                                       path=path,
                                       source='local')
                self.models.append(model)
            else:
                print(f'Файл {wfile} не существует, пропускаем')
        self.loaded = True	# флаг, что модели загрузились

        # return self.models

	# делаем распознавание
    def detect(self, image, model_index=0):
        res = self.models[model_index](image)
        # print(self.class_filter)
        # if len(self.class_filter[model_index]):
        #     res = res[[torch.isin(res[...,4], self.class_filter[model_index])]]
        return res.xyxy[0].cpu().detach().numpy()
        # return res.xyxy[0]