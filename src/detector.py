# классы и методы распознавания объектов
import os
import torch

class DetectorYolo():

    # constructor
    def __init__(self, models_config):
        # загрузка моделей
        self.loaded = False     # статус загрузки моделей
        self.wdir = './weights' # папка с весами yolo
        self.models = []        # здесь будут загруженные модели
        # в будущем классы модели будут загружаться из файла (ToDo)
        self.config_file = models_config
        self.models_config = {'heridal_070922.pt':{0: 'person'},
                              'land_full_270922.pt':{0:'man', 1:'car', 2:'car', 3:'bike'},
                              'mil_111022.pt':{0: 'arms1', 1: 'arms2', 2: 'transport', 3: 'equip', 4: 'man',
                                               5: 'plane', 6: 'warship', 7: 'boat', 8: 'plane'}}
        self.weights = list(self.models_config.keys())
        self.class_names = [value for key, value in self.models_config.items()]

    # загружаем веса yolo из папки self.wdir = './weights'
    def load_weights(self):
        for i, wfile in enumerate(self.weights):
            path = os.path.join(self.wdir,self.weights[i])
            if os.path.exists(path):
                print(f'Файл {self.weights[0]} существует, загружаем')
                model = torch.hub.load('yolov5-master', 'custom',
                                       path=path,
                                       source='local')
                self.models.append(model)
            else:
                print(f'Файл {self.weights[0]} не существует')
        self.loaded = True

        return self.models
