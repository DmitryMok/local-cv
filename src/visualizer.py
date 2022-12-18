# классы и методы визуализации объектов
import os.path

import cv2
import numpy as np

def mean(a,b):
  return (a+b)//4

class Annotator:
    # визуализирует результаты распознавания
    def __init__(self):
        self.lw = 1  # lтолщина линии
        self.pict_list = [] # массив пиктограмм
        self.mask_list = [] # массив масок пиктограмм
        self.cmap_default = [(0,0,220), (0,150,0),
                             (0,150,0), (200,0,0),
                             (0,0,150), (0,200,200),
                             (0,0,200), (0,200,0),
                             (200,0,0), (200,200,0),
                             (0,0,150), (0,200,200),
                             (0,0,200), (0,200,0),
                             ]
        self.cmap_list = self.cmap_default.copy() # массив RGB цвета каждого класса
        self.sc = 4     # масштаб текста (чем больше, тем мельче)

    # если в функцию передано изображение, работаем с ним (для исключений)
    def setImage(self, im):
        if hasattr(im, "__len__"):  # если массив, значит передано изображение и работаем с ним
            self.im = im
            # print('if!!!')
            assert self.im.data.contiguous, 'Image not contiguous'

    # визуализирует реузльтаты распознавания (массив bouding boxes)
    def drawDetections(self, im, bboxes, bbox_type=1, lw=1, picts = False, conf_text = False):
        self.setImage(im)
        self.lw = lw
        self.picts = picts              # выводим пиктограммы
        self.conf_text = conf_text      # выводим уверенность
        for bbox in bboxes:
            self.drawBbox(bbox[:], bbox_type=bbox_type, color=self.cmap_list[int(bbox[-1])])
        return self.im

    # визуализирует bounding box
    def drawBbox(self, box, bbox_type=1, color=(128,128,128)):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        w, h = int(p2[0] - p1[0]), int(p2[1] - p1[1])
        dc = 5  # смещение пиктограммы от угла
        if bbox_type == 1:      # рамка
            cv2.rectangle(self.im, p1, p2, color=color, thickness=self.lw, lineType=cv2.LINE_AA)
        elif bbox_type == 2:    # окружность
            x, y = p1[0] + w // 2, p1[1] + h // 2
            cv2.circle(self.im, (x,y), h//2, (0, 255, 0), 1)
        elif bbox_type == 3:    # скобки
            ow = min(w,h)//4   # offset width
            # выводим рамку
            pts = np.array([[p1[0]+ow, p1[1]],[p1[0],p1[1]],[p1[0], p2[1]],[p1[0]+ow, p2[1]]])
            cv2.polylines(self.im, [pts], False, color, self.lw)
            pts = np.array([[p2[0]-ow, p2[1]],[p2[0],p2[1]],[p2[0], p1[1]],[p2[0]-ow, p1[1]]])
            cv2.polylines(self.im, [pts], False, color, self.lw)
        # вывод пиктограммы
        if self.picts:
            # выводим пиктограммы
            if self.pict_list:  # сначала обработать файлы пиктограмм функцией getPicts
                cl = int(box[-1]) # номер класса
                size_w = self.pict_list[cl].shape[0]
                size_h = self.pict_list[cl].shape[1]
                # print(-size_w + p1[0], -size_h + p1[1], 'cl=', cl)
                if ((-size_w + p1[1]) > 0 and (-size_h + p1[0]) > 0):
                    roi = self.im[-size_w + p1[1]:p1[1], -size_h + p1[0]:p1[0]]
                    roi[np.where(self.mask_list[cl])] = 0
                    roi += self.pict_list[cl]
        # вывод текста уверенности
        if self.conf_text:
            label = str(round(box[-2]-0.05,2))    # вероятность
            self.sc = 4  # масштаб текста
            w_text, h_text = cv2.getTextSize(label, 0, fontScale=self.lw / self.sc, thickness=1)[0]  # text width, height  # cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[0] - w_text >= 3
            p2 = p1[0]-w_text if outside else 0, p1[1]+h_text + 2
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label, (p1[0]-w_text, p1[1]+h_text - 0 if outside else p1[1] + h_text + 2),
                        0,
                        self.lw / self.sc,
                        (255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)

    # загружает пиктограммы, создает массив с пиктограммой и маской
    def setPicts(self, pict_files=None):
        self.pict_list, self.mask_list = [], []
        if pict_files:
            for fname in pict_files:
                logo = cv2.imread(os.path.join('.\img', fname))
                # size_w = logo.shape[0]
                # size_h = logo.shape[1]
                # создаем маску пиктограммы
                img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
                logo[np.where(mask)] = 0
                self.pict_list.append(logo) # пиктограмма с альфа каналом
                # инвертируем маску, чтобы фон имел значение 0
                mask = mask * (-1)
                mask += 255
                self.mask_list.append(mask) # маска альфа канала

