# local-cv
<h2>Распознавание и визуализация</h2>

<h3>Структура папок проекта:</h3>
корень
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ weights/
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ src/
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ img/




<h3>Как использовать:</h3>

<p>Создаем конфигурационный файл <b>.yaml<b>, сохраняем в папку <b>/weights<b></p>


```yaml
- weights: VisBig_270922.pt
  classes:
    0: human
    1: car
    2: truck
    3: bike
  description: 'Модель для распознавания людей и автомобилей'
  menu_name: 'Люди и автомобили'
  
  # необязательные параметры
  # пиктограммы в папке ./img, рекомендуемый размер 32px
  picts: ['human.png', 'car.png', 'truck.png', 'bike.png']
  # цвет для рамки каждого класса
  cmap: ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']
```

<p>Пример рабочего кода</p>

```python
import cv2

# загружаем модули из папки src
import src.detector as Detector
import src.visualizer as Visualizer

# класс DetectorYolo содержит все необходимые методы для загрузки весов и распознавания
det = Detector.DetectorYolo()

# загружаем веса из папки weights. 
# загруженная модель будет храниться в объекте det
det.loadWeights('weights-test.yaml')

# класс Annotator служит для разметки изображений
vis = Visualizer.Annotator()

# готовим пиктограммы (метод преобразует их в маску)
# вариант 1 - если пиктограммы были указаны в конфиге и уже загружены
vis.setPicts(pict_files=det.models[0].picts)
# вариант 2 - если не были указаны в конфиге, или надо изменить
vis.setPicts(pict_files=['human.png', 'car.png', 'truck.png', 'bike.png'])

# загружаем изображение для проверки обнаружения
image = cv2.imread('img/image.jpg')

# делаем обнаружение, метод вернет numpy массив bounding box с абсолютными координатам углов x1y1-x2y2, уверенностью и класс объекта, см. пример
# [[     314.13      542.76      354.79       641.6     0.90917           0]
# [     118.47      608.15      161.96      702.94     0.86808           0]]
res = det.getDetect(image)

# тип разметки - bbox_types:
# 1 - обычная рамка
# 2 - окружность
# 3 - скобки
# picts - нарисовать иконку, conf_text - отобразить уверенность
image = vis.drawDetections(image, res[:], bbox_type=3, lw=2, picts = True, conf_text = True)

# выводим результат
cv2.imshow('Image',image)

cv2.waitKey(0)
```
