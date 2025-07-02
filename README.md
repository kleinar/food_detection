# YOLOv8 Food Detection Project

## 📦 Установка зависимостей

Создайте виртуальное окружение (рекомендуется) и установите зависимости:

```bash
pip install -r requirements.txt
```

## 📁 Структура проекта

```
скачать датасет по ссылке 

project_root/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── weights/
│   └── best.pt              # Предобученные веса
├── images_for_test/         # Картинки для инференса
├── graphs/                  # Сюда сохраняются графики
├── train.py                 # Обучение модели
├── infer.py                 # Инференс
├── analyze.py               # Построение графиков по классам
├── data.yaml                # Описание датасета
└── requirements.txt
```

## 🚀 Обучение

```bash
python train.py
```

Результаты сохранятся в папке `runs/train/food_detector`.

## 📊 Просмотр метрик на тестовой выборке

После обучения можешь запустить в Python:

```python
from ultralytics import YOLO
model = YOLO("runs/train/food_detector/weights/best.pt")
model.val()
```

Метрики (Precision, Recall, mAP) будут напечатаны в консоль.

## 🔍 Инференс на изображениях

Положи изображения в папку `images_for_test/` и запусти:

```bash
python infer.py
```

Результаты сохранятся автоматически в `runs/detect/`.

## 📈 Построение графиков распределения классов

```bash
python analyze.py
```

Графики будут сохранены в папку `graphs/`.