# Детекция транспортных средств — YOLOv11s

Проект по детекции транспортных средств с использованием модели **YOLOv11s**.  
Датасет: [Vehicle Detection Dataset на Kaggle](https://www.kaggle.com/datasets/daudshah/vehicle-detection-dataset).

## Обучение
yolo detect train data=data.yaml model=yolo11s.pt epochs=50 imgsz=512

## Валидация
yolo detect val model=weights/yolo_best.pt data=data.yaml split=test

## Демонстрация
python3 obj_det.py
