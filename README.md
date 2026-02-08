Проект по детекции транспортных средств с использованием модели **YOLOv11s**. 
Датасет: [Vehicle Detection Dataset на Kaggle](https://www.kaggle.com/datasets/daudshah/vehicle-detection-dataset).  
# Переходим в директорию с проектом
cd /path/to/obj

# Запуск обучения YOLOv11s с датасетом data.yaml
yolo detect train data=data.yaml model=yolo11s.pt epochs=50 imgsz=512
