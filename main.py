import torch
import cv2
import numpy as np
import argparse
from sort.sort import Sort

# Парсим аргументы командной строки
parser = argparse.ArgumentParser(description='People counting with YOLOv5 and SORT')
parser.add_argument('--source', type=str, default='0', 
                    help='Source video. Can be a file path or camera index (default: 0)')
args = parser.parse_args()

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
model.classes = [0]  # Отслеживаем только людей

# Инициализация трекера SORT
tracker = Sort()

# Захват кадра для выбора 4 точек (углов двери)
source = args.source
# Если источник - строка и состоит только из цифр, преобразуем в целое число (индекс камеры)
if source.isdigit():
    source = int(source)
    
cap = cv2.VideoCapture(source)
ret, frame = cap.read()
if not ret:
    print("Ошибка при захвате кадра!")
    cap.release()
    exit()

# Остальной код остается без изменений
# ...

# Перезапуск видеопотока для основного цикла
cap.release()
cap = cv2.VideoCapture(source)

# Остальная часть кода без изменений
