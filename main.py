import torch
import cv2
import numpy as np
from sort.sort import Sort

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
model.classes = [0]  # Отслеживаем только людей

# Инициализация трекера SORT
tracker = Sort()

# Захват кадра для выбора 4 точек (углов двери)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Ошибка при захвате кадра!")
    cap.release()
    exit()

# Копия кадра для рисования выбранных точек
points = []

def click_event(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        # Рисуем небольшую окружность в точке клика
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("People Counter", frame)

# Используем то же окно "People Counter" для отображения кадра и выбора точек
cv2.namedWindow("People Counter")
cv2.imshow("People Counter", frame)
cv2.setMouseCallback("People Counter", click_event)

# Ожидание выбора 4 точек (или выход по 'q')
while len(points) < 4:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("People Counter")
src_pts = np.array(points, dtype="float32")
print("Выбранные точки:", src_pts)

# Определяем размеры выходного (bird's eye view) изображения.
width, height = 300, 400
dst_pts = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

# Вычисляем матрицу перспективного преобразования
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
# Определяем пороговую линию в bird's eye view (например, горизонтальная линия посередине)
threshold_line_y = height // 2

# Перезапуск видеопотока для основного цикла
cap.release()
cap = cv2.VideoCapture(0)

# Счетчики входов/выходов и история движения (для каждого трека в bird's eye view)
entrances = 0
exits = 0
motion_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    
    # Обнаружение людей с помощью YOLOv5
    with torch.no_grad():
        results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    dets = []
    for *xyxy, conf, cls in detections:
        if conf > 0.4:
            dets.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])
    dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
    
    # Обновление трекера SORT
    tracks = tracker.update(dets)
    
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Преобразование центра в bird's eye view
        center = np.array([[[center_x, center_y]]], dtype="float32")
        center_bev = cv2.perspectiveTransform(center, M)[0][0]
        bev_x, bev_y = int(center_bev[0]), int(center_bev[1])
        
        # Обновление истории движения для данного трека (используем координату Y в bird's eye view)
        motion_history[track_id] = motion_history.get(track_id, []) + [bev_y]
        if len(motion_history[track_id]) > 2:
            prev_bev_y = motion_history[track_id][-2]
            # Если объект движется сверху вниз через пороговую линию – считаем, что он вошёл
            if prev_bev_y < threshold_line_y and bev_y >= threshold_line_y:
                entrances += 1
                print(f"Человек {track_id} вошел")
                motion_history[track_id] = []  # Сброс истории
            # Если объект движется снизу вверх через пороговую линию – считаем, что он вышел
            elif prev_bev_y > threshold_line_y and bev_y <= threshold_line_y:
                exits += 1
                print(f"Человек {track_id} вышел")
                motion_history[track_id] = []  # Сброс истории
        
        # Рисуем bounding box и ID на исходном кадре
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Отрисовка центра и его проекции (для отладки)
        cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
        cv2.circle(frame, (bev_x, bev_y), 3, (0, 0, 255), -1)
    
    # Отрисовка пороговой линии (ориентир – логика подсчёта ведётся в bird's eye view)
    cv2.line(frame, (0, threshold_line_y), (frame.shape[1], threshold_line_y), (0, 0, 255), 2)
    
    cv2.putText(frame, f"Entered: {entrances}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Exited: {exits}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
