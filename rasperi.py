import cv2
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import numpy as np

# Фильтр Калмана
def init_kalman_filter(bbox):
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.eye(7)
    kf.H = np.zeros((4, 7))
    kf.H[:4, :4] = np.eye(4)
    kf.P *= 10.0
    kf.R *= 1.0
    kf.Q *= 0.01
    kf.x[:4] = np.reshape(bbox, (4, 1))
    return kf


# Обновления треков
def update_tracks(tracks, detections):
    for track in tracks:
        track["kf"].predict()
        track["updated"] = False

    for det in detections:
        if len(tracks) > 0:
            distances = [np.linalg.norm(track["kf"].x[:4] - det) for track in tracks]
            closest_track = np.argmin(distances)
            if distances[closest_track] < 50:
                tracks[closest_track]["kf"].update(det)
                tracks[closest_track]["updated"] = True
            else:
                tracks.append({"bbox": det, "kf": init_kalman_filter(det), "updated": True})
        else:
            tracks.append({"bbox": det, "kf": init_kalman_filter(det), "updated": True})

    tracks[:] = [track for track in tracks if track["updated"]]

# Обработка видео
def detect_and_track_from_video(video_path, output_path):
    # Загрузка модели YOLO
    model = YOLO("yolov5s.pt")

    # Треки объектов
    tracks = []

    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обнаружение объектов
        results = model(frame)
        detections = []

        # Извлечение координат класс "человек"
        for res in results[0].boxes:
            if int(res.cls) == 0:
                bbox = res.xyxy[0].cpu().numpy()
                detections.append(bbox)

        # Обновление треков
        update_tracks(tracks, detections)

        # Отображение треков на кадре
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["kf"].x[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Отображение кадра с помощью OpenCV
        cv2.imshow("Frame", frame)

        # Выход из видео, если нажата клавиша 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Запись кадра в файл
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Запуск обработки видео (указать путь)
video_path = "PATH"
output_path = "OUTPUT_PATH"
detect_and_track_from_video(video_path, output_path)
