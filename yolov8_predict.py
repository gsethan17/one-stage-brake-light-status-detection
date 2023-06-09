import cv2
from ultralytics import YOLO
import os
from glob import glob

weight_path = os.path.join(os.getcwd(), 'runs', 'detect', 'train_l_300', 'weights', 'best.pt')
if os.path.isfile(weight_path):
    model = YOLO(weight_path)

    
video_paths = glob(os.path.join(os.getcwd(), 'test_video', '*.mp4'))

if len(video_paths) > 0:
    for video_path in video_paths:
        model.predict(video_path, save=True, imgsz=640, conf=0.5)