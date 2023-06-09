#!/bin/bash

yolo task=detect mode=train model=yolov8n.pt data=./datasets/State-of-vehicle-tail-lamp-detection-2/data.yaml epochs=300 patience=20 batch=-1 plots=True optimizer=Adam lr0=0.01
yolo task=detect mode=train model=yolov8s.pt data=./datasets/State-of-vehicle-tail-lamp-detection-2/data.yaml epochs=300 patience=20 batch=-1 plots=True optimizer=Adam lr0=0.01
yolo task=detect mode=train model=yolov8m.pt data=./datasets/State-of-vehicle-tail-lamp-detection-2/data.yaml epochs=300 patience=20 batch=-1 plots=True optimizer=Adam lr0=0.01
yolo task=detect mode=train model=yolov8l.pt data=./datasets/State-of-vehicle-tail-lamp-detection-2/data.yaml epochs=300 patience=20 batch=-1 plots=True optimizer=Adam lr0=0.01
