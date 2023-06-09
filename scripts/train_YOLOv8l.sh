#!/bin/bash

yolo task=detect mode=train model=yolov8l.pt data=./datasets/State-of-vehicle-tail-lamp-detection-2/data.yaml epochs=100 patience=10 batch=-1 plots=True