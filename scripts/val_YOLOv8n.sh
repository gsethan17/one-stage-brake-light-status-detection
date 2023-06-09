#!/bin/bash

yolo task=detect mode=val model=./runs/detect/train_n_30/weights/best.pt data=./datasets/State-of-vehicle-tail-lamp-detection-2/data.yaml plots=True