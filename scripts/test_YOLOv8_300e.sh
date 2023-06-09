#!/bin/bash

yolo task=detect mode=val model=./runs/detect/train_n_300/weights/best.pt data=./datasets/state-of-vehicle-tail-lamp-detection-test-night-2/data.yaml plots=True
yolo task=detect mode=val model=./runs/detect/train_s_300/weights/best.pt data=./datasets/state-of-vehicle-tail-lamp-detection-test-night-2/data.yaml plots=True
yolo task=detect mode=val model=./runs/detect/train_m_300/weights/best.pt data=./datasets/state-of-vehicle-tail-lamp-detection-test-night-2/data.yaml plots=True
yolo task=detect mode=val model=./runs/detect/train_l_300/weights/best.pt data=./datasets/state-of-vehicle-tail-lamp-detection-test-night-2/data.yaml plots=True
