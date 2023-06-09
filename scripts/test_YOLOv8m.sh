#!/bin/bash

yolo task=detect mode=val model=./runs/detect/train_m_30/weights/best.pt data=./datasets/state-of-vehicle-tail-lamp-detection-test-night-2/data.yaml plots=True