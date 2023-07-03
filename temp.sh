#!/bin/bash

wget -O test_video.mp4 https://www.dropbox.com/s/mesjnk350f0yio8/day_city.mp4?dl=0
wget -O yolov8x_ct.onnx https://www.dropbox.com/s/9mka105rpoax8wy/yolov8x_ct.onnx?dl=0
python3 onnx_infer.py --source test_video.mp4  --model_size x --is_show no --is_to_end no