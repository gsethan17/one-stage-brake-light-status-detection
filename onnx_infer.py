import cv2
import time
import numpy as np
from PIL import Image
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", default='n', type=str)
parser.add_argument("--source", default='0', type=str)
args = parser.parse_args()

weight_path = os.path.join(os.getcwd(), f'yolov8{args.model_size}_ct.onnx')
if os.path.isfile(weight_path):
    net = cv2.dnn.readNetFromONNX(weight_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    print(f">>>> OpenCV version: {cv2.__version__}")
    print(f">>>> OpenCU CUDA Device Count: {cv2.cuda.getCudaEnabledDeviceCount()}")
print(f">>>> {args.model_size} size model loaded.")
    
if len(args.source) == 1:
    source = int(args.source)
else:
    source = os.path.join(os.getcwd(), args.source)
cap = cv2.VideoCapture(source)
print(f">>>> {args.source} video loaded. (integer number means camera source)")


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
OUTPUT_RESIZE_RATIO = 0.7

# Define yolov8 classes
CLASESS_YOLO = ['On', 'Off']

its = []
while cap.isOpened():
    success, image = cap.read()
    # print(image.shape)

    if success:

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        st = time.time()
        preds = net.forward()
        it = time.time() - st
        preds = preds.transpose((0, 2, 1))

        # Extract output detection
        class_ids, confs, boxes = list(), list(), list()

        image_height, image_width, _ = image.shape
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT

        rows = preds[0].shape[0]

        for i in range(rows):
            row = preds[0][i]
            conf = row[4]

            classes_score = row[4:]
            _,_,_, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if (classes_score[class_id] > .25):
                confs.append(conf)
                label = CLASESS_YOLO[int(class_id)]
                class_ids.append(label)

                #extract boxes
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = list(), list(), list()

        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45)
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i])
            r_boxes.append(boxes[i])

            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            cv2.rectangle(image, (left, top), (left + width, top + height), (0,255,0), 3)
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = cv2.resize(image, (int(image.shape[1]*OUTPUT_RESIZE_RATIO), int(image.shape[0]*OUTPUT_RESIZE_RATIO)))
        cv2.imshow("YOLOv8 Inference-ONNX", image)
        print(f'{1/it:.2f} fps')
        its.append(it)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break


if len(its) > 30:
    print(f'Averate Inference Time: {(sum(its[30:]) / (len(its)-30) * 1000):.2f} ms')
cap.release()
cv2.destroyAllWindows()
            

