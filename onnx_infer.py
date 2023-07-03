import cv2
import time
import numpy as np
from PIL import Image
import argparse
from glob import glob
import os
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", default='n', type=str)
parser.add_argument("--source", default='0', type=str)
parser.add_argument("--is_to_end", default='yes', type=str)
parser.add_argument("--is_show", default='yes', type=str)
args = parser.parse_args()

weight_path = os.path.join(os.getcwd(), f'yolov8{args.model_size}_ct.onnx')
if os.path.isfile(weight_path):
    net = cv2.dnn.readNetFromONNX(weight_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    print(f">>>> OpenCV version: {cv2.__version__}")
    print(f">>>> OpenCU CUDA Device Count: {cv2.cuda.getCudaEnabledDeviceCount()}")
print(f">>>> {args.model_size} size model loaded.")
    
current_time = time.time()
datetime_obj = datetime.fromtimestamp(current_time)
# datetime_obj = datetime_obj + timedelta(hours=9)
date_string = datetime_obj.strftime("%Y-%m-%d-%H%M%S")

if len(args.source) == 1:
    source = int(args.source)
    out_name = date_string
else:
    source = os.path.join(os.getcwd(), args.source)
    print(source)
    out_name = os.path.basename(source).split('.')[0] + '_' + date_string
    
cap = cv2.VideoCapture(source)
print(f">>>> {args.source} video loaded. (integer number means camera source)")

out_fps = cap.get(cv2.CAP_PROP_FPS)
out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if len(args.source) == 1:
    out_fps = 4.
    
out_dir = os.path.join(os.getcwd(), 'runs', 'onnx')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    
    



fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename=os.path.join(out_dir, out_name + '.mp4'), 
                      fourcc=fourcc, 
                      fps=out_fps, 
                      frameSize=(out_width, out_height))

# Define yolov8 classes
CLASSES_YOLO = ['BRAKE OFF', 'BRAKE ON']

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.35
OUTPUT_RESIZE_RATIO = 0.7
TEXT_BOX_HEIGHT_RATIO = 0.17
TEXT_BOX_WIDTH_RATIO = {
    # CLASSES_YOLO[0]:0.85,
    CLASSES_YOLO[0]:1.3,
    CLASSES_YOLO[1]:1.3,
}
TEXT_RATIO = 0.005


colors = {
    CLASSES_YOLO[0]:(23, 204, 146),
    CLASSES_YOLO[1]:(56, 56, 255),
}

count = 0

its = []
while cap.isOpened():
    success, image = cap.read()
    # print(image.shape)

    if success:

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        st = time.time()
        preds = net.forward()   # (1, 6, 8400)
        it = time.time() - st
        preds = preds.transpose((0, 2, 1))  # (1, 8400, 6)

        # Extract output detection
        class_ids, confs, boxes = list(), list(), list()

        image_height, image_width, _ = image.shape
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT

        rows = preds[0].shape[0] # 8400

        for i in range(rows):
            row = preds[0][i]   # (6,)
            # conf = row[4]

            classes_score = row[4:]
            _,_,_, max_idx = cv2.minMaxLoc(classes_score)
            class_idx = max_idx[1]   # higher class
            if (classes_score[class_idx] > SCORE_THRESHOLD):
                conf = classes_score[class_idx]
                confs.append(conf)
                label = CLASSES_YOLO[int(class_idx)]
                class_ids.append(label)

                #extract boxes
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        # r_class_ids, r_confs, r_boxes = list(), list(), list()

        indexes = cv2.dnn.NMSBoxes(boxes, confs, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for i in indexes:
            # r_class_ids.append(class_ids[i])
            # r_confs.append(confs[i])
            # r_boxes.append(boxes[i])

            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            cv2.rectangle(image, (left, top), (left + width, top + height), colors[class_ids[i]], 5)
            cv2.rectangle(image, (left-3, top - int(width*TEXT_BOX_HEIGHT_RATIO)), (left + int(width*TEXT_BOX_WIDTH_RATIO[class_ids[i]]), top), colors[class_ids[i]], -1)
            cv2.putText(image, f"{class_ids[i]}  {confs[i]:.2f}", (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, width*TEXT_RATIO, (255, 255, 255), 2)
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = cv2.resize(image, (int(image.shape[1]*OUTPUT_RESIZE_RATIO), int(image.shape[0]*OUTPUT_RESIZE_RATIO)))
        
        if args.is_show == 'yes':
            cv2.imshow("YOLOv8 Inference-ONNX", image)
        print(f'{1/it:.2f} fps')
        its.append(it)
        
        out_image = cv2.resize(image, (out_width, out_height))
        out.write(out_image)

        if args.is_show == 'yes':
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        if args.is_to_end != 'yes':
            count += 1
            if count > 100:
                break

    else:
        break


if len(its) > 30:
    print(f'Averate Inference Time: {(sum(its[30:]) / (len(its)-30) * 1000):.2f} ms')
cap.release()
out.release()
if args.is_show == 'yes':
    cv2.destroyAllWindows()
            

