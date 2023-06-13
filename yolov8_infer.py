import cv2
from ultralytics import YOLO
import os
from glob import glob
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", default='n', type=str)
parser.add_argument("--source", default='0', type=str)
args = parser.parse_args()

weight_path = os.path.join(os.getcwd(), f'yolov8{args.model_size}_ct.pt')
if os.path.isfile(weight_path):
    model = YOLO(weight_path)
print(f">>>> {args.model_size} size model loaded.")
    
if len(args.source) == 1:
    source = int(args.source)
else:
    source = os.path.join(os.getcwd(), args.source)
cap = cv2.VideoCapture(source)
print(f">>>> {args.source} video loaded. (integer number means camera source)")


OUTPUT_RESIZE_RATIO = 1.5

its = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # frame = cv2.resize(frame, (640, 640))
        st = time.time()
        results = model(frame)
        it = time.time() - st

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        reshape_size = (int(annotated_frame.shape[1]*OUTPUT_RESIZE_RATIO), int(annotated_frame.shape[1]*OUTPUT_RESIZE_RATIO))
        reshape_output = cv2.resize(annotated_frame, reshape_size)
        cv2.imshow("YOLOv8 Inference", reshape_output)
        print(f'{1/it:.2f} fps')
        its.append(it)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
if len(its) > 30:
    print(f'Averate Inference Time: {(sum(its[30:]) / (len(its)-30) * 1000):.2f} ms')
cap.release()
cv2.destroyAllWindows()
