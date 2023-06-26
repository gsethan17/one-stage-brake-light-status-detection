import cv2
import time
from datetime import datetime
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", default='0', type=str)
parser.add_argument("--is_show", default='yes', type=str)
args = parser.parse_args()

current_time = time.time()
datetime_obj = datetime.fromtimestamp(current_time)
date_string = datetime_obj.strftime("%Y-%m-%d-%H%M%S")

dic_source = {
    0:'logitec_',
    1:'logitec_',
    2:'realSense_'
}

source = int(args.source)
out_name = dic_source[source] + date_string
cap = cv2.VideoCapture(source)

out_fps = cap.get(cv2.CAP_PROP_FPS)
out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_dir = os.path.join(os.getcwd(), 'data', 'test_video')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename=os.path.join(out_dir, out_name + '.mp4'), 
                      fourcc=fourcc, 
                      fps=out_fps, 
                      frameSize=(out_width, out_height))

while cap.isOpened():
    success, image = cap.read()
    # print(image.shape)

    if success:
        out.write(image)
        cv2.imshow(f"{dic_source[source]}image", image)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
cap.release()
out.release()
cv2.destroyAllWindows()