import argparse
import cv2
import detect_utils
import time
import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/vide')
parser.add_argument('-s', '--min-size', dest='min_size', default=800,
                    help='minimum input size for the FasterRCNN network')
parser.add_argument('-m', '--model', dest='model', help='select model (mobilenetv3/resnet50)')
args = vars(parser.parse_args())

if args['model'] == 'resnet50':
    print("ResNet50")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=args['min_size'])
elif args['model'] == 'mobilenetv3':
    print("MobileNetv3")
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, min_size=args['min_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video')

# get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"

# define codec
out = cv2.VideoWriter(f"output/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))

frame_count = 0
total_fps = 0

model = model.eval().to(device)

# read until end video
while(cap.isOpened()):
    # capture each frame in video
    ret, frame = cap.read()
    if ret == True:
        # get start time
        start_time = time.time()
        with torch.no_grad():
            # get prediction each frame
            boxes, classes, labels = detect_utils.predict(frame, model, device, 0.8)
        
        # draw boxes
        image = detect_utils.draw_boxes(boxes, classes, labels, frame)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1

        out.write(image)
    else:
        break

cap.release()

# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")