import torchvision
import numpy as np
import torch
import argparse
import cv2
import detect_utils

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/vide')
parser.add_argument('-m', '--min-size', dest='min_size', default=800,
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                            min_size=args['min_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = Image.open(args['input'])
model.eval().to(device)

boxes, classes, labels = detect_utils.predict(image, model, device, 0.6)
image = detect_utils.draw_boxes(boxes, classes, labels, image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"output/{save_name}.jpg", image)
