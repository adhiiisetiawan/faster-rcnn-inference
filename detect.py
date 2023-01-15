import argparse
import cv2
import detect_utils
import torch

from PIL import Image
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

image = Image.open(args['input'])
model.eval().to(device)

boxes, classes, labels = detect_utils.predict(image, model, device, 0.6)
image = detect_utils.draw_boxes(boxes, classes, labels, image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"output/{save_name}.jpg", image)
