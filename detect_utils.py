from torchvision.transforms import transforms
import cv2
import numpy as np

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# create different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # check this
    image = transform(image).to(device) # transform the image to tensor
    image = image.unsqueeze(0) #add a batch dimension
    outputs = model(image) # get the prediction on the image

    # get all predicted classes
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]

    # get all predicted score 
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    # get all predicted bounding boxes 
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)

    return boxes, pred_classes, outputs[0]['labels']
