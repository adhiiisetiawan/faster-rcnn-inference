# Faster R-CNN Inference

## Quick Start Examples

### Install
Clone repo and install [requirements.txt](https://github.com/adhiiisetiawan/faster-rcnn-inference/blob/master/requirements.txt) in your environment.
```bash
git clone https://github.com/adhiiisetiawan/faster-rcnn-inference.git  # clone
cd faster-rcnn-inference
pip install -r requirements.txt  # install
```

### Inference Using Image
You can inference this repository using your own images, just put your own images in input folder and run this script
```bash
python3 detect.py -i input/[image-name] -s [min-image-size] -m [backbone you want to use (mobilenetv3/resnet50)]
```
Example usage
```bash
python3 detect.py -i input/image1.jpg -s 1024 -m resnet50
```

### Inference Using Video
You can inference this repository using your own videos, just put your own videos in input folder and run this script
```bash
python3 detect_vid.py -i input/[video-name] -s [min-input-size] -m [backbone you want to use (mobilenetv3/resnet50)]
```
Example usage
```bash
python3 detect_vid.py -i input/video1.mp4 -s 1024 -m mobilenetv3
```
