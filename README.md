# Faster R-CNN Inference

## Quick Start Examples

### Install
Create your environment using venv
```bash
python3 -m venv [your-env-name]
```

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
