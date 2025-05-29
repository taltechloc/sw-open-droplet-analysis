# Portable Microfluidic Droplet Classifier

A lightweight, CNN-based droplet classification system optimized for resource-limited, field-deployable imaging flow cytometry. This repository contains code and model weights for running a customized YoloV4-tiny on a Raspberry Pi 5 (RPi5), achieving real-time classification of microfluidic droplets.

## Features

- **Fast inference**: Classifies “no cell”, “one cell”, or “multiple cells” in ~13 ms per frame on RPi5.
- **High accuracy**: 99.95 % mAP@0.5 on custom microfluidic dataset.
- **Portable**: All dependencies optimized for Raspberry Pi OS on RPi5.
- **Comparison suite**: Scripts to benchmark against alternative platforms (e.g., MaixCam + YoloV5s).

## Citation

If you use this code, please cite:
Afrin, F., Le Moullec, Y., Pardy, T., & Rang, T. (2025). Lightweight CNN-based microfluidic droplet classification for portable imaging flow cytometry. [Proceedings of the Estonian Academy of Sciences].

## Models, setup & use

### YoloV4-Tiny Custom Model

This project uses OpenCV's DNN module with a custom YoloV4-tiny model for droplet classification

#### Structure

```
Custom YoloV4-tiny-inference/
├── infer.py
├── dnn_model/
│   ├── yolov4-tiny-custom_6000.weights
│   ├── yolov4-tiny-custom.cfg
    ├── classes.txt

├── Testing/
│   ├── test1(1).jpg
│   └── ...
├── Processed_Images/
├── training_data/
│   ├── images/
│   │   ├── 1_frame_0.jpg
│   │   ├── 1_frame_1.jpg
│   │   └── ...
│   └── labels/
│       ├── 1_frame_0.txt
│       ├── 1_frame_1.txt
│       └── ...
├── requirements.txt
├── README.md
```

#### Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

#### Usage

Run the inference:

```bash
python infer.py
```

### NIMA Image Aesthetic Quality Evaluation

This project uses a MobileNet-based model to evaluate the aesthetic quality of images using the NIMA (Neural Image Assessment) approach.

#### Structure

```
image-aesthetic-evaluation/
├── evaluate.py
├── weights/
│   └── weights.h5
├── utils/
│   └── score_utils.py
├── requirements.txt
└── README.md
```

#### Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

#### Usage

Evaluate images or directories:

```bash
# Evaluate a single image
python evaluate.py -img path/to/image.jpg

# Evaluate a directory of images
python evaluate.py -dir path/to/images/

# Optional arguments
-resize true    # Resize images to 224x224
-rank false     # Do not rank images by score
```

#### Output

- Prints NIMA score (mean and std) for each image.
- Optionally ranks all images from most to least aesthetically pleasing.

