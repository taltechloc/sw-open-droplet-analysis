# Portable Microfluidic Droplet Classifier

A lightweight, CNN-based droplet classification system optimized for resource-limited, field-deployable imaging flow cytometry. This repository contains code and model weights for running a customized YOLOv4-tiny on a Raspberry Pi 5 (RPi5), achieving real-time classification of microfluidic droplets.

## Features

- **Fast inference**: Classifies “no cell”, “one cell”, or “multiple cells” in ~13 ms per frame on RPi5.
- **High accuracy**: 99.95 % mAP@0.5 on custom microfluidic dataset.
- **Portable**: All dependencies optimized for Raspberry Pi OS on RPi5.
- **Comparison suite**: Scripts to benchmark against alternative platforms (e.g., MaixCam + YoloV5s).

## Repository Structure

- `data/` – Sample images & annotation format.
- `models/` – Pre-trained YOLOv4-tiny weights and config.
- `src/`
  - `train.py` – Training script (requires GPU).
  - `export.py` – Convert model for TensorRT or TFLite.
  - `infer.py` – Real-time inference on RPi5 camera/video.
  - `benchmark.py` – Performance comparison tools.
- `requirements.txt` – Python dependencies.

## Installation

```bash
git clone https://github.com/your-org/microfluidic-classifier.git
cd microfluidic-classifier
pip install -r requirements.txt
```

## Citation

If you use this code, please cite:
Afrin, F., Le Moullec, Y., Pardy, T., & Rang, T. (2025). Lightweight CNN-based microfluidic droplet classification for portable imaging flow cytometry. [Proceedings of the Estonian Academy of Sciences].