# YOLOv12 Household Object Detection

> Real-time household object detection using **YOLOv12s** trained on a custom dataset of everyday items via webcam.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![YOLOv12](https://img.shields.io/badge/YOLOv12-Ultralytics-red)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Colab](https://img.shields.io/badge/Train-Google%20Colab-orange)](notebooks/yolov12_object_detection_on_custom_dataset.ipynb)

---

---

## Overview

This project trains a **YOLOv12s** model to detect common household objects in real time using a webcam. The model was trained on a custom dataset of ~800 images across multiple household item categories including bottles, cups, mugs, caps, and mobile phones.

---

## Detected Objects

| Class | Description |
|-------|-------------|
| Bottle | Water bottles, plastic bottles |
| Cup | Tea cups, paper cups |
| Mug | Coffee mugs, ceramic mugs |
| Cap | Bottle caps, hats |
| Mobile | Smartphones |
| + more | Other common household items |

---

## Model Performance

Trained for **100 epochs** on Google Colab (Tesla T4 GPU):

| Metric | Value |
|--------|-------|
| **mAP50** | **0.995** |
| **mAP50-95** | **0.824** |
| Precision | 0.992 |
| Recall | 1.000 |
| Inference Speed | 10.4ms/image |
| Model Size | 18.9 MB |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Model | YOLOv12s |
| Dataset | Custom (~800 images) |
| Epochs | 100 |
| Image Size | 640×640 |
| GPU | Tesla T4 (15GB) |
| Framework | Ultralytics 8.3.78 |
| PyTorch | 2.5.1+cu124 |

---

## Project Structure

```
yolov12-object-detection/
│
├── src/
│   └── main.py                  ← Real-time webcam detection
│
├── notebooks/
│   └── yolov12_object_detection_on_custom_dataset.ipynb  ← Training notebook
│
├── assets/
│   └── demo.png                 ← Demo screenshot
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/yolov12-object-detection
cd yolov12-object-detection
pip install ultralytics opencv-python
```

---

## Download Model

The trained model weights (`best.pt`) are available in this repository.

Download directly:
```bash
# Clone the repo to get best.pt
git clone https://github.com/rakibulnishat/yolov12-object-detection
```

---

## Usage

### Real-time Webcam Detection

```bash
python src/main.py
```

Press **Q** to quit.

### Run on a Video File

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.predict("your_video.mp4", save=True, show=True)
```

### Run on an Image

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("your_image.jpg", save=True)
```

### Run on Webcam (CLI)

```bash
yolo task=detect mode=predict model=best.pt source=0 show=True
```

---

## Training

Open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/yolov12_object_detection_on_custom_dataset.ipynb)

Or follow these steps:

```bash
# Install dependencies
pip install ultralytics

# Train
yolo task=detect mode=train model=yolo12s.pt data=data.yaml epochs=100 imgsz=640 plots=True
```

---

## How It Works

```
Webcam Frame
     │
     ▼
┌─────────────────────┐
│   YOLOv12s Model    │
│   (159 layers)      │
│   9.2M parameters   │
│   21.2 GFLOPs       │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Bounding Boxes     │
│  Class Labels       │
│  Confidence Scores  │
└─────────────────────┘
     │
     ▼
  Display Output
```

---

## Requirements

```
ultralytics>=8.3.0
opencv-python>=4.8.0
torch>=2.0.0
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Rakibul Hassan Nishat**
- Kaggle: [rakibulhassannishat](https://www.kaggle.com/rakibulhassannishat)
- Hugging Face: [nishaatt](https://huggingface.co/nishaatt)
