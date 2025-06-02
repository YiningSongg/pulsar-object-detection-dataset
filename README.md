# Pulsar Candidate Identification Dataset (Object Detection Version)

This repository provides an object-detection version of the **pulsar candidate dataset** based on [HTRU1](https://github.com/as595/HTRU1) and [FAST](https://github.com/dzuwhf/FAST_label_data). The dataset is designed for training and evaluating deep learning models—particularly YOLO-based models—for automatic pulsar identification.

---

## 📂 Dataset Description

The dataset consists of two key diagnostic subplots commonly used in pulsar candidate identification:

- **Subband plots** (phase vs frequency)
- **Subint plots** (phase vs time)

Each plot has been carefully cropped and verified, and is accompanied by a manually annotated label file (`.txt`) in YOLO-compatible format. These annotations indicate regions of interest containing pulsar signals.

---

## 📥 Data Sources and References

- **HTRU1 Dataset**:  
  GitHub: [https://github.com/as595/HTRU1](https://github.com/as595/HTRU1)  
  Citation: *Morello et al. (2016), MNRAS, 463, 3, 3410–3423*  
  DOI: [10.1093/mnras/stw656](https://doi.org/10.1093/mnras/stw656)


- **FAST Dataset**:  
  GitHub: [https://github.com/dzuwhf/FAST_label_data](https://github.com/dzuwhf/FAST_label_data)  
  Citation: *Zhang et al. (2020), MNRAS, 495, 1, 195–202*  
  DOI: [10.1093/mnras/staa916](https://doi.org/10.1093/mnras/staa916)

---

## 📦 Download Instructions

You can download the dataset from the **GitHub Releases** page:

👉 [Download from Releases](https://github.com/YiningSongg/pulsar-object-detection-dataset)

Or via command-line using a helper script:

`bash download_data.sh`  

```bash
HTRU1/  
├── HTRU1banddata/  
│   ├── images/  
│       ├── train/  
│           ├── cand_000001.png  
│           ├── ...  
│           ├── pulsar_0000.png  
│           ├── ...  
│   ├── images/  
│       ├── labels/  
│           ├── pulsar_0000.txt
│           ├── ...
├── HTRU1intdata/
│   ├── images/
│       ├── train/
│           ├── cand_000001.png
│           ├── ...
│           ├── pulsar_0000.png
│           ├── ...
│   ├── images/
│       ├── labels/
│           ├── pulsar_0000.txt
│           ├── ...
```

```bash
FAST/
├── lables/
│   ├── xxxxxx.txt
│   ├── xxxxxx.txt
├── test/
│   ├── pulsar
│       ├── xxxxxx.png
│       ├── xxxxxx.png
│   ├── rfi
│       ├── xxxxxx.png
│       ├── xxxxxx.png
├── train/
│   ├── pulsar
│       ├── xxxxxx.png
│       ├── xxxxxx.png
│   ├── rfi
│       ├── xxxxxx.png
│       ├── xxxxxx.png
```

## 🚀 How to Use the Dataset

To train a model using **YOLOv8** (recommended version: `yolov8n` for lightweight applications), follow these steps:

### 1. Prepare Your Dataset Directory

Split the dataset into training, validation, and test folders according to your task needs and YOLOv8's requirement. 

### 2. Create the `data.yaml` File

The `data.yaml` file tells YOLOv8 where to find the dataset and how to interpret class labels. Here's an example:

```
yaml
path: path/to/your_dataset  # dataset root directory
train: images/train         # training images path
val: images/val             # validation images path
test: images/test           # test images path

# Class definitions
names:
  0: RFI
  1: pulsar
```

### 3. Train the YOLOv8 Model
To start training with YOLOv8 (using ultralytics CLI), Here's an example:

```
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

### 4. Validate and Test
After training, validate the model or run inference on test images, Here's an example:

```
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=images/test
```

## Suggestions for Further Optimization

Download and examine the original HTRU/FAST datasets to refine and verify the annotation quality.
Tune YOLOv8 hyperparameters (e.g., learning rate, image size, anchor size).
Explore using additional diagnostic plots or metadata (e.g., DM, S/N) as auxiliary inputs.
Consider using ensemble methods or transfer learning for better generalization.
