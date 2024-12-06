# Yoga_Pose_Detection_And_Prediction


## Overview

The Yoga Pose Detector is a deep learning-based project designed to analyze and classify yoga poses from images or video frames. By combining state-of-the-art object detection and image classification techniques, the project aims to assist users in identifying yoga poses accurately, which can be beneficial for fitness tracking, instructional purposes, or automated yoga tutorials.

## Data
The dataset used in this project is the roboflow's Yoga Pose Computer Vision Project, which includes numerous images for each of the 107 yoga poses.

Dataset Source Link: [data url](https://universe.roboflow.com/new-workspace-mujgg/yoga-pose/dataset/1)


## Features

**1. Pose Detection:** Real-time detection of yoga poses using advanced algorithms.<br>
**2. Object Detection with YOLO:** Identifies key body regions to focus on relevant parts of the pose.<br>
**3. Pose Classification with VGG16:** Utilizes transfer learning for accurate pose recognition.<br>
**4. Image Processing:** Preprocesses input images for optimal model performance.<br>
**5. Visualization:** Highlights detected poses and displays results for easy interpretation.


## Tech Stack

**1. Programming Language:** `Python`<br>
**2. Deep Learning Frameworks:** `TensorFlow`, `Keras` <br>
**3. Computer Vision Tools:** `OpenCV`, `YOLO`<br>
**4. Data Visualization:** `Matplotlib`, `Seaborn`<br>


## Workflow

### 1. Input Handling:
Accepts images or video frames for processing.

### 2. Pose Detection (YOLO):
Detects body parts or the entire pose region.<br>
Outputs bounding boxes or key regions of interest.

### 3. Feature Extraction (VGG16):
Processes detected regions for feature extraction.<br>
Classifies poses using a fine-tuned model.

### 4. Visualization:
Displays results with bounding boxes and pose labels.


## Prerequisites

**1. Python Version:** Python 3.7+<br>
**2. Libraries:** `tensorflow`, `keras`, `opencv-python`, `numpy`, `matplotlib`, `seaborn`, `cvzone`, `ultralytics`<br>
**3.** GPU for faster model inference (optional but recommended)<br>


## Usage

1. After downloading the dataset, unzip it and place in the root directory.
2. Clone the repository
```
git clone https://github.com/SnehaGhoshNew2003/Yoga_Pose_Detection_And_Prediction.git
cd yoga-pose-detector
```
3. Install the required libraries:
```
pip install -r requirements.txt
```
3. Run the notebook or script:
```
jupyter notebook Yoga_Pose_Detector.ipynb
```


## Acknowledgements

1. `YOLO` for real-time object detection.
2. `VGG16` for robust pose classification.
3. Open-source libraries and datasets that made this project possible.
