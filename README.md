<img width="1276" height="717" alt="image" src="https://github.com/user-attachments/assets/4e733fa6-0e44-4ca2-9eb9-fc36086b37b4" />


# 🚗 Lane Detection and Vehicle Perception System

A computer vision and deep learning project that combines classical lane detection techniques with YOLOv8 object detection to create a perception pipeline for road-scene analysis.

The system processes dashcam footage, detects lane boundaries, projects a perspective-based distance grid, identifies road users, and estimates their approximate distance from the camera using monocular depth estimation principles.

---

# 📌 Project Overview

This project was built to explore the core perception concepts used in modern Advanced Driver Assistance Systems (ADAS).

The pipeline combines:

* Classical Computer Vision (OpenCV)
* Deep Learning Object Detection (YOLOv8)
* Perspective Geometry
* Monocular Distance Estimation
* Video Processing and Annotation

The final output is an annotated video containing:

* Lane boundary visualization
* Dynamic road distance grid
* Vehicle and pedestrian detection
* Estimated object distance in meters
* Safety-aware object highlighting

---

# 🎯 Key Features

## 1. Lane Detection

The system detects lane boundaries using traditional computer vision techniques.

Pipeline:

1. Convert frame to grayscale
2. Apply Gaussian blur
3. Perform Canny edge detection
4. Apply Region of Interest masking
5. Detect lines using Probabilistic Hough Transform
6. Average detected line segments into continuous lane boundaries

Detected lane lines are stabilized by averaging slopes and intercepts.

---

## 2. Dynamic Perspective Distance Grid

A perspective-based grid is projected onto the road surface.

Distance zones:

| Zone  | Color  |
| ----- | ------ |
| 1.0 m | Red    |
| 3.0 m | Yellow |
| 5.0 m | Green  |

The grid dynamically aligns with detected lane boundaries.

This provides a visual approximation of vehicle position relative to the road ahead.

---

## 3. Fallback Safety Grid

When lane detection temporarily fails due to:

* Poor lighting
* Road glare
* Occlusions
* Sharp turns

the system automatically switches to a predefined perspective template.

This ensures that distance guidance remains visible even when lane information becomes unreliable.

---

## 4. YOLOv8 Object Detection

The perception pipeline uses YOLOv8 Medium (YOLOv8m) for object detection.

Detected classes include:

* Cars
* Trucks
* Buses
* Motorcycles
* Bicycles
* Pedestrians

The detector is accelerated using GPU computation whenever CUDA is available.

---

## 5. Monocular Distance Estimation

The system estimates approximate object distance using the Pinhole Camera Model.

Formula:

Distance = (Real Object Height × Focal Length) / Bounding Box Height

Known average object heights are assigned to each class:

| Class      | Approximate Height |
| ---------- | ------------------ |
| Person     | 1.7 m              |
| Car        | 1.5 m              |
| Motorcycle | 1.0 m              |
| Bus        | 3.0 m              |
| Truck      | 2.8 m              |

The calculated distance is displayed alongside each detected object.

Example:

```text
car [4.2m]
person [7.8m]
```

---

# 🏗 System Architecture

```text
Input Dashcam Video
        │
        ▼
Frame Extraction
        │
        ▼
Grayscale Conversion
        │
        ▼
Gaussian Blur
        │
        ▼
Canny Edge Detection
        │
        ▼
ROI Masking
        │
        ▼
Hough Line Transform
        │
        ▼
Lane Averaging
        │
        ▼
Dynamic Grid Projection
        │
        ▼
YOLOv8 Object Detection
        │
        ▼
Distance Estimation
        │
        ▼
Frame Annotation
        │
        ▼
Output Video Generation
```

---

# 🛠 Technologies Used

## Programming Language

* Python 3.8+

## Deep Learning

* Ultralytics YOLOv8

## Computer Vision

* OpenCV

## Numerical Processing

* NumPy

## GPU Acceleration

* PyTorch
* CUDA

---

# 💻 Hardware Used

Development Machine:

* HP ZBook Studio x360 G5
* 32 GB RAM
* NVIDIA Quadro P1000 GPU

The application automatically switches to GPU inference when CUDA is available.

---

# 📂 Project Structure

```text
project/
│
├── local_perception.py
├── Car.ipynb
├── README.md
├── .gitignore
│
├── models/
│   └── yolov8m.pt
│
├── input/
│   └── your_test_driving_video.mp4
│
└── output/
    └── predicted_output.mp4
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

---

## Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
venv\Scripts\activate.bat
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install PyTorch

CUDA Version:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU Version:

```bash
pip install torch torchvision
```

---

## Install Remaining Dependencies

```bash
pip install ultralytics opencv-python numpy
```

---

# ▶ Running the Project

Place:

```text
your_test_driving_video.mp4
```

inside the project directory.

Run:

```bash
python local_perception.py
```

The pipeline will process the video frame-by-frame and generate:

```text
predicted_output.mp4
```

---

# 📊 Output

The generated video contains:

* Lane boundaries
* Distance grid
* Object bounding boxes
* Object class labels
* Estimated object distance
* Color-coded danger zones

Example:

```text
Car [2.1m]
Truck [4.8m]
Person [7.2m]
```

---

# 🔍 Current Limitations

* Distance estimation is approximate and based on assumed object heights.
* Monocular estimation is less accurate than stereo-camera systems.
* Lane detection performance may degrade in poor weather conditions.
* Curved roads can reduce lane fitting accuracy.
* No object tracking across frames.
* No sensor fusion (LiDAR, Radar, GPS, IMU).

---

# 🚀 Future Improvements

* Multi-object tracking (DeepSORT / ByteTrack)
* Lane curvature prediction
* Bird's Eye View transformation
* Traffic sign recognition
* Traffic light detection
* Collision warning system
* Driver alert generation
* Semantic road segmentation
* Depth estimation using neural networks
* Real-time webcam integration

---

# 📚 Learning Outcomes

Through this project, the following concepts were explored:

* Computer Vision Fundamentals
* Edge Detection
* Hough Transform
* Region of Interest Masking
* Perspective Geometry
* Object Detection
* GPU Acceleration
* Video Processing
* Distance Estimation
* ADAS Perception Pipelines

---

# 📄 License

This project is released under the MIT License.

Feel free to use, modify, and distribute the code for educational and research purposes.
