```markdown
# ADAS Perception Pipeline

An advanced, headless real-time Advanced Driver Assistance System (ADAS) perception simulator. This pipeline integrates computer vision algorithms with deep learning models to perform synchronized lane boundary tracking, dynamic perspective distance grid projection, and precise vehicle detection with absolute depth mapping.

---

## 🚀 Features

- **Object Detection & Classification:** Powered by **YOLOv8m (Medium)** to track vehicles, pedestrians, and road hazards with enhanced stability and minimized class-flickering.
- **Dynamic Perspective Distance Grid:** Projects a real-time range grid (1.0m, 3.0m, and 5.0m intervals) mapped directly to the road plane utilizing automated line-slope analysis.
- **Fail-Safe Fallback Matrix:** Incorporates an automated fallback layout mimicking a vehicle's reversing camera system to maintain baseline safety indicators when dynamic lane tracking is temporarily obscured by severe glare or sharp curves.
- **Headless Optimization:** Engineered to process frame streams directly in the background, maximizing execution throughput by discarding synchronous window-rendering bottlenecks.
- **Hardware Accelerated Architecture:** Programmed to auto-route deep learning computation to available **NVIDIA CUDA Cores** for high-performance inference.

---

## 🛠️ Tech Stack & Dependencies

- **Core Language:** Python 3.8+
- **Deep Learning Framework:** PyTorch (CUDA-optimized)
- **Object Detection Engine:** Ultralytics YOLOv8
- **Computer Vision Operators:** OpenCV (Open Source Computer Vision Library)
- **Numerical Processing Engine:** NumPy

---

## 💻 Hardware Benchmarks & Requirements

While this codebase operates across standard CPU architectures, it is explicitly optimized to utilize discrete workstation GPUs for real-time parallel video matrix processing.

| Component | Minimum Specification | Recommended Specification (Workstation Optimized) |
| :--- | :--- | :--- |
| **Processor** | Intel Core i5 / AMD Ryzen 5 | Intel Core i7 / Xeon or equivalent |
| **Memory** | 8 GB RAM | 32 GB RAM (Multi-stream buffer capacity) |
| **Graphics** | Shared System Architecture | Dedicated NVIDIA GPU (e.g., Quadro P1000 / RTX Series) |
| **Compute API** | CPU Only execution | NVIDIA CUDA Toolkit 11.8 / 12.1+ |

---

## 📥 Installation & Setup

Follow these precise steps to provision a localized development sandbox and initialize the execution pipeline:

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/adas-perception-pipeline.git](https://github.com/YOUR_GITHUB_USERNAME/adas-perception-pipeline.git)
cd adas-perception-pipeline

```

### 2. Isolate with a Virtual Environment

Create an isolated Python execution environment to keep dependencies clean:

```bash
# Initialize environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Windows (Classic CMD)
venv\Scripts\activate.bat

# Activate on Linux / macOS / Git Bash
source venv/bin/activate

```

*Note for Windows users:* If script execution is blocked on PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` before running the activation path.

### 3. Install CUDA-Optimized Binaries

To prevent PyTorch from defaulting to a slower CPU execution layer, explicitly install the CUDA-mapped binaries:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

### 4. Install Component Packages

Deploy the remaining structural vision and deep learning packages:

```bash
pip install ultralytics opencv-python numpy

```

---

## 🏃 Execution Guide

1. Place your input driving video file within the project folder root and name it `your_test_driving_video.mp4` (or update the file path variables in section 2 of the script).
2. Execute the perception script pipeline from your terminal:
```bash
python local_adas_perception.py

```


3. Monitor progress benchmarks inside the command prompt window. Upon completion, the annotated stream will be exported as a permanent video file: **`predicted_output.mp4`**.

---

## 📊 Pipeline Visual Architecture

The matrix transforms video frames through a linear data-driven pipeline:

```text
 [Raw Frame Input] ──> [Grayscale & Gaussian Blur] ──> [Canny Edge Analytics]
                                                               │
 [YOLOv8m Target Inference] <── [Perspective Grid Layer] <── [ROI Masking Filter]
            │
            ▼
 [Matrix Frame Compilation] ──> [Headless MP4 Video Writer Export]

```

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```
