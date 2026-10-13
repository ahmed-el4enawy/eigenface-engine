# EigenFace-Engine ⚡

> **High-Performance C++17 Facial Detection, PCA Eigenspace Projection & Biometric Verification Engine**

[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pybind11](https://img.shields.io/badge/Binding-pybind11%20%2B%20ctypes-4c1?logo=c%2B%2B)](https://github.com/pybind/pybind11)
[![Django](https://img.shields.io/badge/Frontend-Django%205.x-092E20?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()

---

## 📌 Executive Summary

**EigenFace-Engine** is a complete, production-grade facial detection, dimensionality reduction, and recognition pipeline implemented from scratch in **C++17** and interfaced with Python via **`pybind11`** and memory-safe **`ctypes`**. 

Unlike standard implementations that rely on external linear algebra packages (e.g., Eigen, LAPACK, or SciPy), **EigenFace-Engine** implements its entire numerical foundation—including **cyclic Jacobi eigendecomposition**, **Gram matrix covariance reduction**, **bilinear image interpolation**, and **histogram equalization**—from first principles with zero external mathematical dependencies.

The engine powers an interactive full-stack **Django** dashboard enabling multi-scale face detection, real-time eigenspace projection, visual eigenface basis inspection, and automated scientific evaluation (ROC, AUC, EER).

---

## 🚀 Key Highlights & Architectural Strengths

* ⚡ **Zero External Math Dependencies:** Jacobi eigendecomposition solver, covariance dimensionality tricks, and image preprocessing are written entirely in native C++17.
* 📉 **>99.6% Dimensionality Compression:** Reduces input image vectors from $10,304$ pixels ($112 \times 92$) to $k \le 40$ orthogonal eigenfeatures while preserving facial discriminability.
* ⏱️ **Sub-Millisecond Inference:** Nearest-neighbor Euclidean distance query classification executes in `< 0.5 ms` per image.
* 🔌 **Dual Interoperability Layer:** Seamless integration with Python through both high-level `pybind11` bindings and zero-copy, raw memory buffer C ABI calls (`ctypes`).
* 📊 **Rigorous Diagnostic Suite:** Built-in continuous threshold sweeping for Receiver Operating Characteristic (ROC) curves, Area Under Curve (AUC), and Equal Error Rate (EER) computation.
* 🎨 **Interactive Control Dashboard:** Modern, responsive dark-mode web application featuring real-time image upload, parameter tuning, eigenface visualization, and analytics.

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Web Browser (Client UI)                         │
│  ┌───────────────────────┐ ┌────────────────────┐ ┌──────────────────┐ │
│  │ Multi-Scale Detector  │ │ Eigenspace Trainer │ │ Diagnostic Suite │ │
│  └───────────┬───────────┘ └─────────┬──────────┘ └────────▲─────────┘ │
└──────────────│───────────────────────│─────────────────────│───────────┘
               │ (Multipart Upload)    │ (JSON Params)       │ (JSON Data)
┌──────────────▼───────────────────────▼─────────────────────┴───────────┐
│                    Django Application Server (Python)                  │
│  • detector/views.py  : REST Endpoints & Parameter Validation          │
│  • face_utils.py      : Stratified Dataset Split, ROC/AUC/EER Engine   │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ pybind11 + ctypes C ABI
┌───────────────────────────────▼────────────────────────────────────────┐
│               EigenFace Native Core Engine (cv_core.pyd)               │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ cv_custom_algorithms.h                                           │  │
│  │  • Grayscale Conversion & Bilinear Rescaling                     │  │
│  │  • Histogram Equalization (Lighting Invariance)                  │  │
│  │  • Cyclic Jacobi Eigendecomposition Solver                       │  │
│  │  • Covariance Trick (AᵀA Dimension Reduction)                    │  │
│  │  • Eigenspace Projection & L2 Nearest-Neighbor Search            │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ cv_face_algorithms.h                                             │  │
│  │  • Multi-Scale Face Detection with Aspect Normalization          │  │
│  │  • Bresenham Line Drawing & Box Annotation Primitive             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  * OpenCV is used strictly for cv::Mat container storage and image I/O *
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. Vector Space Representation
Given a training dataset of $M$ facial images $\{\Gamma_1, \Gamma_2, \dots, \Gamma_M\}$, each image of dimensions $H \times W$ is flattened into an $N$-dimensional vector $x_i \in \mathbb{R}^N$, where $N = H \cdot W$ (for ORL faces: $N = 112 \times 92 = 10,304$).

The mean facial vector $\mu$ and zero-mean face vectors $\Phi_i$ are computed:
$$\mu = \frac{1}{M}\sum_{i=1}^M x_i, \quad \Phi_i = x_i - \mu$$

The centered data matrix $A \in \mathbb{R}^{N \times M}$ is formed as:
$$A = \begin{bmatrix} \Phi_1 & \Phi_2 & \dots & \Phi_M \end{bmatrix}$$

### 2. The Sirovich & Kirby / Turk & Pentland Covariance Trick
The full sample covariance matrix $C = \frac{1}{M} A A^T$ has dimensions $N \times N$ ($10,304 \times 10,304 \approx 1.06 \times 10^8$ elements), making direct eigendecomposition computationally prohibitive.

Instead, we construct the surrogate inner-product Gram matrix $L \in \mathbb{R}^{M \times M}$:
$$L = A^T A, \quad \text{where } L_{i,j} = \Phi_i^T \Phi_j$$

### 3. Cyclic Jacobi Eigendecomposition
We solve the symmetric eigenvalue problem $L v_i = \lambda_i v_i$ using our from-scratch cyclic Jacobi rotation algorithm. At each iteration, an orthogonal Givens rotation matrix $R(p, q, \theta)$ eliminates the maximal off-diagonal element $L_{pq}$:
$$\tan(2\theta) = \frac{2 L_{pq}}{L_{pp} - L_{qq}}$$
Iterating until $\max_{p \ne q} |L_{pq}| < 10^{-10}$ yields the eigenvalues $\lambda_i$ and eigenvectors $v_i \in \mathbb{R}^M$.

### 4. Eigenface Basis Reconstruction
The eigenvectors $u_i \in \mathbb{R}^N$ of the original covariance matrix $C$ are reconstructed via linear combination:
$$u_i = \frac{A v_i}{\|A v_i\|_2}, \quad i = 1, \dots, k$$
The top $k$ orthonormal eigenfaces form the projection matrix $U = \begin{bmatrix} u_1 & u_2 & \dots & u_k \end{bmatrix} \in \mathbb{R}^{N \times k}$.

### 5. Projection & Classification
Any test face $\Gamma$ is normalized and projected onto the eigenspace:
$$\Omega = U^T (\Gamma - \mu) \in \mathbb{R}^k$$

Classification is performed via minimal Euclidean distance against training projections $\Omega_j$:
$$\epsilon_j = \|\Omega - \Omega_j\|_2, \quad j^* = \arg\min_j \epsilon_j$$

Given rejection threshold $\theta$:
$$\text{Decision} = \begin{cases} \text{Class}(j^*), & \text{if } \epsilon_{j^*} < \theta \\ \text{Unknown / Rejected}, & \text{otherwise} \end{cases}$$

---

## ⚡ Performance & Benchmarks

Benchmarked on **Intel Core i7 / AMD Ryzen** (Release build with MSVC C++17 optimizations) on the **AT&T / ORL Face Database** (40 subjects, 400 images total, $112 \times 92$ resolution):

| Metric | Measured Value | Remarks |
| :--- | :--- | :--- |
| **Input Dimensionality ($N$)** | `10,304` pixels | Raw flattened image vector |
| **Eigenspace Dimensionality ($k$)** | `40` components | **99.61% compression ratio** |
| **Jacobi Decomposition Latency** | `~14.8 ms` | Full $M \times M$ matrix solution |
| **Single-Face Recognition Latency** | `< 0.38 ms` | Projection + $L_2$ search |
| **Rank-1 Recognition Accuracy** | **`96.67%`** | Stratified 70/30 split |
| **Area Under ROC Curve (AUC)** | **`0.984`** | Continuous sweep across thresholds |
| **Equal Error Rate (EER)** | **`0.038`** | Point where FAR = FRR |

---

## 📂 Repository Structure

```text
eigenface-engine/
├── CMakeLists.txt              # Modern CMake build configuration (C++17)
├── build_cv_core.py            # setuptools build script for compiling cv_core extension
├── cv_core.cpp                 # pybind11 & ctypes C ABI entry points
├── cv_custom_algorithms.h      # Core numerical library (Jacobi solver, PCA, projection)
├── cv_face_algorithms.h        # Face detector & annotation routines
├── face_utils.py               # Dataset splitting, ROC/AUC calculation, evaluation engine
├── haarcascade_frontalface_... # Pre-trained cascade model for facial localization
├── manage.py                   # Django management utility
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # Technical documentation
│
├── cv_project/                 # Django project configuration
│   ├── settings.py             # Server settings & media routing
│   ├── urls.py                 # Route definitions
│   └── wsgi.py
│
├── detector/                   # Web application module
│   ├── views.py                # REST API controllers & model caching
│   └── templates/
│       └── detector/home.html  # Interactive single-page dashboard
│
├── static/                     # Frontend assets
│   ├── css/style.css           # Premium dark-mode design system
│   └── js/app.js               # Reactive client-side application logic
│
└── pretrained/                 # AT&T / ORL dataset (40 classes, s1-s40)
```

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.12+**
* **C++17 compliant compiler:** MSVC (Visual Studio 2022+), GCC 9+, or Clang 11+
* **OpenCV 4.x:** Installed with development headers (set `OpenCV_DIR` environment variable)

### 1. Clone & Setup Environment
```bash
git clone https://github.com/ahmed-el4enawy/eigenface-engine.git
cd eigenface-engine

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate
# Activate environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Compile Native C++ Core
You can compile using `setuptools`:
```bash
python build_cv_core.py build_ext --inplace
```

Or via **CMake**:
```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### 3. Launch the Web Interface
```bash
python manage.py runserver 8000
```
Open **`http://localhost:8000`** in your browser.

---

## 💻 API Reference & Code Examples

### Python (via `pybind11`)
```python
import cv_core

# 1. Train Eigenface Model
train_images = ["path/to/img1.pgm", "path/to/img2.pgm"]
labels = [1, 2]
model_info = cv_core.train_eigenfaces(
    image_paths=train_images,
    labels=labels,
    model_output_prefix="media/results/model",
    num_components=40,   # 0 for auto 95% variance
    face_h=112,
    face_w=92
)
print(f"Trained with {model_info['num_components']} components in {model_info['time_ms']:.2f}ms")

# 2. Recognize a Query Face
result = cv_core.recognize_face("path/to/query.pgm", threshold=5000.0)
print(f"Predicted Subject: {result['predicted_label']} (Distance: {result['distance']:.2f})")
```

### Low-Level C ABI (via `ctypes`)
```python
import ctypes
import numpy as np

# Load compiled library
lib = ctypes.CDLL("./cv_core.pyd")

# Memory-safe face recognition from raw numpy buffer
img = np.ascontiguousarray(face_array, dtype=np.uint8)
rows, cols = img.shape[:2]
out_dist = ctypes.c_double()

lib.ctypes_recognize_face.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]
lib.ctypes_recognize_face.restype = ctypes.c_int

subject_id = lib.ctypes_recognize_face(
    img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    rows, cols, ctypes.byref(out_dist)
)
print(f"Recognized Subject #{subject_id} with distance {out_dist.value:.2f}")
```

---

## 👥 Authors & Credits

Developed by:
* **Ahmed Salah Geoshy Elshenawy**
* **Abdullah Mohammad Khalifa**
* **Mohamed Elsayed Attallah**

*Cairo University · Faculty of Engineering · Systems & Biomedical Engineering Department*

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
