# SnapDetect: Traditional ML-Based X-ray Fracture Detection

**SnapDetect** is a lightweight and efficient fracture detection system that uses traditional machine learning techniques to analyze X-ray images and predict bone fractures. Instead of deep learning, this project leverages the power of **HOG (Histogram of Oriented Gradients)** features combined with regression models like **SVR** and **Random Forest** to deliver fast and reliable predictions — **no GPU required**!

---

## 🔍 Features
- **Detects bone fractures**: Applies machine learning to identify fractures in X-ray images.
- **Extracts HOG features**: Implements Histogram of Oriented Gradients to capture critical visual patterns.
- **Predicts fractures**: Uses **SVR** and **Random Forest** models for accurate fracture predictions.
- **Optimized for speed**: **Processes X-ray images** quickly on standard hardware (CPU only, no GPU needed).

---

## 🚀 Getting Started

### Prerequisites
Ensure the following libraries are installed:
```bash
pip install opencv-python numpy scikit-learn matplotlib scikit-image
