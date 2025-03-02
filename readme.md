# Hybrid Uncertainty-Aware Deep Learning for Interpretable Breast and Gynecological Cancer Screening

Welcome to our research repository for **Hybrid Uncertainty-Aware Deep Learning** in **breast and gynecological cancer screening**. This project aims to combine **state-of-the-art conformal prediction (via TorchCP)** with **deep learning** and **interpretability methods** (e.g., Grad-CAM, attention mechanisms) to produce reliable, explainable models that can assist clinicians in early cancer detection.

---

## Table of Contents

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Key Features](#key-features)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Dataset Preparation](#dataset-preparation)  
5. [Usage](#usage)  
   - [Training](#training)  
   - [Evaluation](#evaluation)  
   - [Interpretability and Visualization](#interpretability-and-visualization)  
6. [Results](#results)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [Acknowledgments](#acknowledgments)  
10. [References](#references)  

---

## 1. Overview

**Goal:**  
To develop a **hybrid deep learning** pipeline that provides **uncertainty estimates** alongside **interpretability** insights for breast and gynecological cancer screening. By leveraging **TorchCP** for conformal prediction, the system can deliver **prediction sets** or confidence intervals, highlighting which cases may require additional review. Simultaneously, interpretability techniques help visualize **which regions of the image** most influenced the model’s decision.

**Why This Matters:**  
- In **clinical diagnostics**, minimizing false negatives is critical. Conformal prediction ensures a predefined coverage or error rate, making predictions **safer** for real-world applications.  
- Providing **visual explanations** for each prediction fosters trust and helps **radiologists** or **pathologists** identify subtle abnormalities.

---

## 2. Project Structure

A typical directory layout might look like this:

```
├── README.md
├── environment.yml or requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── EDA.ipynb
│   └── model_experiments.ipynb
├── src/
│   ├── models/
│   │   ├── base_model.py
│   │   └── ...
│   ├── torchcp_integration/
│   │   ├── conformal_utils.py
│   │   └── ...
│   ├── training/
│   │   ├── train.py
│   │   └── ...
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── ...
│   └── interpretability/
│       ├── gradcam.py
│       └── ...
└── scripts/
    ├── run_training.sh
    └── run_evaluation.sh
```

- **notebooks/**: Jupyter notebooks for exploratory data analysis (EDA), data preprocessing, and quick experiments.  
- **src/**: Contains modular code for models, training, evaluation, and interpretability.  
- **scripts/**: Shell scripts to automate training or evaluation on local or cluster environments.

---

## 3. Key Features

1. **Uncertainty Quantification with TorchCP**  
   - Leverages **Conformal Prediction** for classification tasks.  
   - Produces **prediction sets** at a chosen confidence level.

2. **Interpretability**  
   - Integrates saliency methods like **Grad-CAM** or **attention-based** heatmaps.  
   - Overlays highlight regions of interest on mammograms or histopathology slides.

3. **Modular Architecture**  
   - Easy to plug in different backbone models (e.g., ResNet, EfficientNet, ViT).  
   - Conformal modules are independent of the base architecture, allowing flexible experimentation.

4. **Medical Imaging-Focused**  
   - Code optimized for large 2D (mammography) or 3D (MRI) images.  
   - Supports relevant data augmentations and normalization strategies used in clinical imaging.

---

## 4. Getting Started

### 4.1. Prerequisites

- **Operating System:** Linux (Ubuntu 20.04+ recommended) or macOS.  
- **Python 3.8+**  
- **PyTorch 2.0+** with GPU support (CUDA 11.x or above recommended).  
- **TorchCP** (1.0.1 or later).  
- Optional but recommended:
  - **Conda** for environment management.  
  - **Jupyter** or **JupyterLab** for interactive notebooks.

### 4.2. Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/hybrid-uncertainty-cancer-screening.git
   cd hybrid-uncertainty-cancer-screening
   ```

2. **Create a Conda Environment (Optional)**  
   ```bash
   conda create -n hybrid-dl python=3.9 -y
   conda activate hybrid-dl
   ```

3. **Install Dependencies**  
   - Using `pip`:
     ```bash
     pip install -r requirements.txt
     ```
   - Or using `conda`:
     ```bash
     conda install --file requirements.txt
     ```

4. **Install TorchCP**  
   If not already in your `requirements.txt`, install directly:
   ```bash
   pip install torchcp
   ```

<!-- ### 4.3. Dataset Preparation

- **Breast Cancer**: [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) or [INbreast](https://medicalresearch.inesctec.pt/INbreast/)  
- **Cervical (Gynecological) Cancer**: [Herlev Dataset](https://www.kaggle.com/datasets/andrewmvd/cervical-cancer-cell-segmentation) or ISBI Challenges

1. **Download and Organize**  
   - Place raw data in `data/raw/`.  
   - Convert or preprocess images (e.g., normalization, resizing) and store results in `data/processed/`.

2. **Update Config Files**  
   - In `src/configs/data_config.yaml` (if using a YAML-based approach), specify dataset paths, normalization parameters, etc. -->

---

## 5. Usage

### 5.1. Training

1. **Base Training (No Conformal Methods Yet)**  
   ```bash
   python src/training/train.py --config configs/base_config.yaml
   ```
   - This trains a baseline CNN or transformer.  
   - Check logs for accuracy, loss, etc.

2. **Training with Conformal Loss**  
   ```bash
   python src/training/train.py --config configs/conformal_config.yaml
   ```
   - Uses TorchCP-based loss functions (e.g., ConTr, C-Adapter).  
   - Automatically splits or sets aside calibration data if configured.

### 5.2. Evaluation

1. **Classification Metrics**  
   ```bash
   python src/evaluation/eval.py --config configs/conformal_config.yaml
   ```
   - Reports accuracy, sensitivity, specificity, AUC.  
   - Optionally logs conformal metrics (coverage, size) if using TorchCP.

2. **Post-hoc Calibration**  
   ```bash
   python src/torchcp_integration/conformal_utils.py --method APS --alpha 0.05
   ```
   - Generates prediction sets with a target coverage (1 - alpha).  
   - Outputs coverage metrics and average set size.

### 5.3. Interpretability and Visualization

- **Grad-CAM** or **Attention Heatmaps**  
  ```bash
  python src/interpretability/gradcam.py --model-checkpoint best_model.pth \
                                         --image-path data/processed/sample_image.png
  ```
  - Produces heatmaps showing important regions for the model’s decision.  
  - Compare these heatmaps to the model’s uncertainty (e.g., wide prediction sets).

- **Overlaying Conformal Sets**  
  - If multiple classes remain after conformal filtering, you can highlight uncertain regions or classes in the visualization.

---

## 6. Results

- **Performance Summary**:  
  - Baseline model achieves ~X% accuracy, Y% sensitivity, Z% specificity.  
  - Conformal approach maintains ~95% coverage while reducing false negatives.  
- **Interpretability Findings**:  
  - Grad-CAM overlays show the model focuses on suspicious calcifications or mass edges.  
  - High-uncertainty cases often involve dense breast tissue or poor image quality.

*(Feel free to include plots, confusion matrices, or heatmap images here.)*

---

## Acknowledgments

- **TorchCP** developers for providing the conformal prediction framework.  
- **Clinical collaborators** who provided guidance on annotation and dataset selection.  
- **Open-source contributors** and the PyTorch community for libraries and utilities used here.

---

## References

1. **TorchCP Documentation**: [https://torchcp.readthedocs.io/](https://torchcp.readthedocs.io/)  
2. **CBIS-DDSM**: [https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)  
3. **INbreast**: [https://medicalresearch.inesctec.pt/INbreast/](https://medicalresearch.inesctec.pt/INbreast/)  
4. **Herlev Dataset**: [https://www.kaggle.com/datasets/andrewmvd/cervical-cancer-cell-segmentation](https://www.kaggle.com/datasets/andrewmvd/cervical-cancer-cell-segmentation)  

5. ( Relevant papers or resources that guided our methodology.)*

---

**Thank you for your interest in this project!** For further questions or collaboration inquiries, feel free to open an issue or reach out to us at [dsconite@gmail.com](mailto:dsconite@gmail.com).