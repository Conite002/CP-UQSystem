
## 1. Core Task and Dataset

1. **Classification**.
2. **Data Preparation**  
   - Split data into **training**, **validation**, and **test** sets.
3. **Visualization of the data**, the model, and the distributional uncertainty for classification and regres
sion models
---

## 2. Select and Configure a Base Model

1. **Model Architecture**  
   - Suitable CNN or Vision Transformer (ViT) architecture for medical image classification.  

2. **Initial Training (No Conformal Methods Yet)**  
   - Train your baseline model using standard classification losses (e.g., cross-entropy).  
   - Evaluate accuracy, precision, recall, and F1-score to confirm that the model is learning effectively before adding conformal methods.

---

## 3. Incorporate TorchCP’s Loss & Trainer Components

1. **Choice of  Conformal Loss/Trainer**  
   - From the **Loss & Trainer** section of TorchCP (middle-left of the image), experiment with one or more:
     - **ConTr (Conformal Training)**
     - **C-Adapter**
     - **ConfMargin**
2. **Implementation Details**  
   - Replace or augment the standard loss function with the chosen TorchCP loss.  
   - Use TorchCP’s trainer classes to manage training loops, ensuring that conformal-specific parameters (e.g., calibration splits) are properly configured.

3. **Iterative Tuning**  
   - Adjust hyperparameters (learning rate, batch size, etc.) while monitoring both classification metrics and initial coverage metrics from conformal outputs (if provided).

---

## 4. Apply Post-hoc Calibration for Uncertainty Estimation

1. **Calibration Method Selection**  
   - After or alongside training with a conformal-friendly loss, you can apply **Post-hoc Calibration** (middle-right of the image).  
   - Common methods to try:
     - **APS (Adaptive Prediction Sets)**
     - **THR** or **RAPS (Regularized Adaptive Prediction Sets)**
     - **Weighted CP**, **Clustered CP**, or **Class-wise CP** (if you suspect class imbalance or domain shifts)

2. **Implementation**  
   - Use a held-out calibration set (often part of your validation set) to fit these post-hoc methods.  
   - Generate conformal prediction sets (e.g., sets of possible classes with a specified confidence level).

3. **Practical Considerations**  
   - In medical imaging, a small coverage error rate might be critical (e.g., 1% or 5%). Adjust your desired coverage accordingly to ensure minimal false negatives in cancer detection.

---

## 5. Evaluate with TorchCP Metrics (and Additional Interpretability)

1. **Conformal Metrics**  
   - TorchCP provides metrics like **Coverage**, **Size**, **CovGap**, **Selection Ratio**, **SCE (Set Calibration Error)**.  
   - These help you assess how well the conformal sets match your target coverage and how large the predictive sets are on average.

2. **Standard Classification Metrics**  
   - Maintain your usual classification metrics (accuracy, AUC, sensitivity, specificity) to ensure the model’s predictive performance remains robust.

3. **Interpretability Integration**  
   - TorchCP focuses on conformal prediction, so for interpretability, pair it with external libraries (e.g., Captum, Grad-CAM, or attention-based methods).  
   - Overlay saliency maps or attention heatmaps on medical images, highlighting areas the model deems important.  
   - Cross-check regions of high uncertainty with areas highlighted by interpretability methods to see if uncertainty correlates with ambiguous or complex image regions.

---

## 6. Iterative Refinement

1. **Hyperparameter Tuning**  
   - Systematically adjust learning rates, batch sizes, calibration parameters, and conformal thresholds to balance coverage (or error rate) with interpretability and classification accuracy.

2. **Robustness Checks**  
   - Test the pipeline on diverse patient populations or external validation sets to ensure generalization.  
   - Evaluate how conformal intervals/prediction sets change when data distribution shifts or when images contain artifacts.

3. **Ensemble or Multi-Stage Approaches**  
   - Consider using **Ensemble** methods from TorchCP (see “Ensemble” under Loss & Trainer) for better robustness and calibration.  
   - In medical imaging, ensembles often yield more stable and reliable uncertainty estimates.

---

## 7. Deployment and Documentation

1. **Clinical-Grade Reporting**  
   - Provide both classification probabilities and conformal sets, highlighting the coverage/confidence for each diagnosis.  
   - Include interpretability visuals for radiologists or clinicians, explaining why a model flagged an image as high risk or uncertain.

2. **Regulatory Considerations**  
   - If you plan on clinical deployment, document the model’s performance (AUROC, coverage rates, set sizes) thoroughly.  
   - Track any data used for calibration and demonstrate that coverage thresholds meet clinical requirements.

3. **Open-Source and Collaboration**  
   - Keep your code, data splits, and calibration procedures transparent. This promotes reproducibility and collaboration with other researchers or clinical partners.

