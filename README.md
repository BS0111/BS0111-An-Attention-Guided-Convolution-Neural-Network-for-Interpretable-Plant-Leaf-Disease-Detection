# An Attention-Guided Convolution Neural Network for Interpretable Plant Leaf Disease Detection

## Description
This repository presents an **explainable deep learning framework for plant leaf disease classification and localization**. The proposed approach is based on a **CBAM-VGG16 architecture**, where the VGG16 backbone is enhanced with **Convolutional Block Attention Modules (CBAM)** at all convolutional stages to improve feature representation, disease localization, and interpretability.

The model is evaluated on **five publicly available plant disease datasets**—Apple, PlantVillage, Embrapa, Maize, and Rice. In addition to classification performance, multiple explainable AI techniques, including **CBAM attention maps, Grad-CAM, Grad-CAM++, and Layer-wise Relevance Propagation (LRP)**, are used to visualize disease-affected regions, improving transparency and reliability for agricultural decision support.

---

## Datasets
This work uses five publicly available datasets.  
Only dataset links and descriptions are provided; datasets are **not hosted** in this repository.

Detailed dataset information is available in:  
📁 `datasets/README.md`

**Datasets used:**
- Apple Leaf Disease Dataset  
- PlantVillage Dataset  
- Embrapa Plant Disease Dataset  
- Maize Leaf Disease Dataset  
- Rice Leaf Disease Dataset  

---

## Model
- **Architecture:** CBAM-VGG16  
- **Backbone:** VGG16  
- **Attention Mechanism:** Convolutional Block Attention Module (CBAM)  
- **Framework:** PyTorch  

The same CBAM-VGG16 model implementation is used across all five datasets, with dataset-specific configurations handled during training and evaluation.


Model implementation and training scripts are available in:  
📁 `model/`

---

## Explainability Methods
The following explainable AI techniques are used to interpret model predictions:
- CBAM Attention Maps  
- Grad-CAM  
- Grad-CAM++  
- Layer-wise Relevance Propagation (LRP)  

These methods help identify and highlight disease-affected regions in leaf images.

---

## Results
Experimental results for each dataset are stored separately.

📁 `results/`

Each dataset folder contains:
- Quantitative evaluation metrics
- Visual results and explainability outputs (where applicable)




