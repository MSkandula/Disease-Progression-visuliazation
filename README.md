# Lung Disease Progression Visualizer (Pulmonary fibrosis)

## Overview
This project focuses on building a 3D lung disease progression model using CT imaging data. The goal is to bridge the gap between 2D medical images and intuitive 3D visual understanding by mapping imaging-derived signals onto a reference lung model.

The system simulates disease progression from **healthy → early → moderate → advanced stages** using an interactive visualisation slider.- Interactive controls allow users to zoom in/out and rotate the 3D lung model for better inspection of disease progression from different angles.


## Dataset
The current prototype uses CT image data from [Radiopaedia pulmonary fibrosis cases](https://radiopaedia.org/search?lang=gb&q=pulmonary+fibrosis&scope=cases)

- Total cases used as of now : 8 patient cases  
- Data type: 2D CT scan images  
- Each image is processed individually through the pipeline by inserting in the code one at a time and the results are stored

For each case, the following signals are extracted:

- **Severity** (mean intensity)  
- **Texture** (standard deviation)  
- **High-Density Ratio (HDR)** (proportion of dense regions)  

These signals are used to drive the 3D disease progression model.

| Case ID | Severity (Mean Intensity) | Texture (Std Dev) | High-Density Ratio (HDR) |
|--------|----------------------------|-------------------|---------------------------|
| Case A | 0.1661                     | 0.1711            | 0.0174                    |
| Case B | 0.2050                     | 0.1721            | 0.0344                    |
| Case C | 0.1961                     | 0.1674            | 0.0393                    |
| Case D | 0.4603                     | 0.0932            | 0.0774                    |
| Case E | 0.2713                     | 0.0745            | 0.0097                    |
| Case F | 0.3070                     | 0.1209            | 0.0324                    |
| Case G | 0.2772                     | 0.1357            | 0.0411                    |
| Case H | 0.1970                     | 0.1242            | 0.0219                    |


## Work Completed As of today (Current Prototype)

### 1. CT Image Processing Pipeline
- Grayscale conversion  
- Gaussian smoothing (noise reduction)  
- Otsu thresholding (lung segmentation)  
- Connected component analysis  
- Morphological operations (mask refinement)  

This produces a clean lung region mask.


### 2. Feature Extraction
From the segmented lungs:
- Severity → overall abnormality  
- Texture → structural irregularity  
- HDR → dense abnormal regions  

These provide interpretable disease indicators.


### 3. 3D Lung Model Integration
- Reference lung mesh from Human Atlas API  
- Signal-driven mapping instead of full reconstruction  
- Spatial weighting (subpleural + basal emphasis)  


### 4. Disease Simulation
- Gaussian-based lesion generation  
- Region-aware disease spread  
- Progressive intensity scaling
- 

### 5. Structural Deformation
Simulated effects include:
- Volume loss (shrinkage)  
- Pleural retraction  
- Basal collapse  
- Fibrotic surface roughness  
- Mild asymmetry  


### 6. Visualisation
- Interactive slider (0 → 1 progression)  
- Stage labels (Healthy → Advanced)  
- Clinically-inspired colour mapping  
- Realistic rendering using lighting and glossiness  

<img width="452" height="293" alt="image" src="https://github.com/user-attachments/assets/53702c86-5194-49f2-ac5d-293f5f1ffefc" />
<img width="452" height="293" alt="image" src="https://github.com/user-attachments/assets/00eea89e-70dd-41a6-81d5-494893c086c2" />
<img width="452" height="293" alt="image" src="https://github.com/user-attachments/assets/8dd2909f-7325-4a1d-8ad8-290c06e21fc8" />
<img width="452" height="293" alt="image" src="https://github.com/user-attachments/assets/72a10e22-32a1-4a04-b6e7-52be12fb7df2" />
<img width="452" height="293" alt="image" src="https://github.com/user-attachments/assets/3d48026d-00b0-409a-8048-4caa1932ead3" />
<img width="452" height="293" alt="image" src="https://github.com/user-attachments/assets/21aaf59b-a9f8-4e85-a511-02403ebe6729" />






## Current Results
The prototype currently consists of two working components:

- CT image processing and feature extraction pipeline  
- 3D disease progression visualisation model  

The system is able to:
- Extract disease-related signals (severity, texture, HDR) from CT images  
- Simulate disease progression on a 3D lung model using an interactive slider  

However, the direct integration between CT-derived signals and the 3D model is still under development.



## Limitations
- Segmentation is sensitive to CT image quality  
- No direct anatomical mapping from 2D CT to 3D mesh  
- Current model uses signal-based approximation rather than full reconstruction  


## Future Work
- Integrate CT-derived signals directly into the 3D progression model (linking severity, texture, and HDR to deformation and colouring)
- Categorise cases into discrete disease stages (Healthy, Early, Moderate, Advanced)  
- Improve segmentation robustness  
- Strengthen mapping between CT signals and deformation  
- Enhance visual realism of the 3D model  
- Integrate LLM for generating explanations of disease progression  
- Support multiple CT slices for improved accuracy  


## Tech Stack
- Python  
- OpenCV  
- NumPy  
- PyVista  
- Trimesh  


🚧 Work in progress – Currently working on the preliminary prototype and other requireemnts like presentation and also a detailed report. 
