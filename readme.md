# Background Removal with DINOv2 for Improved Object Recognition

## Overview

This repository demonstrates a **background removal pipeline** for live camera images using **DINOv2**. The goal is to improve object recognition performance and control stability by filtering out irrelevant background pixels. The pipeline leverages the features of a reference image to extract the foreground in real-time from live images.

---

## Key Contributions

* **Automatic background removal** using DINOv2 patch-to-global feature similarity.
* **Evaluation of recognition performance** with and without background removal using Accuracy, mIoU, and Confusion Matrix metrics.
* **Analysis of system impact**, e.g., control deviation over time.
* **One-shot method**: no additional training required; simply provide a reference image with background removed.

---

## Method

1. **Prepare Reference Image**

   * Manually remove the background and replace it with black or white.
   * Extract the **global DINOv2 feature vector** from this reference image.

2. **Process Live Camera Images**
   For each live frame:

   * Extract DINOv2 **patch features** (local features).
   * Compute **cosine similarity** between the global reference feature and each patch feature of the live frame.
   * Generate a **similarity map**: bright areas indicate patches similar to the reference foreground.
   * Apply a **threshold** (e.g., 0.5) to extract the foreground.

> ⚠ Note: The resulting mask is lower resolution than the input image.
> DINOv2 typically uses 14×14 pixel patches, so for a 518×518 input image, the mask resolution will be 37×37 pixels. For background removal, this is generally sufficient.

---

## Experimental Setup

* Compare the pipeline **with and without background removal**.
* Metrics to evaluate:

  * **Accuracy**
  * **Mean Intersection over Union (mIoU)**
  * **Confusion Matrix**
* Optionally, plot **control deviations over time** to assess system performance.

**Important:** Always use a reference image with background removed to generate accurate foreground masks.

---

## Advantages

* Works **without additional training**.
* Adaptable to any new object by simply updating the reference image.
* Improves object recognition and downstream control stability by providing a clean foreground input.

---

## Optional Extensions

* Use **multiple reference images** to handle different angles or lighting conditions.
* Experiment with **threshold values** for fine-tuning mask accuracy.
* Compare against standard segmentation models (e.g., Mask R-CNN) for evaluation.

---

## Getting Started

1. Mount your Google Drive or upload images to Colab.
2. Prepare a **reference image** with background removed.
3. Run the notebook to extract features and generate masks for live images.
4. Evaluate recognition metrics and system performance.

---
