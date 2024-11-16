
# Skin Cancer Detection Model: Inclusive, Sensitive, and Precise

## Overview
This repository contains the code and resources for our research paper, *"Augmented Transfer Learning for Skin Cancer Detection: Enhancing Accuracy using Edge Detection"*. Our model is designed to classify skin lesions as benign or malignant with a focus on **sensitivity across all skin tones**, ensuring equity in early diagnosis.

Skin cancer detection has traditionally struggled with bias due to limited representation of diverse skin tones in training datasets. Our approach addresses this limitation through:
- **Advanced Convolutional Neural Networks (CNNs)** with edge-detection enhancements.
- **Augmented datasets** to improve generalization across skin tones.
- **Ethical AI practices** to ensure unbiased diagnostic outcomes.

## Features
- **Preprocessing Pipelines**: Efficient image processing with techniques like rotation, zoom, and flipping for robust model training.
- **Transfer Learning Model**: MobileNetV2 fine-tuned for skin lesion classification.
- **Custom CNN Architectures**:
  - **DermCNN**: A baseline model with batch normalization and dropout.
  - **AccuDermCNN**: Enhanced with Sobel edge detection for improved feature extraction.
- **Focus on Inclusion**: Tested on diverse datasets to ensure accurate detection across different skin tones.

## Dataset
The models were trained and validated on publicly available datasets:
- [ISIC Dataset](https://isic-archive.com/)
- Additional augmented datasets for skin tones underrepresented in conventional medical datasets.

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/skin-cancer-detection.git
   cd skin-cancer-detection
   ```


2. **Run the model**:
   ```bash
   python custom_cnn.py --model AccuDermCNN
   ```

4. **Test on custom images**:
   ```bash
   python predict.py --image path/to/image.jpg
   ```

## Ethical AI Considerations
This project prioritizes inclusivity and transparency in skin cancer diagnostics:
- **Bias Mitigation**: Augmented datasets to balance skin tone representation.
- **Explainability**: Outputs include Grad-CAM visualizations for human verification.
- **Open Source**: Encouraging collaboration to improve diagnostic tools worldwide.

