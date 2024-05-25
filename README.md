# Bone-Fracture-Classification
This project uses PyTorch to classify bone fractures. As well as fine-tuning some famous CNN architectures (like VGG 19, MobileNetV3, RegNet,...), we designed our own architecture. Additionally, we used Transformer architectures (such as Vision Transformer and Swin Transformer). This dataset is Bone Fracture Multi-Region X-ray, available on Kaggle.
## Bone Fracture Multi-Region X-ray Dataset
The dataset includes fractured and non-fractured X-ray images covering all anatomical body regions, including the lower limb, upper limb, lumbar region, hips, knees, etc. The dataset is divided into three folders, each containing fractured and non-fractured radiographic images. The dataset can be accessed at https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data/data.
<p align="center">
<a href="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/data.png"><img src="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/data.png" align="center"></a>
</p>

### Dataset Contents:

This dataset contains  **10,580**  radiographic images (X-ray) data.

**Training Data**  
_Number of Images: 9246_

**Validation Data**  
_Number of Images: 828_

**Test Data**  
_Number of Images: 506_
## Image Pre-Process
In machine learning applications involving image data, it is essential to employ a robust preprocessing routine to enhance model training and ensure consistent evaluation. The preprocessing steps typically include **resizing** all images to a fixed dimension, introducing **random transformations such as flips** to augment the training dataset, and **normalizing the pixel values** across images. These random transformations help the model learn to generalize from varied data presentations, effectively improving its robustness. For validation and testing, the preprocessing simplifies; images are **resized and normalized** in the same way as the training set but without random transformations. This consistency ensures that the model's performance evaluation is based on processing conditions identical to those it trained under, providing a true test of its capabilities on new, unaltered images.
