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
## Image Pre-Processing
In machine learning applications involving image data, it is essential to employ a robust preprocessing routine to enhance model training and ensure consistent evaluation. The preprocessing steps typically include **resizing** all images to a fixed dimension, introducing **random transformations such as flips** to augment the training dataset, and **normalizing the pixel values** across images. These random transformations help the model learn to generalize from varied data presentations, effectively improving its robustness. For validation and testing, the preprocessing simplifies; images are **resized and normalized** in the same way as the training set but without random transformations. This consistency ensures that the model's performance evaluation is based on processing conditions identical to those it trained under, providing a true test of its capabilities on new, unaltered images.

<p align="center">
<a href="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/data2.png"><img src="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/data2.png" align="center"width="280" height="280"></a>
</p>

## Methods

### 1. Our architecture

Below is the summary of the CNN model architecture used in our project:

```plaintext
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1        [-1, 128, 226, 226]           3,584
       BatchNorm2d-2          [-1, 128, 75, 75]             256
            Conv2d-3          [-1, 256, 77, 77]         295,168
       BatchNorm2d-4          [-1, 256, 25, 25]             512
            Conv2d-5          [-1, 256, 27, 27]         590,080
       BatchNorm2d-6            [-1, 256, 9, 9]             512
            Linear-7                  [-1, 256]       5,308,672
            Linear-8                  [-1, 256]          65,792
            Linear-9                  [-1, 256]          65,792
           Linear-10                  [-1, 256]          65,792
           Linear-11                    [-1, 1]             257
================================================================
Total params: 6,396,417
Trainable params: 6,396,417
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 69.76
Params size (MB): 24.40
Estimated Total Size (MB): 94.74
----------------------------------------------------------------
```

#### Result

```bash
               precision    recall  f1-score   support

    fractured       0.93      0.96      0.94       238
not fractured       0.96      0.93      0.95       268

     accuracy                           0.94       506
    macro avg       0.94      0.95      0.94       506
 weighted avg       0.95      0.94      0.94       506
```

### 2. VGG19 architecture
<p align="center">
<a href="https://github.com/mo26-web/Surface-Crack-Detection-with-DL/blob/main/images/vgg19.JPG?raw=true"><img src="https://github.com/mo26-web/Surface-Crack-Detection-with-DL/blob/main/images/vgg19.JPG?raw=true" align="center"width="600" height="300" ></a>
</p>

#### Result

```bash
               precision    recall  f1-score   support

    fractured       0.87      0.90      0.88       238
not fractured       0.91      0.88      0.89       268

     accuracy                           0.89       506
    macro avg       0.89      0.89      0.89       506
 weighted avg       0.89      0.89      0.89       506
```

### 3. MobileNetV3 architecture
<p align="center">
<a href="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/mobilenetv3.png"><img src="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/mobilenetv3.png" align="center"width="600" height="300"></a>
</p>

#### Result

```bash
               precision    recall  f1-score   support

    fractured       0.73      0.95      0.82       238
not fractured       0.93      0.69      0.79       268

     accuracy                           0.81       506
    macro avg       0.83      0.82      0.81       506
 weighted avg       0.84      0.81      0.81       506
```

### 4. RegNet architecture
<p align="center">
<a href="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/RegNet.png"><img src="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/RegNet.png" align="center"width="600" height="300"></a>
</p>

#### Result
```bash
               precision    recall  f1-score   support

    fractured       0.81      0.76      0.79       238
not fractured       0.80      0.84      0.82       268

     accuracy                           0.81       506
    macro avg       0.81      0.80      0.80       506
 weighted avg       0.81      0.81      0.81       506
```

### 5. Wide ResNet architecture
<p align="center">
<a href="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/wideresnet.png"><img src="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/wideresnet.png" align="center"width="600" height="300"></a>
</p>

#### Result
```bash
               precision    recall  f1-score   support

    fractured       0.86      0.89      0.88       238
not fractured       0.90      0.87      0.89       268

     accuracy                           0.88       506
    macro avg       0.88      0.88      0.88       506
 weighted avg       0.88      0.88      0.88       506
```

### 6. Vision transformer (ViT) architecture
<p align="center">
<a href="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/VIT.png"><img src="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/VIT.png" align="center"width="600" height="300"></a>
</p>

#### Result
```bash
               precision    recall  f1-score   support

    fractured       0.95      0.95      0.95       238
not fractured       0.96      0.95      0.95       268

     accuracy                           0.95       506
    macro avg       0.95      0.95      0.95       506
 weighted avg       0.95      0.95      0.95       506
```

### 7. Swin transformer architecture
<p align="center">
<a href="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/swin.png"><img src="https://github.com/mo26-web/Bone-Fracture-Classification/blob/main/images/swin.png" align="center"width="600" height="300"></a>
</p>

#### Result
```bash
              precision    recall  f1-score   support

    fractured       0.88      0.89      0.88       238
not fractured       0.90      0.89      0.89       268

     accuracy                           0.89       506
    macro avg       0.89      0.89      0.89       506
 weighted avg       0.89      0.89      0.89       506
```
## Best result
