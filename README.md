# Overview
This repository reimplements and compares the original Neural Style Transfer (NST) algorithm as introduced in the paper
“A Neural Algorithm of Artistic Style” by Gatys et al. using various convolutional neural network architectures.
Our goal was to analyze the relative performance of preexisting models, such as VGG16 and VGG19, against additional 
architectures (AlexNet, MobileNet, ResNet-50, InceptionNet, and an untrained CNN) in transferring artistic style from one image onto another.


Key experiments explored content retention, style representation, runtime, and computational efficiency using the famous images 
"The Great Wave" by Hokusai (style image) and the Golden Gate Bridge (content image).

---

## Features and Contributions

- Performance Comparison: Quantitative and qualitative analysis of NST outputs across seven models and architectures.
- Alternative Architectures: Integration of MobileNet, ResNet-50, AlexNet, InceptionNet, and an untrained CNN in place of VGG models to test NST efficacy and computational trade-offs.
- Expanded Dataset: Added original content images to assess the generalizability of NST algorithms beyond stylized datasets
- New Analysis of Style Loss Tradeoffs: Investigated how deeper or more efficient architectures balance style rendering, content retention, and execution time.


### Key Findings

- VGG16 and VGG19 demonstrated highest content retention and stylization quality but were computationally expensive.
- AlexNet and ResNet-50 provided strong stylized results while being more efficient, with AlexNet converging faster than deeper architectures.
- MobileNet and InceptionNet struggled to transfer artistic features effectively, showing limited stylization.
- Untrained CNN indicated potential for style transfer but produced low-detail and pixelated results.

--- 

## Methodology
### Implemented Models
- VGG16/VGG19: Pretrained models designed for image classification, benchmarked for their superior feature representations.
- AlexNet: Modified to exclude fully connected layers for style/content feature extraction, leveraging intermediate layers (conv3, conv4, conv5) for NST.
- ResNet-50: Extracts deep features using residual blocks, balancing stylization and content retention.
- InceptionNet: Examined how its multi-scale kernels handle style/content but struggled with representation due to BatchNorm effects.
- MobileNet: Optimized for computational efficiency but demonstrated limited success in transferring stylistic features.
- Untrained CNN: Introduced as a baseline to explore whether stylization could occur without pretrained weights.

### Implementation Details
1. Common Preprocessing: Input images resized to consistent dimensions (except InceptionNet) and normalized using ImageNet statistics.
2. Loss Functions:
- Content Loss: Evaluates preservation of original image structure.
- Style Loss: Incorporates Gram matrices to transfer stylistic features.
- Total Variation Loss: Controls smoothness of output images.
3. Optimizer: L-BFGS optimizer used for high-quality results in 500 iterations.
4. Hyperparameters: Fixed weights (Content: 1x10⁵, Style: 3x10⁴, TV: 1x10⁰) across all experiments for fair comparisons.

--- 

## Usage Instructions
Clone the repository:
```
git clone https://github.com/<your-username>/style-transfer-cnn-comparison.git
cd style-transfer-cnn-comparison
```
Run style transfer:
``` 
python neural_style_transfer.py --model <model-name> --content_img_name <content-img-name> --style_img_name <style-img-name>
```
