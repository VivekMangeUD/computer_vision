# Computer Vision Projects

**Author:** Vivek Mange

This repository contains several computer vision projects that explore various aspects of image processing and machine learning. Below is a brief description of each project, including objectives and concepts.

## 1. Image Transformation

**Objective:** Estimate affine and perspective transformation matrices between original and transformed images.

**Concepts:**
- **Affine Transformation Matrix (X1):** Estimate using a minimum of 3 corresponding points.
- **Perspective Transformation Matrix (X2):** Estimate using a minimum of 4 corresponding points.
- **Methods:** Implement matrix estimation using Least Squares or SVD. Compare with built-in OpenCV functions.

**Deliverables:**
- Estimated matrices X1 and X2.
- Transformed images.
- Errors compared to built-in functions.

## 2. Image Convolve Function Transformation

**Objective:** Implement convolution and multi-resolution image processing functions.

**Concepts:**
- **Convolve:** Apply a kernel to an image to produce a convolved image.
- **Reduce:** Downscale an image by half its width and height.
- **Expand:** Upscale an image by twice its width and height.
- **Gaussian Pyramid:** Create a multi-resolution image pyramid.
- **Laplacian Pyramid:** Generate a Laplacian pyramid from a Gaussian pyramid.
- **Reconstruct:** Rebuild the original image from the Laplacian pyramid.

**Deliverables:**
- Implementations of convolution, reduce, expand, and pyramid functions.
- Error analysis and results.

## 3. Image Classification with VGG16

**Objective:** Train a VGG16 network on the CIFAR100 dataset for image classification.

**Concepts:**
- **Data Preparation:** Resize images to 224x224, normalize, and create data loaders.
- **Model Modification:** Replace the last fully connected layer for CIFAR100.
- **Training:** Train for 10 epochs, using GPU if available.
- **Testing:** Evaluate accuracy and compare with built-in functions.

**Deliverables:**
- Trained VGG16 model.
- Accuracy results on the test set.

## 4. Custom Convolutional Neural Network

**Objective:** Design and train a custom CNN for image classification on the CIFAR100 dataset.

**Concepts:**
- **Architecture:** Include at least 3 convolutional layers, 1 fully connected layer, and 1 max-pooling layer.
- **Training:** Train from scratch for a minimum of 30 epochs.
- **Testing:** Load the best model and evaluate accuracy.

**Deliverables:**
- Trained custom CNN model.
- Accuracy results on the test set.

## 5. Semantic Segmentation using FCN

**Objective:** Perform semantic segmentation using Fully Convolutional Networks (FCN).

**Concepts:**
- **Network Choice:** Use fcn_resnet50 or fcn_resnet101.
- **Feature Maps:** Obtain 21 feature maps from the network.
- **Segmentation:** Create segmentation images with different colors for each class.

**Deliverables:**
- Segmented images for provided datasets.
- Results for different resolutions and feature maps.


## 6. Stereo Analysis System

Design a stereo analysis system with region-based and feature-based matching.

**Concepts:**
- **Region-Based:** Match image regions using metrics like SAD, SSD, NCC.
- **Feature-Based:** Detect and match features using Harris corners.
- **Multi-Resolution:** Perform matching at multiple levels with validity checks.

**Deliverables:**
- Matching results for 5 stereo pairs.
- Customizable parameters for templates, search neighborhoods, and matching methods. 

## 7. Lane Detection using Hough Transform

**Objective:** Detect lanes in images using the Hough Transform.

**Concepts:**
- **Edge Detection:** Identify lane boundaries using edge detection.
- **Hough Transform:** Detect straight lines representing lanes.
- **Lane Marking:** Overlay detected lanes on the original image for visualization.

**Deliverables:**
- Detected lanes overlaid on original images.
- Results and accuracy of lane detection.

### Data and Deliverables

- **Data:** Provided in respective folders for each project.
- **Deliverables:** Include estimated matrices, transformed images, accuracy results, and error comparisons as specified for each project.

### Implementation

- Follow the outlined methods and algorithms for each project.
- Validate results using built-in functions where applicable.
- Document and report all results, including images and errors.

