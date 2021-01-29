# Object Detection with TensorFlow C++ API and MobileNet

This project demonstrates how to perform object detection using the TensorFlow C++ API and the MobileNet model. It utilizes the power of deep learning to detect and localize objects in images with high accuracy and efficiency.

## Features

- Object detection using the MobileNet model, a lightweight and fast convolutional neural network architecture.
- Integration with the TensorFlow C++ API for seamless deployment and inference.
- Real-time object detection on images or video streams.
- Bounding box visualization and class label annotation for detected objects.
- Easy-to-use and extensible codebase for customization and further development.

## Requirements

- C++14 or higher
- CMake 3.10 or higher
- OpenCV library
- TensorFlow C++ API

## Usage

1. Place your input images in the `images` directory.

2. Update the `main.cpp` file to specify the path to your input image:
   ```cpp
   cv::Mat image = cv::imread("images/your-image.jpg");
   ```

3. Customize the detection parameters, such as confidence threshold and class labels, in the `main.cpp` file if needed.

4. Build and run the project as described in the Installation section.

5. The detected objects will be visualized with bounding boxes and class labels on the input image.

## Model

This project utilizes the MobileNet model, a lightweight and efficient convolutional neural network architecture designed for mobile and embedded vision applications. MobileNet achieves a good balance between accuracy and computational efficiency, making it suitable for real-time object detection tasks.

The model is pre-trained on the COCO dataset, which consists of 80 object categories. The `models` directory contains the necessary model files, including the frozen inference graph and class labels.

## TensorFlow C++ API

The TensorFlow C++ API allows for seamless integration of TensorFlow models into C++ applications. It provides a set of powerful tools and libraries for loading, running, and manipulating TensorFlow graphs efficiently.

This project demonstrates how to use the TensorFlow C++ API to load the MobileNet model, perform object detection inference, and extract the detected objects' bounding boxes and class labels.


## Acknowledgments

- The MobileNet model and its pre-trained weights are provided by the TensorFlow Model Zoo.
- The object detection pipeline is inspired by the TensorFlow Object Detection API.
