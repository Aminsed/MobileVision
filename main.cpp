#include "includeLibraries.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "matplotlibcpp.h"
#include "displayImages.h"

namespace {
    constexpr size_t kInputWidth = 300;
    constexpr size_t kInputHeight = 300;
    constexpr double kInputScaleFactor = 1.0 / 127.5;
    constexpr float kConfidenceThreshold = 0.7;
    const cv::Scalar kMeanValue(127.5, 127.5, 127.5);

    const std::string kConfigFile = MODEL_PATH + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
    const std::string kModelFile = MODEL_PATH + "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
    const std::string kClassFile = MODEL_PATH + "coco_class_labels.txt";
}

std::vector<std::string> LoadClasses(const std::string& class_file) {
    std::vector<std::string> classes;
    std::ifstream ifs(class_file.c_str());
    std::string line;
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }
    return classes;
}

cv::Mat DetectObjects(const cv::dnn::Net& net, const cv::Mat& frame) {
    cv::Mat input_blob = cv::dnn::blobFromImage(frame, kInputScaleFactor, cv::Size(kInputWidth, kInputHeight),
                                                kMeanValue, true, false);
    net.setInput(input_blob);
    cv::Mat detection = net.forward("detection_out");
    cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    return detection_mat;
}

void DisplayText(cv::Mat& image, const std::string& text, int x, int y) {
    int baseline;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
    cv::rectangle(image, cv::Point(x, y - text_size.height - baseline), cv::Point(x + text_size.width, y + baseline),
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(image, text, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
}

void DisplayObjects(cv::Mat& image, const cv::Mat& objects, const std::vector<std::string>& classes, float threshold = 0.25) {
    for (int i = 0; i < objects.rows; ++i) {
        int class_id = static_cast<int>(objects.at<float>(i, 1));
        float score = objects.at<float>(i, 2);

        if (score > threshold) {
            int x = static_cast<int>(objects.at<float>(i, 3) * image.cols);
            int y = static_cast<int>(objects.at<float>(i, 4) * image.rows);
            int width = static_cast<int>(objects.at<float>(i, 5) * image.cols - x);
            int height = static_cast<int>(objects.at<float>(i, 6) * image.rows - y);

            double font_scale = 2.0;
            int thickness = 2;
            int baseline;
            cv::Size text_size = cv::getTextSize(classes[class_id], cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

            cv::rectangle(image, cv::Point(x, y - text_size.height - baseline), cv::Point(x + text_size.width, y),
                          cv::Scalar(0, 0, 255), cv::FILLED);
            cv::putText(image, classes[class_id], cv::Point(x, y - baseline), cv::FONT_HERSHEY_SIMPLEX,
                        font_scale, cv::Scalar(255, 255, 255), thickness);
            cv::rectangle(image, cv::Point(x, y), cv::Point(x + width, y + height), cv::Scalar(255, 255, 255), 4);
        }
    }
}

int main() {
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(kModelFile, kConfigFile);
    std::vector<std::string> classes = LoadClasses(kClassFile);

    cv::Mat image = cv::imread("images/street.png");
    cv::Mat objects = DetectObjects(net, image);

    DisplayObjects(image, objects, classes);

    matplotlibcpp::figure_size(1000, 600);
    matplotlibcpp::imshow(image);
    auto plt_image = displayImage(image);
    plt_image;

    return 0;
}