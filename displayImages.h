#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "xtl/xbase64.hpp"
#include "xeus/xjson.hpp"

#include "matplotlib.h"

namespace plt = matplotlib;

namespace im {
    struct Image {
        explicit Image(const std::string& filename) {
            std::ifstream fin(filename, std::ios::binary);
            m_buffer << fin.rdbuf();
        }

        std::stringstream m_buffer;
    };

    xeus::xjson MimeBundleRepr(const Image& i) {
        auto bundle = xeus::xjson::object();
        bundle["image/png"] = xtl::base64encode(i.m_buffer.str());
        return bundle;
    }
}

auto DisplayImage() {
    plt::save("tmp.png");
    auto img = im::Image("tmp.png");
    return img;
}

int Min(int x, int y) {
    return (x < y) ? x : y;
}

cv::Mat GenerateSampleGrayImage() {
    cv::Mat demoImage = cv::Mat::zeros(cv::Size(4, 4), CV_8U);
    demoImage.at<uchar>(0, 0) = 0.7003673 * 255;
    demoImage.at<uchar>(0, 1) = 0.74275081 * 255;
    demoImage.at<uchar>(0, 2) = 0.70928001 * 255;
    demoImage.at<uchar>(0, 3) = 0.56674552 * 255;
    demoImage.at<uchar>(1, 0) = 0.97778533 * 255;
    demoImage.at<uchar>(1, 1) = 0.70633485 * 255;
    demoImage.at<uchar>(1, 2) = 0.24791576 * 255;
    demoImage.at<uchar>(1, 3) = 0.15788335 * 255;
    demoImage.at<uchar>(2, 0) = 0.69769852 * 255;
    demoImage.at<uchar>(2, 1) = 0.71995667 * 255;
    demoImage.at<uchar>(2, 2) = 0.25774443 * 255;
    demoImage.at<uchar>(2, 3) = 0.34154678 * 255;
    demoImage.at<uchar>(3, 0) = 0.96876117 * 255;
    demoImage.at<uchar>(3, 1) = 0.6945071 * 255;
    demoImage.at<uchar>(3, 2) = 0.46638326 * 255;
    demoImage.at<uchar>(3, 3) = 0.7028127 * 255;
    return demoImage;
}

cv::Mat GenerateSampleColorImage() {
    cv::Mat demoImage = cv::Mat::zeros(cv::Size(3, 3), CV_8UC3);
    demoImage.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 100, 0);
    demoImage.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);
    demoImage.at<cv::Vec3b>(0, 2) = cv::Vec3b(255, 0, 0);
    demoImage.at<cv::Vec3b>(1, 0) = cv::Vec3b(0, 0, 255);
    demoImage.at<cv::Vec3b>(1, 1) = cv::Vec3b(255, 100, 0);
    demoImage.at<cv::Vec3b>(1, 2) = cv::Vec3b(0, 255, 0);
    demoImage.at<cv::Vec3b>(2, 0) = cv::Vec3b(255, 0, 0);
    demoImage.at<cv::Vec3b>(2, 1) = cv::Vec3b(0, 0, 255);
    demoImage.at<cv::Vec3b>(2, 2) = cv::Vec3b(0, 100, 255);
    return demoImage;
}

int GetBilinearPixelGray(const cv::Mat& imArr, float posX, float posY) {
    float modXi = static_cast<int>(posX);
    float modYi = static_cast<int>(posY);
    float modXf = posX - modXi;
    float modYf = posY - modYi;
    float modXiPlusOneLim = Min(modXi + 1, imArr.rows - 1);
    float modYiPlusOneLim = Min(modYi + 1, imArr.cols - 1);

    float bottomLeft = static_cast<int>(imArr.at<uchar>(modYi, modXi));
    float bottomRight = static_cast<int>(imArr.at<uchar>(modYi, modXiPlusOneLim));
    float topLeft = static_cast<int>(imArr.at<uchar>(modYiPlusOneLim, modXi));
    float topRight = static_cast<int>(imArr.at<uchar>(modYiPlusOneLim, modXiPlusOneLim));
    float b = modXf * bottomRight + (1.0f - modXf) * bottomLeft;
    float t = modXf * topRight + (1.0f - modXf) * topLeft;
    float pxf = modYf * t + (1.0f - modYf) * b;

    return static_cast<int>(pxf + 0.5f);
}

cv::Vec3b GetBilinearPixelColor(const cv::Mat& imArr, float posX, float posY) {
    float modXi = static_cast<int>(posX);
    float modYi = static_cast<int>(posY);
    float modXf = posX - modXi;
    float modYf = posY - modYi;
    float modXiPlusOneLim = Min(modXi + 1, imArr.rows - 1);
    float modYiPlusOneLim = Min(modYi + 1, imArr.cols - 1);

    int numChannels = imArr.channels();
    cv::Vec3b newColor(0, 0, 0);

    for (int chan = 0; chan < numChannels; ++chan) {
        float bottomLeft = static_cast<int>(imArr.at<cv::Vec3b>(modYi, modXi)[chan]);
        float bottomRight = static_cast<int>(imArr.at<cv::Vec3b>(modYi, modXiPlusOneLim)[chan]);
        float topLeft = static_cast<int>(imArr.at<cv::Vec3b>(modYiPlusOneLim, modXi)[chan]);
        float topRight = static_cast<int>(imArr.at<cv::Vec3b>(modYiPlusOneLim, modXiPlusOneLim)[chan]);
        float b = modXf * bottomRight + (1.0f - modXf) * bottomLeft;
        float t = modXf * topRight + (1.0f - modXf) * topLeft;
        float pxf = modYf * t + (1.0f - modYf) * b;

        newColor[chan] = static_cast<uchar>(pxf + 0.5f);
    }
    return newColor;
}

void BilinearInterpolation(const cv::Mat& inputImg, cv::Mat& outputImg, float scale = 25.0f) {
    outputImg = inputImg.clone();
    if (outputImg.channels() == 4)
        cv::cvtColor(outputImg, outputImg, cv::COLOR_RGBA2BGR);

    if (outputImg.channels() == 3) {
        outputImg = cv::Mat::zeros(cv::Size(inputImg.cols * scale, inputImg.rows * scale), CV_8UC3);
        float rowScale = static_cast<float>(inputImg.rows) / outputImg.rows;
        float colScale = static_cast<float>(inputImg.cols) / outputImg.cols;
        for (float row = 0.0f; row < outputImg.rows; ++row) {
            for (float col = 0.0f; col < outputImg.cols; ++col) {
                float orir = row * rowScale;
                float oric = col * colScale;
                outputImg.at<cv::Vec3b>(row, col) = GetBilinearPixelColor(inputImg, oric, orir);
            }
        }
    } else if (outputImg.channels() == 1) {
        outputImg = cv::Mat::zeros(cv::Size(inputImg.cols * scale, inputImg.rows * scale), CV_8UC1);
        float rowScale = static_cast<float>(inputImg.rows) / outputImg.rows;
        float colScale = static_cast<float>(inputImg.cols) / outputImg.cols;
        for (float row = 0.0f; row < outputImg.rows; ++row) {
            for (float col = 0.0f; col < outputImg.cols; ++col) {
                float orir = row * rowScale;
                float oric = col * colScale;
                outputImg.at<uchar>(row, col) = GetBilinearPixelGray(inputImg, oric, orir);
            }
        }
    }
}

auto DisplayImage(const cv::Mat& image) {
    if (image.channels() < 3) {
        cv::imwrite("tmp.png", image);
        cv::Mat newImage = cv::imread("tmp.png");
        plt::imshow(newImage);
        plt::save("tmp.png");
    } else {
        plt::save("tmp.png");
    }
    auto img = im::Image("tmp.png");
    return img;
}

auto DisplayHist(const cv::Mat& image) {
    if (image.channels() == 1) {
        cv::Mat tmp = image.reshape(1, image.rows * image.cols);
        std::vector<uchar> tmpData(tmp.data, tmp.data + tmp.total());
        plt::hist(tmpData);
        auto pltImg = DisplayImage();
        return pltImg;
    } else if (image.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        std::vector<uchar> tmpData;
        cv::Mat tmp;
        for (int i = 0; i < 3; ++i) {
            plt::subplot(1, 3, i + 1);
            tmp = channels[i].reshape(1, channels[i].rows * channels[i].cols);
            tmpData.assign(tmp.data, tmp.data + tmp.total());
            plt::hist(tmpData);
        }
        auto pltImg = DisplayImage();
        return pltImg;
    }
    auto pltImg = DisplayImage();
    return pltImg;
}