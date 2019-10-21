// text-recognition.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>


#include "tesseract/baseapi.h"

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

#include <string>
#include <vector>


void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
    std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
{
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4);
    CV_Assert(scores.size[0] == 1); CV_Assert(scores.size[1] == 1);
    CV_Assert(geometry.size[0] == 1);  CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]);
    CV_Assert(scores.size[3] == geometry.size[3]);

    detections.clear();
    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y) {
        const auto* scoresData = scores.ptr<float>(0, 0, y);
        const auto* x0_data = geometry.ptr<float>(0, 0, y);
        const auto* x1_data = geometry.ptr<float>(0, 1, y);
        const auto* x2_data = geometry.ptr<float>(0, 2, y);
        const auto* x3_data = geometry.ptr<float>(0, 3, y);
        const auto* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x) {
            float score = scoresData[x];
            if (score < scoreThresh) {
                continue;
            }

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f;
            float offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5F * (p1 + p3), cv::Size2f(w, h), -angle * 180.0F / static_cast<float>(CV_PI));
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}




cv::Mat detectTextAreas(cv::dnn::Net& net, const cv::Mat& image, std::vector<cv::Rect> &areas)
{
    const float confThreshold = 0.5f;
    const float nmsThreshold = 0.4f;
    const int inputWidth = 320;
    const int inputHeight = 320;

    std::vector<cv::Mat> outs;
    std::vector<cv::String> layerNames(2);
    layerNames[0] = "feature_fusion/Conv_7/Sigmoid";
    layerNames[1] = "feature_fusion/concat_3";

    cv::Mat frame = image.clone();
    cv::Mat blob;

    cv::dnn::blobFromImage(
        frame, blob,
        1.0, cv::Size(inputWidth, inputHeight),
        cv::Scalar(123.68, 116.78, 103.94), true, false);
    net.setInput(blob);
    net.forward(outs, layerNames);

    cv::Mat scores = outs[0];
    cv::Mat geometry = outs[1];

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    decode(scores, geometry, confThreshold, boxes, confidences);

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // Render detections.
    cv::Point2f ratio(static_cast<float>(frame.cols) / inputWidth, static_cast<float>(frame.rows) / inputHeight);
    cv::Scalar green = cv::Scalar(0, 255, 0);

    for (size_t i = 0; i < indices.size(); ++i) {
        cv::RotatedRect& box = boxes[indices[i]];
        cv::Rect area = box.boundingRect();
        area.x *= ratio.x;
        area.width *= ratio.x;
        area.y *= ratio.y;
        area.height *= ratio.y;

        if (area.x < 0)
        {
            area.width += area.x;
            area.x = 0;
        }
        if (area.y < 0)
        {
            area.height += area.y;
            area.y = 0;
        }
        if (area.x + area.width > frame.cols)
        {
            area.width = frame.cols - area.x;
        }
        if (area.y + area.height > frame.rows)
        {
            area.height = frame.rows - area.y;
        }

        if (area.width <= 0 || area.height <= 0)
        {
            continue;
        }

        areas.push_back(area);
        cv::rectangle(frame, area, green, 1);
        auto index = std::to_string(i);
        cv::putText(
            frame, index, cv::Point2f(area.x, area.y - 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, green, 1);
    }
    return frame;
}




const char TESSDATA_PREFIX[] = "C:/Program Files (x86)/Tesseract-OCR/tessdata";

const char MODEL[] = "/model/frozen_east_text_detection.pb";



int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Wrong number of arguments: input parameter required.\n";
        return 1;
    }

    char *old_ctype = _strdup(setlocale(LC_ALL, nullptr));
    setlocale(LC_ALL, "C");

    auto tesseractAPI = new tesseract::TessBaseAPI();
    // Initialize tesseract-ocr with English, with specifying tessdata path
    if (tesseractAPI->Init(TESSDATA_PREFIX, "eng") != 0) {
        std::cerr << "Could not initialize tesseract.\n";
        return 1;
    }

    auto net = cv::dnn::readNet(MODEL);

    cv::VideoCapture cap;
    if ((isdigit(argv[1][0]) != 0) && argv[1][0] == '\0') {
        cap.open(argv[1][0] - '0');
    }
    else {
        cap.open(argv[1]);
    }

    const std::string kWinName = "EAST: An Efficient and Accurate Scene Text Detector";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);


    while (cv::waitKey(1) < 0)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        cv::Mat image;
        frame.convertTo(image, CV_8UC3);

        const int stride = image.step[0];

        tesseractAPI->SetImage(image.data, image.cols, image.rows, 3, stride);

        std::vector<cv::Rect> areas;
        cv::Mat newImage = detectTextAreas(net, image, areas);

        imshow(kWinName, newImage);

        for (cv::Rect &rect : areas) {
            tesseractAPI->SetRectangle(rect.x, rect.y, rect.width, rect.height);

            std::unique_ptr<char[]> outText{ tesseractAPI->GetUTF8Text() };

            std::cout << outText << '\n';
        }
    }

    setlocale(LC_ALL, old_ctype);
    free(old_ctype);
}
