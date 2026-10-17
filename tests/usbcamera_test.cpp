#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

int main()
{
    // 曝光/增益/白平衡等相机控制全部删除，交给相机自动决定。
    // 但流格式/分辨率/帧率不能删：OpenCV 默认会协商 YUYV 原始格式，
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "Failed to open USB camera: /dev/video0\n";
        return 1;
    }

    camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    camera.set(cv::CAP_PROP_FPS, 10);

    std::cout << "Opened /dev/video0: " << camera.get(cv::CAP_PROP_FRAME_WIDTH) << 'x'
              << camera.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
              << camera.get(cv::CAP_PROP_FPS) << " fps\n";

    cv::Mat img;
    while (true) {
        if (!camera.read(img) || img.empty()) {
            std::cerr << "Failed to read frame from USB camera\n";
            return 1;
        }

        cv::imshow("img", img);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}
