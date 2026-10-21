#ifndef IO__USBCamera_HPP
#define IO__USBCamera_HPP

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "camera/camera.hpp"
#include "tools/thread_safe_queue.hpp"

class USBCamera : public CameraBase
{
public:
    USBCamera(const std::string & open_name, const std::string & config_path);
    ~USBCamera();
    cv::Mat read();
    void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
    // 带超时的读取：timeout_ms 内无新帧返回 false（不阻塞），供搜索线程避免永久阻塞
    bool try_read(
        cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
        int timeout_ms = 100) override;
    std::string device_name;

private:
    struct CameraData
    {
        cv::Mat img;
        std::chrono::steady_clock::time_point timestamp;
    };

    std::mutex cap_mutex_;
    cv::VideoCapture cap_;
    cv::Mat img_;
    std::string open_name_;
    int usb_frame_rate_;
    int open_count_;
    double image_width_, image_height_;
    bool quit_, ok_;
    std::thread capture_thread_;
    std::thread daemon_thread_;
    ThreadSafeQueue<CameraData> queue_;

    void try_open();
    void open();
    void close();
};

#endif