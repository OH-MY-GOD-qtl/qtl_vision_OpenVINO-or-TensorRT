#ifndef IO__USBCamera_HPP
#define IO__USBCamera_HPP

#include <atomic>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

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

    // videoN -> /dev/videoN；绝对路径与 v4l/by-path 等相对路径保持可用
    static std::string resolve_device_path(const std::string & device);

    // 把 videoN 映射到 by-id / by-path 的稳定别名；找不到返回空
    // search_dirs 仅供测试注入，默认查 /dev/v4l/by-id 与 /dev/v4l/by-path
    static std::string find_stable_device_path(
        const std::string & device,
        const std::vector<std::string> & search_dirs = {
            "/dev/v4l/by-id", "/dev/v4l/by-path"});

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
    std::string resolved_device_path_;  // videoN 对应的 by-id/by-path 稳定路径（失效后会重解析）
    int usb_frame_rate_;
    std::atomic<int> open_count_;  // 连续打开失败次数，成功取到帧后清零
    double image_width_, image_height_;
    std::atomic<bool> quit_{false};
    std::atomic<bool> ok_{false};
    std::thread capture_thread_;
    std::thread daemon_thread_;
    ThreadSafeQueue<CameraData> queue_;

    void try_open();
    bool open();
    void close();
};

#endif