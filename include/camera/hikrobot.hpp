#ifndef IO__HIKROBOT_HPP
#define IO__HIKROBOT_HPP

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "MvCameraControl.h"
#include "camera/camera.hpp"
#include "tools/thread_safe_queue.hpp"

class HikRobot : public CameraBase
{
public:
    // timestamp_latency_ms：曝光中点之后仍未抵消的固定延迟（传感器读出 + USB 传输）。
    // 曝光中点补偿（exposure/2）在采集线程内按帧信息自动完成，不需要配置。
    HikRobot(
        double exposure_ms, double gain, const std::string & vid_pid,
        double timestamp_latency_ms = 0.0, bool balance_white_auto = true);
    ~HikRobot() override;
    void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
    // 带超时的读取：timeout_ms 内无新帧返回 false（不阻塞），供主循环避免掉线时卡死
    bool try_read(
        cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
        int timeout_ms = 100) override;

private:
    struct CameraData
    {
        cv::Mat img;
        std::chrono::steady_clock::time_point timestamp;
    };

    double exposure_us_;
    double gain_;
    double timestamp_latency_ms_;
    // 连续自动白平衡会让通道差逐帧漂移，而传统检测用的是固定色差阈值，
    // 因此开放为可配置：false = 锁定白平衡（A/B 排查闪烁用）
    bool balance_white_auto_;

    std::thread daemon_thread_;
    std::atomic<bool> daemon_quit_;

    void * handle_;
    std::thread capture_thread_;
    std::atomic<bool> capturing_;
    std::atomic<bool> capture_quit_;
    ThreadSafeQueue<CameraData> queue_;

    bool libusb_ok_;
    int vid_, pid_;

    void capture_start();
    void capture_stop();

    void set_float_value(const std::string & name, double value);
    void set_enum_value(const std::string & name, unsigned int value);

    void set_vid_pid(const std::string & vid_pid);
    void reset_usb() const;
};

#endif  // IO__HIKROBOT_HPP