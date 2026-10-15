#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>


class CameraBase
{
public:
    virtual ~CameraBase() = default;
    virtual void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) = 0;

    // 带超时的读取：返回 false 表示 timeout_ms 内没有取到新帧。
    // 默认实现退化为阻塞读（返回 true），只有支持超时的后端（HikRobot / USBCamera）才真正非阻塞；
    // 调用方可借此在"相机缺失/掉线"时保持循环可控，避免退出时永久阻塞。
    virtual bool try_read(
        cv::Mat & img, std::chrono::steady_clock::time_point & timestamp, int timeout_ms)
    {
        (void)timeout_ms;
        read(img, timestamp);
        return true;
    }
};

class Camera
{
public:
    Camera(const std::string & config_path);
    void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
    bool try_read(
        cv::Mat & img, std::chrono::steady_clock::time_point & timestamp, int timeout_ms = 100);

private:
    std::unique_ptr<CameraBase> camera_;
};


#endif  // IO__CAMERA_HPP