#ifndef IO__DAHENGVISION_HPP
#define IO__DAHENGVISION_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

// 大恒相机SDK头文件
#include "GxIAPI.h"
#include "camera/camera.hpp"
#include "tools/thread_safe_queue.hpp"


class DahengVision : public CameraBase
{
public:
  DahengVision(double exposure_ms, double gamma, const std::string & vid_pid);
  ~DahengVision() override;
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  double exposure_ms_, gamma_;
  GX_DEV_HANDLE handle_;
  int64_t height_, width_;
  bool quit_, ok_;
  std::thread capture_thread_;
  std::thread daemon_thread_;
  ThreadSafeQueue<CameraData> queue_;
  int vid_, pid_;

  void open();
  void try_open();
  void close();
  void set_vid_pid(const std::string & vid_pid);
  void reset_usb() const;
};


#endif  // IO__DAHENGVISION_HPP