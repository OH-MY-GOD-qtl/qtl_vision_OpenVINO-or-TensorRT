#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <array>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "armor.hpp"
#include "camera/usbcamera.hpp"
#include "detector/yolo.hpp"
#include "omni/decider.hpp"
#include "omni/detection.hpp"
#include "tools/thread_safe_queue.hpp"

// 全向感知搜索层：3 路 USB 相机 + YOLO。只做检测：不加载独立数字分类器，
// 也不做传统角点二次矫正（那属于海康精确链路）。
// 选目标仍按表观距离，编号不参与优先级；结果带采集时间戳，交由 Decider 做时间对齐。
class Perceptron
{
public:
    Perceptron(
        USBCamera * usbcam1, USBCamera * usbcam2, USBCamera * usbcam3, const Decider & decider,
        const std::string & config_path);

    ~Perceptron();

    // 取走队列中全部结果（非阻塞）
    std::vector<DetectionResult> get_detection_queue();

    // 调试显示用：三路 USB 的最新送检帧（已应用 omni_usb_detect_gain；无帧时为空）
    std::array<cv::Mat, 3> latest_frames() const;

    // 当前送检亮度缩放，供调试画面标注
    double detect_gain() const { return detect_gain_; }

private:
    // USB 搜索线程：cam -> YOLO 检测 -> DetectionResult(USB)
    // 相机标识 usb0/usb1/usb2 由 index 派生，不再手写第二份身份
    void search_thread(USBCamera * cam, int index);

    std::vector<std::thread> threads_;
    // 容量溢出时丢弃最旧结果，保证搜索层始终保留最新方位
    ThreadSafeQueue<DetectionResult, true> detection_queue_;

    // 每线程独立 YOLO 实例（YOLO 不是线程安全的，不能跨线程共享）
    std::array<std::unique_ptr<YOLO>, 3> detectors_;
    double detect_interval_s_ = 0.0;  // 0 表示不限制
    double detect_gain_ = 1.0;        // 送检前亮度缩放，1.0 表示不处理

    const Decider &
        decider_;  // 复用外部 Decider（安装角/FOV 单一来源；delta_angle 为 const，可并发调用）
    bool stop_flag_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;

    // 调试显示帧快照（仅 --display 使用）
    mutable std::mutex frames_mutex_;
    std::array<cv::Mat, 3> latest_frames_;
};

#endif