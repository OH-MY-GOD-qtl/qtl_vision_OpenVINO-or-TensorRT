#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "omni/decider.hpp"
#include "omni/detection.hpp"
#include "camera/usbcamera.hpp"
#include "detector/detector.hpp"
#include "armor/armor.hpp"
#include "tools/thread_safe_queue.hpp"



// 全向感知搜索层：3 路 USB 相机 + 传统检测（仅报“有无目标 + 方位”）。
// 结果带采集时间戳，交由 Decider 做海康优先仲裁与时间对齐。
class Perceptron
{
public:
    Perceptron(
        USBCamera * usbcam1, USBCamera * usbcam2, USBCamera * usbcam3,
        const Decider & decider, const std::string & config_path);

    ~Perceptron();

    // 取走队列中全部结果（非阻塞）
    std::vector<DetectionResult> get_detection_queue();

private:
    // USB 传统检测线程：cam -> Detector(纯灯条几何) -> DetectionResult(USB)
    void search_thread(USBCamera * cam, Detector & detector, const std::string & camera_key);

    std::vector<std::thread> threads_;
    // 容量溢出时丢弃最旧结果，保证搜索层始终保留最新方位
    ThreadSafeQueue<DetectionResult, true> detection_queue_;

    // 每线程独立 Detector 实例（Detector 含帧级缓冲，非线程安全）
    std::shared_ptr<Detector> detector1_;
    std::shared_ptr<Detector> detector2_;
    std::shared_ptr<Detector> detector3_;

    const Decider & decider_;  // 复用外部 Decider（安装角/FOV 单一来源；delta_angle 为 const，可并发调用）
    bool stop_flag_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

#endif