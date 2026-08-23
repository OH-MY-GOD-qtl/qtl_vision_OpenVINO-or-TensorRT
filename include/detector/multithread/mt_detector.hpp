#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <tuple>

#include "detector/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"


namespace multithread
{

class MultiThreadDetector
{
public:
    MultiThreadDetector(const std::string & config_path, bool debug = false);

    void push(cv::Mat img, std::chrono::steady_clock::time_point t);

    std::tuple<std::vector<Armor>, std::chrono::steady_clock::time_point> pop();  //暂时不支持yolov8

    std::tuple<cv::Mat, std::vector<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    std::string device_;
    YOLO yolo_;

    // 修改队列配置，当队列满时弹出旧帧
    // 队列同时持有 letterbox 输入缓冲(input)与原始帧(img)，保证异步推理期间输入数据生命周期有效
    ThreadSafeQueue<
        std::tuple<cv::Mat, cv::Mat, std::chrono::steady_clock::time_point, ov::InferRequest>,
        true>  // 设置PopWhenFull为true
        queue_{16, [] { logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

}  // namespace multithread


#endif  // AUTO_AIM__MT_DETECTOR_HPP