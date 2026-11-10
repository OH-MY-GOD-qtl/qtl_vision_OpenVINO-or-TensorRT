#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <opencv2/opencv.hpp>

#include <optional>

#include "armor.hpp"

class YOLOBase
{
public:
    virtual std::vector<Armor> detect(const cv::Mat &img, int frame_count) = 0;

    virtual std::vector<Armor> postprocess(
        double scale, cv::Mat &output, const cv::Mat &bgr_img, int frame_count) = 0;
};

// 构造选项：覆盖项各自只对特定后端生效，集中成结构体避免位置参数误读
struct YOLOOptions
{
    bool debug = true;
    std::optional<bool> use_tensorrt;    // 仅 YOLOv5：覆盖配置里的 use_tensorrt
    std::optional<bool> use_traditional; // 仅 YOLOv5：覆盖配置里的 use_traditional
    bool use_classifier = true;          // 仅 YOLOv8：false 直接报错（v8 无类别头）
};

class YOLO
{
public:
    YOLO(
        const std::string &config_path, bool debug = true,
        std::optional<bool> use_tensorrt_override = std::nullopt);
    YOLO(const std::string &config_path, const YOLOOptions &options);

    std::vector<Armor> detect(const cv::Mat &img, int frame_count = -1);

    std::vector<Armor> postprocess(
        double scale, cv::Mat &output, const cv::Mat &bgr_img, int frame_count);

private:
    std::unique_ptr<YOLOBase> yolo_;
};

#endif // AUTO_AIM__YOLO_HPP