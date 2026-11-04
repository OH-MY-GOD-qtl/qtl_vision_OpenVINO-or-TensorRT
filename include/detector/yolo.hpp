#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <opencv2/opencv.hpp>

#include <optional>

#include "armor.hpp"

class YOLOBase
{
public:
    virtual std::vector<Armor> detect(const cv::Mat & img, int frame_count) = 0;

    virtual std::vector<Armor> postprocess(
        double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;
};

class YOLO
{
public:
    // use_tensorrt_override：仅 YOLOv5 生效；用于让 USB 搜索链路强制走 OpenVINO
    YOLO(
        const std::string & config_path, bool debug = true,
        std::optional<bool> use_tensorrt_override = std::nullopt);

    std::vector<Armor> detect(const cv::Mat & img, int frame_count = -1);

    std::vector<Armor> postprocess(
        double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

private:
    std::unique_ptr<YOLOBase> yolo_;
};

#endif  // AUTO_AIM__YOLO_HPP