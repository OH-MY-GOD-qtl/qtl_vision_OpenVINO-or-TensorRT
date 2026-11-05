#include "detector/yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "detector/yolos/yolo11.hpp"
#include "detector/yolos/yolov5.hpp"
#include "detector/yolos/yolov8.hpp"
#include "tools/yaml.hpp"

YOLO::YOLO(
    const std::string & config_path, bool debug, std::optional<bool> use_tensorrt_override)
: YOLO(config_path, YOLOOptions{debug, use_tensorrt_override, std::nullopt, true})
{
}

YOLO::YOLO(const std::string & config_path, const YOLOOptions & options)
{
    auto yaml = yaml_load(config_path);
    auto yolo_name = yaml_read<std::string>(yaml, "yolo_name");

    if (yolo_name == "yolov8") {
        // v8 模型只有颜色类别头，编号必须靠独立分类器；静默退化成“所有目标都叫 sentry”更危险
        if (!options.use_classifier)
            throw std::runtime_error(
                "yolov8 需要独立数字分类器给出编号；USB 搜索链路请使用 yolov5 或 yolo11");
        yolo_ = std::make_unique<YOLOV8>(config_path, options.debug);
    }

    else if (yolo_name == "yolo11") {
        // YOLO11 自带分类头，始终不加载独立数字分类器
        yolo_ = std::make_unique<YOLO11>(config_path, options.debug);
    }

    else if (yolo_name == "yolov5") {
        yolo_ = std::make_unique<YOLOV5>(
            config_path, options.debug, options.use_tensorrt, options.use_traditional);
    }

    else {
        throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
    }
}

std::vector<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
    return yolo_->detect(img, frame_count);
}

std::vector<Armor> YOLO::postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
    return yolo_->postprocess(scale, output, bgr_img, frame_count);
}
