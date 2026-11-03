#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <string>
#include <vector>

#include "armor.hpp"

// 检测结果来源：USB(传统检测，仅报“有无目标 + 方位”) / HIK(海康模型，主瞄准)
enum class DetectionSource
{
    USB,
    HIK
};

//一个识别结果可能包含多个armor,需要排序和过滤。armors, timestamp, delta_yaw, delta_pitch
struct DetectionResult
{
    std::vector<Armor> armors;
    std::chrono::steady_clock::time_point timestamp;
    double delta_yaw;    //rad
    double delta_pitch;  //rad

    // 以下字段带默认值，追加在末尾，保持“{armors, timestamp, delta_yaw, delta_pitch}”
    // 形式的聚合初始化仍然可用
    DetectionSource source = DetectionSource::USB;  // 结果来源
    std::string camera;                             // 相机标识: usb0/usb1/usb2/hik

    // 距离代理：装甲像素宽度 / 图像宽度，越大越近（USB 链路未做 PnP，不能用米制距离）
    double size_score = 0.0;
    // armors 中被选中的装甲下标（按敌方颜色过滤后的最大表观尺寸）
    int selected_index = -1;
};

#endif