#ifndef AUTO_AIM__BLOOD_BAR_HPP
#define AUTO_AIM__BLOOD_BAR_HPP

#include <opencv2/opencv.hpp>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "armor.hpp"
#include "common/color.hpp"
#include "image.hpp"

// 血量灯条：186 mm × 22 mm 的专属灯条，满血全亮，掉血时【点亮长度】变短。
// 未点亮段不发光、在图像中不可见，因此不能直接量满条长度，需要用参考尺度换算：
//   1) 同面装甲板外框宽（已知 mm）→ px/mm，首选，10 m 内可用；
//   2) 血条自身高度 22 mm 自归一化，仅在血条足够大（>= self_norm_min_height_px）时兜底。
struct BloodBarParams
{
    bool enabled = false;
    Color enemy_color = Color::blue;

    // ---- 几何真值（mm）----
    double length_mm = 186;       // 血条满条长度
    double height_mm = 22;        // 血条高度（自归一化参考）
    double armor_width_mm = 141;  // 装甲板外框宽（YOLO 四角的上/下边），尺度参考
    double armor_height_mm = 125; // 装甲板外框高（左/右边），同面校验用

    // ---- 颜色阈值（与全局同名键解耦，允许单独调）----
    double color_threshold = 40;
    double brightness_threshold = 100;

    // ---- 候选筛选 ----
    double min_edge_px = 2;       // 短边下限（远距离时 22 mm 只有几像素）
    double min_area_px2 = 25;     // 轮廓面积下限
    double max_aspect_ratio = 12; // 长边/短边上限
    double min_fill_ratio = 0.55; // 轮廓面积/最小外接矩形面积下限

    // ---- 尺度参考 ----
    double self_norm_min_height_px = 12; // 自归一化启用门槛（≈4.4 m）

    // ---- 血条与装甲板配对 ----
    double max_distance_ratio = 3.0;     // 血条与装甲板中心距上限（相对装甲板宽）
    double min_association_score = 0.25; // 配对得分下限

    // ---- 分档与稳定 ----
    double low_upper = 0.30;
    double medium_upper = 0.60;
    double hysteresis = 0.05;
    int median_frames = 5;
    // 估计超过该值视为几何错配（配错板/尺度错配），报 unknown 而不是截断成满血。
    // 留 5% 容差吸收满血时的量测噪声。
    double max_valid_percent = 1.05;
};

// 读取 yaml 的 hp_bar 段；缺键回退默认值（不 exit，兼容旧配置）
BloodBarParams load_blood_bar_params(const std::string &config_path);

// 一个血条候选：旋转矩形的两条边分别记录长度与方向，
// 长度轴由装甲板朝向或图像水平方向决定（低血量时点亮段可能比 22 mm 还短）。
struct BloodBar
{
    cv::RotatedRect rect;
    std::vector<cv::Point2f> points; // 4 角点，顺序沿矩形
    cv::Point2f center;
    double edge_len[2] = {0, 0};
    cv::Point2f edge_dir[2]; // 两条边的单位方向
    double area = 0;
};

class BloodBarDetector
{
public:
    explicit BloodBarDetector(const BloodBarParams &params);

    std::vector<BloodBar> detect(const cv::Mat &bgr);

private:
    BloodBarParams params_;
    ChannelThresholder thresholder_;
    cv::Mat binary_;
    std::vector<std::vector<cv::Point>> contours_;
    cv::Mat erode_kernel_;
};

// 百分比估计。armor_points 为空表示没有可用装甲板，退回自归一化。
// 返回 nullopt 表示无法可靠估计。
std::optional<double> estimate_percent(
    const BloodBar &bar, const std::vector<cv::Point2f> &armor_points,
    const BloodBarParams &params);

// 血条与某块装甲板的同面拟合结果：percent 为血量估计，score 为同面置信度。
struct BarArmorFit
{
    double percent = -1;
    double score = 0;
};

// 四角顺序沿用 Armor::points（左上、右上、右下、左下）。
// 尺度用板宽（偏转时与血条长度同步压缩，可互相抵消）；
// 同面校验用板高（偏转时不压缩）。
std::optional<BarArmorFit> fit_bar_to_armor(
    const BloodBar &bar, const std::vector<cv::Point2f> &armor_points,
    const BloodBarParams &params);

// 血量档位（编码与 ROS2 上报一致：0 低 / 1 中 / 2 高 / 3 unknown）
enum class HpLevel
{
    low = 0,
    medium = 1,
    high = 2,
    unknown = 3
};

struct HpEstimate
{
    HpLevel level = HpLevel::unknown;
    double percent = -1;
    bool valid = false;
};

// 带迟滞的分档：进入/退出边界不同，避免在 30%/60% 附近抖动
HpLevel classify_hp(double percent, HpLevel previous, const BloodBarParams &params);

class HpStabilizer
{
public:
    explicit HpStabilizer(const BloodBarParams &params);

    HpEstimate update(const std::optional<double> &percent);

private:
    BloodBarParams params_;
    std::deque<double> window_;
    HpLevel level_ = HpLevel::unknown;
};

// 门面：检测 → 与装甲板配对 → 估计 → 稳定。主程序只依赖这一个类。
class BloodBarPipeline
{
public:
    explicit BloodBarPipeline(const std::string &config_path);

    // armors 用当前帧检测结果；内部按「同面 + 近距离 + 朝向一致」选最佳配对。
    HpEstimate update(const cv::Mat &bgr, const std::vector<Armor> &armors);

    bool enabled() const { return params_.enabled; }

private:
    BloodBarParams params_;
    BloodBarDetector detector_;
    HpStabilizer stabilizer_;

    std::optional<double> pick(
        const std::vector<BloodBar> &bars, const std::vector<Armor> &armors,
        const cv::Size &img_size) const;
};

#endif // AUTO_AIM__BLOOD_BAR_HPP
