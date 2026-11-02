#include "blood_bar.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
// ---- yaml 读取：缺键一律回退默认值，避免旧配置因新增键 exit(1) ----
bool read_bool(const YAML::Node & parent, const char * key, bool fallback)
{
    if (!parent || !parent.IsMap() || !parent[key]) return fallback;
    return parent[key].as<bool>();
}

double read_double(const YAML::Node & parent, const char * key, double fallback)
{
    if (!parent || !parent.IsMap() || !parent[key]) return fallback;
    return parent[key].as<double>();
}

int read_int(const YAML::Node & parent, const char * key, int fallback)
{
    if (!parent || !parent.IsMap() || !parent[key]) return fallback;
    return parent[key].as<int>();
}

std::string read_string(const YAML::Node & parent, const char * key, const std::string & fallback)
{
    if (!parent || !parent.IsMap() || !parent[key]) return fallback;
    return parent[key].as<std::string>();
}

double median_of(std::deque<double> values)
{
    if (values.empty()) return 0.0;
    auto mid = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), mid, values.end());
    double result = *mid;
    if (values.size() % 2 == 0) {
        // 偶数个取中间两个的均值
        auto lower = std::max_element(values.begin(), mid);
        result = (*lower + result) / 2.0;
    }
    return result;
}

cv::Point2f unit(const cv::Point2f & v)
{
    const double n = cv::norm(v);
    if (n < 1e-9) return {1.f, 0.f};
    return v / static_cast<float>(n);
}

// 两条平行边的平均长度（宽：上/下边；高：左/右边）
double avg_edge(
    const cv::Point2f & a, const cv::Point2f & b, const cv::Point2f & c,
    const cv::Point2f & d)
{
    return (cv::norm(b - a) + cv::norm(d - c)) / 2.0;
}

std::optional<double> normalize_percent(double percent, double max_percent)
{
    if (!std::isfinite(percent) || percent < 0.0 || percent > max_percent) return std::nullopt;
    return std::clamp(percent, 0.0, 1.0);
}
}  // namespace

// 用装甲板外框宽当尺子量点亮长度，并用板高做「同面」校验：
// 偏转时板宽与血条长度同步压缩、互相抵消；板高不压缩，可独立校验。
std::optional<BarArmorFit> fit_bar_to_armor(
    const BloodBar & bar, const std::vector<cv::Point2f> & armor_points,
    const BloodBarParams & params)
{
    if (armor_points.size() < 4) return std::nullopt;

    const cv::Point2f top_dir = unit(armor_points[1] - armor_points[0]);
    const double width_px =
        avg_edge(armor_points[0], armor_points[1], armor_points[3], armor_points[2]);
    if (width_px < 2.0) return std::nullopt;

    // 长度轴 = 与装甲板上下边更平行的那条边；另一条是 22 mm 的高度轴
    const double dot0 = std::abs(bar.edge_dir[0].x * top_dir.x + bar.edge_dir[0].y * top_dir.y);
    const double dot1 = std::abs(bar.edge_dir[1].x * top_dir.x + bar.edge_dir[1].y * top_dir.y);
    const int len_axis = (dot1 > dot0) ? 1 : 0;
    const double lit_px = bar.edge_len[len_axis];
    const double height_px = bar.edge_len[1 - len_axis];

    const double px_per_mm = width_px / params.armor_width_mm;
    if (px_per_mm < 1e-6) return std::nullopt;

    const auto percent =
        normalize_percent(lit_px / (params.length_mm * px_per_mm), params.max_valid_percent);
    if (!percent) return std::nullopt;

    // 同面校验：用板高换算垂直方向 px/mm。偏转时板宽会压缩、板高不会，
    // 所以只能拿板高来验证血条高是否落在同一平面。
    const double armor_height_px =
        avg_edge(armor_points[0], armor_points[3], armor_points[1], armor_points[2]);
    if (armor_height_px < 2.0) return std::nullopt;

    const double px_per_mm_v = armor_height_px / params.armor_height_mm;
    const double expected_height_px = params.height_mm * px_per_mm_v;
    const double height_err =
        (expected_height_px < 1e-6) ? 1.0 : std::abs(height_px / expected_height_px - 1.0);

    const cv::Point2f armor_center = (armor_points[0] + armor_points[1] + armor_points[2] +
                                      armor_points[3]) /
                                     4.0;
    const double dist_norm = cv::norm(bar.center - armor_center) / width_px;
    if (dist_norm > params.max_distance_ratio) return std::nullopt;

    const double cos_dir = std::max(dot0, dot1);
    // 朝向一致为主，尺寸比为软约束（远距离时高只有几像素，噪声大）
    const double score = cos_dir / (1.0 + 2.0 * height_err) / (1.0 + 0.3 * dist_norm);

    BarArmorFit fit;
    fit.percent = *percent;
    fit.score = score;
    return fit;
}

BloodBarParams load_blood_bar_params(const std::string & config_path)
{
    BloodBarParams params;

    YAML::Node yaml;
    try {
        yaml = YAML::LoadFile(config_path);
    } catch (const std::exception &) {
        return params;
    }

    params.enemy_color =
        (read_string(yaml, "enemy_color", "blue") == "red") ? Color::red : Color::blue;

    const YAML::Node hp = yaml["hp_bar"];
    if (!hp || !hp.IsMap()) return params;

    params.enabled = read_bool(hp, "enabled", params.enabled);
    params.length_mm = read_double(hp, "length_mm", params.length_mm);
    params.height_mm = read_double(hp, "height_mm", params.height_mm);
    params.armor_width_mm = read_double(hp, "armor_width_mm", params.armor_width_mm);
    params.armor_height_mm = read_double(hp, "armor_height_mm", params.armor_height_mm);

    params.color_threshold = read_double(hp, "color_threshold", params.color_threshold);
    params.brightness_threshold =
        read_double(hp, "brightness_threshold", params.brightness_threshold);

    params.min_edge_px = read_double(hp, "min_edge_px", params.min_edge_px);
    params.min_area_px2 = read_double(hp, "min_area_px2", params.min_area_px2);
    params.max_aspect_ratio = read_double(hp, "max_aspect_ratio", params.max_aspect_ratio);
    params.min_fill_ratio = read_double(hp, "min_fill_ratio", params.min_fill_ratio);

    params.self_norm_min_height_px =
        read_double(hp, "self_norm_min_height_px", params.self_norm_min_height_px);

    params.max_distance_ratio = read_double(hp, "max_distance_ratio", params.max_distance_ratio);
    params.min_association_score =
        read_double(hp, "min_association_score", params.min_association_score);

    params.low_upper = read_double(hp, "low_upper", params.low_upper);
    params.medium_upper = read_double(hp, "medium_upper", params.medium_upper);
    params.hysteresis = read_double(hp, "hysteresis", params.hysteresis);
    params.median_frames = read_int(hp, "median_frames", params.median_frames);
    params.max_valid_percent = read_double(hp, "max_valid_percent", params.max_valid_percent);

    return params;
}

BloodBarDetector::BloodBarDetector(const BloodBarParams & params)
: params_(params),
  thresholder_(params.enemy_color, params.color_threshold, params.brightness_threshold),
  erode_kernel_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)))
{
}

std::vector<BloodBar> BloodBarDetector::detect(const cv::Mat & bgr)
{
    std::vector<BloodBar> bars;
    if (!params_.enabled || bgr.empty()) return bars;

    // ChannelThresholder 内部做了两次 3x3 膨胀（为装甲灯条补断点），会把血条量大一整圈。
    // 这里用同核腐蚀两次抵消，保证长度/高度是真实像素尺寸。
    thresholder_.binary(bgr, binary_);
    cv::erode(binary_, binary_, erode_kernel_, cv::Point(-1, -1), 2);

    contours_.clear();
    cv::findContours(binary_, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (const auto & contour : contours_) {
        if (contour.size() < 4) continue;
        const double area = cv::contourArea(contour);
        if (area < params_.min_area_px2) continue;

        const cv::RotatedRect rect = cv::minAreaRect(contour);
        const double rect_area = static_cast<double>(rect.size.width) * rect.size.height;
        if (rect_area < 1e-6) continue;
        if (area / rect_area < params_.min_fill_ratio) continue;

        cv::Point2f corners[4];
        rect.points(corners);

        BloodBar bar;
        bar.rect = rect;
        bar.center = rect.center;
        bar.area = area;
        bar.points.assign(corners, corners + 4);

        const cv::Point2f e0 = corners[1] - corners[0];
        const cv::Point2f e1 = corners[2] - corners[1];
        bar.edge_len[0] = cv::norm(e0);
        bar.edge_len[1] = cv::norm(e1);
        bar.edge_dir[0] = unit(e0);
        bar.edge_dir[1] = unit(e1);

        const double min_edge = std::min(bar.edge_len[0], bar.edge_len[1]);
        const double max_edge = std::max(bar.edge_len[0], bar.edge_len[1]);
        if (min_edge < params_.min_edge_px) continue;
        if (max_edge / min_edge > params_.max_aspect_ratio) continue;

        bars.push_back(std::move(bar));
    }

    return bars;
}

std::optional<double> estimate_percent(
    const BloodBar & bar, const std::vector<cv::Point2f> & armor_points,
    const BloodBarParams & params)
{
    if (!armor_points.empty()) {
        auto fit = fit_bar_to_armor(bar, armor_points, params);
        if (!fit) return std::nullopt;
        return fit->percent;
    }

    // 兜底：自归一化。血条近似水平，取更接近水平的那条边作为点亮长度轴。
    const int len_axis = (std::abs(bar.edge_dir[1].x) > std::abs(bar.edge_dir[0].x)) ? 1 : 0;
    const double lit_px = bar.edge_len[len_axis];
    const double height_px = bar.edge_len[1 - len_axis];
    if (height_px < params.self_norm_min_height_px) return std::nullopt;

    return normalize_percent(
        lit_px * params.height_mm / (height_px * params.length_mm), params.max_valid_percent);
}

HpLevel classify_hp(double percent, HpLevel previous, const BloodBarParams & params)
{
    auto plain = [&params](double value) {
        if (value <= params.low_upper) return HpLevel::low;
        if (value <= params.medium_upper) return HpLevel::medium;
        return HpLevel::high;
    };

    const double h = std::max(0.0, params.hysteresis);
    switch (previous) {
        case HpLevel::low:
            return (percent > params.low_upper + h) ? plain(percent) : HpLevel::low;
        case HpLevel::medium:
            if (percent > params.medium_upper + h) return HpLevel::high;
            if (percent <= params.low_upper - h) return HpLevel::low;
            return HpLevel::medium;
        case HpLevel::high:
            return (percent <= params.medium_upper - h) ? plain(percent) : HpLevel::high;
        case HpLevel::unknown:
        default:
            return plain(percent);
    }
}

HpStabilizer::HpStabilizer(const BloodBarParams & params) : params_(params) {}

HpEstimate HpStabilizer::update(const std::optional<double> & percent)
{
    // 规格 C6：丢检立即 unknown，不做保持
    if (!percent) {
        level_ = HpLevel::unknown;
        window_.clear();
        return {HpLevel::unknown, -1, false};
    }

    window_.push_back(*percent);
    const std::size_t window_size = static_cast<std::size_t>(std::max(1, params_.median_frames));
    while (window_.size() > window_size) window_.pop_front();

    const double percent_median = median_of(window_);
    level_ = classify_hp(percent_median, level_, params_);

    return {level_, percent_median, true};
}

BloodBarPipeline::BloodBarPipeline(const std::string & config_path)
: params_(load_blood_bar_params(config_path)),
  detector_(params_),
  stabilizer_(params_)
{
}

HpEstimate BloodBarPipeline::update(const cv::Mat & bgr, const std::vector<Armor> & armors)
{
    if (!params_.enabled) return stabilizer_.update(std::nullopt);

    const auto bars = detector_.detect(bgr);
    return stabilizer_.update(pick(bars, armors, bgr.size()));
}

std::optional<double> BloodBarPipeline::pick(
    const std::vector<BloodBar> & bars, const std::vector<Armor> & armors,
    const cv::Size & img_size) const
{
    if (bars.empty()) return std::nullopt;

    const cv::Point2f img_center(img_size.width / 2.0f, img_size.height / 2.0f);
    const double half_diag = 0.5 * std::hypot(img_size.width, img_size.height) + 1e-6;

    double best_total = -1;
    std::optional<double> best_percent;

    for (const auto & bar : bars) {
        const double center_term =
            1.0 / (1.0 + cv::norm(bar.center - img_center) / half_diag);

        for (const auto & armor : armors) {
            auto fit = fit_bar_to_armor(bar, armor.points, params_);
            if (!fit || fit->score < params_.min_association_score) continue;

            // 瞄准目标在画面中心附近：配对得分之外再加一点中心权重
            const double total = fit->score * (0.7 + 0.3 * center_term);
            if (total > best_total) {
                best_total = total;
                best_percent = fit->percent;
            }
        }
    }

    if (best_percent) return best_percent;

    // 没有可用装甲板：退回自归一化，取离画面中心最近的血条
    const BloodBar * nearest = nullptr;
    double nearest_dist = std::numeric_limits<double>::max();
    for (const auto & bar : bars) {
        const double dist = cv::norm(bar.center - img_center);
        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest = &bar;
        }
    }
    if (!nearest) return std::nullopt;
    return estimate_percent(*nearest, {}, params_);
}
