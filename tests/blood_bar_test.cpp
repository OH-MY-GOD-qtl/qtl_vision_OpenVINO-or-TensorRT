// blood_bar_test：合成图断言（无硬件、无素材）
//
// 覆盖：
//   1) 血条检测（颜色阈值 → 轮廓 → 旋转矩形）
//   2) 装甲板宽度作为尺度的百分比估计（含低血量时点亮段短于条高）
//   3) 10 m 量级的远距离估计
//   4) 自归一化兜底与其启用门槛
//   5) 分档迟滞与丢检 unknown
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "blood_bar.hpp"
#include "comm/hp_reporter.hpp"

namespace
{
int g_failures = 0;

void check(bool ok, const std::string & name)
{
    if (ok) {
        std::cout << "[ OK ] " << name << "\n";
    } else {
        ++g_failures;
        std::cout << "[FAIL] " << name << "\n";
    }
}

void check_near(double got, double want, double tol, const std::string & name)
{
    if (std::abs(got - want) <= tol) {
        std::cout << "[ OK ] " << name << " got=" << got << " want=" << want << "\n";
    } else {
        ++g_failures;
        std::cout << "[FAIL] " << name << " got=" << got << " want=" << want << " tol=" << tol
                  << "\n";
    }
}

// BGR：蓝分量高、灰度过亮度阈值
const cv::Scalar kEnemyBlue(255, 120, 120);

struct Scene
{
    cv::Mat img;
    std::vector<cv::Point2f> armor_points;  // 空表示画面中没有装甲板
};

// 按 px_per_mm 生成一张“点亮长度 = percent × 186 mm”的血条图
Scene make_scene(double percent, double px_per_mm, bool with_armor)
{
    cv::Mat img(1200, 1600, CV_8UC3, cv::Scalar(0, 0, 0));

    const int bar_h = std::max(1, static_cast<int>(std::lround(22.0 * px_per_mm)));
    const int bar_w = std::max(1, static_cast<int>(std::lround(186.0 * percent * px_per_mm)));
    const cv::Point bar_org(300, 300);
    cv::rectangle(
        img, cv::Rect(bar_org.x, bar_org.y, bar_w, bar_h), kEnemyBlue, cv::FILLED);

    Scene scene;
    scene.img = img;

    if (with_armor) {
        const double armor_w = 141.0 * px_per_mm;
        const double armor_h = 125.0 * px_per_mm;
        const cv::Point2f tl(
            static_cast<float>(bar_org.x), static_cast<float>(bar_org.y + bar_h + 20));
        scene.armor_points = {
            tl,
            cv::Point2f(static_cast<float>(tl.x + armor_w), tl.y),
            cv::Point2f(static_cast<float>(tl.x + armor_w), static_cast<float>(tl.y + armor_h)),
            cv::Point2f(tl.x, static_cast<float>(tl.y + armor_h))};
    }

    return scene;
}

void test_armor_reference()
{
    BloodBarParams params;
    params.enabled = true;
    BloodBarDetector detector(params);

    // 2 m 量级：1.2 px/mm -> 血条 223x26 px，装甲板 169x150 px
    const double px_per_mm = 1.2;
    for (double truth : {1.00, 0.80, 0.50, 0.30, 0.10}) {
        const auto scene = make_scene(truth, px_per_mm, true);
        const auto bars = detector.detect(scene.img);
        check(!bars.empty(), "armor-ref: detect @ " + std::to_string(truth));
        if (bars.empty()) continue;

        const auto percent = estimate_percent(bars.front(), scene.armor_points, params);
        check(percent.has_value(), "armor-ref: estimate @ " + std::to_string(truth));
        if (percent) check_near(*percent, truth, 0.04, "armor-ref: value @ " + std::to_string(truth));
    }
}

void test_far_range()
{
    BloodBarParams params;
    params.enabled = true;
    BloodBarDetector detector(params);

    // 10 m 量级：0.24 px/mm -> 血条 45x5 px，装甲板 34x30 px
    const auto scene = make_scene(0.50, 0.24, true);
    const auto bars = detector.detect(scene.img);
    check(!bars.empty(), "far: detect");
    if (bars.empty()) return;

    const auto percent = estimate_percent(bars.front(), scene.armor_points, params);
    check(percent.has_value(), "far: estimate");
    if (percent) check_near(*percent, 0.50, 0.06, "far: value");
}

void test_self_normalization()
{
    BloodBarParams params;
    params.enabled = true;
    BloodBarDetector detector(params);

    // 近距：条高 26 px >= 门槛 12 px，允许自归一化
    {
        const auto scene = make_scene(0.50, 1.2, false);
        const auto bars = detector.detect(scene.img);
        check(!bars.empty(), "self-norm: detect");
        if (!bars.empty()) {
            const auto percent = estimate_percent(bars.front(), {}, params);
            check(percent.has_value(), "self-norm: estimate");
            if (percent) check_near(*percent, 0.50, 0.03, "self-norm: value");
        }
    }

    // 远距：条高 5 px < 门槛，必须拒绝（宁可不报，也不猜）
    {
        const auto scene = make_scene(0.50, 0.24, false);
        const auto bars = detector.detect(scene.img);
        if (!bars.empty()) {
            const auto percent = estimate_percent(bars.front(), {}, params);
            check(!percent.has_value(), "self-norm: reject when bar too small");
        } else {
            check(true, "self-norm: bar too small to detect at all");
        }
    }
}

void test_classification()
{
    BloodBarParams params;  // low_upper=0.30 medium_upper=0.60 hysteresis=0.05

    check(classify_hp(0.20, HpLevel::unknown, params) == HpLevel::low, "classify: 20% -> low");
    check(classify_hp(0.45, HpLevel::unknown, params) == HpLevel::medium, "classify: 45% -> medium");
    check(classify_hp(0.80, HpLevel::unknown, params) == HpLevel::high, "classify: 80% -> high");
    check(classify_hp(0.30, HpLevel::unknown, params) == HpLevel::low, "classify: 30% -> low");

    // 迟滞：已经在 high 时，掉到 0.58 仍是 high，掉到 0.54 才降为 medium
    check(classify_hp(0.58, HpLevel::high, params) == HpLevel::high, "hysteresis: 58% stay high");
    check(classify_hp(0.54, HpLevel::high, params) == HpLevel::medium, "hysteresis: 54% -> medium");
    // 迟滞：已经在 low 时，升到 0.33 仍是 low，升到 0.36 才升为 medium
    check(classify_hp(0.33, HpLevel::low, params) == HpLevel::low, "hysteresis: 33% stay low");
    check(classify_hp(0.36, HpLevel::low, params) == HpLevel::medium, "hysteresis: 36% -> medium");
}

void test_stabilizer()
{
    BloodBarParams params;
    params.median_frames = 1;  // 只验证迟滞与丢检语义，排除中值干扰
    HpStabilizer stabilizer(params);

    check(stabilizer.update(0.90).level == HpLevel::high, "stabilizer: 90% -> high");
    check(stabilizer.update(0.58).level == HpLevel::high, "stabilizer: 58% stays high");
    check(stabilizer.update(0.54).level == HpLevel::medium, "stabilizer: 54% -> medium");
    check(stabilizer.update(0.20).level == HpLevel::low, "stabilizer: 20% -> low");

    // 规格 C6：丢失检测立即 unknown
    const auto lost = stabilizer.update(std::nullopt);
    check(lost.level == HpLevel::unknown && !lost.valid, "stabilizer: lost -> unknown");

    // 恢复后从 unknown 重新按阈值分档
    check(stabilizer.update(0.75).level == HpLevel::high, "stabilizer: recover -> high");
}

void test_median_filter()
{
    BloodBarParams params;
    params.median_frames = 5;
    HpStabilizer stabilizer(params);

    // 单帧跳变不应改变档位
    stabilizer.update(0.55);
    stabilizer.update(0.55);
    stabilizer.update(0.55);
    const auto spike = stabilizer.update(0.95);
    check(spike.level == HpLevel::medium, "median: single spike ignored");
}

// 手工构造一个水平血条（不走检测器），便于精确控制几何
BloodBar make_bar(double lit_px, double height_px, const cv::Point2f & org)
{
    BloodBar bar;
    bar.points = {
        org,
        cv::Point2f(static_cast<float>(org.x + lit_px), org.y),
        cv::Point2f(static_cast<float>(org.x + lit_px), static_cast<float>(org.y + height_px)),
        cv::Point2f(org.x, static_cast<float>(org.y + height_px))};
    bar.center = cv::Point2f(
        static_cast<float>(org.x + lit_px / 2.0), static_cast<float>(org.y + height_px / 2.0));
    bar.edge_len[0] = lit_px;
    bar.edge_len[1] = height_px;
    bar.edge_dir[0] = {1.f, 0.f};
    bar.edge_dir[1] = {0.f, 1.f};
    return bar;
}

void test_coplanar_height_check()
{
    BloodBarParams params;  // armor_width_mm=141, armor_height_mm=125, height_mm=22

    // 60° 偏转：板宽与血条长度同步压缩一半；板高与条高不受影响
    const double scale = 1.2;  // 正对时的 px/mm
    const double cos_yaw = 0.5;
    const double lit_px = 186.0 * 0.5 * scale * cos_yaw;  // 50% 血量
    const double bar_h_px = 22.0 * scale;
    const double armor_w_px = 141.0 * scale * cos_yaw;
    const double armor_h_px = 125.0 * scale;

    const auto bar = make_bar(lit_px, bar_h_px, {100.f, 100.f});
    const cv::Point2f tl(100.f, static_cast<float>(100.f + bar_h_px + 10.f));
    const std::vector<cv::Point2f> armor = {
        tl,
        cv::Point2f(static_cast<float>(tl.x + armor_w_px), tl.y),
        cv::Point2f(static_cast<float>(tl.x + armor_w_px), static_cast<float>(tl.y + armor_h_px)),
        cv::Point2f(tl.x, static_cast<float>(tl.y + armor_h_px))};

    const auto fit = fit_bar_to_armor(bar, armor, params);
    check(fit.has_value(), "coplanar: fit under 60deg yaw");
    if (fit) {
        check_near(fit->percent, 0.50, 0.02, "coplanar: percent unaffected by yaw");
        // 板高不随偏转压缩，用它做同面校验时得分应保持高位
        check(fit->score > 0.60, "coplanar: score stays high under yaw");
    }
}

void test_impossible_percent_is_invalid()
{
    BloodBarParams params;

    // 板宽 169.2 px -> 1.2 px/mm -> 满条 223.2 px；点亮段画成 290 px 显然不成立
    const double scale = 1.2;
    const double armor_w_px = 141.0 * scale;
    const double armor_h_px = 125.0 * scale;

    const auto bar = make_bar(290.0, 22.0 * scale, {100.f, 100.f});
    const cv::Point2f tl(100.f, static_cast<float>(100.f + 22.0 * scale + 10.0));
    const std::vector<cv::Point2f> armor = {
        tl,
        cv::Point2f(static_cast<float>(tl.x + armor_w_px), tl.y),
        cv::Point2f(static_cast<float>(tl.x + armor_w_px), static_cast<float>(tl.y + armor_h_px)),
        cv::Point2f(tl.x, static_cast<float>(tl.y + armor_h_px))};

    const auto fit = fit_bar_to_armor(bar, armor, params);
    check(!fit.has_value(), "impossible: over 100% rejected, not clamped");

    const auto percent = estimate_percent(bar, armor, params);
    check(!percent.has_value(), "impossible: estimate returns no value");
}
}  // namespace

int main(int argc, char * argv[])
{
    // 冒烟模式：按配置创建上报器并连续发几个档位，配合 ros2 topic echo 验证
    //   ./blood_bar_test --ros2-check configs/new_armor.yaml
    if (argc > 1 && std::string(argv[1]) == "--ros2-check") {
        const std::string config_path = (argc > 2) ? argv[2] : "configs/new_armor.yaml";
#ifdef QTL_WITH_ROS2
        auto reporter = make_hp_reporter(config_path);
        for (int i = 0; i < 20; ++i) {
            reporter->report({i % 4, (i % 4) / 3.0, true});
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        std::cout << "published hp levels 0..3 on the configured topic\n";
#else
        (void)config_path;
        std::cout << "ROS2 not enabled in this build; reconfigure with -DQTL_ENABLE_ROS2=ON\n";
#endif
        return 0;
    }

    test_armor_reference();
    test_far_range();
    test_self_normalization();
    test_classification();
    test_stabilizer();
    test_median_filter();
    test_coplanar_height_check();
    test_impossible_percent_is_invalid();

    if (g_failures == 0) {
        std::cout << "\nall blood_bar tests passed\n";
        return 0;
    }
    std::cout << "\n" << g_failures << " test(s) failed\n";
    return 1;
}
