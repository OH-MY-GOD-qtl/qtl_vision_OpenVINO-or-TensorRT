#include "omni/decider.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "comm/gimbal.hpp"

using namespace std::chrono_literals;

namespace
{
void require(bool condition, const std::string & message)
{
    if (!condition) throw std::runtime_error(message);
}

Armor make_armor(float cx, float cy, float width)
{
    std::vector<cv::Point2f> points = {
        {cx - width / 2, cy - 20}, {cx + width / 2, cy - 20},
        {cx + width / 2, cy + 20}, {cx - width / 2, cy + 20}};
    Armor armor(0, 0.9f, cv::Rect(cv::Point(cx - width / 2, cy - 20), cv::Size(width, 40)), points);
    armor.center_norm = {cx / 640.0f, cy / 360.0f};
    return armor;
}

DetectionResult make_result(
    const std::string & camera, double size_score, std::chrono::steady_clock::time_point t,
    double delta_yaw = 0.0)
{
    DetectionResult result;
    result.armors.push_back(make_armor(320, 180, 60));
    result.timestamp = t;
    result.delta_yaw = delta_yaw;
    result.delta_pitch = 0.0;
    result.source = DetectionSource::USB;
    result.camera = camera;
    result.selected_index = 0;
    result.size_score = size_score;
    return result;
}
}  // namespace

int main()
{
    try {
        const auto config_path =
            (std::filesystem::temp_directory_path() /
             ("omni_arbitration_test_" +
              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
              ".yaml"))
                .string();
        {
            std::ofstream config(config_path);
            config << "enemy_color: blue\n"
                      "omni_usb_fov_h: 37.0\n"
                      "omni_usb_fov_v: 40.9\n"
                      "omni_hik_yaw_offset: 0\n"
                      "omni_hik_fov_h: 54.2\n"
                      "omni_hik_fov_v: 44.5\n"
                      "omni_pitch_preset: 0\n"
                      "omni_usb_max_age_s: 0.15\n"
                      "omni_usb_yaw_rate_limit: 1000.0\n"
                      "omni_usb_lpf_alpha: 1.0\n"
                      "omni_guide_coast_s: 1.5\n"
                      "omni_guide_cooldown_s: 0.5\n"
                      "simulate: true\n";
        }

        Gimbal gimbal(config_path, true);
        Decider decider(config_path);

        auto now = std::chrono::steady_clock::now();

        // 1) 首次锁定：表观尺寸最大的 usb1
        auto guide = decider.arbitrate_usb(
            {make_result("usb0", 0.20, now), make_result("usb1", 0.30, now)}, gimbal, now, false);
        require(guide.valid, "initial lock must be valid");
        require(guide.phase == Decider::GuidePhase::LOCKED, "initial phase must be LOCKED");
        require(guide.camera == "usb1", "initial lock must choose the nearest camera");
        const double locked_yaw = guide.yaw;

        // 2) 锁定期间不切换：即使 usb0 更新且更大
        now += 10ms;
        guide = decider.arbitrate_usb(
            {make_result("usb0", 0.50, now), make_result("usb1", 0.30, now)}, gimbal, now, false);
        require(guide.camera == "usb1", "must not switch camera while locked");

        // 3) 锁定相机超过 max_age：进入 coast，保持最后绝对方位
        now += 200ms;
        guide = decider.arbitrate_usb({make_result("usb0", 0.50, now)}, gimbal, now, false);
        require(guide.valid, "coast must keep controlling");
        require(guide.phase == Decider::GuidePhase::COAST, "stale lock must enter COAST");
        require(guide.camera == "usb1", "coast must keep the locked camera");
        require(std::abs(guide.yaw - locked_yaw) < 1e-9, "coast must keep the last absolute yaw");

        // 4) 后置目标穿盲区：无任何相机结果时继续 coast，直到海康接管
        now += 1000ms;
        guide = decider.arbitrate_usb({}, gimbal, now, false);
        require(guide.phase == Decider::GuidePhase::COAST, "coast must survive the blind arc");
        require(std::abs(guide.yaw - locked_yaw) < 1e-9, "blind-arc coast must not change yaw");

        // 5) coast 超时 -> 停止，并进入冷却
        now += 500ms;
        guide = decider.arbitrate_usb({}, gimbal, now, false);
        require(!guide.valid, "coast timeout must stop control");

        now += 400ms;
        guide = decider.arbitrate_usb({make_result("usb0", 0.40, now)}, gimbal, now, false);
        require(!guide.valid, "cooldown must block immediate re-lock");

        // 6) 冷却结束 -> 重新锁定最近目标
        now += 200ms;
        guide = decider.arbitrate_usb({make_result("usb0", 0.40, now)}, gimbal, now, false);
        require(guide.valid && guide.camera == "usb0", "must re-lock after cooldown");

        // 7) 海康接管 -> 清空锁定；下一次海康丢失重新选最近
        guide = decider.arbitrate_usb({}, gimbal, now, true);
        require(!guide.valid, "hik takeover must clear guide");

        now += 10ms;
        guide = decider.arbitrate_usb(
            {make_result("usb0", 0.20, now), make_result("usb2", 0.35, now)}, gimbal, now, false);
        require(guide.camera == "usb2", "after hik takeover must re-select nearest");

        std::filesystem::remove(config_path);
        std::puts("omni_arbitration_test passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "omni_arbitration_test failed: %s\n", e.what());
        return 1;
    }
}
