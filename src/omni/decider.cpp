#include "omni/decider.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tools/math_tools.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"
#include "omni/usb_camera_config.hpp"


Decider::Decider(const std::string & config_path)
{
    auto yaml = yaml_load(config_path);

    enemy_color_ =
        (yaml_read<std::string>(yaml, "enemy_color") == "red") ? Color::red : Color::blue;
    mode_ = yaml["mode"] ? yaml["mode"].as<int>() : MODE_ONE;

    auto opt = [&yaml](const char * key, double fallback) {
        return yaml[key] ? yaml[key].as<double>() : fallback;
    };

    hik_yaw_offset_ = opt("omni_hik_yaw_offset", 0.0);

    // 新配置角度相对海康；旧配置仍按云台零位解释，保持兼容。
    for (const auto & camera : load_usb_camera_configs(yaml)) {
        usb_yaw_offsets_.push_back(
            camera.yaw_offset_deg + (camera.yaw_offset_relative_to_hik ? hik_yaw_offset_ : 0.0));
    }
    usb_fov_h_ =
        yaml["omni_usb_fov_h"] ? yaml["omni_usb_fov_h"].as<double>() : opt("new_fov_h", 37.0);
    usb_fov_v_ = yaml["omni_usb_fov_v"] ? yaml["omni_usb_fov_v"].as<double>() : opt("new_fov_v", 40.9);
    hik_fov_h_ = opt("omni_hik_fov_h", 54.2);
    hik_fov_v_ = opt("omni_hik_fov_v", 44.5);
    pitch_preset_ = opt("omni_pitch_preset", 0.0) / 57.3;
    usb_max_age_s_ = opt("omni_usb_max_age_s", 0.15);
    usb_yaw_rate_limit_ = opt("omni_usb_yaw_rate_limit", 3.0);
    usb_lpf_alpha_ = opt("omni_usb_lpf_alpha", 0.4);
    usb_lpf_alpha_ = std::max(0.0, std::min(1.0, usb_lpf_alpha_));

    logger()->info(
        "[Decider] omni: usb_yaw_offsets={} usb_fov={:.1f}x{:.1f} hik_fov={:.1f}x{:.1f} "
        "pitch_preset={:.1f}deg max_age={:.3f}s rate_limit={:.1f}rad/s",
        usb_yaw_offsets_.size(), usb_fov_h_, usb_fov_v_, hik_fov_h_, hik_fov_v_,
        pitch_preset_ * 57.3, usb_max_age_s_, usb_yaw_rate_limit_);
}

Decider::Guide Decider::arbitrate_usb(
    const std::vector<DetectionResult> & results, Gimbal & gimbal,
    std::chrono::steady_clock::time_point now)
{
    const DetectionResult * best = nullptr;
    double best_age = 0;

    for (const auto & dr : results) {
        if (dr.source != DetectionSource::USB) continue;
        if (dr.armors.empty()) continue;

        double age = delta_time(now, dr.timestamp);
        if (age < 0) age = 0;
        if (age > usb_max_age_s_) continue;  // 老化结果丢弃

        // 选“最新”的；年龄接近时选更居中的（USB 无编号，无法按优先级排）
        bool better = !best || age < best_age - 1e-3 ||
                      (std::abs(age - best_age) <= 1e-3 &&
                       std::abs(dr.delta_yaw) < std::abs(best->delta_yaw));
        if (better) {
            best = &dr;
            best_age = age;
        }
    }

    Guide guide;
    if (!best) {
        has_last_yaw_ = false;  // 丢失目标后重新捕获时不做限速/低通，快速指向
        return guide;
    }

    // 时间对齐：用“采集时刻”的云台姿态，而非当前姿态
    double yaw_at_capture = eulers(gimbal.q(best->timestamp), 2, 1, 0)[0];

    // 延迟补偿：按云台当前 yaw 角速度外推 age
    double yaw_cmd = limit_rad(yaw_at_capture + best->delta_yaw + gimbal.state().yaw_vel * best_age);

    if (has_last_yaw_) {
        // 一阶低通
        yaw_cmd = limit_rad(last_yaw_ + usb_lpf_alpha_ * limit_rad(yaw_cmd - last_yaw_));

        // 限速（防切换相机时跳变）
        double dt = delta_time(now, last_yaw_t_);
        if (dt > 1e-4) {
            double max_step = usb_yaw_rate_limit_ * dt;
            double step = limit_rad(yaw_cmd - last_yaw_);
            if (step > max_step) yaw_cmd = limit_rad(last_yaw_ + max_step);
            else if (step < -max_step) yaw_cmd = limit_rad(last_yaw_ - max_step);
        }
    }

    has_last_yaw_ = true;
    last_yaw_ = yaw_cmd;
    last_yaw_t_ = now;

    guide.valid = true;
    guide.yaw = yaw_cmd;
    guide.pitch = pitch_preset_;
    guide.camera = best->camera;
    guide.age = best_age;
    return guide;
}

Eigen::Vector2d Decider::delta_angle(
    const std::vector<Armor> & armors, const std::string & camera) const
{
    Eigen::Vector2d delta_angle = Eigen::Vector2d::Zero();
    if (armors.empty()) return delta_angle;

    // 相机安装 yaw 偏移 + 各自 FOV（degree）
    double yaw_offset = 0.0;
    double fov_h = usb_fov_h_;
    double fov_v = usb_fov_v_;

    if (camera == "hik") {
        yaw_offset = hik_yaw_offset_;
        fov_h = hik_fov_h_;
        fov_v = hik_fov_v_;
    } else if (camera.rfind("usb", 0) == 0 && camera.size() > 3) {
        int idx = std::stoi(camera.substr(3));
        if (idx >= 0 && idx < static_cast<int>(usb_yaw_offsets_.size()))
            yaw_offset = usb_yaw_offsets_[idx];
    }

    const auto & center = armors.front().center_norm;
    delta_angle[0] = yaw_offset + (fov_h / 2.0) - center.x * fov_h;
    delta_angle[1] = center.y * fov_v - fov_v / 2.0;
    return delta_angle;
}

bool Decider::armor_filter(std::vector<Armor> & armors) const
{
    if (armors.empty()) return true;
    // 过滤非敌方装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor & a) { return a.color != enemy_color_; }),
        armors.end());

    // 25赛季没有5号装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor & a) { return a.name == ArmorName::five; }),
        armors.end());
    // 不打工程
    // armors.erase(std::remove_if(armors.begin(), armors.end(),
    //   [&](const Armor & a) { return a.name == ArmorName::two; }), armors.end());
    // 不打前哨站
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor & a) { return a.name == ArmorName::outpost; }),
        armors.end());

    // 过滤掉刚复活无敌的装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [&](const Armor & a) {
                return std::find(invincible_armor_.begin(), invincible_armor_.end(), a.name) !=
                                invincible_armor_.end();
            }),
        armors.end());

    return armors.empty();
}

void Decider::set_priority(std::vector<Armor> & armors) const
{
    if (armors.empty()) return;

    const PriorityMap & priority_map = (mode_ == MODE_ONE) ? mode1 : mode2;

    if (!armors.empty()) {
        for (auto & armor : armors) {
            armor.priority = priority_map.at(armor.name);
        }
    }
}

void Decider::sort(std::vector<DetectionResult> & detection_queue) const
{
    if (detection_queue.empty()) return;

    // 对每个 DetectionResult 调用 armor_filter 和 set_priority
    for (auto & dr : detection_queue) {
        armor_filter(dr.armors);
        set_priority(dr.armors);

        // 对每个 DetectionResult 中的 armors 进行排序
        std::sort(
            dr.armors.begin(), dr.armors.end(),
            [](const Armor & a, const Armor & b) { return a.priority < b.priority; });
    }

    // 过滤掉过滤后无目标的项，避免下面 front() 越界
    detection_queue.erase(
        std::remove_if(
            detection_queue.begin(), detection_queue.end(),
            [](const DetectionResult & dr) { return dr.armors.empty(); }),
        detection_queue.end());

    // 根据优先级对 DetectionResult 进行排序
    std::sort(
        detection_queue.begin(), detection_queue.end(),
        [](const DetectionResult & a, const DetectionResult & b) {
            return a.armors.front().priority < b.armors.front().priority;
        });
}

Eigen::Vector4d Decider::get_target_info(
    const std::vector<Armor> & armors, const std::vector<Target> & targets) const
{
    if (armors.empty() || targets.empty()) return Eigen::Vector4d::Zero();

    auto target = targets.front();

    for (const auto & armor : armors) {
        if (armor.name == target.name) {
            return Eigen::Vector4d{
                armor.xyz_in_gimbal[0], armor.xyz_in_gimbal[1], 1,
                static_cast<double>(armor.name) + 1};  //避免歧义+1(详见通信协议)
        }
    }

    return Eigen::Vector4d::Zero();
}

void Decider::get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids)
{
    invincible_armor_.clear();

    if (invincible_enemy_ids.empty()) return;

    for (const auto & id : invincible_enemy_ids) {
        logger()->info("invincible armor id: {}", id);
        invincible_armor_.push_back(ArmorName(id - 1));
    }
}

void Decider::get_auto_aim_target(
    std::vector<Armor> & armors, const std::vector<int8_t> & auto_aim_target) const
{
    if (auto_aim_target.empty()) return;

    std::vector<ArmorName> auto_aim_targets;

    for (const auto & target : auto_aim_target) {
        if (target <= 0 || static_cast<size_t>(target) > ARMOR_NAMES.size()) {
            logger()->warn("Received invalid auto_aim target value: {}", int(target));
            continue;
        }
        auto_aim_targets.push_back(static_cast<ArmorName>(target - 1));
        logger()->info("nav send auto_aim target is {}", ARMOR_NAMES[target - 1]);
    }

    if (auto_aim_targets.empty()) return;

    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [&](const Armor & a) {
                return std::find(auto_aim_targets.begin(), auto_aim_targets.end(), a.name) ==
                                auto_aim_targets.end();
            }),
        armors.end());
}
