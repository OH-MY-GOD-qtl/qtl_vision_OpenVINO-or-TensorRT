#include "omni/decider.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "omni/usb_camera_config.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

Decider::Decider(const std::string &config_path)
{
    auto yaml = yaml_load(config_path);

    enemy_color_ =
        (yaml_read<std::string>(yaml, "enemy_color") == "red") ? Color::red : Color::blue;
    mode_ = yaml["mode"] ? yaml["mode"].as<int>() : MODE_ONE;

    auto opt = [&yaml](const char *key, double fallback)
    {
        return yaml[key] ? yaml[key].as<double>() : fallback;
    };

    hik_yaw_offset_ = opt("omni_hik_yaw_offset", 0.0);

    // 新配置角度相对海康；旧配置仍按云台零位解释，保持兼容。
    for (const auto &camera : load_usb_camera_configs(yaml))
    {
        usb_yaw_offsets_.push_back(
            camera.yaw_offset_deg + (camera.yaw_offset_relative_to_hik ? hik_yaw_offset_ : 0.0));
    }
    usb_fov_h_ =
        yaml["omni_usb_fov_h"] ? yaml["omni_usb_fov_h"].as<double>() : opt("new_fov_h", 37.0);
    usb_fov_v_ =
        yaml["omni_usb_fov_v"] ? yaml["omni_usb_fov_v"].as<double>() : opt("new_fov_v", 40.9);
    hik_fov_h_ = opt("omni_hik_fov_h", 54.2);
    hik_fov_v_ = opt("omni_hik_fov_v", 44.5);
    pitch_preset_ = opt("omni_pitch_preset", 0.0) / 57.3;
    usb_max_age_s_ = opt("omni_usb_max_age_s", 0.15);
    usb_yaw_rate_limit_ = opt("omni_usb_yaw_rate_limit", 3.0);
    usb_lpf_alpha_ = opt("omni_usb_lpf_alpha", 0.4);
    usb_lpf_alpha_ = std::max(0.0, std::min(1.0, usb_lpf_alpha_));
    guide_coast_s_ = std::max(0.0, opt("omni_guide_coast_s", 1.5));
    guide_cooldown_s_ = std::max(0.0, opt("omni_guide_cooldown_s", 0.5));

    logger()->info(
        "[Decider] omni: usb_yaw_offsets={} usb_fov={:.1f}x{:.1f} hik_fov={:.1f}x{:.1f} "
        "pitch_preset={:.1f}deg max_age={:.3f}s rate_limit={:.1f}rad/s",
        usb_yaw_offsets_.size(), usb_fov_h_, usb_fov_v_, hik_fov_h_, hik_fov_v_,
        pitch_preset_ * 57.3, usb_max_age_s_, usb_yaw_rate_limit_);
}

void Decider::reset_guide()
{
    has_guide_lock_ = false;
    guide_camera_.clear();
    last_valid_t_ = {};
    cooldown_until_ = {};
    has_last_yaw_ = false;
    last_yaw_ = 0;
    last_yaw_t_ = {};
    last_results_.clear();
}

Decider::Guide Decider::arbitrate_usb(
    const std::vector<DetectionResult> &results, Gimbal &gimbal,
    std::chrono::steady_clock::time_point now, bool hik_active)
{
    // 海康已接管：清空锁定/coast，下一次海康丢失时重新选最近目标
    if (hik_active)
    {
        reset_guide();
        return Guide{};
    }

    // 先把本批结果并入每路缓存（只保留更新的），再按年龄筛新鲜度。
    // 检测被限频到 15 fps，主循环快得多；只看本批会把“这轮没新帧”误判成“丢目标”。
    for (const auto &dr : results)
    {
        if (dr.source != DetectionSource::USB)
            continue;
        if (dr.camera.empty() || dr.armors.empty())
            continue;
        if (dr.selected_index < 0 ||
            dr.selected_index >= static_cast<int>(dr.armors.size()))
            continue;

        auto &slot = last_results_[dr.camera];
        if (slot.timestamp <= dr.timestamp)
            slot = dr;
    }

    std::unordered_map<std::string, const DetectionResult *> latest;
    for (const auto &[camera, dr] : last_results_)
    {
        double age = delta_time(now, dr.timestamp);
        if (age < 0)
            age = 0;
        if (age > usb_max_age_s_)
            continue;
        latest[camera] = &dr;
    }

    if (!has_guide_lock_)
    {
        if (now < cooldown_until_)
            return Guide{};

        // 首次/重新锁定：表观尺寸最大者优先；同分时更新鲜、更居中者优先
        const DetectionResult *best = nullptr;
        for (const auto &[camera, dr] : latest)
        {
            (void)camera;
            bool better = !best || dr->size_score > best->size_score + 1e-9 ||
                          (std::abs(dr->size_score - best->size_score) <= 1e-9 &&
                           std::abs(dr->delta_yaw) < std::abs(best->delta_yaw));
            if (better)
                best = dr;
        }
        if (!best)
            return Guide{};

        has_guide_lock_ = true;
        guide_camera_ = best->camera;
    }

    auto it = latest.find(guide_camera_);
    if (it != latest.end())
    {
        const auto &dr = *it->second;
        const double age = std::max(0.0, delta_time(now, dr.timestamp));
        last_valid_t_ = dr.timestamp;

        // 时间对齐：用“采集时刻”的云台姿态，而非当前姿态
        const double yaw_at_capture = eulers(gimbal.q(dr.timestamp), 2, 1, 0)[0];

        // 延迟补偿：按云台当前 yaw 角速度外推 age
        double yaw_cmd =
            limit_rad(yaw_at_capture + dr.delta_yaw + gimbal.state().yaw_vel * age);

        if (has_last_yaw_)
        {
            // 一阶低通
            yaw_cmd = limit_rad(last_yaw_ + usb_lpf_alpha_ * limit_rad(yaw_cmd - last_yaw_));

            // 限速（防切换相机时跳变）
            double dt = delta_time(now, last_yaw_t_);
            if (dt > 1e-4)
            {
                double max_step = usb_yaw_rate_limit_ * dt;
                double step = limit_rad(yaw_cmd - last_yaw_);
                if (step > max_step)
                    yaw_cmd = limit_rad(last_yaw_ + max_step);
                else if (step < -max_step)
                    yaw_cmd = limit_rad(last_yaw_ - max_step);
            }
        }

        has_last_yaw_ = true;
        last_yaw_ = yaw_cmd;
        last_yaw_t_ = now;

        Guide guide;
        guide.phase = GuidePhase::LOCKED;
        guide.valid = true;
        guide.yaw = yaw_cmd;
        guide.pitch = pitch_preset_;
        guide.camera = guide_camera_;
        guide.age = age;
        return guide;
    }

    // 锁定相机没有新鲜结果：coast，保持最后绝对方位继续转动
    if (!has_last_yaw_)
    {
        reset_guide();
        return Guide{};
    }

    const double lost_s = std::max(0.0, delta_time(now, last_valid_t_));
    if (lost_s > guide_coast_s_)
    {
        reset_guide();
        cooldown_until_ =
            now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                      std::chrono::duration<double>(guide_cooldown_s_));
        return Guide{};
    }

    Guide guide;
    guide.phase = GuidePhase::COAST;
    guide.valid = true;
    guide.yaw = last_yaw_;
    guide.pitch = pitch_preset_;
    guide.camera = guide_camera_;
    guide.age = lost_s;
    return guide;
}

Eigen::Vector2d Decider::delta_angle(const Armor &armor, const std::string &camera) const
{
    return delta_angle_from_center(armor.center_norm, camera);
}

Eigen::Vector2d Decider::delta_angle(
    const std::vector<Armor> &armors, const std::string &camera) const
{
    if (armors.empty())
        return Eigen::Vector2d::Zero();
    return delta_angle(armors.front(), camera);
}

Eigen::Vector2d Decider::delta_angle_from_center(
    const cv::Point2f &center, const std::string &camera) const
{
    // 相机安装 yaw 偏移 + 各自 FOV（degree）
    double yaw_offset = 0.0;
    double fov_h = usb_fov_h_;
    double fov_v = usb_fov_v_;

    if (camera == "hik")
    {
        yaw_offset = hik_yaw_offset_;
        fov_h = hik_fov_h_;
        fov_v = hik_fov_v_;
    }
    else if (camera.rfind("usb", 0) == 0 && camera.size() > 3)
    {
        int idx = std::stoi(camera.substr(3));
        if (idx >= 0 && idx < static_cast<int>(usb_yaw_offsets_.size()))
            yaw_offset = usb_yaw_offsets_[idx];
    }

    Eigen::Vector2d delta_angle = Eigen::Vector2d::Zero();
    delta_angle[0] = yaw_offset + (fov_h / 2.0) - center.x * fov_h;
    delta_angle[1] = center.y * fov_v - fov_v / 2.0;
    return delta_angle;
}

void Decider::filter_enemy_color(std::vector<Armor> &armors) const
{
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor &a)
            { return a.color != enemy_color_; }),
        armors.end());
}

bool Decider::armor_filter(std::vector<Armor> &armors) const
{
    if (armors.empty())
        return true;
    // 过滤非敌方装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor &a)
            { return a.color != enemy_color_; }),
        armors.end());

    // 25赛季没有5号装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [&](const Armor &a)
            { return a.name == ArmorName::five; }),
        armors.end());
    // 不打工程
    // armors.erase(std::remove_if(armors.begin(), armors.end(),
    //   [&](const Armor & a) { return a.name == ArmorName::two; }), armors.end());
    // 不打前哨站
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [&](const Armor &a)
            { return a.name == ArmorName::outpost; }),
        armors.end());

    // 过滤掉刚复活无敌的装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [&](const Armor &a)
            {
                return std::find(invincible_armor_.begin(), invincible_armor_.end(), a.name) !=
                       invincible_armor_.end();
            }),
        armors.end());

    return armors.empty();
}

void Decider::set_priority(std::vector<Armor> &armors) const
{
    if (armors.empty())
        return;

    const PriorityMap &priority_map = (mode_ == MODE_ONE) ? mode1 : mode2;

    if (!armors.empty())
    {
        for (auto &armor : armors)
        {
            armor.priority = priority_map.at(armor.name);
        }
    }
}

void Decider::sort(std::vector<DetectionResult> &detection_queue) const
{
    if (detection_queue.empty())
        return;

    // 对每个 DetectionResult 调用 armor_filter 和 set_priority
    for (auto &dr : detection_queue)
    {
        armor_filter(dr.armors);
        set_priority(dr.armors);

        // 对每个 DetectionResult 中的 armors 进行排序
        std::sort(dr.armors.begin(), dr.armors.end(), [](const Armor &a, const Armor &b)
                  { return a.priority < b.priority; });
    }

    // 过滤掉过滤后无目标的项，避免下面 front() 越界
    detection_queue.erase(
        std::remove_if(
            detection_queue.begin(), detection_queue.end(),
            [](const DetectionResult &dr)
            { return dr.armors.empty(); }),
        detection_queue.end());

    // 根据优先级对 DetectionResult 进行排序
    std::sort(
        detection_queue.begin(), detection_queue.end(),
        [](const DetectionResult &a, const DetectionResult &b)
        {
            return a.armors.front().priority < b.armors.front().priority;
        });
}

Eigen::Vector4d Decider::get_target_info(
    const std::vector<Armor> &armors, const std::vector<Target> &targets) const
{
    if (armors.empty() || targets.empty())
        return Eigen::Vector4d::Zero();

    auto target = targets.front();

    for (const auto &armor : armors)
    {
        if (armor.name == target.name)
        {
            return Eigen::Vector4d{
                armor.xyz_in_gimbal[0], armor.xyz_in_gimbal[1], 1,
                static_cast<double>(armor.name) + 1}; // 避免歧义+1(详见通信协议)
        }
    }

    return Eigen::Vector4d::Zero();
}

void Decider::get_invincible_armor(const std::vector<int8_t> &invincible_enemy_ids)
{
    invincible_armor_.clear();

    if (invincible_enemy_ids.empty())
        return;

    for (const auto &id : invincible_enemy_ids)
    {
        logger()->info("invincible armor id: {}", id);
        invincible_armor_.push_back(ArmorName(id - 1));
    }
}

void Decider::get_auto_aim_target(
    std::vector<Armor> &armors, const std::vector<int8_t> &auto_aim_target) const
{
    if (auto_aim_target.empty())
        return;

    std::vector<ArmorName> auto_aim_targets;

    for (const auto &target : auto_aim_target)
    {
        if (target <= 0 || static_cast<size_t>(target) > ARMOR_NAMES.size())
        {
            logger()->warn("Received invalid auto_aim target value: {}", int(target));
            continue;
        }
        auto_aim_targets.push_back(static_cast<ArmorName>(target - 1));
        logger()->info("nav send auto_aim target is {}", ARMOR_NAMES[target - 1]);
    }

    if (auto_aim_targets.empty())
        return;

    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(),
            [&](const Armor &a)
            {
                return std::find(auto_aim_targets.begin(), auto_aim_targets.end(), a.name) ==
                       auto_aim_targets.end();
            }),
        armors.end());
}
