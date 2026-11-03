#ifndef OMNIPERCEPTION__DECIDER_HPP
#define OMNIPERCEPTION__DECIDER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "armor.hpp"
#include "comm/command.hpp"
#include "comm/gimbal.hpp"
#include "omni/detection.hpp"
#include "tracker/target.hpp"

class Decider
{
public:
    // USB 方位引导阶段：锁定一路相机 -> 丢目标后保持最后绝对方位（coast）
    enum class GuidePhase
    {
        NONE,
        LOCKED,
        COAST
    };

    // USB 方位引导结果：海康无目标时用于把云台 yaw 转向目标方位
    struct Guide
    {
        GuidePhase phase = GuidePhase::NONE;
        bool valid = false;
        double yaw = 0;      // rad，世界系绝对 yaw 指令（不是增量）
        double pitch = 0;    // rad，固定为预设值
        std::string camera;  // usb0/usb1/usb2
        double age = 0;      // s，从采集到现在的时长；coast 时为距最后有效结果
    };

    explicit Decider(const std::string & config_path);

    // 相对相机光轴的偏角(deg)：yaw 水平、pitch 垂直（含安装角与各自 FOV）
    Eigen::Vector2d delta_angle(const Armor & armor, const std::string & camera) const;
    Eigen::Vector2d delta_angle(
        const std::vector<Armor> & armors, const std::string & camera) const;

    // 海康无目标时的 USB 方位引导（时间对齐版）：
    //  - 每路只取最新一条，过滤老化结果（age > omni_usb_max_age_s）
    //  - 首次锁定表观尺寸最大的相机；锁定期间不切换
    //  - 锁定相机丢目标后 coast：保持最后绝对方位，直到海康接管或超时
    //  - 海康一旦接管（hik_active=true）立即清空引导状态
    Guide arbitrate_usb(
        const std::vector<DetectionResult> & results, Gimbal & gimbal,
        std::chrono::steady_clock::time_point now, bool hik_active);

    // 清除锁定/coast/低通状态（退出自瞄或海康接管时调用）
    void reset_guide();

    // 只过滤敌方颜色，不碰未初始化的 name/type/box
    void filter_enemy_color(std::vector<Armor> & armors) const;

    bool armor_filter(std::vector<Armor> & armors) const;

    void set_priority(std::vector<Armor> & armors) const;
    //对队列中的每一个DetectionResult进行过滤，同时将DetectionResult排序
    void sort(std::vector<DetectionResult> & detection_queue) const;

    Eigen::Vector4d get_target_info(
        const std::vector<Armor> & armors, const std::vector<Target> & targets) const;

    void get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids);

    void get_auto_aim_target(
        std::vector<Armor> & armors, const std::vector<int8_t> & auto_aim_target) const;

private:
    // ---- 全向感知相机安装参数 ----
    std::vector<double> usb_yaw_offsets_;  // 三路USB相对云台零位的安装角(deg)
    double usb_fov_h_, usb_fov_v_;         // USB水平/垂直FOV(deg)
    double hik_yaw_offset_;                // 海康相对yaw零位安装角(deg)
    double hik_fov_h_, hik_fov_v_;         // 海康水平/垂直FOV(deg)
    double pitch_preset_;                  // 无海康目标时的pitch预设(rad)
    double usb_max_age_s_;                 // USB结果老化阈值(s)
    double usb_yaw_rate_limit_;            // USB引导yaw限速(rad/s)
    double usb_lpf_alpha_;                 // USB引导yaw一阶低通系数(0~1)
    double guide_coast_s_;                 // 锁定相机丢失后保持方位的上限(s)
    double guide_cooldown_s_;              // coast 超时后的停止时长(s)

    int mode_;

    Color enemy_color_;
    std::vector<ArmorName> invincible_armor_;  //无敌状态机器人编号,英雄为1，哨兵为6

    // 引导输出状态（锁定/coast + 限速/低通）
    bool has_guide_lock_ = false;
    std::string guide_camera_;
    std::chrono::steady_clock::time_point last_valid_t_;
    std::chrono::steady_clock::time_point cooldown_until_;
    bool has_last_yaw_ = false;
    double last_yaw_ = 0;
    std::chrono::steady_clock::time_point last_yaw_t_;

    Eigen::Vector2d delta_angle_from_center(
        const cv::Point2f & center, const std::string & camera) const;

    // 定义ArmorName到ArmorPriority的映射类型
    using PriorityMap = std::unordered_map<ArmorName, ArmorPriority>;

    const PriorityMap mode1 = {
        {ArmorName::one, ArmorPriority::second},     {ArmorName::two, ArmorPriority::forth},
        {ArmorName::three, ArmorPriority::first},    {ArmorName::four, ArmorPriority::first},
        {ArmorName::five, ArmorPriority::third},     {ArmorName::sentry, ArmorPriority::third},
        {ArmorName::outpost, ArmorPriority::fifth},  {ArmorName::base, ArmorPriority::fifth},
        {ArmorName::not_armor, ArmorPriority::fifth}};

    const PriorityMap mode2 = {
        {ArmorName::two, ArmorPriority::first},      {ArmorName::one, ArmorPriority::second},
        {ArmorName::three, ArmorPriority::second},   {ArmorName::four, ArmorPriority::second},
        {ArmorName::five, ArmorPriority::second},    {ArmorName::sentry, ArmorPriority::third},
        {ArmorName::outpost, ArmorPriority::third},  {ArmorName::base, ArmorPriority::third},
        {ArmorName::not_armor, ArmorPriority::third}};
};

enum PriorityMode
{
    MODE_ONE = 1,
    MODE_TWO
};

#endif