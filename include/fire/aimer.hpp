#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <vector>

#include "comm/gimbal.hpp"
#include "comm/cboard.hpp"
#include "comm/command.hpp"
#include "tracker/target.hpp"



struct AimPoint
{
    bool valid;
    Eigen::Vector4d xyza;
};

class Aimer
{
public:
    AimPoint debug_aim_point;
    explicit Aimer(const std::string & config_path);
    Command aim(
        const std::vector<Target> & targets, std::chrono::steady_clock::time_point timestamp,
        double bullet_speed, bool to_now = true);

    Command aim(
        const std::vector<Target> & targets, std::chrono::steady_clock::time_point timestamp,
        double bullet_speed, ShootMode shoot_mode, bool to_now = true);

private:
    double yaw_offset_;
    std::optional<double> left_yaw_offset_, right_yaw_offset_;
    double pitch_offset_;
    double comming_angle_;
    double leaving_angle_;
    double lock_id_ = -1;
    double high_speed_delay_time_;
    double low_speed_delay_time_;
    double decision_speed_;

    // 输出平滑：一阶低通 + 死区
    double lowpass_factor_ = 0.2;  // 0~1，越小越平滑，1 等价于不滤波
    double deadband_ = 0.1;        // 死区（degree），变化量小于该值不更新输出
    double last_yaw_ = 0.0;
    double last_pitch_ = 0.0;
    bool has_last_ = false;

    AimPoint choose_aim_point(const Target & target);

    Command smooth(const Command & command);
};


#endif  // AUTO_AIM__AIMER_HPP