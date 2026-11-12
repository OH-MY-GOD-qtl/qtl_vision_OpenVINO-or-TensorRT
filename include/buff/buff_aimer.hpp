#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <vector>

#include "buff/buff_target.hpp"
#include "buff/buff_type.hpp"
#include "comm/command.hpp"

class BuffAimer
{
public:
    BuffAimer(const std::string &config_path);

    Command aim(
        BuffTarget &target, std::chrono::steady_clock::time_point &timestamp, double bullet_speed,
        bool to_now = true);

    double angle;     ///
    double t_gap = 0; ///

private:
    SmallTarget target_;
    double yaw_offset_;
    double pitch_offset_;

    double fire_gap_time_;
    double predict_time_;

    int mistake_count_ = 0;
    bool switch_fanblade_;

    double last_yaw_ = 0;
    double last_pitch_ = 0;

    std::chrono::steady_clock::time_point last_fire_t_;

    bool get_send_angle(
        BuffTarget &target, const double predict_time, const double bullet_speed, const bool to_now,
        double &yaw, double &pitch);
};
#endif // AUTO_AIM__AIMER_HPP