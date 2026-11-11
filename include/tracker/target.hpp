#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

// 滤波噪声标定系数：P0、Q、R 必须同比例缩放。
// 三者同比例缩放时 K = P Hᵀ (H P Hᵀ + R)⁻¹ 在每一步都不变，
// 滤波输出完全一致，只把协方差拉到与实际创新同一量级，使 NIS 恢复卡方意义。
constexpr double EKF_NOISE_SCALE = 1e-2;

class Target
{
public:
    ArmorName name;
    ArmorType armor_type;
    ArmorPriority priority;
    bool jumped;
    int last_id; // debug only

    Target() = default;
    Target(
        const Armor &armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
        Eigen::VectorXd P0_dig);
    Target(double x, double vyaw, double radius, double h);

    void predict(std::chrono::steady_clock::time_point t);
    void predict(double dt);

    // id 由 Tracker 用重投影关联给出（见 Tracker::associate_armor）
    void update(const Armor &armor, int id);

    const Eigen::VectorXd &ekf_x() const;
    const ExtendedKalmanFilter &ekf() const;
    const std::vector<Eigen::Vector4d> &armor_xyza_list() const;

    bool diverged() const;

    bool convergened();

    bool isinit = false;

    bool checkinit();

    // 调试用：该 Target 建立以来收到的量测更新次数
    int update_count() const;

    // 量测中断时作废连续超限证据
    void reset_nis_streak();

    // 先验失准时放大协方差，让下一帧量测重新主导
    void inflate_prior(double factor);

    // 调试用：最近一次重投影关联的归一化误差与决策裕度（次优/最优）
    double reproj_err() const;
    double reproj_margin() const;
    void set_reproj_stats(double err, double margin);

private:
    int armor_num_;
    int switch_count_;
    int update_count_;

    bool is_switch_, is_converged_;

    ExtendedKalmanFilter ekf_;
    std::chrono::steady_clock::time_point t_;

    // 帧间复用缓冲：避免 armor_xyza_list/h_jacobian 每次调用堆分配
    mutable std::vector<Eigen::Vector4d> armor_xyza_cache_;

    // 仅在调试与阈值标定时使用，不影响滤波
    double reproj_err_ = -1.0;
    double reproj_margin_ = -1.0;

    void update_ypda(const Armor &armor, int id); // yaw pitch distance angle

    Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd &x, int id) const;
    Eigen::Matrix<double, 4, 11> h_jacobian(const Eigen::VectorXd &x, int id) const;
};

#endif // AUTO_AIM__TARGET_HPP