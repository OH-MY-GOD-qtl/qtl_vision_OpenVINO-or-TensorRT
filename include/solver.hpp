#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

#include <Eigen/Dense> // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include <optional>

#include "armor.hpp"

class Solver
{
public:
    explicit Solver(const std::string &config_path);

    Eigen::Matrix3d R_gimbal2world() const;

    void set_R_gimbal2world(const Eigen::Quaterniond &q);

    // yaw_hint：滤波器预测的装甲板 yaw。给出时用它作为 PnP yaw 优化的搜索中心，
    // 不给出时退回云台 yaw（新建目标、无预测可用时）。
    void solve(Armor &armor, std::optional<double> yaw_hint = std::nullopt) const;

    std::vector<cv::Point2f> reproject_armor(
        const Eigen::Vector3d &xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

    double oupost_reprojection_error(Armor armor, const double &picth);

    std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> &worldPoints);

private:
    cv::Mat camera_matrix_;
    cv::Mat distort_coeffs_;
    Eigen::Matrix3d R_gimbal2imubody_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    Eigen::Matrix3d R_gimbal2world_;

    void optimize_yaw(Armor &armor, std::optional<double> yaw_hint) const;

    double armor_reprojection_error(const Armor &armor, double yaw, const double &inclined) const;
    double SJTU_cost(
        const std::vector<cv::Point2f> &cv_refs, const std::vector<cv::Point2f> &cv_pts,
        const double &inclined) const;
};

#endif // AUTO_AIM__SOLVER_HPP