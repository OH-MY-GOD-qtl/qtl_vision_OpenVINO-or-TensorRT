#include "tools/extended_kalman_filter.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace
{
    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }
} // namespace

int main()
{
    try
    {
        Eigen::VectorXd x0(4);
        x0.setZero();
        Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4);
        ExtendedKalmanFilter ekf(x0, P0);

        // 3 维量测：residual 只有 3 个元素，不得读取 residual[3]
        Eigen::VectorXd z3(3);
        z3 << 1.0, 2.0, 3.0;
        Eigen::MatrixXd H3 = Eigen::MatrixXd::Zero(3, 4);
        H3(0, 0) = H3(1, 1) = H3(2, 2) = 1.0;
        Eigen::MatrixXd R3 = Eigen::MatrixXd::Identity(3, 3);
        ekf.update(z3, H3, R3);

        require(std::isfinite(ekf.data["residual_distance"]), "residual_distance not finite");
        require(ekf.data["residual_angle"] == 0.0, "3-dim update must not read residual[3]");

        // 4 维量测：residual_angle 正常记录
        Eigen::VectorXd z4(4);
        z4 << 1.0, 2.0, 3.0, 4.0;
        Eigen::MatrixXd H4 = Eigen::MatrixXd::Identity(4, 4);
        Eigen::MatrixXd R4 = Eigen::MatrixXd::Identity(4, 4);
        ekf.update(z4, H4, R4);

        require(
            std::abs(ekf.data["residual_angle"] - 2.0) < 1e-9,
            "4-dim update must record residual[3]");

        std::puts("ekf_test passed");
        return 0;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "ekf_test failed: %s\n", e.what());
        return 1;
    }
}
