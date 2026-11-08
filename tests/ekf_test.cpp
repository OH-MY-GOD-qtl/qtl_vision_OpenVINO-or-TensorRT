#include "tools/extended_kalman_filter.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

namespace
{
void require(bool condition, const std::string & message)
{
    if (!condition) throw std::runtime_error(message);
}

void require_near(double actual, double expected, double tol, const std::string & message)
{
    if (!(std::abs(actual - expected) <= tol)) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}
}  // namespace

// 3 维量测：residual 只有 3 个元素，不得读取 residual[3]
void test_residual_indexing()
{
    Eigen::VectorXd x0(4);
    x0.setZero();
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4);
    ExtendedKalmanFilter ekf(x0, P0);

    Eigen::VectorXd z3(3);
    z3 << 1.0, 2.0, 3.0;
    Eigen::MatrixXd H3 = Eigen::MatrixXd::Zero(3, 4);
    H3(0, 0) = H3(1, 1) = H3(2, 2) = 1.0;
    Eigen::MatrixXd R3 = Eigen::MatrixXd::Identity(3, 3);
    ekf.update(z3, H3, R3);

    require(std::isfinite(ekf.data["residual_distance"]), "residual_distance not finite");
    require(ekf.data["residual_angle"] == 0.0, "3-dim update must not read residual[3]");

    // 4 维量测：residual_angle 记录的是先验残差（z - h(x_prior)）
    Eigen::VectorXd z4(4);
    z4 << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd H4 = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd R4 = Eigen::MatrixXd::Identity(4, 4);
    ekf.update(z4, H4, R4);

    require_near(ekf.data["residual_angle"], 4.0, 1e-9, "residual_angle must use the prior state");
}

// NIS 必须用先验残差和先验创新协方差，否则不是卡方分布
void test_nis_uses_prior()
{
    const int n = 4;
    Eigen::VectorXd x0(n);
    x0 << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n) * 2.0;
    ExtendedKalmanFilter ekf(x0, P0);

    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n);

    Eigen::VectorXd z(n);
    z << 1.5, 2.5, 3.5, 4.5;  // 先验残差 0.5，S = 3I => NIS = 4 * 0.25 / 3

    ekf.update(z, H, R);

    const double expected = 4.0 * 0.25 / 3.0;
    require_near(ekf.last_nis, expected, 1e-9, "last_nis must be the prior NIS");
    require_near(ekf.data.at("nis"), expected, 1e-9, "data[nis] must be the prior NIS");
}

// *_fail 是逐帧状态，不能在第一次失败后永久锁存
void test_fail_flags_reset()
{
    const int n = 4;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n) * 100.0;

    ExtendedKalmanFilter ekf(Eigen::VectorXd::Zero(n), P0);

    ekf.update(Eigen::VectorXd::Constant(n, 100.0), H, R);
    require(ekf.data.at("nis_fail") == 1.0, "large NIS must set nis_fail");

    ekf.x = Eigen::VectorXd::Zero(n);
    ekf.P = P0;
    ekf.update(Eigen::VectorXd::Zero(n), H, R);
    require(ekf.data.at("nis_fail") == 0.0, "nis_fail must reset after a passing update");
    require(ekf.data.at("nees_fail") == 0.0, "nees_fail must reset after a passing update");
}

// 模型自洽时，NIS 应服从自由度 = 量测维数的卡方分布
void test_nis_statistical_consistency()
{
    const int n = 4;
    const int samples = 2000;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n) * 100.0;

    ExtendedKalmanFilter ekf(Eigen::VectorXd::Zero(n), P0);

    std::mt19937 rng(12345);
    std::normal_distribution<double> noise(0.0, 1.0);

    double sum = 0.0;
    int exceed = 0;
    for (int i = 0; i < samples; i++) {
        ekf.x = Eigen::VectorXd::Zero(n);
        ekf.P = P0;

        // 先验均值误差 ~ N(0, P0)，量测噪声 ~ N(0, R)，则创新 ~ N(0, H P0 Hᵀ + R)
        Eigen::VectorXd z(n);
        for (int j = 0; j < n; j++) z[j] = 10.0 * noise(rng) + noise(rng);

        ekf.update(z, H, R);
        sum += ekf.last_nis;
        if (ekf.last_nis > 9.488) exceed++;  // dof=4 的 95% 上分位
    }

    const double mean = sum / samples;
    const double rate = static_cast<double>(exceed) / samples;

    require_near(mean, 4.0, 0.4, "mean NIS must be close to the measurement dimension");
    require(rate > 0.02 && rate < 0.10, "NIS exceedance rate must be near 5%");
}

// P0、Q、R 同比例缩放时状态轨迹不变、协方差同比例缩放、NIS 反比缩放
void test_scale_invariance()
{
    const int n = 4;
    const double k = 1e-2;

    Eigen::VectorXd x0(n);
    x0 << 1.0, -2.0, 0.5, 3.0;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n) * 5.0;
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n) * 0.3;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n) * 0.7;

    ExtendedKalmanFilter ref(x0, P0);
    ExtendedKalmanFilter scaled(x0, k * P0);

    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 1.0);

    for (int i = 0; i < 100; i++) {
        ref.predict(F, Q);
        scaled.predict(F, k * Q);

        Eigen::VectorXd z(n);
        for (int j = 0; j < n; j++) z[j] = x0[j] + noise(rng);

        ref.update(z, H, R);
        scaled.update(z, H, k * R);

        require(
            (ref.x - scaled.x).norm() < 1e-9,
            "state trajectory must not depend on the global noise scale");
        require(
            (ref.P - scaled.P / k).norm() < 1e-7, "covariance must scale with the noise scale");
        require_near(
            scaled.last_nis, ref.last_nis / k, 1e-6, "NIS must scale inversely with noise scale");
    }
}

int main()
{
    try {
        test_residual_indexing();
        test_nis_uses_prior();
        test_fail_flags_reset();
        test_nis_statistical_consistency();
        test_scale_invariance();

        std::puts("ekf_test passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ekf_test failed: %s\n", e.what());
        return 1;
    }
}
