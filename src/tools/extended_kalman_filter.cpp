#include "tools/extended_kalman_filter.hpp"

#include <Eigen/Cholesky>
#include <cmath>
#include <numeric>

namespace
{
// 卡方分布 95% 上分位：自由度 1..6 取表值，更大自由度用 Wilson-Hilferty 近似
double chi2_95(int dof)
{
    static constexpr double table[] = {0.0, 3.841, 5.991, 7.815, 9.488, 11.070, 12.592};
    if (dof <= 0) return 0.0;
    if (dof <= 6) return table[dof];

    const double k = static_cast<double>(dof);
    const double t = 1.0 - 2.0 / (9.0 * k) + 1.6448536269514722 * std::sqrt(2.0 / (9.0 * k));
    return k * t * t * t;
}
}  // namespace

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
    data["residual_yaw"] = 0.0;
    data["residual_pitch"] = 0.0;
    data["residual_distance"] = 0.0;
    data["residual_angle"] = 0.0;
    data["nis"] = 0.0;
    data["nees"] = 0.0;
    data["nis_fail"] = 0.0;
    data["nees_fail"] = 0.0;
    data["recent_nis_failures"] = 0.0;
    data["nis_mean"] = 0.0;
    data["r_var_yaw"] = 0.0;
    data["r_var_pitch"] = 0.0;
    data["r_var_distance"] = 0.0;
    data["r_var_angle"] = 0.0;
    data["R_est_yaw"] = 0.0;
    data["R_est_pitch"] = 0.0;
    data["R_est_distance"] = 0.0;
    data["R_est_angle"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
    const Eigen::Ref<const Eigen::MatrixXd> & F, const Eigen::Ref<const Eigen::MatrixXd> & Q)
{
    return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
    const Eigen::Ref<const Eigen::MatrixXd> & F, const Eigen::Ref<const Eigen::MatrixXd> & Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
    P = F * P * F.transpose() + Q;
    x = f(x);
    return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
    const Eigen::Ref<const Eigen::VectorXd> & z, const Eigen::Ref<const Eigen::MatrixXd> & H,
    const Eigen::Ref<const Eigen::MatrixXd> & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
    return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
    const Eigen::Ref<const Eigen::VectorXd> & z, const Eigen::Ref<const Eigen::MatrixXd> & H,
    const Eigen::Ref<const Eigen::MatrixXd> & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
    const Eigen::VectorXd x_prior = x;
    const Eigen::MatrixXd P_prior = P;

    // 先验残差与创新协方差：卡方检验必须用先验量，后验残差会系统性偏小
    const Eigen::VectorXd residual = z_subtract(z, h(x_prior));
    const Eigen::MatrixXd S = H * P_prior * H.transpose() + R;
    const Eigen::LDLT<Eigen::MatrixXd> S_ldlt(S);

    const Eigen::MatrixXd PHt = P_prior * H.transpose();
    const Eigen::MatrixXd K = S_ldlt.solve(PHt.transpose()).transpose();

    // Stable Compution of the Posterior Covariance
    // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
    P = (I - K * H) * P_prior * (I - K * H).transpose() + K * R * K.transpose();

    x = x_add(x_prior, K * residual);

    /// 卡方检验（先验量）
    const double nis = residual.dot(S_ldlt.solve(residual));

    // 残差方差与 NIS 的指数滑动平均，仅用于在线标定 R
    constexpr double alpha = 0.02;
    if (residual_var_.rows() != residual.rows()) {
        residual_var_ = Eigen::VectorXd::Zero(residual.rows());
    }
    residual_var_ = (1.0 - alpha) * residual_var_ + alpha * residual.cwiseProduct(residual);
    nis_mean_ = (1.0 - alpha) * nis_mean_ + alpha * nis;

    // 真实创新方差 ≈ var(r)，且 S - R = H P Hᵀ，故 R ≈ var(r) - (S - R)
    const Eigen::VectorXd R_est = residual_var_ - S.diagonal() + R.diagonal();

    // 注：真正的 NEES 需要真值，这里只有“修正幅度 / 先验协方差”，仅作调试量
    const Eigen::VectorXd delta = x - x_prior;
    double nees = 0.0;
    if (delta.squaredNorm() > 0.0) {
        const Eigen::VectorXd solved = P_prior.ldlt().solve(delta);
        if (solved.allFinite()) nees = delta.dot(solved);
    }

    const double nis_threshold = chi2_95(static_cast<int>(residual.rows()));
    const double nees_threshold = chi2_95(static_cast<int>(x.rows()));

    data["nis_fail"] = nis > nis_threshold ? 1.0 : 0.0;
    data["nees_fail"] = nees > nees_threshold ? 1.0 : 0.0;

    if (data["nis_fail"] > 0.0) nis_count_++;
    if (data["nees_fail"] > 0.0) nees_count_++;
    total_count_++;
    last_nis = nis;

    recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);

    if (recent_nis_failures.size() > window_size) {
        recent_nis_failures.pop_front();
    }

    int recent_failures =
        std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
    double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

    data["residual_yaw"] = residual.size() > 0 ? residual[0] : 0.0;
    data["residual_pitch"] = residual.size() > 1 ? residual[1] : 0.0;
    data["residual_distance"] = residual.size() > 2 ? residual[2] : 0.0;
    data["residual_angle"] = residual.size() > 3 ? residual[3] : 0.0;
    data["nis"] = nis;
    data["nis_mean"] = nis_mean_;
    data["nees"] = nees;
    data["recent_nis_failures"] = recent_rate;

    // R 的在线估计，供标定使用
    auto pick = [](const Eigen::VectorXd & v, int i) { return i < v.rows() ? v[i] : 0.0; };
    data["r_var_yaw"] = pick(residual_var_, 0);
    data["r_var_pitch"] = pick(residual_var_, 1);
    data["r_var_distance"] = pick(residual_var_, 2);
    data["r_var_angle"] = pick(residual_var_, 3);
    data["R_est_yaw"] = pick(R_est, 0);
    data["R_est_pitch"] = pick(R_est, 1);
    data["R_est_distance"] = pick(R_est, 2);
    data["R_est_angle"] = pick(R_est, 3);

    return x;
}
