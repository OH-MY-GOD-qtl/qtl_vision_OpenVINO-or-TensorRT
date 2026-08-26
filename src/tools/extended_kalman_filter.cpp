#include "tools/extended_kalman_filter.hpp"

#include <numeric>


ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
    data["nis"] = 0.0;
    data["nees"] = 0.0;
    data["nis_fail"] = 0.0;
    data["nees_fail"] = 0.0;
    data["recent_nis_failures"] = 0.0;
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

    // 观测新息与创新协方差（均基于先验，卡方检验必须用新息而非后验残差）
    const Eigen::VectorXd innovation = z_subtract(z, h(x_prior));
    const Eigen::MatrixXd S = H * P_prior * H.transpose() + R;

    // 卡尔曼增益
    const Eigen::MatrixXd K = P_prior * H.transpose() * S.inverse();

    // 后验状态与协方差（Joseph 稳定形式）
    // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
    x = x_add(x_prior, K * innovation);
    P = (I - K * H) * P_prior * (I - K * H).transpose() + K * R * K.transpose();

    // 卡方检验：NIS 用新息；NEES 用先验协方差
    // 注意：若 x_add 对角度分量做了包裹，NEES 的 (x - x_prior) 在角度分量上可能差 ±2π，仅影响统计显示
    const double nis = innovation.transpose() * S.inverse() * innovation;
    const double nees = (x - x_prior).transpose() * P_prior.inverse() * (x - x_prior);

    last_nis = nis;
    last_residual = innovation;

    const bool nis_fail = nis > nis_threshold;
    const bool nees_fail = nees > nees_threshold;
    data["nis"] = nis;
    data["nees"] = nees;
    data["nis_fail"] = nis_fail ? 1.0 : 0.0;
    data["nees_fail"] = nees_fail ? 1.0 : 0.0;

    recent_nis_failures.push_back(nis_fail ? 1 : 0);
    if (recent_nis_failures.size() > window_size) {
        recent_nis_failures.pop_front();
    }
    data["recent_nis_failures"] =
        static_cast<double>(
            std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0)) /
        recent_nis_failures.size();

    return x;
}

