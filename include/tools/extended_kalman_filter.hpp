#ifndef TOOLS__EXTENDED_KALMAN_FILTER_HPP
#define TOOLS__EXTENDED_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <cmath>
#include <deque>
#include <functional>
#include <map>

class ExtendedKalmanFilter
{
public:
    Eigen::VectorXd x;
    Eigen::MatrixXd P;

    ExtendedKalmanFilter() = default;

    ExtendedKalmanFilter(
        const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
            [](const Eigen::VectorXd &a, const Eigen::VectorXd &b)
        { return a + b; });

    // 使用 Eigen::Ref 使固定尺寸矩阵（如 Matrix<double,11,11>）可零拷贝传入
    Eigen::VectorXd predict(
        const Eigen::Ref<const Eigen::MatrixXd> &F, const Eigen::Ref<const Eigen::MatrixXd> &Q);

    Eigen::VectorXd predict(
        const Eigen::Ref<const Eigen::MatrixXd> &F, const Eigen::Ref<const Eigen::MatrixXd> &Q,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);

    Eigen::VectorXd update(
        const Eigen::Ref<const Eigen::VectorXd> &z, const Eigen::Ref<const Eigen::MatrixXd> &H,
        const Eigen::Ref<const Eigen::MatrixXd> &R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)>
            z_subtract =
                [](const Eigen::VectorXd &a, const Eigen::VectorXd &b)
        { return a - b; });

    Eigen::VectorXd update(
        const Eigen::Ref<const Eigen::VectorXd> &z, const Eigen::Ref<const Eigen::MatrixXd> &H,
        const Eigen::Ref<const Eigen::MatrixXd> &R,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)>
            z_subtract =
                [](const Eigen::VectorXd &a, const Eigen::VectorXd &b)
        { return a - b; });

    std::map<std::string, double> data; // 卡方检验数据
    std::deque<int> recent_nis_failures{0};
    size_t window_size = 100;
    double last_nis = 0.0; // 最近一次先验 NIS

    // 连续 NIS 超限帧数：被一次正常更新打断即清零，供收敛门使用
    int nis_fail_streak() const { return nis_fail_streak_; }

    // 量测中断（丢失/滑行）后旧的连续超限证据应作废
    void reset_nis_streak()
    {
        nis_fail_streak_ = 0;
        data["nis_fail_streak"] = 0.0;
    }

    // 先验失准时临时放大协方差：下一帧量测的卡尔曼增益随之变大，
    // 用来跳出「拒绝量测 → 无修正滑行 → 误差更大 → 继续拒绝」的自锁。
    // 只放大前 count 个分量（默认全部）；用对角相似变换 P ← S P S，保持对称正定。
    void inflate_covariance(double factor, int count = -1)
    {
        const int n = static_cast<int>(P.rows());
        if (count < 0 || count > n)
            count = n;
        Eigen::VectorXd s = Eigen::VectorXd::Ones(n);
        s.head(count).setConstant(std::sqrt(factor));
        P = s.asDiagonal() * P * s.asDiagonal();
    }

private:
    Eigen::MatrixXd I;
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add;

    int nees_count_ = 0;
    int nis_count_ = 0;
    int total_count_ = 0;

    // 残差方差与 NIS 的指数滑动平均（约 50 帧时间常数），仅用于在线标定 R
    Eigen::VectorXd residual_var_;
    double nis_mean_ = 0.0;
    int nis_fail_streak_ = 0;
};

// 收敛门判据：预热期内不判；预热结束后要求连续 streak 次更新超限。
// 预热期用来跳过目标刚建立时速度/角速度尚未收敛的阶段——那段 NIS 天然偏大，
// 但只会持续有限帧数，用「连续」而非「滑窗比例」即可把两者区分开。
inline bool bad_converge_tripped(
    bool enabled, int update_count, int nis_fail_streak, int warmup, int streak)
{
    if (!enabled)
        return false;
    return update_count >= warmup && nis_fail_streak >= streak;
}

#endif // TOOLS__EXTENDED_KALMAN_FILTER_HPP