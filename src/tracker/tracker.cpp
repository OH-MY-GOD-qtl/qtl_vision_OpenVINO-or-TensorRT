#include "tracker/tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"
#include "tracker/reproj_match.hpp"

namespace
{
    // 跳出拒绝自锁时对先验协方差的放大倍数：实测位置误差可达数个板宽，
    // 增益需要从 ~0.1 提到 ~0.5 才能在状态机超时前拉回来。
    constexpr double kReprojEscapeInflate = 10.0;
} // namespace

Tracker::Tracker(const std::string &config_path, Solver &solver)
    : solver_{solver},
      detect_count_(0),
      temp_lost_count_(0),
      state_{"lost"},
      pre_state_{"lost"},
      last_timestamp_(std::chrono::steady_clock::now()),
      omni_target_priority_{ArmorPriority::fifth}
{
    auto yaml = yaml_load(config_path);
    enemy_color_ =
        (yaml_read<std::string>(yaml, "enemy_color") == "red") ? Color::red : Color::blue;
    min_detect_count_ = yaml_read<int>(yaml, "min_detect_count");
    max_temp_lost_count_ = yaml_read<int>(yaml, "max_temp_lost_count");
    outpost_max_temp_lost_count_ = yaml_read<int>(yaml, "outpost_max_temp_lost_count");
    normal_temp_lost_count_ = max_temp_lost_count_;
    switching_max_temp_lost_count_ =
        yaml["switching_max_temp_lost_count"] ? yaml["switching_max_temp_lost_count"].as<int>()
                                              : 200;

    // NIS 收敛门：只在目标收敛之后统计，且要求连续超限。
    // 7765b04 的 NIS 用后验残差计算，数值极小，这个门实际上从未触发；
    // 改成先验 NIS 后判据才真正生效。
    // 但 NIS 衡量的是「运动模型拟合度」而非「是否跟丢」：目标加速时模型失配，
    // NIS 会持续超限，而跟踪本身是正常的。因此默认关闭，避免把机动判成丢目标。
    drop_on_bad_converge_ =
        yaml["ekf_drop_on_bad_converge"] ? yaml["ekf_drop_on_bad_converge"].as<bool>() : false;
    bad_converge_warmup_ =
        yaml["ekf_bad_converge_warmup"] ? yaml["ekf_bad_converge_warmup"].as<int>() : 30;
    bad_converge_streak_ =
        yaml["ekf_bad_converge_streak"] ? yaml["ekf_bad_converge_streak"].as<int>() : 20;
    // 配置兜底：streak<=0 会让判据恒真，warmup<0 没有意义
    bad_converge_warmup_ = std::max(0, bad_converge_warmup_);
    bad_converge_streak_ = std::max(1, bad_converge_streak_);

    // 重投影关联：把预测的装甲板投到图像，与检测四边形的归一化像素误差上限。
    // 超限只表示本帧量测不可信，是否丢目标仍由 state_machine 的帧数迟滞决定。
    // 初值取宽松的 0.5（误差约半个板宽），等实机日志的分布出来再收紧。
    reproj_max_error_ = yaml["reproj_max_error"] ? yaml["reproj_max_error"].as<double>() : 0.5;
    if (!(reproj_max_error_ > 0.0))
        reproj_max_error_ = 0.5;

    // 连续拒绝逃逸：门只应该挡零星离群帧。实测每次拒绝都成段、长度直达 temp_lost
    // 上限，说明连续超限时错的是先验而不是量测，必须放行并放大先验，
    // 否则滤波器会在无修正下滑行，直到状态机超时丢目标。
    reproj_max_reject_streak_ =
        yaml["reproj_max_reject_streak"] ? yaml["reproj_max_reject_streak"].as<int>() : 3;
    reproj_max_reject_streak_ = std::max(0, reproj_max_reject_streak_);
    reproj_reject_streak_ = 0;

    // 从配置文件读取相机矩阵并计算图像中心坐标
    auto camera_matrix_data = yaml_read<std::vector<double>>(yaml, "camera_matrix");
    double cx = camera_matrix_data[2]; // 相机矩阵的第3个元素为图像中心x坐标
    double cy = camera_matrix_data[5]; // 相机矩阵的第6个元素为图像中心y坐标
    img_center_ = cv::Point2f(cx, cy);
}

std::string Tracker::state() const { return state_; }

std::vector<Target> Tracker::track(
    std::vector<Armor> &armors, std::chrono::steady_clock::time_point t, int id,
    bool use_enemy_color)
{
    auto dt = delta_time(t, last_timestamp_);
    last_timestamp_ = t;
    set_enemy_color(id);

    // 时间间隔过长，说明可能发生了相机离线
    if (state_ != "lost" && dt > 0.1)
    {
        logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
        state_ = "lost";
    }
    // 过滤掉非我方装甲板
    armors.erase(
        std::remove_if(
            armors.begin(), armors.end(), [&](const Armor &a)
            { return a.color != enemy_color_; }),
        armors.end());

    // 过滤前哨站顶部装甲板
    // armors.erase(std::remove_if(armors.begin(), armors.end(), [this](const Armor & a) {
    //   return a.name == ArmorName::outpost &&
    //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
    //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
    // }), armors.end());

    // 优先选择靠近图像中心的装甲板
    std::sort(armors.begin(), armors.end(), [this](const Armor &a, const Armor &b)
              {
        auto distance_1 = cv::norm(a.center - img_center_);
        auto distance_2 = cv::norm(b.center - img_center_);
        return distance_1 < distance_2; });

    // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
    std::stable_sort(armors.begin(), armors.end(), [](const Armor &a, const Armor &b)
                     { return a.priority < b.priority; });

    bool found;
    if (state_ == "lost")
    {
        found = set_target(armors, t);
    }

    else
    {
        found = update_target(armors, t);
    }

    state_machine(found);

    // 发散检测
    if (state_ != "lost" && target_.diverged())
    {
        logger()->debug("[Tracker] Target diverged!");
        state_ = "lost";
        return {};
    }

    // 收敛效果检测：预热期之后连续超限才判坏，避免被初始化收敛段污染
    if (
        bad_converge_tripped(
            drop_on_bad_converge_, target_.update_count(), target_.ekf().nis_fail_streak(),
            bad_converge_warmup_, bad_converge_streak_))
    {
        logger()->debug(
            "[Target] Bad Converge Found! streak={} updates={} nis={:.2f}",
            target_.ekf().nis_fail_streak(), target_.update_count(), target_.ekf().last_nis);
        state_ = "lost";
        return {};
    }

    if (state_ == "lost")
        return {};

    std::vector<Target> targets = {target_};
    return targets;
}

std::tuple<DetectionResult, std::vector<Target>> Tracker::track(
    const std::vector<DetectionResult> &detection_queue, std::vector<Armor> &armors,
    std::chrono::steady_clock::time_point t, int id, bool use_enemy_color)
{
    DetectionResult switch_target{std::vector<Armor>(), t, 0, 0};
    DetectionResult temp_target{std::vector<Armor>(), t, 0, 0};
    if (!detection_queue.empty())
    {
        temp_target = detection_queue.front();
    }

    auto dt = delta_time(t, last_timestamp_);
    last_timestamp_ = t;

    // 时间间隔过长，说明可能发生了相机离线
    if (state_ != "lost" && dt > 0.1)
    {
        logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
        state_ = "lost";
    }

    // 优先选择靠近图像中心的装甲板
    std::sort(armors.begin(), armors.end(), [this](const Armor &a, const Armor &b)
              {
        auto distance_1 = cv::norm(a.center - img_center_);
        auto distance_2 = cv::norm(b.center - img_center_);
        return distance_1 < distance_2; });

    // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
    std::stable_sort(armors.begin(), armors.end(), [](const Armor &a, const Armor &b)
                     { return a.priority < b.priority; });

    bool found;
    if (state_ == "lost")
    {
        found = set_target(armors, t);
    }

    // 此时主相机画面中出现了优先级更高的装甲板，切换目标
    else if (
        state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority)
    {
        found = set_target(armors, t);
        logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
    }

    // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
    else if (
        state_ == "tracking" && !temp_target.armors.empty() &&
        temp_target.armors.front().priority < target_.priority && target_.convergened())
    {
        state_ = "switching";
        switch_target =
            DetectionResult{temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
        omni_target_priority_ = temp_target.armors.front().priority;
        found = false;
        logger()->debug("omniperception find higher priority target");
    }

    else if (state_ == "switching")
    {
        found = !armors.empty() && armors.front().priority == omni_target_priority_;
    }

    else if (state_ == "detecting" && pre_state_ == "switching")
    {
        found = set_target(armors, t);
    }

    else
    {
        found = update_target(armors, t);
    }

    pre_state_ = state_;
    // 更新状态机
    state_machine(found);

    // 发散检测
    if (state_ != "lost" && target_.diverged())
    {
        logger()->debug("[Tracker] Target diverged!");
        state_ = "lost";
        return {switch_target, {}}; // 返回switch_target和空的targets
    }

    if (state_ == "lost")
        return {switch_target, {}}; // 返回switch_target和空的targets

    std::vector<Target> targets = {target_};
    return {switch_target, targets};
}

void Tracker::state_machine(bool found)
{
    if (state_ == "lost")
    {
        if (!found)
            return;

        state_ = "detecting";
        detect_count_ = 1;
    }

    else if (state_ == "detecting")
    {
        if (found)
        {
            detect_count_++;
            if (detect_count_ >= min_detect_count_)
                state_ = "tracking";
        }
        else
        {
            detect_count_ = 0;
            state_ = "lost";
        }
    }

    else if (state_ == "tracking")
    {
        if (found)
            return;

        temp_lost_count_ = 1;
        state_ = "temp_lost";
    }

    else if (state_ == "switching")
    {
        if (found)
        {
            state_ = "detecting";
        }
        else
        {
            temp_lost_count_++;
            if (temp_lost_count_ > switching_max_temp_lost_count_)
                state_ = "lost";
        }
    }

    else if (state_ == "temp_lost")
    {
        if (found)
        {
            state_ = "tracking";
        }
        else
        {
            temp_lost_count_++;
            if (target_.name == ArmorName::outpost)
                // 前哨站的temp_lost_count需要设置的大一些
                max_temp_lost_count_ = outpost_max_temp_lost_count_;
            else
                max_temp_lost_count_ = normal_temp_lost_count_;

            if (temp_lost_count_ > max_temp_lost_count_)
                state_ = "lost";
        }
    }
}

bool Tracker::set_target(std::vector<Armor> &armors, std::chrono::steady_clock::time_point t)
{
    // 新目标没有历史：连续拒绝计数从零开始
    reproj_reject_streak_ = 0;

    if (armors.empty())
        return false;

    auto &armor = armors.front();
    solver_.solve(armor);

    // 根据兵种优化初始化参数
    auto is_balance = (armor.type == ArmorType::big) &&
                      (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                       armor.name == ArmorName::five);

    // 状态下标：0 x, 1 vx, 2 y, 3 vy, 4 z, 5 vz, 6 相位, 7 角速度, 8 半径, 9 半径差, 10 高度差
    // r/l/h 是物理常数，Q 里过程噪声为 0，只靠量测推动。初始方差给 1（σ=1 m）时，
    // 单装甲板可视、目标不旋转的情况下 r 不可观测，会被噪声推到 0.05 以下触发 diverged。
    // 收到 1e-2（σ=0.1 m），与前哨站/基地一致。
    if (is_balance)
    {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-2, 1e-2, 1}};
        target_ = Target(armor, t, 0.2, 2, P0_dig);
    }

    else if (armor.name == ArmorName::outpost)
    {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
        target_ = Target(armor, t, 0.2765, 3, P0_dig);
    }

    else if (armor.name == ArmorName::base)
    {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
        target_ = Target(armor, t, 0.3205, 3, P0_dig);
    }

    else
    {
        Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-2, 1e-2, 1}};
        target_ = Target(armor, t, 0.2, 4, P0_dig);
    }

    return true;
}

bool Tracker::associate_armor(const Armor &armor, int &id, double &err_norm, double &margin) const
{
    const auto &xyza_list = target_.armor_xyza_list();
    if (xyza_list.empty())
        return false;

    double best = std::numeric_limits<double>::infinity();
    double second = std::numeric_limits<double>::infinity();
    int best_id = -1;

    for (int i = 0; i < static_cast<int>(xyza_list.size()); i++)
    {
        // 用预测状态把该 id 的装甲板投到图像，和检测四边形比几何。
        // 只用 PnP 的位置，不用它的 yaw——单块平面板的 yaw 在正对相机时是病态的。
        const auto uv =
            solver_.reproject_armor(xyza_list[i].head(3), xyza_list[i][3], armor.type, armor.name);
        const double e = reproj::normalized_quad_distance(uv, armor.points);

        if (e < best)
        {
            second = best;
            best = e;
            best_id = i;
        }
        else if (e < second)
        {
            second = e;
        }
    }

    if (best_id < 0 || !std::isfinite(best))
        return false;

    id = best_id;
    err_norm = best;
    // 次优/最优：越大说明这次 id 决策越不模糊。只有一块装甲板时没有次优，记 inf。
    margin = (best > 1e-9) ? second / best : std::numeric_limits<double>::infinity();
    return true;
}

bool Tracker::update_target(std::vector<Armor> &armors, std::chrono::steady_clock::time_point t)
{
    target_.predict(t);

    // 刚建目标（detecting）时状态由首帧量测直接初始化，重投影误差没有判别力；
    // 而 state_machine 的 detecting 分支没有帧数迟滞，一次拒绝就直接判 lost。
    // 所以闸门只在已经有预测的状态（tracking / temp_lost / switching）下生效。
    const bool gate_active = (state_ != "detecting");

    bool measured = false;
    bool any_candidate = false;
    for (auto &armor : armors)
    {
        if (armor.name != target_.name || armor.type != target_.armor_type)
            continue;

        any_candidate = true;

        // 关联只用检测框和预测状态，不需要 PnP 位姿，所以放在解算之前：
        // 拿到 id 后就能用该装甲板的预测 yaw 作为 PnP yaw 优化的搜索中心。
        int id = 0;
        double err_norm = 0.0, margin = 0.0;
        if (!associate_armor(armor, id, err_norm, margin))
        {
            // 无预测可用：仍做一次常规解算，保证下游拿到的位姿有效
            solver_.solve(armor);
            continue;
        }

        target_.set_reproj_stats(err_norm, margin);

        bool accept = true;
        if (gate_active && err_norm > reproj_max_error_)
        {
            reproj_reject_streak_++;

            // 零星离群帧：丢掉。连续超限：错的是先验，放行。
            if (reproj_reject_streak_ <= reproj_max_reject_streak_)
            {
                logger()->debug(
                    "[Tracker] reproj reject id={} err={:.3f} > {:.3f} streak={}", id, err_norm,
                    reproj_max_error_, reproj_reject_streak_);
                accept = false;
            }
            else
            {
                // 只有逃逸的那一帧放大先验，之后靠量测自己收回来，避免 P 越滚越大
                if (reproj_reject_streak_ == reproj_max_reject_streak_ + 1)
                    target_.inflate_prior(kReprojEscapeInflate);

                logger()->debug(
                    "[Tracker] reproj force accept id={} err={:.3f} streak={}", id, err_norm,
                    reproj_reject_streak_);
            }
        }
        else
        {
            reproj_reject_streak_ = 0;
        }

        // 用该装甲板的预测 yaw 作搜索中心，避免云台 yaw 与真实板 yaw 相差过大时
        // PnP yaw 优化卡在搜索窗边界、把错误角度灌进 EKF。
        // 只在量测被接受时用：err_norm 已经超限说明预测本身不可信，
        // 这时退回云台 yaw 更安全。
        if (accept)
        {
            const double pred_yaw = target_.armor_xyza_list()[id][3];
            solver_.solve(armor, pred_yaw);
            target_.update(armor, id);
            measured = true;
        }
        else
        {
            solver_.solve(armor);
        }
    }

    if (!measured)
    {
        // 本帧没有可用量测：旧的连续超限证据不应跨中断继续累计
        target_.reset_nis_streak();
        // 根本没看到同名装甲板时，「连续拒绝」不成立，计数作废
        if (!any_candidate)
            reproj_reject_streak_ = 0;
    }

    return measured;
}

bool Tracker::set_enemy_color(int id)
{
    // 红方:1-11 蓝方:101-111
    if ((id < 1 || id > 11) && (id < 101 || id > 111))
    {
        // 仅在 id 变化时提示一次，避免刷屏（模拟模式下 id 恒为 0）
        static int last_invalid_id = -1;
        if (id != last_invalid_id)
        {
            logger()->warn("[Tracker] Invalid ID {}. Using enemy color from config file.", id);
            last_invalid_id = id;
        }
        return false; // 无效的id
    }
    if (id < 100)
    {
        enemy_color_ = Color::blue;
    }
    else
    {
        enemy_color_ = Color::red;
    }
    return true;
}

Color Tracker::get_enemy_color() const { return enemy_color_; }
