// omni_standard：全向感知自瞄（3×USB YOLO 方位搜索 + 1×海康 YOLO 精确主瞄准），无显示窗口
//
// 架构要点（与 omnidirectional_sensing 的异步方案的关键区别）：
//  - 云台只有一个写入者（本主循环），不存在 200Hz 异步命令线程复用旧时间戳导致乱晃
//  - 海康优先：海康锁定 -> 由 planner 精确输出；海康无目标 -> USB 引导 yaw + pitch 预设
//  - USB 结果带采集时间戳，Decider 用 gimbal.q(采集时刻) 做时间对齐 + 老化/限速/低通
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "camera/camera.hpp"
#include "camera/usbcamera.hpp"
#include "comm/gimbal.hpp"
#include "detector/yolo.hpp"
#include "omni/decider.hpp"
#include "omni/perceptron.hpp"
#include "omni/usb_camera_config.hpp"
#include "planner.hpp"
#include "solver.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"
#include "tracker/tracker.hpp"

const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | yaml配置文件路径 }"
    "{simulate       | | 只输出模拟控制，不连接云台串口}"
    "{d display      | | 显示海康 + 三路 USB 调试画面}";

using namespace std::chrono_literals;

namespace
{
void draw_armor(
    cv::Mat & canvas, const Armor & armor, const cv::Scalar & color, int thickness = 2)
{
    if (armor.points.size() != 4) return;
    std::vector<cv::Point> points;
    points.reserve(4);
    for (const auto & point : armor.points)
        points.emplace_back(cvRound(point.x), cvRound(point.y));
    cv::polylines(canvas, points, true, color, thickness);
}

// 返回 true 表示用户按 q 退出
bool show_debug_view(
    const cv::Mat & hik_frame, const std::vector<Armor> & hik_armors,
    const std::unordered_map<std::string, DetectionResult> & latest_usb,
    const std::array<cv::Mat, 3> & usb_frames, const Decider::Guide & guide, bool hik_active,
    double detect_gain)
{
    cv::Mat canvas(720, 1280, CV_8UC3, cv::Scalar(20, 20, 20));
    auto put_tile = [&](int index, const cv::Mat & frame, const std::string & title,
                        const std::string & status, const cv::Scalar & status_color) {
        const int x = (index % 2) * 640;
        const int y = (index / 2) * 360;
        // 按比例缩放并居中，避免 4:3 画面被拉成 16:9（USB 相机为 320x240）
        cv::Mat tile(360, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        if (!frame.empty()) {
            const double scale = std::min(640.0 / frame.cols, 360.0 / frame.rows);
            cv::Mat scaled;
            cv::resize(frame, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
            scaled.copyTo(tile(cv::Rect(
                (640 - scaled.cols) / 2, (360 - scaled.rows) / 2, scaled.cols, scaled.rows)));
        }
        tile.copyTo(canvas(cv::Rect(x, y, 640, 360)));
        cv::putText(
            canvas, title, {x + 10, y + 28}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0),
            2);
        cv::putText(
            canvas, status, {x + 10, y + 56}, cv::FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2);
    };

    cv::Mat hik_view = hik_frame.empty() ? cv::Mat() : hik_frame.clone();
    if (!hik_view.empty())
        for (const auto & armor : hik_armors) draw_armor(hik_view, armor, cv::Scalar(0, 255, 0));
    put_tile(
        0, hik_view, "hik", hik_active ? "HIK active" : "HIK empty",
        hik_active ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255));

    for (int i = 0; i < 3; ++i) {
        const std::string key = "usb" + std::to_string(i);
        cv::Mat usb_view = usb_frames[i].empty() ? cv::Mat() : usb_frames[i].clone();
        std::string status = usb_frames[i].empty() ? "no frame" : "no target";
        auto it = latest_usb.find(key);
        // 结果比 0.5 s 还旧就不画框，避免相机停流后旧框一直挂在画面上
        if (it != latest_usb.end() &&
            delta_time(std::chrono::steady_clock::now(), it->second.timestamp) <= 0.5) {
            const auto & dr = it->second;
            if (!usb_view.empty()) {
                for (int k = 0; k < static_cast<int>(dr.armors.size()); ++k) {
                    if (k == dr.selected_index) continue;
                    draw_armor(usb_view, dr.armors[k], cv::Scalar(160, 160, 160), 1);
                }
                if (dr.selected_index >= 0 &&
                    dr.selected_index < static_cast<int>(dr.armors.size()))
                    draw_armor(
                        usb_view, dr.armors[dr.selected_index], cv::Scalar(0, 255, 255), 2);
            }
            status = "size " + std::to_string(dr.size_score).substr(0, 5);
        }
        if (guide.valid && guide.camera == key)
            status += guide.phase == Decider::GuidePhase::COAST ? "  COAST" : "  LOCK";
        // 显示的就是送检图（已压暗），标题标出 gain 方便判断开关是否生效
        const std::string title =
            key + "  gain " + std::to_string(detect_gain).substr(0, 4);
        put_tile(i + 1, usb_view, title, status, cv::Scalar(0, 255, 255));
    }

    cv::imshow("omni debug", canvas);
    return cv::waitKey(1) == 'q';
}
}  // namespace

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>("@config-path");
    if (cli.has("help") || !cli.has("@config-path")) {
        cli.printMessage();
        return 0;
    }
    const bool display = cli.has("display");

    Exiter exiter;

    // ---- USB 设备和安装角从同一配置项读取，确保顺序一致 ----
    auto yaml = yaml_load(config_path);
    const auto usb_cameras = load_usb_camera_configs(yaml);
    const double hik_lost_s =
        yaml["omni_hik_lost_s"] ? yaml["omni_hik_lost_s"].as<double>() : 0.3;

    // ---- 海康精确链路 ----
    Gimbal gimbal(config_path, cli.has("simulate"));
    Camera hik(config_path);
    YOLO yolo(config_path, true);
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Planner planner(config_path);
    Decider decider(config_path);

    // ---- 3 路 USB 方位搜索（YOLO 检测；不加载分类器、不做传统矫正）----
    std::array<std::unique_ptr<USBCamera>, 3> usb_cams;
    for (size_t i = 0; i < usb_cams.size() && i < usb_cameras.size(); ++i)
        usb_cams[i] = std::make_unique<USBCamera>(usb_cameras[i].device, config_path);
    Perceptron perceptron(
        usb_cams[0].get(), usb_cams[1].get(), usb_cams[2].get(), decider, config_path);

    logger()->info("[omni_standard] started, config: {}", config_path);

    cv::Mat img;
    std::chrono::steady_clock::time_point t;
    auto last_mode = GimbalMode::IDLE;
    auto last_hik_target_t = std::chrono::steady_clock::now() - 10s;
    auto last_hik_frame_t = std::chrono::steady_clock::now() - 10s;
    auto last_hik_warn_t = std::chrono::steady_clock::now() - 10s;
    int guide_log_count = 0;
    // 调试显示用：每路最近一条结果（跨轮保留）。检测被限频，主循环更快，
    // 只用本批结果会让框以检测帧率闪烁。
    std::unordered_map<std::string, DetectionResult> usb_latest;

    auto stop_gimbal = [&] { gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true); };

    // USB 引导指令：锁定/coast 都发 mode=1 + is_empty=false（视觉控 yaw、不开火）。
    // guide 无效时只发停止，不 reset_guide——内部可能正处于冷却，清掉会让它立刻重新锁定。
    auto apply_guide = [&](const Decider::Guide & guide) {
        if (!guide.valid) {
            stop_gimbal();
            return;
        }

        gimbal.send(true, false, guide.yaw, 0, 0, guide.pitch, 0, 0, false);
        if (++guide_log_count % 30 == 0) {
            const char * phase = guide.phase == Decider::GuidePhase::COAST ? "coast" : "lock";
            logger()->info(
                "[omni] USB guide {} {} yaw={:.1f}deg age={:.3f}s", phase, guide.camera,
                guide.yaw * 57.3, guide.age);
        }
    };

    while (!exiter.exit()) {
        auto mode = gimbal.mode();
        if (last_mode != mode) {
            logger()->info("Switch to {}", gimbal.str(mode));
            last_mode = mode;
        }

        // 1) 海康精确链路：同步读取 + 帧时间戳对齐（q(t) 为该帧采集时刻的姿态）
        //    用带超时的读取：海康掉线/未出图时不再永久阻塞，Ctrl+C 仍可退出
        bool hik_ok = hik.try_read(img, t, 100);
        if (hik_ok && !img.empty()) last_hik_frame_t = std::chrono::steady_clock::now();

        // 2) USB 搜索层：每轮必 drain，防止队列堆积
        auto usb_results = perceptron.get_detection_queue();
        for (const auto & dr : usb_results) {
            if (dr.camera.empty()) continue;
            auto & slot = usb_latest[dr.camera];
            if (slot.timestamp <= dr.timestamp) slot = dr;
        }

        if (!hik_ok || img.empty()) {
            // 海康相机不可用：三路 USB 仍在检测，按“海康无目标”降级为 USB 引导。
            // 不能直接 continue——那会让云台收不到任何指令，保持上一条带速度前馈的控制。
            const auto now = std::chrono::steady_clock::now();
            // 与“海康有画面但没目标”区分开，掉线是硬件/链路问题
            if (delta_time(now, last_hik_warn_t) > 2.0) {
                last_hik_warn_t = now;
                logger()->warn("[omni] Hik unavailable (no frame)");
            }

            // 刚掉帧时只发停止：给海康 omni_hik_lost_s 的恢复窗口，
            // 避免偶发丢帧就在精确链路与 USB 引导之间来回切。
            const bool hik_recent = delta_time(now, last_hik_frame_t) < hik_lost_s;
            Decider::Guide guide;
            if (mode != GimbalMode::AUTO_AIM) {
                decider.reset_guide();
                stop_gimbal();
            } else if (hik_recent) {
                stop_gimbal();
            } else {
                guide = decider.arbitrate_usb(usb_results, gimbal, now, /*hik_active=*/false);
                apply_guide(guide);
            }

            if (display) {
                if (show_debug_view(
                        cv::Mat(), {}, usb_latest, perceptron.latest_frames(), guide, false,
                        perceptron.detect_gain()))
                    break;
            }
            continue;
        }

        auto gs = gimbal.state();
        solver.set_R_gimbal2world(gimbal.q(t));

        if (mode != GimbalMode::AUTO_AIM) {
            decider.reset_guide();
            stop_gimbal();
            continue;
        }

        // 3) 海康目标检测/跟踪/规划
        auto armors = yolo.detect(img);
        for (auto & armor : armors) solver.solve(armor);
        auto targets = tracker.track(armors, t, gs.id);

        std::optional<Target> target;
        if (!targets.empty()) target = targets.front();

        auto plan = planner.plan(target, gs.bullet_speed);
        const bool is_empty = !target.has_value();

        // 海康回退迟滞：连续空 hik_lost_s 才交给 USB，避免两套目标反复抢控制
        const auto now = std::chrono::steady_clock::now();
        if (!targets.empty()) last_hik_target_t = now;
        const bool hik_active =
            !targets.empty() || delta_time(now, last_hik_target_t) < hik_lost_s;

        // 每轮都调用：hik_active 时内部会清空锁定/coast 状态
        auto guide = decider.arbitrate_usb(usb_results, gimbal, now, hik_active);

        if (hik_active) {
            if (plan.control) {
                // 电控只用了速度环：把位置环上移到视觉端（速度 = MPC 前馈 + 位置误差反馈）
                constexpr double kp_yaw = 8.0;     // (rad/s)/rad
                constexpr double kp_pitch = 12.0;  // (rad/s)/rad
                double yaw_vel_cmd = plan.yaw_vel + kp_yaw * limit_rad(plan.yaw - gs.yaw);
                double pitch_vel_cmd = plan.pitch_vel + kp_pitch * (plan.pitch - gs.pitch);

                gimbal.send(
                    plan.control, plan.fire, plan.yaw, yaw_vel_cmd, plan.yaw_acc, plan.pitch,
                    pitch_vel_cmd, plan.pitch_acc, is_empty);
            } else {
                stop_gimbal();
            }
        } else {
            apply_guide(guide);
        }

        if (display) {
            if (show_debug_view(
                    img, armors, usb_latest, perceptron.latest_frames(), guide, hik_active,
                    perceptron.detect_gain()))
                break;
        }
    }

    if (display) cv::destroyAllWindows();
    stop_gimbal();
    logger()->info("[omni_standard] exit");
    return 0;
}
