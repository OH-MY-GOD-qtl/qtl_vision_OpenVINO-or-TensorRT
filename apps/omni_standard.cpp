// omni_standard：全向感知自瞄（3×USB YOLO 方位搜索 + 1×海康 YOLO 精确主瞄准），无显示窗口
//
// 架构（海康链路与 auto_aim_debug_mpc 同构）：
//  - 视觉线程：海康读帧 -> YOLO -> Tracker -> 把 Target 推进单槽队列，不碰云台
//  - 控制线程：队列取目标 -> MPC -> 速度环反馈 + 开火门控 -> 云台唯一写入者
//    （云台只能有一个写入者，否则会出现异步线程复用旧时间戳导致的乱晃）
//  - 海康优先：海康锁定 -> 由 planner 精确输出；海康无目标 -> USB 引导 yaw + pitch 预设
//  - USB 结果带采集时间戳，Decider 用 gimbal.q(采集时刻) 做时间对齐 + 老化/限速/低通
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <thread>
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
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"
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
    void draw_armor(cv::Mat &canvas, const Armor &armor, const cv::Scalar &color, int thickness = 2)
    {
        if (armor.points.size() != 4)
            return;
        std::vector<cv::Point> points;
        points.reserve(4);
        for (const auto &point : armor.points)
            points.emplace_back(cvRound(point.x), cvRound(point.y));
        cv::polylines(canvas, points, true, color, thickness);
    }

    void draw_target_projection(
        cv::Mat &image, const Target &target, const Solver &solver, const Planner &planner,
        double bullet_speed, std::chrono::steady_clock::time_point timestamp)
    {
        for (const Eigen::Vector4d &xyza : target.armor_xyza_list())
        {
            auto points =
                solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
            draw_points(image, points, {0, 255, 0});
        }

        auto aim_xyza = planner.predict_xyza(target, bullet_speed, timestamp);
        auto aim_points =
            solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        draw_points(image, aim_points, {0, 0, 255});
    }

    // 返回 true 表示用户按 q 退出
    bool show_debug_view(
        const cv::Mat &hik_frame, const std::vector<Armor> &hik_armors,
        const std::unordered_map<std::string, DetectionResult> &latest_usb,
        const std::array<cv::Mat, 3> &usb_frames, const Decider::Guide &guide, bool hik_active,
        double detect_gain)
    {
        cv::Mat canvas(720, 1280, CV_8UC3, cv::Scalar(20, 20, 20));
        auto put_tile = [&](int index, const cv::Mat &frame, const std::string &title,
                            const std::string &status, const cv::Scalar &status_color)
        {
            const int x = (index % 2) * 640;
            const int y = (index / 2) * 360;
            // 按比例缩放并居中，避免 4:3 画面被拉成 16:9（USB 相机为 320x240）
            cv::Mat tile(360, 640, CV_8UC3, cv::Scalar(0, 0, 0));
            if (!frame.empty())
            {
                const double scale = std::min(640.0 / frame.cols, 360.0 / frame.rows);
                cv::Mat scaled;
                cv::resize(frame, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
                scaled.copyTo(tile(
                    cv::Rect(
                        (640 - scaled.cols) / 2, (360 - scaled.rows) / 2, scaled.cols,
                        scaled.rows)));
            }
            tile.copyTo(canvas(cv::Rect(x, y, 640, 360)));
            cv::putText(
                canvas, title, {x + 10, y + 28}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
            cv::putText(
                canvas, status, {x + 10, y + 56}, cv::FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2);
        };

        cv::Mat hik_view = hik_frame.empty() ? cv::Mat() : hik_frame.clone();
        if (!hik_view.empty())
            for (const auto &armor : hik_armors) draw_armor(hik_view, armor, cv::Scalar(0, 255, 0));
        put_tile(
            0, hik_view, "hik", hik_active ? "HIK active" : "HIK empty",
            hik_active ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255));

        for (int i = 0; i < 3; ++i)
        {
            const std::string key = "usb" + std::to_string(i);
            cv::Mat usb_view = usb_frames[i].empty() ? cv::Mat() : usb_frames[i].clone();
            std::string status = usb_frames[i].empty() ? "no frame" : "no target";
            auto it = latest_usb.find(key);
            // 结果比 0.5 s 还旧就不画框，避免相机停流后旧框一直挂在画面上
            if (it != latest_usb.end() &&
                delta_time(std::chrono::steady_clock::now(), it->second.timestamp) <= 0.5)
            {
                const auto &dr = it->second;
                if (!usb_view.empty())
                {
                    for (int k = 0; k < static_cast<int>(dr.armors.size()); ++k)
                    {
                        if (k == dr.selected_index)
                            continue;
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
            const std::string title = key + "  gain " + std::to_string(detect_gain).substr(0, 4);
            put_tile(i + 1, usb_view, title, status, cv::Scalar(0, 255, 255));
        }

        cv::imshow("omni debug", canvas);
        return cv::waitKey(1) == 'q';
    }
} // namespace

int main(int argc, char *argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>("@config-path");
    if (cli.has("help") || !cli.has("@config-path"))
    {
        cli.printMessage();
        return 0;
    }
    const bool display = cli.has("display");

    Exiter exiter;
    Plotter plotter; // UDP 127.0.0.1:9870，供 PlotJuggler 显示

    // ---- USB 设备和安装角从同一配置项读取，确保顺序一致 ----
    auto yaml = yaml_load(config_path);
    const auto usb_cameras = load_usb_camera_configs(yaml);
    const double hik_lost_s = yaml["omni_hik_lost_s"] ? yaml["omni_hik_lost_s"].as<double>() : 0.3;

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

    // ---- 线程间状态（与 auto_aim_debug_mpc 同构）----
    // 视觉线程只「出目标」，控制线程是云台唯一写入者。
    // 单槽队列 + 预置 nullopt：front() 阻塞且不 pop，保证控制线程第一拍就有值。
    ThreadSafeQueue<std::optional<Target>, true> target_queue(1);
    target_queue.push(std::nullopt);

    std::atomic<bool> quit = false;
    std::atomic<bool> hik_frame_ok = false; // 视觉线程写，控制线程读（只用于日志/绘图）

    // 调试显示快照：控制线程写，视觉线程读
    struct ControlSnapshot
    {
        bool hik_active = false;
        Decider::Guide guide;
        // 每路最近一条结果（跨轮保留）。检测被限频，主循环更快，
        // 只用本批结果会让框以检测帧率闪烁。
        std::unordered_map<std::string, DetectionResult> usb_latest;
    };
    std::mutex snapshot_mutex;
    ControlSnapshot snapshot;

    const auto long_ago = std::chrono::steady_clock::now() - 10s;
    const auto t0 = std::chrono::steady_clock::now();

    auto stop_gimbal = [&] { gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true); };

    // PlotJuggler：控制线程每拍发一帧，字段口径与 auto_aim_debug_mpc 一致
    // （UDP 127.0.0.1:9870）。guide_* 是 omni_standard 额外的 USB 引导诊断量，
    // mode 仍是云台模式（不是控制/开火状态），避免破坏现有调参布局。
    auto publish_plot = [&](bool hik_ok, bool hik_active, const Decider::Guide &guide,
                            const std::optional<Target> &target, const Plan &plan, bool fire_cmd,
                            bool is_empty)
    {
        nlohmann::json data;
        data["t"] = delta_time(std::chrono::steady_clock::now(), t0);
        data["mode"] = static_cast<int>(gimbal.mode());
        data["hik_ok"] = hik_ok ? 1 : 0;
        data["hik_target"] = target.has_value() ? 1 : 0;
        data["hik_active"] = hik_active ? 1 : 0;
        data["is_empty"] = is_empty ? 1 : 0;

        // phase: 0=none 1=lock 2=coast；camera: usb0..2 -> 0..2，未锁定 -1
        data["guide_valid"] = guide.valid ? 1 : 0;
        data["guide_phase"] = guide.phase == Decider::GuidePhase::LOCKED  ? 1
                              : guide.phase == Decider::GuidePhase::COAST ? 2
                                                                          : 0;
        int camera_index = -1;
        if (guide.camera.size() == 4 && guide.camera.rfind("usb", 0) == 0)
            camera_index = guide.camera[3] - '0';
        data["guide_camera"] = camera_index;
        data["guide_yaw"] = guide.yaw * 57.3;
        data["guide_pitch"] = guide.pitch * 57.3;
        data["guide_age"] = guide.age;

        auto gs = gimbal.state();
        data["gimbal_yaw"] = gs.yaw * 57.3;
        data["gimbal_pitch"] = gs.pitch * 57.3;
        data["bullet_speed"] = gs.bullet_speed;
        data["robot_id"] = gs.id;

        data["target_yaw"] = plan.target_yaw * 57.3;
        data["target_pitch"] = plan.target_pitch * 57.3;
        data["plan_yaw"] = plan.yaw * 57.3;
        data["plan_pitch"] = plan.pitch * 57.3;
        data["control"] = plan.control ? 1 : 0;
        data["fire"] = fire_cmd ? 1 : 0;
        // 必须 wrap：跨越 ±π 时不处理会出现 ±360° 跳变，调参时容易误判
        data["diff_yaw"] = limit_rad(plan.yaw - gs.yaw) * 57.3;
        data["diff_pitch"] = (plan.pitch - gs.pitch) * 57.3;

        if (target)
        {
            data["distance"] = std::hypot(target->ekf_x()[0], target->ekf_x()[2]);
            data["w"] = target->ekf_x()[7];
            data["target_z"] = target->ekf_x()[4];
            data["target_vz"] = target->ekf_x()[5];
        }

        plotter.plot(data);
    };

    // ---- 控制线程：MPC + 速度环反馈 + 开火门控，云台唯一写入者 ----
    auto control_thread = std::thread(
        [&]
        {
            auto last_mode = GimbalMode::IDLE;
            auto last_hik_target_t = long_ago;
            bool was_hik_active = false;
            bool was_hik_ok = true;
            int guide_log_count = 0;
            std::unordered_map<std::string, DetectionResult> usb_latest;

            // USB 引导指令：锁定/coast 都发 mode=1 + is_empty=false（视觉控 yaw、不开火）。
            // guide 无效时只发停止，不 reset_guide——内部可能正处于冷却，清掉会让它立刻重新锁定。
            auto apply_guide = [&](const Decider::Guide &guide)
            {
                if (!guide.valid)
                {
                    stop_gimbal();
                    return;
                }

                gimbal.send(true, false, guide.yaw, 0, 0, guide.pitch, 0, 0, false);
                if (++guide_log_count % 30 == 0)
                {
                    const char *phase =
                        guide.phase == Decider::GuidePhase::COAST ? "coast" : "lock";
                    logger()->info(
                        "[omni] USB guide {} {} yaw={:.1f}deg age={:.3f}s", phase, guide.camera,
                        guide.yaw * 57.3, guide.age);
                }
            };

            while (!quit)
            {
                auto mode = gimbal.mode();
                if (last_mode != mode)
                {
                    logger()->info("Switch to {}", gimbal.str(mode));
                    last_mode = mode;
                }

                // USB 搜索层：每拍必 drain，防止队列堆积。
                // 放在控制线程而不是视觉线程，海康掉线时这条链路仍然活着。
                auto usb_results = perceptron.get_detection_queue();
                for (const auto &dr : usb_results)
                {
                    if (dr.camera.empty())
                        continue;
                    auto &slot = usb_latest[dr.camera];
                    if (slot.timestamp <= dr.timestamp)
                        slot = dr;
                }

                auto target = target_queue.front();

                auto gs = gimbal.state();
                auto plan = planner.plan(target, gs.bullet_speed);

                // MPC 两次求解 + 22 次弹道求解要几 ms，反馈统一用求解后的最新姿态，
                // 否则位置环会带一个固定滞后
                gs = gimbal.state();
                const double yaw_err = limit_rad(plan.yaw - gs.yaw);
                const double pitch_err = plan.pitch - gs.pitch;

                // 电控只用了速度环（无位置闭环）：把位置环上移到视觉端。
                // 速度指令 = MPC 前馈速度 + 位置误差反馈，避免纯前馈导致的稳态角度偏差/漂移。
                // 注意：plan.yaw/pitch 取的是 HALF_HORIZON（命中时刻）的解，不是当前时刻，
                // 所以位置误差里天然含 LOS 角速度 ×(delay+fly) 的超前，调 kp 会同时改变超前量。
                double yaw_vel_cmd = plan.yaw_vel;
                double pitch_vel_cmd = plan.pitch_vel;
                bool fire_cmd = plan.fire;
                if (plan.control)
                {
                    constexpr double kp_yaw = 8.0;       // yaw 角度误差 -> 速度 增益 (rad/s)/rad
                    constexpr double kp_pitch = 12.0;    // pitch 角度误差 -> 速度 增益 (rad/s)/rad
                    constexpr double kMaxYawVel = 15.0;  // rad/s，云台能力上限的保守值
                    constexpr double kMaxPitchVel = 8.0; // rad/s
                    constexpr double kFireSlack = 1.0 / 57.3; // rad，点火门控的固定余量

                    yaw_vel_cmd += kp_yaw * yaw_err;
                    pitch_vel_cmd += kp_pitch * pitch_err;

                    // 切目标/误检时误差可达 180°，不限幅会给出远超云台能力的指令
                    yaw_vel_cmd = std::clamp(yaw_vel_cmd, -kMaxYawVel, kMaxYawVel);
                    pitch_vel_cmd = std::clamp(pitch_vel_cmd, -kMaxPitchVel, kMaxPitchVel);

                    // plan.fire 只比较 MPC 解与它自己的参考（planner.cpp:99），完全不含真实云台
                    // 姿态，静止目标实测恒为 true。这里再要求实际滞后不超过一阶跟随的稳态滞后，
                    // 否则说明云台还在追参考（切目标/被遮挡/发散），此时点火必然打飞。
                    // 模拟模式没有姿态回传（read_thread 不启动，state() 恒为初值），闭环永远建
                    // 不起来，门控会一直挡着开火，所以只在实机生效。
                    if (!gimbal.simulate())
                    {
                        const double yaw_lag_allow = std::abs(plan.yaw_vel) / kp_yaw + kFireSlack;
                        const double pitch_lag_allow =
                            std::abs(plan.pitch_vel) / kp_pitch + kFireSlack;
                        fire_cmd = plan.fire && std::abs(yaw_err) <= yaw_lag_allow &&
                                   std::abs(pitch_err) <= pitch_lag_allow;
                    }
                }

                const bool is_empty = !target.has_value();
                const auto now = std::chrono::steady_clock::now();

                // 海康回退迟滞：连续空 hik_lost_s 才交给 USB，避免两套目标反复抢控制
                if (target)
                    last_hik_target_t = now;
                const bool hik_active =
                    !is_empty || delta_time(now, last_hik_target_t) < hik_lost_s;
                if (hik_active != was_hik_active)
                {
                    was_hik_active = hik_active;
                    logger()->info("[omni] Hik target {}", hik_active ? "acquired" : "lost");
                }

                // 与「海康有画面但没目标」区分开，掉线是硬件/链路问题
                const bool hik_ok = hik_frame_ok.load();
                if (hik_ok != was_hik_ok)
                {
                    was_hik_ok = hik_ok;
                    if (!hik_ok)
                        logger()->warn("[omni] Hik unavailable (no frame)");
                }

                Decider::Guide guide;
                if (mode != GimbalMode::AUTO_AIM)
                {
                    decider.reset_guide();
                    stop_gimbal();
                }
                else
                {
                    // 每拍都调用：hik_active 时内部会清空锁定/coast 状态
                    guide = decider.arbitrate_usb(usb_results, gimbal, now, hik_active);
                    if (hik_active)
                    {
                        gimbal.send(
                            plan.control, fire_cmd, plan.yaw, yaw_vel_cmd, plan.yaw_acc, plan.pitch,
                            pitch_vel_cmd, plan.pitch_acc, is_empty);
                    }
                    else
                    {
                        apply_guide(guide);
                    }
                }

                publish_plot(hik_ok, hik_active, guide, target, plan, fire_cmd, is_empty);

                {
                    std::lock_guard<std::mutex> lock(snapshot_mutex);
                    snapshot.hik_active = hik_active;
                    snapshot.guide = guide;
                    snapshot.usb_latest = usb_latest;
                }

                std::this_thread::sleep_for(10ms);
            }
        });

    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit())
    {
        // 1) 海康精确链路：同步读取（q(t) 为该帧采集时刻的姿态）
        //    与 auto_aim_debug_mpc 唯一的差异：这里用带超时的读。阻塞 read 在海康掉线时
        //    会让视觉线程永久挂住，exiter 再也检查不到，Ctrl+C 也退不出去。
        //    超时时推一个空目标，控制线程看到的就是「海康没有目标」，
        //    按 hik_lost_s 迟滞后自动转 USB 引导。
        if (!hik.try_read(img, t, 100) || img.empty())
        {
            hik_frame_ok = false;
            target_queue.push(std::nullopt);
            continue;
        }
        hik_frame_ok = true;

        auto gs = gimbal.state();
        solver.set_R_gimbal2world(gimbal.q(t));

        // 2) 海康目标检测/跟踪：Tracker::track 内部会调用 solver.solve，不需要额外循环
        auto armors = yolo.detect(img);
        auto targets = tracker.track(armors, t, gs.id);

        if (targets.empty())
            target_queue.push(std::nullopt);
        else
            target_queue.push(targets.front());

        if (!display)
            continue;

        ControlSnapshot snap;
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex);
            snap = snapshot;
        }

        if (!targets.empty())
            draw_target_projection(img, targets.front(), solver, planner, gs.bullet_speed, t);

        if (show_debug_view(
                img, armors, snap.usb_latest, perceptron.latest_frames(), snap.guide,
                snap.hik_active, perceptron.detect_gain()))
            break;
    }

    quit = true;
    if (control_thread.joinable())
        control_thread.join();

    if (display)
        cv::destroyAllWindows();
    stop_gimbal();
    logger()->info("[omni_standard] exit");
    return 0;
}
