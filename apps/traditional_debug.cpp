// traditional_debug：传统视觉检测 + MPC 规划 + 串口云台，带检测调试窗口
#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "aimer.hpp"
#include "camera/camera.hpp"
#include "detector/detector.hpp" // 使用传统检测器
#include "planner.hpp"
#include "shooter.hpp"
#include "solver.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tracker/tracker.hpp"

const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | yaml配置文件路径 }"
    "{d diag | | 逐帧诊断：检测/跟踪判据写入 logs/flicker_diag.csv }";

using namespace std::chrono_literals;

int main(int argc, char *argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>("@config-path");
    if (cli.has("help") || !cli.has("@config-path"))
    {
        cli.printMessage();
        return 0;
    }

    // 逐帧诊断：每帧落一行，记录检测数、跟踪状态和每块装甲板的判据，
    // 用于区分“检测层抖”（n_armors 变）还是“匹配层抖”（n_armors 不变但 state 变）
    std::ofstream diag_csv;
    if (cli.has("diag"))
    {
        diag_csv.open("logs/flicker_diag.csv", std::ios::out | std::ios::trunc);
        diag_csv << "frame,t_ms,dt_ms,n_armors,state,target,armors\n";
    }

    Exiter exiter;
    Recorder recorder;
    Plotter plotter; // UDP 127.0.0.1:9870，供 PlotJuggler 显示

    Gimbal gimbal(config_path);
    Camera camera(config_path);

    Detector detector(config_path, true); // 使用传统检测器替代YOLO
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Planner planner(config_path);

    ThreadSafeQueue<std::optional<Target>, true> target_queue(1);
    target_queue.push(std::nullopt);

    std::atomic<bool> quit = false;

    std::atomic<GimbalMode> mode{GimbalMode::IDLE};
    auto last_mode{GimbalMode::IDLE};
    // 冻结入口：只为消除模式切换瞬间的双写者窗口，不改变架构（见 issue #6）
    std::mutex send_mutex;

    auto plan_thread = std::thread(
        [&]()
        {
            auto t0 = std::chrono::steady_clock::now();
            auto last_fps_time = std::chrono::steady_clock::now();
            int frame_count = 0;

            while (!quit)
            {
                if (!target_queue.empty() && mode == GimbalMode::AUTO_AIM)
                {
                    auto target = target_queue.front();
                    auto gs = gimbal.state();
                    auto plan = planner.plan(target, gs.bullet_speed);
                    auto is_empty = target.has_value() ? false : true;

                    {
                        std::lock_guard<std::mutex> lock(send_mutex);
                        if (mode != GimbalMode::AUTO_AIM)
                            continue;
                        gimbal.send(
                            plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc,
                            plan.pitch, plan.pitch_vel, plan.pitch_acc, is_empty);
                    }
                    nlohmann::json data;

                    // 计算帧率
                    frame_count++;
                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       current_time - last_fps_time)
                                       .count();
                    double fps = 0.0;
                    if (elapsed >= 1000)
                    {
                        fps = frame_count * 1000.0 / elapsed;
                        frame_count = 0;
                        last_fps_time = current_time;
                    }

                    data["t"] = delta_time(std::chrono::steady_clock::now(), t0);
                    data["control"] = plan.control;
                    data["fire"] = plan.fire;
                    data["plan_yaw"] = plan.yaw * 57.3;
                    data["gimbal_yaw"] = gs.yaw * 57.3;
                    data["plan_pitch"] = plan.pitch * 57.3;
                    data["gimbal_pitch"] = gs.pitch * 57.3;
                    data["is_empty"] = is_empty;
                    data["bullet_speed"] = gs.bullet_speed;
                    data["fps"] = fps; // 在data中添加帧率信息

                    plotter.plot(data);

                    std::this_thread::sleep_for(10ms);
                }
                else
                    std::this_thread::sleep_for(200ms);
            }
        });

    cv::Mat img;
    std::chrono::steady_clock::time_point t;
    std::chrono::steady_clock::time_point prev_t;
    const auto t_start = std::chrono::steady_clock::now();
    std::size_t frame_idx = 0;
    bool show_raw = true; // 'r' 切换原始检测框（灰色细框）

    while (!exiter.exit())
    {
        {
            std::lock_guard<std::mutex> lock(send_mutex);
            mode = gimbal.mode();
        }

        if (last_mode != mode)
        {
            logger()->info("Switch to {}", gimbal.str(mode));
            last_mode = mode.load();
        }

        camera.read(img, t);
        // 帧间隔：与 Tracker 内部同口径，用于判断是否触发 dt>0.1s 的强制 lost
        const double dt_ms = (prev_t.time_since_epoch().count() == 0)
                                 ? 0.0
                                 : std::chrono::duration<double, std::milli>(t - prev_t).count();
        prev_t = t;
        auto q = gimbal.q(t);
        solver.set_R_gimbal2world(q);

        // recorder.record(img, q, t);

        /// 自瞄
        if (mode.load() == GimbalMode::AUTO_AIM)
        {
            auto armors = detector.detect(img); // 使用传统检测方法

            // 先跟踪、再画框：tracker 内部会按敌方颜色过滤 armors，且只有它确认过的目标
            // （tracking / temp_lost）才画亮框。原始检测只画细暗框，避免把检测抖动误读成
            // 跟踪状态抖动（排查“闪烁”时按 'r' 可关掉原始框）。
            auto targets = tracker.track(armors, t, gimbal.state().id);
            if (!targets.empty())
                target_queue.push(targets.front());
            else
                target_queue.push(std::nullopt);

            const bool has_target = !targets.empty();

            for (const auto &armor : armors)
            {
                if (armor.points.size() != 4)
                    continue;

                const bool locked = has_target && armor.name == targets.front().name &&
                                    armor.type == targets.front().armor_type;
                if (!locked && !show_raw)
                    continue;

                const auto color = locked ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 170, 255);
                const int thickness = locked ? 2 : 1;

                for (int i = 0; i < 4; i++)
                    cv::line(
                        img, cv::Point(armor.points[i].x, armor.points[i].y),
                        cv::Point(armor.points[(i + 1) % 4].x, armor.points[(i + 1) % 4].y), color,
                        thickness);
            }

            // HUD：跟踪状态 + 过滤后的检测数，用于区分“检测掉”还是“状态掉”
            cv::putText(
                img, fmt::format("{} armors={} raw={}", tracker.state(), armors.size(), show_raw),
                {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 255}, 2);

            if (diag_csv.is_open())
            {
                // 每块板：name/type/conf/ratio/side_ratio/rect_deg/cx/cy
                std::string desc;
                for (const auto &armor : armors)
                {
                    if (!desc.empty())
                        desc += ';';
                    desc += fmt::format(
                        "{}/{}/{:.2f}/{:.2f}/{:.2f}/{:.1f}/{:.0f}/{:.0f}", ARMOR_NAMES[armor.name],
                        ARMOR_TYPES[armor.type], armor.confidence, armor.ratio, armor.side_ratio,
                        armor.rectangular_error * 57.3, armor.center.x, armor.center.y);
                }
                const auto target_desc = targets.empty()
                                             ? std::string("-")
                                             : fmt::format(
                                                   "{}/{}", ARMOR_NAMES[targets.front().name],
                                                   ARMOR_TYPES[targets.front().armor_type]);
                diag_csv << frame_idx << ','
                         << std::chrono::duration<double, std::milli>(t - t_start).count() << ','
                         << dt_ms << ',' << armors.size() << ',' << tracker.state() << ','
                         << target_desc << ',' << desc << '\n';
                if (frame_idx % 30 == 0)
                    diag_csv.flush();
            }
        }

        else
        {
            std::lock_guard<std::mutex> lock(send_mutex);
            if (mode != GimbalMode::AUTO_AIM)
                gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
        }

        // 显示检测窗口
        cv::resize(img, img, {}, 0.5, 0.5); // 缩小图片尺寸以便显示
        cv::imshow("Detection", img);
        auto key = cv::waitKey(1);
        if (key == 'q')
            break;
        if (key == 'r')
            show_raw = !show_raw;
        frame_idx++;
    }

    quit = true;
    if (plan_thread.joinable())
        plan_thread.join();
    gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    cv::destroyAllWindows(); // 关闭窗口

    return 0;
}