// omni_standard：全向感知自瞄（3×USB 传统方位搜索 + 1×海康 YOLO 精确主瞄准），无显示窗口
//
// 架构要点（与 omnidirectional_sensing 的异步方案的关键区别）：
//  - 云台只有一个写入者（本主循环），不存在 200Hz 异步命令线程复用旧时间戳导致乱晃
//  - 海康优先：海康锁定 -> 由 planner 精确输出；海康无目标 -> USB 引导 yaw + pitch 预设
//  - USB 结果带采集时间戳，Decider 用 gimbal.q(采集时刻) 做时间对齐 + 老化/限速/低通
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "camera/camera.hpp"
#include "camera/usbcamera.hpp"
#include "comm/gimbal.hpp"
#include "detector/yolo.hpp"
#include "omni/decider.hpp"
#include "omni/perceptron.hpp"
#include "planner/planner.hpp"
#include "solve/solver.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"
#include "tracker/tracker.hpp"

const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>("@config-path");
    if (cli.has("help") || !cli.has("@config-path")) {
        cli.printMessage();
        return 0;
    }

    Exiter exiter;

    // ---- USB 设备名（顺序对应 usb0/usb1/usb2，与 omni_usb_yaw_offsets 索引一致）----
    auto yaml = yaml_load(config_path);
    std::vector<std::string> usb_devices = {"video0", "video2", "video4"};
    if (yaml["omni_usb_devices"]) {
        usb_devices.clear();
        for (const auto & v : yaml["omni_usb_devices"]) usb_devices.push_back(v.as<std::string>());
    }

    // ---- 海康精确链路 ----
    Gimbal gimbal(config_path);
    Camera hik(config_path);
    YOLO yolo(config_path, true);
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Planner planner(config_path);
    Decider decider(config_path);

    // ---- 3 路 USB 方位搜索（传统检测，不加载分类器）----
    std::array<std::unique_ptr<USBCamera>, 3> usb_cams;
    for (size_t i = 0; i < usb_cams.size() && i < usb_devices.size(); ++i)
        usb_cams[i] = std::make_unique<USBCamera>(usb_devices[i], config_path);
    Perceptron perceptron(
        usb_cams[0].get(), usb_cams[1].get(), usb_cams[2].get(), decider, config_path);

    logger()->info("[omni_standard] started, config: {}", config_path);

    cv::Mat img;
    std::chrono::steady_clock::time_point t;
    auto last_mode = GimbalMode::IDLE;
    int guide_log_count = 0;

    while (!exiter.exit()) {
        auto mode = gimbal.mode();
        if (last_mode != mode) {
            logger()->info("Switch to {}", gimbal.str(mode));
            last_mode = mode;
        }

        // 1) 海康精确链路：同步读取 + 帧时间戳对齐（q(t) 为该帧采集时刻的姿态）
        hik.read(img, t);
        if (img.empty()) continue;

        auto gs = gimbal.state();
        solver.set_R_gimbal2world(gimbal.q(t));

        // 2) USB 搜索层：每轮必 drain，防止队列堆积
        auto usb_results = perceptron.get_detection_queue();

        if (mode != GimbalMode::AUTO_AIM) {
            gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true);
            continue;
        }

        // 3) 海康目标检测/跟踪/规划
        auto armors = yolo.detect(img);
        for (auto & armor : armors) solver.solve(armor);
        auto targets = tracker.track(armors, t, gs.id);

        std::optional<Target> target;
        if (!targets.empty()) target = targets.front();

        auto plan = planner.plan(target, gs.bullet_speed);
        bool is_empty = !target.has_value();

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
            // 海康无目标 -> USB 方位引导（时间对齐 + 老化/限速/低通在 Decider 内完成）
            auto now = std::chrono::steady_clock::now();
            auto guide = decider.arbitrate_usb(usb_results, gimbal, now);
            if (guide.valid) {
                gimbal.send(true, false, guide.yaw, 0, 0, guide.pitch, 0, 0, false);
                if (++guide_log_count % 30 == 0)
                    logger()->info(
                        "[omni] USB guide {} yaw={:.1f}deg age={:.3f}s", guide.camera,
                        guide.yaw * 57.3, guide.age);
            } else {
                gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true);
            }
        }
    }

    gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true);
    logger()->info("[omni_standard] exit");
    return 0;
}
