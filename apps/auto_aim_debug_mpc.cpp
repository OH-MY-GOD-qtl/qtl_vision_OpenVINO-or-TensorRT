// auto_aim_debug_mpc：YOLO + EKF 跟踪 + MPC 规划 + 串口云台，带重投影调试窗口
#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "camera/camera.hpp"
#include "comm/gimbal.hpp"
#include "detector/yolo.hpp"
#include "planner.hpp"
#include "solver.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tracker/tracker.hpp"

using namespace std::chrono_literals;

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/standard1.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
    Exiter exiter;
    Plotter plotter;

    cv::CommandLineParser cli(argc, argv, keys);
    auto config_path = cli.get<std::string>(0);
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }

    Gimbal gimbal(config_path);
    Camera camera(config_path);

    YOLO yolo(config_path, true);
    Solver solver(config_path);
    Tracker tracker(config_path, solver);
    Planner planner(config_path);

    ThreadSafeQueue<std::optional<Target>, true> target_queue(1);
    target_queue.push(std::nullopt);

    std::atomic<bool> quit = false;
    auto plan_thread = std::thread([&]() {
        auto t0 = std::chrono::steady_clock::now();

        while (!quit) {
            auto target = target_queue.front();
            auto gs = gimbal.state();
            auto plan = planner.plan(target, gs.bullet_speed);
            auto is_empty = target.has_value() ? false : true;

            // MPC 两次求解 + 22 次弹道求解要几 ms，反馈和绘图统一用求解后的最新姿态，
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
            if (plan.control) {
                constexpr double kp_yaw = 8.0;     // yaw 角度误差 -> 速度 增益 (rad/s)/rad
                constexpr double kp_pitch = 12.0;  // pitch 角度误差 -> 速度 增益 (rad/s)/rad
                constexpr double kMaxYawVel = 15.0;        // rad/s，云台能力上限的保守值
                constexpr double kMaxPitchVel = 8.0;       // rad/s
                constexpr double kFireSlack = 1.0 / 57.3;  // rad，点火门控的固定余量

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
                if (!gimbal.simulate()) {
                    const double yaw_lag_allow = std::abs(plan.yaw_vel) / kp_yaw + kFireSlack;
                    const double pitch_lag_allow =
                        std::abs(plan.pitch_vel) / kp_pitch + kFireSlack;
                    fire_cmd = plan.fire && std::abs(yaw_err) <= yaw_lag_allow &&
                               std::abs(pitch_err) <= pitch_lag_allow;
                }
            }

            gimbal.send(
                plan.control, fire_cmd, plan.yaw, yaw_vel_cmd, plan.yaw_acc, plan.pitch,
                pitch_vel_cmd, plan.pitch_acc, is_empty);

            nlohmann::json data;
            data["t"] = delta_time(std::chrono::steady_clock::now(), t0);

            data["gimbal_yaw"] = gs.yaw * 57.2958;  // 世界坐标系中云台当前yaw角度（单位：度）
            //data["gimbal_yaw_vel"] = gs.yaw_vel;
            data["gimbal_pitch"] = gs.pitch * 57.2958;  // 世界坐标系中云台当前pitch角度（单位：度）
            //data["gimbal_pitch_vel"] = gs.pitch_vel;

            data["target_yaw"] =
                plan.target_yaw * 57.2958;  //目标在世界坐标系中的yaw角度（单位：度）
            data["target_pitch"] =
                plan.target_pitch * 57.2958;  //目标在世界坐标系中的pitch角度（单位：度）

            data["plan_yaw"] = plan.yaw * 57.2958;  // 期望在世界坐标系中的yaw角度（单位：度）
            // data["plan_yaw_vel"] = plan.yaw_vel;
            // data["plan_yaw_acc"] = plan.yaw_acc;

            data["plan_pitch"] = plan.pitch * 57.2958;  // 期望在世界坐标系中的pitch角度（单位：度）
            // data["plan_pitch_vel"] = plan.pitch_vel;
            // data["plan_pitch_acc"] = plan.pitch_acc;

            data["fire"] = fire_cmd ? 1 : 0;
            data["is_empty"] = is_empty ? 1 : 0;
            data["bullet_speed"] = gs.bullet_speed;
            data["robot_id"] = gs.id;
            data["mode"] = plan.control ? (fire_cmd ? 2 : 1) : 0;
            if (target.has_value()) {
                data["target_z"] = target->ekf_x()[4];   //z目标在世界坐标系中的z轴位置坐标
                data["target_vz"] = target->ekf_x()[5];  //vz目标在z轴方向的速度分量
            }

            if (target.has_value()) {
                data["w"] = target->ekf_x()[7];
            } else {
                data["w"] = 0.0;
            }

            // 必须 wrap：跨越 ±π 时不处理会出现 ±360° 跳变，调参时容易误判
            float diff_yaw = limit_rad(plan.yaw - gs.yaw) * 57.2958;
            float diff_pitch = (plan.pitch - gs.pitch) * 57.2958;
            data["diff_yaw"] = diff_yaw;
            data["diff_pitch"] = diff_pitch;
            data["distance"] =
                target.has_value() ? std::hypot(target->ekf_x()[0], target->ekf_x()[2]) : 0.0;

            plotter.plot(data);

            std::this_thread::sleep_for(10ms);
        }
    });

    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
        camera.read(img, t);
        auto q = gimbal.q(t);

        solver.set_R_gimbal2world(q);
        auto armors = yolo.detect(img);
        auto targets = tracker.track(armors, t, gimbal.state().id);
        if (!targets.empty())
            target_queue.push(targets.front());
        else
            target_queue.push(std::nullopt);

        if (!targets.empty()) {  //绘制目标投影
            auto target = targets.front();

            // 当前帧target更新后
            std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
            for (const Eigen::Vector4d & xyza : armor_xyza_list) {
                auto image_points =
                    solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
                draw_points(img, image_points, {0, 255, 0});
            }

            // 红框：按当前帧时间戳重新计算预测点，与绿框同帧对齐（原 planner.debug_xyza
            // 由计划线程异步写入，存在数据竞争且与显示帧率不同步，导致运动顿挫）
            auto aim_xyza = planner.predict_xyza(target, gimbal.state().bullet_speed, t);
            auto image_points = solver.reproject_armor(
                aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
            draw_points(img, image_points, {0, 0, 255});
        }

        cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
        cv::imshow("reprojection", img);
        auto key = cv::waitKey(1);
        if (key == 'q') break;
    }

    quit = true;
    if (plan_thread.joinable()) plan_thread.join();
    gimbal.send(false, false, 0, 0, 0, 0, 0, 0, true);

    return 0;
}