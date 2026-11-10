#include "omni/perceptron.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "detector/yolo.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

Perceptron::Perceptron(
    USBCamera *usbcam1, USBCamera *usbcam2, USBCamera *usbcam3, const Decider &decider,
    const std::string &config_path)
    : detection_queue_(30), decider_(decider), stop_flag_(false)
{
    auto yaml = yaml_load(config_path);
    const bool use_tensorrt =
        yaml["omni_usb_use_tensorrt"] ? yaml["omni_usb_use_tensorrt"].as<bool>() : false;
    const double detect_fps =
        yaml["omni_usb_detect_fps"] ? yaml["omni_usb_detect_fps"].as<double>() : 15.0;
    detect_interval_s_ = detect_fps > 0 ? 1.0 / detect_fps : 0.0;

    // 送检前亮度缩放：相机无曝光控件时的软件兜底，只影响检测，不影响调试显示
    const double detect_gain =
        yaml["omni_usb_detect_gain"] ? yaml["omni_usb_detect_gain"].as<double>() : 1.0;
    detect_gain_ = (detect_gain > 0.0 && detect_gain <= 1.0) ? detect_gain : 1.0;
    if (detect_gain_ != detect_gain)
        logger()->warn(
            "[omni] omni_usb_detect_gain={} out of (0, 1]; falling back to 1.0", detect_gain);

    // 结果间隔比老化阈值还长时，Decider 会把每条结果都当过期丢掉（表现为永远锁不上）
    const double max_age_s =
        yaml["omni_usb_max_age_s"] ? yaml["omni_usb_max_age_s"].as<double>() : 0.15;
    if (detect_interval_s_ > max_age_s)
        logger()->warn(
            "[omni] omni_usb_detect_fps={:.1f} -> {:.3f}s interval exceeds "
            "omni_usb_max_age_s={:.3f}s; USB results will always be stale",
            detect_fps, detect_interval_s_, max_age_s);

    // USB 搜索链路只需要检测：
    //   - 不加载独立数字分类器（编号由检测器的分类头直接给出）
    //   - 不做传统角点二次矫正（那是海康精确链路的行为）
    //   - 默认走 OpenVINO，即使海康配置 use_tensorrt=true
    YOLOOptions options;
    options.debug = false;
    options.use_tensorrt = use_tensorrt;
    options.use_traditional = false;
    options.use_classifier = false;
    for (auto &detector : detectors_)
        detector = std::make_unique<YOLO>(config_path, options);

    // 相机标识 usb0/usb1/usb2 与 omni_usb_cameras 索引一一对应，由 index 派生
    threads_.emplace_back([this, usbcam1]
                          { search_thread(usbcam1, 0); });
    threads_.emplace_back([this, usbcam2]
                          { search_thread(usbcam2, 1); });
    threads_.emplace_back([this, usbcam3]
                          { search_thread(usbcam3, 2); });

    logger()->info(
        "Perceptron initialized (3 USB YOLO, {:.0f} fps, tensorrt={}, classifier=off, "
        "traditional=off, detect_gain={:.2f}).",
        detect_fps, use_tensorrt, detect_gain_);
}

Perceptron::~Perceptron()
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_flag_ = true; // 设置退出标志
    }
    condition_.notify_all(); // 唤醒所有等待的线程

    // 等待线程结束
    for (auto &t : threads_)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    logger()->info("Perceptron destructed.");
}

std::vector<DetectionResult> Perceptron::get_detection_queue()
{
    std::vector<DetectionResult> result;
    DetectionResult temp;

    // 注意：这里的 pop 不阻塞（队列为空时直接返回空 vector）
    while (!detection_queue_.empty())
    {
        detection_queue_.pop(temp);
        result.push_back(std::move(temp));
    }

    return result;
}

std::array<cv::Mat, 3> Perceptron::latest_frames() const
{
    std::lock_guard<std::mutex> lock(frames_mutex_);
    return latest_frames_;
}

// USB 搜索线程
void Perceptron::search_thread(USBCamera *cam, int index)
{
    const std::string camera_key = "usb" + std::to_string(index);
    if (!cam)
    {
        logger()->error("USB camera pointer is null ({})!", camera_key);
        return;
    }

    // 超时读取（而非阻塞 read）：相机缺失 / 运行中拔线 / usbipd 掉线时，
    // 采集线程不会 push 数据，这里必须靠超时返回，否则析构 join() 会永久死锁。
    constexpr int read_timeout_ms = 100;
    int timeout_count = 0;
    bool unavailable = false;
    std::chrono::steady_clock::time_point last_detect_t{};

    try
    {
        while (true)
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (stop_flag_)
                    break; // 检查是否需要退出
            }

            cv::Mat img;
            std::chrono::steady_clock::time_point ts;
            if (!cam->try_read(img, ts, read_timeout_ms))
            {
                // 连续 ~1s 无帧才报一次"暂不可用"，恢复时再报一次，避免刷屏
                if (++timeout_count >= 10 && !unavailable)
                {
                    unavailable = true;
                    logger()->warn(
                        "[omni] {} temporarily unavailable (no frame for ~{}ms)", camera_key,
                        timeout_count * read_timeout_ms);
                }
                continue;
            }
            if (unavailable)
            {
                unavailable = false;
                logger()->info("[omni] {} recovered", camera_key);
            }
            timeout_count = 0;

            if (img.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // 送检图像：过曝时先压暗。调试显示也用它，保证看到的就是检测器看到的画面
            cv::Mat detect_img;
            if (detect_gain_ < 0.999)
                img.convertTo(detect_img, -1, detect_gain_, 0);
            else
                detect_img = img;

            {
                std::lock_guard<std::mutex> lock(frames_mutex_);
                latest_frames_[index] = detect_img;
            }

            const auto now = std::chrono::steady_clock::now();
            if (detect_interval_s_ > 0 && last_detect_t != std::chrono::steady_clock::time_point{} &&
                std::chrono::duration<double>(now - last_detect_t).count() < detect_interval_s_)
                continue;
            last_detect_t = now;

            auto armors = detectors_[index]->detect(detect_img);
            decider_.filter_enemy_color(armors);
            if (armors.empty())
                continue;

            // 表观宽度统一用四点几何，YOLO/传统路径通用
            auto armor_width = [](const Armor &armor)
            {
                if (armor.points.size() != 4)
                    return 0.0;
                return std::max(
                    cv::norm(armor.points[0] - armor.points[1]),
                    cv::norm(armor.points[3] - armor.points[2]));
            };

            int selected_index = -1;
            double best_width = 0.0;
            for (int i = 0; i < static_cast<int>(armors.size()); ++i)
            {
                const double width = armor_width(armors[i]);
                if (width > best_width)
                {
                    best_width = width;
                    selected_index = i;
                }
            }
            if (selected_index < 0 || img.cols <= 0)
                continue;

            auto delta_angle = decider_.delta_angle(armors[selected_index], camera_key);

            DetectionResult dr;
            dr.armors = std::move(armors);
            dr.timestamp = ts; // 采集时刻，供上层做时间对齐
            dr.delta_yaw = delta_angle[0] / 57.3;
            dr.delta_pitch = delta_angle[1] / 57.3;
            dr.source = DetectionSource::USB;
            dr.camera = camera_key;
            dr.selected_index = selected_index;
            dr.size_score = best_width / static_cast<double>(img.cols);
            detection_queue_.push(dr);
        }
    }
    catch (const std::exception &e)
    {
        logger()->error("Exception in search_thread({}): {}", camera_key, e.what());
    }
}
