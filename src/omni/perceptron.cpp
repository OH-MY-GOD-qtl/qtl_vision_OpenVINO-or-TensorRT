#include "omni/perceptron.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "tools/logger.hpp"

Perceptron::Perceptron(
    USBCamera * usbcam1, USBCamera * usbcam2, USBCamera * usbcam3, const Decider & decider,
    const std::string & config_path)
: detection_queue_(30), decider_(decider), stop_flag_(false)
{
    // 三路 USB 走传统检测、不加载分类器（只报“有无目标 + 方位”）
    detector1_ = std::make_shared<Detector>(config_path, false, false);
    detector2_ = std::make_shared<Detector>(config_path, false, false);
    detector3_ = std::make_shared<Detector>(config_path, false, false);

    // 相机标识 usb0/usb1/usb2 与 omni_usb_yaw_offsets 索引一一对应
    threads_.emplace_back([this, usbcam1] { search_thread(usbcam1, *detector1_, "usb0", 0); });
    threads_.emplace_back([this, usbcam2] { search_thread(usbcam2, *detector2_, "usb1", 1); });
    threads_.emplace_back([this, usbcam3] { search_thread(usbcam3, *detector3_, "usb2", 2); });

    logger()->info("Perceptron initialized (3 USB traditional search).");
}

Perceptron::~Perceptron()
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_flag_ = true;  // 设置退出标志
    }
    condition_.notify_all();  // 唤醒所有等待的线程

    // 等待线程结束
    for (auto & t : threads_) {
        if (t.joinable()) {
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
    while (!detection_queue_.empty()) {
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

// USB 传统检测线程
void Perceptron::search_thread(
    USBCamera * cam, Detector & detector, const std::string & camera_key, int index)
{
    if (!cam) {
        logger()->error("USB camera pointer is null ({})!", camera_key);
        return;
    }

    // 超时读取（而非阻塞 read）：相机缺失 / 运行中拔线 / usbipd 掉线时，
    // 采集线程不会 push 数据，这里必须靠超时返回，否则析构 join() 会永久死锁。
    constexpr int read_timeout_ms = 100;
    int timeout_count = 0;
    bool unavailable = false;

    try {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (stop_flag_) break;  // 检查是否需要退出
            }

            cv::Mat img;
            std::chrono::steady_clock::time_point ts;
            if (!cam->try_read(img, ts, read_timeout_ms)) {
                // 连续 ~1s 无帧才报一次"暂不可用"，恢复时再报一次，避免刷屏
                if (++timeout_count >= 10 && !unavailable) {
                    unavailable = true;
                    logger()->warn(
                        "[omni] {} temporarily unavailable (no frame for ~{}ms)", camera_key,
                        timeout_count * read_timeout_ms);
                }
                continue;
            }
            if (unavailable) {
                unavailable = false;
                logger()->info("[omni] {} recovered", camera_key);
            }
            timeout_count = 0;

            if (img.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(frames_mutex_);
                latest_frames_[index] = img;
            }

            auto armors = detector.detect(img);
            decider_.filter_enemy_color(armors);
            if (armors.empty()) continue;

            // 传统 Armor 的 type/name/box 未初始化，只能用 points 算表观宽度
            auto armor_width = [](const Armor & armor) {
                if (armor.points.size() != 4) return 0.0;
                return std::max(
                    cv::norm(armor.points[0] - armor.points[1]),
                    cv::norm(armor.points[3] - armor.points[2]));
            };

            int selected_index = -1;
            double best_width = 0.0;
            for (int i = 0; i < static_cast<int>(armors.size()); ++i) {
                const double width = armor_width(armors[i]);
                if (width > best_width) {
                    best_width = width;
                    selected_index = i;
                }
            }
            if (selected_index < 0 || img.cols <= 0) continue;

            auto delta_angle = decider_.delta_angle(armors[selected_index], camera_key);

            DetectionResult dr;
            dr.armors = std::move(armors);
            dr.timestamp = ts;  // 采集时刻，供上层做时间对齐
            dr.delta_yaw = delta_angle[0] / 57.3;
            dr.delta_pitch = delta_angle[1] / 57.3;
            dr.source = DetectionSource::USB;
            dr.camera = camera_key;
            dr.selected_index = selected_index;
            dr.size_score = best_width / static_cast<double>(img.cols);
            detection_queue_.push(dr);
        }
    } catch (const std::exception & e) {
        logger()->error("Exception in search_thread({}): {}", camera_key, e.what());
    }
}
