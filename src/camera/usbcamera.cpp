#include "camera/usbcamera.hpp"

#include <filesystem>
#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

std::string USBCamera::resolve_device_path(const std::string &device)
{
    if (device.rfind("/dev/", 0) == 0)
        return device;
    if (!device.empty() && device.front() == '/')
        return device;
    return "/dev/" + device;
}

// 把 /dev/videoN 映射到 /dev/v4l/by-id 或 by-path 的稳定别名。
// 相机缺少序列号时只有 by-path 可靠（同一物理 USB 口重插后路径不变）。
std::string USBCamera::find_stable_device_path(
    const std::string &device, const std::vector<std::string> &search_dirs)
{
    if (device.empty())
        return {};
    const std::filesystem::path node = device.front() == '/' ? device : "/dev/" + device;
    if (node.filename().string().rfind("video", 0) != 0)
        return {};

    std::error_code ec;
    const auto canonical_node = std::filesystem::canonical(node, ec);
    if (ec)
        return {};

    for (const auto &dir : search_dirs)
    {
        if (!std::filesystem::exists(dir))
            continue;
        for (const auto &entry : std::filesystem::directory_iterator(dir, ec))
        {
            if (ec)
                break;
            if (entry.path().filename().string().find("-video-index0") == std::string::npos)
                continue;

            std::error_code entry_ec;
            if (std::filesystem::canonical(entry.path(), entry_ec) == canonical_node)
                return entry.path().string();
        }
    }
    return {};
}

bool USBCamera::needs_alias_resolution(const std::string &device)
{
    if (device.empty())
        return false;
    const std::filesystem::path node = device.front() == '/' ? device : "/dev/" + device;
    // 只有 /dev/videoN 会随枚举顺序变化；by-id/by-path 等显式路径本身已经稳定
    return node.parent_path() == "/dev" && node.filename().string().rfind("video", 0) == 0;
}

USBCamera::USBCamera(const std::string &open_name, const std::string &config_path)
    : open_name_(open_name), open_count_(0), queue_(1)
{
    auto yaml = yaml_load(config_path);
    image_width_ = yaml_read<double>(yaml, "image_width");
    image_height_ = yaml_read<double>(yaml, "image_height");
    usb_frame_rate_ = yaml_read<double>(yaml, "usb_frame_rate");

    // videoN 会随枚举顺序变化：解析成 by-id/by-path；显式路径（by-id/by-path）原样使用
    if (needs_alias_resolution(open_name_))
    {
        resolved_device_path_ = find_stable_device_path(open_name_);
        has_stable_alias_ = !resolved_device_path_.empty();
        if (resolved_device_path_.empty())
            logger()->warn(
                "[USBCamera] {} has no stable by-id/by-path alias yet; will re-resolve on retry",
                open_name_);
        else
            logger()->info("[USBCamera] {} -> {}", open_name_, resolved_device_path_);
    }
    else
    {
        resolved_device_path_ = resolve_device_path(open_name_);
        logger()->info("[USBCamera] explicit path {}", resolved_device_path_);
    }

    try_open();

    // 守护线程：相机缺失/拔插时持续重试，不永久放弃
    daemon_thread_ =
        std::thread{[this]
                    {
                        while (!quit_)
                        {
                            std::this_thread::sleep_for(100ms);

                            if (ok_)
                                continue;

                            if (capture_thread_.joinable())
                                capture_thread_.join();

                            {
                                std::lock_guard<std::mutex> lock(cap_mutex_);
                                close();
                            }
                            try_open();

                            if (!ok_ && open_count_ % 50 == 0)
                                logger()->warn(
                                    "[USBCamera] {} unavailable, retrying ({} attempts)",
                                    open_name_, open_count_.load());
                        }
                    }};
}

USBCamera::~USBCamera()
{
    quit_ = true;
    {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        close();
    }
    if (daemon_thread_.joinable())
        daemon_thread_.join();
    if (capture_thread_.joinable())
        capture_thread_.join();
    logger()->info("USBCamera destructed.");
}

cv::Mat USBCamera::read()
{
    std::lock_guard<std::mutex> lock(cap_mutex_);
    if (!cap_.isOpened())
    {
        logger()->warn("Failed to read {} USB camera", this->device_name);
        return cv::Mat();
    }
    cap_ >> img_;
    return img_;
}

void USBCamera::read(cv::Mat &img, std::chrono::steady_clock::time_point &timestamp)
{
    CameraData data;
    queue_.pop(data);

    img = data.img;
    timestamp = data.timestamp;
}

bool USBCamera::try_read(
    cv::Mat &img, std::chrono::steady_clock::time_point &timestamp, int timeout_ms)
{
    CameraData data;
    // 队列由采集线程 push；相机缺失/掉线时无人 push，这里超时返回 false 而不是永久阻塞
    if (!queue_.try_pop(data, timeout_ms))
        return false;

    img = data.img;
    timestamp = data.timestamp;
    return true;
}

bool USBCamera::open()
{
    std::lock_guard<std::mutex> lock(cap_mutex_);

    // 别名会随拔插失效（换物理口、by-id 名字变化）：每次重试都检查一次。
    // 曾解析到过稳定别名的相机，别名消失期间不回退到 videoN —— 那个编号可能已属于另一台相机。
    if (needs_alias_resolution(open_name_))
    {
        // 别名还在就直接用；失效了才重新解析。旧值保留：同口重插后它可能原样恢复。
        if (resolved_device_path_.empty() || !std::filesystem::exists(resolved_device_path_))
        {
            auto refreshed = find_stable_device_path(open_name_);
            if (!refreshed.empty() && refreshed != resolved_device_path_)
            {
                logger()->info("[USBCamera] {} -> {}", open_name_, refreshed);
                resolved_device_path_ = refreshed;
                has_stable_alias_ = true;
            }
        }
        if (resolved_device_path_.empty() && has_stable_alias_)
            return false; // 相机不在：等守护线程下一轮重试
    }

    std::string true_device_name =
        resolved_device_path_.empty() ? resolve_device_path(open_name_) : resolved_device_path_;
    cap_.open(true_device_name, cv::CAP_V4L);
    if (!cap_.isOpened())
    {
        // 不在每次重试时刷 warn，由守护线程限频输出
        return false;
    }
    device_name = true_device_name;
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
    cap_.set(cv::CAP_PROP_FPS, usb_frame_rate_);

    // V4L 可能静默给出最接近的分辨率，这里打印实际协商结果，避免配了 320x240 实际却是别的尺寸
    logger()->info(
        "[USBCamera] {} opened: {}x{} @ {:.1f} fps (requested {}x{} @ {})", device_name,
        static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)), cap_.get(cv::CAP_PROP_FPS),
        static_cast<int>(image_width_), static_cast<int>(image_height_), usb_frame_rate_);

    // 取图线程
    capture_thread_ = std::thread{
        [this]
        {
            ok_ = true;
            std::this_thread::sleep_for(50ms);
            logger()->info("[{} USB camera] capture thread started ", this->device_name);
            while (!quit_)
            {
                std::this_thread::sleep_for(1ms);

                cv::Mat img;
                bool success;
                {
                    std::lock_guard<std::mutex> lock(cap_mutex_);
                    if (!cap_.isOpened())
                    {
                        break;
                    }
                    success = cap_.read(img);
                }

                if (!success)
                {
                    logger()->warn("Failed to read frame, exiting capture thread");
                    break;
                }

                // 已恢复取流：重置连续失败计数，避免累计到上限后永久放弃
                open_count_ = 0;

                auto timestamp = std::chrono::steady_clock::now();
                queue_.push({img, timestamp});
            }
            ok_ = false;
        }};
    return true;
}

void USBCamera::try_open()
{
    try
    {
        open();
        open_count_++;
    }
    catch (const std::exception &e)
    {
        logger()->warn("{}", e.what());
    }
}

void USBCamera::close()
{
    if (cap_.isOpened())
    {
        cap_.release();
        // 拔插重试期间避免每次 info 刷屏
        logger()->debug("[USBCamera] {} released", device_name);
    }
}
