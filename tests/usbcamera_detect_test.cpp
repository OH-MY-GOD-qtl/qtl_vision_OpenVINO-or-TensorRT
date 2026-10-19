#include <opencv2/opencv.hpp>

#include <array>
#include <chrono>
#include <memory>
#include <string>

#include "camera/usbcamera.hpp"
#include "detector/yolo.hpp"
#include "tools/img_tools.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/omni.yaml      | yaml配置文件路径}"
    "{@camera1       | video0                 | 第一路 USB 相机设备名}"
    "{@camera2       | video2                 | 第二路 USB 相机设备名}"
    "{@camera3       | video4                 | 第三路 USB 相机设备名}"
    "{d display      |                        | 显示视频流}";

int main(int argc, char * argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    Exiter exiter;
    const auto config_path = cli.get<std::string>(0);
    const std::array<std::string, 3> device_names = {
        cli.get<std::string>(1), cli.get<std::string>(2), cli.get<std::string>(3)};
    const auto display = cli.has("display");

    std::array<std::unique_ptr<USBCamera>, 2> cameras;
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        cameras[i] = std::make_unique<USBCamera>(device_names[i], config_path);
        if (display) cv::namedWindow(device_names[i], cv::WINDOW_NORMAL);
    }

    YOLO yolo(config_path, false);
    std::array<std::chrono::steady_clock::time_point, 2> previous_timestamps{};
    std::array<bool, 2> has_previous_timestamp{};

    while (!exiter.exit()) {
        for (std::size_t i = 0; i < cameras.size(); ++i) {
            cv::Mat img;
            std::chrono::steady_clock::time_point timestamp;
            if (!cameras[i]->try_read(img, timestamp, 10) || img.empty()) continue;

            const auto armors = yolo.detect(img);
            if (has_previous_timestamp[i]) {
                const auto fps = 1.0 / delta_time(timestamp, previous_timestamps[i]);
                logger()->info(
                    "{}: {:.2f} fps, {} detections", device_names[i], fps, armors.size());
            }
            previous_timestamps[i] = timestamp;
            has_previous_timestamp[i] = true;

            if (display) {
                auto detection = img.clone();
                for (const auto & armor : armors) {
                    cv::rectangle(detection, armor.box, {0, 255, 0}, 2);
                    draw_text(
                        detection, ARMOR_NAMES[armor.name], armor.box.tl(), {0, 255, 0}, 0.6, 1);
                }
                cv::imshow(device_names[i], detection);
            }
        }

        if (display && cv::waitKey(1) == 'q') break;
    }

    if (display) cv::destroyAllWindows();
    return 0;
}
