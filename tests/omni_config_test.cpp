#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "comm/gimbal.hpp"
#include "omni/decider.hpp"
#include "omni/usb_camera_config.hpp"

void require(bool condition, const std::string & message)
{
    if (!condition) throw std::runtime_error(message);
}

int main()
{
    const auto yaml = YAML::Load(R"(
enemy_color: blue
mode: 1
omni_usb_cameras:
  - device: video0
    yaw_offset_from_hik: 30
  - device: video2
    yaw_offset_from_hik: -30
  - device: video4
    yaw_offset_from_hik: 180
omni_usb_fov_h: 40
omni_usb_fov_v: 40
omni_hik_yaw_offset: 10
simulate: false
com_port: /definitely/not/a/real/serial/device
baud_rate: 115200
)");

    const auto cameras = load_usb_camera_configs(yaml);
    require(cameras.size() == 3, "Expected three configured USB cameras");
    require(cameras[0].device == "video0", "Camera 0 device was not loaded");
    require(cameras[1].device == "video2", "Camera 1 device was not loaded");
    require(cameras[2].device == "video4", "Camera 2 device was not loaded");
    require(cameras[0].yaw_offset_deg == 30, "Camera 0 angle was not loaded");
    require(cameras[1].yaw_offset_deg == -30, "Camera 1 angle was not loaded");
    require(cameras[2].yaw_offset_deg == 180, "Camera 2 angle was not loaded");
    require(cameras[0].yaw_offset_relative_to_hik, "New mapping must be relative to Hik");

    const auto legacy = load_usb_camera_configs(YAML::Load(R"(
omni_usb_devices: [video1, video3]
omni_usb_yaw_offsets: [15, -15]
)"));
    require(legacy.size() == 2, "Legacy camera mapping was not loaded");
    require(legacy[0].device == "video1", "Legacy camera 0 device was not loaded");
    require(legacy[1].device == "video3", "Legacy camera 1 device was not loaded");
    require(!legacy[0].yaw_offset_relative_to_hik, "Legacy offsets must remain absolute");

    const auto config_path =
        (std::filesystem::temp_directory_path() /
         ("omni_config_test_" +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml"))
            .string();
    {
        std::ofstream config(config_path);
        config << YAML::Dump(yaml);
    }

    Decider decider(config_path);
    std::vector<cv::Point2f> points = {
        {0, 0}, {20, 0}, {20, 10}, {0, 10}};
    Armor armor(0, 0.9f, {0, 0, 20, 10}, points);
    armor.center_norm = {0.5f, 0.5f};
    const std::vector<Armor> armors = {armor};
    require(
        std::abs(decider.delta_angle(armors, "usb0")[0] - 40.0) < 1e-6,
        "Relative USB angle was not combined with Hik yaw offset");
    require(
        std::abs(decider.delta_angle(armors, "usb1")[0] + 20.0) < 1e-6,
        "Negative relative USB angle was converted incorrectly");
    require(
        std::abs(decider.delta_angle(armors, "usb2")[0] - 190.0) < 1e-6,
        "Rear USB angle was converted incorrectly");

    Gimbal gimbal(config_path, true);
    require(gimbal.mode() == GimbalMode::AUTO_AIM, "Simulated gimbal is not in AUTO_AIM mode");
    gimbal.send(true, false, 0, 0, 0, 0, 0, 0);

    auto legacy_yaml = YAML::Load(R"(
enemy_color: blue
mode: 1
omni_usb_devices: [video1, video3]
omni_usb_yaw_offsets: [15, -15]
omni_usb_fov_h: 40
omni_hik_yaw_offset: 10
)");
    {
        std::ofstream config(config_path);
        config << YAML::Dump(legacy_yaml);
    }
    Decider legacy_decider(config_path);
    require(
        std::abs(legacy_decider.delta_angle(armors, "usb0")[0] - 15.0) < 1e-6,
        "Legacy offsets must remain relative to the gimbal yaw origin");

    std::remove(config_path.c_str());
    return 0;
}
