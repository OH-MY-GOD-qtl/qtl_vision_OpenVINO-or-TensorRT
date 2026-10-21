#ifndef OMNI__USB_CAMERA_CONFIG_HPP
#define OMNI__USB_CAMERA_CONFIG_HPP

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

struct OmniUsbCameraConfig
{
    std::string device;
    double yaw_offset_deg;
    bool yaw_offset_relative_to_hik;
};

inline std::vector<OmniUsbCameraConfig> load_usb_camera_configs(const YAML::Node & yaml)
{
    std::vector<OmniUsbCameraConfig> cameras;
    if (yaml["omni_usb_cameras"]) {
        const auto configured_cameras = yaml["omni_usb_cameras"];
        if (!configured_cameras.IsSequence() || configured_cameras.size() != 3) {
            throw std::runtime_error("omni_usb_cameras must contain exactly three entries");
        }

        for (std::size_t i = 0; i < configured_cameras.size(); ++i) {
            const auto camera = configured_cameras[i];
            if (!camera["device"] || !camera["yaw_offset_from_hik"]) {
                throw std::runtime_error(
                    "Each omni_usb_cameras entry requires device and yaw_offset_from_hik");
            }
            cameras.push_back(
                {camera["device"].as<std::string>(), camera["yaw_offset_from_hik"].as<double>(),
                 true});
        }
        return cameras;
    }

    std::vector<std::string> devices = {"video0", "video2", "video4"};
    if (yaml["omni_usb_devices"]) {
        devices.clear();
        for (const auto & device : yaml["omni_usb_devices"])
            devices.push_back(device.as<std::string>());
    }

    std::vector<double> yaw_offsets = {60, -60, -180};
    if (yaml["omni_usb_yaw_offsets"]) {
        yaw_offsets.clear();
        for (const auto & offset : yaml["omni_usb_yaw_offsets"])
            yaw_offsets.push_back(offset.as<double>());
    }

    for (std::size_t i = 0; i < devices.size(); ++i) {
        const double yaw_offset = i < yaw_offsets.size() ? yaw_offsets[i] : 0.0;
        cameras.push_back({devices[i], yaw_offset, false});
    }
    return cameras;
}

#endif
