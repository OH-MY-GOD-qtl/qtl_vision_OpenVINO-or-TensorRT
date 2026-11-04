#include "camera/usbcamera.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace
{
void require(bool condition, const std::string & message)
{
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main()
{
    try {
        require(
            USBCamera::resolve_device_path("video0") == "/dev/video0",
            "legacy videoN must resolve to /dev/videoN");
        require(
            USBCamera::resolve_device_path("v4l/by-path/pci-0000:00:14.0-usb-0:1.2:1.0-video-index0") ==
                "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:1.2:1.0-video-index0",
            "relative v4l path must resolve under /dev");
        require(
            USBCamera::resolve_device_path("/dev/v4l/by-id/usb-camera-0") ==
                "/dev/v4l/by-id/usb-camera-0",
            "absolute /dev path must be preserved");

        std::puts("usbcamera_path_test passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "usbcamera_path_test failed: %s\n", e.what());
        return 1;
    }
}
