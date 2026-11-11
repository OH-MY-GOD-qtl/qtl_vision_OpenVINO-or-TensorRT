#include "camera/usbcamera.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

        // 只有裸 videoN 才去查别名；by-id/by-path 显式路径必须原样使用
        require(
            USBCamera::needs_alias_resolution("video0") &&
                USBCamera::needs_alias_resolution("/dev/video12"),
            "bare videoN must need alias resolution");
        require(
            !USBCamera::needs_alias_resolution("/dev/v4l/by-path/port-1-video-index0") &&
                !USBCamera::needs_alias_resolution("v4l/by-id/cam-video-index0"),
            "explicit by-id/by-path must be used verbatim");

        // 别名查找：用注入目录模拟 /dev/v4l，覆盖「有别名 / 没别名 / 重插后改名」
        namespace fs = std::filesystem;
        const auto root = fs::temp_directory_path() / "usbcamera_path_test";
        fs::remove_all(root);
        fs::create_directories(root / "by-path");
        std::ofstream(root / "video7").put('x');
        std::ofstream(root / "video8").put('x');

        const std::string video7 = (root / "video7").string();
        const std::vector<std::string> dirs = {(root / "by-path").string()};

        fs::create_symlink(root / "video7", root / "by-path" / "port-1-video-index0");
        require(
            USBCamera::find_stable_device_path(video7, dirs) ==
                (root / "by-path" / "port-1-video-index0").string(),
            "existing alias must be found");
        require(
            USBCamera::find_stable_device_path((root / "video8").string(), dirs).empty(),
            "device without alias must yield empty");

        // 模拟重插到另一个口：旧别名消失、新别名出现
        fs::remove(root / "by-path" / "port-1-video-index0");
        require(
            USBCamera::find_stable_device_path(video7, dirs).empty(),
            "removed alias must yield empty");
        fs::create_symlink(root / "video7", root / "by-path" / "port-9-video-index0");
        require(
            USBCamera::find_stable_device_path(video7, dirs) ==
                (root / "by-path" / "port-9-video-index0").string(),
            "re-enumerated alias must be found");

        fs::remove_all(root);

        std::puts("usbcamera_path_test passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "usbcamera_path_test failed: %s\n", e.what());
        return 1;
    }
}
