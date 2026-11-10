#include "buff/buff_detector.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace
{
    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }
} // namespace

int main()
{
    try
    {
        // Buff_Detector 构造时会加载模型，这里用仓库内资产；测试需在仓库根目录运行
        const auto config_path =
            (std::filesystem::temp_directory_path() /
             ("buff_detector_test_" +
              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml"))
                .string();
        {
            std::ofstream config(config_path);
            config << "model: assets/yolo11_buff_int8.xml\n";
        }

        Buff_Detector detector(config_path);
        require(detector.status() == LOSE, "initial status must be LOSE");

        // 纯黑图像应检测不到目标，持续走 handle_lose()
        cv::Mat black = cv::Mat::zeros(640, 640, CV_8UC3);
        detector.detect(black);
        require(detector.status() == TEM_LOSE, "first miss must be TEM_LOSE");

        for (int i = 1; i < LOSE_MAX; ++i)
            detector.detect(black);
        require(detector.status() == LOSE, "after LOSE_MAX misses status must be LOSE");

        std::filesystem::remove(config_path);
        std::puts("buff_detector_test passed");
        return 0;
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "buff_detector_test failed: %s\n", e.what());
        return 1;
    }
}
