#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

struct CameraStream
{
    std::string device;
    cv::VideoCapture camera;
    cv::Mat frame;
    std::mutex frame_mutex;
    std::thread capture_thread;
    std::atomic<bool> running{false};
};

int main()
{
    constexpr std::array<const char *, 3> devices = {"video0", "video2", "video4"};
    std::array<CameraStream, devices.size()> streams;

    for (std::size_t i = 0; i < streams.size(); ++i) {
        auto & stream = streams[i];
        stream.device = std::string("/dev/") + devices[i];
        if (!stream.camera.open(stream.device, cv::CAP_V4L)) {
            std::cerr << "Failed to open USB camera: " << stream.device << '\n';
            return 1;
        }

        stream.camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        stream.camera.set(cv::CAP_PROP_FRAME_WIDTH, 320);
        stream.camera.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
        stream.camera.set(cv::CAP_PROP_FPS, 30);

        // std::cout << "Opened " << stream.device << ": "
        //           << stream.camera.get(cv::CAP_PROP_FRAME_WIDTH) << 'x'
        //           << stream.camera.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
        //           << stream.camera.get(cv::CAP_PROP_FPS) << " fps\n";
    }

    std::atomic<bool> capture_failed{false};
    for (auto & stream : streams) {
        stream.running = true;
        auto * stream_ptr = &stream;
        stream.capture_thread = std::thread([stream_ptr, &capture_failed] {
            while (stream_ptr->running) {
                cv::Mat frame;
                if (!stream_ptr->camera.read(frame) || frame.empty()) {
                    std::cerr << "Failed to read frame from " << stream_ptr->device << '\n';
                    capture_failed = true;
                    stream_ptr->running = false;
                    break;
                }

                std::lock_guard<std::mutex> lock(stream_ptr->frame_mutex);
                stream_ptr->frame = std::move(frame);
            }
        });
    }

    while (!capture_failed) {
        for (auto & stream : streams) {
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(stream.frame_mutex);
                if (!stream.frame.empty()) frame = stream.frame.clone();
            }
            if (!frame.empty()) cv::imshow(stream.device, frame);
        }

        if (cv::waitKey(1) == 'q') break;
    }

    for (auto & stream : streams) stream.running = false;
    for (auto & stream : streams) {
        if (stream.capture_thread.joinable()) stream.capture_thread.join();
    }
    cv::destroyAllWindows();

    return capture_failed ? 1 : 0;
}
