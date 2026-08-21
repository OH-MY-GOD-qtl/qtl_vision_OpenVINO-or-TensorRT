#include "camera/dahengvision.hpp"

#include <libusb-1.0/libusb.h>

#include "tools/logger.hpp"


// 构造函数：初始化参数并启动相机
DahengVision::DahengVision(double exposure_ms, double gamma, const std::string & vid_pid)
: exposure_ms_(exposure_ms),
  gamma_(gamma),
  queue_(5),  // 增加队列大小，提高缓冲能力
  handle_(nullptr),
  height_(0),
  width_(0),
  quit_(false),
  ok_(false),
  vid_(-1),
  pid_(-1)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) logger()->warn("Unable to init libusb!");

  try_open();

  // 守护线程
  daemon_thread_ = std::thread{[this] {
    while (!quit_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      if (ok_) continue;

      if (capture_thread_.joinable()) capture_thread_.join();

      close();
      reset_usb();
      try_open();
    }
  }};
}

// 析构函数：释放资源
DahengVision::~DahengVision()
{
  quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();

  close();
  GXCloseLib();  // 反初始化SDK
}

// 解析 vid_pid 字符串（格式："vid:pid" 或 "vid,pid"，支持十六进制）
void DahengVision::set_vid_pid(const std::string & vid_pid)
{
  if (vid_pid.empty()) return;

  std::stringstream ss(vid_pid);
  std::string part;
  int idx = 0;
  char delimiter = vid_pid.find(':') != std::string::npos ? ':' : ',';

  while (std::getline(ss, part, delimiter)) {
    if (idx == 0) {
      // 支持十六进制转换（如 "2ba2" 或 "0x2ba2"）
      vid_ = std::stoi(part, nullptr, 16);
    } else if (idx == 1) {
      // 支持十六进制转换（如 "4d55" 或 "0x4d55"）
      pid_ = std::stoi(part, nullptr, 16);
    }
    idx++;
  }
}

// USB设备复位（使用libusb库，与mindvision和hikrobot保持一致）
void DahengVision::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // 使用libusb进行USB复位
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    logger()->warn("Unable to open usb!");
    return;
  }

  if (libusb_reset_device(handle))
    logger()->warn("Unable to reset usb!");
  else
    logger()->info("Reset usb successfully :)");

  libusb_close(handle);
}

// 打开相机：初始化SDK + 枚举设备 + 配置参数 + 启动采集线程
void DahengVision::open()
{
  // 先关闭日志
  GXSetLogType(GX_LOG_TYPE_OFF);

  // 初始化SDK
  GX_STATUS status = GXInitLib();
  if (status != GX_STATUS_SUCCESS) {
    logger()->error("SDK initialization failed, error code: {}", status);
    return;
  }

  // 枚举设备
  uint32_t device_num = 0;
  status = GXUpdateAllDeviceList(&device_num, 1000);
  if (status != GX_STATUS_SUCCESS || device_num == 0) {
    logger()->error("No camera device found, error code: {}", status);
    GXCloseLib();
    return;
  }

  logger()->info("Found {} camera devices", device_num);

  // 打开第一台设备（可根据vid/pid过滤，此处简化为默认第一台）
  GX_OPEN_PARAM open_param;
  memset(&open_param, 0, sizeof(GX_OPEN_PARAM));
  open_param.accessMode = GX_ACCESS_CONTROL;        // 控制模式
  open_param.openMode = GX_OPEN_INDEX;              // 按索引打开
  open_param.pszContent = const_cast<char *>("1");  // 设备索引从1开始

  status = GXOpenDevice(&open_param, &handle_);
  if (status != GX_STATUS_SUCCESS) {
    logger()->error("Camera open failed, error code: {}", status);
    GXCloseLib();
    return;
  }

  // 配置相机参数
  try {
    // 1. 设置Gamma值（支持时）
    GX_NODE_ACCESS_MODE gamma_access;
    if (
      GXGetNodeAccessMode(handle_, "Gamma", &gamma_access) == GX_STATUS_SUCCESS &&
      gamma_access == GX_NODE_ACCESS_MODE_RW) {
      status = GXSetFloatValue(handle_, "Gamma", gamma_);
      if (status != GX_STATUS_SUCCESS) throw std::runtime_error("Failed to set Gamma");
    }

    // 2. 设置曝光时间（单位：微秒）
    status = GXSetEnumValueByString(handle_, "ExposureMode", "Timed");
    if (status != GX_STATUS_SUCCESS) throw std::runtime_error("Failed to set exposure mode");
    status = GXSetFloatValue(handle_, "ExposureTime", exposure_ms_ * 1000);
    if (status != GX_STATUS_SUCCESS) throw std::runtime_error("Failed to set exposure time");

    // 3. 设置采集模式为连续采集
    status = GXSetEnumValueByString(handle_, "AcquisitionMode", "Continuous");
    if (status != GX_STATUS_SUCCESS) throw std::runtime_error("Failed to set acquisition mode");

    //  设置自动白平衡（新增）
    GX_ENUM_VALUE stEnumValue;
    status = GXSetEnumValueByString(handle_, "BalanceWhiteAuto", "Continuous");
    if (status != GX_STATUS_SUCCESS) {
      logger()->error("Failed to set auto white balance");
      throw std::runtime_error("Failed to set auto white balance");
    } else {
      logger()->info("Auto white balance set to continuous mode");
    }
    
    // 4. 设置像素格式为BayerBG8（兼容OpenCV转换）
    GX_ENUM_VALUE pixel_format;
    status = GXGetEnumValue(handle_, "PixelFormat", &pixel_format);
    if (status == GX_STATUS_SUCCESS) {
      bool support_bayer_bg8 = false;
      for (uint32_t i = 0; i < pixel_format.nSupportedNum; ++i) {
        if (strcmp(pixel_format.nArrySupportedValue[i].strCurSymbolic, "BayerBG8") == 0) {
          support_bayer_bg8 = true;
          break;
        }
      }
      if (support_bayer_bg8) GXSetEnumValueByString(handle_, "PixelFormat", "BayerBG8");
    }

    // 5. 获取图像尺寸
    GX_INT_VALUE width_val, height_val;
    GXGetIntValue(handle_, "Width", &width_val);
    GXGetIntValue(handle_, "Height", &height_val);
    width_ = static_cast<int64_t>(width_val.nCurValue);
    height_ = static_cast<int64_t>(height_val.nCurValue);

    // 6. 流层配置（减少残帧，优化帧率）
    GX_DS_HANDLE data_stream;
    if (GXGetDataStreamHandleFromDev(handle_, 1, &data_stream) == GX_STATUS_SUCCESS) {
      GXSetIntValue(data_stream, "BlockTimeout", 200);   // 减少数据块超时到200ms
      GXSetIntValue(data_stream, "ResendTimeout", 100);  // 减少重传超时到100ms
      // 增加缓冲区数量以提高帧率
      GXSetIntValue(data_stream, "BufferHandlingMode", 1);  // 启用循环缓冲区模式
      GXSetIntValue(data_stream, "BufferCount", 8);  // 设置缓冲区数量为8
    }

    // 7. 启动流传输（Linux专用）
    status = GXStreamOn(handle_);
    if (status != GX_STATUS_SUCCESS)
      throw std::runtime_error("Failed to start stream transmission");

    // 8. 启动采集线程（流传输启动后再启动采集线程）
    ok_ = true;
    capture_thread_ = std::thread([this]() {
      GX_FRAME_BUFFER * frame_buffer = nullptr;
      while (ok_ && !quit_) {
        // 获取图像缓冲区
        GX_STATUS status = GXDQBuf(handle_, &frame_buffer, 50);  // 减少超时时间到50ms
        if (status == GX_STATUS_SUCCESS && frame_buffer != nullptr) {
          if (frame_buffer->nStatus == GX_FRAME_STATUS_SUCCESS) {
            // BayerBG8转RGB，但直接转换为BGR以解决红蓝颠倒问题
            cv::Mat bayer_img(height_, width_, CV_8UC1, frame_buffer->pImgBuf);
            // 水平+垂直翻转
            //cv::flip(bayer_img, bayer_img, -1);  // 水平+垂直翻转
            cv::Mat rgb_img;
            cv::cvtColor(bayer_img, rgb_img, cv::COLOR_BayerBG2BGR);

            // 存入线程安全队列
            CameraData data;
            data.img = rgb_img.clone();
            data.timestamp = std::chrono::steady_clock::now();
            queue_.push(data);
          }
          // 归还缓冲区
          GXQBuf(handle_, frame_buffer);
          frame_buffer = nullptr;
        } else if (status != GX_STATUS_TIMEOUT) {
          logger()->error("Image capture failed, error code: {}", status);
          ok_ = false;
          break;
        }

        // 移除等待时间，提高帧率
        // std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });

    logger()->info(
      "Camera started successfully: {}x{} | Exposure time: {}ms", width_, height_, exposure_ms_);
  } catch (const std::exception & e) {
    logger()->error("Camera configuration failed: {}", e.what());
    close();
    GXCloseLib();
  }
}

// 关闭相机：停止采集 + 释放句柄
void DahengVision::close()
{
  ok_ = false;
  if (capture_thread_.joinable()) capture_thread_.join();

  if (handle_ != nullptr) {
    GXStreamOff(handle_);    // 停止流传输
    GXCloseDevice(handle_);  // 关闭设备
    handle_ = nullptr;
  }

  queue_.clear();  // 清空图像队列
}

// 读取图像：从线程安全队列获取
void DahengVision::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  img.release();
  CameraData data;
  if (queue_.try_pop(data, 1000))  // 1秒超时
  {
    // 直接使用队列中的图像，相机已在硬件层面完成翻转
    img = data.img.clone();
    timestamp = data.timestamp;
  } else {
    logger()->warn("Image read timeout");
    // 返回一个空的有效图像，避免后续处理崩溃
    img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
    timestamp = std::chrono::steady_clock::now();
  }
}

// 守护线程：监测相机状态，自动重连
void DahengVision::try_open()
{
  if (ok_) return;

  logger()->info("Trying to open camera...");
  open();
}

