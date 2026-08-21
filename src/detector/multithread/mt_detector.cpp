#include "detector/multithread/mt_detector.hpp"

#include <yaml-cpp/yaml.h>
#include "image/image.hpp"
#include "tools/yaml.hpp"


namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = yaml_load(config_path);
  auto yolo_name = yaml_read<std::string>(yaml, "yolo_name");
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml_read<std::string>(yaml, "device");

  auto model = core_.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})  // TODO
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    // .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, 
    // 调整性能模式，优先保证低延迟和稳定帧率
    ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
    // 启用CPU优化
    ov::hint::enable_cpu_pinning(true));

  logger()->info("[MultiThreadDetector] initialized !");
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  // preproces（letterbox 见 image 模块）
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  double scale;
  letterbox(img, input, scale);

  auto input_port = compiled_model_.input();
  auto infer_request = compiled_model_.create_infer_request();
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  infer_request.set_input_tensor(input_tensor);
  infer_request.start_async();
  // 输入缓冲随队列移动以延续生命周期；img 为引用计数浅拷贝，无需逐帧 clone 整幅图
  queue_.push({std::move(input), img, t, std::move(infer_request)});
}

std::tuple<std::vector<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [input, img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::vector<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [input, img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {img, std::move(armors), t};
}

}  // namespace multithread

