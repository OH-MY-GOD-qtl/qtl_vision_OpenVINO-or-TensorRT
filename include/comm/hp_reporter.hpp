#ifndef IO__HP_REPORTER_HPP
#define IO__HP_REPORTER_HPP

#include <memory>
#include <string>

// 血量档位（与 ROS2 上报编码一致）：0 低 / 1 中 / 2 高 / 3 unknown。
// 血量灯条识别见 blood_bar 模块；本接口只负责把结果送出去。
struct HpReport
{
    int level = 3;
    double percent = -1;
    bool valid = false;
};

class HpReporter
{
public:
    virtual ~HpReporter() = default;

    virtual void report(const HpReport & report) = 0;
};

// 按配置创建上报器：
//   - 构建时打开 QTL_ENABLE_ROS2 -> ROS2 发布器（std_msgs/String，内容为档位数字）
//   - 否则返回空实现（不影响主链路，仍可通过日志观察）
std::unique_ptr<HpReporter> make_hp_reporter(const std::string & config_path);

#endif  // IO__HP_REPORTER_HPP
