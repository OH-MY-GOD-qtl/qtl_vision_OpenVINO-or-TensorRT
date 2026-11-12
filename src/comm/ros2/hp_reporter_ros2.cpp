#include "comm/hp_reporter.hpp"

#ifdef QTL_WITH_ROS2

#include <chrono>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <yaml-cpp/yaml.h>

namespace
{
    struct Ros2HpConfig
    {
        std::string topic = "enemy_hp";
        double rate_hz = 10.0;
    };

    Ros2HpConfig load_ros2_config(const std::string &config_path)
    {
        Ros2HpConfig config;
        try
        {
            const auto yaml = YAML::LoadFile(config_path);
            const auto hp = yaml["hp_bar"];
            if (hp && hp.IsMap())
            {
                const auto ros2 = hp["ros2"];
                if (ros2 && ros2.IsMap())
                {
                    if (ros2["topic"])
                        config.topic = ros2["topic"].as<std::string>();
                    if (ros2["rate_hz"])
                        config.rate_hz = ros2["rate_hz"].as<double>();
                }
            }
        }
        catch (const std::exception &)
        {
            // 配置缺失/非法时用默认值，不打断主链路
        }
        if (config.rate_hz <= 0)
            config.rate_hz = 10.0;
        return config;
    }

    // 发布 std_msgs/String，内容为档位数字：0 低 / 1 中 / 2 高 / 3 unknown
    class Ros2HpReporter : public HpReporter
    {
    public:
        Ros2HpReporter(const std::string &topic, double rate_hz) : period_s_(1.0 / rate_hz)
        {
            if (!rclcpp::ok())
                rclcpp::init(0, nullptr);
            node_ = std::make_shared<rclcpp::Node>("enemy_hp_publisher");
            publisher_ = node_->create_publisher<std_msgs::msg::String>(topic, 10);
            spin_thread_ = std::thread([node = node_]() { rclcpp::spin(node); });
            last_publish_ = std::chrono::steady_clock::now();
        }

        ~Ros2HpReporter() override
        {
            if (rclcpp::ok())
                rclcpp::shutdown();
            if (spin_thread_.joinable())
                spin_thread_.join();
        }

        void report(const HpReport &report) override
        {
            const auto now = std::chrono::steady_clock::now();
            const bool level_changed = report.level != last_level_;
            last_level_ = report.level;

            const double elapsed_s = std::chrono::duration<double>(now - last_publish_).count();
            // 档位变化立即发（丢检 -> unknown 不能被 10 Hz 心跳拖后），其余按心跳节流
            if (!level_changed && elapsed_s < period_s_)
                return;
            last_publish_ = now;

            std_msgs::msg::String message;
            message.data = std::to_string(report.level);
            publisher_->publish(message);
        }

    private:
        const double period_s_;
        std::chrono::steady_clock::time_point last_publish_;
        int last_level_ = -1;
        std::shared_ptr<rclcpp::Node> node_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
        std::thread spin_thread_;
    };
} // namespace

std::unique_ptr<HpReporter> make_hp_reporter(const std::string &config_path)
{
    const auto config = load_ros2_config(config_path);
    return std::make_unique<Ros2HpReporter>(config.topic, config.rate_hz);
}

#endif // QTL_WITH_ROS2
