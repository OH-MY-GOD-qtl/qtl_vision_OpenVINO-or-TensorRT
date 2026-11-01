#include "comm/hp_reporter.hpp"

// 无 ROS2 构建时的空实现：识别链路照常运行，只是不对外发布。
#ifndef QTL_WITH_ROS2

namespace
{
class NullHpReporter : public HpReporter
{
public:
    void report(const HpReport &) override {}
};
}  // namespace

std::unique_ptr<HpReporter> make_hp_reporter(const std::string &) {
    return std::make_unique<NullHpReporter>();
}

#endif  // QTL_WITH_ROS2
