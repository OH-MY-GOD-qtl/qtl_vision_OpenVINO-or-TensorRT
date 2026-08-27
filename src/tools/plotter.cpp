#include "tools/plotter.hpp"

#include <arpa/inet.h>   // htons, inet_addr
#include <sys/socket.h>  // socket, sendto
#include <unistd.h>      // close

#include <sstream>


Plotter::Plotter(std::string host, uint16_t port)
{
    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);

    destination_.sin_family = AF_INET;
    destination_.sin_port = ::htons(port);
    destination_.sin_addr.s_addr = ::inet_addr(host.c_str());
}

Plotter::~Plotter() { ::close(socket_); }

void Plotter::plot(const nlohmann::json & json)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // snap 版 PlotJuggler 无 JSON 解析插件，但自带 InfluxDB Line Protocol 解析器，
    // 故把扁平 JSON 转成 Line Protocol 文本再发送：
    //   vision gimbal_yaw=10.5,plan_pitch=20.3,fire=0i,...
    std::ostringstream oss;
    oss << "vision";
    bool first = true;
    for (auto it = json.begin(); it != json.end(); ++it) {
        const std::string & key = it.key();
        const auto & val       = it.value();

        if (val.is_boolean()) {
            oss << (first ? ' ' : ',') << key << '=' << (val.get<bool>() ? 1 : 0) << 'i';
            first = false;
        } else if (val.is_number_integer()) {
            oss << (first ? ' ' : ',') << key << '=' << val.get<int64_t>() << 'i';
            first = false;
        } else if (val.is_number_unsigned()) {
            oss << (first ? ' ' : ',') << key << '=' << val.get<uint64_t>() << 'i';
            first = false;
        } else if (val.is_number_float()) {
            oss << (first ? ' ' : ',') << key << '=' << val.get<double>();
            first = false;
        }
        // 字符串等其他类型跳过
    }

    const std::string line = oss.str();
    ::sendto(
        socket_, line.c_str(), line.length(), 0, reinterpret_cast<sockaddr *>(&destination_),
        sizeof(destination_));
}

