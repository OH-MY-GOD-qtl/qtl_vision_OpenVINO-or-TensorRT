#ifndef COMMON_HPP
#define COMMON_HPP

#include "tools/logger.hpp"

// 全局常量
constexpr float MIN_BULLET_SPEED = 14.0f;      // 弹速下限（m/s），低于视为无效
constexpr float MAX_BULLET_SPEED = 25.0f;      // 弹速上限（m/s），高于视为配置异常
constexpr float DEFAULT_BULLET_SPEED = 23.0f;  // 默认弹速兜底值（m/s）

// 弹速异常时统一兜底，并保留来源信息便于定位问题
inline double sanitize_bullet_speed(double bullet_speed, const char * source)
{
    if (bullet_speed < MIN_BULLET_SPEED || bullet_speed > MAX_BULLET_SPEED) {
        logger()->warn(
            "[{}] invalid bullet speed {:.2f} m/s, fallback to {:.2f} m/s", source, bullet_speed,
            DEFAULT_BULLET_SPEED);
        return DEFAULT_BULLET_SPEED;
    }
    return bullet_speed;
}

#endif
