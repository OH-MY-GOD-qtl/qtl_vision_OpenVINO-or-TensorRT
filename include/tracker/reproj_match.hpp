#ifndef AUTO_AIM__REPROJ_MATCH_HPP
#define AUTO_AIM__REPROJ_MATCH_HPP

#include <algorithm>
#include <limits>
#include <opencv2/core/types.hpp>
#include <vector>

namespace reproj
{
    // 预测四边形与观测四边形的归一化像素距离。
    //
    // 装甲板是矩形，从正面和背面看过去会得到循环移位、甚至绕向相反的角点顺序，
    // 所以对 4 个循环移位 × 2 个绕向取最小。
    //
    // 分母用**观测**四边形的平均边长，而不是预测四边形，理由三点：
    // 1. 观测边长是图像上的直接测量，不依赖当前状态，不会因预测偏差而失真；
    // 2. 候选比较才公平——每个候选的预测尺寸不同，各除各的边长会把「预测得小」
    //    的候选系统性放大误差，argmin 本身就会偏；共用同一分母后 margin 才有意义；
    // 3. 目标接近边缘朝向时预测框会被压缩，用预测边长归一化会把正常误差放大成
    //    假警报。
    inline double normalized_quad_distance(
        const std::vector<cv::Point2f> &pred, const std::vector<cv::Point2f> &obs)
    {
        if (pred.size() != 4 || obs.size() != 4)
            return std::numeric_limits<double>::infinity();

        double best = std::numeric_limits<double>::infinity();
        for (int dir = 0; dir < 2; dir++)
        {
            for (int shift = 0; shift < 4; shift++)
            {
                double sum = 0.0;
                for (int j = 0; j < 4; j++)
                {
                    const int k = (dir == 0) ? (j + shift) % 4 : (shift - j + 8) % 4;
                    sum += cv::norm(pred[j] - obs[k]);
                }
                best = std::min(best, sum / 4.0);
            }
        }

        double edge_sum = 0.0;
        for (int j = 0; j < 4; j++) edge_sum += cv::norm(obs[j] - obs[(j + 1) % 4]);

        const double scale = edge_sum / 4.0;
        if (scale < 1e-6)
            return std::numeric_limits<double>::infinity();

        return best / scale;
    }
} // namespace reproj

#endif // AUTO_AIM__REPROJ_MATCH_HPP
