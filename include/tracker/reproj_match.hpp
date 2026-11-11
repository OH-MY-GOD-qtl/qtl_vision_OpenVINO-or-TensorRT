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
    // 结果再除以预测四边形的平均边长，使阈值与目标距离无关（远处与近处可比）。
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
        for (int j = 0; j < 4; j++) edge_sum += cv::norm(pred[j] - pred[(j + 1) % 4]);

        const double scale = edge_sum / 4.0;
        if (scale < 1e-6)
            return std::numeric_limits<double>::infinity();

        return best / scale;
    }
} // namespace reproj

#endif // AUTO_AIM__REPROJ_MATCH_HPP
