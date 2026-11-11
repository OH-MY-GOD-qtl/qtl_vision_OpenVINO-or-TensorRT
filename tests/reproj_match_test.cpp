#include "tracker/reproj_match.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    void require_near(double actual, double expected, double tol, const std::string &message)
    {
        if (!(std::abs(actual - expected) <= tol))
        {
            throw std::runtime_error(
                message + ": actual=" + std::to_string(actual) +
                " expected=" + std::to_string(expected));
        }
    }

    // 一个 40x20 像素的矩形，按顺时针给出
    std::vector<cv::Point2f> make_quad(float scale = 1.0f, float dx = 0.0f, float dy = 0.0f)
    {
        return {
            {0.0f * scale + dx, 0.0f * scale + dy},
            {40.0f * scale + dx, 0.0f * scale + dy},
            {40.0f * scale + dx, 20.0f * scale + dy},
            {0.0f * scale + dx, 20.0f * scale + dy}};
    }

    std::vector<cv::Point2f> rotate_left(const std::vector<cv::Point2f> &quad, int n)
    {
        std::vector<cv::Point2f> out;
        for (int i = 0; i < 4; i++) out.push_back(quad[(i + n) % 4]);
        return out;
    }
} // namespace

// 完全重合时误差为 0
void test_identical()
{
    const auto quad = make_quad();
    require_near(
        reproj::normalized_quad_distance(quad, quad), 0.0, 1e-9, "identical quads must score 0");
}

// 角点起始位置不同（换板/不同角点顺序）不应影响结果
void test_cyclic_shift_invariance()
{
    const auto quad = make_quad();
    for (int shift = 1; shift < 4; shift++)
    {
        require_near(
            reproj::normalized_quad_distance(quad, rotate_left(quad, shift)), 0.0, 1e-9,
            "cyclic shift must not change the score");
    }
}

// 正/背面观察会得到相反绕向的角点顺序
void test_reversed_invariance()
{
    const auto quad = make_quad();
    const std::vector<cv::Point2f> reversed{quad[3], quad[2], quad[1], quad[0]};
    require_near(
        reproj::normalized_quad_distance(quad, reversed), 0.0, 1e-9,
        "reversed corner order must not change the score");
}

// 平移一个板宽：归一化后误差应约为 1（板宽 40，平均边长 (40+20)/2=30，
// 四角平均位移 40，所以 40/30 ≈ 1.333）
void test_translation_is_normalized_by_edge()
{
    const auto pred = make_quad();
    const auto obs = make_quad(1.0f, 40.0f, 0.0f);
    require_near(
        reproj::normalized_quad_distance(pred, obs), 40.0 / 30.0, 1e-6,
        "one board-width shift must normalize to width/mean-edge");
}

// 同比例放大后误差与四边形同时放大，归一化结果不变
void test_scale_invariance()
{
    const auto pred = make_quad();
    const auto obs_small = make_quad(1.0f, 10.0f, 0.0f);
    const auto obs_big = make_quad(4.0f, 40.0f, 0.0f);
    require_near(
        reproj::normalized_quad_distance(pred, obs_small),
        reproj::normalized_quad_distance(make_quad(4.0f), obs_big), 1e-6,
        "normalized error must not depend on how large the quad appears");
}

// 退化四边形（边长趋近 0）不能产生有限结果，否则阈值判据会误通过
void test_degenerate_returns_infinity()
{
    const std::vector<cv::Point2f> point{
        {100.0f, 100.0f}, {100.0f, 100.0f}, {100.0f, 100.0f}, {100.0f, 100.0f}};
    require(
        !std::isfinite(reproj::normalized_quad_distance(point, make_quad())),
        "degenerate prediction must return infinity");
}

// 角点数量不是 4 时直接判无效
void test_wrong_size_returns_infinity()
{
    const std::vector<cv::Point2f> three{{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
    require(
        !std::isfinite(reproj::normalized_quad_distance(three, make_quad())),
        "non-quad prediction must return infinity");
}

int main()
{
    try
    {
        test_identical();
        test_cyclic_shift_invariance();
        test_reversed_invariance();
        test_translation_is_normalized_by_edge();
        test_scale_invariance();
        test_degenerate_returns_infinity();
        test_wrong_size_returns_infinity();

        std::puts("reproj_match_test passed");
        return 0;
    }
    catch (const std::exception &e)
    {
        std::printf("reproj_match_test failed: %s\n", e.what());
        return 1;
    }
}
