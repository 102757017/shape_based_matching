// 调试用: 逐级输出 ellipse detection 中间结果
#include <opencv2/opencv.hpp>
#include "cvcannyapi.h"
#include "defines.h"
#include "compute.h"
#include "detect.h"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    cv::Mat img = cv::imread(argc >= 2 ? argv[1] : "synth_ellipse.bmp", cv::IMREAD_GRAYSCALE);
    if (img.empty()) { std::printf("load failed\n"); return 1; }

    // 1. Canny3 边缘
    std::vector<double> angles((size_t)img.rows * img.cols, 0);
    zgh::calculateGradient3(img.data, img.rows, img.cols, angles.data());
    int n_angle_def = 0;
    for (double a : angles) if (a != ANGLE_NOT_DEF) ++n_angle_def;
    std::printf("angles defined: %d / %d\n", n_angle_def, img.rows * img.cols);

    // 2. 弧段分组(与 getValidInitialEllipseSet 内部一致)
    {
        double scale = 0.8, sigma_scale = 0.6;
        int sr = (int)ceil(img.rows * scale), sc = (int)ceil(img.cols * scale);
        std::vector<double> sdata((size_t)sr * sc);
        zgh::gaussianSampler(img.data, img.rows, img.cols, sdata.data(), sr, sc, scale, sigma_scale);
        std::vector<std::shared_ptr<zgh::Arc>> arcs;
        zgh::lsdgroups(sdata.data(), sr, sc, scale, arcs);
        std::printf("arc groups: %zu\n", arcs.size());
        for (size_t i = 0; i < std::min(arcs.size(), (size_t)15); ++i)
            std::printf("  arc[%zu] polarity=%d coverages=%.3f rad (%.1f deg) lines=%zu\n",
                        i, arcs[i]->polarity, arcs[i]->coverages,
                        arcs[i]->coverages * 180.0 / CV_PI, arcs[i]->lines.size());

        // 3. 拟合单个弧段
        int fitted = 0;
        for (size_t i = 0; i < arcs.size(); ++i) {
            auto ell = zgh::calcElliseParam(arcs[i], nullptr, angles.data(), img.rows, img.cols);
            if (ell) {
                ++fitted;
                if (fitted <= 5)
                    std::printf("  fit arc[%zu]: o=(%.1f,%.1f) a=%.1f b=%.1f legal=%d\n",
                                i, ell->o.x, ell->o.y, ell->a, ell->b, (int)ell->isLegal());
            }
        }
        std::printf("arc fits succeeded: %d\n", fitted);
    }

    // 4. 初始椭圆候选
    std::vector<std::shared_ptr<zgh::Ellipse>> cands;
    zgh::getValidInitialEllipseSet(img.data, angles.data(), img.rows, img.cols, cands, 0);
    std::fprintf(stderr, "[stage] initial candidates: %zu\n", cands.size());
    std::fflush(stderr);
    for (size_t i = 0; i < std::min(cands.size(), (size_t)10); ++i)
        std::fprintf(stderr, "  cand[%zu] o=(%.1f,%.1f) a=%.1f b=%.1f legal=%d\n",
                     i, cands[i]->o.x, cands[i]->o.y, cands[i]->a, cands[i]->b,
                     (int)cands[i]->isLegal());

    // 5. generateEllipseCandidates
    {
        std::vector<std::shared_ptr<zgh::Ellipse>> gc;
        zgh::generateEllipseCandidates(img.data, angles.data(), img.rows, img.cols, gc, 0);
        std::fprintf(stderr, "[stage] generateEllipseCandidates: %zu\n", gc.size());
        std::fflush(stderr);
        for (size_t i = 0; i < std::min(gc.size(), (size_t)10); ++i)
            std::fprintf(stderr, "  gc[%zu] o=(%.1f,%.1f) a=%.1f b=%.1f\n",
                         i, gc[i]->o.x, gc[i]->o.y, gc[i]->a, gc[i]->b);
    }

    // 6. 完整检测
    std::vector<std::shared_ptr<zgh::Ellipse>> ells;
    std::fprintf(stderr, "[stage] detectEllipse begin\n");
    std::fflush(stderr);
    bool ok = zgh::detectEllipse(img.data, img.rows, img.cols, ells, 0, 2.0);
    std::fprintf(stderr, "[stage] detectEllipse ok=%d, ellipses=%zu\n", (int)ok, ells.size());
    std::fflush(stderr);
    for (size_t i = 0; i < ells.size(); ++i)
        std::printf("  [%zu] o=(%.1f,%.1f) a=%.1f b=%.1f phi=%.2f goodness=%.2f\n",
                    i, ells[i]->o.x, ells[i]->o.y, ells[i]->a, ells[i]->b,
                    ells[i]->phi * 180.0 / CV_PI, ells[i]->goodness);
    return 0;
}
