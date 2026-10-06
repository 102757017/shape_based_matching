/*
 * ellipse_bench.cpp
 * 椭圆检测性能基准 + 等价性校验工具。
 *
 * 用法:
 *   ellipse_bench <image> [--iters N] [--runs M]
 *
 *   --iters N   重复检测 N 次取中位数(默认 5, 首次结果用于等价性打印)
 *   --runs M    打印 M 次结果签名(默认 1)
 *
 * 编译期需要 -DSBM_ELLIPSE_PROFILE 才会有分阶段耗时。
 */

#include <opencv2/opencv.hpp>

#include "detect.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 保存首次检测出的椭圆, 供 --micro 复用作扫描基准
static std::vector<std::shared_ptr<zgh::Ellipse>> g_global_ells;

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n == 0) return 0.0;
    if (n % 2) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 结果签名: 用于确认优化前后检测结果逐字节一致
static void print_signature(const std::vector<std::shared_ptr<zgh::Ellipse>>& ells, int idx) {
    std::printf("[run %d] ellipses=%zu\n", idx, ells.size());
    char buf[512];
    size_t off = 0;
    for (size_t i = 0; i < ells.size(); ++i) {
        const auto& e = ells[i];
        int n = std::snprintf(buf + off, sizeof(buf) - off,
                              "%.2f,%.2f,%.2f,%.2f,%.4f,%.3f,%.2f | ",
                              e->o.x, e->o.y, e->a, e->b, e->phi,
                              e->goodness, e->coverangle);
        if (n < 0) break;
        off += (size_t)n;
    }
    if (off) std::printf("  sig: %s\n", buf);
}

// 微基准: 单独测 distFast / EllipseIter 推进 / 栈操作, 定位真实热点
static int micro_benchmark2(int n_iter, const std::vector<std::shared_ptr<zgh::Ellipse>>& src) {
    if (src.empty()) return 1;
    auto ell = src[0];
    double sink = 0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; ++i) {
            sink += ell->distFast(224.4 + i * 0.001, 179.7 + (i % 7) * 0.01);
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("micro: distFast  %8.2f ns/call (n=%d)\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / n_iter, n_iter);
    }
    {
        const double tol = 1.5;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; ++i) {
            sink += ell->distFast(224.4 + i * 0.0001, 179.7 - (i % 5) * 0.01);
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("micro: distFast2 %8.2f ns/call\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / n_iter);
    }
    {
        std::stack<zgh::Pixel> st;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; ++i) {
            st.push(zgh::Pixel(3, 7));
            st.pop();
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("micro: stack     %8.2f ns/op\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / n_iter);
    }
    {
        const int row = 291, col = 473;
        double* angles = new double[row * col]();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; ++i) {
            sink += angles[(i % row) * col + (i % col)];
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("micro: rand-rd   %8.2f ns/op\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / n_iter);
        delete[] angles;
    }
    std::printf("        (sink=%.1f)\n", sink);
    return 0;
}

// 微基准: 直接测 EllipseIter 的推进成本, 不受检测其它阶段干扰
static int micro_benchmark(int n_iter, const std::vector<std::shared_ptr<zgh::Ellipse>>& src) {
    if (src.empty()) {
        std::printf("micro: no ellipse available\n");
        return 1;
    }
    auto ell = src[0];
    double tol = 1.5;
    int steps = 0;
    double sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_iter; ++i) {
        zgh::EllipseIter it(ell, tol);
        while (!it.isEnd()) {
            ++it;
            ++steps;
            sink += it.np.x;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::printf("micro: %d iters, %d steps, %.2f us/iter (%.2f ns/step), sink=%.1f\n",
                n_iter, steps, us / n_iter,
                steps ? us * 1000.0 / steps : 0.0, sink);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <image> [--iters N] [--runs M] [--micro K] [--threads T]\n", argv[0]);
        return 1;
    }
    int micro = 0, threads = 0;
    for (int i = 2; i < argc; i += 2) {
        if (std::string(argv[i]) == "--micro") micro = std::atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "--threads") threads = std::atoi(argv[i + 1]);
    }
    const std::string in_path = argv[1];
    int iters = 5, runs = 1;
    for (int i = 2; i < argc; i += 2) {
        const std::string a = argv[i];
        if (i + 1 < argc) {
            if (a == "--iters") iters = std::atoi(argv[i + 1]);
            else if (a == "--runs") runs = std::atoi(argv[i + 1]);
        }
    }
    iters = std::max(iters, 1);
    runs = std::max(runs, 1);

    cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::printf("failed to load image: %s\n", in_path.c_str());
        return 1;
    }
    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img;

    zgh::DetectParams params;
    params.num_threads = threads;
    std::printf("threads request: %d\n", threads);
    std::vector<double> times;
    times.reserve(iters);

    std::vector<std::shared_ptr<zgh::Ellipse>> ells;
    for (int r = 0; r < runs; ++r) {
        ells.clear();
        const auto t0 = std::chrono::steady_clock::now();
        zgh::detectEllipse(gray.data, gray.rows, gray.cols, ells, params);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        print_signature(ells, r);
        if (r == 0) {
            g_global_ells = ells;
            times.push_back(ms);
        }
    }

    // 正式计时轮
    for (int i = 0; i < iters; ++i) {
        ells.clear();
        const auto t0 = std::chrono::steady_clock::now();
        zgh::detectEllipse(gray.data, gray.rows, gray.cols, ells, params);
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    if (micro > 0) {
        micro_benchmark2(micro * 500, g_global_ells);
        return micro_benchmark(micro, g_global_ells);
    }

    const double med = median_of(times);
    double sum = 0;
    for (double t : times) sum += t;
    std::printf("image %dx%d: median %.2f ms, mean %.2f ms, min %.2f ms, n=%zu\n",
                gray.cols, gray.rows, med, sum / times.size(), times.front(), times.size());
    zgh::printDetectStats(zgh::detectStats());
    return 0;
}
