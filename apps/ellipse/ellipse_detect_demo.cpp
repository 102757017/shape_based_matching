/*
 * ellipse_detect_demo.cpp
 * 演示如何在本工程中调用 src/ellipse/ellipse_detection 的椭圆检测。
 *
 * 用法:
 *   ellipse_detect_demo <图片路径> [输出图片路径] [选项]
 *
 * 选项:
 *   --polarity <0|-1|1>          椭圆极性, 0=检测所有极性 (默认 0)
 *   --line-width <像素>          椭圆线宽 (默认 2.0)
 *   --min-cover-angle <度>       完整度门槛, 弧覆盖角低于该值的椭圆被丢弃 (默认 240)
 *   --min-goodness <0~1>         最终质量门槛, 越低越容易检出但误检增多 (默认 0.4)
 *   --candidate-goodness <0~1>   候选粗筛门槛 (默认 0.3)
 *   --help                       显示帮助
 *
 * 接口: zgh::detectEllipse(gray.data, rows, cols, ells, params)
 *   输出 ells: 椭圆列表(中心 o / 半轴 a,b / 旋转角 phi(弧度) / 评分 goodness 等)
 */

#include <opencv2/opencv.hpp>

#include "detect.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void print_usage(const char* prog) {
    std::printf(
        "usage: %s <image> [output.png] [options]\n"
        "options:\n"
        "  --polarity <0|-1|1>          ellipse polarity, 0=all (default 0)\n"
        "  --line-width <px>            ellipse line width in pixels (default 2.0)\n"
        "  --min-cover-angle <deg>      coverage threshold in degrees (default 240)\n"
        "  --min-goodness <0~1>         final quality threshold (default 0.4)\n"
        "  --candidate-goodness <0~1>   candidate pre-filter threshold (default 0.3)\n"
        "  --help                       show this help\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    const std::string in_path = argv[1];
    std::string out_path = "ellipse_result.png";

    zgh::DetectParams params;  // 默认值与库内硬编码行为一致

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::printf("error: %s requires a value\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--polarity") {
            params.polarity = std::atoi(need_value("--polarity"));
        } else if (arg == "--line-width") {
            params.line_width = std::atof(need_value("--line-width"));
        } else if (arg == "--min-cover-angle") {
            params.min_cover_angle = std::atof(need_value("--min-cover-angle"));
        } else if (arg == "--min-goodness") {
            params.min_goodness = std::atof(need_value("--min-goodness"));
        } else if (arg == "--candidate-goodness") {
            params.candidate_goodness = std::atof(need_value("--candidate-goodness"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            // 未识别的选项: 第一个当作输出路径兼容旧用法, 其余报错
            if (out_path == "ellipse_result.png" && arg.rfind("--", 0) != 0) {
                out_path = arg;
            } else {
                std::printf("error: unknown option: %s\n", arg.c_str());
                print_usage(argv[0]);
                return 1;
            }
        }
    }

    cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::printf("failed to load image: %s\n", in_path.c_str());
        return 1;
    }

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    std::vector<std::shared_ptr<zgh::Ellipse>> ells;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = zgh::detectEllipse(gray.data, gray.rows, gray.cols, ells, params);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        std::printf("detectEllipse failed\n");
        return 2;
    }

    std::printf("image: %dx%d, ellipses: %zu, time: %.1f ms\n"
                "params: polarity=%d line_width=%.1f min_cover_angle=%.0f "
                "min_goodness=%.2f candidate_goodness=%.2f\n",
                gray.cols, gray.rows, ells.size(), ms,
                params.polarity, params.line_width, params.min_cover_angle,
                params.min_goodness, params.candidate_goodness);

    // 库的坐标约定: o.x = 行(y), o.y = 列(x), phi 为 (行,列) 平面内的角度。
    // cv::ellipse 需要 (列,行) 与相对列轴的角度, 故做 90-phi 换算。
    for (size_t i = 0; i < ells.size(); ++i) {
        const auto& e = ells[i];
        std::printf("  [%zu] center=(row %.1f, col %.1f) a=%.1f b=%.1f phi=%.1f deg "
                    "goodness=%.2f coverangle=%.1f\n",
                    i, e->o.x, e->o.y, e->a, e->b,
                    e->phi * 180.0 / CV_PI, e->goodness, e->coverangle);
        double draw_angle = 90.0 - e->phi * 180.0 / CV_PI;
        cv::ellipse(img, cv::Point(cvRound(e->o.y), cvRound(e->o.x)),
                    cv::Size(cvRound(e->a), cvRound(e->b)),
                    draw_angle, 0, 360,
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    cv::imwrite(out_path, img);
    std::printf("result saved to: %s\n", out_path.c_str());
    return 0;
}
