/*
 * aamed_demo.cpp
 * 演示如何在本工程中调用 AAMED(弧邻接矩阵)椭圆检测, 与根目录的
 * ellipse_detect_demo.cpp (standard-ellipse-detection 那一路) 对照使用。
 *
 * 用法:
 *   aamed_demo <图片路径> [输出图片路径] [选项]
 *
 * 选项:
 *   --theta-fsa <度>       邻域分组角度约束, 默认 60 (等价 CV_PI/3)。调大更宽松
 *                          更容易成组, 误检增多; 调小则更严。
 *   --length-fsa <值>      邻域分组长度约束, 默认 3.4
 *   --t-val <0~1>          验证评分门槛, 默认 0.77。调低能检出更残缺的椭圆
 *   --min-goodness <0~1>   最终输出评分门槛, 默认 0 (不筛)
 *   --nms-iou <0~1>        非极大抑制重叠门槛, 默认 0.7
 *   --help                 显示帮助
 *
 * 接口: sbm::AamedDetector::detect(gray, params, out)
 *   输出 out: 椭圆列表 (中心 cx/cy / 半轴 a,b / 旋转角 phi(弧度) / 评分 goodness)
 *   其中 cx=列(x), cy=行(y), phi 为 cv::ellipse 口径, 可直接拿去画。
 *
 * 注意内存: AAMED 内部按 图像rows*图像cols 个 Node_FC 预分配, 约 272 Byte/像素,
 * 1080p 单帧约 550 MB。detect() 内部只增不减地复用实例, 反复调用同一尺寸不会
 * 重复分配; 想完全掌控内存生命周期就直接用 sbm::AamedDetector 长期持有。
 */

#include <opencv2/opencv.hpp>

#include "ellipse/aamed_detector.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void print_usage(const char* prog) {
    std::printf(
        "usage: %s <image> [output.png] [options]\n"
        "options:\n"
        "  --theta-fsa <deg>        arc grouping angle constraint, default 60 (CV_PI/3)\n"
        "  --length-fsa <val>       arc grouping length constraint, default 3.4\n"
        "  --t-val <0~1>            validation score threshold, default 0.77\n"
        "  --min-goodness <0~1>     final quality threshold, default 0 (no filter)\n"
        "  --nms-iou <0~1>          non-maximum suppression IoU threshold, default 0.7\n"
        "  --help                   show this help\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    const std::string in_path = argv[1];
    std::string out_path = "aamed_result.png";

    sbm::AamedParams params = sbm::aamed_params_default();
    double theta_fsa_deg = 60.0;   // 展示用: 命令行给角度, 内部转弧度

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::printf("error: %s requires a value\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--theta-fsa") {
            theta_fsa_deg = std::atof(need_value("--theta-fsa"));
        } else if (arg == "--length-fsa") {
            params.length_fsa = std::atof(need_value("--length-fsa"));
        } else if (arg == "--t-val") {
            params.t_val = std::atof(need_value("--t-val"));
        } else if (arg == "--min-goodness") {
            params.min_goodness = std::atof(need_value("--min-goodness"));
        } else if (arg == "--nms-iou") {
            params.nms_iou = std::atof(need_value("--nms-iou"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            // 未识别的选项: 第一个当作输出路径兼容旧用法, 其余报错
            if (out_path == "aamed_result.png" && arg.rfind("--", 0) != 0) {
                out_path = arg;
            } else {
                std::printf("error: unknown option: %s\n", arg.c_str());
                print_usage(argv[0]);
                return 1;
            }
        }
    }
    if (theta_fsa_deg > 0.0) params.theta_fsa = theta_fsa_deg * CV_PI / 180.0;

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

    sbm::AamedDetector detector;
    std::vector<sbm::AamedEllipse> ells;

    const auto t0 = std::chrono::steady_clock::now();
    int n = 0;
    try {
        n = detector.detect(gray, params, ells);
    } catch (const std::exception& e) {
        std::printf("AAMED detect failed: %s\n", e.what());
        return 2;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("image: %dx%d, ellipses: %d, time: %.1f ms (cache %dx%d)\n"
                "params: theta_fsa=%.1f deg (%.4f rad) length_fsa=%.2f t_val=%.2f "
                "min_goodness=%.2f nms_iou=%.2f\n",
                gray.cols, gray.rows, n, ms,
                detector.capacity_cols(), detector.capacity_rows(),
                theta_fsa_deg, params.theta_fsa, params.length_fsa, params.t_val,
                params.min_goodness, params.nms_iou);

    // 画回原图: sbm::AamedEllipse 已是 (列,行) + cv::ellipse 口径, 直接用
    for (int i = 0; i < n; ++i) {
        const sbm::AamedEllipse& e = ells[i];
        std::printf("  [%d] center=(col %.1f, row %.1f) a=%.1f b=%.1f phi=%.1f deg "
                    "goodness=%.3f\n",
                    i, e.cx, e.cy, e.a, e.b, e.phi * 180.0 / CV_PI, e.goodness);
        cv::ellipse(img, cv::Point(cvRound(e.cx), cvRound(e.cy)),
                    cv::Size(cvRound(e.a), cvRound(e.b)),
                    e.phi * 180.0 / CV_PI, 0, 360,
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    cv::imwrite(out_path, img);
    std::printf("result saved to: %s\n", out_path.c_str());
    return 0;
}
