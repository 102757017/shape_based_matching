/* ============================================================================
 * ellipse_compare.cpp — 两种椭圆检测算法的公平对比基准
 *
 *   路线 A: standard-ellipse-detection   (zgh::detectEllipse, 弧段扫描 + 精化)
 *   路线 B: AAMED / FLED                 (弧邻接矩阵 + 弧段分组投票)
 *
 * 对比两个维度:
 *   1) 速度 —— 交错(interleaved)计时, 交替跑 A/B 各一轮, 抵消频率漂移/调度
 *      噪声对单算法的不公平偏置; 同时单独量一次 AAMED 的首次内存分配开销。
 *   2) 精度 —— 合成场景带已知真值, 用栅格化 IoU 做一对一贪心匹配, 统计
 *      TP / FP / 召回 / 精确率 / F1 / TP 的平均 IoU。
 *
 * 两路输出都归约到同一个 sbm_ellipse_t 口径 (cx=列, cy=行, a/b 是半轴,
 * phi 是相对列轴的弧度), 换算方式与 C ABI 层的 sbm_detect_ellipses 完全一致,
 * 所以这里量出来的数字可以直接代表线上两套 C API 的表现。
 *
 * 用法:
 *   ellipse_compare [--iters N] [--threads T] [--dump <outdir>]
 */

#include <opencv2/opencv.hpp>

#include "detect.h"
#include "../ellipse/aamed_detector.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

typedef sbm::AamedEllipse Ell;          /* 布局与 sbm_ellipse_t 一致 */

static double kMs = 1.0;                 /* 计时单位: ms (0=秒, 由 main 设) */

/* ---------------- 通用小工具 ---------------- */

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

static cv::RotatedRect rect_of(const Ell& e) {
    return cv::RotatedRect(cv::Point2f(static_cast<float>(e.cx),
                                       static_cast<float>(e.cy)),
                           cv::Size2f(static_cast<float>(e.a * 2.0),
                                      static_cast<float>(e.b * 2.0)),
                           static_cast<float>(e.phi * 180.0 / CV_PI));
}

/* 把一个椭圆栅格成描边掩膜, thick 越大带越宽 */
static cv::Mat mask_of(const cv::Size& sz, const Ell& e, int thick) {
    cv::Mat m(sz, CV_8UC1, cv::Scalar(0));
    cv::ellipse(m, rect_of(e), cv::Scalar(255), thick);
    return m;
}

/* 椭圆"内部区域"掩膜。
 * 注意不能用描边环 (cv::ellipse thickness>0) 来算 IoU —— 环的面积与线宽线性
 * 相关, 真值用 3px 描边、预测用 1px 描边时光 IoU 就封顶在 ~0.33, 会把几何
 * 完全一致的结果判成不匹配。填充区域内的 IoU 才真正反映中心/半轴的拟合精度。 */
static cv::Mat fill_mask(const cv::Size& sz, const Ell& e) {
    cv::Mat m(sz, CV_8UC1, cv::Scalar(0));
    cv::ellipse(m, rect_of(e), cv::Scalar(255), cv::FILLED);
    return m;
}

static double iou_of(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat inter, uni;
    cv::bitwise_and(a, b, inter);
    cv::bitwise_or(a, b, uni);
    const int denom = std::max(1, cv::countNonZero(uni));
    return static_cast<double>(cv::countNonZero(inter)) / denom;
}

/* ---------------- 路线 A: standard-ellipse-detection ---------------- */

static std::vector<Ell> run_standard(const cv::Mat& gray, const zgh::DetectParams& p) {
    std::vector<std::shared_ptr<zgh::Ellipse>> ells;
    zgh::detectEllipse(gray.data, gray.rows, gray.cols, ells, p);
    std::vector<Ell> out;
    out.reserve(ells.size());
    for (const auto& e : ells) {
        Ell o{};
        /* 与 matcher_c.cpp::sbm_detect_ellipses 505-511 行完全一致的换算 */
        o.cx = e->o.y;                    /* o.y 是列 -> cx */
        o.cy = e->o.x;                    /* o.x 是行 -> cy */
        o.a = e->a;
        o.b = e->b;
        o.phi = CV_PI / 2.0 - e->phi;
        o.goodness = e->goodness;
        o.coverangle = e->coverangle;
        out.push_back(o);
    }
    return out;
}

/* ---------------- 路线 B: AAMED ---------------- */

static std::vector<Ell> run_aamed(const cv::Mat& gray, sbm::AamedDetector& det,
                                  const sbm::AamedParams& p) {
    std::vector<Ell> out;
    det.detect(gray, p, out);
    return out;   // AamedEllipse 与 sbm_ellipse_t 同构, 可直接用
}

/* ---------------- 精度评估 ---------------- */

struct Metrics {
    int tp = 0, fp = 0, fn = 0;
    double mean_iou = 0.0;
    double precision() const { return tp + fp ? (double)tp / (tp + fp) : 0.0; }
    double recall()    const { return tp + fn ? (double)tp / (tp + fn) : 0.0; }
    double f1()        const {
        const double pp = precision(), rr = recall();
        return (pp + rr) > 0 ? 2 * pp * rr / (pp + rr) : 0.0;
    }
};

/* 真值 vs 预测 的一对一贪心匹配 (IoU 阈值以上才算 TP) */
static Metrics evaluate(const cv::Size& sz,
                        const std::vector<Ell>& truth,
                        const std::vector<Ell>& pred,
                        double iou_thr = 0.5) {
    Metrics m;
    m.fn = static_cast<int>(truth.size());
    const size_t P = pred.size();
    std::vector<std::vector<double>> iou(truth.size(), std::vector<double>(P, 0.0));
    std::vector<cv::Mat> pm;
    pm.reserve(P);
    for (size_t j = 0; j < P; ++j) pm.push_back(fill_mask(sz, pred[j]));

    for (size_t i = 0; i < truth.size(); ++i) {
        const cv::Mat tm = fill_mask(sz, truth[i]);
        for (size_t j = 0; j < P; ++j) iou[i][j] = iou_of(tm, pm[j]);
    }

    std::vector<std::pair<double, std::pair<size_t, size_t>>> pairs;
    for (size_t i = 0; i < truth.size(); ++i)
        for (size_t j = 0; j < P; ++j)
            if (iou[i][j] >= iou_thr) pairs.emplace_back(iou[i][j], std::make_pair(i, j));
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& x, const auto& y) { return x.first > y.first; });

    std::vector<char> ut(truth.size(), 0), up(P, 0);
    double iou_sum = 0;
    for (const auto& q : pairs) {
        if (ut[q.second.first] || up[q.second.second]) continue;
        ut[q.second.first] = 1;
        up[q.second.second] = 1;
        ++m.tp;
        iou_sum += q.first;
    }
    m.fp = static_cast<int>(P) - m.tp;
    m.fn = static_cast<int>(truth.size()) - m.tp;
    m.mean_iou = m.tp ? iou_sum / m.tp : 0.0;
    return m;
}

/* ---------------- 合成场景 ---------------- */

struct Scene {
    const char* name;
    cv::Size size;
    std::vector<Ell> truth;
    double noise;      /* 高斯噪声 sigma; 0 = 干净 */
    double blur;       /* 高斯模糊 sigma; 0 = 不模糊 */
    int occlude;       /* 遮挡矩形宽度(0 = 不遮挡) */
};

static Ell mk(double cx, double cy, double a, double b, double phi_deg) {
    Ell e{};
    e.cx = cx; e.cy = cy; e.a = a; e.b = b;
    e.phi = phi_deg * CV_PI / 180.0;
    e.goodness = 1.0;
    return e;
}

static std::vector<Scene> build_scenes() {
    using S = Scene;
    const cv::Size sz(640, 480);
    std::vector<Scene> scenes;

    {   /* 完整清晰椭圆: 两算法的主战场 */
        S s{"full-circle", sz, {}, 0, 0, 0};
        s.truth = { mk(160, 150, 90, 90, 0),
                    mk(450, 170, 110, 60, 30),
                    mk(320, 380, 80, 80, 0) };
        scenes.push_back(s);
    }
    {   /* 小目标 */
        S s{"small", sz, {}, 0, 0, 0};
        s.truth = { mk(120, 110, 22, 22, 0),
                    mk(300, 150, 28, 20, 45),
                    mk(480, 120, 25, 25, 0),
                    mk(520, 380, 24, 18, 20) };
        scenes.push_back(s);
    }
    {   /* 局部残缺: 只画约 2/3 圆弧 */
        S s{"partial-arc", sz, {}, 0, 0, 0};
        s.truth = { mk(160, 160, 70, 70, 0),
                    mk(420, 200, 80, 45, 15),
                    mk(300, 380, 60, 60, 0) };
        s.occlude = 100;   // 一条竖白带压过去, 切掉一段弧
        scenes.push_back(s);
    }
    {   /* 细长椭圆 */
        S s{"elongated", sz, {}, 0, 0, 0};
        s.truth = { mk(200, 200, 120, 28, 10),
                    mk(430, 320, 100, 24, 70),
                    mk(300, 120, 80, 20, 0) };
        scenes.push_back(s);
    }
    {   /* 低对比 + 噪声 */
        S s{"noisy-lowcontrast", sz, {mk(180, 180, 85, 85, 0),
                                      mk(420, 300, 70, 40, 25)}, 30.0, 1.5, 0};
        scenes.push_back(s);
    }
    {   /* 密集 + 遮挡 */
        S s{"dense-occluded", sz, {}, 0, 0, 0};
        s.truth = { mk(140, 140, 45, 45, 0),
                    mk(250, 150, 45, 25, 30),
                    mk(360, 140, 45, 45, 0),
                    mk(470, 160, 40, 30, 60),
                    mk(180, 360, 50, 35, 15),
                    mk(420, 380, 55, 55, 0),
                    mk(520, 300, 42, 30, 0) };
        s.noise = 12.0;
        s.occlude = 110;
        scenes.push_back(s);
    }
    {   /* 1080p 大图: 看时间/内存随尺寸的增长曲线 (AAMED 按像素规模预分配) */
        S s{"hd-1080p", cv::Size(1920, 1080), {}, 0, 0, 0};
        s.truth = { mk(500, 400, 260, 260, 0),
                    mk(1300, 500, 300, 170, 25),
                    mk(950, 850, 220, 220, 0) };
        scenes.push_back(s);
    }
    return scenes;
}

static cv::Mat synth(const Scene& s) {
    cv::Mat img(s.size, CV_8UC1, cv::Scalar(255));
    for (const Ell& e : s.truth) cv::ellipse(img, rect_of(e), cv::Scalar(0), 2);
    if (s.occlude > 0)
        cv::rectangle(img, cv::Rect(s.size.width / 2 - s.occlude, 0, s.occlude * 2, s.size.height),
                      cv::Scalar(255), cv::FILLED);
    if (s.blur > 0) {
        cv::Mat b;
        cv::GaussianBlur(img, b, cv::Size(0, 0), s.blur);
        img = b;
    }
    if (s.noise > 0) {
        cv::Mat n(s.size, CV_8UC1);
        cv::randn(n, cv::Scalar(0), cv::Scalar(s.noise));
        cv::add(img, n, img, cv::noArray(), CV_8U);
    }
    return img;
}

/* ---------------- 主流程 ---------------- */

static double now_ms() {
    using namespace std::chrono;
    const auto t = steady_clock::now().time_since_epoch();
    return duration<double, std::milli>(t).count();
}



int main(int argc, char** argv) {
    int iters = 7;
    zgh::DetectParams sp;
    sbm::AamedParams ap = sbm::aamed_params_default();
    std::string dump;
    std::vector<std::string> real_imgs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) sp.num_threads = std::atoi(argv[++i]);
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (a == "--real" && i + 1 < argc) {
            std::string list = argv[++i];
            size_t p = 0;
            while (p <= list.size()) {
                const size_t c = list.find(',', p);
                real_imgs.push_back(list.substr(p, c == std::string::npos ? c : c - p));
                if (c == std::string::npos) break;
                p = c + 1;
            }
        }
    }
    iters = std::max(iters, 1);
    if (sp.num_threads <= 0) sp.num_threads = 0;   // 0 = 自动
    kMs = 1.0;

    std::printf("==== ellipse_compare: standard-ellipse-detection vs AAMED ====\n");
    std::printf("iters=%d  threads=%d  AAMED(theta_fsa=%.2f, length_fsa=%.2f, t_val=%.2f, nms=%.2f)\n\n",
                iters, sp.num_threads, ap.theta_fsa, ap.length_fsa, ap.t_val, ap.nms_iou);

    std::vector<Scene> scenes = build_scenes();
    struct Row { const char* name; double ms_std, ms_aam, f1_std, f1_aam, iou_std, iou_aam;
                 int tp_std, tp_aam, fp_std, fp_aam, n_truth; };
    std::vector<Row> rows;

    for (const Scene& sc : scenes) {
        const cv::Mat img = synth(sc);
        sbm::AamedDetector det;
        // 预热一轮(不计入统计), 让 AAMED 完成内存分配 + 代码走热
        run_standard(img, sp);
        run_aamed(img, det, ap);

        std::vector<double> t_std, t_aamed;
        std::vector<Ell> p_std, p_aam;
        for (int i = 0; i < iters; ++i) {
            // 交错: A/B 交替, 抵消机器状态漂移
            {
                const double t0 = now_ms();
                p_std = run_standard(img, sp);
                t_std.push_back(now_ms() - t0);
            }
            {
                const double t0 = now_ms();
                p_aam = run_aamed(img, det, ap);
                t_aamed.push_back(now_ms() - t0);
            }
        }

        const Metrics ms = evaluate(sc.size, sc.truth, p_std);
        const Metrics ma = evaluate(sc.size, sc.truth, p_aam);

        const double msStd = median_of(t_std), msAam = median_of(t_aamed);
        rows.push_back(Row{sc.name, msStd, msAam, ms.f1(), ma.f1(), ms.mean_iou, ma.mean_iou,
                           ms.tp, ma.tp, ms.fp, ma.fp, (int)sc.truth.size()});

        std::printf("[%-20s] %dx%d  truth=%zu\n", sc.name, sc.size.width, sc.size.height,
                    sc.truth.size());
        std::printf("    standard : TP=%2d/%-2d FP=%2d  mIoU=%.3f  F1=%.3f   median %8.2f ms"
                    "  (mean %.2f, min %.2f)\n",
                    ms.tp, (int)sc.truth.size(), ms.fp, ms.mean_iou, ms.f1(),
                    msStd, mean_of(t_std), *std::min_element(t_std.begin(), t_std.end()));
        std::printf("    AAMED    : TP=%2d/%-2d FP=%2d  mIoU=%.3f  F1=%.3f   median %8.2f ms"
                    "  (mean %.2f, min %.2f)\n",
                    ma.tp, (int)sc.truth.size(), ma.fp, ma.mean_iou, ma.f1(),
                    msAam, mean_of(t_aamed), *std::min_element(t_aamed.begin(), t_aamed.end()));
        std::printf("    speed    : AAMED / standard = %.2fx  (standard is %.1fx faster)\n\n",
                    msStd > 0 ? msAam / msStd : 0.0, msAam > 0 ? msStd / msAam : 0.0);

        if (!dump.empty()) {
            cv::Mat vis;
            cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
            for (const Ell& e : sc.truth)
                cv::ellipse(vis, rect_of(e), cv::Scalar(255, 0, 0), 2);
            for (const Ell& e : p_std) cv::ellipse(vis, rect_of(e), cv::Scalar(0, 255, 0), 1);
            for (const Ell& e : p_aam) cv::ellipse(vis, rect_of(e), cv::Scalar(0, 0, 255), 1);
            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s/%s.jpg", dump.c_str(), sc.name);
            cv::imwrite(buf, vis);
        }
    }

    /* ---- 汇总表 ---- */
    std::printf("\n================ 汇总 (median ms, IoU thr=0.50) ================\n");
    std::printf("%-20s | %-28s | %-28s | %8s\n", "scene", "standard (ms / mIoU / F1)",
                "AAMED (ms / mIoU / F1)", "AAMED/standard");
    std::printf("--------------------+---------------------------+---------------------------+----------\n");
    double agg_std_ms = 0, agg_aam_ms = 0, agg_f1_std = 0, agg_f1_aam = 0;
    for (const Row& r : rows) {
        agg_std_ms += r.ms_std; agg_aam_ms += r.ms_aam;
        agg_f1_std += r.f1_std; agg_f1_aam += r.f1_aam;
        std::printf("%-20s | %7.2f ms  %.3f  %.3f | %7.2f ms  %.3f  %.3f | %8.2fx\n",
                    r.name, r.ms_std, r.iou_std, r.f1_std,
                    r.ms_aam, r.iou_aam, r.f1_aam,
                    r.ms_std > 0 ? r.ms_aam / r.ms_std : 0.0);
    }
    const size_t n = rows.size();
    std::printf("-------------------------------------------------------------------------------\n");
    std::printf("TOTAL (sum of scenes, F1 = mean): standard %.2f ms / F1 %.3f  |  "
                "AAMED %.2f ms / F1 %.3f  |  AAMED takes %.2fx\n",
                agg_std_ms, agg_f1_std / n, agg_aam_ms, agg_f1_aam / n,
                agg_std_ms > 0 ? agg_aam_ms / agg_std_ms : 0.0);

    /* ---- 真实图像(无真值, 只看检出数 + 速度) ---- */
    if (!real_imgs.empty()) {
        std::printf("\n================ 真实图像 (无真值, 只看检出数与速度) ================\n");
        for (const std::string& path : real_imgs) {
            cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
            if (img.empty()) { std::printf("  skip (cannot read): %s\n", path.c_str()); continue; }
            cv::Mat gray;
            if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            else gray = img;

            sbm::AamedDetector det;
            double t_first_aam = 0;   // 首次调用含内存分配, 单独记
            std::vector<double> t_std, t_aam;
            std::vector<Ell> p_std, p_aam;
            {
                const double t0 = now_ms();
                p_aam = run_aamed(gray, det, ap);
                t_first_aam = now_ms() - t0;
                p_std = run_standard(gray, sp);
            }
            for (int i = 0; i < iters; ++i) {
                const double a0 = now_ms();
                p_std = run_standard(gray, sp);
                t_std.push_back(now_ms() - a0);
                const double b0 = now_ms();
                p_aam = run_aamed(gray, det, ap);
                t_aam.push_back(now_ms() - b0);
            }
            const double gx_std = (p_std.empty() ? 0.0 : p_std.front().goodness);
            const double gx_aam = (p_aam.empty() ? 0.0 : p_aam.front().goodness);
            std::printf("  %-46s %dx%d\n", path.c_str(), gray.cols, gray.rows);
            std::printf("    standard : %3zu ellipses (top goodness %.3f)  median %8.2f ms\n",
                        p_std.size(), gx_std, median_of(t_std));
            std::printf("    AAMED    : %3zu ellipses (top goodness %.3f)  median %8.2f ms"
                        "  [first call incl. alloc: %.2f ms]\n",
                        p_aam.size(), gx_aam, median_of(t_aam), t_first_aam);
        }
    }
    return 0;
}
