/* 临时校验 (调试用, 验完即删): AAMED 适配层输出的 (cx, cy, a, b, phi) 能否还原输入几何。
 * 用 cv::ellipse 画已知真值的圆环喂给 AamedDetector, 取评分最高的那个输出原样还原成
 * cv::ellipse 掩膜, 与真值掩膜算 IoU; 同时出叠加图肉眼看。 */
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <vector>

#include "aamed_detector.h"

static cv::Mat maskOf(const cv::Mat& img, const sbm::AamedEllipse& e) {
    cv::Mat m = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::ellipse(m, cv::Point2f((float)e.cx, (float)e.cy),
                cv::Size2f((float)e.a, (float)e.b),
                (float)(e.phi * 180.0 / CV_PI), 0, 360, cv::Scalar(1), cv::FILLED);
    return m;
}

int main() {
    const int W = 460, H = 460;
    const double gt_cx = 230.0, gt_cy = 230.0, gt_a = 115.0, gt_b = 58.0;
    const double angles[] = {0.0, 30.0, 60.0, 90.0, 120.0, 150.0};

    std::printf("%-8s %-11s %-9s %-6s\n", "gtAngle", "count/good", "IoU", "n");
    double sum = 0.0;
    int cnt = 0;

    for (double ang : angles) {
        cv::Mat img = cv::Mat::zeros(H, W, CV_8UC1);
        img.setTo(0);
        cv::ellipse(img, cv::Point2f((float)gt_cx, (float)gt_cy),
                    cv::Size((int)gt_a, (int)gt_b), (float)ang,
                    0, 360, cv::Scalar(255), 4);

        cv::Mat gt = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::ellipse(gt, cv::Point2f((float)gt_cx, (float)gt_cy),
                    cv::Size((int)(gt_a - 12), (int)(gt_b - 6)), (float)ang,
                    0, 360, cv::Scalar(1), cv::FILLED);

        sbm::AamedDetector det;
        det.prepare(H, W);
        std::vector<sbm::AamedEllipse> ells;
        det.detect(img, sbm::aamed_params_default(), ells);

        // 取评分最高的
        sbm::AamedEllipse bestE{};
        double best = -1.0;
        for (const sbm::AamedEllipse& e : ells) if (e.goodness > best) { best = e.goodness; bestE = e; }

        cv::Mat out = ells.empty() ? cv::Mat::zeros(img.size(), CV_8UC1) : maskOf(img, bestE);

        cv::Mat inter;
        cv::bitwise_and(gt, out, inter);
        int ni = cv::countNonZero(inter),
            nu = cv::countNonZero(gt),
            nm = cv::countNonZero(out);
        int uni = ni + nu + nm - 2 * ni;
        double iou = uni > 0 ? (double)ni / (double)uni : 0.0;

        std::printf("%-8.1f %-11.2f %-9.4f %-6zu", ang, best, iou, ells.size());
        for (size_t i = 0; i < ells.size(); ++i)
            std::printf(" | [%zu] c=(%.0f,%.0f) a=%.0f b=%.0f phi=%.1f",
                        i, ells[i].cx, ells[i].cy, ells[i].a, ells[i].b,
                        ells[i].phi * 180.0 / CV_PI);
        std::printf("\n");

        sum += iou;
        cnt++;

        cv::Mat overlay = img.clone();
        cv::ellipse(overlay, cv::Point2f((float)gt_cx, (float)gt_cy),
                    cv::Size((int)gt_a, (int)gt_b), (float)ang, 0, 360,
                    cv::Scalar(0, 0, 255), 4);
        if (!ells.empty())
            cv::ellipse(overlay, cv::Point2f((float)bestE.cx, (float)bestE.cy),
                        cv::Size2f((float)bestE.a, (float)bestE.b),
                        (float)(bestE.phi * 180.0 / CV_PI), 0, 360,
                        cv::Scalar(0, 255, 0), 4);
        char fn[64];
        std::sprintf(fn, "fit_%03.0f.png", ang);
        cv::Mat bgr;
        cv::cvtColor(overlay, bgr, cv::COLOR_GRAY2BGR);
        cv::imwrite(fn, bgr);
    }
    std::printf("MEAN IoU = %.4f\n", sum / cnt);
    return 0;
}
