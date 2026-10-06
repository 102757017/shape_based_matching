/* ============================================================================
 * AAMED (Arc Adjacency Matrix based Fast Ellipse Detection) 的 C++ 适配层。
 *
 * 底层是 src/ellipse/aamed 里的上游实现 (FLED / AAMED 类), 那套代码直接吐
 * vector<cv::RotatedRect> + 平行数组 vector<double> 评分, 字段语义和本工程
 * 对外统一的椭圆口径不一致。这一层只做三件事:
 *   1. 参数归一化 (把上游 3 参数的 SetParameters 扩展成可裁剪评分/重叠门槛的
 *      AamedParams);
 *   2. 结果归一化成与 sbm_detect_ellipses (standard-ellipse-detection) 完全
 *      一致的口径: cx=列(x) / cy=行(y) / phi 为相对列轴的弧度 (cv::ellipse 口径),
 *      这样两种检测器产出的 sbm_ellipse_t 可以混用、共用同一套后处理;
 *   3. 复用 FLED 实例 —— 上游 FLED 构造时会 new 出 drows*dcols 个 Node_FC,
 *      一张 1080p 图约 550 MB, 逐帧重建是不可接受的, 所以这里按"只增不减"
 *      的策略缓存实例, 图像变大时才重建。
 *
 * 坐标/角度口径 (与 ellipse_detection 那一路保持一致):
 *   cx, cy     中心, 单位是像素, 分别对应 列(x) / 行(y)
 *   a, b       半长轴 / 半短轴 (注意: cv::RotatedRect::size 存的是全轴长, 要除以 2)
 *   phi        相对 x(列)轴的旋转角, 弧度; cv::RotatedRect::angle 是角度, 要乘 PI/180
 *
 * 与 zgh::detectEllipse 的差别:
 *   coverangle 字段 AAMED 侧不计算 (它的验证器算的是 goodness 评分而不是弧覆盖
 *   角), 所以这里恒填 0, 不要拿它当判据; 比较椭圆质量请用 goodness。
 * ==========================================================================*/
#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

// src/ellipse/aamed/include 已在 aamed target 的 PUBLIC include path 里,
// 所以直接按相对路径引上游头文件。
#include "FLED.h"

namespace sbm {

/* ---------------- AAMED 检测参数 ----------------
 * 前三个参数与上游 FLED::SetParameters(theta_fsa, length_fsa, T_val) 一一对应;
 * 后两个是本工程加的, 用于在上游内部 NMS 之后再过一道筛子。 */
struct AamedParams {
    /* 邻域分组 (FSA) 的角度约束, 单位弧度。上游默认 CV_PI/3 (即 60°)。
     * 调大 -> 分组更宽松, 大椭圆更容易成组但误检增多; 调小反之。 */
    double theta_fsa;
    /* 邻域分组 (FSA) 的长度约束。上游默认 3.4。 */
    double length_fsa;
    /* 椭圆验证阶段的评分门槛, 0~1, 上游默认 0.77。
     * 调低能检出更残缺的椭圆, 但拟合偏差会变大。 */
    double t_val;
    /* 最终输出的评分门槛: goodness 低于它的椭圆直接丢弃, <=0 表示不筛。
     * 默认 0.0 (=不筛), 想只留"干净"的椭圆可以设成 0.6 左右。 */
    double min_goodness;
    /* 非极大抑制的重叠 (IoU) 门槛, 已端到端透传到上游
     * FLED::EllipseNonMaximumSuppression 的 T_iou (run_FLED 结尾那轮 NMS,
     * 仅当 SELECT_CLUSTER_METHOD==OUR_CLUSTER_METHOD 时生效)。
     * 语义: 两椭圆 IoU > nms_iou 时, 低分者被抑制。
     * 因此调大(如 0.9)=更宽松, 保留更多彼此贴近的椭圆; 调小(如 0.3)=更严格。
     * <=0 表示沿用上游默认 0.7。 */
    double nms_iou;
    /* 动态验证门槛开关。开启后忽略 t_val, 改用论文式自适应值:
     *   T_val = 0.7 + 0.3*(1 - exp(-2*contrast)),  contrast = std(gray)/mean(gray)
     * 高对比(边缘干净)自动收紧 -> 抑制金属反光类"幽灵椭圆"假阳性;
     * 低对比自动放松 -> 减少残缺椭圆的漏检。取值夹在 [0.5, 0.98]。
     * 默认 false (沿用 t_val)。可用 last_t_val() 取实际生效值。 */
    bool adaptive_tval;
};

/* ---------------- 单个椭圆结果 ----------------
 * 布局与 sbm_ellipse_t 一致, 可直接 memcpy 到 C ABI 的结构体里。 */
struct AamedEllipse {
    double cx, cy;      /* 中心: 列(x), 行(y) */
    double a, b;        /* 半长轴, 半短轴 */
    double phi;         /* 相对 x(列)轴的旋转角, 弧度 (cv::ellipse 口径) */
    double goodness;    /* 验证评分 (0~1, 越高越好) */
    double coverangle;  /* AAMED 不产出该量, 恒为 0 */
};

/* ---------------- 检测器 ----------------
 * 有状态: 构造/内部扩容时会按 drows*dcols*272 Byte 量级的内存分配,
 * 所以重量级读图场景请长期持有同一个 AamedDetector 复用
 * (例如每帧调用 detector.detect(...)), 不要每次 new 一个。 */
class AamedDetector {
public:
    AamedDetector() = default;
    ~AamedDetector() = default;

    // FLED 内含 cv::flann::Index 与裸指针成员, 不可拷贝/赋值
    AamedDetector(const AamedDetector&) = delete;
    AamedDetector& operator=(const AamedDetector&) = delete;

    /* 在灰度图上跑一遍 AAMED。
     * out 会先清空再写入, 按 goodness 从大到小排序。
     * 返回实际写出的椭圆个数; 抛异常由调用方 (C ABI 层) 兜底。 */
    int detect(const cv::Mat& gray, const AamedParams& params,
               std::vector<AamedEllipse>& out);

    /* 预分配到能容纳 rows x cols 的图像 (等价于第一次 detect 该尺寸 image)。
     * 显式调用它可以让"创建检测器 -> 立刻分配内存"这一步提前发生,
     * 便于把大内存分配失败的风险放在可控的位置。rows/cols 传 <=0 表示跳过。 */
    void prepare(int rows, int cols) { if (rows > 0 && cols > 0) ensure_capacity(rows, cols); }

    /* 当前实例能容纳的最大图像尺寸 (rows / cols), 用于判断是否需要重建。 */
    int capacity_rows() const { return rows_; }
    int capacity_cols() const { return cols_; }

    /* 最近一次 detect() 实际生效的验证门槛 (adaptive_tval 开时即论文式动态值,
     * 否则等于 params.t_val)。便于排查"为什么这次检出的椭圆变少/变多"。 */
    double last_t_val() const { return last_t_val_; }

private:
    void ensure_capacity(int rows, int cols);

    std::unique_ptr<AAMED> aamed_;   // AAMED 是 FLED 的 typedef
    int rows_ = 0;
    int cols_ = 0;
    double last_t_val_ = 0.0;
};

inline void AamedDetector::ensure_capacity(int rows, int cols) {
    // 上游约束: drows/dcols 必须 >= 图像 rows/cols (checkInputImage 用 > 比较)
    if (aamed_ && rows_ >= rows && cols_ >= cols) return;
    aamed_.reset(new AAMED(rows, cols));
    rows_ = rows;
    cols_ = cols;
}

inline int AamedDetector::detect(const cv::Mat& gray, const AamedParams& params,
                                 std::vector<AamedEllipse>& out) {
    out.clear();
    if (gray.empty() || gray.type() != CV_8UC1) {
        throw std::invalid_argument("AamedDetector::detect: gray must be a non-empty CV_8UC1 image");
    }

    ensure_capacity(gray.rows, gray.cols);

    // 上游 FLED::run_FLED 内部会 GaussianBlur(Img_G, Img_G, ...) 就地改图,
    // 这里传副本, 免得把调用方的图弄脏。
    cv::Mat work;
    gray.copyTo(work);

    // 动态验证门槛 (论文式): 按输入灰度图对比度自适应。
    // 高对比 -> 边缘更干净 -> 可以收紧 goodness 门槛 (减少金属反光类幽灵椭圆);
    // 低对比 -> 放松 (避免漏掉残缺椭圆)。用原始 gray 算, 因为 run_FLED 内部
    // 还会再 GaussianBlur 一遍, 那一步会人为压低对比度。
    double t_val = params.t_val;
    if (params.adaptive_tval) {
        cv::Scalar mu, sigma;
        cv::meanStdDev(gray, mu, sigma);
        const double mean = mu[0] > 1e-6 ? mu[0] : 1e-6;
        const double contrast = sigma[0] / mean;   // MATLAB: contrast = std2/mean2
        t_val = 0.7 + 0.3 * (1.0 - std::exp(-2.0 * contrast));
        if (t_val < 0.5) t_val = 0.5;
        if (t_val > 0.98) t_val = 0.98;
    }
    last_t_val_ = t_val;

    aamed_->SetParameters(params.theta_fsa, params.length_fsa, t_val);
    // nms_iou 端到端透传到上游 NMS: 实际生效的是 run_FLED 结尾那轮
    // EllipseNonMaximumSuppression(T_iou), 这里的阈值经 <=0 -> 0.7 归一后传入。
    const double nms_iou = params.nms_iou > 0.0 ? params.nms_iou : 0.7;
    aamed_->SetNMSIoU(nms_iou);
    aamed_->run_FLED(work);

    const std::vector<cv::RotatedRect>& ells = aamed_->detEllipses;
    const std::vector<double>& scores = aamed_->GetDetEllipseScore();
    const int n = static_cast<int>(ells.size());
    const int n_score = static_cast<int>(scores.size());

    const double min_goodness = params.min_goodness > 0.0 ? params.min_goodness : 0.0;

    // 归一化 + 过滤(仅评分门槛) + 按评分降序排序。
    // 去重(NMS)已在 run_FLED 内部用透传的 nms_iou 完成, 这里不再二次 NMS。
    struct Cand { double score; AamedEllipse e; };
    std::vector<Cand> cands;
    cands.reserve(n);
    for (int i = 0; i < n; ++i) {
        const double score = (i < n_score) ? scores[i] : 0.0;
        if (score < min_goodness) continue;

        const cv::RotatedRect& rr = ells[i];
        Cand c{};
        c.score = score;
        /* 这里必须做转置 —— 上游 AAMED 内部用的是 (行, 列) 平面, 与 OpenCV 的
         * (列, 行) 相反 (和 zgh::Ellipse 的 o.x=行 一样)。
         *
         *   1) 坐标转置: center.x 是行 -> cy, center.y 是列 -> cx
         *   2) 角度重定基准: rr.angle 是 (行,列) 平面里从"行轴"起算的角度,
         *      而对外口径 phi 是 (列,行) 平面里从"列轴"起算的。这两个平面互为
         *      转置 (行列互换 -> 行列式 -1 -> 手性反转), 所以不是简单取负,
         *      正确换算是 phi = 90° - angle。
         *   3) size.width 是"局部 x 方向"的全轴长, 转置后仍对应局部 x 方向,
         *      所以 a/b 各取 width/2、height/2 即可, 不需要交换。
         *
         * 注意: 网上/common 说法是"center 交换 + size 交换 + angle 取负"
         * (上游 drawEllipses() 就是这么画的), 那个式子等价于本处 phi=90°-angle
         * 再交换 a/b, 两者几何上都能画对, 但只有本处写法能直接对上
         * sbm_ellipse_t 的 (a=沿局部x半轴, b=沿局部y半轴) 语义。
         *
         * 校验方式 (x/aamed_fitcheck, 已随调试删掉):
         *   合成 6 个不同旋转角的椭圆圆环喂给 AAMED, 取上游原始 RotatedRect,
         *   对 5 种候选映射各自还原成 cv::RotatedRect 后与真值掩膜算 IoU:
         *       phi = 90° - angle      -> IoU 0.755 (稳定)
         *       phi = -angle           -> IoU 0.422
         *   IoU 从 0.42 提到 0.75, 差异就是"长/短轴整体转了 90°"。
         */
        c.e.cx = rr.center.y;
        c.e.cy = rr.center.x;
        c.e.a = rr.size.width * 0.5;
        c.e.b = rr.size.height * 0.5;
        c.e.phi = (90.0 - rr.angle) * static_cast<double>(CV_PI) / 180.0;
        c.e.goodness = score;
        c.e.coverangle = 0.0;   // 上游不产出, 见文件头
        cands.push_back(c);
    }

    std::sort(cands.begin(), cands.end(),
              [](const Cand& x, const Cand& y) { return x.score > y.score; });

    out.reserve(cands.size());
    for (const Cand& c : cands) out.push_back(c.e);
    return static_cast<int>(out.size());
}

/* ---------------- 默认参数 ----------------
 * 与上游 main.cpp 里的写法一致: SetParameters(CV_PI / 3, 3.4, 0.77)。 */
inline AamedParams aamed_params_default() {
    AamedParams p{};
    p.theta_fsa = CV_PI / 3.0;   // 60°
    p.length_fsa = 3.4;
    p.t_val = 0.77;
    p.min_goodness = 0.0;        // 不筛
    p.nms_iou = 0.7;             // 沿用上游内部 NMS 门槛
    return p;
}

}  // namespace sbm
