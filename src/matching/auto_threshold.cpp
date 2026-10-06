// ============================================================================
// 训练阈值(弱/强)自动探测
//
// 背景: line2D 的 weak_thresh / strong_thresh 是"贴着图走"的参数 ——
//   强阈值 = 训练图上梯度够强才能当模板特征点,
//   弱阈值 = 场景图上多弱的边可以参与匹配,
// 不同图片对比度差一个量级, 手调非常痛苦。这里直接按训练图自身的梯度分布搜索。
//
// 做法(全部复用训练时的口径, 绝不另起炉灶):
//   1. 按 train() 同样的方式做 padding / 组合掩码(正向-负向-排除区);
//   2. 梯度分布直接取自训练同款的 ColorGradientPyramid(它是 public 成员),
//      而不是自己复刻一遍 Sobel, 免得口径漂移;
//   3. 沿"梯度幅值的分位数"撒一遍强阈值候选, 每个候选真实跑一遍
//      ColorGradientPyramid(弱=比例*强) + extractTemplate,
//      拿到候选点数 n_cand 与实际特征点数 n_feat;
//   4. 打分选最优:
//        - 候选点数相对期望特征点数的富余度接近 3 倍(太少填不满点, 太多摊得太散且掺噪声)
//        - 特征点数达到期望值的 60% 以上(至少 10 个)
//   5. 可选做一次"自匹配自检": 用训练图自己当场景匹配一次, 给出置信度做冒烟测试。
// ============================================================================
#include "matcher_core.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <streambuf>
#include <vector>

namespace sbm {

namespace {

// ---------------------------------------------------------------------------
// line2Dup 的 extractTemplate 在特征点不够时会往 std::cout 打日志, 扫阈值时会刷屏,
// 这里临时把 cout 接到一个空缓冲上(不影响别处输出)。
// ---------------------------------------------------------------------------
class NullStreamBuf : public std::streambuf {
public:
    int overflow(int c) override { return c == std::char_traits<char>::eof() ? 0 : c; }
    int sync() override { return 0; }
};

class CoutSuspend {
public:
    CoutSuspend() : prev_(std::cout.rdbuf(&buf_)) {}
    ~CoutSuspend() { std::cout.rdbuf(prev_); }
    CoutSuspend(const CoutSuspend&) = delete;
    CoutSuspend& operator=(const CoutSuspend&) = delete;

private:
    NullStreamBuf buf_;
    std::streambuf* prev_;
};

// 已排序序列的分位取值
float quantile_sorted(const std::vector<float>& v, float q) {
    if (v.empty()) return 0.f;
    double pos = static_cast<double>(v.size() - 1) * static_cast<double>(q);
    size_t i = static_cast<size_t>(pos);
    if (i >= v.size() - 1) return v.back();
    double f = pos - static_cast<double>(i);
    return v[i] * (1.0f - static_cast<float>(f)) + v[i + 1] * static_cast<float>(f);
}

// 复刻 extractTemplate 里的候选点判定: 5x5 局部极大 + 幅值过强阈值 + 方向量化有效
int count_candidates(const cv::Mat& magnitude,
                     const cv::Mat& angle,
                     const cv::Mat& local_mask,
                     float strong) {
    const int k = 2;   // nms_kernel_size / 2
    const float threshold_sq = strong * strong;
    int count = 0;

    for (int r = k; r < magnitude.rows - k; ++r) {
        const float* mag_center = magnitude.ptr<const float>(r);
        const float* mag_up1 = magnitude.ptr<const float>(r - 1);
        const float* mag_up2 = magnitude.ptr<const float>(r - 2);
        const float* mag_dn1 = magnitude.ptr<const float>(r + 1);
        const float* mag_dn2 = magnitude.ptr<const float>(r + 2);
        const uchar* mask_r = local_mask.empty() ? nullptr : local_mask.ptr<const uchar>(r);
        const uchar* ang_r = angle.ptr<const uchar>(r);

        for (int c = k; c < magnitude.cols - k; ++c) {
            if (mask_r && mask_r[c] == 0) continue;
            const float* mag_r = mag_center + c;
            float score = mag_r[0];
            if (score <= threshold_sq) continue;
            if (ang_r[c] == 0) continue;

            bool is_max = true;
            for (int dr = -k; dr <= k && is_max; ++dr) {
                const float* row = dr == -2 ? mag_up2 : (dr == -1 ? mag_up1 :
                                   (dr == 1 ? mag_dn1 : mag_dn2));
                const float* p = row + c;
                for (int dc = -k; dc <= k; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    if (score < p[dc]) { is_max = false; break; }
                }
            }
            if (is_max) ++count;
        }
    }
    return count;
}

// 最终阈值下再跑一次, 把特征点画到 ROI 上, 便于肉眼核对
cv::Mat draw_features_on_roi(const cv::Mat& padded_img,
                             const cv::Mat& padded_mask,
                             float weak, float strong, int feature_num,
                             int padding, int w, int h) {
    cv::Mat preview = padded_img(cv::Rect(padding, padding, w, h)).clone();
    line2Dup::Template templ;
    {
        CoutSuspend muted;
        line2Dup::ColorGradientPyramid pyr(padded_img, padded_mask, weak,
                                           static_cast<size_t>(feature_num), strong);
        if (!pyr.extractTemplate(templ)) return preview;
        // 注意: ColorGradientPyramid::extractTemplate 只填 width/height(-1)/pyramid_level,
        // 不填 tl_x/tl_y (那是 addTemplate -> cropTemplates 才做的)。特征点坐标
        // 直接就是这张 padded 图上的坐标, 减掉 padding 才是 ROI 内坐标。
        for (const auto& f : templ.features) {
            int px = static_cast<int>(f.x) - padding;
            int py = static_cast<int>(f.y) - padding;
            if (px >= 0 && px < w && py >= 0 && py < h)
                cv::circle(preview, cv::Point(px, py), 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    return preview;
}

}  // namespace

// ===========================================================================
ThresholdEstimate estimate_train_thresholds(
    const cv::Mat& train_image, const std::vector<int>& roi,
    const cv::Mat& positive_mask, const cv::Mat& negative_mask,
    const std::vector<ExclusionZone>& exclusion_zones,
    const ThresholdSearchOptions& opt) {
    ThresholdEstimate out;
    out.requested_features = opt.feature_num > 0 ? opt.feature_num : 100;
    out.ok = false;

    // ---------- 1. 参数校验 / ROI 裁剪(与 train() 保持同样的口径) ----------
    if (train_image.empty()) throw std::invalid_argument("训练图为空。");
    if (roi.size() != 4) throw std::invalid_argument("roi 必须是 [x, y, w, h]。");
    const int x = roi[0], y = roi[1], w = roi[2], h = roi[3];
    if (w <= 0 || h <= 0) throw std::invalid_argument("ROI区域为空或无效。");
    cv::Rect roi_rect(x, y, w, h);
    if (roi_rect.x < 0 || roi_rect.y < 0 ||
        roi_rect.x + roi_rect.width > train_image.cols ||
        roi_rect.y + roi_rect.height > train_image.rows)
        throw std::invalid_argument("ROI区域超出训练图范围。");
    cv::Mat roi_image = train_image(roi_rect).clone();
    // line2D 内部的 Sobel 在 16 位输入上会读到越界内存, 统一按实际最大值压到 0~255,
    // 这样阈值单位也和 UI 上的 0~255 口径一致。
    if (roi_image.depth() != CV_8U) {
        double mx = 0.0;
        cv::minMaxLoc(roi_image, nullptr, &mx);
        cv::Mat conv;
        roi_image.convertTo(conv, CV_8U, mx > 0.0 ? 255.0 / mx : 1.0, 0.0);
        roi_image = conv;
    }

    double diagonal = std::sqrt(static_cast<double>(w) * w + static_cast<double>(h) * h);
    double max_scale = std::max(opt.scale_end, 1.0f);
    int padding = static_cast<int>(diagonal * max_scale * 1.5) + 50;
    const int padded_w = w + 2 * padding, padded_h = h + 2 * padding;

    cv::Mat padded_img = cv::Mat::zeros(padded_h, padded_w, roi_image.type());
    roi_image.copyTo(padded_img(cv::Rect(padding, padding, w, h)));

    // ---------- 2. 组合掩码(正向 / 负向 / 排除区), 与 train() 第 4 步一致 ----------
    cv::Mat padded_mask = cv::Mat::zeros(padded_h, padded_w, CV_8UC1);
    bool have_positive = false;
    if (!positive_mask.empty()) {
        cv::Mat pm = positive_mask.clone();
        if (pm.channels() == 3) cv::cvtColor(pm, pm, cv::COLOR_BGR2GRAY);
        if (pm.rows != h || pm.cols != w)
            throw std::invalid_argument(
                "positive_mask 尺寸 (" + std::to_string(pm.rows) + ", " + std::to_string(pm.cols) +
                ") 与 ROI 尺寸 (" + std::to_string(h) + ", " + std::to_string(w) + ") 不一致。");
        if (cv::countNonZero(pm) > 0) {
            have_positive = true;
            pm.copyTo(padded_mask(cv::Rect(padding, padding, w, h)));
        }
    }
    if (!have_positive)
        cv::rectangle(padded_mask, cv::Rect(padding, padding, w, h), cv::Scalar(255), -1);

    if (!negative_mask.empty()) {
        cv::Mat nm = negative_mask.clone();
        if (nm.channels() == 3) cv::cvtColor(nm, nm, cv::COLOR_BGR2GRAY);
        if (nm.rows != h || nm.cols != w)
            throw std::invalid_argument(
                "negative_mask 尺寸 (" + std::to_string(nm.rows) + ", " + std::to_string(nm.cols) +
                ") 与 ROI 尺寸 (" + std::to_string(h) + ", " + std::to_string(w) + ") 不一致。");
        if (cv::countNonZero(nm) > 0)
            padded_mask(cv::Rect(padding, padding, w, h)).setTo(0, nm);
    }
    for (const ExclusionZone& zone : exclusion_zones) {
        int px = zone.x + padding, py = zone.y + padding;
        if (zone.type == "exclude_rect") {
            cv::rectangle(padded_mask, cv::Rect(px, py, zone.w, zone.h), cv::Scalar(0), -1);
        } else if (zone.type == "exclude_ellipse") {
            cv::ellipse(padded_mask, cv::Point(px + zone.w / 2, py + zone.h / 2),
                        cv::Size(zone.w / 2, zone.h / 2), 0, 0, 360, cv::Scalar(0), -1);
        }
    }

    // ---------- 3. 梯度分布统计 ----------
    // 直接拿训练同款的 ColorGradientPyramid(public 成员)当数据源, 不另写一套 Sobel。
    // 这里用"很宽松"的弱阈值先建一次: hysteresis 几乎不放过任何像素, magnitude
    // 就接近原始梯度分布, 用来撒候选强阈值最合适。
    cv::Mat local_mask;
    cv::erode(padded_mask, local_mask, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);

    std::vector<float> grads;      // 有效区内的等效梯度幅值(已排序)
    grads.reserve(static_cast<size_t>(local_mask.total()));
    {
        line2Dup::ColorGradientPyramid ref_pyr(padded_img, padded_mask,
                                               opt.strong_min * opt.weak_ratio,
                                               static_cast<size_t>(out.requested_features),
                                               opt.strong_min);
        for (int r = 0; r < local_mask.rows && r < ref_pyr.magnitude.rows; ++r) {
            const uchar* mr = local_mask.ptr<const uchar>(r);
            const float* vr = ref_pyr.magnitude.ptr<const float>(r);
            for (int c = 0; c < local_mask.cols && c < ref_pyr.magnitude.cols; ++c)
                if (mr[c]) grads.push_back(std::sqrt(vr[c]));
        }
    }
    out.mask_pixels = static_cast<int>(grads.size());
    out.features_image = roi_image.clone();

    if (out.mask_pixels < 20) {
        out.note = "有效区域太小(疑似涂抹掩码把 ROI 屏蔽光了), 无法探测阈值。";
        return out;
    }
    std::sort(grads.begin(), grads.end());
    out.median_gradient = quantile_sorted(grads, 0.5f);
    out.p95_gradient = std::max(quantile_sorted(grads, 0.95f), 1.f);

    if (out.p95_gradient < 1.5f) {   // 图里几乎没有梯度: 给默认值 + 提示
        out.note = "ROI 内几乎没有梯度(疑似纯色 / 严重模糊), 无法自动探测, 已回退到默认阈值 (弱 30 / 强 60)。";
        return out;
    }

    // ---------- 4. 沿梯度分位数撒候选强阈值 ----------
    static const float kQuantiles[] = {
        0.999f, 0.995f, 0.99f, 0.98f, 0.97f, 0.95f, 0.93f, 0.90f, 0.87f, 0.85f,
        0.82f, 0.80f, 0.77f, 0.75f, 0.72f, 0.70f, 0.65f, 0.60f, 0.55f, 0.50f,
        0.45f, 0.40f, 0.35f, 0.30f, 0.25f, 0.20f, 0.15f
    };
    std::vector<float> ladder;
    for (float q : kQuantiles) {
        float v = std::min(quantile_sorted(grads, q), opt.strong_max);
        v = std::max(v, opt.strong_min);
        if (!ladder.empty() && std::fabs(ladder.back() - v) < 0.5f) continue;
        ladder.push_back(v);
    }

    const int min_features = std::max(10, static_cast<int>(std::round(out.requested_features * 0.6f)));
    const float target_surplus = 3.0f;   // 候选点数 / 期望特征点数

    struct Cand {
        float strong = 0.f;
        float weak = 0.f;
        int n_cand = 0;
        int n_feat = 0;
        float cost = 0.f;
    };
    std::vector<Cand> scored;
    scored.reserve(ladder.size());

    for (float s : ladder) {
        const float wk = s * opt.weak_ratio;
        Cand c;
        c.strong = s;
        c.weak = wk;

        line2Dup::ColorGradientPyramid pyr(padded_img, padded_mask, wk,
                                           static_cast<size_t>(out.requested_features), s);
        c.n_cand = count_candidates(pyr.magnitude, pyr.angle, local_mask, s);

        line2Dup::Template templ;
        {
            CoutSuspend muted;
            if (pyr.extractTemplate(templ))
                c.n_feat = static_cast<int>(templ.features.size());
        }

        float surplus = static_cast<float>(c.n_cand) / static_cast<float>(out.requested_features);
        c.cost = std::fabs(std::log(std::max(surplus, 1e-3f) / target_surplus));
        if (c.n_cand <= 4) c.cost += 5.0f;    // 候选都不够, 模板会直接失败
        if (c.n_feat < min_features)
            c.cost += 2.0f * static_cast<float>(min_features - c.n_feat) / static_cast<float>(min_features);
        else
            c.cost += 0.1f * static_cast<float>(c.n_feat - out.requested_features) /
                      static_cast<float>(out.requested_features);
        scored.push_back(c);
    }

    const Cand* best = nullptr;
    for (const Cand& c : scored) {
        if (best == nullptr) { best = &c; continue; }
        if (c.cost < best->cost - 1e-6f) { best = &c; continue; }
        // 打分打平时选更强的阈值: 特征点更"硬", 抗噪一点
        if (c.cost <= best->cost + 1e-6f && c.strong > best->strong) best = &c;
    }
    if (best == nullptr) {
        out.note = "未能从候选阈值中选出可用组合, 已回退到默认阈值。";
        return out;
    }
    // 所有候选都取不到像样的特征点: 这不是"阈值选得不好", 而是 ROI 选得不好,
    // 别硬塞一个看着像模像样的数字给用户。
    if (best->n_feat < 10 || best->n_cand == 0) {
        out.note = "ROI 内取不到足够的强边缘(候选点 " + std::to_string(best->n_cand) +
                   " / 特征点 " + std::to_string(best->n_feat) +
                   "), 该 ROI 不适合模板训练: 请确认 ROI 是否框住目标本体, 或用涂抹掩码去掉杂纹理。";
        return out;
    }

    out.strong_thresh = std::round(best->strong * 10.f) / 10.f;
    out.weak_thresh = std::round(best->weak * 10.f) / 10.f;
    out.candidates = best->n_cand;
    out.features = best->n_feat;
    out.ok = true;

    // ---------- 5. 提示语 ----------
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "建议阈值: 弱 %.1f / 强 %.1f (期望特征点 %d, 实得 %d, 候选点 %d, "
                  "ROI 梯度中位数 %.1f)",
                  out.weak_thresh, out.strong_thresh, out.requested_features,
                  out.features, out.candidates, out.median_gradient);
    out.note = buf;

    if (out.features < min_features)
        out.note += " | 注意: ROI 内够强的边缘不足, 已尽量取满; 确认 ROI 是否框住目标本体, 或用涂抹掩码去掉杂纹理。";
    if (out.median_gradient > 60.f)
        out.note += " | 注意: ROI 纹理丰富, 建议阈值已按'够强但不吃纹理'的口径给出。";
    if (out.strong_thresh < 12.f)
        out.note += " | 注意: 该图整体对比很低, 阈值已压得很低; 若匹配误报多, 请把强阈值手动抬高 10~20。";

    // ---------- 6. 自匹配自检(冒烟测试) ----------
    if (opt.run_self_check && static_cast<size_t>(padded_h * padded_w) <= 4000000) {
        try {
            std::vector<int> levels = opt.pyramid_levels;
            if (levels.empty()) levels = {4, 8};

            // 场景 = 训练图本身。不做任何滤波: 与 MatcherCore::match 的口径一致
            // (模板与场景两侧都不滤), 这样自检得分就是真实可达的上限。
            cv::Mat scene = padded_img.clone();
            if (scene.rows % 16 != 0 || scene.cols % 16 != 0) {
                const int nh = (scene.rows + 15) / 16 * 16;
                const int nw = (scene.cols + 15) / 16 * 16;
                cv::Mat padded = cv::Mat::zeros(nh, nw, scene.type());
                scene.copyTo(padded(cv::Rect(0, 0, scene.cols, scene.rows)));
                scene = padded;
            }

            cv::Mat obj_mask = cv::Mat::zeros(padded_h, padded_w, CV_8UC1);
            cv::rectangle(obj_mask, cv::Rect(padding, padding, w, h), cv::Scalar(255), -1);

            line2Dup::Detector det(out.requested_features, levels,
                                   out.weak_thresh, out.strong_thresh);
            int tid = -1;
            {
                CoutSuspend muted;
                tid = det.addTemplate(padded_img, "self", obj_mask, 0);
            }
            if (tid >= 0) {
                line2Dup::MatchParams mp;
                mp.min_confidence = 0.f;
                std::vector<line2Dup::Match> ms;
                {
                    CoutSuspend muted;
                    ms = det.match(scene, mp);
                }
                if (!ms.empty()) out.self_score = static_cast<float>(ms[0].similarity);
            }
        } catch (...) {
            out.self_score = -1.0;   // 自检失败不影响主结果
        }
        if (out.self_score >= 0.0) {
            std::snprintf(buf, sizeof(buf), " | 自匹配自检得分 %.1f%%", out.self_score);
            out.note += buf;
        }
    }

    // ---------- 7. 特征点预览图 ----------
    out.features_image = draw_features_on_roi(padded_img, padded_mask, out.weak_thresh,
                                              out.strong_thresh, out.requested_features,
                                              padding, w, h);
    return out;
}

}  // namespace sbm
