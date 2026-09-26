#ifndef PY_MATCHER_H
#define PY_MATCHER_H
// Python 薄壳: 只做 py::dict / py::object <-> 内核结构体的转发, 不承载任何业务逻辑。
//
// 业务逻辑在 ../core/matcher_core.h 的 sbm::MatcherCore (纯 C++, 无 Python 依赖),
// 这样同一个内核还能挂 C ABI (c_api/) 给 C# 用。
//
// 对外签名与字段与旧版完全一致, matcher.py 与 py_tests 不需要任何修改。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "../core/matcher_core.h"

namespace py = pybind11;

class PyMatcher {
public:
    PyMatcher() = default;

    /// 清空所有已加载的模板和检测器实例 (对应旧 Matcher.clear)
    void clear();

    /// 训练并保存单个模板 (对应旧 Matcher.train)
    /// :param train_image: 训练图 (灰度或 BGR)
    /// :param roi: [x, y, w, h]
    /// :param class_id: 类别名 (已由 Python 门面做过乱码修复与非法字符过滤)
    /// :param train_params: dict, 含 feature_num / pyramid_levels / weak_thresh /
    ///                      strong_thresh / angle_start / angle_extent / angle_step /
    ///                      scale_start / scale_end / scale_step
    /// :param save_dir: 模板保存目录 (UTF-8, 可以是中文路径)
    /// :param exclusion_zones: list[dict] 或 None,
    ///                         每项 {'type': 'exclude_rect'/'exclude_ellipse', 'rect': [x,y,w,h]}
    /// :param positive_mask: ROI 尺寸 (h,w) uint8 ndarray 或 None
    /// :param negative_mask: ROI 尺寸 (h,w) uint8 ndarray 或 None
    /// :return dict: yaml_path / info_path / preview_path / save_dir / base_name /
    ///               features_image / template_count
    py::dict train(const cv::Mat& train_image, std::vector<int> roi,
                   const std::string& class_id, const py::dict& train_params,
                   const std::string& save_dir, py::object exclusion_zones,
                   py::object positive_mask, py::object negative_mask);

    /// 加载模板类别 (对应旧 Matcher.add_template_class)
    /// :return (class_id, 最终参数 dict)
    py::tuple add_template_class(const std::string& path, py::object override_params);

    py::list get_loaded_class_ids() const;

    py::list get_base_template_features(const std::string& class_id) const;

    /// 自动探测训练图(ROI)合适的弱/强阈值 (对应 sbm::estimate_train_thresholds)
    /// :param train_image 训练图 (灰度或 BGR)
    /// :param roi         [x, y, w, h]
    /// :param options     dict, 可选: feature_num / pyramid_levels / scale_end /
    ///                    weak_ratio / strong_min / strong_max / run_self_check
    /// :return dict: weak_thresh / strong_thresh / ok / candidates / features /
    ///               requested_features / median_gradient / p95_gradient /
    ///               self_score / note / features_image
    py::dict estimate_thresholds(const cv::Mat& train_image, std::vector<int> roi,
                                 py::object positive_mask, py::object negative_mask,
                                 py::object exclusion_zones, py::object options);

    /// 执行模板匹配 (对应旧 Matcher.match), 返回 dict 列表
    /// 每个 dict 的字段: class_id / template_id / score / x / y / icp_refined /
    /// refined_box_points / matched_features / refined_x / refined_y / refined_angle /
    /// angle(模板训练角) / scale(模板训练缩放) / refined_scale(ICP 额外缩放) /
    /// fitness / overlap / grasp_points / grasp_point
    py::list match(const cv::Mat& image, double score_threshold,
                   py::object class_ids_to_match,
                   bool use_nms, double nms_threshold,
                   py::object grasp_points_config,
                   int max_matches, double min_fitness, bool use_refine,
                   double max_overlap, py::object masks, bool fill_overlap);

private:
    sbm::MatcherCore core_;
};

#endif // PY_MATCHER_H
