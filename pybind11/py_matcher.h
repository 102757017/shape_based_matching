#ifndef PY_MATCHER_H
#define PY_MATCHER_H
// 高层门面类: 把原 matcher.py 的训练 / 加载 / 匹配逻辑整体搬进 C++。
//
// 目标:
//   - Python 调用方只面对一个 PyMatcher 类的 4 个方法, 不再接触
//     Detector / shapeInfo_producer / MatchParams / 临时目录搬运等细节;
//   - 中文路径问题在 C++ 侧根治: YAML 用 FileStorage MEMORY 模式读写内存,
//     再以 std::filesystem(UTF-8) 落盘, 不再需要 Python 层的 ASCII 临时目录绕行;
//   - 返回值与旧版 matcher.py 完全一致 (dict / list 字段同名同型),
//     旧产物 (xxx.yaml / xxx.info.json / xxx.preview.png) 的格式保持兼容。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <vector>

#include "../line2Dup.h"

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
    /// :return dict: yaml_path / info_path / preview_path / save_dir / base_name /
    ///               features_image / template_count
    py::dict train(const cv::Mat& train_image, std::vector<int> roi,
                   const std::string& class_id, const py::dict& train_params,
                   const std::string& save_dir, py::object exclusion_zones);

    /// 加载模板类别 (对应旧 Matcher.add_template_class)
    /// yaml_path 可传 xxx.yaml / xxx.info.json / xxx.json 中的任意一份,
    /// 会自动推算同目录下另一份, 两份都存在才能加载。
    /// :param override_params: dict 或 None, 覆盖 info.json 中记录的训练参数
    /// :return (class_id, 最终参数 dict)
    py::tuple add_template_class(const std::string& path, py::object override_params);

    /// 已加载的类别 id 列表 (按加载顺序)
    py::list get_loaded_class_ids() const;

    /// 某类别训练时主模板的特征点 (ROI 坐标), 对应旧 base_template_features_map
    py::list get_base_template_features(const std::string& class_id) const;

    /// 执行模板匹配 (对应旧 Matcher.match), 返回结果列表, 每项为 dict:
    ///   class_id / template_id / score / x / y / icp_refined /
    ///   refined_box_points / matched_features / refined_x / refined_y / refined_angle /
    ///   fitness / overlap / grasp_point / grasp_points
    py::list match(const cv::Mat& image, double score_threshold,
                   py::object class_ids_to_match,
                   bool use_nms, double nms_threshold,
                   py::object grasp_points_config,
                   int max_matches, double min_fitness, bool use_refine,
                   double max_overlap, py::object masks, bool fill_overlap);

private:
    struct ClassMeta {
        // template_id -> (训练角度, 训练尺度)
        std::map<int, std::pair<double, double>> templates;
        // 主尺度组基础模板的特征点 (ROI 坐标)
        std::vector<std::pair<double, double>> base_features;
        int original_w = 0;
        int original_h = 0;
        int padding = 0;
    };

    void ensure_params_dict();

    line2Dup::Detector detector_;
    bool detector_ready_ = false;
    py::dict detector_params_;   // 最近一次初始化检测器所用的合并参数
    std::vector<std::string> class_order_;               // 保持加载顺序
    std::map<std::string, ClassMeta> class_meta_;        // class_id -> 元信息
};

#endif // PY_MATCHER_H
