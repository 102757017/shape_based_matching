#ifndef SBM_MATCHER_CORE_H
#define SBM_MATCHER_CORE_H
// 纯 C++ 内核: 训练 / 加载 / 匹配的完整实现, 不含任何 pybind11 / Python / NumPy 依赖。
//
// 设计要点:
//   - 对外接口只用 POD 结构体 + cv::Mat, 方便再挂任意前端 (pybind11 / C ABI / C++/CLI);
//   - 中文路径在内核侧根治: YAML 用 FileStorage MEMORY 模式读写内存,
//     再以 std::filesystem(UTF-8) 落盘;
//   - 语义与旧版 matcher.py 完全一致 (字段名 / 计算口径 / 产物格式)。
//
// 本文件(及其 .cpp)不允许 include 任何 pybind11 头文件。

#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../line2Dup.h"

namespace sbm {

// ---------------- 训练参数 ----------------
struct TrainParams {
    int feature_num = 100;
    std::vector<int> pyramid_levels{4, 8};
    float weak_thresh = 30.f;
    float strong_thresh = 60.f;
    float angle_start = 0.f;
    float angle_extent = 0.f;
    float angle_step = 15.f;
    float scale_start = 1.f;
    float scale_end = 1.f;
    float scale_step = 0.5f;
};

// add_template_class 的覆盖参数。每项独立带 has_* 标记:
// 调用方没给就用 has=false, 保留原有值。
struct TrainParamsOverride {
    bool has_feature_num = false;
    int feature_num = 0;
    bool has_pyramid_levels = false;
    std::vector<int> pyramid_levels;
    bool has_weak_thresh = false;
    float weak_thresh = 0.f;
    bool has_strong_thresh = false;
    float strong_thresh = 0.f;
};

// 排除区: type 为 "exclude_rect" 或 "exclude_ellipse"
struct ExclusionZone {
    std::string type;
    int x = 0, y = 0, w = 0, h = 0;
};

// ---------------- 训练结果 ----------------
struct TrainResult {
    std::string yaml_path;
    std::string info_path;
    std::string preview_path;    // 空串 = 预览图未写出 (对 Python 侧表现为 None)
    std::string save_dir;
    std::string base_name;
    cv::Mat features_image;      // ROI + 特征点红点标注图
    int template_count = 0;
};

// ---------------- 匹配结果 ----------------
struct MatchResult {
    std::string class_id;
    int template_id = 0;
    double score = 0.0;
    double x = 0.0, y = 0.0;
    bool icp_refined = false;
    std::vector<std::pair<double, double>> refined_box_points;   // 4 个角点
    std::vector<std::pair<double, double>> matched_features;     // 精修后的特征点
    double refined_x = 0.0, refined_y = 0.0, refined_angle = 0.0;
    double fitness = -1.0;
    double overlap = 0.0;
    // 抓取点, 至少 1 个; 第 1 个即旧版 dict 里的 grasp_point
    std::vector<std::pair<double, double>> grasp_points;
};

// ---------------- 加载结果 ----------------
struct LoadedClass {
    std::string class_id;
    TrainParams params;          // 最终生效的训练参数
};

class MatcherCore {
public:
    MatcherCore() = default;

    /// 清空所有已加载的模板和检测器实例
    void clear();

    /// 训练并保存单个模板
    /// :param roi: [x, y, w, h]
    /// :param positive_mask / negative_mask: ROI 尺寸的 uint8 单通道图; 空 Mat = 不使用
    TrainResult train(const cv::Mat& train_image, const std::vector<int>& roi,
                      const std::string& class_id, const TrainParams& params,
                      const std::string& save_dir,
                      const std::vector<ExclusionZone>& exclusion_zones,
                      const cv::Mat& positive_mask, const cv::Mat& negative_mask);

    /// 加载模板类别, path 可以是 xxx.yaml / xxx.info.json / xxx.json
    LoadedClass add_template_class(const std::string& path,
                                   const TrainParamsOverride& override_params);

    std::vector<std::string> get_loaded_class_ids() const;

    /// 某类别训练时主模板的特征点 (ROI 坐标)
    std::vector<std::pair<double, double>> get_base_template_features(
        const std::string& class_id) const;

    /// 执行匹配
    /// :param class_ids: nullptr = 匹配全部已加载类别
    /// :param grasp_cfg: nullptr = 不使用抓取点配置 (退化为精修中心)
    /// :param masks: 场景 mask, 空 Mat = 不加限制
    std::vector<MatchResult> match(
        const cv::Mat& image, double score_threshold,
        const std::vector<std::string>* class_ids,
        bool use_nms, double nms_threshold,
        const std::map<std::string, std::vector<std::pair<double, double>>>* grasp_cfg,
        int max_matches, double min_fitness, bool use_refine,
        double max_overlap, const cv::Mat& masks, bool fill_overlap);

private:
    struct ClassMeta {
        std::map<int, std::pair<double, double>> templates;      // template_id -> (angle, scale)
        std::vector<std::pair<double, double>> base_features;
        int original_w = 0;
        int original_h = 0;
        int padding = 0;
    };

    line2Dup::Detector detector_;
    bool detector_ready_ = false;
    TrainParams detector_params_;                 // 最近一次初始化检测器所用的合并参数
    std::vector<std::string> class_order_;        // 保持加载顺序
    std::map<std::string, ClassMeta> class_meta_;
};

}  // namespace sbm

#endif  // SBM_MATCHER_CORE_H
