#include "py_matcher.h"
#include "np2mat/ndarray_converter.h"

#include <string>
#include <vector>

namespace py = pybind11;

// ============================ py::object <-> 内核结构体 ============================

namespace {

// dict -> TrainParams (缺必填项时报的异常与旧版一致)
sbm::TrainParams params_from_dict(const py::dict& train_params) {
    if (!train_params.contains("feature_num") || !train_params.contains("pyramid_levels") ||
        !train_params.contains("weak_thresh") || !train_params.contains("strong_thresh"))
        throw std::invalid_argument(
            "train_params 缺少必填项 (feature_num/pyramid_levels/weak_thresh/strong_thresh)。");

    sbm::TrainParams p;
    p.feature_num = train_params["feature_num"].cast<int>();
    p.pyramid_levels = train_params["pyramid_levels"].cast<std::vector<int>>();
    p.weak_thresh = train_params["weak_thresh"].cast<float>();
    p.strong_thresh = train_params["strong_thresh"].cast<float>();
    if (train_params.contains("angle_start"))
        p.angle_start = train_params["angle_start"].cast<float>();
    if (train_params.contains("angle_extent"))
        p.angle_extent = train_params["angle_extent"].cast<float>();
    if (train_params.contains("angle_step"))
        p.angle_step = train_params["angle_step"].cast<float>();
    if (train_params.contains("scale_start"))
        p.scale_start = train_params["scale_start"].cast<float>();
    if (train_params.contains("scale_end"))
        p.scale_end = train_params["scale_end"].cast<float>();
    if (train_params.contains("scale_step"))
        p.scale_step = train_params["scale_step"].cast<float>();
    return p;
}

py::dict params_to_dict(const sbm::TrainParams& p) {
    py::dict out;
    out["feature_num"] = p.feature_num;
    py::list levels;
    for (int l : p.pyramid_levels) levels.append(l);
    out["pyramid_levels"] = levels;
    out["weak_thresh"] = p.weak_thresh;
    out["strong_thresh"] = p.strong_thresh;
    return out;
}

// 覆盖参数: None -> 全 false; dict -> 只认这 4 个键
sbm::TrainParamsOverride override_from_dict(const py::object& obj) {
    sbm::TrainParamsOverride ov;
    if (obj.is_none()) return ov;
    py::dict d = obj.cast<py::dict>();
    if (d.contains("feature_num")) {
        ov.has_feature_num = true;
        ov.feature_num = d["feature_num"].cast<int>();
    }
    if (d.contains("pyramid_levels")) {
        ov.has_pyramid_levels = true;
        ov.pyramid_levels = d["pyramid_levels"].cast<std::vector<int>>();
    }
    if (d.contains("weak_thresh")) {
        ov.has_weak_thresh = true;
        ov.weak_thresh = d["weak_thresh"].cast<float>();
    }
    if (d.contains("strong_thresh")) {
        ov.has_strong_thresh = true;
        ov.strong_thresh = d["strong_thresh"].cast<float>();
    }
    return ov;
}

// list[dict] -> vector<ExclusionZone>
std::vector<sbm::ExclusionZone> zones_from_list(const py::object& obj) {
    std::vector<sbm::ExclusionZone> zones;
    if (obj.is_none()) return zones;
    for (py::handle zone_obj : obj) {
        py::dict zone = zone_obj.cast<py::dict>();
        std::vector<int> r = zone["rect"].cast<std::vector<int>>();
        sbm::ExclusionZone z;
        z.type = zone["type"].cast<std::string>();
        z.x = r[0]; z.y = r[1]; z.w = r[2]; z.h = r[3];
        zones.push_back(std::move(z));
    }
    return zones;
}

// ndarray -> cv::Mat; None -> 空 Mat
cv::Mat mat_from_object(const py::object& obj) {
    if (obj.is_none()) return cv::Mat();
    return obj.cast<cv::Mat>();
}

py::list pts_to_list(const std::vector<std::pair<double, double>>& pts) {
    py::list out;
    for (const auto& p : pts) {
        py::list pt;
        pt.append(p.first);
        pt.append(p.second);
        out.append(pt);
    }
    return out;
}

}  // namespace

// ============================ PyMatcher ============================

void PyMatcher::clear() {
    core_.clear();
}

py::dict PyMatcher::train(const cv::Mat& train_image, std::vector<int> roi,
                          const std::string& class_id, const py::dict& train_params,
                          const std::string& save_dir, py::object exclusion_zones,
                          py::object positive_mask, py::object negative_mask) {
    sbm::TrainResult r = core_.train(train_image, roi, class_id,
                                     params_from_dict(train_params), save_dir,
                                     zones_from_list(exclusion_zones),
                                     mat_from_object(positive_mask),
                                     mat_from_object(negative_mask));

    py::dict result;
    result["yaml_path"] = r.yaml_path;
    result["info_path"] = r.info_path;
    result["preview_path"] = r.preview_path.empty() ? py::none() : py::cast(r.preview_path);
    result["save_dir"] = r.save_dir;
    result["base_name"] = r.base_name;
    result["features_image"] = py::cast(r.features_image);
    result["template_count"] = r.template_count;
    return result;
}

py::tuple PyMatcher::add_template_class(const std::string& path, py::object override_params) {
    sbm::LoadedClass lc = core_.add_template_class(path, override_from_dict(override_params));

    py::dict final_params = params_to_dict(lc.params);
    // 调用方额外传进来的键原样回写 (旧版 dict 合并的语义)
    if (!override_params.is_none()) {
        py::dict ov = override_params.cast<py::dict>();
        for (auto item : ov) final_params[item.first] = item.second;
    }
    return py::make_tuple(lc.class_id, final_params);
}

py::list PyMatcher::get_loaded_class_ids() const {
    py::list out;
    for (const auto& c : core_.get_loaded_class_ids()) out.append(c);
    return out;
}

py::list PyMatcher::get_base_template_features(const std::string& class_id) const {
    return pts_to_list(core_.get_base_template_features(class_id));
}

// dict -> ThresholdSearchOptions (缺项用默认值)
sbm::ThresholdSearchOptions thr_options_from_dict(const py::dict& d) {
    sbm::ThresholdSearchOptions o;
    if (d.contains("feature_num")) o.feature_num = d["feature_num"].cast<int>();
    if (d.contains("pyramid_levels")) o.pyramid_levels = d["pyramid_levels"].cast<std::vector<int>>();
    if (d.contains("scale_end")) o.scale_end = d["scale_end"].cast<float>();
    if (d.contains("weak_ratio")) o.weak_ratio = d["weak_ratio"].cast<float>();
    if (d.contains("strong_min")) o.strong_min = d["strong_min"].cast<float>();
    if (d.contains("strong_max")) o.strong_max = d["strong_max"].cast<float>();
    if (d.contains("run_self_check")) o.run_self_check = d["run_self_check"].cast<bool>();
    return o;
}

py::dict PyMatcher::estimate_thresholds(const cv::Mat& train_image, std::vector<int> roi,
                                        py::object positive_mask, py::object negative_mask,
                                        py::object exclusion_zones, py::object options) {
    sbm::ThresholdSearchOptions opt;
    if (!options.is_none()) opt = thr_options_from_dict(options.cast<py::dict>());

    sbm::ThresholdEstimate r = sbm::estimate_train_thresholds(
        train_image, roi, mat_from_object(positive_mask), mat_from_object(negative_mask),
        zones_from_list(exclusion_zones), opt);

    py::dict out;
    out["weak_thresh"] = r.weak_thresh;
    out["strong_thresh"] = r.strong_thresh;
    out["ok"] = r.ok;
    out["candidates"] = r.candidates;
    out["features"] = r.features;
    out["requested_features"] = r.requested_features;
    out["median_gradient"] = r.median_gradient;
    out["p95_gradient"] = r.p95_gradient;
    out["self_score"] = r.self_score;
    out["note"] = r.note;
    out["features_image"] = py::cast(r.features_image);
    return out;
}

py::list PyMatcher::match(const cv::Mat& image, double score_threshold,
                          py::object class_ids_to_match,
                          bool use_nms, double nms_threshold,
                          py::object grasp_points_config,
                          int max_matches, double min_fitness, bool use_refine,
                          double max_overlap, py::object masks, bool fill_overlap) {
    // class_ids: None = 全部已加载类别
    std::vector<std::string> ids;
    const std::vector<std::string>* ids_ptr = nullptr;
    if (!class_ids_to_match.is_none()) {
        if (py::isinstance<py::str>(class_ids_to_match)) {
            ids.push_back(class_ids_to_match.cast<std::string>());
        } else {
            for (py::handle item : class_ids_to_match)
                ids.push_back(item.cast<std::string>());
        }
        ids_ptr = &ids;
    }

    // 抓取点配置: {类别名: [[x, y], ...]}
    std::map<std::string, std::vector<std::pair<double, double>>> gcfg;
    const std::map<std::string, std::vector<std::pair<double, double>>>* gcfg_ptr = nullptr;
    if (!grasp_points_config.is_none()) {
        py::dict gd = grasp_points_config.cast<py::dict>();
        for (auto item : gd) {
            std::vector<std::pair<double, double>> pts;
            for (py::handle pt_obj : py::reinterpret_borrow<py::object>(item.second)) {
                py::sequence pt = py::reinterpret_borrow<py::sequence>(pt_obj);
                pts.emplace_back(pt[0].cast<double>(), pt[1].cast<double>());
            }
            gcfg[item.first.cast<std::string>()] = std::move(pts);
        }
        gcfg_ptr = &gcfg;
    }

    std::vector<sbm::MatchResult> core_results = core_.match(
        image, score_threshold, ids_ptr, use_nms, nms_threshold, gcfg_ptr,
        max_matches, min_fitness, use_refine, max_overlap,
        mat_from_object(masks), fill_overlap);

    py::list results;
    for (const auto& r : core_results) {
        py::dict res;
        res["class_id"] = r.class_id;
        res["template_id"] = r.template_id;
        res["score"] = r.score;
        res["x"] = r.x;
        res["y"] = r.y;
        res["icp_refined"] = r.icp_refined;
        res["refined_box_points"] = pts_to_list(r.refined_box_points);
        res["matched_features"] = pts_to_list(r.matched_features);
        res["refined_x"] = r.refined_x;
        res["refined_y"] = r.refined_y;
        res["refined_angle"] = r.refined_angle;
        res["fitness"] = r.fitness;
        res["overlap"] = r.overlap;
        res["grasp_points"] = pts_to_list(r.grasp_points);
        py::list first_pt;
        first_pt.append(r.grasp_points.empty() ? r.refined_x : r.grasp_points[0].first);
        first_pt.append(r.grasp_points.empty() ? r.refined_y : r.grasp_points[0].second);
        res["grasp_point"] = first_pt;
        results.append(res);
    }
    return results;
}
