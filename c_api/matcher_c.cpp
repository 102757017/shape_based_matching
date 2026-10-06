/* C ABI 实现: 把 sbm::MatcherCore 包装成 extern "C" 的平铺接口。
 * 这里只做参数搬运与异常捕获, 业务逻辑一律不在这里。 */
#define SBM_BUILD
#include "matcher_c.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../core/matcher_core.h"
#include "../third_party/ellipse_detection/include/detect.h"
#include "../ellipse/aamed_detector.h"   // AAMED 适配层 (输出 sbm_ellipse_t 口径)

namespace {

/* sbm_last_error 收的是 void*, 而精配器句柄与 AAMED 检测器句柄是两种不同布局。
 * 用 offset 0 的 magic 区分, 免得把 AAMED 句柄当成精配器句柄读野内存。 */
enum { SBM_HANDLE_MAGIC_MATCHER = 1, SBM_HANDLE_MAGIC_AAMED = 2 };

struct SbmHandle {
    int magic;
    sbm::MatcherCore core;
    std::string last_error;
    std::string last_class_id;                 // sbm_add_template_class 的返回值
    std::string str_yaml, str_info, str_preview, str_save_dir, str_base;
    cv::Mat features_image;                    // 上一次训练的标注图 (保持连续)
    std::vector<sbm::MatchResult> matches;
    std::vector<std::vector<double>> feature_bufs;
    std::vector<std::vector<double>> grasp_bufs;
    std::map<std::string, std::vector<std::pair<double, double>>> grasp_cfg;
    std::vector<std::string> loaded_ids;
    std::string last_note;                      // sbm_estimate_thresholds 的返回值
    cv::Mat est_features_image;                 // 上次探测的特征点标注图 (与训练的那份分开存)
};

SbmHandle* as_handle(void* h) { return static_cast<SbmHandle*>(h); }

cv::Mat image_to_mat(const sbm_image_t* img) {
    if (!img || !img->data || img->width <= 0 || img->height <= 0) return cv::Mat();
    int ch = img->channels <= 1 ? 1 : 3;
    int step = img->step > 0 ? img->step : img->width * ch;
    return cv::Mat(img->height, img->width, CV_MAKETYPE(CV_8U, ch),
                   const_cast<uchar*>(img->data), static_cast<size_t>(step)).clone();
}

sbm::TrainParams params_from_c(const sbm_train_params_t& p) {
    sbm::TrainParams out;
    out.feature_num = p.feature_num;
    int n = p.pyramid_level_count;
    if (n <= 0) n = 0;
    if (n > SBM_MAX_PYRAMID_LEVELS) n = SBM_MAX_PYRAMID_LEVELS;
    out.pyramid_levels.assign(p.pyramid_levels, p.pyramid_levels + n);
    if (out.pyramid_levels.empty()) out.pyramid_levels = {4, 8};
    out.weak_thresh = static_cast<float>(p.weak_thresh);
    out.strong_thresh = static_cast<float>(p.strong_thresh);
    out.angle_start = static_cast<float>(p.angle_start);
    out.angle_extent = static_cast<float>(p.angle_extent);
    out.angle_step = static_cast<float>(p.angle_step);
    out.scale_start = static_cast<float>(p.scale_start);
    out.scale_end = static_cast<float>(p.scale_end);
    out.scale_step = static_cast<float>(p.scale_step);
    return out;
}

void params_to_c(const sbm::TrainParams& src, sbm_train_params_t* out) {
    out->feature_num = src.feature_num;
    int n = static_cast<int>(src.pyramid_levels.size());
    if (n > SBM_MAX_PYRAMID_LEVELS) n = SBM_MAX_PYRAMID_LEVELS;
    out->pyramid_level_count = n;
    for (int i = 0; i < n; ++i) out->pyramid_levels[i] = src.pyramid_levels[i];
    out->weak_thresh = src.weak_thresh;
    out->strong_thresh = src.strong_thresh;
    out->angle_start = src.angle_start;
    out->angle_extent = src.angle_extent;
    out->angle_step = src.angle_step;
    out->scale_start = src.scale_start;
    out->scale_end = src.scale_end;
    out->scale_step = src.scale_step;
}

// 探测参数 -> 内核选项。约定: C 侧的 0 / 负数一律 meaning "用内核默认值",
// 这样调用方填一份全零结构体也能拿到合理结果。
sbm::ThresholdSearchOptions search_options_from_c(const sbm_threshold_search_params_t& p) {
    sbm::ThresholdSearchOptions opt;                 // 一份默认值
    opt.feature_num = p.feature_num > 0 ? p.feature_num : opt.feature_num;
    int n = p.pyramid_level_count;
    if (n > 0) {
        if (n > SBM_MAX_PYRAMID_LEVELS) n = SBM_MAX_PYRAMID_LEVELS;
        opt.pyramid_levels.assign(p.pyramid_levels, p.pyramid_levels + n);
    } else {
        opt.pyramid_levels.clear();                  // 内核会退回 {4, 8}
    }
    if (p.scale_end > 0.0) opt.scale_end = static_cast<float>(p.scale_end);
    if (p.weak_ratio > 0.0) opt.weak_ratio = static_cast<float>(p.weak_ratio);
    if (p.strong_min > 0.0) opt.strong_min = static_cast<float>(p.strong_min);
    if (p.strong_max > 0.0) opt.strong_max = static_cast<float>(p.strong_max);
    if (p.run_self_check > 0) opt.run_self_check = true;
    else if (p.run_self_check < 0) opt.run_self_check = false;
    return opt;
}

}  // namespace

extern "C" {

void* sbm_create(void) {
    SbmHandle* h = new SbmHandle();
    h->magic = SBM_HANDLE_MAGIC_MATCHER;
    return h;
}

void sbm_destroy(void* handle) {
    delete as_handle(handle);
}

void sbm_clear(void* handle) {
    SbmHandle* h = as_handle(handle);
    if (!h) return;
    h->core.clear();
    h->matches.clear();
    h->feature_bufs.clear();
    h->grasp_bufs.clear();
    h->grasp_cfg.clear();
    h->loaded_ids.clear();
}

void sbm_train_params_init(sbm_train_params_t* params) {
    if (!params) return;
    sbm::TrainParams d;
    params_to_c(d, params);
}

int sbm_train(void* handle, const sbm_image_t* image, const int roi[4],
              const char* class_id, const sbm_train_params_t* params,
              const char* save_dir, const sbm_exclusion_zone_t* zones, int zone_count,
              const sbm_image_t* positive_mask, const sbm_image_t* negative_mask,
              sbm_train_result_t* out) {
    SbmHandle* h = as_handle(handle);
    if (!h || !out || !roi || !class_id || !params) return -1;
    try {
        std::vector<int> roi_v{roi[0], roi[1], roi[2], roi[3]};

        std::vector<sbm::ExclusionZone> zone_v;
        if (zones && zone_count > 0) {
            for (int i = 0; i < zone_count; ++i) {
                sbm::ExclusionZone z;
                z.type = zones[i].type ? zones[i].type : "exclude_rect";
                z.x = zones[i].x; z.y = zones[i].y; z.w = zones[i].w; z.h = zones[i].h;
                zone_v.push_back(std::move(z));
            }
        }

        sbm::TrainResult r = h->core.train(image_to_mat(image), roi_v, class_id,
                                           params_from_c(*params),
                                           save_dir ? save_dir : ".",
                                           zone_v,
                                           image_to_mat(positive_mask),
                                           image_to_mat(negative_mask));

        h->str_yaml = r.yaml_path;
        h->str_info = r.info_path;
        h->str_preview = r.preview_path;
        h->str_save_dir = r.save_dir;
        h->str_base = r.base_name;
        h->features_image = r.features_image.clone();

        out->yaml_path = h->str_yaml.c_str();
        out->info_path = h->str_info.c_str();
        out->preview_path = h->str_preview.empty() ? nullptr : h->str_preview.c_str();
        out->save_dir = h->str_save_dir.c_str();
        out->base_name = h->str_base.c_str();
        out->template_count = r.template_count;
        out->features_image.data = h->features_image.data;
        out->features_image.width = h->features_image.cols;
        out->features_image.height = h->features_image.rows;
        out->features_image.channels = h->features_image.channels();
        out->features_image.step = static_cast<int>(h->features_image.step[0]);
        return 0;
    } catch (const std::exception& e) {
        h->last_error = e.what();
        return -2;
    } catch (...) {
        h->last_error = "unknown error in sbm_train";
        return -3;
    }
}

const char* sbm_add_template_class(void* handle, const char* path,
                                   const sbm_train_params_t* override_params,
                                   sbm_train_params_t* out_final_params) {
    SbmHandle* h = as_handle(handle);
    if (!h || !path) return nullptr;
    try {
        sbm::TrainParamsOverride ov;
        if (override_params) {
            if (override_params->pyramid_level_count > 0) {
                ov.has_pyramid_levels = true;
                int n = override_params->pyramid_level_count;
                if (n > SBM_MAX_PYRAMID_LEVELS) n = SBM_MAX_PYRAMID_LEVELS;
                ov.pyramid_levels.assign(override_params->pyramid_levels,
                                         override_params->pyramid_levels + n);
            }
            if (override_params->feature_num > 0) {
                ov.has_feature_num = true;
                ov.feature_num = override_params->feature_num;
            }
            if (override_params->weak_thresh > 0) {
                ov.has_weak_thresh = true;
                ov.weak_thresh = static_cast<float>(override_params->weak_thresh);
            }
            if (override_params->strong_thresh > 0) {
                ov.has_strong_thresh = true;
                ov.strong_thresh = static_cast<float>(override_params->strong_thresh);
            }
        }
        sbm::LoadedClass lc = h->core.add_template_class(path, ov);
        h->last_class_id = lc.class_id;
        if (out_final_params) params_to_c(lc.params, out_final_params);
        return h->last_class_id.c_str();
    } catch (const std::exception& e) {
        h->last_error = e.what();
        return nullptr;
    } catch (...) {
        h->last_error = "unknown error in sbm_add_template_class";
        return nullptr;
    }
}

int sbm_loaded_class_count(void* handle) {
    SbmHandle* h = as_handle(handle);
    if (!h) return -1;
    try {
        h->loaded_ids = h->core.get_loaded_class_ids();
        return static_cast<int>(h->loaded_ids.size());
    } catch (const std::exception& e) {
        h->last_error = e.what();
        return -2;
    }
}

const char* sbm_loaded_class_id(void* handle, int index) {
    SbmHandle* h = as_handle(handle);
    if (!h || index < 0 || index >= static_cast<int>(h->loaded_ids.size())) return nullptr;
    return h->loaded_ids[index].c_str();
}

int sbm_base_features(void* handle, const char* class_id, double* out_xy, int max_points) {
    SbmHandle* h = as_handle(handle);
    if (!h || !class_id || !out_xy) return -1;
    try {
        std::vector<std::pair<double, double>> pts = h->core.get_base_template_features(class_id);
        int n = static_cast<int>(pts.size());
        if (n > max_points) n = max_points;
        for (int i = 0; i < n; ++i) {
            out_xy[2 * i] = pts[i].first;
            out_xy[2 * i + 1] = pts[i].second;
        }
        return n;
    } catch (const std::exception& e) {
        h->last_error = e.what();
        return -2;
    }
}

void sbm_threshold_search_params_init(sbm_threshold_search_params_t* params) {
    if (!params) return;
    // 全部从内核的结构体默认值派生, 免得两边默认值写歪了还不一致
    sbm::ThresholdSearchOptions d;
    std::memset(params, 0, sizeof(*params));
    params->feature_num = d.feature_num;
    params->pyramid_level_count = static_cast<int>(d.pyramid_levels.size());
    for (size_t i = 0; i < d.pyramid_levels.size(); ++i)
        params->pyramid_levels[i] = d.pyramid_levels[i];
    params->scale_end = d.scale_end;
    params->weak_ratio = d.weak_ratio;
    params->strong_min = d.strong_min;
    params->strong_max = d.strong_max;
    params->run_self_check = 1;
}

int sbm_estimate_thresholds(void* handle, const sbm_image_t* image, const int roi[4],
                            const sbm_threshold_search_params_t* params,
                            const sbm_exclusion_zone_t* zones, int zone_count,
                            const sbm_image_t* positive_mask, const sbm_image_t* negative_mask,
                            sbm_threshold_estimate_t* out) {
    SbmHandle* h = as_handle(handle);
    if (!h || !out || !roi) return -1;
    try {
        std::vector<int> roi_v{roi[0], roi[1], roi[2], roi[3]};

        std::vector<sbm::ExclusionZone> zone_v;
        if (zones && zone_count > 0) {
            for (int i = 0; i < zone_count; ++i) {
                sbm::ExclusionZone z;
                z.type = zones[i].type ? zones[i].type : "exclude_rect";
                z.x = zones[i].x; z.y = zones[i].y; z.w = zones[i].w; z.h = zones[i].h;
                zone_v.push_back(std::move(z));
            }
        }

        sbm::ThresholdEstimate r = sbm::estimate_train_thresholds(
            image_to_mat(image), roi_v,
            image_to_mat(positive_mask), image_to_mat(negative_mask),
            zone_v,
            params ? search_options_from_c(*params) : sbm::ThresholdSearchOptions());

        // note / 标注图都存进句柄, 再让 out 指向它们 (下次调用或 destroy 前有效)
        h->last_note = r.note;
        h->est_features_image = r.features_image.clone();

        out->weak_thresh = r.weak_thresh;
        out->strong_thresh = r.strong_thresh;
        out->ok = r.ok ? 1 : 0;
        out->mask_pixels = r.mask_pixels;
        out->requested_features = r.requested_features;
        out->candidates = r.candidates;
        out->features = r.features;
        out->median_gradient = r.median_gradient;
        out->p95_gradient = r.p95_gradient;
        out->self_score = r.self_score;
        out->note = h->last_note.c_str();
        out->features_image.data = h->est_features_image.data;
        out->features_image.width = h->est_features_image.cols;
        out->features_image.height = h->est_features_image.rows;
        out->features_image.channels = h->est_features_image.channels();
        out->features_image.step = h->est_features_image.empty()
            ? 0 : static_cast<int>(h->est_features_image.step[0]);
        return 0;
    } catch (const std::exception& e) {
        h->last_error = e.what();
        return -2;
    } catch (...) {
        h->last_error = "unknown error in sbm_estimate_thresholds";
        return -3;
    }
}

int sbm_set_grasp_points(void* handle, const char* class_id, const double* xy, int count) {
    SbmHandle* h = as_handle(handle);
    if (!h || !class_id) return -1;
    try {
        std::vector<std::pair<double, double>> pts;
        if (xy && count > 0) {
            for (int i = 0; i < count; ++i) pts.emplace_back(xy[2 * i], xy[2 * i + 1]);
        }
        if (pts.empty()) h->grasp_cfg.erase(class_id);
        else h->grasp_cfg[class_id] = std::move(pts);
        return 0;
    } catch (const std::exception& e) {
        h->last_error = e.what();
        return -2;
    }
}

int sbm_match(void* handle, const sbm_image_t* image, double score_threshold,
              const char* const* class_ids, int class_id_count,
              int use_nms, double nms_threshold,
              int max_matches, double min_fitness, int use_refine,
              double max_overlap, const sbm_image_t* masks, int fill_overlap) {
    SbmHandle* h = as_handle(handle);
    if (!h || !image) return -1;
    try {
        std::vector<std::string> ids;
        const std::vector<std::string>* ids_ptr = nullptr;
        if (class_ids && class_id_count > 0) {
            for (int i = 0; i < class_id_count; ++i)
                if (class_ids[i]) ids.emplace_back(class_ids[i]);
            if (!ids.empty()) ids_ptr = &ids;
        }

        const std::map<std::string, std::vector<std::pair<double, double>>>* gcfg =
            h->grasp_cfg.empty() ? nullptr : &h->grasp_cfg;

        h->matches = h->core.match(image_to_mat(image), score_threshold, ids_ptr,
                                   use_nms != 0, nms_threshold, gcfg,
                                   max_matches, min_fitness, use_refine != 0,
                                   max_overlap, image_to_mat(masks), fill_overlap != 0);

        // 为 C ABI 准备连续的双精度缓冲 (指针在下次 match / destroy 前有效)
        h->feature_bufs.clear();
        h->grasp_bufs.clear();
        h->feature_bufs.reserve(h->matches.size());
        h->grasp_bufs.reserve(h->matches.size());
        for (const auto& m : h->matches) {
            std::vector<double> fb;
            fb.reserve(m.matched_features.size() * 2);
            for (const auto& p : m.matched_features) { fb.push_back(p.first); fb.push_back(p.second); }
            h->feature_bufs.push_back(std::move(fb));

            std::vector<double> gb;
            gb.reserve(m.grasp_points.size() * 2);
            for (const auto& p : m.grasp_points) { gb.push_back(p.first); gb.push_back(p.second); }
            h->grasp_bufs.push_back(std::move(gb));
        }
        return static_cast<int>(h->matches.size());
    } catch (const std::exception& e) {
        h->last_error = e.what();
        return -2;
    } catch (...) {
        h->last_error = "unknown error in sbm_match";
        return -3;
    }
}

int sbm_match_result(void* handle, int index, sbm_match_result_t* out) {
    SbmHandle* h = as_handle(handle);
    if (!h || !out) return -1;
    if (index < 0 || index >= static_cast<int>(h->matches.size())) return -4;
    const sbm::MatchResult& r = h->matches[index];

    out->class_id = r.class_id.c_str();
    out->template_id = r.template_id;
    out->score = r.score;
    out->x = r.x;
    out->y = r.y;
    out->icp_refined = r.icp_refined ? 1 : 0;
    for (int i = 0; i < 4; ++i) {
        double px = i < static_cast<int>(r.refined_box_points.size()) ? r.refined_box_points[i].first : 0.0;
        double py = i < static_cast<int>(r.refined_box_points.size()) ? r.refined_box_points[i].second : 0.0;
        out->box[2 * i] = px;
        out->box[2 * i + 1] = py;
    }
    out->feature_count = static_cast<int>(h->feature_bufs[index].size() / 2);
    out->features = h->feature_bufs[index].empty() ? nullptr : h->feature_bufs[index].data();
    out->refined_x = r.refined_x;
    out->refined_y = r.refined_y;
    out->refined_angle = r.refined_angle;
    out->fitness = r.fitness;
    out->overlap = r.overlap;
    out->grasp_count = static_cast<int>(h->grasp_bufs[index].size() / 2);
    out->grasp_points = h->grasp_bufs[index].empty() ? nullptr : h->grasp_bufs[index].data();
    out->angle = r.angle;
    out->scale = r.scale;
    out->refined_scale = r.refined_scale;
    return 0;
}

/* ============================ 椭圆检测 ============================ */
// 无句柄的纯函数式接口; 错误串放在文件级缓冲, 用 sbm_last_error(NULL) 取。
namespace {
std::string g_ellipse_last_error;

zgh::DetectParams ellipse_params_from_c(const sbm_ellipse_params_t* p) {
    // 约定与其它参数结构一致: NULL / <=0 一律解释为"用默认"
    zgh::DetectParams out;   // 默认成员值 == 原版硬编码行为
    if (!p) return out;
    if (p->polarity != 0) out.polarity = p->polarity;
    if (p->line_width > 0.0) out.line_width = p->line_width;
    if (p->min_cover_angle > 0.0) out.min_cover_angle = p->min_cover_angle;
    if (p->min_goodness > 0.0) out.min_goodness = p->min_goodness;
    if (p->candidate_goodness > 0.0) out.candidate_goodness = p->candidate_goodness;
    // num_threads: 0/负数=自动, >=1 用请求值
    if (p->num_threads > 0) out.num_threads = p->num_threads;
    return out;
}
}  // namespace

SBM_API void sbm_ellipse_params_init(sbm_ellipse_params_t* params) {
    if (!params) return;
    std::memset(params, 0, sizeof(*params));
    params->polarity = 0;
    params->line_width = 2.0;
    params->min_cover_angle = 240.0;
    params->min_goodness = 0.4;
    params->candidate_goodness = 0.3;
    params->num_threads = 0;   /* 0 = 自动(用满可用核) */
}

SBM_API int sbm_detect_ellipses(const sbm_image_t* image,
                                const sbm_ellipse_params_t* params,
                                sbm_ellipse_t* out_ellipses,
                                int max_ellipses) {
    g_ellipse_last_error.clear();
    try {
        if (max_ellipses < 0) max_ellipses = 0;
        cv::Mat m = image_to_mat(image);
        if (m.empty()) {
            g_ellipse_last_error = "sbm_detect_ellipses: invalid image (null data or zero size)";
            return -1;
        }
        cv::Mat gray;
        if (m.channels() == 3) {
            cv::cvtColor(m, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = m;
        }

        const zgh::DetectParams ep = ellipse_params_from_c(params);
        std::vector<std::shared_ptr<zgh::Ellipse>> ells;
        zgh::detectEllipse(gray.data, gray.rows, gray.cols, ells, ep);

        const int n = static_cast<int>(ells.size());
        // out_ellipses 为 NULL 时只统计数量, 不写结果
        int write_n = 0;
        if (out_ellipses && max_ellipses > 0) {
            write_n = (max_ellipses < n) ? max_ellipses : n;
        }
        for (int i = 0; i < write_n; ++i) {
            const auto& e = ells[i];
            sbm_ellipse_t& o = out_ellipses[i];
            // 库内部约定: o.x=行, o.y=列, phi 在(行,列)平面测量。
            // 对外统一换成 (列=x, 行=y) + cv::ellipse 口径: phi_cv = 90° - phi。
            o.cx = e->o.y;
            o.cy = e->o.x;
            o.a = e->a;
            o.b = e->b;
            o.phi = CV_PI / 2.0 - e->phi;
            o.goodness = e->goodness;
            o.coverangle = e->coverangle;
        }
        return n;   // 返回总检出数; 只写了前 write_n 个
    } catch (const std::exception& e) {
        g_ellipse_last_error = std::string("sbm_detect_ellipses: ") + e.what();
        return -2;
    } catch (...) {
        g_ellipse_last_error = "unknown error in sbm_detect_ellipses";
        return -3;
    }
}

/* ============================ AAMED 椭圆检测 ============================
 * 与上面 standard-ellipse-detection 那一路共用 g_ellipse_last_error 作为
 * 无句柄调用的错误串缓冲, 所以 sbm_last_error(NULL) 两种接口都能取到。 */
namespace {

struct SbmAamedHandle {
    int magic;
    sbm::AamedDetector detector;
    std::string last_error;
};

sbm::AamedParams aamed_params_from_c(const sbm_aamed_params_t* p) {
    sbm::AamedParams out = sbm::aamed_params_default();
    if (!p) return out;
    // 与其它参数结构一致的约定: NULL / <=0 一律解释为"用默认"
    if (p->theta_fsa > 0.0) out.theta_fsa = p->theta_fsa;
    if (p->length_fsa > 0.0) out.length_fsa = p->length_fsa;
    if (p->t_val > 0.0) out.t_val = p->t_val;
    if (p->min_goodness > 0.0) out.min_goodness = p->min_goodness;
    if (p->nms_iou > 0.0) out.nms_iou = p->nms_iou;
    return out;
}

// 把 sbm::AamedEllipse 搬到 C ABI 的 sbm_ellipse_t。
// 两个结构体字段布局相同 (6 个 double), 这里逐字段写, 免得以后加字段时悄悄错位。
void write_aamed_ellipse(const sbm::AamedEllipse& src, sbm_ellipse_t* dst) {
    dst->cx = src.cx;
    dst->cy = src.cy;
    dst->a = src.a;
    dst->b = src.b;
    dst->phi = src.phi;
    dst->goodness = src.goodness;
    dst->coverangle = src.coverangle;   // AAMED 不产出, 恒 0
}

}  // namespace

SBM_API void sbm_aamed_params_init(sbm_aamed_params_t* params) {
    if (!params) return;
    const sbm::AamedParams d = sbm::aamed_params_default();
    params->theta_fsa = d.theta_fsa;
    params->length_fsa = d.length_fsa;
    params->t_val = d.t_val;
    params->min_goodness = d.min_goodness;
    params->nms_iou = d.nms_iou;
}

SBM_API void* sbm_aamed_create(int rows, int cols) {
    try {
        auto* h = new SbmAamedHandle();
        h->magic = SBM_HANDLE_MAGIC_AAMED;
        h->detector.prepare(rows, cols);
        return h;
    } catch (const std::exception& e) {
        g_ellipse_last_error = std::string("sbm_aamed_create: ") + e.what();
        return nullptr;
    } catch (...) {
        g_ellipse_last_error = "unknown error in sbm_aamed_create";
        return nullptr;
    }
}

SBM_API void sbm_aamed_destroy(void* handle) {
    delete static_cast<SbmAamedHandle*>(handle);
}

SBM_API int sbm_aamed_detect(void* handle, const sbm_image_t* image,
                             const sbm_aamed_params_t* params,
                             sbm_ellipse_t* out_ellipses, int max_ellipses) {
    auto* h = static_cast<SbmAamedHandle*>(handle);
    if (!h) {
        g_ellipse_last_error = "sbm_aamed_detect: null detector handle";
        return -1;
    }
    h->last_error.clear();
    try {
        if (max_ellipses < 0) max_ellipses = 0;
        cv::Mat m = image_to_mat(image);
        if (m.empty()) {
            h->last_error = "sbm_aamed_detect: invalid image (null data or zero size)";
            return -1;
        }
        cv::Mat gray;
        if (m.channels() == 3) {
            cv::cvtColor(m, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = m;
        }

        const sbm::AamedParams p = aamed_params_from_c(params);
        std::vector<sbm::AamedEllipse> ells;
        const int n = h->detector.detect(gray, p, ells);

        if (out_ellipses && max_ellipses > 0) {
            const int write_n = (max_ellipses < n) ? max_ellipses : n;
            for (int i = 0; i < write_n; ++i) write_aamed_ellipse(ells[i], &out_ellipses[i]);
        }
        return n;
    } catch (const std::exception& e) {
        h->last_error = std::string("sbm_aamed_detect: ") + e.what();
        return -2;
    } catch (...) {
        h->last_error = "unknown error in sbm_aamed_detect";
        return -3;
    }
}

SBM_API int sbm_detect_ellipses_aamed(const sbm_image_t* image,
                                      const sbm_aamed_params_t* params,
                                      sbm_ellipse_t* out_ellipses,
                                      int max_ellipses) {
    g_ellipse_last_error.clear();
    try {
        if (max_ellipses < 0) max_ellipses = 0;
        cv::Mat m = image_to_mat(image);
        if (m.empty()) {
            g_ellipse_last_error = "sbm_detect_ellipses_aamed: invalid image (null data or zero size)";
            return -1;
        }
        cv::Mat gray;
        if (m.channels() == 3) {
            cv::cvtColor(m, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = m;
        }

        const sbm::AamedParams p = aamed_params_from_c(params);
        // 按线程缓存: AAMED 内部按像素量级分配内存, 每次调用重建太贵;
        // 同一尺寸重复调用只会重算, 不重复分配。
        thread_local sbm::AamedDetector cache;
        std::vector<sbm::AamedEllipse> ells;
        const int n = cache.detect(gray, p, ells);

        if (out_ellipses && max_ellipses > 0) {
            const int write_n = (max_ellipses < n) ? max_ellipses : n;
            for (int i = 0; i < write_n; ++i) write_aamed_ellipse(ells[i], &out_ellipses[i]);
        }
        return n;
    } catch (const std::exception& e) {
        g_ellipse_last_error = std::string("sbm_detect_ellipses_aamed: ") + e.what();
        return -2;
    } catch (...) {
        g_ellipse_last_error = "unknown error in sbm_detect_ellipses_aamed";
        return -3;
    }
}

const char* sbm_last_error(void* handle) {
    // 无句柄调用(如 sbm_detect_ellipses / sbm_detect_ellipses_aamed)的错误串
    if (!handle) {
        if (!g_ellipse_last_error.empty()) return g_ellipse_last_error.c_str();
        return nullptr;
    }
    if (static_cast<SbmHandle*>(handle)->magic == SBM_HANDLE_MAGIC_AAMED) {
        auto* ah = static_cast<SbmAamedHandle*>(handle);
        if (!ah->last_error.empty()) return ah->last_error.c_str();
        if (!g_ellipse_last_error.empty()) return g_ellipse_last_error.c_str();
        return nullptr;
    }
    SbmHandle* h = as_handle(handle);
    if (h->last_error.empty()) return nullptr;
    return h->last_error.c_str();
}

}  // extern "C"
