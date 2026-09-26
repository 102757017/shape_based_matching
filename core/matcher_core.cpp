#include "matcher_core.h"

#include "mini_json.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

// ============================ 工具函数 ============================

namespace {

// UTF-8 字符串 -> std::filesystem::path (Windows 上按 UTF-8 处理中文路径)
fs::path utf8_path(const std::string& s) {
    return fs::u8path(s);
}

// 把整段字节按二进制写到 UTF-8 路径 (替代 cv::imwrite / FileStorage 直写,
// 绕开它们在 Windows 上用 ANSI fopen 打不开中文路径的问题)
void write_bytes_utf8(const fs::path& path, const void* data, size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("文件写入失败: " + path.u8string());
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("文件写入失败: " + path.u8string());
}

std::string read_bytes_utf8(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("文件打开失败: " + path.u8string());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// 写图片(规避 cv::imwrite 在 Windows 上对非 ASCII 路径静默失败): imencode 后二进制落盘
bool write_image_utf8(const cv::Mat& image, const fs::path& path) {
    std::string ext = path.extension().u8string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") ext = ".png";
    std::vector<uchar> buf;
    if (!cv::imencode(ext, image, buf)) return false;
    try {
        write_bytes_utf8(path, buf.data(), buf.size());
    } catch (...) {
        return false;
    }
    return true;
}

// 把类别名转成安全的模板文件名(过滤 Windows 非法字符)。乱码修复在上层前端完成。
std::string safe_file_name_cpp(std::string name, const std::string& def = "template") {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!name.empty() && is_space(static_cast<unsigned char>(name.front()))) name.erase(name.begin());
    while (!name.empty() && is_space(static_cast<unsigned char>(name.back()))) name.pop_back();

    std::string out;
    out.reserve(name.size());
    for (unsigned char c : name) {
        bool bad = (c == '\\' || c == '/' || c == ':' || c == '*' || c == '?' ||
                    c == '"' || c == '<' || c == '>' || c == '|' || c < 0x20);
        out += bad ? '_' : static_cast<char>(c);
    }
    while (!out.empty() && out.front() == '.') out.erase(out.begin());
    while (!out.empty() && out.back() == '.') out.pop_back();
    return out.empty() ? def : out;
}

// [start, end] 范围收敛: produce_infos 在 2 元素时断言严格递增,
// "不变化的范围"必须收敛成单元素
std::vector<float> collapse_range(double start, double end) {
    if (end > start) return {static_cast<float>(start), static_cast<float>(end)};
    return {static_cast<float>(start)};
}

// 金字塔每层的位移容差 T, 每层至少为 4
void validate_pyramid_levels(const std::vector<int>& levels) {
    bool ok = !levels.empty();
    for (int l : levels)
        if (l < 4) ok = false;
    if (!ok)
        throw std::invalid_argument(
            "金字塔层级参数必须是一个列表, 且每层的位移容差 T 为不小于 4 的整数 (例如 [4, 8])。");
}

}  // namespace

namespace sbm {

// ============================ MatcherCore ============================

void MatcherCore::clear() {
    detector_ = line2Dup::Detector();
    detector_ready_ = false;
    detector_params_ = TrainParams();
    class_order_.clear();
    class_meta_.clear();
}

// ---------------- train ----------------

TrainResult MatcherCore::train(const cv::Mat& train_image, const std::vector<int>& roi,
                               const std::string& class_id_in, const TrainParams& params,
                               const std::string& save_dir_in,
                               const std::vector<ExclusionZone>& exclusion_zones,
                               const cv::Mat& positive_mask, const cv::Mat& negative_mask) {
    // ---------- 1. 参数校验 ----------
    validate_pyramid_levels(params.pyramid_levels);

    if (roi.size() != 4)
        throw std::invalid_argument("roi 必须是 [x, y, w, h]。");
    int x = roi[0], y = roi[1], w = roi[2], h = roi[3];
    if (w <= 0 || h <= 0) throw std::invalid_argument("ROI区域为空或无效。");
    if (train_image.empty()) throw std::invalid_argument("训练图为空。");
    cv::Rect roi_rect(x, y, w, h);
    if (roi_rect.x < 0 || roi_rect.y < 0 ||
        roi_rect.x + roi_rect.width > train_image.cols ||
        roi_rect.y + roi_rect.height > train_image.rows)
        throw std::invalid_argument("ROI区域超出训练图范围。");
    cv::Mat roi_image = train_image(roi_rect).clone();
    if (roi_image.empty()) throw std::invalid_argument("ROI区域为空或无效。");

    // ---------- 2. 名称与目录 (中文路径在内核侧直接支持) ----------
    std::string base_name = safe_file_name_cpp(class_id_in);
    const std::string& class_id = base_name;   // 内部类名 = 文件名前缀, 保证两边一致
    std::string dir_str = save_dir_in.empty() ? "." : save_dir_in;
    fs::path save_dir = utf8_path(dir_str);
    std::error_code ec;
    fs::create_directories(save_dir, ec);
    fs::path img_path = save_dir / utf8_path(base_name + ".jpg");
    if (!write_image_utf8(roi_image, img_path))
        throw std::runtime_error("ROI 图片写入失败, 请检查保存目录权限或路径: " + img_path.u8string());

    // ---------- 3. padding + 基础掩码 ----------
    double diagonal = std::sqrt(static_cast<double>(w) * w + static_cast<double>(h) * h);
    double max_scale = static_cast<double>(params.scale_end);
    int padding = static_cast<int>(diagonal * max_scale * 1.5) + 50;
    int padded_w = w + 2 * padding, padded_h = h + 2 * padding;

    cv::Mat padded_img = cv::Mat::zeros(padded_h, padded_w, roi_image.type());
    roi_image.copyTo(padded_img(cv::Rect(padding, padding, w, h)));

    // ---------- 4. 组合训练掩码: 正向(识别区) - 负向(排除区) - 矩形/椭圆排除区 ----------
    cv::Mat padded_mask = cv::Mat::zeros(padded_h, padded_w, CV_8UC1);
    bool have_positive = false;
    if (!positive_mask.empty()) {
        cv::Mat pm = positive_mask;
        if (pm.channels() == 3) cv::cvtColor(pm, pm, cv::COLOR_BGR2GRAY);
        if (pm.rows != h || pm.cols != w)
            throw std::invalid_argument(
                "positive_mask 尺寸 (" + std::to_string(pm.rows) + ", " + std::to_string(pm.cols) +
                ") 与 ROI 尺寸 (" + std::to_string(h) + ", " + std::to_string(w) + ") 不一致。");
        if (cv::countNonZero(pm) > 0) {   // 全 0 = 未使用 -> 默认整个 ROI 参与
            have_positive = true;
            pm.copyTo(padded_mask(cv::Rect(padding, padding, w, h)));
        }
    }
    if (!have_positive)
        cv::rectangle(padded_mask, cv::Rect(padding, padding, w, h), cv::Scalar(255), -1);

    if (!negative_mask.empty()) {
        cv::Mat nm = negative_mask;
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

    // ---------- 5. 生成角度/尺度信息并训练 ----------
    shape_based_matching::shapeInfo_producer producer(padded_img, padded_mask);
    producer.angle_range = collapse_range(params.angle_start,
                                          params.angle_start + params.angle_extent);
    producer.angle_step = params.angle_step;
    producer.scale_range = collapse_range(params.scale_start, params.scale_end);
    producer.scale_step = params.scale_step;
    if (producer.angle_step <= 0 || producer.scale_step <= 0)
        throw std::invalid_argument("angle_step / scale_step 必须大于 0。");
    producer.produce_infos();
    if (producer.infos.empty())
        throw std::runtime_error("未能生成任何形状信息 (infos)。");

    line2Dup::Detector det_tmp(params.feature_num, params.pyramid_levels,
                               params.weak_thresh, params.strong_thresh);

    using Info = shape_based_matching::shapeInfo_producer::Info;
    std::map<int, std::vector<const Info*>> infos_by_scale;   // key = round(scale*1000)
    std::map<int, float> key_to_scale;
    for (const auto& info : producer.infos) {
        int key = static_cast<int>(std::lround(info.scale * 1000.0));
        infos_by_scale[key].push_back(&info);
        key_to_scale[key] = info.scale;
    }
    if (infos_by_scale.empty()) throw std::runtime_error("没有可用的尺度信息。");

    // 主尺度 = 最接近 1.0 的那组
    int main_key = 0;
    double best = -1.0;
    for (const auto& kv : key_to_scale) {
        double d = std::fabs(static_cast<double>(kv.second) - 1.0);
        if (best < 0 || d < best) { best = d; main_key = kv.first; }
    }

    std::map<int, std::pair<double, double>> template_info;              // templ_id -> (angle, scale)
    std::vector<std::pair<double, double>> base_template_features;       // 仅主尺度组基础模板
    auto train_group = [&](std::vector<const Info*>& group, bool collect_features) {
        std::stable_sort(group.begin(), group.end(), [](const Info* a, const Info* b) {
            return std::fabs(static_cast<double>(a->angle)) < std::fabs(static_cast<double>(b->angle));
        });
        const Info* base = group[0];
        int adjusted_feature_num =
            std::max(10, std::min(static_cast<int>(params.feature_num * base->scale),
                                  params.feature_num));
        cv::Mat rotated_img = producer.src_of(*base);
        cv::Mat rotated_mask = producer.mask_of(*base);
        int base_id = det_tmp.addTemplate(rotated_img, class_id, rotated_mask, adjusted_feature_num);
        if (base_id == -1) return;
        template_info[base_id] = {base->angle, base->scale};
        if (collect_features) {
            // 特征在 src_of(angle, scale) 这张"旋转缩放后的图"上; 画回原始 ROI 必须做逆仿射,
            // 否则 scale != 1 / angle != 0 时特征点整体偏移甚至超出 ROI。
            const auto& templ0 = det_tmp.getTemplates(class_id, base_id)[0];
            cv::Point2f rot_center(padded_w / 2.0f, padded_h / 2.0f);
            cv::Mat M = cv::getRotationMatrix2D(rot_center, base->angle, base->scale);  // 2x3
            cv::Mat Mi;
            cv::invertAffineTransform(M, Mi);                                           // 2x3 逆
            for (const auto& feat : templ0.features) {
                cv::Point2d p_pad(feat.x + templ0.tl_x, feat.y + templ0.tl_y);  // 旋转缩放图坐标
                cv::Point2d p_orig(
                    Mi.at<double>(0, 0) * p_pad.x + Mi.at<double>(0, 1) * p_pad.y + Mi.at<double>(0, 2),
                    Mi.at<double>(1, 0) * p_pad.x + Mi.at<double>(1, 1) * p_pad.y + Mi.at<double>(1, 2));
                base_template_features.emplace_back(p_orig.x - padding, p_orig.y - padding);
            }
        }
        float base_angle = base->angle;
        cv::Point2f rotation_center(padded_w / 2.0f, padded_h / 2.0f);
        for (size_t i = 1; i < group.size(); ++i) {
            const Info* info = group[i];
            int tid = det_tmp.addTemplate_rotate(class_id, base_id, info->angle - base_angle,
                                                 rotation_center);
            if (tid != -1) template_info[tid] = {info->angle, info->scale};
        }
    };

    // 先训练主尺度组, 再训练其余尺度组 (与旧 Python 实现的顺序一致)
    train_group(infos_by_scale[main_key], true);
    for (auto& kv : infos_by_scale) {
        if (kv.first == main_key) continue;
        train_group(kv.second, false);
    }
    int num_templates_added = static_cast<int>(template_info.size());
    if (num_templates_added == 0) throw std::runtime_error("未能成功添加任何模板。");

    // ---------- 6. 预览图 (ROI + 特征点红点标注) ----------
    cv::Mat display_roi_with_features = roi_image.clone();
    for (const auto& f : base_template_features) {
        if (0 <= f.first && f.first < w && 0 <= f.second && f.second < h)
            cv::circle(display_roi_with_features,
                       cv::Point(static_cast<int>(f.first), static_cast<int>(f.second)),
                       2, cv::Scalar(0, 0, 255), -1);
    }

    // ---------- 7. 写 YAML (MEMORY 模式 + UTF-8 落盘, 中文目录/类别名均可用) ----------
    cv::FileStorage yfs("templates.yaml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
    det_tmp.writeClass(class_id, yfs);
    std::string yaml_buf = yfs.releaseAndGetString();
    fs::path yaml_path = save_dir / utf8_path(base_name + ".yaml");
    write_bytes_utf8(yaml_path, yaml_buf.data(), yaml_buf.size());

    // ---------- 8. 写 info.json (schema 与旧版完全一致) ----------
    std::ostringstream oss;
    oss << "{\n    \"training_params\": {\n"
        << "        \"feature_num\": " << minijson::dump_number(params.feature_num) << ",\n"
        << "        \"pyramid_levels\": [";
    for (size_t i = 0; i < params.pyramid_levels.size(); ++i) {
        if (i) oss << ", ";
        oss << params.pyramid_levels[i];
    }
    oss << "],\n"
        << "        \"weak_thresh\": " << minijson::dump_number(params.weak_thresh) << ",\n"
        << "        \"strong_thresh\": " << minijson::dump_number(params.strong_thresh) << "\n"
        << "    },\n"
        << "    \"templates\": {";
    {
        bool first = true;
        for (const auto& kv : template_info) {
            if (!first) oss << ",";
            first = false;
            oss << "\n        " << minijson::escape_string(std::to_string(kv.first))
                << ": {\n"
                << "            \"angle\": " << minijson::dump_number(kv.second.first) << ",\n"
                << "            \"scale\": " << minijson::dump_number(kv.second.second) << "\n"
                << "        }";
        }
    }
    oss << "\n    },\n"
        << "    \"base_template_features\": [";
    for (size_t i = 0; i < base_template_features.size(); ++i) {
        if (i) oss << ", ";
        oss << "[" << minijson::dump_number(base_template_features[i].first) << ", "
            << minijson::dump_number(base_template_features[i].second) << "]";
    }
    oss << "],\n"
        << "    \"original_w\": " << w << ",\n"
        << "    \"original_h\": " << h << ",\n"
        << "    \"padding\": " << padding << "\n"
        << "}";
    fs::path info_path = save_dir / utf8_path(base_name + ".info.json");
    std::string json_str = oss.str();
    write_bytes_utf8(info_path, json_str.data(), json_str.size());

    // ---------- 9. 预览图落盘 (失败不影响训练结果) ----------
    fs::path preview_path = save_dir / utf8_path(base_name + ".preview.png");
    bool preview_ok = false;
    try {
        preview_ok = write_image_utf8(display_roi_with_features, preview_path);
    } catch (...) {
        preview_ok = false;
    }

    TrainResult result;
    result.yaml_path = yaml_path.u8string();
    result.info_path = info_path.u8string();
    result.preview_path = preview_ok ? preview_path.u8string() : std::string();
    result.save_dir = dir_str;
    result.base_name = base_name;
    result.features_image = display_roi_with_features;
    result.template_count = num_templates_added;
    return result;
}

// ---------------- add_template_class ----------------

LoadedClass MatcherCore::add_template_class(const std::string& path_str,
                                            const TrainParamsOverride& override_params) {
    // ---------- 1. 解析 yaml / info.json 两份文件的路径 ----------
    fs::path p = utf8_path(path_str);
    std::string suffix = p.extension().u8string();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    fs::path yaml_path, info_path;
    if (suffix == ".json") {
        std::string name = p.filename().u8string();
        const std::string info_suffix = ".info.json";
        std::string base;
        if (name.size() >= info_suffix.size() &&
            name.compare(name.size() - info_suffix.size(), info_suffix.size(), info_suffix) == 0) {
            base = name.substr(0, name.size() - info_suffix.size());
        } else {
            base = p.stem().u8string();
        }
        yaml_path = p.parent_path() / utf8_path(base + ".yaml");
        info_path = p.parent_path() / utf8_path(base + ".info.json");
    } else {
        yaml_path = p;
        info_path = p.parent_path() / utf8_path(p.stem().u8string() + ".info.json");
    }
    if (!fs::exists(yaml_path) || !fs::exists(info_path))
        throw std::runtime_error("模板文件或信息文件未找到: " + yaml_path.u8string() +
                                 " / " + info_path.u8string());

    std::string class_id = yaml_path.stem().u8string();

    // ---------- 2. 解析 info.json ----------
    minijson::Value root = minijson::parse(read_bytes_utf8(info_path));
    const minijson::Value& saved = root.get("training_params");
    const minijson::Value& templates_j = root.get("templates");
    const minijson::Value& feats_j = root.get("base_template_features");

    // ---------- 3. 合并参数: 当前实例参数 <- info.json 记录 <- override ----------
    TrainParams final_params = detector_ready_ ? detector_params_ : TrainParams();
    if (saved.is_obj()) {
        if (const minijson::Value* lv = saved.find("pyramid_levels")) {
            std::vector<int> levels;
            for (const auto& v : lv->arr) levels.push_back(v.as_int());
            final_params.pyramid_levels = levels;
        }
        if (saved.get("feature_num").is_num())
            final_params.feature_num = saved.get("feature_num").as_int();
        if (saved.get("weak_thresh").is_num())
            final_params.weak_thresh = static_cast<float>(saved.get("weak_thresh").as_num());
        if (saved.get("strong_thresh").is_num())
            final_params.strong_thresh = static_cast<float>(saved.get("strong_thresh").as_num());
    }
    if (override_params.has_pyramid_levels) final_params.pyramid_levels = override_params.pyramid_levels;
    if (override_params.has_feature_num)   final_params.feature_num = override_params.feature_num;
    if (override_params.has_weak_thresh)   final_params.weak_thresh = override_params.weak_thresh;
    if (override_params.has_strong_thresh) final_params.strong_thresh = override_params.strong_thresh;

    validate_pyramid_levels(final_params.pyramid_levels);

    // ---------- 4. 初始化检测器 (仅首次) ----------
    // 与旧实现一致: detector_params_ 只记录"首次初始化检测器"时的参数快照,
    // 后续加载类别时若 info.json 缺项, 回退到这份快照而不是上一个类别的参数。
    if (!detector_ready_) {
        detector_ = line2Dup::Detector(final_params.feature_num, final_params.pyramid_levels,
                                       final_params.weak_thresh, final_params.strong_thresh);
        detector_ready_ = true;
        detector_params_ = final_params;
    }

    // ---------- 5. 读 YAML (整文件进内存, MEMORY 模式解析, 中文目录无碍) ----------
    std::string yaml_buf = read_bytes_utf8(yaml_path);
    cv::FileStorage rfs(yaml_buf, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    detector_.readClass(rfs.root());

    // ---------- 6. 登记元信息 ----------
    ClassMeta meta;
    if (templates_j.is_obj()) {
        for (const auto& kv : templates_j.members) {
            int tid = std::atoi(kv.first.c_str());
            double angle = kv.second.get("angle").is_num() ? kv.second.get("angle").as_num() : 0.0;
            double scale = kv.second.get("scale").is_num() ? kv.second.get("scale").as_num() : 1.0;
            meta.templates[tid] = {angle, scale};
        }
    }
    if (feats_j.is_arr()) {
        for (const auto& pt : feats_j.arr) {
            if (pt.is_arr() && pt.arr.size() == 2)
                meta.base_features.emplace_back(pt.arr[0].as_num(), pt.arr[1].as_num());
        }
    }
    const minijson::Value& ow = root.get("original_w");
    const minijson::Value& oh = root.get("original_h");
    const minijson::Value& pd = root.get("padding");
    meta.original_w = ow.is_num() ? ow.as_int() : 0;
    meta.original_h = oh.is_num() ? oh.as_int() : 0;
    meta.padding = pd.is_num() ? pd.as_int() : 0;

    if (std::find(class_order_.begin(), class_order_.end(), class_id) == class_order_.end())
        class_order_.push_back(class_id);
    class_meta_[class_id] = std::move(meta);

    return LoadedClass{class_id, final_params};
}

std::vector<std::string> MatcherCore::get_loaded_class_ids() const {
    return class_order_;
}

std::vector<std::pair<double, double>> MatcherCore::get_base_template_features(
    const std::string& class_id) const {
    auto it = class_meta_.find(class_id);
    if (it == class_meta_.end()) return {};
    return it->second.base_features;
}

// ---------------- match ----------------

std::vector<MatchResult> MatcherCore::match(
    const cv::Mat& image, double score_threshold,
    const std::vector<std::string>* class_ids,
    bool use_nms, double nms_threshold,
    const std::map<std::string, std::vector<std::pair<double, double>>>* grasp_cfg,
    int max_matches, double min_fitness, bool use_refine,
    double max_overlap, const cv::Mat& masks, bool fill_overlap) {
    if (!detector_ready_ || class_order_.empty())
        throw std::runtime_error("没有加载任何模板类别，请先加载或训练模板。");

    // 1. 图像预处理: 只补边到 16 的倍数 (MIPP 步长要求), 不做任何滤波。
    //    模板特征是在"未滤波"的训练图 ROI 上提取的, 场景若先滤波 (旧版这里做过 medianBlur(3)),
    //    就会破坏/改变 1~3px 尺度的边缘方向, 这部分特征点永久对不上 —— 连训练图自匹配都拿不到 100 分。
    //    两侧口径必须完全一致: 要么都不滤, 要么训练时也滤同样的核。
    cv::Mat image_to_match = image;
    int orig_h = image_to_match.rows, orig_w = image_to_match.cols;
    const int stride = 16;
    bool need_pad = (orig_h % stride != 0 || orig_w % stride != 0);
    if (need_pad) {
        int new_h = (orig_h + stride - 1) / stride * stride;
        int new_w = (orig_w + stride - 1) / stride * stride;
        cv::Mat padded = cv::Mat::zeros(new_h, new_w, image_to_match.type());
        image_to_match.copyTo(padded(cv::Rect(0, 0, orig_w, orig_h)));
        image_to_match = padded;
    }
    image_to_match = image_to_match.clone();   // 保证内存连续 (MIPP 16 字节步长要求)

    // 场景 mask 预处理: 尺寸须与输入一致, 跟随补边, 非0即参与
    cv::Mat mask_to_use;
    if (!masks.empty()) {
        cv::Mat mask_arr = masks;
        if (mask_arr.channels() == 3)
            cv::cvtColor(mask_arr, mask_arr, cv::COLOR_BGR2GRAY);
        if (mask_arr.rows != orig_h || mask_arr.cols != orig_w)
            throw std::invalid_argument(
                "masks 尺寸 (" + std::to_string(mask_arr.rows) + ", " + std::to_string(mask_arr.cols) +
                ") 与输入图像尺寸 (" + std::to_string(orig_h) + ", " + std::to_string(orig_w) + ") 不一致。");
        cv::threshold(mask_arr, mask_arr, 0, 255, cv::THRESH_BINARY);
        if (need_pad) {
            cv::Mat padded = cv::Mat::zeros(image_to_match.rows, image_to_match.cols, CV_8UC1);
            mask_arr.copyTo(padded(cv::Rect(0, 0, orig_w, orig_h)));
            mask_arr = padded;
        }
        mask_to_use = mask_arr.clone();
    }

    // 2. 确定搜索范围并组装 MatchParams
    std::vector<std::string> search_ids = class_ids ? *class_ids : class_order_;
    if (search_ids.empty()) return {};

    line2Dup::MatchParams params;
    params.class_ids = search_ids;
    params.min_confidence = static_cast<float>(score_threshold);
    params.nms = use_nms;
    params.nms_overlap = static_cast<float>(nms_threshold);
    params.use_refine = use_refine;
    params.min_fitness = use_refine ? static_cast<float>(min_fitness) : 0.f;
    params.max_matches = max_matches;
    params.max_overlap = static_cast<float>(max_overlap);
    params.fill_overlap = fill_overlap;
    if (!mask_to_use.empty()) params.masks = mask_to_use;

    // 核心检测: 类别过滤 + 重叠度过滤 + NMS + ICP 精修一次完成
    std::vector<line2Dup::Match> raw_matches = detector_.match(image_to_match, params);
    std::sort(raw_matches.begin(), raw_matches.end());   // similarity 降序

    std::vector<MatchResult> results;
    // 3. 遍历每个匹配结果, 计算精确的红点和旋转外框
    for (const auto& m : raw_matches) {
        try {
            auto meta_it = class_meta_.find(m.class_id);
            if (meta_it == class_meta_.end()) continue;
            const ClassMeta& meta = meta_it->second;

            double angle_deg = 0.0, scale = 1.0;
            auto tit = meta.templates.find(m.template_id);
            if (tit != meta.templates.end()) {
                angle_deg = tit->second.first;
                scale = tit->second.second;
            }
            double angle_rad = angle_deg * CV_PI / 180.0;

            int roi_w = meta.original_w, roi_h = meta.original_h, padding = meta.padding;
            if (roi_w == 0 || roi_h == 0) {
                std::fprintf(stderr, "[MatcherCore] 类别 %s 缺少原始尺寸信息，跳过\n", m.class_id.c_str());
                continue;
            }
            double center_x = (roi_w + 2.0 * padding) / 2.0;
            double center_y = (roi_h + 2.0 * padding) / 2.0;

            const std::vector<line2Dup::Template>& tps =
                detector_.getTemplates(m.class_id, m.template_id);
            const line2Dup::Template* templ0 = nullptr;
            for (const auto& t : tps)
                if (t.pyramid_level == 0) { templ0 = &t; break; }
            if (templ0 == nullptr && !tps.empty()) templ0 = &tps[0];
            if (templ0 == nullptr) continue;
            double tl_x = templ0->tl_x, tl_y = templ0->tl_y;

            // 解出精修变换: 优先 Match.transform (2x3, 已合成平移), 否则回退手动 refine 增量矩阵
            bool have_transform = false;
            cv::Matx33d T = cv::Matx33d::eye();
            if (!m.transform.empty() && m.transform.rows == 2 && m.transform.cols == 3) {
                cv::Mat tf64;
                m.transform.convertTo(tf64, CV_64F);
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 3; ++c)
                        T(r, c) = tf64.at<double>(r, c);
                have_transform = true;
            }
            cv::Matx33d M = cv::Matx33d::eye();   // 3x3 增量矩阵 (回退路径)
            if (!have_transform) {
                line2Dup::RegistrationResult rr = detector_.refine(m);
                if (rr.transformation.size() == 3)
                    for (int r = 0; r < 3; ++r) {
                        if (rr.transformation[r].size() < 3) { have_transform = false; break; }
                        for (int c = 0; c < 3; ++c)
                            M(r, c) = rr.transformation[r][c];
                    }
            }

            // 场景坐标 = A * [模板点 + 偏移, 1]
            auto to_scene = [&](cv::Point2d p) -> cv::Point2d {
                double ox = have_transform ? tl_x : static_cast<double>(m.x);
                double oy = have_transform ? tl_y : static_cast<double>(m.y);
                double tx = p.x + ox, ty = p.y + oy;
                const cv::Matx33d& A = have_transform ? T : M;
                return {A(0, 0) * tx + A(0, 1) * ty + A(0, 2),
                        A(1, 0) * tx + A(1, 1) * ty + A(1, 2)};
            };
            double delta_angle = [&]() {
                const cv::Matx33d& A = have_transform ? T : M;
                return std::atan2(A(1, 0), A(0, 0)) * 180.0 / CV_PI;
            }();
            // 精修变换的线性部分模长 = ICP 带来的额外缩放 (sim3, 只有旋转+均匀缩放)
            double refine_scale_delta = [&]() {
                const cv::Matx33d& A = have_transform ? T : M;
                return std::sqrt(A(0, 0) * A(0, 0) + A(1, 0) * A(1, 0));
            }();

            // ROI 点 -> 模板坐标系
            auto roi_to_templ = [&](cv::Point2d p) -> cv::Point2d {
                double x_rel = p.x + padding - center_x;
                double y_rel = p.y + padding - center_y;
                double cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);
                double x_rot = center_x + scale * (x_rel * cos_a + y_rel * sin_a);
                double y_rot = center_y + scale * (-x_rel * sin_a + y_rel * cos_a);
                return {x_rot - tl_x, y_rot - tl_y};
            };

            MatchResult res;
            res.class_id = m.class_id;
            res.template_id = m.template_id;
            res.score = static_cast<double>(m.similarity);
            res.x = static_cast<double>(m.x);
            res.y = static_cast<double>(m.y);
            res.icp_refined = use_refine;
            res.angle = angle_deg;      // 模板自身的训练角
            res.scale = scale;          // 模板自身的训练缩放
            res.refined_scale = refine_scale_delta;

            const std::vector<cv::Point2d> roi_corners = {
                {0, 0}, {static_cast<double>(roi_w), 0},
                {static_cast<double>(roi_w), static_cast<double>(roi_h)}, {0, static_cast<double>(roi_h)}};
            cv::Point2d center(0, 0);
            for (const auto& p : roi_corners) {
                cv::Point2d sp = to_scene(roi_to_templ(p));
                res.refined_box_points.emplace_back(sp.x, sp.y);
                center += sp;
            }
            center *= 0.25;

            for (const auto& f : templ0->features) {
                cv::Point2d sp = to_scene(cv::Point2d(f.x, f.y));
                res.matched_features.emplace_back(sp.x, sp.y);
            }

            res.refined_x = center.x;
            res.refined_y = center.y;
            res.refined_angle = angle_deg - delta_angle;   // 最终角度 = 训练角度 - ICP 增量角
            // 最终缩放 = 模板训练缩放 * ICP 增量缩放
            res.fitness = static_cast<double>(m.fitness);
            res.overlap = static_cast<double>(m.overlap);

            // 抓取点: 优先用调用方配置 (ROI 坐标), 无则用精修中心
            if (grasp_cfg) {
                auto git = grasp_cfg->find(m.class_id);
                if (git != grasp_cfg->end() && !git->second.empty()) {
                    for (const auto& gp : git->second) {
                        cv::Point2d sp = to_scene(roi_to_templ(cv::Point2d(gp.first, gp.second)));
                        res.grasp_points.emplace_back(sp.x, sp.y);
                    }
                }
            }
            if (res.grasp_points.empty())
                res.grasp_points.emplace_back(center.x, center.y);

            results.push_back(std::move(res));
        } catch (const std::exception& e) {
            // 单个结果处理失败不中断整体匹配
            std::fprintf(stderr, "[MatcherCore] 处理匹配结果 [%s] 时出错: %s\n",
                         m.class_id.c_str(), e.what());
            continue;
        }
    }

    return results;
}

}  // namespace sbm
