#include "py_matcher.h"
#include "mini_json.h"
#include "np2mat/ndarray_converter.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <sstream>

namespace py = pybind11;
namespace fs = std::filesystem;

// ============================ 工具函数 ============================

// UTF-8 字符串 -> std::filesystem::path (Windows 上按 UTF-8 处理中文路径)
static fs::path utf8_path(const std::string& s) {
    return fs::u8path(s);
}

// 把整段字节按二进制写到 UTF-8 路径 (替代 cv::imwrite / FileStorage 直写,
// 绕开它们在 Windows 上用 ANSI fopen 打不开中文路径的问题)
static void write_bytes_utf8(const fs::path& path, const void* data, size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("文件写入失败: " + path.u8string());
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("文件写入失败: " + path.u8string());
}

static std::string read_bytes_utf8(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("文件打开失败: " + path.u8string());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// 写图片(规避 cv::imwrite 在 Windows 上对非 ASCII 路径静默失败): imencode 后二进制落盘
static bool write_image_utf8(const cv::Mat& image, const fs::path& path) {
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

// 把类别名转成安全的模板文件名(过滤 Windows 非法字符)。乱码修复在 Python 门面完成。
static std::string safe_file_name_cpp(std::string name, const std::string& def = "template") {
    // 去首尾空白
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!name.empty() && is_space((unsigned char)name.front())) name.erase(name.begin());
    while (!name.empty() && is_space((unsigned char)name.back())) name.pop_back();

    std::string out;
    out.reserve(name.size());
    for (unsigned char c : name) {
        bool bad = (c == '\\' || c == '/' || c == ':' || c == '*' || c == '?' ||
                    c == '"' || c == '<' || c == '>' || c == '|' || c < 0x20);
        out += bad ? '_' : char(c);
    }
    while (!out.empty() && out.front() == '.') out.erase(out.begin());
    while (!out.empty() && out.back() == '.') out.pop_back();
    return out.empty() ? def : out;
}

// [start, end] 范围收敛: C++ 侧 produce_infos 在 2 元素时断言严格递增,
// "不变化的范围"必须收敛成单元素
static std::vector<float> collapse_range(double start, double end) {
    if (end > start) return {float(start), float(end)};
    return {float(start)};
}

static py::list pt_to_list(double x, double y) {
    py::list pt;
    pt.append(x);
    pt.append(y);
    return pt;
}

static py::list pts_to_list(const std::vector<cv::Point2d>& pts) {
    py::list out;
    for (const auto& p : pts) out.append(pt_to_list(p.x, p.y));
    return out;
}

// 参数校验: 金字塔每层的位移容差 T, 每层至少为 4
static std::vector<int> parse_pyramid_levels(const py::object& obj) {
    std::vector<int> levels = obj.cast<std::vector<int>>();
    bool ok = !levels.empty();
    for (int l : levels)
        if (l < 4) ok = false;
    if (!ok)
        throw std::invalid_argument(
            "金字塔层级参数必须是一个列表, 且每层的位移容差 T 为不小于 4 的整数 (例如 [4, 8])。");
    return levels;
}

// ============================ PyMatcher ============================

void PyMatcher::clear() {
    detector_ = line2Dup::Detector();
    detector_ready_ = false;
    detector_params_ = py::dict();
    class_order_.clear();
    class_meta_.clear();
}

// ---------------- train ----------------

py::dict PyMatcher::train(const cv::Mat& train_image, std::vector<int> roi,
                          const std::string& class_id_in, const py::dict& train_params,
                          const std::string& save_dir_in, py::object exclusion_zones) {
    // ---------- 1. 参数解析 ----------
    if (!train_params.contains("feature_num") || !train_params.contains("pyramid_levels") ||
        !train_params.contains("weak_thresh") || !train_params.contains("strong_thresh"))
        throw std::invalid_argument("train_params 缺少必填项 (feature_num/pyramid_levels/weak_thresh/strong_thresh)。");

    int feature_num = train_params["feature_num"].cast<int>();
    std::vector<int> pyramid_levels = parse_pyramid_levels(train_params["pyramid_levels"]);
    float weak_thresh = train_params["weak_thresh"].cast<float>();
    float strong_thresh = train_params["strong_thresh"].cast<float>();
    float angle_start = train_params.contains("angle_start") ? train_params["angle_start"].cast<float>() : 0.f;
    float angle_extent = train_params.contains("angle_extent") ? train_params["angle_extent"].cast<float>() : 0.f;
    float angle_step = train_params.contains("angle_step") ? train_params["angle_step"].cast<float>() : 15.f;
    float scale_start = train_params.contains("scale_start") ? train_params["scale_start"].cast<float>() : 1.f;
    float scale_end = train_params.contains("scale_end") ? train_params["scale_end"].cast<float>() : 1.f;
    float scale_step = train_params.contains("scale_step") ? train_params["scale_step"].cast<float>() : 0.5f;

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

    // ---------- 2. 名称与目录 (中文路径在 C++ 侧直接支持) ----------
    std::string base_name = safe_file_name_cpp(class_id_in);
    const std::string& class_id = base_name;  // C++ 内部类名 = 文件名前缀, 保证两边一致
    std::string dir_str = save_dir_in.empty() ? "." : save_dir_in;
    fs::path save_dir = utf8_path(dir_str);
    std::error_code ec;
    fs::create_directories(save_dir, ec);
    fs::path img_path = save_dir / utf8_path(base_name + ".jpg");
    if (!write_image_utf8(roi_image, img_path))
        throw std::runtime_error("ROI 图片写入失败, 请检查保存目录权限或路径: " + img_path.u8string());

    // ---------- 3. padding + 全白基础掩码 + 排除区 ----------
    double diagonal = std::sqrt(double(w) * double(w) + double(h) * double(h));
    double max_scale = double(scale_end);
    int padding = int(diagonal * max_scale * 1.5) + 50;
    int padded_w = w + 2 * padding, padded_h = h + 2 * padding;

    cv::Mat padded_img = cv::Mat::zeros(
        padded_h, padded_w, roi_image.type());
    roi_image.copyTo(padded_img(cv::Rect(padding, padding, w, h)));

    cv::Mat padded_mask = cv::Mat::zeros(padded_h, padded_w, CV_8UC1);
    cv::rectangle(padded_mask, cv::Rect(padding, padding, w, h), cv::Scalar(255), -1);

    // 在掩码上绘制排除区域 (实心黑)
    if (!exclusion_zones.is_none()) {
        for (py::handle zone_obj : exclusion_zones) {
            py::dict zone = zone_obj.cast<py::dict>();
            std::string ztype = zone["type"].cast<std::string>();
            std::vector<int> r = zone["rect"].cast<std::vector<int>>();
            int ex = r[0], ey = r[1], ew = r[2], eh = r[3];
            int px = ex + padding, py_ = ey + padding;  // 排除区在 padded_mask 中的左上角
            if (ztype == "exclude_rect") {
                cv::rectangle(padded_mask, cv::Rect(px, py_, ew, eh), cv::Scalar(0), -1);
            } else if (ztype == "exclude_ellipse") {
                cv::ellipse(padded_mask, cv::Point(px + ew / 2, py_ + eh / 2),
                            cv::Size(ew / 2, eh / 2), 0, 0, 360, cv::Scalar(0), -1);
            }
        }
    }

    // ---------- 4. 生成角度/尺度信息并训练 ----------
    shape_based_matching::shapeInfo_producer producer(padded_img, padded_mask);
    producer.angle_range = collapse_range(angle_start, angle_start + angle_extent);
    producer.angle_step = angle_step;
    producer.scale_range = collapse_range(scale_start, scale_end);
    producer.scale_step = scale_step;
    if (producer.angle_step <= 0 || producer.scale_step <= 0)
        throw std::invalid_argument("angle_step / scale_step 必须大于 0。");
    producer.produce_infos();
    if (producer.infos.empty())
        throw std::runtime_error("未能生成任何形状信息 (infos)。");

    line2Dup::Detector det_tmp(feature_num, pyramid_levels, weak_thresh, strong_thresh);

    using Info = shape_based_matching::shapeInfo_producer::Info;
    std::map<int, std::vector<const Info*>> infos_by_scale;   // key = round(scale*1000)
    std::map<int, float> key_to_scale;
    for (const auto& info : producer.infos) {
        int key = int(std::lround(info.scale * 1000.0));
        infos_by_scale[key].push_back(&info);
        key_to_scale[key] = info.scale;
    }
    if (infos_by_scale.empty()) throw std::runtime_error("没有可用的尺度信息。");

    // 主尺度 = 最接近 1.0 的那组
    int main_key = 0;
    double best = -1.0;
    for (const auto& kv : key_to_scale) {
        double d = std::fabs(double(kv.second) - 1.0);
        if (best < 0 || d < best) { best = d; main_key = kv.first; }
    }

    // 一个尺度组: 先加该组基础模板, 其余角度用 addTemplate_rotate 旋转生成
    std::map<int, std::pair<double, double>> template_info;   // templ_id -> (angle, scale)
    std::vector<std::pair<double, double>> base_template_features;  // 仅主尺度组基础模板
    auto train_group = [&](std::vector<const Info*>& group, bool collect_features) {
        std::stable_sort(group.begin(), group.end(), [](const Info* a, const Info* b) {
            return std::fabs(double(a->angle)) < std::fabs(double(b->angle));
        });
        const Info* base = group[0];
        int adjusted_feature_num =
            std::max(10, std::min(int(feature_num * base->scale), feature_num));
        cv::Mat rotated_img = producer.src_of(*base);
        cv::Mat rotated_mask = producer.mask_of(*base);
        int base_id = det_tmp.addTemplate(rotated_img, class_id, rotated_mask, adjusted_feature_num);
        if (base_id == -1) return;
        template_info[base_id] = {base->angle, base->scale};
        if (collect_features) {
            // 特征在 src_of(angle, scale) 这张"旋转缩放后的图"上; 画回原始 ROI 必须做逆仿射,
            // 否则 scale != 1 / angle != 0 时特征点整体偏移甚至超出 ROI (旧版 Python 同样有此 bug)。
            const auto& templ0 = det_tmp.getTemplates(class_id, base_id)[0];
            cv::Point2f rot_center(padded_w / 2.0f, padded_h / 2.0f);
            cv::Mat M = cv::getRotationMatrix2D(rot_center, base->angle, base->scale);  // 2x3
            cv::Mat Mi;
            cv::invertAffineTransform(M, Mi);                                           // 2x3 逆
            for (const auto& feat : templ0.features) {
                cv::Point2d p_pad(feat.x + templ0.tl_x, feat.y + templ0.tl_y);          // 旋转缩放图坐标
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
    int num_templates_added = int(template_info.size());
    if (num_templates_added == 0) throw std::runtime_error("未能成功添加任何模板。");

    // ---------- 5. 预览图 (ROI + 特征点红点标注) ----------
    cv::Mat display_roi_with_features = roi_image.clone();
    for (const auto& f : base_template_features) {
        if (0 <= f.first && f.first < w && 0 <= f.second && f.second < h)
            cv::circle(display_roi_with_features,
                       cv::Point(int(f.first), int(f.second)), 2, cv::Scalar(0, 0, 255), -1);
    }

    // ---------- 6. 写 YAML (MEMORY 模式 + UTF-8 落盘, 中文目录/类别名均可用) ----------
    cv::FileStorage yfs("templates.yaml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
    det_tmp.writeClass(class_id, yfs);
    std::string yaml_buf = yfs.releaseAndGetString();
    fs::path yaml_path = save_dir / utf8_path(base_name + ".yaml");
    write_bytes_utf8(yaml_path, yaml_buf.data(), yaml_buf.size());

    // ---------- 7. 写 info.json (schema 与旧版完全一致) ----------
    std::ostringstream oss;
    oss << "{\n    \"training_params\": {\n"
        << "        \"feature_num\": " << minijson::dump_number(feature_num) << ",\n"
        << "        \"pyramid_levels\": [";
    for (size_t i = 0; i < pyramid_levels.size(); ++i) {
        if (i) oss << ", ";
        oss << pyramid_levels[i];
    }
    oss << "],\n"
        << "        \"weak_thresh\": " << minijson::dump_number(weak_thresh) << ",\n"
        << "        \"strong_thresh\": " << minijson::dump_number(strong_thresh) << "\n"
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

    // ---------- 8. 预览图落盘 (失败不影响训练结果) ----------
    fs::path preview_path = save_dir / utf8_path(base_name + ".preview.png");
    bool preview_ok = false;
    try {
        preview_ok = write_image_utf8(display_roi_with_features, preview_path);
    } catch (...) {
        preview_ok = false;
    }

    py::dict result;
    result["yaml_path"] = yaml_path.u8string();
    result["info_path"] = info_path.u8string();
    result["preview_path"] = preview_ok ? py::cast(preview_path.u8string()) : py::none();    result["save_dir"] = dir_str;
    result["base_name"] = base_name;
    result["features_image"] = py::cast(display_roi_with_features);
    result["template_count"] = num_templates_added;
    return result;
}

// ---------------- add_template_class ----------------

py::tuple PyMatcher::add_template_class(const std::string& path_str, py::object override_params) {
    // ---------- 1. 解析 yaml / info.json 两份文件的路径 ----------
    fs::path p = utf8_path(path_str);
    std::string suffix = p.extension().u8string();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });

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
    ensure_params_dict();
    py::dict final_params;
    if (!detector_params_.empty())
        final_params = py::reinterpret_steal<py::dict>(PyDict_Copy(detector_params_.ptr()));
    if (saved.is_obj()) {
        if (const minijson::Value* lv = saved.find("pyramid_levels")) {
            py::list levels;
            for (const auto& v : lv->arr) levels.append(v.as_int());
            final_params["pyramid_levels"] = levels;
        }
        if (saved.get("feature_num").is_num())
            final_params["feature_num"] = saved.get("feature_num").as_int();
        if (saved.get("weak_thresh").is_num())
            final_params["weak_thresh"] = saved.get("weak_thresh").as_num();
        if (saved.get("strong_thresh").is_num())
            final_params["strong_thresh"] = saved.get("strong_thresh").as_num();
    }
    if (!override_params.is_none()) {
        py::dict ov = override_params.cast<py::dict>();
        for (auto item : ov) final_params[item.first] = item.second;
    }
    if (!final_params.contains("pyramid_levels") || !final_params.contains("feature_num") ||
        !final_params.contains("weak_thresh") || !final_params.contains("strong_thresh"))
        throw std::invalid_argument("类别 '" + class_id + "' 的参数不完整。");

    // ---------- 4. 初始化检测器 (仅首次) ----------
    if (!detector_ready_) {
        int feature_num = final_params["feature_num"].cast<int>();
        std::vector<int> pyramid_levels = parse_pyramid_levels(final_params["pyramid_levels"]);
        float weak = final_params["weak_thresh"].cast<float>();
        float strong = final_params["strong_thresh"].cast<float>();
        detector_ = line2Dup::Detector(feature_num, pyramid_levels, weak, strong);
        detector_ready_ = true;
        detector_params_ =
            py::reinterpret_steal<py::dict>(PyDict_Copy(final_params.ptr()));
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

    return py::make_tuple(class_id, final_params);
}

void PyMatcher::ensure_params_dict() {
    if (detector_params_.is_none() || !PyDict_Check(detector_params_.ptr()))
        detector_params_ = py::dict();
}

py::list PyMatcher::get_loaded_class_ids() const {
    py::list out;
    for (const auto& c : class_order_) out.append(c);
    return out;
}

py::list PyMatcher::get_base_template_features(const std::string& class_id) const {
    py::list out;
    auto it = class_meta_.find(class_id);
    if (it == class_meta_.end()) return out;
    for (const auto& f : it->second.base_features) {
        py::list pt;
        pt.append(f.first);
        pt.append(f.second);
        out.append(pt);
    }
    return out;
}

// ---------------- match ----------------

py::list PyMatcher::match(const cv::Mat& image, double score_threshold,
                          py::object class_ids_to_match,
                          bool use_nms, double nms_threshold,
                          py::object grasp_points_config,
                          int max_matches, double min_fitness, bool use_refine,
                          double max_overlap, py::object masks, bool fill_overlap) {
    if (!detector_ready_ || class_order_.empty())
        throw std::runtime_error("没有加载任何模板类别，请先加载或训练模板。");

    // 1. 图像预处理: 中值滤波 + 补边到 16 的倍数 (MIPP 步长要求)
    cv::Mat image_to_match;
    cv::medianBlur(image, image_to_match, 3);
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
    // 保证内存连续 (MIPP 16 字节步长要求)
    image_to_match = image_to_match.clone();

    // 场景 mask 预处理: 尺寸须与输入一致, 跟随补边, 非0即参与
    cv::Mat mask_to_use;
    if (!masks.is_none()) {
        cv::Mat mask_arr = masks.cast<cv::Mat>();
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
    std::vector<std::string> search_ids;
    if (class_ids_to_match.is_none()) {
        search_ids = class_order_;
    } else if (py::isinstance<py::str>(class_ids_to_match)) {
        search_ids.push_back(class_ids_to_match.cast<std::string>());
    } else {
        for (py::handle item : class_ids_to_match)
            search_ids.push_back(item.cast<std::string>());
    }
    py::list results;
    if (search_ids.empty()) return results;

    line2Dup::MatchParams params;
    params.class_ids = search_ids;
    params.min_confidence = float(score_threshold);
    params.nms = use_nms;
    params.nms_overlap = float(nms_threshold);
    params.use_refine = use_refine;
    params.min_fitness = use_refine ? float(min_fitness) : 0.f;
    params.max_matches = max_matches;
    params.max_overlap = float(max_overlap);
    params.fill_overlap = fill_overlap;
    if (!mask_to_use.empty()) params.masks = mask_to_use;

    // 核心检测: 类别过滤 + 重叠度过滤 + NMS + ICP 精修一次完成
    std::vector<line2Dup::Match> raw_matches = detector_.match(image_to_match, params);
    std::sort(raw_matches.begin(), raw_matches.end());   // similarity 降序

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
                std::fprintf(stderr, "[PyMatcher] 类别 %s 缺少原始尺寸信息，跳过\n", m.class_id.c_str());
                continue;
            }
            double center_x = (roi_w + 2.0 * padding) / 2.0;
            double center_y = (roi_h + 2.0 * padding) / 2.0;

            // 模板对象 (最高分辨率层级)
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
            cv::Matx33d M = cv::Matx33d::eye();  // 3x3 增量矩阵 (回退路径)
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
            //   transform 分支: 偏移 = (tl_x, tl_y), A = T
            //   refine 回退分支: 偏移 = (match.x, match.y), A = M  (与旧 Python 实现一致)
            auto to_scene = [&](cv::Point2d p) -> cv::Point2d {
                double ox = have_transform ? tl_x : double(m.x);
                double oy = have_transform ? tl_y : double(m.y);
                double tx = p.x + ox, ty = p.y + oy;
                const cv::Matx33d& A = have_transform ? T : M;
                return {A(0, 0) * tx + A(0, 1) * ty + A(0, 2),
                        A(1, 0) * tx + A(1, 1) * ty + A(1, 2)};
            };
            double delta_angle = [&]() {
                const cv::Matx33d& A = have_transform ? T : M;
                return std::atan2(A(1, 0), A(0, 0)) * 180.0 / CV_PI;
            }();

            // ROI 点 -> 模板坐标系: 平移到填充图, 绕中心顺时针旋转缩放, 再裁剪偏移
            auto roi_to_templ = [&](cv::Point2d p) -> cv::Point2d {
                double x_rel = p.x + padding - center_x;
                double y_rel = p.y + padding - center_y;
                double cos_a = std::cos(angle_rad), sin_a = std::sin(angle_rad);
                double x_rot = center_x + scale * (x_rel * cos_a + y_rel * sin_a);
                double y_rot = center_y + scale * (-x_rel * sin_a + y_rel * cos_a);
                return {x_rot - tl_x, y_rot - tl_y};
            };

            // 原始 ROI 四个角点 (ROI 坐标系) -> 精修后场景坐标
            std::vector<cv::Point2d> refined_corners;
            const std::vector<cv::Point2d> roi_corners = {
                {0, 0}, {double(roi_w), 0}, {double(roi_w), double(roi_h)}, {0, double(roi_h)}};
            for (const auto& p : roi_corners)
                refined_corners.push_back(to_scene(roi_to_templ(p)));

            // 精修后的特征点 (模板特征坐标 -> 场景坐标)
            std::vector<cv::Point2d> refined_features;
            for (const auto& f : templ0->features)
                refined_features.push_back(to_scene(cv::Point2d(f.x, f.y)));

            cv::Point2d center(0, 0);
            for (const auto& p : refined_corners) center += p;
            center *= 0.25;

            py::dict res;
            res["class_id"] = m.class_id;
            res["template_id"] = m.template_id;
            res["score"] = double(m.similarity);
            res["x"] = double(m.x);
            res["y"] = double(m.y);
            res["icp_refined"] = use_refine;
            res["refined_box_points"] = pts_to_list(refined_corners);
            res["matched_features"] = pts_to_list(refined_features);
            res["refined_x"] = center.x;
            res["refined_y"] = center.y;
            res["refined_angle"] = angle_deg - delta_angle;   // 最终角度 = 训练角度 - ICP 增量角
            res["fitness"] = double(m.fitness);
            res["overlap"] = double(m.overlap);

            // 抓取点: 优先用调用方配置 (ROI 坐标), 无则用精修中心
            std::vector<cv::Point2d> grasp_pts;
            if (!grasp_points_config.is_none()) {
                py::dict gcfg = grasp_points_config.cast<py::dict>();
                if (gcfg.contains(m.class_id.c_str())) {
                    for (py::handle item : gcfg[m.class_id.c_str()]) {
                        py::sequence pt = py::reinterpret_borrow<py::sequence>(item);
                        grasp_pts.emplace_back(pt[0].cast<double>(), pt[1].cast<double>());
                    }
                }
            }
            if (!grasp_pts.empty()) {
                std::vector<cv::Point2d> refined_grasp;
                for (const auto& p : grasp_pts)
                    refined_grasp.push_back(to_scene(roi_to_templ(p)));
                res["grasp_points"] = pts_to_list(refined_grasp);
                py::list first_pt;
                first_pt.append(refined_grasp[0].x);
                first_pt.append(refined_grasp[0].y);
                res["grasp_point"] = first_pt;
            } else {
                py::list center_pt;
                center_pt.append(center.x);
                center_pt.append(center.y);
                res["grasp_point"] = center_pt;
                py::list center_pts;
                center_pts.append(center_pt);
                res["grasp_points"] = center_pts;
            }

            results.append(res);
        } catch (const std::exception& e) {
            // 与旧 Python 实现一致: 单个结果处理失败不中断整体匹配
            std::fprintf(stderr, "[PyMatcher] 处理匹配结果 [%s] 时出错: %s\n",
                         m.class_id.c_str(), e.what());
            continue;
        }
    }

    return results;
}
