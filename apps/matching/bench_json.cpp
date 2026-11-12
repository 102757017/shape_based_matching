// bench_json —— JSON 配置驱动的模板匹配基准测试
//
// 用法:
//   bench_json --config bench/bench.json [--repeat 10] [--warmup 2] [--out result.png]
//
// JSON 配置(与 MatcherCore 语义对齐, 字段均可省略, 省略用默认值):
// {
//   "class_id": "part",
//   "save_dir": "bench_out",
//   "train": {
//     "image": "test/case3/train.png",
//     "roi": [660, 300, 420, 420],
//     "feature_num": 128,
//     "pyramid_levels": [4, 8],
//     "weak_thresh": 30,
//     "strong_thresh": 60,
//     "angle_start": 0, "angle_extent": 360, "angle_step": 1,
//     "scale_start": 1.0, "scale_end": 1.0, "scale_step": 0.5
//   },
//   "match": {
//     "scene": "test/case3/test.png",
//     "score_threshold": 90,
//     "use_nms": true, "nms_threshold": 0.5,
//     "max_matches": 10,
//     "use_refine": true, "min_fitness": 0.3,
//     "max_overlap": 1.0
//   }
// }
//
// 流程: 配置 -> (模板不存在则)训练落盘 -> 加载 -> 预热 -> 重复计时取最小/平均
//      -> 输出 top 匹配(得分/位置/角度) -> 可视化结果图
// 输出格式刻意做成一行一个结果的稳定文本, 便于优化前后 diff 对比精度。

#include "matcher_core.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mini_json.h"

using namespace std;
using namespace sbm;

static std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("打不开配置文件: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static cv::Mat imread_gray_or_bgr(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("读图失败: " + path);
    return img;
}

struct Config {
    std::string class_id = "part";
    std::string save_dir = "bench_out";
    // train
    std::string train_image;
    std::vector<int> roi{130, 110, 270, 270};
    int feature_num = 128;
    std::vector<int> pyramid_levels{4, 8};
    float weak_thresh = 30.f;
    float strong_thresh = 60.f;
    float angle_start = 0.f, angle_extent = 360.f, angle_step = 1.f;
    float scale_start = 1.f, scale_end = 1.f, scale_step = 0.5f;
    // match
    std::string scene;
    double score_threshold = 90.0;
    bool use_nms = true;
    double nms_threshold = 0.5;
    int max_matches = 10;
    bool use_refine = true;
    double min_fitness = 0.3;
    double max_overlap = 1.0;
};

static Config parseConfig(const std::string& json_text) {
    Config c;
    minijson::Value root = minijson::parse(json_text);

    if (root.get("class_id").is_str()) c.class_id = root.get("class_id").as_str();
    if (root.get("save_dir").is_str()) c.save_dir = root.get("save_dir").as_str();

    const minijson::Value* tr = root.find("train");
    if (tr && tr->is_obj()) {
        if (tr->get("image").is_str()) c.train_image = tr->get("image").as_str();
        if (const minijson::Value* roi = tr->find("roi"); roi && roi->is_arr() && roi->arr.size() == 4) {
            for (int i = 0; i < 4; ++i) c.roi[i] = roi->arr[i].as_int();
        }
        if (tr->get("feature_num").is_num()) c.feature_num = tr->get("feature_num").as_int();
        if (const minijson::Value* lv = tr->find("pyramid_levels"); lv && lv->is_arr()) {
            std::vector<int> ls;
            for (const auto& v : lv->arr) ls.push_back(v.as_int());
            if (!ls.empty()) c.pyramid_levels = ls;
        }
        if (tr->get("weak_thresh").is_num())   c.weak_thresh   = (float)tr->get("weak_thresh").as_num();
        if (tr->get("strong_thresh").is_num()) c.strong_thresh = (float)tr->get("strong_thresh").as_num();
        if (tr->get("angle_start").is_num())  c.angle_start  = (float)tr->get("angle_start").as_num();
        if (tr->get("angle_extent").is_num()) c.angle_extent = (float)tr->get("angle_extent").as_num();
        if (tr->get("angle_step").is_num())   c.angle_step   = (float)tr->get("angle_step").as_num();
        if (tr->get("scale_start").is_num())  c.scale_start  = (float)tr->get("scale_start").as_num();
        if (tr->get("scale_end").is_num())    c.scale_end    = (float)tr->get("scale_end").as_num();
        if (tr->get("scale_step").is_num())   c.scale_step   = (float)tr->get("scale_step").as_num();
    }

    const minijson::Value* mt = root.find("match");
    if (mt && mt->is_obj()) {
        if (mt->get("scene").is_str()) c.scene = mt->get("scene").as_str();
        if (mt->get("score_threshold").is_num()) c.score_threshold = mt->get("score_threshold").as_num();
        if (const minijson::Value* b = mt->find("use_nms"); b && b->type == minijson::Value::BOOL)
            c.use_nms = b->boolean;
        if (mt->get("nms_threshold").is_num())   c.nms_threshold = mt->get("nms_threshold").as_num();
        if (mt->get("max_matches").is_num())     c.max_matches = mt->get("max_matches").as_int();
        if (const minijson::Value* b = mt->find("use_refine"); b && b->type == minijson::Value::BOOL)
            c.use_refine = b->boolean;
        if (mt->get("min_fitness").is_num())     c.min_fitness = mt->get("min_fitness").as_num();
        if (mt->get("max_overlap").is_num())     c.max_overlap = mt->get("max_overlap").as_num();
    }
    return c;
}

int main(int argc, char** argv) {
    std::string configPath = "bench/bench.json";
    int repeat = 10, warmup = 2;
    std::string outPath;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--config") configPath = next();
        else if (a == "--repeat") repeat = std::max(1, std::atoi(next().c_str()));
        else if (a == "--warmup") warmup = std::max(0, std::atoi(next().c_str()));
        else if (a == "--out") outPath = next();
        else { std::cerr << "未知参数 " << a << "\n"; return 1; }
    }

    try {
        Config cfg = parseConfig(slurp(configPath));
        std::cout << "[bench] class=" << cfg.class_id
                  << " features=" << cfg.feature_num
                  << " pyramid=";
        for (int t : cfg.pyramid_levels) std::cout << t << ",";
        std::cout << " angle_step=" << cfg.angle_step
                  << " scale=[" << cfg.scale_start << "," << cfg.scale_end << "]" << std::endl;

        MatcherCore core;

        // ---------- 1. 训练(或复用已有模板) ----------
        std::string yaml_path = cfg.save_dir + "/" + cfg.class_id + ".yaml";
        {
            std::ifstream probe(yaml_path, std::ios::binary);
            if (probe.good()) {
                std::cout << "[bench] 复用已训练模板: " << yaml_path << std::endl;
            } else {
                cv::Mat train_img = imread_gray_or_bgr(cfg.train_image);
                TrainParams tp;
                tp.feature_num = cfg.feature_num;
                tp.pyramid_levels = cfg.pyramid_levels;
                tp.weak_thresh = cfg.weak_thresh;
                tp.strong_thresh = cfg.strong_thresh;
                tp.angle_start = cfg.angle_start;
                tp.angle_extent = cfg.angle_extent;
                tp.angle_step = cfg.angle_step;
                tp.scale_start = cfg.scale_start;
                tp.scale_end = cfg.scale_end;
                tp.scale_step = cfg.scale_step;

                auto t0 = std::chrono::steady_clock::now();
                TrainResult tr = core.train(train_img, cfg.roi, cfg.class_id, tp,
                                            cfg.save_dir, {}, cv::Mat(), cv::Mat());
                auto t1 = std::chrono::steady_clock::now();
                std::cout << "[bench] 训练完成: " << tr.template_count << " 个模板, 耗时 "
                          << std::chrono::duration<double, std::milli>(t1 - t0).count()
                          << " ms" << std::endl;
            }
        }

        // ---------- 2. 加载 ----------
        TrainParamsOverride ov;   // 不覆盖, 用训练时记录的参数
        LoadedClass lc = core.add_template_class(yaml_path, ov);
        std::cout << "[bench] 已加载类别: " << lc.class_id
                  << " (feature_num=" << lc.params.feature_num << ")" << std::endl;

        // ---------- 3. 场景图 ----------
        cv::Mat scene = imread_gray_or_bgr(cfg.scene);
        // 优化配套: 预填充到 64 的倍数。Detector::match 内部会裁剪到 lcm(T)<<levels 的倍数
        // (如金字塔 [4,8,16] 裁到 64 倍数), 若不预填充, 场景右/下边缘会被裁掉导致目标丢失。
        // 零填充区梯度全零, 不产生任何响应, 对结果无副作用。
        {
            int pw = (scene.cols + 63) / 64 * 64;
            int ph = (scene.rows + 63) / 64 * 64;
            if (pw != scene.cols || ph != scene.rows) {
                cv::Mat padded = cv::Mat::zeros(ph, pw, scene.type());
                scene.copyTo(padded(cv::Rect(0, 0, scene.cols, scene.rows)));
                scene = padded;
            }
        }
        cv::Mat scene_bgr = scene;
        if (scene.channels() == 1) cv::cvtColor(scene, scene_bgr, cv::COLOR_GRAY2BGR);

        // ---------- 4. 预热 + 计时 ----------
        std::vector<MatchResult> last;
        for (int i = 0; i < warmup; ++i) {
            last = core.match(scene, cfg.score_threshold, nullptr,
                              cfg.use_nms, cfg.nms_threshold, nullptr,
                              cfg.max_matches, cfg.min_fitness, cfg.use_refine,
                              cfg.max_overlap, cv::Mat(), true);
        }
        double best_ms = 1e18, sum_ms = 0;
        for (int i = 0; i < repeat; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            last = core.match(scene, cfg.score_threshold, nullptr,
                              cfg.use_nms, cfg.nms_threshold, nullptr,
                              cfg.max_matches, cfg.min_fitness, cfg.use_refine,
                              cfg.max_overlap, cv::Mat(), true);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best_ms = std::min(best_ms, ms);
            sum_ms += ms;
        }
        std::cout << "[bench] match " << repeat << " 次: min=" << best_ms
                  << " ms, avg=" << sum_ms / repeat << " ms" << std::endl;

        // ---------- 5. 结果(稳定格式, 便于 diff) ----------
        std::cout << "[result] count=" << last.size() << std::endl;
        for (size_t i = 0; i < last.size(); ++i) {
            const MatchResult& m = last[i];
            std::printf("[result] #%zu score=%.2f x=%.2f y=%.2f angle=%.3f scale=%.4f "
                        "refined=(%.2f,%.2f,%.3f) fitness=%.3f tpl=%d\n",
                        i, m.score, m.x, m.y, m.angle, m.scale,
                        m.refined_x, m.refined_y, m.refined_angle, m.fitness, m.template_id);
        }

        // ---------- 6. 可视化 ----------
        if (!outPath.empty()) {
            cv::Mat vis = scene_bgr.clone();
            for (size_t i = 0; i < last.size(); ++i) {
                const MatchResult& m = last[i];
                // 用精修后的特征点画框
                std::vector<cv::Point> pts;
                for (auto& p : m.refined_box_points)
                    pts.emplace_back((int)std::lround(p.first), (int)std::lround(p.second));
                if (pts.size() == 4) {
                    cv::polylines(vis, pts, true, cv::Scalar(0, 255, 0), 2);
                    cv::putText(vis, cv::format("%.1f", m.score),
                                pts[0] + cv::Point(0, -6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                }
                for (auto& p : m.matched_features)
                    cv::circle(vis, {(int)std::lround(p.first), (int)std::lround(p.second)},
                               2, cv::Scalar(255, 0, 0), -1);
            }
            if (!cv::imwrite(outPath, vis))
                std::cerr << "[bench] 可视化写盘失败: " << outPath << std::endl;
            else
                std::cout << "[bench] 可视化: " << outPath << std::endl;
        }
        std::cout << "[bench] done" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[bench] ERROR: " << e.what() << std::endl;
        return 2;
    }
}
