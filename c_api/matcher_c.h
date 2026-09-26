#ifndef SBM_MATCHER_C_H
#define SBM_MATCHER_C_H
/* ============================================================================
 * shape_based_matching 的 C ABI (供 C# P/Invoke, 也可被 C/C++/Rust 等直接链接)
 *
 * 约定:
 *   1. 只暴露 POD 结构体与裸指针, 不暴露 std::string / std::vector / cv::Mat;
 *   2. 调用约定为 cdecl (C# 侧 DllImport 要写 CallingConvention.Cdecl);
 *   3. 返回值: 0 表示成功, 负数为错误码; 返回指针的函数失败时返回 NULL;
 *      错误详情用 sbm_last_error() 取 (UTF-8 字符串);
 *   4. 结果里的 const char* / const double* 都指向库内部缓冲,
 *      在下一次调用或 sbm_destroy() 之前有效, 调用方不要 free。
 * ==========================================================================*/

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
    #ifdef SBM_BUILD
        #define SBM_API __declspec(dllexport)
    #else
        #define SBM_API __declspec(dllimport)
    #endif
#else
    #define SBM_API __attribute__((visibility("default")))
#endif

/* ---------------- 图像: 行优先, channels=1 灰度 / 3 为 BGR ---------------- */
typedef struct sbm_image_t {
    const unsigned char* data;   /* 像素数据 */
    int width;
    int height;
    int channels;                /* 1 或 3 */
    int step;                    /* 每行字节数; 0 表示 width*channels (无对齐填充) */
} sbm_image_t;

/* ---------------- 训练参数 ---------------- */
#define SBM_MAX_PYRAMID_LEVELS 8
typedef struct sbm_train_params_t {
    int feature_num;
    int pyramid_level_count;
    int pyramid_levels[SBM_MAX_PYRAMID_LEVELS];
    double weak_thresh;
    double strong_thresh;
    double angle_start;
    double angle_extent;
    double angle_step;
    double scale_start;
    double scale_end;
    double scale_step;
} sbm_train_params_t;

/* ---------------- 排除区 ---------------- */
typedef struct sbm_exclusion_zone_t {
    const char* type;   /* "exclude_rect" 或 "exclude_ellipse" */
    int x, y, w, h;
} sbm_exclusion_zone_t;

/* ---------------- 训练结果 ---------------- */
typedef struct sbm_train_result_t {
    const char* yaml_path;
    const char* info_path;
    const char* preview_path;    /* 预览图未写出时为 NULL */
    const char* save_dir;
    const char* base_name;
    int template_count;
    sbm_image_t features_image;  /* ROI + 特征点红点图, 指向库内部缓冲 */
} sbm_train_result_t;

/* ---------------- 阈值自动探测参数 ----------------
 * sbm_estimate_thresholds 用来按"这张训练图自己"搜出合适的 weak/strong。
 * 零值一律解释为"用默认", 所以调 sbm_threshold_search_params_init() 拿一份
 * 填好默认值的结构体是最省事的用法; 想只改其中一项就手动覆盖对应字段。 */
typedef struct sbm_threshold_search_params_t {
    int feature_num;                            /* <=0 -> 100 */
    int pyramid_level_count;                    /* <=0 -> [4, 8] (只影响自检) */
    int pyramid_levels[SBM_MAX_PYRAMID_LEVELS];
    double scale_end;                           /* <=0 -> 1.0   (与训练参数一致, 只影响 padding) */
    double weak_ratio;                          /* <=0 -> 0.5   (弱阈值 = weak_ratio * 强阈值) */
    double strong_min;                          /* <=0 -> 4.0   (强阈值搜索下界) */
    double strong_max;                          /* <=0 -> 255.0 (强阈值搜索上界) */
    int run_self_check;                         /* 0=默认(开) / 1=强制开 / -1=强制关 */
} sbm_threshold_search_params_t;

/* ---------------- 阈值自动探测结果 ----------------
 * ok = true 时 weak_thresh / strong_thresh 才是可信的建议值;
 * ok = false 时这两个数是默认值 30/60, 请直接看 note 判断是图不适合还是 ROI 选得不好。
 * note 与 features_image 都指向库内部缓冲, 在下一次调用或 sbm_destroy() 前有效,
 * 调用方不要 free(如需长期持有请自行拷贝)。 */
typedef struct sbm_threshold_estimate_t {
    double weak_thresh;
    double strong_thresh;
    int ok;
    /* ---- 诊断信息, 便于人工复核 ---- */
    int mask_pixels;            /* 参与训练的有效像素数 */
    int requested_features;     /* 期望特征点数 */
    int candidates;             /* 该阈值下的候选点数 */
    int features;               /* 该阈值下实际取到的特征点数 */
    double median_gradient;     /* 有效区梯度中位数 */
    double p95_gradient;
    double self_score;          /* 自匹配自检得分, -1 = 未计算 */
    const char* note;           /* 库内部缓冲 */
    sbm_image_t features_image; /* ROI 原图 + 特征点红点, 指向库内部缓冲 */
} sbm_threshold_estimate_t;

/* ---------------- 匹配结果 ---------------- */
typedef struct sbm_match_result_t {
    const char* class_id;
    int template_id;
    double score;
    double x, y;
    int icp_refined;
    double box[8];               /* 旋转外框 4 个角点: x0,y0,x1,y1,x2,y2,x3,y3 */
    int feature_count;
    const double* features;      /* 匹配到的特征点: x,y,x,y,... 共 2*feature_count */
    double refined_x, refined_y, refined_angle;
    double fitness;
    double overlap;
    int grasp_count;
    const double* grasp_points;  /* 抓取点: x,y,... 共 2*grasp_count */
    /* --- 以下三项为后加字段, 追加在结构体末尾, 前面所有字段的偏移都不变 --- */
    double angle;                /* 命中的模板训练时的旋转角 (度)。
                                  * 注意与 refined_angle 不是一回事:
                                  *   angle        模板自身训练就用的角度 (来自 info.json)
                                  *   refined_angle 最终姿态 = angle - ICP 增量角 */
    double scale;                /* 命中的模板训练时的缩放系数 (来自 info.json 的
                                  * templates[template_id].scale), 缺省 1.0 */
    double refined_scale;        /* ICP 精修引入的额外缩放, 与 refined_angle 同口径;
                                  * 目标最终相对原图的大小 = scale * refined_scale, 缺省 1.0 */
} sbm_match_result_t;

/* ============================ 句柄 ============================ */

SBM_API void* sbm_create(void);
SBM_API void  sbm_destroy(void* handle);
SBM_API void  sbm_clear(void* handle);   /* 清空已加载的模板 */

/* 填一份默认训练参数 (feature_num=100, pyramid=[4,8], weak=30, strong=60 ...) */
SBM_API void  sbm_train_params_init(sbm_train_params_t* params);

/* ============================ 训练 ============================ */
SBM_API int   sbm_train(void* handle,
                        const sbm_image_t* image,
                        const int roi[4],            /* x, y, w, h */
                        const char* class_id,        /* UTF-8, 中文可用 */
                        const sbm_train_params_t* params,
                        const char* save_dir,        /* UTF-8, 中文可用 */
                        const sbm_exclusion_zone_t* zones,   /* 可为 NULL */
                        int zone_count,
                        const sbm_image_t* positive_mask,    /* 可为 NULL */
                        const sbm_image_t* negative_mask,    /* 可为 NULL */
                        sbm_train_result_t* out);

/* ============================ 加载 ============================ */
/* path 可为 xxx.yaml / xxx.info.json / xxx.json。
 * override 传 NULL 表示不覆盖; 返回 class_id (库内部缓冲), 失败返回 NULL。 */
SBM_API const char* sbm_add_template_class(void* handle,
                                           const char* path,
                                           const sbm_train_params_t* override_params,
                                           sbm_train_params_t* out_final_params /* 可为 NULL */);

SBM_API int         sbm_loaded_class_count(void* handle);
SBM_API const char* sbm_loaded_class_id(void* handle, int index);

/* 某类别训练时主模板的特征点 (ROI 坐标); 返回写入的点数, 负数为错误码 */
SBM_API int sbm_base_features(void* handle, const char* class_id,
                              double* out_xy, int max_points);

/* ============================ 阈值自动探测 ============================ */
/* 按训练图自身的梯度分布给出 weak_thresh / strong_thresh 的建议值。
 * 传 NULL 给 params 表示用全部默认。zones / positive_mask / negative_mask 与
 * sbm_train 语义一致(可为 NULL / 空)。返回 0 成功, 负数为错误码。 */
SBM_API void  sbm_threshold_search_params_init(sbm_threshold_search_params_t* params);

SBM_API int   sbm_estimate_thresholds(void* handle,
                                      const sbm_image_t* image,
                                      const int roi[4],                /* x, y, w, h */
                                      const sbm_threshold_search_params_t* params,
                                      const sbm_exclusion_zone_t* zones,   /* 可为 NULL */
                                      int zone_count,
                                      const sbm_image_t* positive_mask,    /* 可为 NULL */
                                      const sbm_image_t* negative_mask,    /* 可为 NULL */
                                      sbm_threshold_estimate_t* out);

/* ============================ 匹配 ============================ */
/* 抓取点配置: {class_id -> [[x,y],...]} (ROI 坐标), count=0 表示清除该类别配置。
 * 必须在 sbm_match 之前调用; 不设置则抓取点退化为精修中心。 */
SBM_API int sbm_set_grasp_points(void* handle, const char* class_id,
                                 const double* xy, int count);

/* 返回匹配到的结果个数 (可能为 0), 负数为错误码。
 * class_ids 传 NULL 或 class_id_count<=0 表示匹配全部已加载类别。 */
SBM_API int sbm_match(void* handle,
                      const sbm_image_t* image,
                      double score_threshold,
                      const char* const* class_ids,
                      int class_id_count,
                      int use_nms, double nms_threshold,
                      int max_matches, double min_fitness, int use_refine,
                      double max_overlap,
                      const sbm_image_t* masks,      /* 可为 NULL */
                      int fill_overlap);

/* 取第 index 个匹配结果 (0 <= index < sbm_match 的返回值) */
SBM_API int sbm_match_result(void* handle, int index, sbm_match_result_t* out);

/* ============================ 错误 ============================ */
SBM_API const char* sbm_last_error(void* handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  /* SBM_MATCHER_C_H */
