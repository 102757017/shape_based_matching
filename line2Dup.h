#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>

#include "mipp.h"  // for SIMD in different platforms

namespace line2Dup
{

    struct Feature
    {
        int x;
        int y;
        int label;
        float theta;

        void read(const cv::FileNode& fn);
        void write(cv::FileStorage& fs) const;

        Feature() : x(0), y(0), label(0) {}
        Feature(int x, int y, int label);
    };
    inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

    struct Template
    {
        int width;
        int height;
        int tl_x;
        int tl_y;
        int pyramid_level;
        std::vector<Feature> features;

        void read(const cv::FileNode& fn);
        void write(cv::FileStorage& fs) const;
    };

    struct RegistrationResult {
        std::vector<std::vector<float>> transformation; // 3x3 矩阵
        float fitness;
        float inlier_rmse;
    };

    // 匹配结果与实际目标的重叠度(mask 面积交并比)
    struct OverlapResult {
        float iou = 0.f;         // 交集 / 并集, 0~1
        float inter_area = 0.f;  // 交集像素数
        float union_area = 0.f;  // 并集像素数
        float pred_area = 0.f;   // 预测 mask 像素数(可用于 precision = inter/pred)
        float gt_area = 0.f;     // 真值 mask 像素数(可用于 recall = inter/gt)

        float precision() const { return pred_area > 0 ? inter_area / pred_area : 0.f; }
        float recall() const { return gt_area > 0 ? inter_area / gt_area : 0.f; }
    };

    class ColorGradientPyramid
    {
    public:
        ColorGradientPyramid(const cv::Mat& src, const cv::Mat& mask,
            float weak_threshold, size_t num_features,
            float strong_threshold);

        void quantize(cv::Mat& dst) const;

        bool extractTemplate(Template& templ) const;

        void pyrDown();
        void quantizedOrientations(const cv::Mat& src, cv::Mat& magnitude,
            cv::Mat& angle, cv::Mat& angle_ori, float threshold);
    public:
        void update();
        /// Candidate feature with a score
        struct Candidate
        {
            Candidate(int x, int y, int label, float score);

            /// Sort candidates with high score to the front
            bool operator<(const Candidate& rhs) const
            {
                return score > rhs.score;
            }

            Feature f;
            float score;
        };

        bool has_pydown = false;
        cv::Mat dx_, dy_;

        cv::Mat src;
        cv::Mat mask;

        int pyramid_level;
        cv::Mat angle;
        cv::Mat magnitude;
        cv::Mat angle_ori;

        float weak_threshold;
        size_t num_features;
        float strong_threshold;
        static bool selectScatteredFeatures(const std::vector<Candidate>& candidates,
            std::vector<Feature>& features,
            size_t num_features, float distance);
    };
    inline ColorGradientPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

    class ColorGradient
    {
    public:
        ColorGradient();
        ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

        std::string name() const;

        float weak_threshold;
        size_t num_features;
        float strong_threshold;
        void read(const cv::FileNode& fn);
        void write(cv::FileStorage& fs) const;

        cv::Ptr<ColorGradientPyramid> process(const cv::Mat src, const cv::Mat& mask = cv::Mat()) const
        {
            return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features, strong_threshold);
        }
    };

    /**
     * \brief 一次匹配的结果
     *
     * 术语约定(与工业视觉 / halcon 一致, 两者语义不同, 不要混用):
     *
     *   置信度 (confidence / similarity)
     *       = 匹配上的特征点数 / 模板总特征点数, 取值 0~1(库中 similarity 是它的百分制 0~100)
     *       回答"这个结果有多像模板"。
     *
     *   重叠度 (overlap)
     *       = 本目标矩形 mask 与其它目标矩形 mask 的重叠面积 / 自身面积, 取值 0~1
     *       回答"这个结果有多大一块被别人盖住了"。
     *       注意: 不是 IoU! IoU 的分母是并集(交/并), 这里的分母是自身面积(交/自身),
     *       所以同样两块区域, overlap 通常比 IoU 大, 它衡量的是"被遮挡/被重复检测"的程度。
     */
    struct Match
    {
        Match();

        Match(int x, int y, float similarity, const std::string& class_id, int template_id);

        /// Sort matches with high similarity to the front
        bool operator<(const Match& rhs) const
        {
            // Secondarily sort on template_id for the sake of duplicate removal
            if (similarity != rhs.similarity)
                return similarity > rhs.similarity;
            else
                return template_id < rhs.template_id;
        }

        bool operator==(const Match& rhs) const
        {
            return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
        }

        // ---- 位置 ----
        int x;              // 模板包围盒左上角在场景图中的 x
        int y;              // 模板包围盒左上角在场景图中的 y
        int width;          // 目标矩形 mask 的宽(= 模板包围盒宽, 未精修时)
        int height;         // 目标矩形 mask 的高

        // ---- 置信度 ----
        float similarity;   // 置信度的百分制形式, 0~100(= confidence * 100)
        std::string class_id;
        int template_id;

        // ---- 扩展信息(由 match(img, MatchParams) 填充) ----
        float confidence;   // 置信度 0~1, 即 similarity / 100
        float overlap;      // 重叠度 0~1: 与其它目标的矩形 mask 重叠面积 / 自身面积,
                            //            0 表示没有和任何目标相交。是"被遮挡比例", 不是 IoU

        cv::Mat transform;  // 2x3 CV_32F 仿射变换, 训练图(模板)坐标系 -> 场景图坐标系。
                            //    用法: cv::warpAffine(模板图/mask, dst, m.transform, 场景图.size())
                            //    即可把模板轮廓画到匹配位置。未精修时退化为纯平移+训练时的旋转缩放
        float angle;        // 从 transform 解出的旋转角(度), 相对训练时用的那一版模板
        float scale;        // 从 transform 解出的缩放系数

        float fitness;      // ICP 内点率 0~1(精修后有效, 未精修为 -1): 模板点中在场景边缘上找到对应点的比例
        float inlier_rmse;  // ICP 内点残差(像素), 未精修为 -1
    };

    inline Match::Match()
        : x(0), y(0), width(0), height(0), similarity(0), template_id(0),
        confidence(0), overlap(0), angle(0), scale(1), fitness(-1), inlier_rmse(-1)
    {
    }

    inline Match::Match(int _x, int _y, float _similarity, const std::string& _class_id, int _template_id)
        : x(_x), y(_y), width(0), height(0), similarity(_similarity), class_id(_class_id),
        template_id(_template_id), confidence(_similarity / 100.f), overlap(0),
        angle(0), scale(1), fitness(-1), inlier_rmse(-1)
    {
    }

    /**
     * \brief 匹配参数集合, 一次把"匹配 + 过滤 + 精修"说清楚
     *
     * 过滤的执行顺序(见 Detector::match(img, MatchParams)):
     *   1. class_ids        只匹配指定类别
     *   2. min_confidence   丢掉置信度不够的
     *   3. 计算每个结果的重叠度 overlap
     *   4. max_overlap      丢掉被遮挡太厉害的
     *   5. nms              非极大值抑制, 同一目标只留置信度最高的那个
     *   6. max_matches      截断到最大数量
     *   7. use_refine       对活下来的结果做 ICP 精修, 再按 min_fitness 过滤
     */
    struct MatchParams
    {
        // ---------- 类别 ----------
        std::vector<std::string> class_ids;  // 只匹配这些类别; 空 = 匹配全部类别

        // ---------- 置信度 ----------
        float min_confidence = 90.f;         // 置信度下限, 0~100(对应旧接口的 threshold)

        // ---------- 数量 ----------
        int max_matches = 0;                 // 最多返回几个结果, 0 = 不限

        // ---------- ICP 精修开关 ----------
        bool use_refine = false;             // true: 匹配后自动对每个结果跑 ICP, 填 transform/angle/scale/fitness
                                             // false: 不做精修, 速度快, transform 只有平移部分
        float min_fitness = 0.f;             // 精修后的内点率下限 0~1, 低于就丢弃(仅 use_refine 时生效)

        // ---------- 重叠度 ----------
        float max_overlap = 1.f;             // 重叠度上限 0~1; 超过说明该结果大部分被别的目标盖住, 丢弃
                                             // 1.0 = 不过滤, 0.5 = 自身一半以上被盖住就不要

        // ---------- 非极大值抑制 ----------
        bool nms = true;                     // 是否做 NMS(同一目标被多个模板/角度重复命中时只留最好的)
        float nms_overlap = 0.5f;            // NMS 的重叠度阈值 0~1; 与已保留结果重叠超过它就被抑制
                                             // 0 = 只要相交就抑制, 1 = 几乎不抑制

        // ---------- 其它 ----------
        cv::Mat masks;                       // 场景 mask, 只在该区域内匹配; 空 = 整幅图
        bool fill_overlap = true;            // 是否计算 overlap(关闭可省一点点时间)
    };

    class Detector
    {
    public:
        /**
             * \brief Empty constructor, initialize with read().
             */
        Detector();

        Detector(std::vector<int> T);
        /**
         * \brief 构造检测器
         * \param num_features  每个模板最多取多少个特征点(越大越准、越慢)。
         *                      注意: 它同时决定了置信度的分母 —— 置信度 = 匹配上的点数 / 总点数。
         *                      实际取不到这么多时会退化成"有多少用多少"(<=4 个点则放弃该模板)
         * \param T             金字塔每层的"位移容差"(像素)。T 越大越能容忍形变/噪声, 但定位越粗;
         *                      每层至少 4, 一般给 {4, 8}; 层数 = T.size()
         * \param weak_thresh   场景图(min_det_contrast)的最小梯度幅值, 低于它的像素不参与匹配。
         *                      调高可以抗噪声, 调低可以在低对比度图上找到目标
         * \param strong_thresh 训练图(min_train_contrast)的最小梯度幅值, 只有大于它的点才会被选为模板特征点
         */
        Detector(int num_features, std::vector<int> T, float weak_thresh = 30.0f, float strong_thresh = 60.0f);

        /**
         * \brief 在场景图上做匹配(完整参数版)
         * \param sources 场景图, 单通道灰度; 尺寸需为 16 的倍数(否则自己补边)
         * \param params  见 MatchParams: 类别 / 置信度 / 重叠度 / NMS / 最大数量 / ICP 开关
         * \return 过滤后的结果, 默认按置信度降序; 每个结果带 confidence / overlap / transform
         */
        std::vector<Match> match(cv::Mat sources, const MatchParams& params);

        /**
         * \brief 简化版, 等价于只设了 min_confidence 和 class_ids 的 MatchParams(不做 NMS、不精修)
         * \param threshold 置信度下限(0~100), 与 MatchParams::min_confidence 同义
         * \param class_ids 只匹配这些类别, 空 = 全部
         * \param masks     场景 mask, 只在该区域内匹配
         */
        std::vector<Match> match(cv::Mat sources, float threshold,
            const std::vector<std::string>& class_ids = std::vector<std::string>(),
            const cv::Mat masks = cv::Mat());

        /**
         * \brief 训练一个模板
         * \param sources     训练图(灰度)
         * \param class_id    类别名, 匹配时可按类别过滤
         * \param object_mask 训练区域 mask(CV_8UC1, 非零处参与), 用来"涂抹掉"不想要的区域。
         *                    传空 = 整幅图都参与。注意内部会先 erode 1 像素, 避免取到区域边界外的梯度
         * \param num_features 覆盖构造时的 num_features; 0 = 用构造时的值
         * \return template_id(该类别内从 0 递增), 失败返回 -1
         */
        int addTemplate(const cv::Mat sources, const std::string& class_id,
            const cv::Mat& object_mask, int num_features = 0);

        int addTemplate_rotate(const std::string& class_id, int zero_id, float theta, cv::Point2f center);

        const cv::Ptr<ColorGradient>& getModalities() const { return modality; }

        int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

        int pyramidLevels() const { return pyramid_levels; }

        const std::vector<Template>& getTemplates(const std::string& class_id, int template_id) const;

        // 训练时用的 mask, 与 template_id 一一对应(空表示未记录, 例如从 yaml 读入的模板)
        const cv::Mat& getTemplateMask(const std::string& class_id, int template_id) const;
        // 从 yaml 读入模板后, 可用它补登记对应的 mask
        void setTemplateMask(const std::string& class_id, int template_id, const cv::Mat& mask);

        /**
         * \brief 计算某个 match 与真值 mask 的重叠度(IoU)
         * \param match        match() 返回的结果
         * \param gt_mask      场景图中目标的真值 mask(CV_8UC1, 非0即前景)
         * \param templ_mask   该 template 对应的训练 mask, 应与当时的训练 src 同尺寸;
         *                     留空则用 Detector 内部缓存的那份(addTemplate 时自动记录)
         * \param use_refine   true: 先跑 icp 精修再算; false: 只用 match 的平移
         * \return OverlapResult, 若失败(无 mask / 尺寸不符)返回全 0
         */
        OverlapResult computeIoU(const Match& match, const cv::Mat& gt_mask,
            const cv::Mat& templ_mask = cv::Mat(), bool use_refine = true);

        int numTemplates() const;
        int numTemplates(const std::string& class_id) const;
        int numClasses() const { return static_cast<int>(class_templates.size()); }

        std::vector<std::string> classIds() const;

        void read(const cv::FileNode& fn);
        void write(cv::FileStorage& fs) const;

        std::string readClass(const cv::FileNode& fn, const std::string& class_id_override = "");
        void writeClass(const std::string& class_id, cv::FileStorage& fs) const;

        void readClasses(const std::vector<std::string>& class_ids,
            const std::string& format = "templates_%s.yml.gz");
        void writeClasses(const std::string& format = "templates_%s.yml.gz") const;

        // 添加 clear_classes 方法(同时清掉训练 mask 缓存)
        void clear_classes() { class_templates.clear(); class_masks.clear(); }

        cv::Mat dx_, dy_; // dx dy recorded for icp

        /**
         * \brief 对单个 match 做 ICP 精修(手动版)
         *
         * ICP 的开关有两种用法:
         *   1. 自动: MatchParams::use_refine = true, match() 内部会对过滤后留下来的结果逐个精修
         *   2. 手动: 关闭 use_refine, 自己挑几个重要的结果再调本函数
         *
         * 返回的 transformation 是 3x3 相似变换(旋转+缩放+平移), 是**增量**矩阵:
         *   它作用在"模板点已经摆到 (match.x, match.y) 之后"的全局坐标上, 即 model -> scene 的修正量。
         *   想直接把模板画到场景上, 用 Match::transform(已合成好平移), 不用自己拼。
         *
         * 注意: 必须在 match() 之后调用, 它依赖 match() 填充的 dx_ / dy_, 否则抛 StsBadArg。
         *       fitness = 内点数 / 模板点数, 即"ICP 意义上的置信度"。
         */
        RegistrationResult refine(const Match& match);

        /// 某个 match 对应的目标矩形(模板包围盒摆到 match.x/y)
        cv::Rect matchRect(const Match& match) const;


    protected:
        cv::Ptr<ColorGradient> modality;
        int pyramid_levels;
        std::vector<int> T_at_level;

        typedef std::vector<Template> TemplatePyramid;
        typedef std::map<std::string, std::vector<TemplatePyramid>> TemplatesMap;
        TemplatesMap class_templates;

        // 与 class_templates 平行的训练 mask 缓存: [class_id][template_id]
        std::map<std::string, std::vector<cv::Mat>> class_masks;

        typedef std::vector<cv::Mat> LinearMemories;
        // Indexed as [pyramid level][ColorGradient][quantized label]
        typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;

        void matchClass(const LinearMemoryPyramid& lm_pyramid,
            const std::vector<cv::Size>& sizes,
            float threshold, std::vector<Match>& matches,
            const std::string& class_id,
            const std::vector<TemplatePyramid>& template_pyramids) const;
    };

} // namespace line2Dup

namespace shape_based_matching {
    class shapeInfo_producer {
    public:
        cv::Mat src;
        cv::Mat mask;

        std::vector<float> angle_range;
        std::vector<float> scale_range;

        float angle_step = 15;
        float scale_step = 0.5;
        float eps = 0.00001f;

        class Info {
        public:
            float angle;
            float scale;

            Info() : angle(0), scale(1) {}  // 添加默认构造函数
            Info(float angle_, float scale_) {
                angle = angle_;
                scale = scale_;
            }
        };
        std::vector<Info> infos;

        // 添加默认构造函数
        shapeInfo_producer() {}

        shapeInfo_producer(cv::Mat src, cv::Mat mask = cv::Mat()) {
            this->src = src;
            if (mask.empty()) {
                // make sure we have masks
                this->mask = cv::Mat(src.size(), CV_8UC1, { 255 });
            }
            else {
                this->mask = mask;
            }
        }

        static cv::Mat transform(cv::Mat src, float angle, float scale) {
            cv::Mat dst;

            cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
            cv::warpAffine(src, dst, rot_mat, src.size());

            return dst;
        }
        static void save_infos(std::vector<shapeInfo_producer::Info>& infos, std::string path = "infos.yaml") {
            cv::FileStorage fs(path, cv::FileStorage::WRITE);

            fs << "infos"
                << "[";
            for (int i = 0; i < infos.size(); i++)
            {
                fs << "{";
                fs << "angle" << infos[i].angle;
                fs << "scale" << infos[i].scale;
                fs << "}";
            }
            fs << "]";
        }
        static std::vector<Info> load_infos(std::string path = "info.yaml") {
            cv::FileStorage fs(path, cv::FileStorage::READ);

            std::vector<Info> infos;

            cv::FileNode infos_fn = fs["infos"];
            cv::FileNodeIterator it = infos_fn.begin(), it_end = infos_fn.end();
            for (int i = 0; it != it_end; ++it, i++)
            {
                infos.emplace_back(float((*it)["angle"]), float((*it)["scale"]));
            }
            return infos;
        }

        void produce_infos() {
            infos.clear();

            assert(angle_range.size() <= 2);
            assert(scale_range.size() <= 2);
            assert(angle_step > eps * 10);
            assert(scale_step > eps * 10);

            // make sure range not empty
            if (angle_range.size() == 0) {
                angle_range.push_back(0);
            }
            if (scale_range.size() == 0) {
                scale_range.push_back(1);
            }

            if (angle_range.size() == 1 && scale_range.size() == 1) {
                float angle = angle_range[0];
                float scale = scale_range[0];
                infos.emplace_back(angle, scale);

            }
            else if (angle_range.size() == 1 && scale_range.size() == 2) {
                assert(scale_range[1] > scale_range[0]);
                float angle = angle_range[0];
                for (float scale = scale_range[0]; scale <= scale_range[1] + eps; scale += scale_step) {
                    infos.emplace_back(angle, scale);
                }
            }
            else if (angle_range.size() == 2 && scale_range.size() == 1) {
                assert(angle_range[1] > angle_range[0]);
                float scale = scale_range[0];
                for (float angle = angle_range[0]; angle <= angle_range[1] + eps; angle += angle_step) {
                    infos.emplace_back(angle, scale);
                }
            }
            else if (angle_range.size() == 2 && scale_range.size() == 2) {
                assert(scale_range[1] > scale_range[0]);
                assert(angle_range[1] > angle_range[0]);
                for (float scale = scale_range[0]; scale <= scale_range[1] + eps; scale += scale_step) {
                    for (float angle = angle_range[0]; angle <= angle_range[1] + eps; angle += angle_step) {
                        infos.emplace_back(angle, scale);
                    }
                }
            }
        }

        cv::Mat src_of(const Info& info) {
            return transform(src, info.angle, info.scale);
        }

        cv::Mat mask_of(const Info& info) {
            return (transform(mask, info.angle, info.scale) > 0);
        }
    };

}

#endif