/*
 * 性能分析脚手架(本仓库新增, 不来自上游)。
 * 只有编译期定义 SBM_ELLIPSE_PROFILE 时才有开销; 默认发布构建里
 * ScopedPhase 是空操作, sbmCounter 返回哑变量, 调用点零成本。
 */
#ifndef _INCLUDE_PROFILE_H_
#define _INCLUDE_PROFILE_H_

#if defined(SBM_ELLIPSE_PROFILE)
#include <chrono>
#include <cstring>
#include <unordered_map>
#endif

namespace zgh {

#if defined(SBM_ELLIPSE_PROFILE)
// 简易具名全局计数器(供 types.hpp 内联函数打点)
long long& sbmCounter(const char* key);
void resetSbmCounters();
#endif

struct PhaseRecord {
  const char* name;
  double begin_ms;
  double end_ms;
  double* dest;          // 非 null = 该阶段是有效阶段, 耗时累加到 dest
};

struct DetectStats {
  double total_ms = 0.0;            // detectEllipse 整体耗时(锚点, 校验各阶段之和)
  double gradient_ms = 0.0;        // 梯度场计算
  double lsd_ms = 0.0;              // LSD 线段提取
  double arc_group_ms = 0.0;        // 线段 -> 弧段分组
  double initial_ellipses_ms = 0.0; // 弧段/弧段对 -> 初始椭圆集
  double clustering_ms = 0.0;       // 中心/phi/长短轴 三级聚类
  double candidate_scan_ms = 0.0;   // 候选椭圆内点扫描 + 打分
  double refine_ms = 0.0;           // subdetect 精化
#if defined(SBM_ELLIPSE_PROFILE)
  std::chrono::steady_clock::time_point total_begin_ns;
#else
  double total_begin_ms_unused_ = 0.0;
#endif
  // 顶层阶段(带 dest 的)绝对时间轴: 用于确认各阶段串行无重叠
  PhaseRecord rec[16];
  int rec_count = 0;
  // 子阶段(不累加到主字段)按名聚合: 只保留前 12 种名字
  char sub_names[12][32];
  double sub_ms[12] = {};
  long long sub_n[12] = {};
  int sub_count = 0;

  long long line_count = 0;         // LSD 线段数
  long long arc_count = 0;          // 弧段数
  long long initial_ellipse_count = 0;  // 初始椭圆数(聚类前)
  long long candidate_count = 0;     // 通过 candidate_goodness 的候选数
  long long inlier_count = 0;        // 所有内点总像素数
  long long iter_pixels = 0;         // EllipseIter 扫描过的像素数

  void reset() {
    total_ms = gradient_ms = lsd_ms = arc_group_ms = initial_ellipses_ms = 0.0;
    clustering_ms = candidate_scan_ms = refine_ms = 0.0;
    rec_count = 0; sub_count = 0;
    std::memset(sub_ms, 0, sizeof(sub_ms));
    std::memset(sub_n, 0, sizeof(sub_n));
    line_count = arc_count = initial_ellipse_count = 0;
    candidate_count = inlier_count = iter_pixels = 0;
  }
};

// 全局统计(每次 detectEllipse 开始时清零)
DetectStats& detectStats();

// 计时作用域
class ScopedPhase {
 public:
  ScopedPhase(const char* name, double* dest);
  ~ScopedPhase();
 private:
  double* dest_;
#if defined(SBM_ELLIPSE_PROFILE)
  std::chrono::steady_clock::time_point t0_;
  int rec_id_;
  int sub_id_;
#endif
};

// 嵌套子阶段: 按名聚合耗时/次数, 不进顶层时间轴
#define SBM_SUBPHASE(var, name) ScopedPhase var(name, nullptr)

void printDetectStats(const DetectStats& s);

}  // namespace zgh
#endif  // _INCLUDE_PROFILE_H_
