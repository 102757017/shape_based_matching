/*
 *Copyright: Copyright (c) 2019
 *Created on 2019-5-21
 *Author:zhengaohong@zgheye.cc
 *Version 1.0.1
*/


#include <vector>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <algorithm>
#include <map>
#include <cfloat>

#include "defines.h"
#include "detect.h"
#include "unitily.h"
#include "compute.h"
#include "cvcannyapi.h"
#include "parallel.hpp"

#if defined(SBM_ELLIPSE_PROFILE)
#include <chrono>
#endif


namespace zgh {

// 计时脚手架(DetectStats / ScopedPhase / sbmCounter)见 profile.hpp

static DetectStats g_stats;

// 解析请求的线程数: 0/负数 -> 自动(用满硬件并发数); >=1 -> 取请求值。
// 开启 SBM_ELLIPSE_PROFILE 时各阶段累加器没有加锁, 强制单线程,
// 避免多线程累加互相踩踏(profile 本来只用于单线程调参)。
static int resolveThreads(int requested) {
#if defined(SBM_ELLIPSE_PROFILE)
  (void)requested;
  return 1;
#else
  return ThreadPool::resolveThreads(requested);
#endif
}

DetectStats& detectStats() {
  return g_stats;
}

long long& sbmCounter(const char* key) {
#if defined(SBM_ELLIPSE_PROFILE)
  static std::unordered_map<const char*, long long> ctr;
  auto it = ctr.find(key);
  if (it == ctr.end()) {
    it = ctr.emplace(key, 0LL).first;
  }
  return it->second;
#else
  static long long dummy = 0;
  return dummy;
#endif
}

void resetSbmCounters() {
#if defined(SBM_ELLIPSE_PROFILE)
  static std::unordered_map<const char*, long long> ctr;
  for (auto &kv : ctr) {
    kv.second = 0;
  }
#endif
}

#if defined(SBM_ELLIPSE_PROFILE)
namespace {
inline double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t).count();
}
}  // namespace

// 顶层阶段(dest != null): 记绝对时间轴 + 累加到 dest
// 子阶段(dest == null):  按名聚合耗时与次数, 不进时间轴
ScopedPhase::ScopedPhase(const char* name, double* dest)
    : dest_(dest), t0_(std::chrono::steady_clock::now()) {
  rec_id_ = -1;
  if (dest_) {
    if (g_stats.rec_count < (int)(sizeof(g_stats.rec) / sizeof(g_stats.rec[0]))) {
      rec_id_ = g_stats.rec_count++;
      g_stats.rec[rec_id_].name = name;
      g_stats.rec[rec_id_].dest = dest_;
      g_stats.rec[rec_id_].begin_ms = ms_since(g_stats.total_begin_ns);
    }
  } else {
    sub_id_ = -1;
    for (int i = 0; i < g_stats.sub_count; ++i) {
      if (std::strcmp(g_stats.sub_names[i], name) == 0) { sub_id_ = i; break; }
    }
    if (sub_id_ < 0 && g_stats.sub_count < 12) {
      sub_id_ = g_stats.sub_count++;
      std::strncpy(g_stats.sub_names[sub_id_], name, 31);
      g_stats.sub_names[sub_id_][31] = '\0';
    }
  }
}

ScopedPhase::~ScopedPhase() {
  const double dur = ms_since(t0_);
  if (dest_) {
    *dest_ += dur;
  }
  if (rec_id_ >= 0) {
    g_stats.rec[rec_id_].end_ms = ms_since(g_stats.total_begin_ns);
  }
  if (sub_id_ >= 0) {
    g_stats.sub_ms[sub_id_] += dur;
    ++g_stats.sub_n[sub_id_];
  }
}
#else
ScopedPhase::ScopedPhase(const char*, double*) {}
ScopedPhase::~ScopedPhase() {}
#endif

void printDetectStats(const DetectStats& s) {
  // 绝对时间轴: 确认各阶段串行无重叠
  double prev_end = 0.0;
  std::printf("  --- absolute timeline (ms from detectEllipse start) ---\n");
  for (int i = 0; i < s.rec_count; ++i) {
    const PhaseRecord &r = s.rec[i];
    std::printf("    %-16s %8.2f -> %8.2f  (dur %6.2f, gap %5.2f)\n",
                r.name, r.begin_ms, r.end_ms, r.end_ms - r.begin_ms,
                r.begin_ms - prev_end);
    prev_end = r.end_ms;
  }

#if defined(SBM_ELLIPSE_PROFILE)
  {
    const char* keys[] = {"distFast", "push_outer", "push_endcap", "calcYAxis", "stack_push"};
    for (const char* k : keys) {
      std::printf("    ctr:%-16s %lld\n", k, sbmCounter(k));
    }
  }
#endif

  for (int i = 0; i < s.sub_count; ++i) {
    std::printf("    sub:%-16s %8.2f  (n=%lld, avg %.3f)\n", s.sub_names[i],
                s.sub_ms[i], s.sub_n[i],
                s.sub_n[i] ? s.sub_ms[i] / (double)s.sub_n[i] : 0.0);
  }

  const double phase_sum = s.gradient_ms + s.lsd_ms + s.arc_group_ms +
      s.initial_ellipses_ms + s.clustering_ms + s.candidate_scan_ms +
      s.refine_ms;
  std::printf(
      "  --- phase timing (ms, cumulative over all runs) ---\n"
      "    [total anchor]    %8.2f\n"
      "    gradient           %8.2f   (%.1f%%)\n"
      "    lsd                %8.2f   (%.1f%%)\n"
      "    arc grouping       %8.2f   (%.1f%%)\n"
      "    initial ellipses   %8.2f   (%.1f%%)\n"
      "    clustering         %8.2f   (%.1f%%)\n"
      "    candidate scan     %8.2f   (%.1f%%)\n"
      "    refine             %8.2f   (%.1f%%)\n"
      "    sum(non-nested)    %8.2f   (%.1f%%)\n"
      "  --- counts ---\n"
      "    lines=%lld arcs=%lld initial_ellipses=%lld candidates=%lld "
      "inliers=%lld iter_pixels=%lld\n",
      s.total_ms,
      s.gradient_ms, 100.0 * s.gradient_ms / s.total_ms,
      s.lsd_ms, 100.0 * s.lsd_ms / s.total_ms,
      s.arc_group_ms, 100.0 * s.arc_group_ms / s.total_ms,
      s.initial_ellipses_ms, 100.0 * s.initial_ellipses_ms / s.total_ms,
      s.clustering_ms, 100.0 * s.clustering_ms / s.total_ms,
      s.candidate_scan_ms, 100.0 * s.candidate_scan_ms / s.total_ms,
      s.refine_ms, 100.0 * s.refine_ms / s.total_ms,
      phase_sum, 100.0 * phase_sum / s.total_ms,
      s.line_count, s.arc_count, s.initial_ellipse_count, s.candidate_count,
      s.inlier_count, s.iter_pixels);
}

bool lsdgroups(const double *image, int row, int col, double scale, std::vector<std::shared_ptr<Arc> >& arcs) {
#if defined(SBM_ELLIPSE_PROFILE)
  auto t_lsd0 = std::chrono::steady_clock::now();
  auto ms_since = [](const std::chrono::steady_clock::time_point &t) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t).count();
  };
#endif
  std::vector<std::shared_ptr<Lined> > lines;
  lineSegmentDetection(image, row, col, lines);
#if defined(SBM_ELLIPSE_PROFILE)
  g_stats.lsd_ms += ms_since(t_lsd0);
  g_stats.line_count = (long long)lines.size();
  auto t_arc0 = std::chrono::steady_clock::now();
#endif

  int *pixlabel = new int[row * col];
  bool *used = new bool[(int)lines.size()];
  memset(used, 0, lines.size() * sizeof(bool));
  memset(pixlabel, -1, sizeof(int) * row * col);
  for (size_t idx = 0; idx < lines.size(); ++idx) {
    for (auto &pix : lines[idx]->getregs()) {
      pixlabel[pix.x * col + pix.y] = idx;
    }
  }

  for (size_t begin_line_id = 0; begin_line_id < lines.size(); ++begin_line_id) {
    if (!used[begin_line_id]) {
      // 4 X 4 邻域进行搜索
      int xsize = 4, ysize = 4;
      // 向后搜索能组合成弧的线段
      std::vector<std::shared_ptr<Lined> > group_back;
      group_back.push_back(lines[begin_line_id]);
      size_t current_id = begin_line_id;
      bool isFound = true;
      while (isFound) {
        used[current_id] = true;
        auto updir = lines[current_id]->dir;
        auto downdir = updir;
        if (lines[current_id]->polarity == SAME_POL) {
          downdir.rotation(PI_4);
        } else {
          downdir.rotation(-PI_4);
        }
        std::unordered_map<int, int> vote;
        int max_vote = 0, max_idx = -1;

        for (int idx = 1; idx <= xsize; ++idx) {
          for (int idy = 1; idy <= ysize; ++idy) {
            int x = (int)(lines[current_id]->ep.x + updir.x * idx + downdir.x * idy);
            int y = (int)(lines[current_id]->ep.y + updir.y * idx + downdir.y * idy);

            if (inRect(row, col, x, y) && pixlabel[x * col + y] != -1) {
              int line_id = pixlabel[x * col + y];
              pixlabel[x * col + y] = -1;
              int cnt = ++vote[line_id];
              if (max_idx == -1) {
                max_idx = line_id;
                max_vote = 1;
              } else if (cnt > max_vote){
                max_idx = line_id;
                max_vote = cnt;
              }
            }
          }
        }

        if (max_vote >= 5 && !used[max_idx] && lines[begin_line_id]->polarity == lines[max_idx]->polarity) {
          double start_angle = atan2(lines[current_id]->dir.y, lines[current_id]->dir.x);
          double end_angle = atan2(lines[max_idx]->dir.y, lines[max_idx]->dir.x);
          double angle_delta = rotateAngle(start_angle, end_angle, lines[begin_line_id]->polarity);;
          if (angle_delta <= 3.0 * PI_8) {
            group_back.push_back(lines[max_idx]);
            current_id = max_idx;
          } else {
            isFound = false;
          }
        } else {
          isFound = false;
        }
      }

      // 向前搜索能组合成弧的线段

      std::vector<std::shared_ptr<Lined> > group_front;
      group_front.push_back(lines[begin_line_id]);
      current_id = begin_line_id;
      isFound = true;
      while (isFound) {
        used[current_id] = true;
        auto updir = Pointd(0, 0) - lines[current_id]->dir;
        auto downdir = updir;
        if (lines[current_id]->polarity == SAME_POL) {
          downdir.rotation(-PI_4);
        } else {
          downdir.rotation(PI_4);
        }


        std::unordered_map<int, int> vote;
        int max_vote = 0, max_idx = -1;

        for (int idx = 1; idx <= xsize; ++idx) {
          for (int idy = 1; idy <= ysize; ++idy) {
            int x = (int)(lines[current_id]->sp.x + updir.x * idx + downdir.x * idy);
            int y = (int)(lines[current_id]->sp.y + updir.y * idx + downdir.y * idy);

            if (inRect(row, col, x, y) && pixlabel[x * col + y] != -1) {
              int line_id = pixlabel[x * col + y];
              pixlabel[x * col + y] = -1;
              int cnt = ++vote[line_id];
              if (max_idx == -1) {
                max_idx = line_id;
                max_vote = 1;
              } else if (cnt > max_vote){
                max_idx = line_id;
                max_vote = cnt;
              }
            }
          }
        }
        if (max_vote >= 5 && !used[max_idx] && lines[begin_line_id]->polarity == lines[max_idx]->polarity) {
          double start_angle = atan2(lines[current_id]->dir.y, lines[current_id]->dir.x);
          double end_angle = atan2(lines[max_idx]->dir.y, lines[max_idx]->dir.x);
          // 注意此时需要调换一下，因为是从尾部开始搜索
          double angle_delta = rotateAngle(end_angle, start_angle, lines[begin_line_id]->polarity);
          if (angle_delta <= 3.0 * PI_8) {
            group_front.push_back(lines[max_idx]);
            current_id = max_idx;
          } else {
            isFound = false;
          }
        } else {
            isFound = false;
        }
      }

      std::shared_ptr<Arc> arc(nullptr);
      for (int idx = (int)group_front.size() - 1; idx >= 0; --idx) {
        group_front[idx]->sp /= scale;
        group_front[idx]->ep /= scale;
        group_front[idx]->length /= scale;
        if (!arc) {
          arc = std::make_shared<Arc>(std::move(group_front[idx]));
        } else {
          arc->merge(std::move(group_front[idx]));
        }
      }

      for (size_t idx = 1; idx < group_back.size(); ++idx) {
        group_back[idx]->sp /= scale;
        group_back[idx]->ep /= scale;
        group_back[idx]->length /= scale;
        arc->merge(std::move(group_back[idx]));
      }
      arcs.push_back(arc);
    }
  }

  delete [] pixlabel;
  delete [] used;
#if defined(SBM_ELLIPSE_PROFILE)
  g_stats.arc_group_ms += ms_since(t_arc0);
  g_stats.arc_count = (long long)arcs.size();
#endif
  return true;
}

bool getValidInitialEllipseSet(const uint8_t *image,
                               const double *angles,
                               int row, int col,
                               std::vector<std::shared_ptr<Ellipse> > &ells,
                               int polarity,
                               int num_threads) {

  double scale = 0.8;
  double sigma_scale = 0.6;
  int scale_row = (int)ceil(1.0 * row * scale);
  int scale_col = (int)ceil(1.0 * col * scale);
  double *scale_data = new double[scale_row * scale_col];
  gaussianSampler(image, row, col, scale_data, scale_row, scale_col, scale, sigma_scale,
                  num_threads);
  std::vector<std::shared_ptr<Arc> > arcs;

  ScopedPhase _ph_init("initial_ellipses", &g_stats.initial_ellipses_ms);
  lsdgroups(scale_data, scale_row, scale_col, scale, arcs);
  delete [] scale_data;

  int groupsNum = (int)arcs.size();

  for (int id = 0; id < groupsNum; ++id) {
    if (polarity == 0 || arcs[id]->polarity == polarity) {
      if (arcs[id]->coverages >= 4.0 * PI / 9.0) {
        SBM_SUBPHASE(_ps_singlefit, "fit(arc_single)");
        auto ell = calcElliseParam(arcs[id], nullptr, angles, row, col);
        if (ell) {
          ells.push_back(std::move(ell));
        }
      }
    }
  }

  // 弧段两两拟合。每个 (i, j) 组合只读 arcs/angles -> 相互独立, 可并行。
  // 结果先落到 per_i[i], 再按 i 升序合并, 保证与串行的 ells 顺序完全一致
  // (下游聚类依赖该顺序, 顺序变了结果就会漂)。
  const int nt = resolveThreads(num_threads);
  std::vector<std::vector<std::shared_ptr<Ellipse> > > per_i(
      groupsNum > 0 ? groupsNum : 1);
  ThreadPool::instance().parallelFor(
      groupsNum - 1, [&](int i) {
    auto &out_i = per_i[i];
    for (int j = i + 1; j < groupsNum; ++j) {
      if ((arcs[i]->polarity == polarity || polarity == NONE_POL) &&
          (arcs[j]->polarity == polarity || polarity == NONE_POL) &&
          (int)arcs[i]->lines.size() + (int)arcs[j]->lines.size() >= 3 &&
          arcs[i]->coverages + arcs[j]->coverages >= 3.0 * PI / 9.0
          ) {
#if defined(SBM_ELLIPSE_PROFILE)
        SBM_SUBPHASE(_ps_region, "region_limitation");
#endif
        if (regionLimitation(arcs[i], arcs[j])) {
#if defined(SBM_ELLIPSE_PROFILE)
          SBM_SUBPHASE(_ps_pairfit, "fit(arc_pair)");
#endif
          auto ell = calcElliseParam(arcs[i], arcs[j], angles, row, col);
          if (ell) {
            out_i.push_back(std::move(ell));
          }
        }
      }
    }
  }, nt);
  for (int i = 0; i < groupsNum; ++i) {
    for (auto &e : per_i[i]) {
      ells.push_back(std::move(e));
    }
  }
  return true;
}



static bool meanShift(const std::vector<double> &data, std::vector<double> &init_data, int dims,
                      double sigma, double windos_size, double accuracy_tolerance, int iter_times) {

  double temparr[8];
  int nquerrues = (int)init_data.size() / dims;
  int data_num = (int)data.size() / dims;
  double sigma2 = sigma * sigma;
  double radius2 = windos_size * windos_size;
  double tolerance = accuracy_tolerance;
  int maxiters = iter_times;

  std::vector<double> dis(data_num);
  for (int loop = 0; loop < nquerrues; ++loop) {
    int iters = 0;
    while (iters < maxiters) {
      bool flag = false;
      double denominator = 0.0;
      for (int i = 0; i < data_num; ++i) {
        double temp = 0;
        for (int d = 0; d < dims; ++d) {
          temp += sqr(data[dims * i + d] - init_data[loop * dims + d]);
        }
        dis[i] = temp;
        if (dis[i] <= radius2) {
          flag = true;
          denominator += exp(-dis[i] / sigma2);
        }
      }
      if (!flag) {
        break;
      }
      for (int d = 0; d < dims; ++d) {
        temparr[d] = init_data[loop * dims + d];
        init_data[loop * dims + d] = 0;
      }
      for (int i = 0; i < data_num; ++i) {
        if (dis[i] <= radius2) {
          for (int d = 0; d < dims; ++d) {
            init_data[loop * dims + d] += exp(-dis[i] / sigma2) * data[i * dims + d];
          }
        }
      }
      double temp = 0;
      for (int d = 0; d < dims; ++d) {
        init_data[loop * dims + d] /= denominator;
        temp += sqr(init_data[loop * dims + d] - temparr[d]);
      }

      if (sqrt(temp) < tolerance) {
        break;
      }
      ++iters;
    }
  }
  return true;
}

static bool clusterByDistance(std::vector<double> &data, int dims, double distance_threshold,
                              double number_control) {

  double threshold2 = distance_threshold * distance_threshold;
  int npoints = (int)data.size() / dims;
  if (npoints == 1) {
    return true;
  }
  int nout = 0;
  std::vector<double> data_out;
  std::vector<double> counts;
  std::vector<bool> labeled(npoints, false);
  for (int idx = 0; idx < npoints; ++idx) {
    if (!labeled[idx]) {
      ++nout;
      labeled[idx] = true;
      for (int d = 0; d < dims; ++d) {
        data_out.push_back(data[idx * dims + d]);
      }
      counts.push_back(1);
      for (int idy = idx + 1; idy < npoints; ++idy) {
        if (!labeled[idy]) {
          double dis = 0;
          for (int d = 0; d < dims; ++d) {
            dis += sqr(data_out[(nout - 1) * dims + d] / counts[nout - 1] - data[idy * dims + d]);
          }
          if (dis <= threshold2) {
            ++counts[nout - 1];
            labeled[idy] = true;
            for (int d = 0; d < dims; ++d) {
              data_out[(nout - 1) * dims + d] += data[idy * dims + d];
            }
            if (counts[nout - 1] >= number_control) {
              // 聚类数量控制，防止均值中心漂的太远  圆心聚类时 20  半径聚类时 10
              break;
            }
          }
        }
      }
    }
  }
  for (int id = 0; id < nout; ++id) {
    for (int d = 0; d < dims; ++d) {
      data_out[id * dims + d] /= counts[id];
    }
  }
  data.clear();
  data = std::move(data_out);
  return true;
}

static bool cluster2DPoints(const std::vector<std::shared_ptr<Ellipse> > &ells,
                            std::vector<Pointd> &cluster_center,
                            double distance_tolerance,
                            int data_type) {

  int nbinx, nbiny;
  double xmax, ymax, xmin, ymin;
  xmax = ymax = 0;
  xmin = ymin = DBL_MAX;
  std::vector<double> data;
  int npoints = (int)ells.size();
  for (auto &ell : ells) {
    if (data_type == 0) {
      data.push_back(ell->o.x);
      data.push_back(ell->o.y);
      xmax = max(xmax, ell->o.x);
      xmin = min(xmin, ell->o.x);
      ymax = max(ymax, ell->o.y);
      ymin = min(ymin, ell->o.y);
    } else {
      data.push_back(ell->a);
      data.push_back(ell->b);
      xmax = max(xmax, ell->a);
      xmin = min(xmin, ell->a);
      ymax = max(ymax, ell->b);
      ymin = min(ymin, ell->b);
    }
  }
  xmax += xmax * 0.02;
  xmin -= xmin * 0.02;
  ymax += ymax * 0.02;
  ymin -= ymin * 0.02;
  double xdelta = xmax - xmin;
  double ydelta = ymax - ymin;
  nbinx = (int)ceil(xdelta / distance_tolerance);
  nbiny = (int)ceil(ydelta / distance_tolerance);
  if (nbinx <= 0) {
    nbinx = 1;
  }
  if (nbiny <= 0) {
    nbiny = 1;
  }
  std::map<Pointi, std::pair<Pointd, int> > bindata;
  for (int id = 0; id < npoints; ++id) {
    int x = (int)floor((data[id * 2] - xmin) / xdelta * nbinx + 0.5);
    int y = (int)floor((data[id * 2 + 1] - ymin) / ydelta * nbiny + 0.5);
    if (x >= nbinx) {
      x = nbinx - 1;
    }
    if (y >= nbiny - 1) {
      y = nbiny - 1;
    }
    Pointi cell(x, y);
    if (!bindata.count(cell)) {
      bindata.insert(std::make_pair(cell, std::make_pair(Pointd(data[id * 2], data[id * 2 + 1]), 1)));
    } else {
      bindata[cell].first += Pointd(data[id * 2], data[id * 2 + 1]);
      ++bindata[cell].second;
    }
  }
  std::vector<double> init_data;
  for (auto &bin : bindata) {
    init_data.push_back(bin.second.first.x / bin.second.second);
    init_data.push_back(bin.second.first.y / bin.second.second);
  }

  meanShift(data, init_data, 2, 1, distance_tolerance, 1e-6, 50);

  clusterByDistance(init_data, 2, distance_tolerance / 2, 40);

  for (int id = 0; id < (int)init_data.size(); id += 2) {
    cluster_center.push_back(Pointd(init_data[id], init_data[id + 1]));
  }
  return true;
}

static bool cluster1DDatas(const std::vector<std::shared_ptr<Ellipse> > &ells,
                           std::vector<double> &cluster_center,
                           double distance_tolerance) {

  double val_max = 0;
  double val_min = DBL_MAX;
  std::vector<double> data;
  for (auto &ell : ells) {
    val_max = max(val_max, ell->phi);
    val_min = min(val_min, ell->phi);
    data.push_back(ell->phi);
  }
  val_max += val_min * 0.02; // avoid rmax - rmin = 0
  val_min -= val_min * 0.02;

  double val_delta = val_max - val_min;
  int nbins = (int)ceil(val_delta / distance_tolerance);
  if (nbins <= 0) {
    nbins = 1;
  }
  // first sum, second vote;
  std::vector<std::pair<double, int> > bindata(nbins, std::make_pair(0.0, 0));
  for (auto &ell : ells) {
    int r = (int)floor((ell->phi - val_min) / val_delta * nbins + 0.5);
    if (r >= nbins) {
      r = nbins - 1;
    }
    bindata[r].first += ell->phi;
    ++bindata[r].second;
  }
  auto pend = std::remove_if(bindata.begin(), bindata.end(), [](std::pair<double, int> &data) {
    if (data.second == 0) {
      return true;
    }
    return false;
  });
  cluster_center.clear();
  for (auto iter = bindata.begin(); iter != pend; ++iter) {
    cluster_center.push_back(iter->first / iter->second);
  }
  bindata.clear();


  // 均值漂移
  meanShift(data, cluster_center, 1, 1, distance_tolerance, 1e-6, 20);

  // 按照距离阈值聚类
  clusterByDistance(cluster_center, 1, distance_tolerance / 2, 40);

  return true;
}

bool generateEllipseCandidates(const uint8_t *image, const double *angles,
                               int row, int col,
                               std::vector<std::shared_ptr<Ellipse> > &ells, int polarity,
                               int num_threads) {

  std::vector<std::shared_ptr<Ellipse> > ells_init;

  getValidInitialEllipseSet(image, angles, row, col, ells_init, polarity, num_threads);

  int init_size = (int)ells_init.size();
  if (init_size == 0) {
    return true;
  }


#if defined(SBM_ELLIPSE_PROFILE)
  g_stats.initial_ellipse_count = (long long)ells_init.size();
#endif

  // 最外层椭圆中心聚类　第二层椭圆 phi　聚类　第三层椭圆长短轴聚类

  ScopedPhase _ph_clu("clustering", &g_stats.clustering_ms);
  std::vector<Pointd> cluster_center;
  cluster2DPoints(ells_init, cluster_center, MIN_ELLIPSE_THRESHOLD_LENGTH, 0);

  int center_num = (int)cluster_center.size();

  std::vector<std::vector<std::shared_ptr<Ellipse> > > ells_center(center_num);

  // TODO(using KD tree optimization if necessary)
  for (auto &ell : ells_init) {
    double dis_min = DBL_MAX;
    int idx = -1;
    for (int centerid = 0; centerid < center_num; ++centerid) {
      double temp_dis = (ell->o - cluster_center[centerid]).length();
      if (temp_dis < dis_min) {
        dis_min = temp_dis;
        idx = centerid;
      }
    }
    ells_center[idx].push_back(ell);
  }

  for (int center_id = 0; center_id < center_num; ++center_id)  {
    auto &ells_c = ells_center[center_id];
    if ((int)ells_c.size() == 0) {
      continue;
    }

    //  phi　聚类
    std::vector<double> cluster_phi;
    cluster1DDatas(ells_c, cluster_phi, 0.0873);
    int phi_num = (int)cluster_phi.size();
    int ells_cnum = (int)ells_c.size();

    std::sort(cluster_phi.begin(), cluster_phi.end());
    std::sort(ells_c.begin(), ells_c.end(),
      [](const std::shared_ptr<Ellipse> &ea, const std::shared_ptr<Ellipse> &eb) {
        return ea->phi < eb->phi;
      });
    int p = 0;
    for (int phi_id = 0; phi_id < phi_num; ++phi_id) {
      std::vector<std::shared_ptr<Ellipse> > ells_p;
      while (p < ells_cnum && (phi_id == phi_num - 1 ||
             fabs(ells_c[p]->phi - cluster_phi[phi_id])
             < fabs(ells_c[p]->phi - cluster_phi[phi_id + 1]))) {
        ells_p.push_back(ells_c[p]);
        ++p;
      }

      // 椭圆长短轴聚类
      int ells_pnum = (int)ells_p.size();
      if (ells_pnum == 0) {
        continue;
      }

      std::vector<Pointd> cluster_axis;
      cluster2DPoints(ells_p, cluster_axis, MIN_ELLIPSE_THRESHOLD_LENGTH, 1);
      for (auto &p : cluster_axis) {
        ells.push_back(std::make_shared<Ellipse>(cluster_center[center_id], p.x, p.y, cluster_phi[phi_id]));
      }
    }
  }
  return true;
}


static int improveInliers(std::vector<Pixel> &inliers, Pointd ell_center, int tbins) {

  // 基于联通性分析提升内点质量
  double tmin = -PI, tmax = PI;
  std::vector<int> votes(tbins);
  std::vector<int> tids;
  for (auto &pix : inliers) {
    double theta = atan2(1.0 * pix.y - ell_center.y, 1.0 * pix.x - ell_center.x);
    double tid = (int)floor((theta - tmin) / (tmax - tmin) * tbins + 0.5);
    if (tid >= tbins) {
      tid = tbins - 1;
    }
    tids.push_back(tid);
    ++votes[tid];
  }

  // bfs
  std::vector<int> labels(tbins, -1);
  std::vector<int> comp_lengths;
  int npoints = (int)inliers.size();
  int ncomp = 0;
  int head = 0, tail = tbins - 1;
  int lens = 0;
  if (votes[0]) {
    while (head <= tail && votes[head]) {
      labels[head] = ncomp;
      ++head;
      ++lens;
    }
    while (head <= tail && votes[tail]) {
      labels[tail] = ncomp;
      --tail;
      ++lens;
    }
    comp_lengths.push_back(lens);
    ++ncomp;
  }

  for (int id = head; id <= tail; ++id) {
    if (votes[id]) {
      lens = 0;
      while (id <= tail && votes[id]) {
        labels[id] = ncomp;
        ++lens;
        ++id;
      }
      comp_lengths.push_back(lens);
      ++ncomp;
    }
  }
  votes.clear();

  int max_len = 0;
  for (auto &len : comp_lengths) {
    max_len = max(max_len, len);
  }
  std::vector<Pixel> inliers_out;
  for (int id = 0; id < npoints; ++id) {
    int &label = labels[tids[id]];
    if (label != -1 && 10 * comp_lengths[label] >= max_len && comp_lengths[label] > 10) {
      inliers_out.push_back(inliers[id]);
    } else {
      label = -1;
    }
  }
  inliers.clear();
  inliers = std::move(inliers_out);
  int bin_count = 0;
  for (int id = 0; id < tbins; ++id) {
    if (labels[id] != -1) {
      ++bin_count;
    }
  }
  return bin_count;
}

static bool subdetect(double *angles, int row, int col, double min_cover_angle,
                      double min_goodness,
                      double distance_tolerance, double normal_tolerance, double tr,
                      std::vector<std::shared_ptr<Ellipse> > &ells) {

#if defined(SBM_ELLIPSE_PROFILE)
  g_stats.candidate_count = (long long)ells.size();
#endif

  std::vector<std::shared_ptr<Ellipse> > ells_out;
  for (int id = 0; id < (int)ells.size(); ++id) {
    auto &ell = ells[id];
    bool issame = false;
    for (int idcompare = 0; idcompare < id; ++idcompare) {
      if (ell->equal(ells[idcompare])) {
        issame = true;
        break;
      }
    }
    if (issame) {
      continue;
    }

    double beta = PI * (1.5 * (ell->a + ell->b) - sqrt(ell->a * ell->b));
    int tbins = min(180, (int)floor(beta * tr));
    std::vector<Pixel> inliers_t;

    for (auto &pix : ell->inliers) {
      int addr = pix.x * col + pix.y;
      if (!equal(angles[addr], ANGLE_NOT_DEF)) {
        inliers_t.push_back(pix);
      }
    }
    ell->inliers.clear();
    ell->inliers = std::move(inliers_t);
    std::shared_ptr<Ellipse> ell_t;
    {
      SBM_SUBPHASE(_ps_fit1, "fit(inliers)");
      ell_t = fitEllipse(ell->inliers);
    }
    if (ell_t) {
      ell_t->polarity = ell->polarity;
      if (ell_t->equal(ell, 3 * distance_tolerance, 0.1, 0.1, 0.1, 0.9)) {

        // 重新计算内点
        beta = PI * (1.5 * (ell_t->a + ell_t->b) - sqrt(ell_t->a * ell_t->b));
        tbins = min(180, (int)floor(beta * tr));

        std::vector<Pixel> new_inliers;
        for (EllipseIter iter(ell_t, distance_tolerance); !iter.isEnd(); ++iter) {
          if (inRect(row, col, iter.np.x, iter.np.y)) {
            int addr = iter.np.x * col + iter.np.y;
            if (!equal(angles[addr], ANGLE_NOT_DEF)) {
              Vectord tanline = ell_t->getTangent(iter.np);
              Vectord grad_normal(sin(angles[addr]), cos(angles[addr]));
              if (tanline.dot(grad_normal) >= normal_tolerance &&
                  (ell->polarity == NONE_POL || ell->polarity == SAME_POL)) {
                new_inliers.push_back(iter.np);
              }
              if (tanline.dot(grad_normal) <= -normal_tolerance &&
                  (ell->polarity == NONE_POL || ell->polarity == OPP_POL)) {
                new_inliers.push_back(iter.np);
              }
            }
          }
        }
        int bin_count = 0;
        {
          SBM_SUBPHASE(_ps_improve2, "ell_improve2");
          bin_count = improveInliers(new_inliers, ell_t->o, tbins);
        }
        if ((int)new_inliers.size() - (int)ell->inliers.size() >= -10) {
          std::shared_ptr<Ellipse> ell_tt;
          {
            SBM_SUBPHASE(_ps_fit2, "fit(new_inliers)");
            ell_tt = fitEllipse(new_inliers);
          }
          if (ell_tt) {
            ell_tt->polarity = ell_t->polarity;
            ell_tt->inliers = std::move(new_inliers);
            double support_inliers_ratio = 1.0 * ell_tt->inliers.size() / distance_tolerance / beta;
            double completeness_ratio = 1.0 * bin_count / tbins;
            ell_tt->coverangle = completeness_ratio * 360.0;
            ell_tt->goodness = sqrt(support_inliers_ratio * completeness_ratio);
            ell_t = ell_tt;
          } else {
            ell_t->inliers = std::move(ell->inliers);
            ell_t->goodness = ell->goodness;
            ell_t->coverangle = ell->coverangle;
          }
        } else {
          ell_t->inliers = std::move(ell->inliers);
          ell_t->goodness = ell->goodness;
          ell_t->coverangle = ell->coverangle;
        }
      } else {
        ell_t->inliers = std::move(ell->inliers);
        ell_t->goodness = ell->goodness;
        ell_t->coverangle = ell->coverangle;
      }
    } else {
      ell_t = ell;
    }


    if ((double)ell_t->inliers.size() / distance_tolerance >= beta * tr / 2.0) {
      bool isComplete = (ell_t->coverangle >= min_cover_angle && ell_t->goodness >= min_goodness);
      if (isComplete) {
        bool isUnique = true;
        for (auto &new_ell : ells_out) {
          if (new_ell->equal(ell, distance_tolerance)) {
            isUnique = false;
            break;
          }
        }
        if (isUnique) {
          ells_out.push_back(ell_t);
          for (auto &pix : ell_t->inliers) {
            int addr = pix.x * col + pix.y;
            angles[addr] = ANGLE_NOT_DEF;
          }
        }
      }
    }
  }

  ells.clear();
  ells = std::move(ells_out);
  return true;
}

bool detectEllipse(const uint8_t *image, int row, int col,
                   std::vector<std::shared_ptr<Ellipse> > &ells, int polarity, double width) {
  DetectParams params;
  params.polarity = polarity;
  params.line_width = width;
  return detectEllipse(image, row, col, ells, params);
}

bool detectEllipse(const uint8_t *image, int row, int col,
                   std::vector<std::shared_ptr<Ellipse> > &ells, const DetectParams &params) {

#if defined(SBM_ELLIPSE_PROFILE)
  g_stats.reset();
  resetSbmCounters();
  g_stats.total_begin_ns = std::chrono::steady_clock::now();
#endif

  // calc the gradient
  double *angles = new double[row * col];

#if defined(SBM_ELLIPSE_PROFILE)
  {
    ScopedPhase _ph_grad("gradient", &g_stats.gradient_ms);
#endif
    calculateGradient3(image, row, col, angles, params.num_threads);
#if defined(SBM_ELLIPSE_PROFILE)
  }
#endif

  generateEllipseCandidates(image, angles, row, col, ells, params.polarity,
                            params.num_threads);



  ScopedPhase _ph_scan("candidate_scan", &g_stats.candidate_scan_ms);
  // control parameter
  double tr = 0.6;
  double distance_tolerance = MIN_ELLIPSE_THRESHOLD_LENGTH;
  double normal_tolerance = cos(PI / 12.0);

  // 每个候选只是读 angles 并把结果写进自己这个 Ellipse 对象 -> 相互独立。
  // 结果按"候选原索引"回填 kept[], 最后按索引升序取出, 顺序与串行完全一致
  // (下游 std::sort 不是稳定排序, 顺序一变结果就可能漂)。
  const int nt = resolveThreads(params.num_threads);
  const int n_ells = (int)ells.size();
  std::vector<std::shared_ptr<Ellipse> > kept(n_ells, nullptr);

  ThreadPool::instance().parallelFor(
      n_ells, [&](int idx) {
    auto &ell = ells[idx];
    // 这几个容器原来在循环外复用; 并行下必须每个候选一份
    std::vector<Pixel> inliers_positive, inliers_negative, inliers_all;
    double beta = PI * (1.5 * (ell->a + ell->b) - sqrt(ell->a * ell->b));
    int tbins = min(180, (int)floor(beta * tr));

#if defined(SBM_ELLIPSE_PROFILE)
    SBM_SUBPHASE(_ps_iter, "ell_iter_scan");
#endif
    for (EllipseIter iter(ell, distance_tolerance); !iter.isEnd(); ++iter) {
#if defined(SBM_ELLIPSE_PROFILE)
      ++g_stats.iter_pixels;
      SBM_SUBPHASE(_ps_body, "iter_body");
#endif
      if (inRect(row, col, iter.np.x, iter.np.y)) {
        int addr = iter.np.x * col + iter.np.y;
        if (!equal(angles[addr], ANGLE_NOT_DEF)) {
          Vectord tanline = ell->getTangent(iter.np);
          Vectord grad_normal(sin(angles[addr]), cos(angles[addr]));
          double dv = tanline.dot(grad_normal);
          if (dv >= normal_tolerance) {
            inliers_positive.push_back(iter.np);
            inliers_all.push_back(iter.np);
          } else if (dv <= -normal_tolerance){
            inliers_negative.push_back(iter.np);
            inliers_all.push_back(iter.np);
          }
        }
      }
    }

    if ((int)inliers_positive.size() >= 10 * (int)inliers_negative.size()) {
      ell->polarity = SAME_POL;
      ell->inliers.clear();
      ell->inliers = std::move(inliers_positive);
    } else if ((int)inliers_negative.size() >= 10 * (int)inliers_positive.size()) {
      ell->polarity = OPP_POL;
      ell->inliers.clear();
      ell->inliers = std::move(inliers_negative);
    } else {
      ell->polarity = NONE_POL;
      ell->inliers.clear();
      ell->inliers = std::move(inliers_all);
    }
    // 原来是 continue; lambda 体内没有可 continue 的外层循环
    if (params.polarity != NONE_POL && ell->polarity != params.polarity) {
      return;
    }

    int bin_count = 0;
    {
#if defined(SBM_ELLIPSE_PROFILE)
      SBM_SUBPHASE(_ps_improve, "ell_improve");
#endif
      bin_count = improveInliers(ell->inliers, ell->o, tbins);
    }
    double support_inliers_ratio = 1.0 * ell->inliers.size() / distance_tolerance / beta;
    double completeness_ratio = 1.0 * bin_count / tbins;
    ell->coverangle = completeness_ratio * 360.0;
    ell->goodness = sqrt(support_inliers_ratio * completeness_ratio);
#if defined(SBM_ELLIPSE_PROFILE)
    g_stats.inlier_count += (long long)ell->inliers.size();
#endif
    if (ell->goodness >= params.candidate_goodness) {
      kept[idx] = ell;
    }
  }, nt);

  std::vector<std::shared_ptr<Ellipse> > ells_temp;
  for (int idx = 0; idx < n_ells; ++idx) {
    if (kept[idx]) {
      ells_temp.push_back(kept[idx]);
    }
  }
  ells.clear();
  ells = std::move(ells_temp);
  std::sort(ells.begin(), ells.end(),
    [](const std::shared_ptr<Ellipse> &ea, const std::shared_ptr<Ellipse> &eb) {
      return ea->goodness > eb->goodness;
    });



  ScopedPhase _ph_ref("refine", &g_stats.refine_ms);
  subdetect(angles, row, col, params.min_cover_angle, params.min_goodness,
            params.line_width, normal_tolerance, tr, ells);



  delete [] angles;
#if defined(SBM_ELLIPSE_PROFILE)
  g_stats.total_ms += ms_since(g_stats.total_begin_ns);
#endif
  return true;
}

} // namespace zgh
