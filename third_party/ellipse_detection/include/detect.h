/*
 *Copyright: Copyright (c) 2019
 *Created on 2019-5-22
 *Author:zhengaohong@zgheye.cc
 *Version 1.0.1
*/

#ifndef _INCLUDE_DETECT_H_
#define _INCLUDE_DETECT_H_

#include "types.hpp"   // 已间接引入 profile.hpp(DetectStats / ScopedPhase / sbmCounter)

namespace zgh {

bool lineSegmentDetection(const double *image, int row, int col,
                          std::vector<std::shared_ptr<Lined> > &lines);

bool lsdgroups(const double *image, int row, int col, double scale,
               std::vector<std::shared_ptr<Arc> >& arcs);

bool getValidInitialEllipseSet(const uint8_t *image,
                               const double *angles,
                               int row, int col, 
                               std::vector<std::shared_ptr<Ellipse> > &ells,
                               int polarity = 0,
                               int num_threads = 0);
                              

bool generateEllipseCandidates(const uint8_t *image,
                               const double *angles,
                               int row, int col, 
                               std::vector<std::shared_ptr<Ellipse> > &ells,
                               int polarity,
                               int num_threads = 0);


bool detectEllipse(const uint8_t *image, int row, int col,
                   std::vector<std::shared_ptr<Ellipse> > &ells,
                   int polarity = 0, double width = 2.0);

// 可配置检测参数(默认值与原版硬编码行为一致)
struct DetectParams {
  int polarity = 0;                // 椭圆极性: -1/0/1, 0=检测所有极性
  double line_width = 2.0;         // 椭圆线宽(像素)
  double min_cover_angle = 240.0;  // 完整度门槛(角度), 低于此覆盖角的椭圆被丢弃
  double min_goodness = 0.4;       // 质量门槛(最终输出), 越低越容易检出但误检增多
  double candidate_goodness = 0.3; // 候选质量门槛(进入精化阶段前的粗筛)
  int num_threads = 0;             // 并行线程数; 0=自动(用满可用核), 1=关闭多线程
};

// 带参数版本
bool detectEllipse(const uint8_t *image, int row, int col,
                   std::vector<std::shared_ptr<Ellipse> > &ells,
                   const DetectParams &params);

}

//namespace zgh
#endif // _INCLUDE_DETECT_H_
