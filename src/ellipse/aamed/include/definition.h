#pragma once


#define FLED_EDGECONTOURS               0x21
#define FLED_FSAARCCONTOURS             0x22
#define FLED_FSALINES                   0x23
#define FLED_FSAARCSLINKMATRIX          0x24
#define FLED_DPCONTOURS                 0x25

//分组的4种情况，其中F代表距离远，C代表距离近
#define FLED_GROUPING_IBmA1_IAnB1      0x00//初始距离
#define FLED_GROUPING_FBmA1_FAnB1      0x00//搜索点远，尾点远
#define FLED_GROUPING_FBmA1_CAnB1      0x01//搜索点近，尾点远
#define FLED_GROUPING_CBmA1_FAnB1      0x10//搜索点远，尾点近
#define FLED_GROUPING_CBmA1_CAnB1      0x11//搜索点近，尾点近
#define FLED_GROUPING_CAnB1            0x01//搜索点近
#define FLED_GROUPING_CBmA1            0x10//尾点近

// 在邻域分组中，去掉了根据颜色判断是否连接的条件。因为就算去掉了，在后面的全局分组也会重新合并一次进行判断，这次提出的kd树算法可以
// 避免这个问题。


#define FLED_SEARCH_LINKING            true
#define FLED_SEARCH_LINKED             false



// ---------------------功能切换部分-----------------------
// 自适应RDP算法
#define ADAPT_APPROX_CONTOURS 1
// 自适应曲率估计算法
#define DEFINITE_ERROR_BOUNDED 1
// 椭圆验证加速
#define FASTER_ELLIPSE_VALIDATION 0

// 选择椭圆聚类算法
#define NONE_CLUSTER_METHOD   0
#define PRASAD_CLUSTER_METHOD 1
#define OUR_CLUSTER_METHOD    2
// 用 OUR_CLUSTER_METHOD (EllipseNonMaximumSuppression, 基于 IoU 的 NMS),
// 这样 sbm_aamed_params_t.nms_iou 才能端到端生效 (T_iou 由 FLED::SetNMSIoU 注入)。
// 原默认 PRASAD_CLUSTER_METHOD 用几何距离聚类, 没有 IoU 门槛, nms_iou 对其无效。
#define SELECT_CLUSTER_METHOD OUR_CLUSTER_METHOD
// ------------------------------------------------------


// 是否需要统计时间
#define DETAIL_BREAKDOWN
