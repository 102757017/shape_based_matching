#ifndef MASK_PAINTER_H
#define MASK_PAINTER_H

// 交互式"涂抹"工具: 用鼠标在训练图上刷出/擦除区域, 生成 shape_based_matching 需要的 object_mask
//
// 用法:
//   #include "mask_painter.h"
//   cv::Mat img = cv::imread("templ.png");
//   cv::Mat mask = mask_painter::paint(img);          // 从空白开始刷
//   cv::Mat mask2 = mask_painter::paint(img, old_mask); // 在已有 mask 上补刷
//   int id = detector.addTemplate(img, "cls", mask);
//
// 操作:
//   左键拖动   涂抹(加入前景)
//   右键拖动   擦除(排除该区域)
//   滚轮       调整笔刷半径
//   r          清空重来
//   q / Enter / 空格  完成并退出
//   ESC        放弃, 返回空 mask

#include <opencv2/opencv.hpp>
#include <string>

namespace mask_painter
{
    namespace detail
    {
        struct PainterState
        {
            cv::Mat base;      // 原始图(BGR), 用于绘制叠加显示
            cv::Mat view;      // 显示图
            cv::Mat mask;      // CV_8UC1 结果
            int radius = 15;
            int mode = 0;      // 0 无, 1 涂抹, 2 擦除
            cv::Point last;
            bool cancelled = false;

            void refresh()
            {
                view = base.clone();
                // 前景叠半透明绿色
                cv::Mat green(view.size(), view.type(), cv::Scalar(0, 0, 0));
                green.setTo(cv::Scalar(40, 180, 40), mask > 0);
                cv::addWeighted(view, 1.0, green, 0.35, 0, view);
                // 笔刷光标
                if (last.x >= 0)
                    cv::circle(view, last, radius, cv::Scalar(0, 255, 255), 1);
                cv::putText(view, "L:paint  R:erase  wheel:size  r:reset  q/Enter:ok  ESC:cancel",
                    cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
                cv::putText(view, "L:paint  R:erase  wheel:size  r:reset  q/Enter:ok  ESC:cancel",
                    cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        };

        inline void onMouse(int event, int x, int y, int flags, void* userdata)
        {
            PainterState* s = static_cast<PainterState*>(userdata);
            int value = (s->mode == 2) ? 0 : 255;

            switch (event)
            {
            case cv::EVENT_LBUTTONDOWN:
            case cv::EVENT_RBUTTONDOWN:
                s->mode = (event == cv::EVENT_RBUTTONDOWN) ? 2 : 1;
                s->last = cv::Point(x, y);
                cv::circle(s->mask, s->last, s->radius, cv::Scalar(value), -1);
                s->refresh();
                cv::imshow("paint mask", s->view);
                break;
            case cv::EVENT_MOUSEMOVE:
                s->last = cv::Point(x, y);
                if (s->mode != 0)
                {
                    // 用线段连接, 避免快速拖动时断点
                    cv::line(s->mask, s->last, cv::Point(x, y), cv::Scalar(value), s->radius * 2);
                    cv::circle(s->mask, s->last, s->radius, cv::Scalar(value), -1);
                }
                s->refresh();
                cv::imshow("paint mask", s->view);
                break;
            case cv::EVENT_LBUTTONUP:
            case cv::EVENT_RBUTTONUP:
                s->mode = 0;
                break;
            case cv::EVENT_MOUSEWHEEL:
                s->radius += (cv::getMouseWheelDelta(flags) > 0) ? 2 : -2;
                s->radius = std::max(1, std::min(s->radius, 300));
                s->refresh();
                cv::imshow("paint mask", s->view);
                break;
            default:
                break;
            }
        }
    }

    // img: 训练图(灰度或 BGR 均可); initial: 可选初始 mask
    // 返回 CV_8UC1 mask, 非零处参与模板特征选取
    inline cv::Mat paint(const cv::Mat& img, const cv::Mat& initial = cv::Mat(),
        const std::string& win_name = "paint mask")
    {
        CV_Assert(!img.empty());

        detail::PainterState s;
        if (img.channels() == 1)
            cv::cvtColor(img, s.base, cv::COLOR_GRAY2BGR);
        else
            s.base = img.clone();

        if (initial.empty())
            s.mask = cv::Mat::zeros(img.size(), CV_8UC1);
        else
        {
            CV_Assert(initial.size() == img.size());
            cv::Mat bin = initial > 0;      // MatExpr -> Mat, 不能直接 .clone()
            bin.convertTo(s.mask, CV_8UC1);
        }
        s.last = cv::Point(-1, -1);
        s.refresh();

        cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(win_name, detail::onMouse, &s);
        cv::imshow(win_name, s.view);

        while (true)
        {
            int key = cv::waitKey(50);
            if (key == 'r' || key == 'R')
            {
                s.mask = cv::Mat::zeros(img.size(), CV_8UC1);
                s.refresh();
                cv::imshow(win_name, s.view);
            }
            else if (key == 27)  // ESC
            {
                s.cancelled = true;
                break;
            }
            else if (key == 'q' || key == 'Q' || key == 13 || key == 32)
            {
                break;
            }
            // OpenCV 窗口被关闭(点 X)也退出; 窗口已销毁时 getWindowProperty 会抛异常
            bool visible = true;
            try { visible = cv::getWindowProperty(win_name, cv::WND_PROP_VISIBLE) >= 1; }
            catch (const cv::Exception&) { visible = false; }
            if (!visible)
                break;
        }
        try
        {
            cv::setMouseCallback(win_name, nullptr, nullptr);
            cv::destroyWindow(win_name);
        }
        catch (const cv::Exception&) {}

        if (s.cancelled)
            return cv::Mat();
        return s.mask;
    }

    // 非交互式: 直接从多边形/矩形生成 mask, 适合脚本批量制作
    inline cv::Mat fromPolygons(cv::Size size,
        const std::vector<std::vector<cv::Point>>& polygons)
    {
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
        if (!polygons.empty())
            cv::fillPoly(mask, polygons, cv::Scalar(255));
        return mask;
    }

    inline cv::Mat fromRect(cv::Size size, cv::Rect roi)
    {
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
        cv::Rect safe = roi & cv::Rect(0, 0, size.width, size.height);
        if (safe.area() > 0)
            mask(safe) = 255;
        return mask;
    }
}

#endif
