// 涂抹制模 demo: 用鼠标在训练图上刷出/擦除区域, 生成 object_mask, 并可选地直接验证匹配效果
//
// 用法:
//   paint_mask_demo.exe <image> [out_mask.png] [--init old_mask.png] [--no-paint] [--train]
//     --init     在已有 mask 上补刷
//     --no-paint 不弹窗, 直接用 --init 的 mask(没给则全选), 便于脚本批量跑
//     --train    用该 mask 训练模板, 在图上匹配并打印 IoU 自测
//
// 窗口操作: 左键涂抹 / 右键擦除 / 滚轮改笔刷 / r 重置 / q 或 Enter 完成 / ESC 放弃

#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>

#include <opencv2/opencv.hpp>

#include "mask_painter.h"
#include "line2Dup.h"

static void usage()
{
    std::cout << "usage: paint_mask_demo <image> [out_mask.png] [--init old_mask.png] [--no-paint] [--train]\n"
              << "  paint: L-button  erase: R-button  wheel: brush size  r: reset  q/Enter: ok  ESC: cancel\n"
              << "  --no-paint: skip the window, use --init mask or full image\n"
              << "  --thresh N : match threshold for --train (default 90)\n";
}

// 库要求输入尺寸为 16 的倍数
static cv::Mat pad16(const cv::Mat& img, int* pad_h = nullptr, int* pad_w = nullptr)
{
    // 注意 C++ 里 (-473) % 16 == -9(负数), 不能直接用 -rows % 16
    int ph = (16 - img.rows % 16) % 16;
    int pw = (16 - img.cols % 16) % 16;
    if (pad_h) *pad_h = ph;
    if (pad_w) *pad_w = pw;
    if (ph == 0 && pw == 0)
        return img;
    cv::Mat dst;
    cv::copyMakeBorder(img, dst, 0, ph, 0, pw, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return dst;
}

int main(int argc, char** argv)
{
    std::string img_path, out_path = "mask.png", init_path;
    bool do_train = false;
    bool no_paint = false;
    float thresh = 90;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--train") do_train = true;
        else if (a == "--no-paint") no_paint = true;
        else if (a == "--init")
        {
            if (i + 1 >= argc) { usage(); return 1; }
            init_path = argv[++i];
        }
        else if (a == "--thresh")
        {
            if (i + 1 >= argc) { usage(); return 1; }
            thresh = static_cast<float>(std::atof(argv[++i]));
        }
        else positional.push_back(a);
    }

    if (positional.empty()) { usage(); return 1; }
    img_path = positional[0];
    if (positional.size() > 1) out_path = positional[1];

    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "cannot read image: " << img_path << "\n";
        return 1;
    }

    std::cout << "image: " << img_path << " " << img.cols << "x" << img.rows << "\n";

    cv::Mat initial;
    if (!init_path.empty())
    {
        initial = cv::imread(init_path, cv::IMREAD_GRAYSCALE);
        if (initial.empty())
        {
            std::cerr << "cannot read init mask: " << init_path << "\n";
            return 1;
        }
        if (initial.size() != img.size())
        {
            std::cerr << "init mask size " << initial.size() << " != image size " << img.size() << "\n";
            return 1;
        }
    }

    cv::Mat mask;
    if (no_paint)
    {
        // 不弹窗: 用 --init 的 mask, 没给就整幅全选
        mask = initial.empty() ? cv::Mat(img.size(), CV_8UC1, cv::Scalar::all(255)) : (initial > 0);
        mask.convertTo(mask, CV_8UC1);
    }
    else
    {
        std::cout << "painting... (see the window)\n";
        mask = mask_painter::paint(img, initial);
    }
    if (mask.empty())
    {
        std::cout << "cancelled, nothing saved\n";
        return 0;
    }

    int fg = cv::countNonZero(mask);
    std::cout << "mask: " << fg << " / " << (mask.rows * mask.cols) << " pixels selected ("
              << (100.0 * fg / (mask.rows * mask.cols)) << "%)\n";

    if (!cv::imwrite(out_path, mask))
    {
        std::cerr << "failed to write " << out_path << "\n";
        return 1;
    }
    std::cout << "saved -> " << out_path << "\n";

    if (!do_train)
        return 0;

    // ---- 用刚刷好的 mask 训练, 并在同一张图上匹配, 自测 IoU ----
    // matchClass 内部要求 8*T 的边界余量, 模板几乎占满整图时会匹配不到,
    // 所以和 test.cpp 一样先留一圈 padding(黑色), 再补到 16 的倍数
    const int padding = 100;
    cv::Mat src_pad, mask_pad;
    cv::copyMakeBorder(img, src_pad, padding, padding, padding, padding,
        cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::copyMakeBorder(mask, mask_pad, padding, padding, padding, padding,
        cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat src = pad16(src_pad);
    cv::Mat train_mask = pad16(mask_pad);

    std::cout << "train size: " << src.cols << "x" << src.rows
              << " (padding " << padding << " + pad to 16)\n";

    line2Dup::Detector det(128, std::vector<int>{4, 8});
    int tid = det.addTemplate(src, "obj", train_mask);
    if (tid < 0)
    {
        std::cerr << "addTemplate failed: too few features, try painting a larger area\n";
        return 1;
    }
    std::cout << "template id: " << tid << "\n";

    std::vector<line2Dup::Match> matches = det.match(src, thresh);
    std::cout << "matches (threshold " << thresh << "): " << matches.size() << "\n";
    if (matches.empty())
    {
        std::cerr << "no match (threshold 80)\n";
        return 1;
    }

    const line2Dup::Match& m = matches[0];
    std::cout << "best match: x=" << m.x << " y=" << m.y
              << " similarity=" << m.similarity << "\n";

    // GT 就是涂抹的 mask 本身(按训练时的摆放), 所以 IoU 应该接近 1
    line2Dup::OverlapResult r = det.computeIoU(m, train_mask);
    std::cout << "self-check IoU = " << r.iou << "  (precision " << r.precision()
              << ", recall " << r.recall() << ")\n";

    return 0;
}
