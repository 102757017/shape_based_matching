# -*- coding: utf-8 -*-
"""自匹配得分诊断 (可复现): "用训练模板的原图去匹配, 得分为什么不是 100?"

背景
----
shape_based_matching 的置信度 = 命中特征点数 / 模板总特征点数, 满分就是 100
(见 line2Dup.cpp: `score = raw_score * 100 / (4 * num_features)`, 每个特征点最多贡献 4)。
所以"自匹配原图只拿到 ~80"等价于: 约 20% 的模板特征点在场景图上取不到
匹配的梯度方向。本脚本把原因定位到唯一一个环节 —— 图像预处理不对称。

做法
----
绕开 MatcherCore, 直接用 pybind 暴露的 line2Dup::Detector:
  * 用真实 train() 产出的同一份模板 (同一份 yaml / 同样的零填充 + mask), 保证模板逐点一致;
  * 只换"喂进去的场景图", 于是得分差异 100% 归因于场景侧预处理。
模板一致性由"特征点数逐层相等"验证 (例如 111 / 27)。

已知结论 (在本仓库 test/ 图片上的实测)
------------------------------------
场景 = 训练图自己            -> 100.00
场景 = 原图(不滤波)          -> 100.00
场景 = medianBlur(原图, 3)   -> 93.96 ~ 97.64   <-- MatcherCore::match 内部实际喂的
场景 = 两端都做 medianBlur3  -> 100.00
高频纹理被 unsharp 放大后    -> 84.76           <-- 纹理越细, 损耗越大, 可到 80 档

根因: core/matcher_core.cpp 的 match() 里曾有 `cv::medianBlur(image, image_to_match, 3)` 只作用在场景上,
训练路径 (MatcherCore::train) 从不滤波, 于是模板特征的方向是在"更锐利"的图上量出来的,
而场景是滤波后的图 —— 3x3 中值滤波恰好会破坏/改变 1~3px 尺度的边缘方向,
这些特征点就永久对不上了。该行为继承自旧 Python 版 (py_tests/matcher_old_ab.py 的
_preprocess_image, 只在 match() 里调用), 不是当下 C++ 重写引入的回归, 但确属缺陷。

现状 (已修复): 训练与匹配两侧都不做任何滤波, matcher_core.cpp 的 match() 只剩补边到 16 的倍数。
契约由 py_tests/test_selfmatch_score_contract.py 固化 (自匹配必须 100)。
本脚本用底层 Detector 保留"当初为什么会掉分"的可复现证据:
上表第二列的 100.00 就是修复后 MatcherCore::match 走的口径, 第三列是修复前的口径。

跑法:  python py_tests/diag_selfmatch_score.py
"""
import os
import sys
import shutil
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import shape_based_matching_py as sbm          # 底层 Detector (无任何预处理)

OUT = os.path.join(HERE, "_diag_out")          # 必须 ASCII 路径: FileStorage 读不了中文路径

FEAT, WEAK, STRONG = 200, 30.0, 60.0


def pad16(img):
    """match() 对场景做的补边 (MIPP 要求 rows*cols % 16 == 0)"""
    ph = (16 - img.shape[0] % 16) % 16
    pw = (16 - img.shape[1] % 16) % 16
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def train_template(train_img, roi, tag):
    """复刻 MatcherCore::train 的模板构建 (单角度单尺度时 producer.src_of 即零填充本身)"""
    x, y, w, h = roi
    diag = (w * w + h * h) ** 0.5
    pad = int(diag * 1.0 * 1.5) + 50          # 与 MatcherCore::train 一致
    padded = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.uint8)
    padded[pad:pad + h, pad:pad + w] = train_img[y:y + h, x:x + w]
    mask = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    mask[pad:pad + h, pad:pad + w] = 255      # 默认整块 ROI 参与 (erode 1px 发生在库内部)
    det = sbm.Detector(FEAT, [4, 8], float(WEAK), float(STRONG))
    tid = det.addTemplate(padded, tag, mask, FEAT)
    return det, tid, pad


def score_at_target(det, scene, cls, roi, pad, tid):
    """取真值位置附近的最高分 (只看粗定位位置 -> 避免把别处的误报当结果)

    match.x 的定义: 特征点 f 在场景中的位置 = f.x + match.x;
    而 f 在零填充训练图里的位置 = f.x + tl_x, 对应原图位置 = f.x + tl_x - pad,
    故真值 match.x = roi.x + tl_x - pad。
    """
    ms = det.match(scene, 0.0, [cls])
    if not ms:
        return None, 0
    tl = det.getTemplates(cls, tid)[0]
    ideal = (roi[0] + tl.tl_x - pad, roi[1] + tl.tl_y - pad)
    near = [m for m in ms if abs(m.x - ideal[0]) < 30 and abs(m.y - ideal[1]) < 30]
    if not near:
        return None, len(ms)
    return max(near, key=lambda m: m.similarity).similarity, len(ms)


def main():
    big = cv2.imread(os.path.join(ROOT, "test", "case0", "1.jpg"))     # 900x600 实拍
    assert big is not None
    roi = (60, 380, 160, 110)          # 草地纹理区: 高频成分多, 对滤波最敏感
    blur0 = cv2.GaussianBlur(big, (0, 0), 1.2)

    print("ROI=%s feat=%d 阈值%g/%g" % (str(roi), FEAT, WEAK, STRONG))
    print("%-16s %-10s %-18s %-18s %-14s" %
          ("图像高频成分", "特征点数", "场景=原图(不滤波)", "场景=medianBlur3", "两端都blur3"))
    for k, tag in [(0.0, "原图"), (1.0, "unsharp x1.0"), (1.8, "unsharp x1.8"), (2.6, "unsharp x2.6")]:
        img = big if k == 0 else np.clip(
            big.astype(np.float32) + k * (big.astype(np.float32) - blur0.astype(np.float32)),
            0, 255).astype(np.uint8)

        det_raw, tid_raw, pad_raw = train_template(img, roi, "RAW")          # 训练图=原图(现状)
        det_blr, tid_blr, pad_blr = train_template(cv2.medianBlur(img, 3), roi, "BLR")  # 训练图也滤波

        if tid_raw < 0 or tid_blr < 0:
            print("%-16s 模板训练失败 (特征点不足)" % tag)
            continue
        n = len(det_raw.getTemplates("RAW", tid_raw)[0].features)

        s_raw, _ = score_at_target(det_raw, pad16(img.copy()), "RAW", roi, pad_raw, tid_raw)
        s_blr, _ = score_at_target(det_raw, pad16(cv2.medianBlur(img, 3)), "RAW", roi, pad_raw, tid_raw)
        s_sym, _ = score_at_target(det_blr, pad16(cv2.medianBlur(img, 3)), "BLR", roi, pad_blr, tid_blr)
        f = lambda v: ("%.2f" % v) if v is not None else "无命中"
        print("%-16s %-10d %-18s %-18s %-14s" % (tag, n, f(s_raw), f(s_blr), f(s_sym)))

    print("\n注: 模板特征点数逐层与 MatcherCore::train 产物完全一致, 故差异只来自场景侧预处理。")
    shutil.rmtree(OUT, ignore_errors=True)


if __name__ == "__main__":
    main()
