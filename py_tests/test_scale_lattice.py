# -*- coding: utf-8 -*-
"""尺度格点回归: 训练尺度集合必须包含 1.0, 且能自匹配原图。

背景 (bug):
    shapeInfo_producer::produce_infos 旧实现从 scale_range[0] 起按 scale_step 单向铺格点
    (循环条件 `<= hi+eps`), 于是:
      - 步长不整除时上端点被悄悄丢弃;
      - 更要命的是 1.0 常常不在格点上 —— 例如 0.90~1.10 / step 0.15 只生成 {0.90, 1.05}。
    训练图里的目标恰好处于 1.0 尺度, 最近模板却偏 5%~10%, 特征点整体径向外漂,
    导致"在训练模版的原图上都匹配不到任何对象"。

本测试固化修复后的契约:
  A. 0.90~1.10 / 0.15 -> {0.90, 1.00, 1.05, 1.10}   (注入 1.0 + 上端点)
  B. 0.95~1.05 / 0.10 -> {0.95, 1.00, 1.05}          (格点本就不含 1.0)
  C. 0.90~1.10 / 0.10 -> {0.90, 1.00, 1.10}          (回归: 原本就正确的配置不变)
  D. 0.80~0.95 / 0.10 -> {0.80, 0.90, 0.95}          (区间不含 1.0, 不得硬塞)
  E. 自匹配: 用 A 的配置训练, 把训练图原样贴进场景, threshold=90 必须命中,
     且命中模板 scale == 1.0、位置/角度落在真实目标上。
"""
import os
import sys
import json
import shutil
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from matcher import Matcher

SAVE_DIR = os.path.join(HERE, "输出_尺度回归")

TRAIN_PARAMS_BASE = {
    "pyramid_levels": [4, 8], "feature_num": 100,
    "weak_thresh": 30.0, "strong_thresh": 60.0,
    "angle_start": 0.0, "angle_extent": 0.0, "angle_step": 10.0,
}


def pad16(img):
    ph = (16 - img.shape[0] % 16) % 16
    pw = (16 - img.shape[1] % 16) % 16
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def train_and_get_scales(m, train_img, w, h, class_id, lo, hi, step):
    params = dict(TRAIN_PARAMS_BASE)
    params.update(scale_start=lo, scale_end=hi, scale_step=step)
    res = m.train(train_img, (0, 0, w, h), class_id, params, SAVE_DIR)
    with open(res["info_path"], "r", encoding="utf-8") as f:
        info = json.load(f)
    scales = sorted({round(float(t["scale"]), 6) for t in info["templates"].values()})
    return res, info, scales


def train_and_get_angles(m, train_img, w, h, class_id, start, extent, step):
    params = dict(TRAIN_PARAMS_BASE)
    params.update(angle_start=start, angle_extent=extent, angle_step=step,
                  scale_start=1.0, scale_end=1.0, scale_step=0.1)
    res = m.train(train_img, (0, 0, w, h), class_id, params, SAVE_DIR)
    with open(res["info_path"], "r", encoding="utf-8") as f:
        info = json.load(f)
    angles = sorted({round(float(t["angle"]), 6) for t in info["templates"].values()})
    return res, info, angles


def check(label, got, want):
    ok = len(got) == len(want) and all(abs(a - b) < 1e-6 for a, b in zip(got, want))
    print("  %-26s -> %-34s %s" % (label, got, "OK" if ok else "FAIL  期望 %s" % want))
    assert ok, "%s: 得到 %s, 期望 %s" % (label, got, want)


def main():
    train_img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"))
    assert train_img is not None, "找不到 test/case1/train.png"
    th, tw = train_img.shape[:2]
    print("训练图: %dx%d" % (tw, th))

    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    m = Matcher()

    # ---------- A/B/C/D: 尺度集合契约 ----------
    print("[A] 0.90~1.10 / step 0.15  (本次 bug 的原始配置)")
    res_a, _, sc_a = train_and_get_scales(m, train_img, tw, th, "尺A", 0.9, 1.1, 0.15)
    check("A", sc_a, [0.9, 1.0, 1.05, 1.1])
    assert 1.0 in sc_a, "A 必须包含 1.0"
    assert res_a["template_count"] == 4, "A 应生成 4 个模板, 实得 %s" % res_a["template_count"]

    print("[B] 0.95~1.05 / step 0.10")
    _, _, sc_b = train_and_get_scales(m, train_img, tw, th, "尺B", 0.95, 1.05, 0.10)
    check("B", sc_b, [0.95, 1.0, 1.05])

    print("[C] 0.90~1.10 / step 0.10  (回归: 修复前本来就对)")
    _, _, sc_c = train_and_get_scales(m, train_img, tw, th, "尺C", 0.9, 1.1, 0.10)
    check("C", sc_c, [0.9, 1.0, 1.1])

    print("[D] 0.80~0.95 / step 0.10  (区间不含 1.0, 不应硬塞)")
    _, _, sc_d = train_and_get_scales(m, train_img, tw, th, "尺D", 0.8, 0.95, 0.10)
    check("D", sc_d, [0.8, 0.9, 0.95])
    assert 1.0 not in sc_d, "D 的区间不含 1.0, 不该出现 1.0"

    # ---------- E: 角度格点: 必须含恒等档 0° ----------
    # 同一类 bug: -45~45 / step 2 单方向铺出来全是奇数, 没有 0 (最近的是 ±1)。
    # ROI 一大, 1 度偏差就让两端特征点漂出金字塔容差 T(4,8): C# 夹具 (1375px ROI,
    # -45~45/step2) 实测自匹配只有 81.15 分, 补上 0 后 100.00。
    print("[E] 角度格点契约")
    res_e, info_e, ang_e = train_and_get_angles(m, train_img, tw, th, "角E", -45.0, 90.0, 2.0)
    assert 0.0 in ang_e, "E: -45~45/step2 必须包含恒等档 0°, 实得 %s" % ang_e
    assert len(ang_e) == 47 and ang_e[0] == -45.0 and ang_e[-1] == 45.0, \
        "E: 期望 47 档 (-45..45 奇数档 + 0), 实得 %d 档 %s" % (len(ang_e), ang_e)
    print("  %-26s -> 共 %d 档, 含 0°=%s  OK" % ("E -45~45 / step 2", len(ang_e), 0.0 in ang_e))

    _, _, ang_f = train_and_get_angles(m, train_img, tw, th, "角F", -44.0, 88.0, 2.0)
    assert 0.0 in ang_f and len(ang_f) == 45, "F: -44~44/step2 本就含 0, 应仍为 45 档, 实得 %s" % len(ang_f)
    print("  %-26s -> 共 %d 档, 含 0°=True  OK" % ("F -44~44 / step 2", len(ang_f)))

    _, _, ang_g = train_and_get_angles(m, train_img, tw, th, "角G", 10.0, 50.0, 5.0)
    assert 0.0 not in ang_g, "G: 区间 10~60 不含 0, 不得硬塞: %s" % ang_g
    assert ang_g == [10.0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], ang_g
    print("  %-26s -> %s (不含 0°, 未硬塞)  OK" % ("G 10~60 / step 5", ang_g))

    _, _, ang_h = train_and_get_angles(m, train_img, tw, th, "角H", 350.0, 50.0, 2.0)
    assert 360.0 in ang_h, "H: 350~400 的恒等档是 360°, 必须在内: %s" % ang_h
    print("  %-26s -> 含 360°(等价不旋转)  OK" % "H 350~400 / step 2")

    # ---------- I: 大 ROI + 奇偶性错开的角度格点 -> 自匹配必须 100 ----------
    # 判别性用例: 补 0° 之前这里是 ~90 分档 (1° 偏差让 382px 半对角上的特征点漂 6.7px > T=4)
    print("[I] 大 ROI 自匹配: -45~45/step2 (缺 0 时两端特征点漂出容差)")
    m3 = Matcher()
    m3.add_template_class(res_e["info_path"])
    scene = pad16(np.zeros((th + 300, tw + 300, 3), np.uint8))
    OX, OY = 101, 97
    scene[OY:OY + th, OX:OX + tw] = train_img
    results = m3.match(scene, 99.0, use_refine=False)
    assert results, "I: 补上 0° 后 threshold=99 自匹配必须命中"
    r = results[0]
    print("  %-26s -> 得分 %.2f (角度 %.1f°)  OK" % ("I 自匹配", r["score"], r["angle"]))
    assert r["score"] >= 99.99, "I: 大 ROI 自匹配应满 100, 实得 %.2f" % r["score"]
    assert abs(r["angle"]) < 1e-6, "I: 应命中 angle=0 的那个模板, 实得 %s" % r["angle"]

    # ---------- E: 自匹配原图 ----------
    print("[E] 自匹配: 用 A 的模板跑训练图自身 (threshold=90)")
    m2 = Matcher()
    m2.add_template_class(res_a["info_path"])
    scene = pad16(np.zeros((th + 300, tw + 300, 3), np.uint8))
    OX, OY = 101, 97
    scene[OY:OY + th, OX:OX + tw] = train_img

    results = m2.match(scene, 90.0, use_refine=True)
    print("  命中数: %d" % len(results))
    assert results, "修复后: threshold=90 自匹配必须至少命中 1 个目标"
    r = results[0]
    print("  top: score=%.2f scale=%.4f angle=%.3f refined=(%.1f, %.1f) fitness=%.3f"
          % (r["score"], r["scale"], r["refined_angle"], r["refined_x"], r["refined_y"], r["fitness"]))
    assert abs(r["scale"] - 1.0) < 1e-6, "自匹配应命中 scale=1.0 的模板, 实得 %s" % r["scale"]
    assert abs(r["refined_x"] - (OX + tw / 2)) < 15 and abs(r["refined_y"] - (OY + th / 2)) < 15, \
        "自匹配中心偏差过大: (%.1f, %.1f)" % (r["refined_x"], r["refined_y"])
    assert abs(r["refined_angle"]) < 8, "自匹配角度偏差过大: %s" % r["refined_angle"]

    # 模板数 = 4 个尺度 x 1 个角度
    with open(res_a["info_path"], "r", encoding="utf-8") as f:
        info_a = json.load(f)
    assert len(info_a["templates"]) == 4, "A 应生成 4 个模板, 实得 %d" % len(info_a["templates"])

    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    print("\n全部通过")


if __name__ == "__main__":
    main()
