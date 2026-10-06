# -*- coding: utf-8 -*-
"""
复现: "先做一次不过 ICP 的匹配(use_refine=false, 或走 4 参 match), 之后再精修" 的场景。

对应 C# 侧两种真实调用顺序:
  a) sbm_match(use_refine=0) 之后某处又调 refine / 需要 ICP 精修的结果;
  b) Detector::match(img, threshold) 简化版(match(use_refine=false)) 之后再 computeIoU(use_refine=True)。
"""
import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "build", "Release"))
import shape_based_matching_py as sb

p = os.path.join(ROOT, "test", "case1", "train.png")
src = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
ph, pw = (-src.shape[0]) % 16, (-src.shape[1]) % 16
src = cv2.copyMakeBorder(src, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
h, w = src.shape

full = np.full(src.shape, 255, np.uint8)
OX, OY = 61, 97
scene = np.zeros((h + 200, w + 200), np.uint8)
scene = cv2.copyMakeBorder(scene, 0, (-scene.shape[0]) % 16, 0, (-scene.shape[1]) % 16,
                           cv2.BORDER_CONSTANT, value=0)
scene[OY:OY + h, OX:OX + w] = src

det = sb.Detector(128, [4, 8])
det.addTemplate(src, "A", full)

print("A. 简化版 4 参 match(img, threshold)  -> 再 refine()")
m4 = det.match(scene, 60)
print(f"   匹配到 {len(m4)} 个, dx_ 尺寸 = {np.asarray(det.dx_).shape}")
if m4:
    try:
        det.refine(m4[0])
        print("   refine OK")
    except Exception as e:
        print(f"   !! refine 抛异常: {type(e).__name__}: {e}")

print("\nB. MatchParams(use_refine=False) -> 再 refine()")
p0 = sb.MatchParams(); p0.min_confidence = 60; p0.use_refine = False
ms = det.match(scene, p0)
print(f"   匹配到 {len(ms)} 个, dx_ 尺寸 = {np.asarray(det.dx_).shape}")
if ms:
    try:
        det.refine(ms[0])
        print("   refine OK")
    except Exception as e:
        print(f"   !! refine 抛异常: {type(e).__name__}: {e}")
    try:
        det.computeIoU(ms[0], np.full(scene.shape, 255, np.uint8))
        print("   computeIoU(use_refine=True) OK")
    except Exception as e:
        print(f"   !! computeIoU 抛异常: {type(e).__name__}: {e}")

print("\nC. 同一个 detector 上 '先 False 再 True'")
p1 = sb.MatchParams(); p1.min_confidence = 60; p1.use_refine = True
try:
    ms2 = det.match(scene, p1)
    print(f"   匹配到 {len(ms2)} 个, refine: ", end="")
    det.refine(ms2[0])
    print("OK")
except Exception as e:
    print(f"!! {type(e).__name__}: {e}")
