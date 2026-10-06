# -*- coding: utf-8 -*-
"""
诊断: fusion 流水线下 use_refine=True 时 dx_/dy_ 是否真的被填充。
对应 C# 侧 sbm_match(use_refine=1) 的失败现场:
    Gradient maps (dx_, dy_) are empty. Call match() first.

用法: python diag_refine_dxy.py [灰度|彩色|小图|kernel]
"""
import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build", "Release"))
import shape_based_matching_py as sb


def pad16(img):
    ph = (16 - img.shape[0] % 16) % 16
    pw = (16 - img.shape[1] % 16) % 16
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def dxy_info(det):
    dx, dy = np.asarray(det.dx_), np.asarray(det.dy_)
    return (f"dx={dx.shape}/{dx.dtype} 非零={int(np.count_nonzero(dx))}/{dx.size} | "
            f"dy={dy.shape}/{dy.dtype} 非零={int(np.count_nonzero(dy))}/{dy.size}")


def case(name, scene, color_train, use_refine=True):
    print("\n" + "=" * 66)
    print(f"{name}  scene={scene.shape} train_color={color_train} use_refine={use_refine}")
    full = np.full(scene.shape[:2], 255, np.uint8)
    det = sb.Detector(128, [4, 8])
    if color_train:
        det.addTemplate(scene, "A", full)          # 直接喂 3 通道, 走 BGR2GRAY 节点
    else:
        gray = scene if scene.ndim == 2 else cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        det.addTemplate(gray, "A", full)
    print("  模板数:", det.numTemplates())

    p = sb.MatchParams()
    p.min_confidence = 60
    p.use_refine = use_refine
    try:
        ms = det.match(scene, p)
        print(f"  match -> {len(ms)} 个")
        for m in ms[:3]:
            print(f"    pos=({m.x},{m.y}) fit={m.fitness:.4f} angle={m.angle:.2f} "
                  f"scale={m.scale:.4f}")
    except Exception as e:
        print(f"  !! match 抛异常: {type(e).__name__}: {e}")
    print("  " + dxy_info(det))
    if ms:
        try:
            reg = det.refine(ms[0])
            print(f"  手动 refine OK: fitness={reg.fitness:.4f} rmse={reg.inlier_rmse:.4f}")
        except Exception as e:
            print(f"  !! 手动 refine 抛异常: {type(e).__name__}: {e}")


def main(which):
    p = os.path.join(ROOT, "test", "case1", "train.png")
    src = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    src = pad16(src)
    h, w = src.shape
    full = np.full(src.shape, 255, np.uint8)

    OX, OY = 61, 97
    scene = np.zeros((h + 200, w + 200), np.uint8)
    scene = pad16(scene)
    scene[OY:OY + h, OX:OX + w] = src

    if which in ("灰度", "all"):
        case("[1] 灰度 大场景", scene, False)
    if which in ("彩色", "all"):
        case("[2] 彩色 (BGR2GRAY 节点)", cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR), True)
    if which in ("小图", "all"):
        small = scene[:96, :160]                       # 小于一个 tile (32x256)
        case("[3] 小于一个 tile", small, False)
    if which in ("小图", "all"):
        odd = cv2.copyMakeBorder(src, 17, 23, 29, 31, cv2.BORDER_CONSTANT, value=0)
        odd = odd[:, :]
        case("[4] 非 16 倍数 / 非 lcm 倍数边界", odd, False)
    if which in ("kernel", "all"):
        # 反复调用: 先关后开, 检查 dx_ 是否被 4 参 match 清掉
        det = sb.Detector(128, [4, 8])
        det.addTemplate(src, "A", full)
        p0 = sb.MatchParams(); p0.min_confidence = 60; p0.use_refine = False
        det.match(scene, p0)
        print("\n[5] use_refine=False 之后:", dxy_info(det))
        p1 = sb.MatchParams(); p1.min_confidence = 60; p1.use_refine = True
        det.match(scene, p1)
        print("[5] 再 use_refine=True 之后:", dxy_info(det))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
