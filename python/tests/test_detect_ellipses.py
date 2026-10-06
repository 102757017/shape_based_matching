# -*- coding: utf-8 -*-
"""椭圆检测 Python 绑定冒烟测试: shape_based_matching_py.detect_ellipses"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "Release"))

import cv2
import numpy as np
import shape_based_matching_py as sbm

def main():
    print("exports:", [x for x in dir(sbm) if "llipse" in x])

    # ---- 1. 合成图 (真值: 中心 col=320,row=240, a=180,b=90, 25°) ----
    img = cv2.imread("synth_ellipse.bmp", cv2.IMREAD_GRAYSCALE)
    assert img is not None
    ells = sbm.detect_ellipses(img)
    assert len(ells) >= 1, "synthetic ellipse should be detected"
    e = ells[0]
    print("synth:", {k: round(v, 2) for k, v in e.items()})
    assert abs(e["center_col"] - 320) < 5 and abs(e["center_row"] - 240) < 5
    assert abs(e["a"] - 180) < 6 and abs(e["b"] - 90) < 6
    assert abs(e["phi"] - 25) < 3
    assert e["coverangle"] > 350

    # ---- 2. 参数对象 (放宽阈值) ----
    p = sbm.EllipseParams()
    p.min_cover_angle = 150
    p.min_goodness = 0.25
    ells2 = sbm.detect_ellipses(img, p)
    assert len(ells2) >= len(ells)

    # ---- 3. BGR 3通道自动转灰度 ----
    img3 = cv2.imread("synth_ellipse.bmp", cv2.IMREAD_COLOR)
    ells3 = sbm.detect_ellipses(img3)
    assert len(ells3) >= 1

    # ---- 4. 空白图应返回 0 个 ----
    blank = np.full((300, 400), 255, np.uint8)
    assert len(sbm.detect_ellipses(blank)) == 0

    # ---- 5. 真实图 test5 (默认阈值 = 3 个蛋) ----
    t5 = cv2.imread(os.path.join(os.path.dirname(__file__), "..",
                                 "src", "ellipse", "ellipse_detection", "images", "test5.jpg"),
                    cv2.IMREAD_GRAYSCALE)
    ells5 = sbm.detect_ellipses(t5)
    print("test5 count:", len(ells5))
    assert len(ells5) == 3

    print("ALL PYTHON TESTS PASSED")

if __name__ == "__main__":
    main()
