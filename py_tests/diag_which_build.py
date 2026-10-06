# -*- coding: utf-8 -*-
"""一键自检: 打印"你实际加载的是哪个 build / 模板 / 得分", 用来定位 UI 与命令行结论不一致。

用法 (请用**你启动 ui.py 的那个 python**) :
    <你的python> py_tests/diag_which_build.py

参数可改: TEMPLATE_INFO / TEST_IMAGE / ROI (模板训练时的 ROI, 用于定位真值位置)
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TEMPLATE_INFO = r"C:\Users\Administrator\Documents\模版\template_model.info.json"
TEST_IMAGE = r"E:\programing\pathon\vision-inspect\app\setting\measure\sample\temple.jpg"

# ui.py 的默认匹配参数
UI_MIN_SCORE = 80.0
UI_NMS, UI_NMS_THR = True, 0.5
UI_REFINE, UI_MIN_FITNESS, UI_MAX_OVERLAP, UI_MAX_MATCHES = True, 0.0, 1.0, 20


def imread_u(p):
    import cv2
    import numpy as np
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def pad16(img):
    import cv2
    ph = (16 - img.shape[0] % 16) % 16
    pw = (16 - img.shape[1] % 16) % 16
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def main():
    print("=" * 78)
    print("python    :", sys.version.split()[0], sys.executable)
    try:
        import shape_based_matching_py as sbm
        p = getattr(sbm, "__file__", "?")
        print("pyd       :", p)
        if os.path.isfile(p):
            print("pyd 时间  :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))))
            # realpath 里带 shape_based_matching_py/ 说明是 site-packages 的旧包
            print("pyd 大小  :", os.path.getsize(p), "字节")
        print("PyMatcher :", hasattr(sbm, "PyMatcher"),
              "(False = 这是旧版仅 Detector 的包, 不是本仓库的 C++ 内核)")
    except Exception as e:
        print("导入 shape_based_matching_py 失败:", e)
        return
    print("=" * 78)

    from matcher import Matcher
    print("matcher.py:", sys.modules["matcher"].__file__)

    big = imread_u(TEST_IMAGE)
    print("测试图    :", TEST_IMAGE, big.shape)
    info = json.load(open(TEMPLATE_INFO, encoding="utf-8"))
    print("模板      : feature_num=%s pyramid=%s weak=%s strong=%s padding=%s ROI=%dx%d"
          % (info["training_params"]["feature_num"], info["training_params"]["pyramid_levels"],
             info["training_params"]["weak_thresh"], info["training_params"]["strong_thresh"],
             info["padding"], info["original_w"], info["original_h"]))

    scene = pad16(big)
    m = Matcher()
    cid, params = m.add_template_class(TEMPLATE_INFO)
    print("加载类别  :", cid, "  已加载:", m.get_loaded_class_ids())
    print("-" * 78)

    for tag, masks in [("masks=None (UI 没画匹配ROI)", None),
                       ("masks=整幅图", "full")]:
        mk = None
        if masks == "full":
            import numpy as np
            mk = np.full(scene.shape[:2], 255, np.uint8)
        out = m.match(scene, UI_MIN_SCORE, None, UI_NMS, UI_NMS_THR,
                      max_matches=UI_MAX_MATCHES, min_fitness=UI_MIN_FITNESS,
                      use_refine=UI_REFINE, max_overlap=UI_MAX_OVERLAP, masks=mk)
        print("%s -> 命中 %d 条" % (tag, len(out)))
        for i, r in enumerate(out[:5]):
            print("    #%d 得分=%.2f 中心=(%.1f,%.1f) 角度=%.1f tid=%d 内点率=%.3f 重叠=%.3f"
                  % (i + 1, r["score"], r["refined_x"], r["refined_y"], r["refined_angle"],
                     r["template_id"], r["fitness"], r["overlap"]))
    print("-" * 78)
    print("参照: 本机用本仓库代码 + 这份模板 + temple.jpg, 上面两行都应给出 得分=100.00")
    print("      若 UI 里看到 ~80, 请确认: (1) UI 是否重开过 (旧进程仍旧 DLL);")
    print("      (2) 是否画过'匹配ROI'(掩码裁到模板足迹会让得分按比例掉到 80 档);")
    print("      (3) 测试图是不是 temple.jpg 本身 (另一张 2.jpg 只有 34 分, 不会命中)。")


if __name__ == "__main__":
    main()
