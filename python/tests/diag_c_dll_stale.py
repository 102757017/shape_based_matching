# -*- coding: utf-8 -*-
"""
对照实验: 用 ctypes 分别加载 build/Release/ 与 build/ 根目录下的 shape_based_matching_c.dll,
跑同一条 C ABI 链路 (训练 -> 加载 -> match(use_refine=1)), 看是否复现
C# 侧报的 "Gradient maps (dx_, dy_) are empty"。

用法: python diag_c_dll_stale.py [dll 路径]
"""
import os
import sys
import ctypes
import shutil
from ctypes import (c_int, c_double, c_char_p, c_void_p, c_ubyte,
                    POINTER, Structure, byref)

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from test_c_api import (SbmImage, SbmTrainParams, SbmTrainResult, SbmMatchResult,
                        to_sbm_image, make_params, pad16)


def run(dll_path):
    print("=" * 70)
    print("DLL:", dll_path, " 修改时间:",
          __import__("time").strftime("%Y-%m-%d %H:%M:%S",
                                      __import__("time").localtime(os.path.getmtime(dll_path))))
    if not os.path.isfile(dll_path):
        print("  找不到该 DLL, 跳过")
        return
    lib = ctypes.CDLL(dll_path)
    lib.sbm_create.restype = c_void_p
    lib.sbm_last_error.restype = c_char_p
    lib.sbm_last_error.argtypes = [c_void_p]
    lib.sbm_train_params_init.argtypes = [POINTER(SbmTrainParams)]
    lib.sbm_add_template_class.restype = c_char_p
    lib.sbm_add_template_class.argtypes = [c_void_p, c_char_p, POINTER(SbmTrainParams), POINTER(SbmTrainParams)]
    lib.sbm_loaded_class_count.restype = c_int
    lib.sbm_loaded_class_count.argtypes = [c_void_p]
    lib.sbm_match.restype = c_int
    lib.sbm_match.argtypes = [c_void_p, POINTER(SbmImage), c_double,
                              POINTER(c_char_p), c_int, c_int, c_double,
                              c_int, c_double, c_int, c_double, POINTER(SbmImage), c_int]
    lib.sbm_match_result.restype = c_int
    lib.sbm_match_result.argtypes = [c_void_p, c_int, POINTER(SbmMatchResult)]

    train_img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"))
    th, tw = train_img.shape[:2]
    save_dir = os.path.join(HERE, "输出_stale")
    shutil.rmtree(save_dir, ignore_errors=True)

    h2 = lib.sbm_create()
    try:
        params = make_params()
        roi = (c_int * 4)(0, 0, tw, th)
        out = SbmTrainResult()
        rc = lib.sbm_train_placeholder if False else None
        # 训练: 直接复用导出符号 sbm_train
        lib.sbm_train.restype = c_int
        lib.sbm_train.argtypes = [c_void_p, POINTER(SbmImage), POINTER(c_int * 4), c_char_p,
                                  POINTER(SbmTrainParams), c_char_p, c_void_p, c_int,
                                  POINTER(SbmImage), POINTER(SbmImage), POINTER(SbmTrainResult)]
        rc = lib.sbm_train(h2, byref(to_sbm_image(train_img)), byref(roi),
                           "obj".encode("utf-8"), byref(params),
                           save_dir.encode("utf-8"), None, 0, None, None, byref(out))
        print("  sbm_train rc =", rc, lib.sbm_last_error(h2).decode() if rc else "")
        if rc != 0:
            return
        cid = lib.sbm_add_template_class(h2, out.yaml_path.decode("utf-8").encode("utf-8"), None, byref(params))
        if not cid:
            print("  加载失败:", lib.sbm_last_error(h2).decode())
            return

        OX, OY = 101, 97
        scene = np.zeros((th + 300, tw + 300, 3), np.uint8)
        scene[OY:OY + th, OX:OX + tw] = train_img
        scene = pad16(scene)
        n = lib.sbm_match(h2, byref(to_sbm_image(scene)), 50.0, None, 0,
                          1, 0.5, 0, 0.0, 1, 1.0, None, 1)     # useRefine = 1
        print("  sbm_match(useRefine=1) rc =", n)
        if n < 0:
            print("  错误信息:", lib.sbm_last_error(h2).decode())
        else:
            r = SbmMatchResult()
            lib.sbm_match_result(h2, 0, byref(r))
            print(f"  结果: 中心=({r.refined_x:.2f},{r.refined_y:.2f}) "
                  f"角度={r.refined_angle:.3f} fitness={r.fitness:.4f}")
    finally:
        lib.sbm_destroy.argtypes = [c_void_p]
        lib.sbm_destroy(h2)
        shutil.rmtree(save_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        run(os.path.join(ROOT, "build", "Release", "shape_based_matching_c.dll"))
        run(os.path.join(ROOT, "build", "shape_based_matching_c.dll"))
