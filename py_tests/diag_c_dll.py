# -*- coding: utf-8 -*-
"""校验某个 shape_based_matching_c.dll 里到底有没有"两侧都不滤波"的修复。

用法:
    python py_tests/diag_c_dll.py [dll路径]      # 默认 build/Release/shape_based_matching_c.dll

判据 (必须一正一反, 光看用户那张 X 光图分辨不出来 —— 它边缘太强, 旧版也是 100):
  [A] 判别用例: test/case0/1.jpg 的草地纹理 ROI(60,380,160,110) feat=200
      修复前 = 93.96, 修复后 = 100.00
  [B] 用户数据: Documents\\模版\\template_model + temple.jpg -> 应为 100.00
"""
import os
import sys
import time
import ctypes
import shutil
from ctypes import (c_int, c_double, c_char_p, c_void_p, c_ubyte,
                    POINTER, Structure, byref)

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SBM_MAX_PYRAMID_LEVELS = 8

USER_INFO = r"C:\Users\Administrator\Documents\模版\template_model.info.json"
USER_BIG = r"E:\programing\pathon\vision-inspect\app\setting\measure\sample\temple.jpg"


class SbmImage(Structure):
    _fields_ = [("data", POINTER(c_ubyte)), ("width", c_int), ("height", c_int),
                ("channels", c_int), ("step", c_int)]


class SbmTrainParams(Structure):
    _fields_ = [("feature_num", c_int), ("pyramid_level_count", c_int),
                ("pyramid_levels", c_int * SBM_MAX_PYRAMID_LEVELS),
                ("weak_thresh", c_double), ("strong_thresh", c_double),
                ("angle_start", c_double), ("angle_extent", c_double),
                ("angle_step", c_double), ("scale_start", c_double),
                ("scale_end", c_double), ("scale_step", c_double)]


class SbmTrainResult(Structure):
    _fields_ = [("yaml_path", c_char_p), ("info_path", c_char_p),
                ("preview_path", c_char_p), ("save_dir", c_char_p),
                ("base_name", c_char_p), ("template_count", c_int),
                ("features_image", SbmImage)]


class SbmMatchResult(Structure):
    _fields_ = [("class_id", c_char_p), ("template_id", c_int),
                ("score", c_double), ("x", c_double), ("y", c_double),
                ("icp_refined", c_int), ("box", c_double * 8),
                ("feature_count", c_int), ("features", POINTER(c_double)),
                ("refined_x", c_double), ("refined_y", c_double),
                ("refined_angle", c_double), ("fitness", c_double),
                ("overlap", c_double), ("grasp_count", c_int),
                ("grasp_points", POINTER(c_double)),
                ("angle", c_double), ("scale", c_double),
                ("refined_scale", c_double)]


def to_sbm_image(img):
    assert img.flags["C_CONTIGUOUS"]
    ch = 1 if img.ndim == 2 else int(img.shape[2])
    return SbmImage(img.ctypes.data_as(POINTER(c_ubyte)), img.shape[1], img.shape[0],
                    ch, int(img.strides[0]))


def make_params(feature_num, pyramid, weak, strong,
                angle_start=0.0, angle_extent=0.0, angle_step=10.0,
                scale_start=1.0, scale_end=1.0, scale_step=0.1):
    p = SbmTrainParams()
    p.feature_num = feature_num
    p.pyramid_level_count = len(pyramid)
    for i, lv in enumerate(pyramid):
        p.pyramid_levels[i] = lv
    p.weak_thresh, p.strong_thresh = weak, strong
    p.angle_start, p.angle_extent, p.angle_step = angle_start, angle_extent, angle_step
    p.scale_start, p.scale_end, p.scale_step = scale_start, scale_end, scale_step
    return p


def pad16(img):
    ph, pw = (-img.shape[0]) % 16, (-img.shape[1]) % 16
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def top_score(lib, h, scene, thr=80.0):
    n = lib.sbm_match(h, byref(to_sbm_image(scene)), thr, None, 0,
                      1, 0.5, 20, 0.0, 1, 1.0, None, 1)
    if n < 0:
        raise RuntimeError("sbm_match 失败: %s" % lib.sbm_last_error(h).decode("utf-8", "replace"))
    if n == 0:
        return None, 0
    r = SbmMatchResult()
    lib.sbm_match_result(h, 0, byref(r))
    return r.score, n


def main():
    dll = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "Release",
                                                             "shape_based_matching_c.dll")
    print("=" * 80)
    if not os.path.isfile(dll):
        raise SystemExit("找不到 DLL: %s" % dll)
    print("DLL      :", dll)
    print("大小/时间:", os.path.getsize(dll), "字节",
          time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(dll))))

    os.add_dll_directory(os.path.dirname(os.path.abspath(dll)))
    lib = ctypes.CDLL(dll)
    lib.sbm_create.restype = c_void_p
    lib.sbm_destroy.argtypes = [c_void_p]
    lib.sbm_last_error.restype = c_char_p
    lib.sbm_last_error.argtypes = [c_void_p]
    lib.sbm_train.restype = c_int
    lib.sbm_train.argtypes = [c_void_p, POINTER(SbmImage), POINTER(c_int * 4), c_char_p,
                              POINTER(SbmTrainParams), c_char_p, c_void_p, c_int,
                              POINTER(SbmImage), POINTER(SbmImage), POINTER(SbmTrainResult)]
    lib.sbm_add_template_class.restype = c_char_p
    lib.sbm_add_template_class.argtypes = [c_void_p, c_char_p, POINTER(SbmTrainParams),
                                           POINTER(SbmTrainParams)]
    lib.sbm_match.restype = c_int
    lib.sbm_match.argtypes = [c_void_p, POINTER(SbmImage), c_double, POINTER(c_char_p), c_int,
                              c_int, c_double, c_int, c_double, c_int, c_double,
                              POINTER(SbmImage), c_int]
    lib.sbm_match_result.restype = c_int
    lib.sbm_match_result.argtypes = [c_void_p, c_int, POINTER(SbmMatchResult)]

    save_dir = os.path.join(HERE, "_cdll_out")

    # ---------------- [A] 判别用例 ----------------
    print("-" * 80)
    print("[A] 判别用例: test/case0/1.jpg 草地纹理 ROI (修复前 93.96 / 修复后 100.00)")
    photo = cv2.imread(os.path.join(ROOT, "test", "case0", "1.jpg"))
    shutil.rmtree(save_dir, ignore_errors=True)
    h = lib.sbm_create()
    try:
        p = make_params(200, (4, 8), 30.0, 60.0)
        roi = (c_int * 4)(60, 380, 160, 110)
        out = SbmTrainResult()
        rc = lib.sbm_train(h, byref(to_sbm_image(photo)), byref(roi), b"grass",
                           byref(p), save_dir.encode("utf-8"), None, 0, None, None, byref(out))
        if rc != 0:
            print("    训练失败:", lib.sbm_last_error(h).decode("utf-8", "replace"))
        else:
            # train 只落盘产物, 不把类别装进句柄 (与 Python 门面一致) -> 再加载一次
            cid = lib.sbm_add_template_class(h, out.info_path, None, None)
            if not cid:
                print("    加载失败:", lib.sbm_last_error(h).decode("utf-8", "replace"))
            else:
                s, n = top_score(lib, h, pad16(photo.copy()))
                print("    自匹配得分: %s (命中 %d 条)  -> %s"
                      % ("%.2f" % s if s is not None else "无命中", n,
                         "含修复" if (s or 0) >= 99.99 else ("疑似旧版(带 medianBlur)" if s else "?")))
    finally:
        lib.sbm_destroy(h)
        shutil.rmtree(save_dir, ignore_errors=True)

    # ---------------- [B] 用户数据 ----------------
    print("-" * 80)
    print("[B] 用户数据: Documents\\模版\\template_model + temple.jpg (期望 100.00)")
    if not (os.path.isfile(USER_INFO) and os.path.isfile(USER_BIG)):
        print("    跳过: 文件不存在")
        return
    big = pad16(imread_u(USER_BIG))
    h2 = lib.sbm_create()
    try:
        final = SbmTrainParams()
        cid = lib.sbm_add_template_class(h2, USER_INFO.encode("utf-8"), None, byref(final))
        if not cid:
            print("    加载模板失败:", lib.sbm_last_error(h2).decode("utf-8", "replace"))
        else:
            print("    类别:", cid.decode("utf-8"))
            s, n = top_score(lib, h2, big)
            print("    自匹配得分: %s (命中 %d 条)"
                  % ("%.2f" % s if s is not None else "无命中", n))
    finally:
        lib.sbm_destroy(h2)


    # ---------------- [C] C# 夹具配置: 大 ROI + 奇偶性错开的角度格点 ----------------
    print("-" * 80)
    print("[C] C# 工具配置 (temple.jpg, ROI 259,173,1375x491, 角度 -45~45 step2) 重训后")
    d = os.path.join(HERE, "_cdll_cs")
    shutil.rmtree(d, ignore_errors=True)
    if os.path.isfile(USER_BIG):
        big2 = imread_u(USER_BIG)
        h3 = lib.sbm_create()
        try:
            p3 = make_params(128, (4, 8), 30.0, 60.0, angle_start=-45.0, angle_extent=90.0,
                             angle_step=2.0, scale_start=1.0, scale_end=1.0, scale_step=0.1)
            roi3 = (c_int * 4)(259, 173, 1375, 491)
            out3 = SbmTrainResult()
            rc = lib.sbm_train(h3, byref(to_sbm_image(big2)), byref(roi3), b"template",
                               byref(p3), d.encode("utf-8"), None, 0, None, None, byref(out3))
            if rc != 0:
                print("    训练失败:", lib.sbm_last_error(h3).decode("utf-8", "replace"))
            else:
                print("    模板数:", out3.template_count, "(含 0° 后应为 47)")
                cid3 = lib.sbm_add_template_class(h3, out3.info_path, None, None)
                if not cid3:
                    print("    加载失败:", lib.sbm_last_error(h3).decode("utf-8", "replace"))
                else:
                    s, n = top_score(lib, h3, pad16(big2.copy()))
                    print("    自匹配得分: %s (命中 %d 条)  -> %s"
                          % ("%.2f" % s if s is not None else "无命中", n,
                             "OK (含 0° 修复)" if (s or 0) >= 99.99 else "仍偏低!"))
        finally:
            lib.sbm_destroy(h3)
            shutil.rmtree(d, ignore_errors=True)
    else:
        print("    跳过: 找不到", USER_BIG)


if __name__ == "__main__":
    main()
