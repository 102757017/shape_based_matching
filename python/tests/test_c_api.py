# -*- coding: utf-8 -*-
"""C ABI 冒烟测试: 用 ctypes 直接调用 shape_based_matching_c.dll。

C# 侧 P/Invoke 走的就是这套 ABI, 所以这里能跑通即证明 C# 可用。
验证链路: 训练(中文目录/中文类别名) -> 加载 -> 匹配 -> 几何校验,
再与 Python 门面 (matcher.py) 的结果交叉核对, 确认两个前端口径一致。
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
ROOT = os.path.dirname(os.path.dirname(HERE))                     # 项目根目录
sys.path.insert(0, ROOT)

DLL_PATH = os.path.join(ROOT, "build", "Release", "shape_based_matching_c.dll")

SBM_MAX_PYRAMID_LEVELS = 8


# ---------------- ABI 结构体 (必须与 c_api/matcher_c.h 一致) ----------------

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
                # 后加字段, 与 c_api/matcher_c.h 的追加顺序一致
                ("angle", c_double), ("scale", c_double),
                ("refined_scale", c_double)]


def to_sbm_image(img: np.ndarray) -> SbmImage:
    assert img.flags["C_CONTIGUOUS"], "图像必须是连续内存"
    ch = 1 if img.ndim == 2 else int(img.shape[2])
    return SbmImage(img.ctypes.data_as(POINTER(c_ubyte)), img.shape[1], img.shape[0],
                    ch, int(img.strides[0]))


def make_params(feature_num=100, pyramid=(4, 8), weak=30.0, strong=60.0,
                angle_start=0.0, angle_extent=0.0, angle_step=10.0,
                scale_start=1.0, scale_end=1.0, scale_step=0.1) -> SbmTrainParams:
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


def main():
    if not os.path.isfile(DLL_PATH):
        raise SystemExit("找不到 DLL: %s (先构建 shape_based_matching_c 目标)" % DLL_PATH)

    lib = ctypes.CDLL(DLL_PATH)
    lib.sbm_create.restype = c_void_p
    lib.sbm_create.argtypes = []
    lib.sbm_destroy.argtypes = [c_void_p]
    lib.sbm_clear.argtypes = [c_void_p]
    lib.sbm_last_error.restype = c_char_p
    lib.sbm_last_error.argtypes = [c_void_p]
    lib.sbm_train_params_init.argtypes = [POINTER(SbmTrainParams)]
    lib.sbm_train.restype = c_int
    lib.sbm_train.argtypes = [c_void_p, POINTER(SbmImage), POINTER(c_int * 4), c_char_p,
                              POINTER(SbmTrainParams), c_char_p, c_void_p, c_int,
                              POINTER(SbmImage), POINTER(SbmImage), POINTER(SbmTrainResult)]
    lib.sbm_add_template_class.restype = c_char_p
    lib.sbm_add_template_class.argtypes = [c_void_p, c_char_p,
                                           POINTER(SbmTrainParams), POINTER(SbmTrainParams)]
    lib.sbm_loaded_class_count.restype = c_int
    lib.sbm_loaded_class_count.argtypes = [c_void_p]
    lib.sbm_loaded_class_id.restype = c_char_p
    lib.sbm_loaded_class_id.argtypes = [c_void_p, c_int]
    lib.sbm_match.restype = c_int
    lib.sbm_match.argtypes = [c_void_p, POINTER(SbmImage), c_double,
                              POINTER(c_char_p), c_int, c_int, c_double,
                              c_int, c_double, c_int, c_double, POINTER(SbmImage), c_int]
    lib.sbm_match_result.restype = c_int
    lib.sbm_match_result.argtypes = [c_void_p, c_int, POINTER(SbmMatchResult)]
    lib.sbm_set_grasp_points.argtypes = [c_void_p, c_char_p, POINTER(c_double), c_int]

    # ---------- 0. 默认参数自检 ----------
    dp = SbmTrainParams()
    lib.sbm_train_params_init(byref(dp))
    assert dp.feature_num == 100 and dp.pyramid_level_count == 2, "默认参数不对"
    print("[0] 默认参数: feature_num=%d pyramid=[%d,%d] weak=%.1f strong=%.1f"
          % (dp.feature_num, dp.pyramid_levels[0], dp.pyramid_levels[1],
             dp.weak_thresh, dp.strong_thresh))

    train_img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"))
    assert train_img is not None
    th, tw = train_img.shape[:2]
    save_dir = os.path.join(HERE, "输出_CABI")
    shutil.rmtree(save_dir, ignore_errors=True)

    # ---------- 1. 训练 ----------
    h = lib.sbm_create()
    assert h
    try:
        params = make_params()
        roi = (c_int * 4)(0, 0, tw, th)
        out = SbmTrainResult()
        rc = lib.sbm_train(h, byref(to_sbm_image(train_img)), byref(roi),
                           "夹具A".encode("utf-8"), byref(params),
                           save_dir.encode("utf-8"), None, 0, None, None, byref(out))
        assert rc == 0, "sbm_train 失败: %s" % lib.sbm_last_error(h)
        info_path = out.info_path.decode("utf-8")
        yaml_path = out.yaml_path.decode("utf-8")
        assert out.template_count > 0
        for p in (yaml_path, info_path):
            assert os.path.isfile(p), p
        assert os.path.isfile(os.path.join(save_dir, "夹具A.jpg"))
        # features_image 指针可以读回有效的 ndarray
        fi = out.features_image
        npy = np.ctypeslib.as_array(fi.data, shape=(fi.height, fi.width, fi.channels))
        assert npy.shape == (th, tw, 3)
        print("[1] 训练通过: template_count=%d, 类别=%s, 标注图=%dx%d"
              % (out.template_count, out.base_name.decode("utf-8"), fi.width, fi.height))

        # ---------- 2. 加载 ----------
        h2 = lib.sbm_create()
        assert h2
        final = SbmTrainParams()
        cid = lib.sbm_add_template_class(h2, info_path.encode("utf-8"), None, byref(final))
        assert cid, "sbm_add_template_class 失败: %s" % lib.sbm_last_error(h2)
        class_id = cid.decode("utf-8")
        assert class_id == "夹具A", class_id
        assert final.feature_num == 100 and final.pyramid_levels[0] == 4
        n_cls = lib.sbm_loaded_class_count(h2)
        assert n_cls == 1 and lib.sbm_loaded_class_id(h2, 0).decode("utf-8") == "夹具A"
        print("[2] 加载通过: class_id=%s, 参数 feature_num=%d" % (class_id, final.feature_num))

        # ---------- 3. 匹配 ----------
        OX, OY = 101, 97
        scene = np.zeros((th + 300, tw + 300, 3), np.uint8)
        scene[OY:OY + th, OX:OX + tw] = train_img
        scene = pad16(scene)
        n = lib.sbm_match(h2, byref(to_sbm_image(scene)), 50.0, None, 0,
                          1, 0.5, 0, 0.0, 1, 1.0, None, 1)
        assert n >= 1, "sbm_match 未匹配到目标 (返回 %d)" % n
        r = SbmMatchResult()
        assert lib.sbm_match_result(h2, 0, byref(r)) == 0
        assert r.class_id.decode("utf-8") == "夹具A"
        assert abs(r.refined_x - (OX + tw / 2)) < 15 and abs(r.refined_y - (OY + th / 2)) < 15, \
            (r.refined_x, r.refined_y)
        assert abs(r.refined_angle) < 8, r.refined_angle
        assert r.fitness >= 0
        box = [(r.box[2 * i], r.box[2 * i + 1]) for i in range(4)]
        feats = [(r.features[2 * i], r.features[2 * i + 1]) for i in range(r.feature_count)]
        grasps = [(r.grasp_points[2 * i], r.grasp_points[2 * i + 1])
                  for i in range(r.grasp_count)]
        assert len(feats) > 0 and len(grasps) >= 1
        # scale / angle / refined_scale 必须透出: angle 应等于模板训练角(默认 0),
        # refined_scale 是 ICP 的额外缩放(自匹配应贴近 1.0)
        assert r.scale != 0 and abs(r.angle) < 1e-6, (r.angle, r.scale)
        assert abs(r.refined_scale - 1.0) < 0.5, r.refined_scale
        print("[3] 匹配通过: %d 个结果, 中心=(%.1f,%.1f) 角度=%.3f 特征点=%d 抓取点=%d"
              % (n, r.refined_x, r.refined_y, r.refined_angle, len(feats), len(grasps)))
        print("[3b] 姿态字段透出: angle=%.3f scale=%.4f refined_scale=%.4f (最终大小 = scale*refined_scale)"
              % (r.angle, r.scale, r.refined_scale))

        # ---------- 4. 自定义抓取点 ----------
        gp = (c_double * 2)(10.0, 20.0)
        assert lib.sbm_set_grasp_points(h2, "夹具A".encode("utf-8"), gp, 1) == 0
        n = lib.sbm_match(h2, byref(to_sbm_image(scene)), 50.0, None, 0,
                          1, 0.5, 0, 0.0, 1, 1.0, None, 1)
        assert n >= 1
        r2 = SbmMatchResult()
        assert lib.sbm_match_result(h2, 0, byref(r2)) == 0
        assert abs(r2.grasp_points[0] - (OX + 10)) < 15 and \
               abs(r2.grasp_points[1] - (OY + 20)) < 15, (r2.grasp_points[0], r2.grasp_points[1])
        print("[4] 抓取点通过: (%.1f,%.1f) 期望 (%.1f,%.1f)"
              % (r2.grasp_points[0], r2.grasp_points[1], OX + 10, OY + 20))

        # ---------- 5. 与 Python 门面交叉核对 ----------
        from matcher import Matcher
        py_dir = os.path.join(HERE, "输出_CABI_PY")
        shutil.rmtree(py_dir, ignore_errors=True)
        m = Matcher()
        m.train(train_img, (0, 0, tw, th), "夹具A",
                {"pyramid_levels": [4, 8], "feature_num": 100,
                 "weak_thresh": 30.0, "strong_thresh": 60.0,
                 "angle_start": 0.0, "angle_extent": 0.0, "angle_step": 10.0,
                 "scale_start": 1.0, "scale_end": 1.0, "scale_step": 0.1},
                py_dir)
        m2 = Matcher()
        m2.add_template_class(os.path.join(py_dir, "夹具A.info.json"))
        py_res = m2.match(scene, 50.0, use_refine=True)
        assert py_res, "Python 侧应匹配到目标"
        p0 = py_res[0]
        # ICP 精修是并行的, 迭代顺序会带来极小浮点差(实测 <1e-4 px), 这里留 0.01px 容差
        # (参照 test_ab_compare 用的 1.5px 容差)。
        diffs = {
            "refined_x": abs(p0["refined_x"] - r.refined_x),
            "refined_y": abs(p0["refined_y"] - r.refined_y),
            "refined_angle": abs(p0["refined_angle"] - r.refined_angle),
            "score": abs(p0["score"] - r.score),
        }
        for k, v in diffs.items():
            assert v < 0.01, "%s 与 Python 结果不一致: 差 %.3g" % (k, v)
        print("[5] 与 Python 门面一致: 最大偏差 %.3g px (容差 0.01)"
              % max(diffs.values()))

        # ---------- 6. clear ----------
        lib.sbm_clear(h2)
        assert lib.sbm_loaded_class_count(h2) == 0
        print("[6] clear 通过")

        lib.sbm_destroy(h2)
        h2 = None
    finally:
        if h2:
            lib.sbm_destroy(h2)
        lib.sbm_destroy(h)
        shutil.rmtree(save_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(HERE, "输出_CABI_PY"), ignore_errors=True)

    print("\nC ABI 全部通过")


if __name__ == "__main__":
    main()
