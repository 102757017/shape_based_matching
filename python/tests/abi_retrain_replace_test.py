# -*- coding: utf-8 -*-
# 用 ctypes 直接测 shape_based_matching_c.dll (C# P/Invoke 同一条 ABI):
# 同一句柄 训练(无排除区)->加载->匹配 -> 训练(带排除区)->加载->匹配
# 验证"重训 = 替换旧模板"语义
import os, sys, shutil, ctypes as C
from ctypes import wintypes

_ROOT = r'E:\programing\CSharp\shape_based_matching'
REL = os.path.join(_ROOT, 'build', 'Release')
sys.path.insert(0, REL)
import numpy as np
import cv2

os.add_dll_directory(REL)
dll = C.CDLL(os.path.join(REL, 'shape_based_matching_c.dll'))

SBM_MAX_PYRAMID_LEVELS = 8

class SbmImage(C.Structure):
    _fields_ = [('data', C.POINTER(C.c_ubyte)), ('width', C.c_int),
                ('height', C.c_int), ('channels', C.c_int), ('step', C.c_int)]

class TrainParams(C.Structure):
    _fields_ = [('feature_num', C.c_int), ('pyramid_level_count', C.c_int),
                ('pyramid_levels', C.c_int * SBM_MAX_PYRAMID_LEVELS),
                ('weak_thresh', C.c_double), ('strong_thresh', C.c_double),
                ('angle_start', C.c_double), ('angle_extent', C.c_double),
                ('angle_step', C.c_double), ('scale_start', C.c_double),
                ('scale_end', C.c_double), ('scale_step', C.c_double)]

class ExclusionZone(C.Structure):
    _fields_ = [('type', C.c_char_p), ('x', C.c_int), ('y', C.c_int),
                ('w', C.c_int), ('h', C.c_int)]

class TrainResult(C.Structure):
    _fields_ = [('yaml_path', C.c_char_p), ('info_path', C.c_char_p),
                ('preview_path', C.c_char_p), ('save_dir', C.c_char_p),
                ('base_name', C.c_char_p), ('template_count', C.c_int),
                ('features_image', SbmImage)]

class MatchResult(C.Structure):
    _fields_ = [('class_id', C.c_char_p), ('template_id', C.c_int),
                ('score', C.c_double), ('x', C.c_double), ('y', C.c_double),
                ('icp_refined', C.c_int), ('box', C.c_double * 8),
                ('feature_count', C.c_int), ('features', C.POINTER(C.c_double)),
                ('refined_x', C.c_double), ('refined_y', C.c_double),
                ('refined_angle', C.c_double), ('fitness', C.c_double),
                ('overlap', C.c_double), ('grasp_count', C.c_int),
                ('grasp_points', C.POINTER(C.c_double)),
                ('angle', C.c_double), ('scale', C.c_double),
                ('refined_scale', C.c_double)]

def bind(name, rest, args):
    f = getattr(dll, name); f.restype = rest; f.argtypes = args; return f

sbm_create = bind('sbm_create', C.c_void_p, [])
sbm_destroy = bind('sbm_destroy', None, [C.c_void_p])
sbm_clear = bind('sbm_clear', None, [C.c_void_p])
sbm_train_params_init = bind('sbm_train_params_init', None, [C.POINTER(TrainParams)])
sbm_train = bind('sbm_train', C.c_int, [C.c_void_p, C.POINTER(SbmImage), C.POINTER(C.c_int),
                                        C.c_char_p, C.POINTER(TrainParams), C.c_char_p,
                                        C.POINTER(ExclusionZone), C.c_int,
                                        C.POINTER(SbmImage), C.POINTER(SbmImage),
                                        C.POINTER(TrainResult)])
sbm_add_template_class = bind('sbm_add_template_class', C.c_char_p,
                              [C.c_void_p, C.c_char_p, C.POINTER(TrainParams), C.POINTER(TrainParams)])
sbm_loaded_class_count = bind('sbm_loaded_class_count', C.c_int, [C.c_void_p])
sbm_match = bind('sbm_match', C.c_int, [C.c_void_p, C.POINTER(SbmImage), C.c_double,
                                        C.POINTER(C.c_char_p), C.c_int, C.c_int, C.c_double,
                                        C.c_int, C.c_double, C.c_int, C.c_double,
                                        C.POINTER(SbmImage), C.c_int])
sbm_match_result = bind('sbm_match_result', C.c_int, [C.c_void_p, C.c_int, C.POINTER(MatchResult)])
sbm_last_error = bind('sbm_last_error', C.c_char_p, [C.c_void_p])

def make_image(arr):
    arr = np.ascontiguousarray(arr)
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    return SbmImage(arr.ctypes.data_as(C.POINTER(C.c_ubyte)),
                    arr.shape[1], arr.shape[0], ch, 0), arr  # arr 保活

def do_train(h, img, roi, cls, zones, save_dir, tag):
    tp = TrainParams()
    sbm_train_params_init(C.byref(tp))
    tp.feature_num = 60
    tp.pyramid_level_count = 2
    tp.pyramid_levels[0] = 4; tp.pyramid_levels[1] = 8
    tp.weak_thresh = 30.0; tp.strong_thresh = 60.0
    tp.angle_start = 0.0; tp.angle_extent = 0.0; tp.angle_step = 15.0
    tp.scale_start = 1.0; tp.scale_end = 1.0; tp.scale_step = 0.5
    zarr = (ExclusionZone * len(zones))() if zones else None
    zptr = C.cast(zarr, C.POINTER(ExclusionZone)) if zones else None
    for i, (t, x, y, w, hh) in enumerate(zones or []):
        zarr[i].type = t.encode(); zarr[i].x = x; zarr[i].y = y; zarr[i].w = w; zarr[i].h = hh
    simg, keep1 = make_image(img)
    tr = TrainResult()
    rc = sbm_train(h, C.byref(simg), (C.c_int * 4)(*roi), cls.encode(),
                   C.byref(tp), save_dir.encode(), zptr, len(zones),
                   None, None, C.byref(tr))
    assert rc == 0, 'sbm_train(%s) rc=%d err=%s' % (tag, rc, sbm_last_error(h))
    cid = sbm_add_template_class(h, tr.yaml_path, None, None)
    assert cid is not None, 'sbm_add_template_class(%s) 失败: %s' % (tag, sbm_last_error(h))
    assert cid.decode() == cls
    return tr.template_count

def do_match(h, img, cls, th=30.0):
    simg, keep = make_image(img)
    cid = (C.c_char_p * 1)(cls.encode())
    n = sbm_match(h, C.byref(simg), th, cid, 1, 1, 0.5, 0, 0.0, 0, 1.0, None, 1)
    assert n >= 0, 'sbm_match err=%s' % sbm_last_error(h)
    out = []
    for i in range(n):
        mr = MatchResult()
        assert sbm_match_result(h, i, C.byref(mr)) == 0
        feats = [(mr.features[2*j], mr.features[2*j+1]) for j in range(mr.feature_count)]
        out.append(dict(score=mr.score, x=mr.x, y=mr.y, feats=feats,
                        templ_id=mr.template_id))
    return out

TMP = os.path.join(_ROOT, '.workbuddy', 'tmp_abi_retrain')
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP, exist_ok=True)
train_img = cv2.imread(os.path.join(_ROOT, 'test', 'case1', 'train.png'), cv2.IMREAD_GRAYSCALE)
scene_img = cv2.imread(os.path.join(_ROOT, 'test', 'case1', 'test.png'), cv2.IMREAD_GRAYSCALE)
ROI = [130, 110, 270, 270]
CLS = 'abi_repl'

h = sbm_create()
try:
    n1 = do_train(h, train_img, ROI, CLS, [], TMP, '第1次')
    print('train#1 templates=%d, loaded=%d' % (n1, sbm_loaded_class_count(h)))
    ms1 = do_match(h, scene_img, CLS)
    assert ms1, '第一次训练后应有匹配'
    top = ms1[0]
    print('match#1: %d 个, top score=%.2f (%.0f,%.0f) feats=%d' % (
        len(ms1), top['score'], top['x'], top['y'], len(top['feats'])))

    # 排除区罩住命中目标特征包围盒的中心一半
    fxs = [p[0] for p in top['feats']]; fys = [p[1] for p in top['feats']]
    bw = max(8, int(max(fxs) - min(fxs))); bh = max(8, int(max(fys) - min(fys)))
    zx, zy = int(min(fxs)) + bw // 4, int(min(fys)) + bh // 4
    zw, zh = max(8, bw // 2), max(8, bh // 2)
    print('exclude zone:', (zx, zy, zw, zh))

    n2 = do_train(h, train_img, ROI, CLS,
                  [('exclude_rect', zx, zy, zw, zh)], TMP, '第2次')
    print('train#2 templates=%d, loaded=%d' % (n2, sbm_loaded_class_count(h)))
    assert sbm_loaded_class_count(h) == 1, '重训加载后句柄里应只有 1 个类别'

    ms2 = do_match(h, scene_img, CLS)
    assert ms2, '重训后应有匹配'
    print('match#2: %d 个, top score=%.2f (%.0f,%.0f)' % (
        len(ms2), ms2[0]['score'], ms2[0]['x'], ms2[0]['y']))

    def sig(ms):
        return [(round(m['score'], 2), int(m['x']), int(m['y'])) for m in ms[:10]]
    assert sig(ms1) != sig(ms2), '重训后结果一模一样 -> 旧模板没被替换!'
    inside = [m for m in ms2 if zx <= m['x'] < zx+zw and zy <= m['y'] < zy+zh]
    print('hits inside zone: %d / %d' % (len(inside), len(ms2)))
    assert not inside, '重训后仍有匹配落在排除区内!'
    print('ABI RETRAIN-REPLACE OK')
finally:
    sbm_destroy(h)
