# -*- coding: utf-8 -*-
# 复现/验证: 同一句柄内重训(带排除区)必须"替换"旧模板, 而不是静默保留
import os, sys, shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'build', 'Release'))
import numpy as np
import cv2
import shape_based_matching_py as sbm

TMP = os.path.join(_ROOT, '.workbuddy', 'tmp_retrain_test')
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)

train_img = cv2.imread(os.path.join(_ROOT, 'test', 'case1', 'train.png'), cv2.IMREAD_GRAYSCALE)
scene_img = cv2.imread(os.path.join(_ROOT, 'test', 'case1', 'test.png'), cv2.IMREAD_GRAYSCALE)
assert train_img is not None and scene_img is not None
ROI = [130, 110, 270, 270]

TP = dict(feature_num=60, pyramid_levels=[4, 8], weak_thresh=30.0, strong_thresh=60.0,
          angle_start=0.0, angle_extent=0.0, angle_step=15.0,
          scale_start=1.0, scale_end=1.0, scale_step=0.5)
CLS = 'repl_test'

m = sbm.PyMatcher()

# ---------- 第 1 次: 不带排除区训练 ----------
r1 = m.train(train_img, ROI, CLS, TP, save_dir=TMP)
cid1, _ = m.add_template_class(r1['yaml_path'])
assert cid1 == CLS
ms1 = m.match(scene_img, 30.0, CLS, use_refine=False)
print('load#1: templates=%s, matches=%d' % (m.get_loaded_class_ids(), len(ms1)))
top1 = ms1[0] if ms1 else None
if top1:
    print('  top: score=%.2f x=%d y=%d feats=%d' % (
        top1['score'], top1['x'], top1['y'], len(top1['matched_features'])))

assert len(ms1) > 0, '第一次训练后应有匹配结果'
assert m.get_loaded_class_ids() == [CLS]

# ---------- 第 2 次: 用排除区罩住第一次命中目标的中心区域重训 ----------
top = ms1[0]
feats = top['matched_features']
fxs = [p[0] for p in feats]; fys = [p[1] for p in feats]
bw = max(8, int(max(fxs) - min(fxs)))
bh = max(8, int(max(fys) - min(fys)))
# 取特征包围盒的中心一半, 保证重训后模板仍可生成(只是中心被挖掉)
zx, zy = int(min(fxs)) + bw // 4, int(min(fys)) + bh // 4
zw, zh = max(8, bw // 2), max(8, bh // 2)
zone = [{'type': 'exclude_rect', 'rect': [zx, zy, zw, zh]}]
print('exclude zone:', zone[0]['rect'])

r2 = m.train(train_img, ROI, CLS, TP, save_dir=TMP, exclusion_zones=zone)
assert r2['template_count'] > 0
cid2, _ = m.add_template_class(r2['yaml_path'])   # <-- 修复前这里 CV_Assert 崩掉/静默失败
assert cid2 == CLS
assert m.get_loaded_class_ids() == [CLS], '重训加载后句柄里应只有一个同名类别'
ms2 = m.match(scene_img, 30.0, CLS, use_refine=False)
print('load#2: matches=%d' % len(ms2))
for mm in ms2[:3]:
    print('  score=%.2f x=%d y=%d' % (mm['score'], mm['x'], mm['y']))

# ---------- 断言: 重训后结果必须变化, 且命中位置不再落在排除区内 ----------
def sig(ms):
    return [(round(mm['score'], 2), mm['x'], mm['y']) for mm in ms[:10]]

s1, s2 = sig(ms1), sig(ms2)
assert s1 != s2, '重训(带排除区)后匹配结果竟然和重训前一模一样! 旧模板没有被替换'

inside = [mm for mm in ms2
          if zx <= mm['x'] < zx + bw and zy <= mm['y'] < zy + bh]
print('hits inside exclusion zone: %d / %d' % (len(inside), len(ms2)))
assert not inside, '重训后仍有匹配落在排除区内'

print('RETRAIN-REPLACE OK')
