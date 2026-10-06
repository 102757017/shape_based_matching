# -*- coding: utf-8 -*-
# 合并后冒烟测试: 用 fusion 流水线构建响应图 + 读入模板 + 匹配
import os, sys, time
import numpy as np
import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'build', 'Release'))
import shape_based_matching_py as sbm

TEST_DIR = os.path.join(_ROOT, 'test', 'case1')

det = sbm.Detector(128, [4, 8])
det.readClasses(['test'], os.path.join(TEST_DIR, '%s_templ.yaml'))
print('templates loaded:', det.numTemplates('test'))

img = cv2.imread(os.path.join(TEST_DIR, 'test.png'), cv2.IMREAD_GRAYSCALE)
assert img is not None and img.size > 0, 'test.png load failed'
print('scene img:', img.shape)

t0 = time.time()
matches = det.match(img, 70.0, ['test'])
dt = time.time() - t0
print('match time: %.3fs, matches: %d' % (dt, len(matches)))
for m in matches[:5]:
    print('  Match class=%s conf=%.1f x=%d y=%d templ_id=%d' % (
        m.class_id, m.confidence, m.x, m.y, m.template_id))

# 带 mask 的调用: 走经典路径(fusion 流水线暂不支持 mask)
mask = np.full(img.shape, 255, dtype=np.uint8)
matches_masked = det.match(img, 70.0, ['test'], mask)
print('mask path: %d matches' % len(matches_masked))
for m in matches_masked[:3]:
    print('  Match class=%s conf=%.1f x=%d y=%d' % (m.class_id, m.confidence, m.x, m.y))
assert len(matches_masked) > 0, 'mask path returned no matches'

# 完整参数版(含 MatchParams / NMS / ICP 精修开关)验证
params = sbm.MatchParams()
params.min_confidence = 60.0
params.use_refine = True
ms = det.match(img, params)
print('MatchParams path: %d matches' % len(ms))
for m in ms[:5]:
    print('  conf=%.3f angle=%.1f scale=%.3f fitness=%.3f overlap=%.3f' % (
        m.confidence, m.angle, m.scale, m.fitness, m.overlap))
print('SMOKE OK')
