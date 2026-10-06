# -*- coding: utf-8 -*-
"""离屏验证: ui.py 的"自动探测阈值"按钮能拿到建议值并回填到弱/强阈值输入框。

覆盖两条路径:
  1) 正常路径 —— 按钮走完整链路 (train_image + ROI + 涂抹掩码 -> C++ 探测 -> 回填);
  2) 无训练图 —— 必须给出明确提示, 不能抛异常 / 不能静默填默认值。
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))            # py_tests/
ROOT = os.path.dirname(os.path.dirname(HERE))                                  # 项目根目录
sys.path.insert(0, ROOT)                                      # 让根目录的 ui.py 可导入

import numpy as np
import cv2
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QRect

app = QApplication(sys.argv)
import ui

win = ui.TemplateMatchingApp()

# 弹窗会阻塞, 这里拦下来只记参数
captured = {}
win._show_thresholds_dialog = lambda r, applied: captured.update(result=r, applied=applied)

img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"), cv2.IMREAD_UNCHANGED)
if img.ndim == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# ---- 路径1: 没加载训练图必须先拒绝 ----
win.train_image = None
win.estimate_thresholds()
msg = win.status_bar.currentMessage()
print("无训练图提示:", msg)
assert "请先加载训练图" in msg, f"无训练图时未给出提示: {msg}"

# ---- 路径2: 正常探测并回填 ----
win.train_image = img.copy()
win.roi_rect = QRect(150, 118, 300, 236)
win.create_feature_num.setValue(128)
win.create_weak_thresh.setValue(30.0)
win.create_strong_thresh.setValue(60.0)

win.estimate_thresholds()

weak, strong = win.create_weak_thresh.value(), win.create_strong_thresh.value()
res = captured["result"]
print("回填后阈值: 弱 %.1f / 强 %.1f (原 30.0 / 60.0)" % (weak, strong))
print("状态栏:", win.status_bar.currentMessage())

assert captured.get("applied") is True, "探测失败, UI 未标记为已回填"
assert weak != 30.0 and strong != 60.0, "探测结果没回填到输入框"
assert abs(weak - res["weak_thresh"]) < 1e-6, "UI 回填值与 C++ 返回值不一致"
assert abs(strong - res["strong_thresh"]) < 1e-6, "UI 回填值与 C++ 返回值不一致"
assert 0.0 < weak < strong <= 255.0, f"阈值区间不合理: {weak} / {strong}"
assert int(res["features"]) >= 60, f"特征点数太少: {res['features']}"
assert res["features_image"] is not None, "没有返回带特征点红点的预览图"

# 预览图上确实画了红点
fi = res["features_image"]
if fi.ndim == 3 and fi.shape[2] == 4:
    fi = cv2.cvtColor(fi, cv2.COLOR_BGRA2BGR)
b, g, r = fi[:, :, 0].astype(int), fi[:, :, 1].astype(int), fi[:, :, 2].astype(int)
n_red = int(((r - np.maximum(b, g)) > 60).sum())
print("预览图红点像素:", n_red)
assert n_red > 0, "预览图上没有特征点红点"

print("== 离屏验证通过: 自动探测阈值按钮会回填弱/强阈值 ==")
