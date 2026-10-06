# -*- coding: utf-8 -*-
"""离屏验证: UI 两种涂抹模式 -> train_template 真实链路。

1. 涂抹识别区(正向): 特征只落在涂抹处;
2. 涂抹排除区(负向): 特征避开涂抹处;
3. UI 控件渲染出半透明叠加 (绿/红);
4. 正向涂满又全擦光 -> 训练被拒绝。
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))            # py_tests/
ROOT = os.path.dirname(HERE)                                  # 项目根目录
sys.path.insert(0, ROOT)                                      # 让根目录的 ui.py 可导入

import json
import shutil
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtCore import QPoint

app = QApplication(sys.argv)
import ui

SAVE_DIR = os.path.join(HERE, "离屏涂抹验证")
win = ui.TemplateMatchingApp()
img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"))
th, tw = img.shape[:2]
win.train_image = img.copy()
win.original_image_for_display = img.copy()
win.current_cv_image = img.copy()
from PySide6.QtCore import QRect
win.set_main_roi(QRect(0, 0, tw, th))
win.class_id_input.setText("涂抹测试")

PARAMS_OK = True


def feats_of(info_path):
    return np.array(json.load(open(info_path, encoding='utf-8'))['base_template_features'], float)


# ---------- 1. 正向涂抹: 只留左半边 ----------
win.toggle_brush('brush_positive')
assert win.image_label.drawing_mode == 'brush_positive'
# 模拟鼠标轨迹: 沿左半图涂几笔 (直接调 paint_brush_stroke = mouseEvent 的实际落笔函数)
for yy in range(20, th, 60):
    win.paint_brush_stroke(QPoint(10, yy), QPoint(tw // 2 - 10, yy), erase=False)
assert win.positive_mask is not None and np.any(win.positive_mask)
print("1) 正向涂抹后掩码非零像素:", int(np.count_nonzero(win.positive_mask)))

ov = win.build_mask_overlay()
assert ov is not None and np.any((ov[:, :, 1] > 0) & (ov[:, :, 3] > 0)), "叠加层应有绿色"

win.toggle_brush('brush_positive')  # 退出涂抹模式
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: SAVE_DIR)
win.train_template()
info1 = os.path.join(SAVE_DIR, "涂抹测试.info.json")
f1 = feats_of(info1)
# 语义验证: 每个特征点处的正向掩码必须是 255 (涂抹过的识别区)
pos_vals = [win.positive_mask[int(fy), int(fx)] for fx, fy in f1 if 0 <= int(fy) < th and 0 <= int(fx) < tw]
print("1) 正向训练: %d 个特征全部落在涂抹识别区: %s" % (len(f1), all(v > 0 for v in pos_vals)))
assert all(v > 0 for v in pos_vals), "特征跑到屏蔽区!"
assert f1[:, 0].max() <= tw // 2 + 10, "特征 x 超出涂抹范围"

# ---------- 2. 负向涂抹: 抠掉中心大块 (大笔刷 + 密集轨迹, 避免缝隙) ----------
win.clear_brush_masks()
win.brush_size_spin.setValue(100)   # 笔刷直径 100 (半径 50)
win.toggle_brush('brush_negative')
for yy in range(th // 2 - 80, th // 2 + 81, 40):
    win.paint_brush_stroke(QPoint(tw // 2 - 150, yy), QPoint(tw // 2 + 150, yy), erase=False)
win.toggle_brush('brush_negative')
win.train_template()
f2 = feats_of(info1)
# 语义验证: 每个特征点处的负向掩码必须是 0 (未涂抹)
neg_vals = [win.negative_mask[int(fy), int(fx)] for fx, fy in f2 if 0 <= int(fy) < win.negative_mask.shape[0] and 0 <= int(fx) < win.negative_mask.shape[1]]
print("2) 负向训练: %d 个特征全部落在未涂抹区: %s" % (len(f2), all(v == 0 for v in neg_vals)))
assert all(v == 0 for v in neg_vals), "特征落进涂抹排除区!"
assert np.any(win.negative_mask), "负向掩码应为空 (有涂抹)"

# ---------- 3. UI 渲染叠加 ----------
win.toggle_brush('brush_negative')
grab_img = win.image_label.grab().toImage()
ptr = grab_img.constBits()
arr = np.array(ptr).reshape(grab_img.height(), grab_img.bytesPerLine() // 4, 4)[:, :grab_img.width(), :]
# toImage() 是 BGRA(小端, 可能预乘 alpha): 红色叠加 -> R 通道明显高于 B/G
red_px = int(((arr[:, :, 2].astype(int) - np.maximum(arr[:, :, 0], arr[:, :, 1]).astype(int)) > 40).sum())
print("3) UI 渲染红色叠加像素:", red_px)
assert red_px > 500, "UI 渲染看不到排除区叠加"
win.toggle_brush('brush_negative')

# ---------- 4. 正向全擦光 -> 拒绝训练 ----------
win.clear_brush_masks()
win.toggle_brush('brush_positive')
win.paint_brush_stroke(QPoint(10, 10), QPoint(50, 50), erase=False)
win.paint_brush_stroke(QPoint(10, 10), QPoint(50, 50), erase=True)  # 全擦掉
status_before = win.status_indicator.text()
win.train_template()
print("4) 正向全空时训练状态:", win.status_indicator.text(), " (应为 FAIL)")
assert win.status_indicator.text() == "FAIL", "正向掩码全空应拒绝训练"

shutil.rmtree(SAVE_DIR, ignore_errors=True)
print("\n== 离屏涂抹链路全部通过 ==")
