# -*- coding: utf-8 -*-
"""离屏验证: ui.py 训练后预览控件上必须能看到特征点红点。

复现 train_template 的真实显示代码路径 (绕过 QFileDialog),
抓取 image_label 的渲染结果, 数 ROI 区域内的红点像素。
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

app = QApplication(sys.argv)
import ui

win = ui.TemplateMatchingApp()
img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"))
th, tw = img.shape[:2]

# 模拟 UI 加载训练图 + 画 ROI
win.train_image = img.copy()
win.original_image_for_display = img.copy()
from PySide6.QtCore import QRect
win.roi_rect = QRect(0, 0, tw, th)
win.class_id_input.setText("离屏验证")
win.display_image(win.original_image_for_display.copy())
app.processEvents()

# ---- 修复前的调用顺序 (复现 bug) ----
win.current_cv_image = win.original_image_for_display.copy()
win.current_cv_image[0:th, 0:tw] = np.zeros((th, tw, 3), np.uint8)  # 占位
result = win.matcher.train(
    img, (0, 0, tw, th), "离屏验证",
    {
        'pyramid_levels': [4, 8], 'feature_num': 128, 'weak_thresh': 30.0, 'strong_thresh': 60.0,
        'angle_start': 0.0, 'angle_extent': 0.0, 'angle_step': 3.0,
        'scale_start': 1.0, 'scale_end': 1.0, 'scale_step': 0.1,
    },
    os.path.join(HERE, "离屏验证输出"),
)

def red_pixels(cv_img):
    b, g, r = cv_img[:, :, 0].astype(int), cv_img[:, :, 1].astype(int), cv_img[:, :, 2].astype(int)
    return int(((r - np.maximum(b, g)) > 60).sum())

# ---- 修复后的 train_template 显示顺序: 先 clear_results_table, 再贴特征点 ----
win.clear_results_table()          # 内部会还原底图 (旧代码在这之后贴图 -> 被抹掉)
x, y, w, h = win.roi_rect.getRect()
win.current_cv_image = win.original_image_for_display.copy()
win.current_cv_image[y:y+h, x:x+w] = result['features_image']
win.display_image(win.current_cv_image)
app.processEvents()

# ---- 验证1: 显示缓冲 current_cv_image 上有红点 ----
n_buf = red_pixels(win.current_cv_image[0:th, 0:tw])
print("显示缓冲红点像素:", n_buf)
assert n_buf > 500, f"显示缓冲上没有特征点红点 ({n_buf})"

# ---- 验证2: 抓取 image_label 实际渲染的 pixmap, 数红点 ----
grab = win.image_label.grab().toImage().convertToFormat(win.image_label.grab().toImage().Format.Format_BGR888)
ptr = grab.constBits()
arr = np.array(ptr).reshape(grab.height(), grab.bytesPerLine() // 3, 3)[:, :grab.width(), :]
n_ui = int(((arr[:, :, 2].astype(int) - np.maximum(arr[:, :, 0], arr[:, :, 1]).astype(int)) > 60).sum())
print("UI 控件渲染红点像素:", n_ui)
assert n_ui > 500, f"UI 渲染结果上没有特征点红点 ({n_ui})"

import shutil
shutil.rmtree(os.path.join(HERE, "离屏验证输出"), ignore_errors=True)
print("== 离屏验证通过: 训练后 UI 预览控件能看到特征点 ==")
