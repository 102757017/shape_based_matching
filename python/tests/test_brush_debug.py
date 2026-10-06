# -*- coding: utf-8 -*-
"""离屏调试: 用真实 QMouseEvent 序列走涂抹路径, ROI 放在图中央 (非原点),
逐环节检查 mask / overlay / 渲染结果, 定位"涂抹生效但画面无痕迹"。
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
from PySide6.QtCore import Qt, QPoint, QPointF, QEvent, QRect
from PySide6.QtGui import QMouseEvent

app = QApplication(sys.argv)
import ui

win = ui.TemplateMatchingApp()
img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"))
th, tw = img.shape[:2]
win.train_image = img.copy()
win.original_image_for_display = img.copy()
win.current_cv_image = img.copy()
win.set_main_roi(QRect(100, 80, 400, 300))   # ROI 在图中央, 不是 (0,0)
win.display_image(win.original_image_for_display.copy())   # 真实 UI 加载图后 pixmap 已设置
app.processEvents()

lbl = win.image_label
print("zoom_factor:", lbl.zoom_factor, " main_window is win:", lbl.main_window is win)


def send_mouse(type_, btn, buttons, label_pos):
    ev = QMouseEvent(type_, QPointF(label_pos), btn, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(lbl, ev)


def to_label(img_x, img_y):
    """图像坐标 -> label 控件坐标 (zoom=1 时相同)"""
    return QPoint(img_x, img_y)


# 进入正向涂抹模式
win.toggle_brush('brush_positive')
print("drawing_mode:", lbl.drawing_mode)

# 真实鼠标事件序列: 按下 -> 移动 -> 释放 (图内坐标 150..450, y=150)
send_mouse(QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton,
           Qt.MouseButton.LeftButton, to_label(150, 150))
print("按下后 brushing:", lbl.brushing, " mask 非零:", 0 if win.positive_mask is None else int(np.count_nonzero(win.positive_mask)))
for x in range(170, 451, 20):
    send_mouse(QEvent.Type.MouseMove, Qt.MouseButton.NoButton,
               Qt.MouseButton.LeftButton, to_label(x, 150))
send_mouse(QEvent.Type.MouseButtonRelease, Qt.MouseButton.LeftButton,
           Qt.MouseButton.NoButton, to_label(450, 150))
print("释放后 brushing:", lbl.brushing, " mode:", lbl.drawing_mode, "(应保持涂抹模式)")
print("positive_mask 非零:", int(np.count_nonzero(win.positive_mask)))

# overlay 检查
ov = win.build_mask_overlay()
print("overlay 非零像素:", int((ov[:, :, 3] > 0).sum()))
assert ov is not None and (ov[:, :, 3] > 0).sum() > 500

# 渲染检查
app.processEvents()
grab_img = lbl.grab().toImage()
ptr = grab_img.constBits()
arr = np.array(ptr).reshape(grab_img.height(), grab_img.bytesPerLine() // 4, 4)[:, :grab_img.width(), :]
green_px = int(((arr[:, :, 1].astype(int) - np.maximum(arr[:, :, 0], arr[:, :, 2]).astype(int)) > 40).sum())
print("渲染绿色叠加像素:", green_px)

# 无叠加时的基线渲染 (清空 mask 再 grab)
win.clear_brush_masks()
app.processEvents()
grab2 = lbl.grab().toImage()
ptr2 = grab2.constBits()
arr2 = np.array(ptr2).reshape(grab2.height(), grab2.bytesPerLine() // 4, 4)[:, :grab2.width(), :]
diff = (np.abs(arr.astype(int) - arr2.astype(int)).sum(axis=2) > 30).sum()
print("有/无叠加渲染差异像素:", diff)
print("== 完成 ==")
