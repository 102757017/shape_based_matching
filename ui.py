# ui.py

import numpy as np
import sys
import os
import time
import cv2
import traceback
import random
from typing import Dict, List, Tuple, Any, Optional
import logging
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QFormLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QStatusBar, QFileDialog, QCheckBox, QScrollArea,
    QLineEdit, QListWidget, QAbstractItemView, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
from PySide6.QtCore import Qt, QPoint, QRect, QTimer
import pprint

try:
    from matcher import Matcher, repair_mojibake, safe_file_name
except ImportError:
    print("错误: 找不到 'matcher.py' 文件。请确保它与 ui.py 在同一目录下。")
    sys.exit(1)

def excepthook(exc_type, exc_value, exc_tb):
    print(f"未捕获的主线程异常: {exc_type.__name__}: {exc_value}")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.exit(1)

sys.excepthook = excepthook

class ZoomableROIImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        self.drawing_mode = 'none' # 'none', 'roi', 'exclude_rect', 'exclude_ellipse'
        self.is_drawing = False
        self.draw_start_point = QPoint()
        self.draw_end_point = QPoint()
        
        self.zoom_factor = 1.0
        self.original_pixmap = None
        self.panning = False
        self.last_mouse_pos = QPoint()
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        
    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.update_pixmap_display()

    def update_pixmap_display(self):
        if self.original_pixmap is None:
            super().setPixmap(QPixmap())
            self.setFixedSize(0, 0)
            return
        scaled_pixmap = self.original_pixmap.scaled(
            int(self.original_pixmap.width() * self.zoom_factor),
            int(self.original_pixmap.height() * self.zoom_factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)
        self.setFixedSize(scaled_pixmap.size())

    def set_drawing_mode(self, mode: str):
        """设置当前的绘图模式并更新光标。"""
        self.drawing_mode = mode
        if mode != 'none':
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def map_label_point_to_image(self, label_point):
        if self.original_pixmap is None or self.pixmap().isNull() or self.pixmap().width() == 0: return None
        scaled_pixmap_width = self.pixmap().width()
        scaled_pixmap_height = self.pixmap().height()
        img_x = int(label_point.x() * (self.original_pixmap.width() / scaled_pixmap_width))
        img_y = int(label_point.y() * (self.original_pixmap.height() / scaled_pixmap_height))
        return QPoint(img_x, img_y)
    
    def map_image_point_to_label(self, image_point):
        if self.original_pixmap is None or self.pixmap().isNull() or self.original_pixmap.width() == 0: return None
        scaled_pixmap_width = self.pixmap().width()
        scaled_pixmap_height = self.pixmap().height()
        label_x = int(image_point.x() * (scaled_pixmap_width / self.original_pixmap.width()))
        label_y = int(image_point.y() * (scaled_pixmap_height / self.original_pixmap.height()))
        return QPoint(label_x, label_y)
    
    def wheelEvent(self, event):
        old_zoom_factor = self.zoom_factor
        mouse_on_label = event.position().toPoint() 
        if event.angleDelta().y() > 0: self.zoom_factor *= 1.1
        else: self.zoom_factor /= 1.1
        self.zoom_factor = max(0.05, min(self.zoom_factor, 10.0))
        if self.zoom_factor != old_zoom_factor:
            mouse_on_image_before_zoom = self.map_label_point_to_image(mouse_on_label)
            self.update_pixmap_display()
            if mouse_on_image_before_zoom:
                new_mouse_on_label = self.map_image_point_to_label(mouse_on_image_before_zoom)
                scroll_area = self.parent().parent()
                if isinstance(scroll_area, QScrollArea):
                    h_bar = scroll_area.horizontalScrollBar(); v_bar = scroll_area.verticalScrollBar()
                    h_bar.setValue(int(h_bar.value() + new_mouse_on_label.x() - mouse_on_label.x()))
                    v_bar.setValue(int(v_bar.value() + new_mouse_on_label.y() - mouse_on_label.y()))
            self.update()
            
    def mousePressEvent(self, event):
        if self.drawing_mode != 'none' and event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.draw_start_point = event.position().toPoint()
            self.draw_end_point = event.position().toPoint()
            self.update()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        image_pos = self.map_label_point_to_image(event.position().toPoint())
        if image_pos: self.main_window.update_status_bar_pos(image_pos.x(), image_pos.y())
        
        if self.is_drawing and event.buttons() & Qt.MouseButton.LeftButton:
            self.draw_end_point = event.position().toPoint()
            self.update()
        elif self.panning and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position().toPoint() - self.last_mouse_pos
            self.last_mouse_pos = event.position().toPoint()
            scroll_area = self.parent().parent()
            if isinstance(scroll_area, QScrollArea):
                h_bar, v_bar = scroll_area.horizontalScrollBar(), scroll_area.verticalScrollBar()
                h_bar.setValue(h_bar.value() - delta.x())
                v_bar.setValue(v_bar.value() - delta.y())

    def mouseReleaseEvent(self, event):
        if self.is_drawing and event.button() == Qt.MouseButton.LeftButton:
            start_img_pt = self.map_label_point_to_image(self.draw_start_point)
            end_img_pt = self.map_label_point_to_image(self.draw_end_point)
            
            if start_img_pt and end_img_pt:
                rect_in_image_coords = QRect(start_img_pt, end_img_pt).normalized()
                
                if self.drawing_mode == 'roi':
                    self.main_window.set_main_roi(rect_in_image_coords)
                elif self.drawing_mode == 'match_roi':
                    self.main_window.set_match_roi(rect_in_image_coords)
                elif self.drawing_mode in ['exclude_rect', 'exclude_ellipse']:
                    if self.main_window.roi_rect:
                        main_roi_tl = self.main_window.roi_rect.topLeft()
                        relative_rect = rect_in_image_coords.translated(-main_roi_tl)
                        self.main_window.add_exclusion_zone(self.drawing_mode, relative_rect.getRect())
                    else:
                        self.main_window.update_status("请先设置主ROI区域", "fail")
            
            self.is_drawing = False
            self.set_drawing_mode('none')
            self.update() # 触发重绘
            
        elif self.panning and event.button() == Qt.MouseButton.LeftButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            
    def leaveEvent(self, event): self.main_window.update_status_bar_pos(None, None)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        # 1. 绘制主ROI (绿色)
        if self.main_window.roi_rect and not self.main_window.roi_rect.isNull():
            pen = QPen(Qt.GlobalColor.green, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            start_point_scaled = self.map_image_point_to_label(self.main_window.roi_rect.topLeft())
            end_point_scaled = self.map_image_point_to_label(self.main_window.roi_rect.bottomRight())
            if start_point_scaled and end_point_scaled:
                painter.drawRect(QRect(start_point_scaled, end_point_scaled).normalized())

        # 2. 绘制匹配ROI (橙色实线, 匹配时只在该区域内找目标)
        if self.main_window.match_roi_rect and not self.main_window.match_roi_rect.isNull():
            pen = QPen(QColor(255, 165, 0), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            start_scaled = self.map_image_point_to_label(self.main_window.match_roi_rect.topLeft())
            end_scaled = self.map_image_point_to_label(self.main_window.match_roi_rect.bottomRight())
            if start_scaled and end_scaled:
                painter.drawRect(QRect(start_scaled, end_scaled).normalized())

        # 3. 绘制排除区域 (半透明红色)
        if self.main_window.roi_rect and self.main_window.exclusion_zones:
            painter.save()
            pen = QPen(QColor(255, 0, 0, 200), 1, Qt.PenStyle.DashLine)
            brush = QBrush(QColor(255, 0, 0, 100))
            painter.setPen(pen)
            painter.setBrush(brush)
            
            main_roi_tl = self.main_window.roi_rect.topLeft()
            
            for zone in self.main_window.exclusion_zones:
                zone_rect_relative = QRect(*zone['rect'])
                zone_rect_global = zone_rect_relative.translated(main_roi_tl)
                
                start_scaled = self.map_image_point_to_label(zone_rect_global.topLeft())
                end_scaled = self.map_image_point_to_label(zone_rect_global.bottomRight())

                if start_scaled and end_scaled:
                    scaled_rect = QRect(start_scaled, end_scaled).normalized()
                    if zone['type'] == 'exclude_rect':
                        painter.drawRect(scaled_rect)
                    elif zone['type'] == 'exclude_ellipse':
                        painter.drawEllipse(scaled_rect)
            painter.restore()

        # 4. 绘制当前正在画的形状 (红色虚线)
        if self.is_drawing and not self.draw_start_point.isNull() and not self.draw_end_point.isNull():
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRect(self.draw_start_point, self.draw_end_point)
            if self.drawing_mode in ['roi', 'exclude_rect']:
                painter.drawRect(rect)
            elif self.drawing_mode == 'exclude_ellipse':
                painter.drawEllipse(rect)


class TemplateMatchingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多类别模板匹配应用程序")
        self.setGeometry(100, 100, 1200, 800)
        self.train_image, self.test_image, self.current_cv_image, self.original_image_for_display, self.roi_rect = None, None, None, None, None
        self.exclusion_zones = [] # 存储 {'type': str, 'rect': [x, y, w, h]}
        self.match_roi_rect = None # 匹配阶段的有效区域 (对应 MatchParams.masks)
        self.matcher = Matcher()
        self.match_results = []     # 最近一次匹配的全部结果 (结果表格数据源)
        self.match_base_image = None # 叠加前的原始测试图, 用于表格选中时重绘高亮
        self.highlight_index = -1   # 表格中被选中、需要在图上高亮的结果下标
        self.class_color_map = {}
        self.timer = QTimer(self); self.timer.timeout.connect(self.update_runtime); self.start_time = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget); v_layout = QVBoxLayout(main_widget)
        main_layout = QHBoxLayout()
        self.create_left_panel(main_layout)
        self.create_right_panel(main_layout)
        self.create_bottom_panel()
        v_layout.addLayout(main_layout); v_layout.addLayout(self.bottom_layout)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.pos_label = QLabel("X:---- Y:---- | VAL:---"); self.status_bar.addPermanentWidget(self.pos_label)
        
    def create_left_panel(self, parent_layout):
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        self.tabs = QTabWidget()
        self.train_tab = self.create_train_tab()
        self.match_tab = self.create_match_tab()
        self.tabs.addTab(self.train_tab, "模板训练"); self.tabs.addTab(self.match_tab, "模板匹配")
        left_layout.addWidget(self.tabs); left_widget.setFixedWidth(400); parent_layout.addWidget(left_widget)

    def create_right_panel(self, parent_layout):
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        self.image_label = ZoomableROIImageLabel(self); self.image_label.setStyleSheet("background-color: black;")
        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(False); self.scroll_area.setWidget(self.image_label); self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setMinimumHeight(200)
        right_layout.addWidget(self.scroll_area)
        # 匹配结果表格: 图片预览控件的下方
        right_layout.addWidget(self.create_results_table())
        parent_layout.addWidget(right_widget)

    RESULTS_TABLE_HEADERS = ["#", "类别", "得分", "中心X", "中心Y", "角度(°)", "模板ID", "特征点", "ICP内点率", "重叠度"]

    def create_results_table(self):
        self.results_table = QTableWidget(0, len(self.RESULTS_TABLE_HEADERS))
        self.results_table.setHorizontalHeaderLabels(self.RESULTS_TABLE_HEADERS)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.doubleClicked.connect(self.copy_row_to_clipboard)
        hh = self.results_table.horizontalHeader()
        for col in range(len(self.RESULTS_TABLE_HEADERS)):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        last = len(self.RESULTS_TABLE_HEADERS) - 1
        hh.setSectionResizeMode(last, QHeaderView.ResizeMode.Stretch)
        # 点击表格行 -> 在预览图上高亮对应目标
        self.results_table.itemSelectionChanged.connect(self.on_result_row_selected)
        self.results_table.setFixedHeight(110)  # 高度随结果条数自适应, 见 _fit_table_height

        group = QGroupBox("匹配结果表 (0)")
        vbox = QVBoxLayout(group); vbox.setContentsMargins(4, 4, 4, 4); vbox.addWidget(self.results_table)
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.results_group = group
        return group

    def _fit_table_height(self):
        """表格高度自适应行数 (100~320px), 多余空间留给上方预览图。"""
        rows = self.results_table.rowCount()
        if rows == 0:
            h = 110
        else:
            h = self.results_table.horizontalHeader().height() + 2 * self.results_table.frameWidth()
            h += sum(self.results_table.rowHeight(r) for r in range(rows)) + 6
        self.results_table.setFixedHeight(max(110, min(h, 320)))

    def create_bottom_panel(self):
        self.bottom_layout = QHBoxLayout()
        self.run_button = QPushButton("运行"); self.run_button.clicked.connect(self.run_process)
        self.close_button = QPushButton("关闭应用"); self.close_button.clicked.connect(self.close)
        self.bottom_layout.addStretch(); self.bottom_layout.addWidget(QLabel("耗时:"))
        self.time_label = QLabel("0.00s"); self.bottom_layout.addWidget(self.time_label)
        self.status_indicator = QLabel("READY"); self.status_indicator.setStyleSheet("background-color: lightgray; color: black; padding: 2px 10px; border-radius: 3px;")
        self.bottom_layout.addWidget(self.status_indicator); self.bottom_layout.addSpacing(20)
        self.bottom_layout.addWidget(self.run_button); self.bottom_layout.addWidget(self.close_button)

    def create_train_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        
        img_source_group = QGroupBox("1. 图像源设置"); img_source_layout = QVBoxLayout(img_source_group)
        self.btn_load_train_img = QPushButton("加载训练图"); self.btn_load_train_img.clicked.connect(lambda: self.load_image('train'))
        img_source_layout.addWidget(self.btn_load_train_img); layout.addWidget(img_source_group)
        
        roi_group = QGroupBox("2. 区域设置"); roi_layout = QVBoxLayout(roi_group)
        main_roi_layout = QHBoxLayout()
        self.btn_create_roi = QPushButton("1. 创建主ROI"); self.btn_create_roi.clicked.connect(self.start_roi_drawing)
        self.btn_clear_roi = QPushButton("清除主ROI"); self.btn_clear_roi.clicked.connect(self.clear_roi)
        main_roi_layout.addWidget(self.btn_create_roi); main_roi_layout.addWidget(self.btn_clear_roi)
        roi_layout.addLayout(main_roi_layout)
        exclusion_layout = QHBoxLayout()
        self.btn_add_exclude_rect = QPushButton("2. 添加排除矩形"); self.btn_add_exclude_rect.clicked.connect(lambda: self.start_exclusion_drawing('exclude_rect'))
        self.btn_add_exclude_ellipse = QPushButton("3. 添加排除椭圆"); self.btn_add_exclude_ellipse.clicked.connect(lambda: self.start_exclusion_drawing('exclude_ellipse'))
        exclusion_layout.addWidget(self.btn_add_exclude_rect); exclusion_layout.addWidget(self.btn_add_exclude_ellipse)
        roi_layout.addLayout(exclusion_layout)
        self.btn_clear_exclusions = QPushButton("清除所有排除区"); self.btn_clear_exclusions.clicked.connect(self.clear_exclusion_zones)
        roi_layout.addWidget(self.btn_clear_exclusions)
        layout.addWidget(roi_group)
        
        create_group = QGroupBox("3. 模板创建参数"); create_layout = QFormLayout(create_group)
        self.class_id_input = QLineEdit("template_model"); self.class_id_input.setPlaceholderText("为模板指定一个类别名称")
        self.create_angle_start = QSpinBox(); self.create_angle_start.setRange(-180, 180); self.create_angle_start.setValue(-10)
        self.create_angle_extent = QSpinBox(); self.create_angle_extent.setRange(0, 360); self.create_angle_extent.setValue(10)
        self.create_angle_step = QDoubleSpinBox(); self.create_angle_step.setRange(0.1, 10.0); self.create_angle_step.setValue(2.0); self.create_angle_step.setSingleStep(0.1)
        self.create_scale_start = QDoubleSpinBox(); self.create_scale_start.setRange(0.1, 5.0); self.create_scale_start.setValue(0.9); self.create_scale_start.setSingleStep(0.1)
        self.create_scale_end = QDoubleSpinBox(); self.create_scale_end.setRange(0.1, 5.0); self.create_scale_end.setValue(1.1); self.create_scale_end.setSingleStep(0.01)
        self.create_scale_step = QDoubleSpinBox(); self.create_scale_step.setRange(0.01, 1.0); self.create_scale_step.setValue(0.1); self.create_scale_step.setSingleStep(0.01)
        self.create_feature_num = QSpinBox(); self.create_feature_num.setRange(10, 500); self.create_feature_num.setValue(128)
        self.create_weak_thresh = QDoubleSpinBox(); self.create_weak_thresh.setRange(0.0, 255.0); self.create_weak_thresh.setValue(30.0); self.create_weak_thresh.setSingleStep(1.0)
        self.create_strong_thresh = QDoubleSpinBox(); self.create_strong_thresh.setRange(0.0, 255.0); self.create_strong_thresh.setValue(60.0); self.create_strong_thresh.setSingleStep(1.0)
        self.pyramid_levels_input = QLineEdit("4,8"); self.pyramid_levels_input.setPlaceholderText("每层位移容差T(>=4), 逗号分隔, 例如: 4,8")
        create_layout.addRow("类别名称 (Class ID):", self.class_id_input)
        create_layout.addRow("起始角度:", self.create_angle_start); create_layout.addRow("扩展角度:", self.create_angle_extent)
        create_layout.addRow("角度步长:", self.create_angle_step); create_layout.addRow("起始尺度:", self.create_scale_start)
        create_layout.addRow("结束尺度:", self.create_scale_end); create_layout.addRow("尺度步长:", self.create_scale_step)
        create_layout.addRow("特征点数量:", self.create_feature_num); create_layout.addRow("弱阈值:", self.create_weak_thresh)
        create_layout.addRow("强阈值:", self.create_strong_thresh); create_layout.addRow("金字塔容差 (T):", self.pyramid_levels_input)
        layout.addWidget(create_group); layout.addStretch(); return tab

    def create_match_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        template_group = QGroupBox("1. 模板管理"); template_layout = QVBoxLayout(template_group)
        self.template_list_widget = QListWidget(); self.template_list_widget.setFixedHeight(120)
        # 多选模式: 选中项作为类别过滤, 全不选 = 匹配所有已加载类别
        self.template_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.template_list_widget.setToolTip("按住Ctrl多选, 匹配时只匹配选中的类别;\n一个都不选 = 匹配全部已加载类别")
        template_layout.addWidget(QLabel("已加载的模板类别 (选中项=类别过滤, 不选=全部):"))
        template_layout.addWidget(self.template_list_widget)
        btn_layout = QHBoxLayout()
        self.btn_add_template = QPushButton("添加模板类别"); self.btn_add_template.clicked.connect(self.add_template_class)
        self.btn_clear_templates = QPushButton("清空所有模板"); self.btn_clear_templates.clicked.connect(self.clear_all_templates)
        btn_layout.addWidget(self.btn_add_template); btn_layout.addWidget(self.btn_clear_templates)
        template_layout.addLayout(btn_layout)
        layout.addWidget(template_group)
        img_source_group = QGroupBox("2. 图像源"); img_source_layout = QVBoxLayout(img_source_group)
        self.btn_load_test_img = QPushButton("加载测试图"); self.btn_load_test_img.clicked.connect(lambda: self.load_image('test'))
        match_roi_layout = QHBoxLayout()
        self.btn_create_match_roi = QPushButton("创建匹配ROI"); self.btn_create_match_roi.clicked.connect(self.start_match_roi_drawing)
        self.btn_create_match_roi.setToolTip("在测试图上拖动画框, 匹配时只在该区域内找目标 (对应 MatchParams.masks)")
        self.btn_clear_match_roi = QPushButton("清除匹配ROI"); self.btn_clear_match_roi.clicked.connect(self.clear_match_roi)
        match_roi_layout.addWidget(self.btn_create_match_roi); match_roi_layout.addWidget(self.btn_clear_match_roi)
        img_source_layout.addWidget(self.btn_load_test_img); img_source_layout.addLayout(match_roi_layout)
        layout.addWidget(img_source_group)
        search_group = QGroupBox("3. 搜索与过滤参数"); search_layout = QFormLayout(search_group)
        self.search_score = QSpinBox(); self.search_score.setRange(0, 100); self.search_score.setValue(80)
        self.search_num_matches = QSpinBox(); self.search_num_matches.setRange(1, 500); self.search_num_matches.setValue(20)
        self.search_num_matches.setToolTip("对应匹配接口的 max_matches, 在 C++ 侧截断结果数量")
        self.max_overlap_spinbox = QDoubleSpinBox(); self.max_overlap_spinbox.setRange(0.0, 1.0); self.max_overlap_spinbox.setValue(1.0); self.max_overlap_spinbox.setSingleStep(0.05)
        self.max_overlap_spinbox.setToolTip("重叠度上限: 结果被其它目标遮挡超过该比例(交/自身面积, 非IoU)则丢弃;\n1.0 = 不过滤, 0.5 = 一半以上被盖住就不要")
        self.enable_nms_checkbox = QCheckBox("启用NMS"); self.enable_nms_checkbox.setChecked(True)
        self.nms_threshold_spinbox = QDoubleSpinBox(); self.nms_threshold_spinbox.setRange(0.0, 1.0); self.nms_threshold_spinbox.setValue(0.5); self.nms_threshold_spinbox.setSingleStep(0.05)
        self.enable_nms_checkbox.stateChanged.connect(lambda state: self.nms_threshold_spinbox.setEnabled(state == Qt.CheckState.Checked.value))
        self.enable_icp_checkbox = QCheckBox("启用ICP精修"); self.enable_icp_checkbox.setChecked(True)
        self.enable_icp_checkbox.setToolTip("关闭后跳过ICP精修, 返回粗匹配位置 (更快), 结果中 fitness = -1")
        self.min_fitness_spinbox = QDoubleSpinBox(); self.min_fitness_spinbox.setRange(0.0, 1.0); self.min_fitness_spinbox.setValue(0.0); self.min_fitness_spinbox.setSingleStep(0.05)
        self.min_fitness_spinbox.setToolTip("ICP 内点率(0~1)下限, 低于该值的结果被丢弃;\n0 = 不过滤。仅启用ICP精修时生效")
        self.enable_icp_checkbox.stateChanged.connect(lambda state: self.min_fitness_spinbox.setEnabled(state == Qt.CheckState.Checked.value))
        search_layout.addRow("最小分数:", self.search_score); search_layout.addRow("最大匹配数量:", self.search_num_matches)
        search_layout.addRow("重叠度上限:", self.max_overlap_spinbox)
        search_layout.addRow(self.enable_nms_checkbox); search_layout.addRow("NMS 阈值:", self.nms_threshold_spinbox)
        search_layout.addRow(self.enable_icp_checkbox); search_layout.addRow("精修内点率下限:", self.min_fitness_spinbox)
        layout.addWidget(search_group); layout.addStretch(); return tab

    def add_template_class(self):
        # 同时接受 yaml (检测器模板) 与 json (模板信息文件), matcher 内部会自动补齐另一份
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择一个或多个模板文件", "",
                                                     "模板文件 (*.yaml *.json);;YAML Files (*.yaml);;JSON Files (*.json)")
        if not file_paths: return
        for file_path in file_paths:
            self.update_status(f"正在添加 {os.path.basename(file_path)}...", "running")
            try:
                class_id, _ = self.matcher.add_template_class(file_path)
                if not self.template_list_widget.findItems(class_id, Qt.MatchFlag.MatchExactly):
                    self.template_list_widget.addItem(class_id)
                self.update_status(f"模板类别 '{class_id}' 添加成功", "success")
            except Exception as e:
                self.update_status(f"添加失败: {e}", "fail"); traceback.print_exc()

    def clear_all_templates(self):
        self.matcher.clear()
        self.template_list_widget.clear()
        self.class_color_map.clear()
        self.clear_results_table()
        self.update_status("所有模板已清空", "success")

    # ---------------- 匹配结果表格 ----------------

    def clear_results_table(self):
        """清空结果表格, 并撤销预览图上的结果叠加。"""
        self.results_table.blockSignals(True)
        self.results_table.setRowCount(0)
        self.results_table.clearSelection()
        self.results_table.blockSignals(False)
        self.match_results = []
        self.match_base_image = None
        self.highlight_index = -1
        self._fit_table_height()
        self._set_results_group_title(0)
        if self.original_image_for_display is not None:
            self.current_cv_image = self.original_image_for_display.copy()
            self.display_image(self.current_cv_image)

    def _set_results_group_title(self, count: int):
        if hasattr(self, 'results_group'): self.results_group.setTitle(f"匹配结果表 ({count})")

    def populate_results_table(self, results: List[Dict]):
        """把匹配结果填入表格 (顺序与预览图中绘制顺序一致)。"""
        table = self.results_table
        self.match_results = list(results)
        self.match_base_image = self.test_image.copy() if self.test_image is not None else None
        table.blockSignals(True)
        table.setRowCount(0)
        for idx, m in enumerate(self.match_results):
            row = table.rowCount(); table.insertRow(row)
            fitness = m.get('fitness', -1.0)
            icp_refined = m.get('icp_refined', False)
            fit_text = f"{fitness:.3f}" if (icp_refined and fitness >= 0) else "--"
            cells = [
                str(idx + 1),
                str(m.get('class_id', '')),
                f"{float(m.get('score', 0.0)):.1f}",
                f"{float(m.get('refined_x', 0.0)):.1f}",
                f"{float(m.get('refined_y', 0.0)):.1f}",
                f"{float(m.get('refined_angle', 0.0)):.1f}",
                str(m.get('template_id', '')),
                str(len(m.get('matched_features', []))),
                fit_text,
                f"{float(m.get('overlap', 0.0)):.3f}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip(f"{self.RESULTS_TABLE_HEADERS[col]}: {text}")
                if col == 2:  # 得分: 高分标绿
                    try:
                        if float(text) >= 90.0: item.setForeground(QColor(0, 130, 0))
                        elif float(text) >= 50.0: item.setForeground(QColor(190, 120, 0))
                        else: item.setForeground(QColor(200, 0, 0))
                    except ValueError: pass
                if col == 8 and fit_text != "--":  # ICP 内点率偏低标红
                    try:
                        if float(fit_text) < 0.5: item.setForeground(QColor(200, 0, 0))
                    except ValueError: pass
                table.setItem(row, col, item)
            table.setRowHeight(row, 20)
        table.blockSignals(False)
        self.highlight_index = -1
        self._fit_table_height()
        self._set_results_group_title(len(self.match_results))
        # 按类别分组再按得分排序不方便阅读, 这里保持 C++/置信度降序, 与图上标注顺序一致

    def on_result_row_selected(self):
        """选中表格行时, 在预览图上高亮对应的那个目标。"""
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.match_results):
            if self.highlight_index != -1:
                self.highlight_index = -1
                self.redraw_results_overlay()
            return
        if row == self.highlight_index: return
        self.highlight_index = row
        self.redraw_results_overlay()

    def redraw_results_overlay(self):
        """以原始测试图为底重绘匹配叠加, 可选高亮某一行。"""
        if self.match_base_image is None or not self.match_results: return
        overlay = self.draw_multiclass_results(
            self.match_base_image, self.match_results,
            self.search_num_matches.value(), highlight_index=self.highlight_index
        )
        self.current_cv_image = overlay
        self.display_image(overlay)

    def copy_row_to_clipboard(self, index=None):
        """双击表格 = 把该行所有字段复制成制表符分隔的文本, 可直接粘进 Excel。"""
        table = self.results_table
        if index is None:
            index = table.currentIndex()
        if not index.isValid(): return
        row = index.row()
        text = "\t".join(
            (table.item(row, col).text() if table.item(row, col) else "")
            for col in range(table.columnCount())
        )
        QApplication.clipboard().setText(text)
        self.update_status(f"已复制第 {row + 1} 行结果到剪贴板", "success")



    def match_template(self):
        if self.test_image is None: self.update_status("请先加载测试图", "fail"); return
        if not self.matcher.get_loaded_class_ids(): self.update_status("请先添加模板类别", "fail"); return
        self.update_status("开始匹配...", "running"); self.start_timer()
        try:
            # 类别过滤: 取模板列表中选中的项; 一个都不选 = 匹配全部已加载类别
            selected_classes = [item.text() for item in self.template_list_widget.selectedItems()]
            # 匹配ROI -> 场景 mask (MatchParams.masks): 非0区域才参与匹配
            match_mask = None
            if self.match_roi_rect and not self.match_roi_rect.isNull():
                match_mask = np.zeros(self.test_image.shape[:2], np.uint8)
                mx, my, mw, mh = self.match_roi_rect.getRect()
                match_mask[max(0, my):my+mh, max(0, mx):mx+mw] = 255
            # 其余过滤参数全部对应 MatchParams, 在 C++ 侧一次完成:
            #   max_matches 截断数量 / use_refine ICP开关 / min_fitness 内点率下限
            #   max_overlap 重叠度上限 / masks 场景有效区域 / fill_overlap 是否计算重叠度
            results = self.matcher.match(
                self.test_image, float(self.search_score.value()),
                selected_classes if selected_classes else None,
                self.enable_nms_checkbox.isChecked(), self.nms_threshold_spinbox.value(),
                max_matches=self.search_num_matches.value(),
                use_refine=self.enable_icp_checkbox.isChecked(),
                min_fitness=self.min_fitness_spinbox.value(),
                max_overlap=self.max_overlap_spinbox.value(),
                masks=match_mask
            )
            #pprint.pprint(results)
            # 先记录原始测试图, 再把结果同时画到预览图并填进下方表格 (数据同源)
            display_img = self.draw_multiclass_results(self.test_image, results, self.search_num_matches.value())
            self.current_cv_image = display_img; self.display_image(self.current_cv_image)
            self.populate_results_table(results)
            if results: self.update_status(f"匹配完成, 找到 {len(results)} 个目标", "success")
            else: self.update_status("未找到匹配目标", "success")
        except Exception as e:
            traceback.print_exc(); self.update_status(f"匹配失败: {e}", "fail")
        finally: self.stop_timer()

    def get_class_color(self, class_id):
        if class_id not in self.class_color_map:
            self.class_color_map[class_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        return self.class_color_map[class_id]

    def draw_multiclass_results(self, image: np.ndarray, results: List[Dict], max_matches_to_draw: int,
                                highlight_index: int = -1) -> np.ndarray:
        display_img = image.copy()
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        for i, match in enumerate(results):
            if i >= max_matches_to_draw:
                break

            highlighted = (i == highlight_index)
            class_id = match['class_id']
            color = self.get_class_color(class_id)
            if highlighted:
                color = (0, 255, 255)  # BGR 黄色: 表格选中行 -> 加粗框 + 黄色标签
            score = match.get('score', 0.0)
            angle = match.get('refined_angle', 0.0)
            center_x = match.get('refined_x', 0.0)
            center_y = match.get('refined_y', 0.0)
            # ICP 内点率 (当前匹配接口直接提供, 未精修时为 -1)
            fitness = match.get('fitness', -1.0)
            icp_refined = match.get('icp_refined', False)
            
            # 绘制外框（优先使用精修后的外框，若不存在则跳过）
            if 'refined_box_points' in match:
                points = np.array(match['refined_box_points'], dtype=np.int32)
                cv2.polylines(display_img, [points], isClosed=True, color=color, thickness=3 if highlighted else 2)
                # 使用外框最上方的点作为文本绘制位置
                text_pos = tuple(points[np.argmin(points[:, 1])])
            else:
                continue  # 缺少外框信息，跳过当前匹配
            
            # 构建标签文本：显示类别、得分、角度、ICP内点率和中心坐标 (未精修时内点率显示 --)
            fit_text = f"F:{fitness:.2f}" if icp_refined and fitness >= 0 else "F:--"
            label_text = f"{class_id} | {score:.1f} | A:{angle:.1f} | {fit_text} | ({int(center_x)},{int(center_y)})"
            cv2.putText(display_img, label_text, (text_pos[0], text_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制特征点（红色圆点）
            if 'matched_features' in match:
                for point in match['matched_features']:
                    cv2.circle(display_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
        
        return display_img

    
    def _parse_pyramid_levels(self, line_edit_widget):
        """解析金字塔容差输入。当前接口中该参数是每层的位移容差 T (像素), 每层至少为 4。"""
        text = line_edit_widget.text()
        if not text: raise ValueError("金字塔容差不能为空。")
        try: levels = [int(part.strip()) for part in text.split(',')]
        except ValueError: raise ValueError("金字塔容差格式错误，应为逗号分隔的整数 (例如: 4,8)。")
        if any(l < 4 for l in levels):
            raise ValueError("金字塔每层的位移容差 T 不能小于 4 (例如: 4,8)。")
        return levels

    def load_image(self, mode):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not file_path: return
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None: self.update_status(f"无法加载图像: {file_path}", "fail"); return
        self.clear_roi()
        self.match_roi_rect = None  # 新图加载后旧匹配ROI失效
        if mode == 'train': self.train_image = image; self.tabs.setCurrentWidget(self.train_tab); self.update_status("训练图加载成功", "success")
        else: self.test_image = image; self.tabs.setCurrentWidget(self.match_tab); self.update_status("测试图加载成功", "success")
        self.clear_results_table()   # 换了图, 旧的匹配结果一并作废
        self.current_cv_image = image.copy(); self.original_image_for_display = image.copy()
        self.display_image(self.current_cv_image); self.image_label.zoom_factor = 1.0; self.image_label.update_pixmap_display()

    def display_image(self, cv_img):
        if cv_img is None: self.image_label.setPixmap(QPixmap()); return
        h, w, ch = cv_img.shape; q_img = QImage(cv_img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img)); self.image_label.update()
        
    def set_main_roi(self, rect: QRect):
        self.roi_rect = rect
        self.clear_exclusion_zones() # 清除旧的排除区，因为它们是相对于旧ROI的
        self.update_status("主ROI区域已选择", "success")
        self.image_label.update()

    def set_match_roi(self, rect: QRect):
        self.match_roi_rect = rect
        self.update_status("匹配ROI已选择, 匹配时只在该区域内找目标", "success")
        self.image_label.update()

    def start_match_roi_drawing(self):
        if self.test_image is None: self.update_status("请先加载测试图", "fail"); return
        self.image_label.set_drawing_mode('match_roi')
        self.update_status("请在测试图上拖动鼠标绘制匹配ROI", "running")

    def clear_match_roi(self):
        self.match_roi_rect = None
        self.update_status("匹配ROI已清除", "success")
        self.image_label.update()

    def add_exclusion_zone(self, zone_type: str, rect_tuple: Tuple[int, int, int, int]):
        self.exclusion_zones.append({'type': zone_type, 'rect': list(rect_tuple)})
        self.update_status(f"已添加一个排除区域. 总数: {len(self.exclusion_zones)}", "success")
        self.image_label.update()

    def start_roi_drawing(self):
        if self.train_image is None: self.update_status("请先加载训练图像", "fail"); return
        self.image_label.set_drawing_mode('roi')
        self.update_status("请在图像上拖动鼠标绘制主ROI", "running")

    def start_exclusion_drawing(self, mode: str):
        if self.train_image is None: self.update_status("请先加载训练图像", "fail"); return
        if self.roi_rect is None: self.update_status("请先创建主ROI区域", "fail"); return
        self.image_label.set_drawing_mode(mode)
        self.update_status(f"请在主ROI内绘制排除区域 ({mode.split('_')[1]})", "running")

    def clear_roi(self):
        self.roi_rect = None
        self.clear_exclusion_zones()
        if self.original_image_for_display is not None:
            self.current_cv_image = self.original_image_for_display.copy()
            self.display_image(self.current_cv_image)
        self.update_status("主ROI已清除", "success")
        self.image_label.update()
        
    def clear_exclusion_zones(self):
        self.exclusion_zones.clear()
        self.update_status("所有排除区域已清除", "success")
        self.image_label.update()

    def run_process(self):
        if self.tabs.currentWidget() == self.train_tab: self.train_template()
        else: self.match_template()

    def train_template(self):
        if self.train_image is None: self.update_status("请先加载训练图", "fail"); return
        if self.roi_rect is None or self.roi_rect.isNull(): self.update_status("请先绘制ROI", "fail"); return
        class_id = repair_mojibake(self.class_id_input.text().strip())
        if not class_id: self.update_status("请输入类别名称 (Class ID)", "fail"); return
        base_name = safe_file_name(class_id)
        if base_name != class_id:
            self.update_status(f"类别名含非法字符, 将以 '{base_name}' 命名模板文件", "running")
        self.update_status("开始训练...", "running"); self.start_timer()
        try:
            train_params = {
                'pyramid_levels': self._parse_pyramid_levels(self.pyramid_levels_input),
                'feature_num': self.create_feature_num.value(), 'weak_thresh': self.create_weak_thresh.value(),
                'strong_thresh': self.create_strong_thresh.value(), 'angle_start': self.create_angle_start.value(),
                'angle_extent': self.create_angle_extent.value(), 'angle_step': self.create_angle_step.value(),
                'scale_start': self.create_scale_start.value(), 'scale_end': self.create_scale_end.value(),
                'scale_step': self.create_scale_step.value(),
            }
            # 只选"目录": 文件名统一由类别名称决定, 避免文件名/目录语义混淆
            save_dir = QFileDialog.getExistingDirectory(self, "选择模板保存目录", "")
            if not save_dir: self.update_status("训练已取消", "success"); self.stop_timer(); return
            save_dir = repair_mojibake(save_dir)  # 修复中文目录名可能被 ANSI 层转码成乱码的问题

            result = self.matcher.train(
                self.train_image, self.roi_rect.getRect(), class_id, train_params,
                save_dir, exclusion_zones=self.exclusion_zones
            )
            if result:
                x, y, w, h = self.roi_rect.getRect()
                # 先清结果表(会把预览图还原成原始训练图), 再叠加特征点预览,
                # 否则 clear_results_table 会把刚贴上去的特征点抹掉
                self.clear_results_table()
                self.current_cv_image = self.original_image_for_display.copy()
                self.current_cv_image[y:y+h, x:x+w] = result['features_image']
                self.display_image(self.current_cv_image)
                files = [os.path.basename(result['yaml_path']), os.path.basename(result['info_path'])]
                if result.get('preview_path'): files.append(os.path.basename(result['preview_path']))
                files.append(os.path.basename(result['yaml_path']).replace('.yaml', '.jpg'))
                self.update_status(f"训练成功, 保存目录: {result['save_dir']} | {', '.join(files)}", "success")
                logging.getLogger('ui').info(f"模板产物: {result}")
        except (ValueError, RuntimeError) as e: self.update_status(f"训练失败: {e}", "fail")
        except Exception as e: traceback.print_exc(); self.update_status(f"发生未知错误: {e}", "fail")
        finally: self.stop_timer()

    def start_timer(self): self.start_time = time.time(); self.timer.start(10)
    
    def stop_timer(self):
        if self.timer.isActive(): self.timer.stop()
        if self.start_time: self.time_label.setText(f"{(time.time() - self.start_time):.2f}s"); self.start_time = None

    def update_runtime(self):
        if self.start_time: self.time_label.setText(f"{(time.time() - self.start_time):.2f}s")
    
    def update_status(self, message, status_type):
        self.status_bar.showMessage(message, 5000)
        style = "color: white; padding: 2px 10px; border-radius: 3px;"
        if status_type == "success": self.status_indicator.setText("PASS"); self.status_indicator.setStyleSheet(f"background-color: #4CAF50; {style}")
        elif status_type == "fail": self.status_indicator.setText("FAIL"); self.status_indicator.setStyleSheet(f"background-color: #F44336; {style}")
        elif status_type == "running": self.status_indicator.setText("RUN"); self.status_indicator.setStyleSheet(f"background-color: #2196F3; {style}")
        else: self.status_indicator.setText("READY"); self.status_indicator.setStyleSheet("background-color: lightgray; color: black; padding: 2px 10px; border-radius: 3px;")
    
    def update_status_bar_pos(self, x, y):
        if x is not None and self.current_cv_image is not None and 0 <= y < self.current_cv_image.shape[0] and 0 <= x < self.current_cv_image.shape[1]:
            val = self.current_cv_image[y, x]
            val_str = f"({val[0]:3d},{val[1]:3d},{val[2]:3d})" if len(self.current_cv_image.shape) > 2 else f"{val:3d}"
            self.pos_label.setText(f"X:{x:04d} Y:{y:04d} | VAL:{val_str}")
        else: self.pos_label.setText("X:---- Y:---- | VAL:---")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app = QApplication(sys.argv)
    window = TemplateMatchingApp()
    window.show()
    sys.exit(app.exec())
