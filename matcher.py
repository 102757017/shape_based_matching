# -*- coding: utf-8 -*-
"""shape_based_matching 的 Python 门面 (薄壳)。

训练 / 模板加载 / 匹配 / 几何精修的全部逻辑已下沉到 C++ (pybind11/py_matcher.h):
  - 中文路径在 C++ 侧通过 FileStorage MEMORY 模式 + UTF-8 落盘根治,
    不再需要 ASCII 临时目录搬运;
  - match() 的逐结果几何计算 (旋转外框 / 特征点 / 抓取点投影) 也在 C++ 完成。

本文件只保留三件事:
  1. pyd 模块搜索路径处理;
  2. repair_mojibake / safe_file_name (依赖 OS 编码转换, 且 ui.py 直接 import 它们);
  3. Matcher 薄壳: 透传给 C++ 的 PyMatcher, 对外签名与旧版完全一致。
"""
import logging
import re
import sys
from pathlib import Path

# --- 路径设置: 把编译产物所在目录加入 sys.path, 保证能找到 pybind 模块 ---
script_dir = Path(__file__).parent
_module_dirs = [
    script_dir / "build" / "Release",
    script_dir / "build" / "Debug",
    script_dir / "build",
    script_dir / "out" / "build" / "x64-Release",
    script_dir / "out" / "build" / "x64-Debug",
]
for _d in _module_dirs:
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

try:
    import shape_based_matching_py
except ImportError:
    logging.error("错误: 无法导入 'shape_based_matching_py' 模块, 请先编译并确认 pyd 位于 build/Release 等目录。")
    raise


# 还原后只允许出现: ASCII + 中文标点 + 常用汉字(含全角字符), 否则视为误判
_CJK_ONLY_RE = re.compile('[\\x00-\\x7F\\u3000-\\u303F\\u4E00-\\u9FFF\\uFF00-\\uFFEF]*'
                          '[\\u4E00-\\u9FFF\\u3000-\\u303F\\uFF00-\\uFFEF]+'
                          '[\\x00-\\x7F\\u3000-\\u303F\\u4E00-\\u9FFF\\uFF00-\\uFFEF]*')


def repair_mojibake(text: str) -> str:
    """修复"UTF-8 字节被 GBK/GB18030 误解码"产生的乱码。

    Qt/Windows 文件对话框经过 ANSI 层时, 中文路径可能被错误转码成
    "鏂板缓..." 这类乱码, 再拿去建目录/写文件就会得到乱码文件名。
    该转换是可逆的: 乱码串按 GBK 编码回原始字节, 再按 UTF-8 解码即还原。

    只有满足"能完整往返"且"不引入新的 ASCII 字符"才认为确实是乱码,
    否则连"新建文件夹"这类正常中文也会被误伤(某些 GBK 字节恰好是合法 UTF-8, 会多出一个 '\\')。
    """
    if not text:
        return text
    try:
        repaired = text.encode('gbk').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text                      # 回不去, 原样返回
    if repaired == text:
        return text
    if not _CJK_ONLY_RE.fullmatch(repaired):
        return text                      # 还原后混进了非中文/非 ASCII 字符, 多半是误判
    return repaired


def safe_file_name(name: str, default: str = 'template') -> str:
    """把类别名转成安全的模板文件名(修复乱码 + 过滤 Windows 非法字符)。"""
    cleaned = repair_mojibake(str(name).strip())
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', cleaned).strip().strip('.')
    return cleaned or default


class Matcher:
    """内部核心匹配算法引擎的薄壳, 逻辑在 C++ (shape_based_matching_py.PyMatcher)。

    对外签名与旧版完全一致:
      - train() 训练并保存单类别模板 (产物: xxx.jpg / xxx.yaml / xxx.info.json / xxx.preview.png)
      - add_template_class() 加载已有模板类别
      - match() 一体化匹配 (类别过滤 + NMS + ICP 精修 + 几何投影, 全部在 C++ 侧完成)
      - clear() / get_loaded_class_ids()
    """

    def __init__(self):
        self._impl = shape_based_matching_py.PyMatcher()

    def clear(self):
        """清空所有已加载的模板和检测器实例。"""
        self._impl.clear()
        logging.info("Matcher已清空。")

    def train(self, train_image, roi_rect_tuple, class_id, train_params,
              save_dir=".", exclusion_zones=None, positive_mask=None, negative_mask=None):
        """训练并保存单个模板文件。所有产物写到 save_dir, 文件名前缀为类别名。

        :param roi_rect_tuple: (x, y, w, h)
        :param save_dir: 模板保存目录(不是完整文件路径), 支持中文路径
        :param exclusion_zones: 排除区域字典列表,
            每项形如 {'type': 'exclude_rect'/'exclude_ellipse', 'rect': [x, y, w, h]}
        :param positive_mask: ROI 尺寸 (h,w) uint8 掩码或 None。
            正向涂抹模式: 非0处才参与特征提取, 未涂抹区域全部屏蔽;
            None 或全 0 = 默认整个 ROI 参与
        :param negative_mask: ROI 尺寸 (h,w) uint8 掩码或 None。
            负向涂抹模式: 非0处排除特征点(干扰点), 未涂抹区域全部参与
        """
        base_name = safe_file_name(str(class_id).strip())
        save_dir = repair_mojibake(str(save_dir or "."))
        return self._impl.train(
            train_image, list(roi_rect_tuple), base_name, train_params,
            save_dir, list(exclusion_zones) if exclusion_zones else None,
            positive_mask, negative_mask)

    def add_template_class(self, yaml_path, override_params=None):
        """加载模板类别。yaml_path 可传训练产物中的任意一份:
        - xxx.yaml (检测器模板文件)
        - xxx.info.json (模板信息文件)
        - xxx.json (等价按 xxx.info.json 处理)
        会自动推算同目录下另一份文件, 两份都存在才能加载。
        返回 (class_id, 最终使用的参数字典)。
        """
        return self._impl.add_template_class(str(yaml_path), override_params)

    def get_loaded_class_ids(self):
        return list(self._impl.get_loaded_class_ids())

    def get_base_template_features(self, class_id):
        """某类别训练时主模板的特征点 (ROI 坐标)。"""
        return list(self._impl.get_base_template_features(class_id))

    def match(self, image, score_threshold, class_ids_to_match=None,
              use_nms=True, nms_threshold=0.5, grasp_points_config=None,
              max_matches=0, min_fitness=0.0,
              use_refine=True, max_overlap=1.0,
              masks=None, fill_overlap=True):
        """执行模板匹配，返回包含精确外框（原始宽高）和特征点（红点）的结果列表。

        :param score_threshold: 置信度下限 (0~100, 对应 MatchParams.min_confidence)
        :param class_ids_to_match: 只匹配这些类别, None = 全部已加载类别
        :param use_nms: 是否做非极大值抑制 (对应 MatchParams.nms)
        :param nms_threshold: NMS 重叠度阈值 (对应 MatchParams.nms_overlap)
        :param grasp_points_config: {类别名: [[x, y], ...]} 自定义抓取点 (ROI 坐标)
        :param max_matches: 最多返回几个结果, 0 = 不限 (对应 MatchParams.max_matches)
        :param min_fitness: ICP 内点率下限 0~1, 低于则丢弃 (仅 use_refine 时生效)
        :param use_refine: 是否做 ICP 精修; False 时结果为粗匹配位置, fitness = -1
        :param max_overlap: 重叠度上限 0~1, 1.0 = 不过滤 (是"交/自身面积", 不是 IoU)
        :param masks: 场景 mask (与输入图像同尺寸, 非0即参与匹配); None = 整幅图
        :param fill_overlap: 是否计算结果的重叠度 overlap; 关闭可省一点时间
        """
        return self._impl.match(
            image, float(score_threshold), class_ids_to_match,
            bool(use_nms), float(nms_threshold), grasp_points_config,
            int(max_matches), float(min_fitness), bool(use_refine),
            float(max_overlap), masks, bool(fill_overlap))
