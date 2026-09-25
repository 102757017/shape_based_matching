import os
import sys
import re
import json
import shutil
import tempfile
import cv2
import numpy as np
import traceback
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path
import math


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


def write_image(image: np.ndarray, path: Union[str, Path]) -> bool:
    """写图片(规避 cv2.imwrite 在 Windows 上对非 ASCII 路径静默失败的问题)。

    cv2.imwrite 对中文路径会直接返回 False 且不落地任何文件, 因此改成
    imencode 拿到字节流后用 Python 的 open('wb') 写出(UTF-8 路径没问题)。
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
        ext = '.png'
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        return False
    with open(str(path), 'wb') as f:
        f.write(buffer.tobytes())
    return True


def _ascii_tmp_dir() -> Path:
    """在 %TEMP% 下创建一个纯 ASCII 路径的临时目录。

    OpenCV 的 FileStorage(readClasses/writeClasses) 在 Windows 上用 fopen + 当前 ANSI 代码页
    打开路径, 一旦路径里有中文就会直接失败 (日志: Can't open file ... in write mode),
    即使不报错也拿不到文件。所以 C++ 侧的读写都先落到这个 ASCII 目录, 再在 Python 层搬运。
    """
    return Path(tempfile.mkdtemp(prefix='sbm_tmpl_'))


_TMP_YAML_NAME = 'sbm_template.yaml'  # 全 ASCII, 保证 OpenCV FileStorage 能打开


def write_detector_classes(detector, save_dir: Path, base_name: str) -> Path:
    """调用 C++ writeClasses 写 YAML, 并把它搬到真正的保存目录。返回目标 yaml 路径。

    C++ 侧会按 "format % class_id" 拼接文件名, 所以先用 ASCII 的中间名落盘, 再重命名成
    "<类别名>.yaml" 搬进目标目录 —— 这样即使类别名/目录名是中文也能正常保存。
    """
    target = Path(save_dir) / f"{base_name}.yaml"
    if target.exists():
        target.unlink()
    tmp_dir = _ascii_tmp_dir()
    try:
        detector.writeClasses(str(tmp_dir / _TMP_YAML_NAME))
        tmp_yaml = tmp_dir / _TMP_YAML_NAME
        if not tmp_yaml.exists():
            raise RuntimeError(f"C++ 侧未写出 YAML 模板文件, 期望: {tmp_yaml}")
        shutil.move(str(tmp_yaml), str(target))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return target


def read_detector_classes(detector, class_id: str, yaml_path: Path):
    """调用 C++ readClasses 读 YAML, 兼容中文保存目录与中文类别名。

    C++ 侧用 "format % class_id" 拼文件名, 中文类别名会让 FileStorage 打不开文件。
    这里把 YAML 拷到 ASCII 临时目录, 并把 format 直接传成这个 ASCII 绝对路径(不含 %s,
    cv::format 会原样返回), 于是文件能打开, 类别名则以 YAML 里记录的 class_id 为准。
    """
    yaml_path = Path(yaml_path)
    tmp_dir = _ascii_tmp_dir()
    try:
        tmp_yaml = tmp_dir / _TMP_YAML_NAME
        shutil.copyfile(str(yaml_path), str(tmp_yaml))
        detector.readClasses([class_id], str(tmp_yaml))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# --- 路径设置: 把编译产物所在目录加入 sys.path, 保证能找到 pybind 模块 ---
script_dir = Path(__file__).parent  # 使用 pathlib
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

class Matcher:
    """内部核心匹配算法引擎，支持多类别模板管理。

    兼容当前 pybind 接口 (MatchParams 版 match / Match.transform / Match.angle 等):
      - match() 使用 MatchParams 一体化匹配: 类别过滤 + NMS + ICP 精修都在 C++ 侧完成
      - 几何精修直接使用 Match.transform (2x3, 训练图坐标系 -> 场景图坐标系)，
        与手动 refine() 增量矩阵结果一致(已验证误差 < 1e-3 像素)
    """
    def __init__(self):
        self.detector = None
        self.detector_params = None
        self.class_info_map = {}
        self.base_template_features_map = {}

    def clear(self):
        """清空所有已加载的模板和检测器实例。"""
        self.detector = None
        self.detector_params = None
        self.class_info_map.clear()
        self.base_template_features_map.clear()
        logging.info("Matcher已清空。")

    def _initialize_detector(self, params: Dict):
        if self.detector is None:
            if not params: raise ValueError("首次加载模板时必须提供检测器参数。")
            self.detector_params = params
            pyramid_levels = self._parse_pyramid_levels(params['pyramid_levels'])
            self.detector = shape_based_matching_py.Detector(
                params['feature_num'], pyramid_levels,
                params['weak_thresh'], params['strong_thresh']
            )
            logging.info(f"Detector已使用参数初始化: {params}")

    def _parse_pyramid_levels(self, levels_list):
        # 当前接口中该参数是金字塔每层的"位移容差 T"(像素), 每层至少为 4
        if not levels_list or not all(isinstance(l, int) and l >= 4 for l in levels_list):
            raise ValueError("金字塔层级参数必须是一个列表, 且每层的位移容差 T 为不小于 4 的整数 (例如 [4, 8])。")
        return levels_list

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        对输入图像进行预处理。
        :param image: 待处理的图像
        :return: 处理后的图像
        """
        processed_image = cv2.medianBlur(image, 3)  #中值滤波
        return processed_image

    @staticmethod
    def _collapse_range(start: float, end: float) -> List[float]:
        """把 [start, end] 范围收敛为 pybind 接口要求的形式。

        当前 C++ 接口在范围列表有 2 个元素时断言 range[1] > range[0] (严格大于)，
        因此"不变化的范围"(如 scale 1.0~1.0) 必须收敛成单元素列表。
        """
        if end > start:
            return [start, end]
        return [start]

    def train(self, train_image, roi_rect_tuple, class_id, train_params, save_dir: Union[str, Path] = ".", exclusion_zones: List[Dict] = None):
        """
        训练并保存单个模板文件。

        所有产物 (xxx.jpg / xxx.yaml / xxx.info.json / xxx.preview.png) 都写到同一个 save_dir,
        文件名前缀统一为类别名 (class_id)。save_dir 可以是相对路径, 相对当前工作目录解析。
        :param save_dir: 模板保存目录(不是完整文件路径)
        :param exclusion_zones: 一个包含排除区域信息的字典列表。
                                每个字典形如 {'type': 'exclude_rect'/'exclude_ellipse', 'rect': [x, y, w, h]}
        """
        temp_matcher = Matcher() # 使用临时实例以不影响当前会话
        try:
            detector_init_params = {
                'feature_num': train_params['feature_num'], 'pyramid_levels': train_params['pyramid_levels'],
                'weak_thresh': train_params['weak_thresh'], 'strong_thresh': train_params['strong_thresh']
            }
            temp_matcher._initialize_detector(detector_init_params)

            x, y, w, h = roi_rect_tuple
            if w <= 0 or h <= 0: raise ValueError("ROI区域为空或无效。")
            roi_image = train_image[y:y+h, x:x+w]

            # 统一: 修复可能的乱码路径 + 目录只认目录, 所有产物都落在这里
            base_name = safe_file_name(class_id)
            class_id = base_name  # C++ 内部类名 = 文件名前缀, 保证两边一致
            save_dir = Path(repair_mojibake(str(save_dir or ".")))
            save_dir.mkdir(parents=True, exist_ok=True)
            img_path = save_dir / f"{base_name}.jpg"
            if roi_image.size == 0: raise ValueError("ROI区域为空或无效。")
            if not write_image(roi_image, img_path):
                raise RuntimeError(f"ROI 图片写入失败, 请检查保存目录权限或路径: {img_path}")

            diagonal = np.sqrt(w**2 + h**2)
            max_scale = train_params.get('scale_end', 1.0)
            padding = int(diagonal * max_scale * 1.5) + 50
            padded_shape = (h + 2 * padding, w + 2 * padding, 3) if len(roi_image.shape) == 3 else (h + 2 * padding, w + 2 * padding)
            padded_img = np.zeros(padded_shape, dtype=np.uint8)
            padded_img[padding:padding + h, padding:padding + w] = roi_image

            # 创建基础掩码 (全白)
            padded_mask = np.zeros(padded_shape[0:2], np.uint8)
            padded_mask[padding:padding + h, padding:padding + w] = 255

            # 在掩码上绘制排除区域 (全黑)
            if exclusion_zones:
                for zone in exclusion_zones:
                    # 将ROI内坐标转换为padded_mask的坐标
                    zone_rect = zone['rect']
                    ex, ey, ew, eh = zone_rect

                    # 排除区在 padded_mask 中的左上角坐标
                    px, py = ex + padding, ey + padding

                    if zone['type'] == 'exclude_rect':
                        # 绘制实心黑色矩形
                        cv2.rectangle(padded_mask, (px, py), (px + ew, py + eh), (0), thickness=-1)
                    elif zone['type'] == 'exclude_ellipse':
                        # 绘制实心黑色椭圆
                        center = (px + ew // 2, py + eh // 2)
                        axes = (ew // 2, eh // 2)
                        cv2.ellipse(padded_mask, center, axes, 0, 0, 360, (0), thickness=-1)

            producer = shape_based_matching_py.shapeInfo_producer(padded_img, padded_mask)
            # 角度/尺度范围适配当前接口断言 (2 元素时要求严格递增, 单点范围用 1 元素列表)
            angle_start = float(train_params['angle_start'])
            angle_extent = float(train_params['angle_extent'])
            producer.angle_range = self._collapse_range(angle_start, angle_start + angle_extent)
            producer.angle_step = float(train_params['angle_step'])
            producer.scale_range = self._collapse_range(float(train_params['scale_start']), float(train_params['scale_end']))
            producer.scale_step = float(train_params['scale_step'])
            if producer.angle_step <= 0 or producer.scale_step <= 0:
                raise ValueError("angle_step / scale_step 必须大于 0。")
            producer.produce_infos()
            if not producer.infos: raise RuntimeError("未能生成任何形状信息 (infos)。")

            current_template_info = {}
            base_template_features_local = []
            num_templates_added = 0

            center_x_padded = padded_img.shape[1] / 2.0; center_y_padded = padded_img.shape[0] / 2.0
            rotation_center = shape_based_matching_py.CV_Point2f(center_x_padded, center_y_padded)
            infos_by_scale = {}
            for info in producer.infos:
                scale_key = round(info.scale, 3)
                if scale_key not in infos_by_scale: infos_by_scale[scale_key] = []
                infos_by_scale[scale_key].append(info)
            if not infos_by_scale: raise RuntimeError("没有可用的尺度信息。")
            main_scale = min(infos_by_scale.keys(), key=lambda x: abs(x - 1.0))
            main_scale_group = sorted(infos_by_scale[main_scale], key=lambda x: abs(x.angle))
            if not main_scale_group: raise RuntimeError("主尺度组为空。")

            main_base_info = main_scale_group[0]
            adjusted_feature_num = max(10, min(int(train_params['feature_num'] * main_base_info.scale), train_params['feature_num']))
            rotated_scaled_img = producer.src_of(main_base_info)
            rotated_scaled_mask = producer.mask_of(main_base_info)
            main_base_templ_id = temp_matcher.detector.addTemplate(rotated_scaled_img, class_id, rotated_scaled_mask, adjusted_feature_num)

            if main_base_templ_id != -1:
                base_template_obj = temp_matcher.detector.getTemplates(class_id, main_base_templ_id)[0]
                for feat in base_template_obj.features:
                    x_in_roi = feat.x + base_template_obj.tl_x - padding
                    y_in_roi = feat.y + base_template_obj.tl_y - padding
                    base_template_features_local.append([x_in_roi, y_in_roi])
                current_template_info[main_base_templ_id] = {'angle': main_base_info.angle, 'scale': main_base_info.scale}
                num_templates_added += 1
                base_angle = main_base_info.angle
                for info in main_scale_group[1:]:
                    templ_id = temp_matcher.detector.addTemplate_rotate(class_id, main_base_templ_id, info.angle - base_angle, rotation_center)
                    if templ_id != -1: current_template_info[templ_id] = {'angle': info.angle, 'scale': info.scale}; num_templates_added += 1
            for scale_key in sorted(infos_by_scale.keys()):
                if scale_key == main_scale: continue
                scale_group = sorted(infos_by_scale[scale_key], key=lambda x: abs(x.angle))
                if not scale_group: continue
                scale_base_info = scale_group[0]
                adjusted_feature_num = max(10, min(int(train_params['feature_num'] * scale_base_info.scale), train_params['feature_num']))
                rotated_scaled_img = producer.src_of(scale_base_info)
                rotated_scaled_mask = producer.mask_of(scale_base_info)
                scale_base_templ_id = temp_matcher.detector.addTemplate(rotated_scaled_img, class_id, rotated_scaled_mask, adjusted_feature_num)
                if scale_base_templ_id != -1:
                    current_template_info[scale_base_templ_id] = {'angle': scale_base_info.angle, 'scale': scale_base_info.scale}; num_templates_added += 1
                    base_angle = scale_base_info.angle
                    for info in scale_group[1:]:
                        templ_id = temp_matcher.detector.addTemplate_rotate(class_id, scale_base_templ_id, info.angle - base_angle, rotation_center)
                        if templ_id != -1: current_template_info[templ_id] = {'angle': info.angle, 'scale': info.scale}; num_templates_added += 1
            if num_templates_added == 0: raise RuntimeError("未能成功添加任何模板。")

            display_roi_with_features = roi_image.copy()
            if base_template_features_local:
                for x_feat, y_feat in base_template_features_local:
                    if 0 <= x_feat < w and 0 <= y_feat < h:
                        cv2.circle(display_roi_with_features, (int(x_feat), int(y_feat)), 2, (0, 0, 255), -1)

            # C++ 侧的 YAML 经由 ASCII 临时目录搬运, 才能在中文保存目录里正确落盘
            yaml_path = write_detector_classes(temp_matcher.detector, save_dir, base_name)
            info_path = save_dir / f"{base_name}.info.json"
            preview_path = save_dir / f"{base_name}.preview.png"

            saved_data = {
                "training_params": detector_init_params,
                "templates": current_template_info,
                "base_template_features": base_template_features_local,
                "original_w": w,  # 保存原始 ROI 宽
                "original_h": h,  # 保存原始 ROI 高
                "padding": padding # 保存训练时的填充偏移
            }
            with info_path.open('w', encoding='utf-8') as f:
                json.dump(saved_data, f, indent=4)

            # 保存模板预览图 (ROI + 特征点标注), 失败不影响训练结果
            try:
                if not write_image(display_roi_with_features, preview_path):
                    logging.warning(f"预览图保存失败: {preview_path}")
                    preview_path = None
            except Exception as pe:
                logging.warning(f"预览图保存异常: {pe}")
                preview_path = None

            return {
                'yaml_path': str(yaml_path),  # 返回字符串以保持兼容性
                'info_path': str(info_path),
                'preview_path': str(preview_path) if preview_path else None,
                'save_dir': str(save_dir),
                'base_name': base_name,
                'features_image': display_roi_with_features,
                'template_count': num_templates_added
            }
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"训练过程中发生错误: {e}") from e

    def add_template_class(self, yaml_path: str, override_params: Dict = None):
        """加载模板类别。yaml_path 可传训练产物中的任意一份文件:
        - xxx.yaml (检测器模板文件)
        - xxx.info.json (模板信息文件)
        - xxx.json (等价按 xxx.info.json 处理)
        会自动推算同目录下另一份文件, 两份都存在才能加载。
        """
        path_obj = Path(yaml_path)
        suffix = path_obj.suffix.lower()
        if suffix == '.json':
            name = path_obj.name
            if name.endswith('.info.json'):
                base = name[:-len('.info.json')]
            else:
                base = path_obj.stem
            yaml_path_obj = path_obj.parent / f"{base}.yaml"
            info_path = path_obj.parent / f"{base}.info.json"
        else:
            yaml_path_obj = path_obj
            info_path = yaml_path_obj.with_suffix('.info.json')

        if not yaml_path_obj.exists() or not info_path.exists():
            raise FileNotFoundError(f"模板文件或信息文件未找到: {yaml_path_obj} / {info_path}")

        try:
            # 使用 Path 的属性
            class_id = yaml_path_obj.stem  # 获取文件名（不含扩展名）

            if class_id in self.class_info_map:
                logging.warning(f"类别 '{class_id}' 已存在，将被覆盖。")

            # 使用 Path.open() 读取 JSON
            with info_path.open('r', encoding='utf-8') as f:
                data = json.load(f)

            saved_params = data.get("training_params", {})
            final_params = self.detector_params.copy() if self.detector_params else {}
            final_params.update(saved_params)
            if override_params:
                final_params.update(override_params)

            if any(p not in final_params for p in ['pyramid_levels', 'feature_num', 'weak_thresh', 'strong_thresh']):
                raise ValueError(f"类别 '{class_id}' 的参数不完整。")

            self._initialize_detector(final_params)

            # 读取走 ASCII 临时目录, 兼容中文保存目录/中文类别名
            read_detector_classes(self.detector, class_id, yaml_path_obj)

            self.base_template_features_map[class_id] = data.get("base_template_features", [])
            self.class_info_map[class_id] = {
                "templates": {int(k): v for k, v in data.get("templates", {}).items()},
                "original_w": data.get("original_w"),
                "original_h": data.get("original_h"),
                "padding": data.get("padding", 0)
            }

            logging.info(f"模板类别 '{class_id}' 加载成功。")
            return class_id, final_params
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"加载模板类别 '{yaml_path_obj}' 失败: {e}") from e

    def get_loaded_class_ids(self) -> List[str]:
        return list(self.class_info_map.keys())

    def match(self, image, score_threshold, class_ids_to_match: Union[str, List[str]] = None,
              use_nms=True, nms_threshold=0.5, grasp_points_config: Dict[str, List[List[float]]] = None,
              max_matches: int = 0, min_fitness: float = 0.0,
              use_refine: bool = True, max_overlap: float = 1.0,
              masks: np.ndarray = None, fill_overlap: bool = True):
        """
        执行模板匹配，返回包含精确外框（原始宽高）和特征点（红点）的结果列表。

        使用当前 pybind 接口的 MatchParams 一体化匹配 (全部字段均已透传):
          - 类别过滤 / 重叠度过滤 / NMS / ICP 精修均在 C++ 侧完成 (过滤顺序见 MatchParams 注释)
          - 精修后的几何变换直接取 Match.transform (2x3, 训练图坐标系 -> 场景图坐标系)
        :param score_threshold: 置信度下限 (0~100, 对应 MatchParams.min_confidence)
        :param class_ids_to_match: 只匹配这些类别, None = 全部已加载类别 (对应 MatchParams.class_ids)
        :param use_nms: 是否做非极大值抑制 (对应 MatchParams.nms)
        :param nms_threshold: NMS 重叠度阈值 (对应 MatchParams.nms_overlap)
        :param max_matches: 最多返回几个结果, 0 = 不限 (对应 MatchParams.max_matches)
        :param min_fitness: ICP 内点率下限 0~1, 低于则丢弃 (对应 MatchParams.min_fitness, 仅 use_refine 时生效)
        :param use_refine: 是否做 ICP 精修 (对应 MatchParams.use_refine);
                           False 时跳过精修, 结果为粗匹配位置, fitness = -1
        :param max_overlap: 重叠度上限 0~1, 结果被其它目标遮挡超过该比例则丢弃 (对应 MatchParams.max_overlap);
                            1.0 = 不过滤, 0.5 = 一半以上被盖住就不要。注意是"交/自身面积", 不是 IoU
        :param masks: 场景 mask (与输入图像同尺寸, 非0即参与匹配), 只在该区域内匹配
                      (对应 MatchParams.masks); None = 整幅图。会自动跟随图像补边到 16 的倍数
        :param fill_overlap: 是否计算结果的重叠度 overlap (对应 MatchParams.fill_overlap);
                             关闭可省一点时间, 但结果中 overlap 恒为 0
        """
        if self.detector is None or not self.class_info_map:
            raise RuntimeError("没有加载任何模板类别，请先加载或训练模板。")

        # 1. 图像预处理：确保图像连续且步长满足 MIPP 要求 (16的倍数)
        image_to_match = self._preprocess_image(image)
        orig_h, orig_w = image_to_match.shape[:2]
        stride = 16
        need_pad = (orig_h % stride != 0 or orig_w % stride != 0)
        if need_pad:
            new_h = (orig_h + stride - 1) // stride * stride
            new_w = (orig_w + stride - 1) // stride * stride
            padded_img = np.zeros((new_h, new_w, 3) if len(image_to_match.shape) == 3 else (new_h, new_w), dtype=np.uint8)
            padded_img[0:orig_h, 0:orig_w] = image_to_match
            image_to_match = padded_img

        if not image_to_match.flags['C_CONTIGUOUS']:
            image_to_match = np.ascontiguousarray(image_to_match)

        # 场景 mask 预处理: 尺寸须与输入图一致, 并跟随图像补边到 16 的倍数 (补边区域不参与匹配)
        mask_to_use = None
        if masks is not None:
            mask_arr = np.asarray(masks)
            if mask_arr.ndim == 3:
                mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)
            if mask_arr.shape[:2] != (orig_h, orig_w):
                raise ValueError(f"masks 尺寸 {mask_arr.shape[:2]} 与输入图像尺寸 {(orig_h, orig_w)} 不一致。")
            mask_arr = (mask_arr > 0).astype(np.uint8) * 255
            if need_pad:
                mask_full = np.zeros((new_h, new_w), dtype=np.uint8)
                mask_full[0:orig_h, 0:orig_w] = mask_arr
                mask_arr = mask_full
            mask_to_use = np.ascontiguousarray(mask_arr)

        # 确定搜索范围
        search_ids = self.get_loaded_class_ids() if class_ids_to_match is None else \
                    ([class_ids_to_match] if isinstance(class_ids_to_match, str) else class_ids_to_match)

        if not search_ids: return []

        # 2. 调用核心检测器 (MatchParams 完整参数版):
        #    类别过滤 + 重叠度过滤 + NMS + ICP 精修(use_refine) 一次完成
        params = shape_based_matching_py.MatchParams()
        params.class_ids = list(search_ids)
        params.min_confidence = float(score_threshold)
        params.nms = bool(use_nms)
        params.nms_overlap = float(nms_threshold)
        params.use_refine = bool(use_refine)
        params.min_fitness = float(min_fitness) if use_refine else 0.0
        params.max_matches = int(max_matches)
        params.max_overlap = float(max_overlap)
        params.fill_overlap = bool(fill_overlap)
        if mask_to_use is not None:
            params.masks = mask_to_use
        raw_matches = self.detector.match(image_to_match, params)

        # C++ 侧默认按置信度降序返回, 这里再排一次以保险
        final_matches_raw = sorted(raw_matches, key=lambda m: m.similarity, reverse=True)

        results = []
        # 3. 遍历每个匹配结果，计算精确的红点和旋转外框
        for match in final_matches_raw:
            try:
                class_id = match.class_id
                meta = self.class_info_map.get(class_id)
                if not meta: continue

                # 获取该模板对应的训练信息（角度、缩放等）
                templ_info = meta.get("templates", {}).get(match.template_id, {})
                angle_deg = templ_info.get("angle", 0.0)
                scale = templ_info.get("scale", 1.0)
                angle_rad = np.radians(angle_deg)

                # 获取原始 ROI 尺寸和填充信息
                roi_w = meta.get("original_w", 0)
                roi_h = meta.get("original_h", 0)
                padding = meta.get("padding", 0)
                if roi_w == 0 or roi_h == 0:
                    logging.warning(f"类别 {class_id} 缺少原始尺寸信息，跳过")
                    continue

                # 计算填充图像尺寸和旋转中心
                padded_w = roi_w + 2 * padding
                padded_h = roi_h + 2 * padding
                center_x = padded_w / 2.0
                center_y = padded_h / 2.0

                # 获取模板对象（最高分辨率层级）
                templates = self.detector.getTemplates(class_id, match.template_id)
                templ0 = None
                for t in templates:
                    if hasattr(t, 'pyramid_level') and t.pyramid_level == 0:
                        templ0 = t
                        break
                if templ0 is None:
                    templ0 = templates[0]  # 降级使用第一个
                tl_x = templ0.tl_x
                tl_y = templ0.tl_y

                # 解出精修变换矩阵, 优先使用 match.transform (C++ 侧已精修合成好),
                # 若为空则回退到手动 refine() 增量矩阵
                A = None  # 2x3, 训练图坐标系 -> 场景图坐标系
                try:
                    tf = np.array(match.transform, dtype=np.float64)
                    if tf.ndim == 2 and tf.shape[0] == 2 and tf.shape[1] == 3 and tf.size > 0:
                        A = tf
                except Exception:
                    A = None
                if A is not None:
                    # match.transform 作用于"训练图坐标"(模板坐标 + tl 偏移)
                    def to_scene(pts_templ):
                        pts = np.asarray(pts_templ, dtype=np.float64)
                        pts_train = pts + [tl_x, tl_y]
                        ones = np.ones((len(pts_train), 1), dtype=np.float64)
                        return (A @ np.hstack([pts_train, ones]).T).T[:, :2]
                    delta_angle = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
                else:
                    refine_res = self.detector.refine(match)
                    M = np.array(refine_res.transformation, dtype=np.float64)  # 3x3 增量矩阵
                    def to_scene(pts_templ):
                        pts = np.asarray(pts_templ, dtype=np.float64)
                        coarse = pts + [match.x, match.y]
                        ones = np.ones((len(coarse), 1), dtype=np.float64)
                        return (M @ np.hstack([coarse, ones]).T).T[:, :2]
                    delta_angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

                # 定义变换函数：将 ROI 点转换到模板坐标系
                def roi_to_templ(pts):
                    # pts: Nx2 array in ROI coordinates
                    # 步骤1：平移到填充图像
                    pts_pad = pts + [padding, padding]
                    # 步骤2：绕中心顺时针旋转缩放 (图像坐标系，y向下)
                    x_rel = pts_pad[:, 0] - center_x
                    y_rel = pts_pad[:, 1] - center_y
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    # 顺时针旋转公式
                    x_rot = center_x + scale * (x_rel * cos_a + y_rel * sin_a)
                    y_rot = center_y + scale * (-x_rel * sin_a + y_rel * cos_a)
                    # 步骤3：裁剪偏移
                    x_templ = x_rot - tl_x
                    y_templ = y_rot - tl_y
                    return np.stack([x_templ, y_templ], axis=1)

                # 原始 ROI 四个角点（ROI 坐标系）
                roi_corners = np.array([
                    [0, 0],
                    [roi_w, 0],
                    [roi_w, roi_h],
                    [0, roi_h]
                ], dtype=np.float32)

                # 精修后的角点 (ROI -> 模板坐标 -> 场景坐标)
                refined_corners = to_scene(roi_to_templ(roi_corners))

                # 初始化结果字典
                res = {
                    'class_id': class_id,
                    'template_id': match.template_id,
                    'score': match.similarity,
                    'x': float(match.x),
                    'y': float(match.y),
                    'icp_refined': bool(use_refine)
                }

                res['refined_box_points'] = refined_corners.tolist()

                # 计算精修后的特征点 (模板特征坐标 -> 场景坐标)
                feat_pts = np.array([[f.x, f.y] for f in templ0.features], dtype=np.float64)
                if len(feat_pts) > 0:
                    refined_pts = to_scene(feat_pts)
                    res['matched_features'] = refined_pts.tolist()
                else:
                    res['matched_features'] = []

                # 计算精修后的旋转中心
                center = np.mean(refined_corners, axis=0)
                res['refined_x'] = float(center[0])
                res['refined_y'] = float(center[1])

                # 最终角度 = 训练角度 - ICP 增量角
                res['refined_angle'] = angle_deg - delta_angle
                # ICP 内点率与重叠度 (当前接口 Match 直接提供)
                res['fitness'] = float(match.fitness)
                res['overlap'] = float(match.overlap)

                # --- 计算抓取点 ---
                grasp_pts_roi = grasp_points_config.get(class_id, []) if grasp_points_config else []
                if grasp_pts_roi:
                    # 将抓取点从 ROI 坐标系转换到模板坐标系, 再变换到场景坐标系
                    grasp_pts = np.array(grasp_pts_roi, dtype=np.float64)
                    refined_grasp = to_scene(roi_to_templ(grasp_pts))
                    res['grasp_points'] = refined_grasp.tolist()
                    res['grasp_point'] = refined_grasp[0].tolist()   # 取第一个作为主抓取点
                else:
                    # 无自定义抓取点，使用中心点
                    res['grasp_point'] = [center[0], center[1]]
                    res['grasp_points'] = [[center[0], center[1]]]

                results.append(res)

            except Exception as e:
                logging.error(f"处理匹配结果 [{match.class_id}] 时出错: {e}")
                traceback.print_exc()
                continue

        return results



    @staticmethod
    def draw_results(image: np.ndarray, results: Dict[str, List[Dict]], **kwargs) -> np.ndarray:
        display_img = image.copy()
        for roi_name, eval_results in results.items():
            if not eval_results: continue
            eval_result = eval_results[0]

            roi_bbox = eval_result["geometry"]["value"]
            x, y, w, h = [int(c) for c in roi_bbox]
            roi_color = (0, 255, 0) if eval_result["label"] == LabelStatus.GOOD else (0, 0, 255)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), roi_color, 2)
            cv2.putText(display_img, f"{roi_name}: {eval_result['label'].value}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)

            matches_to_draw = eval_result.get("metadata", {}).get("found_matches", [])
            for match in matches_to_draw:
                if match["geometry"]["type"] == GeometryType.ROTATED_BBOX:
                    points = np.array(match["geometry"]["value"], dtype=np.int32)
                    # 为不同类别使用不同颜色 (此处简化为统一颜色，可在UI层做得更好)
                    match_color = (255, 128, 0)
                    cv2.polylines(display_img, [points], isClosed=True, color=match_color, thickness=2)
        return display_img
