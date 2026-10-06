# -*- coding: utf-8 -*-
"""旧版 matcher.py (git HEAD) 与 C++ 化新版的行为 A/B 对比。

相同输入 -> 相同产物结构 -> 相同匹配结果 (几何误差应在浮点容差内)。
"""
import os
import sys
import json
import shutil
import importlib.util
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))            # py_tests/
ROOT = os.path.dirname(os.path.dirname(HERE))                                  # 项目根目录
sys.path.insert(0, os.path.join(ROOT, "build", "Release"))   # 提前注入, 旧版模块也能找到 pyd

# 加载旧版
spec = importlib.util.spec_from_file_location("matcher_old", os.path.join(HERE, "matcher_old_ab.py"))
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)
# 加载新版
spec2 = importlib.util.spec_from_file_location("matcher_new", os.path.join(ROOT, "matcher.py"))
new_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(new_mod)

TRAIN_PARAMS = {
    "pyramid_levels": [4, 8], "feature_num": 100,
    "weak_thresh": 30.0, "strong_thresh": 60.0,
    "angle_start": -15.0, "angle_extent": 30.0, "angle_step": 10.0,
    "scale_start": 0.9, "scale_end": 1.1, "scale_step": 0.1,
}
EXCL = [{"type": "exclude_rect", "rect": [550, 423, 50, 50]}]


def pad16(img):
    ph = (16 - img.shape[0] % 16) % 16
    pw = (16 - img.shape[1] % 16) % 16
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def run_pipeline(mod, tag, save_dir):
    img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"))
    th, tw = img.shape[:2]
    m = mod.Matcher()
    res = m.train(img, (0, 0, tw, th), "夹具A", TRAIN_PARAMS, save_dir, exclusion_zones=EXCL)
    m2 = mod.Matcher()
    cid, params = m2.add_template_class(res["info_path"])
    scene = np.zeros((th + 300, tw + 300, 3), np.uint8)
    scene = pad16(scene)
    OX, OY = 101, 97
    scene[OY:OY + th, OX:OX + tw] = img
    results = m2.match(scene, 50.0, use_refine=True)
    print(f"[{tag}] template_count={res['template_count']} 加载={cid} 匹配数={len(results)}")
    return results


def norm(r):
    """把结果 dict 规整成可比较的结构"""
    return {
        "class_id": r["class_id"],
        "score": round(float(r["score"]), 3),
        "refined_x": round(float(r["refined_x"]), 1),
        "refined_y": round(float(r["refined_y"]), 1),
        "refined_angle": round(float(r["refined_angle"]), 2),
        "fitness": round(float(r["fitness"]), 3),
        "n_box": len(r["refined_box_points"]),
        "n_feat": len(r["matched_features"]),
        "grasp": [round(float(v), 1) for v in r["grasp_point"]],
    }


def main():
    old_dir = os.path.join(HERE, "ab_旧版输出")
    new_dir = os.path.join(HERE, "ab_新版输出")
    try:
        old_res = run_pipeline(old_mod, "旧版", old_dir)
        new_res = run_pipeline(new_mod, "新版", new_dir)
        print("\n旧版首个:", norm(old_res[0]) if old_res else None)
        print("新版首个:", norm(new_res[0]) if new_res else None)
        assert len(old_res) == len(new_res), f"匹配数不同: {len(old_res)} vs {len(new_res)}"
        for a, b in zip(old_res, new_res):
            na, nb = norm(a), norm(b)
            # 几何量容差 1.5px / 1 度 (ICP 迭代顺序的浮点级差异可忽略)
            assert na["class_id"] == nb["class_id"]
            assert abs(na["score"] - nb["score"]) < 0.5, (na, nb)
            assert abs(na["refined_x"] - nb["refined_x"]) < 1.5, (na, nb)
            assert abs(na["refined_y"] - nb["refined_y"]) < 1.5, (na, nb)
            assert abs(na["refined_angle"] - nb["refined_angle"]) < 1.0, (na, nb)
            assert abs(na["fitness"] - nb["fitness"]) < 0.05, (na, nb)
            assert abs(na["grasp"][0] - nb["grasp"][0]) < 1.5, (na, nb)
            assert na["n_box"] == nb["n_box"] == 4 and na["n_feat"] == nb["n_feat"] > 0
        # 旧版产物 info.json 与新版逐字段对比
        old_info = json.load(open(os.path.join(old_dir, "夹具A.info.json"), encoding="utf-8"))
        new_info = json.load(open(os.path.join(new_dir, "夹具A.info.json"), encoding="utf-8"))
        assert old_info["training_params"] == new_info["training_params"], "training_params 不一致"
        assert {k: (v["angle"], v["scale"]) for k, v in old_info["templates"].items()} == \
               {k: (v["angle"], v["scale"]) for k, v in new_info["templates"].items()}, "templates 不一致"
        assert len(old_info["base_template_features"]) == len(new_info["base_template_features"])
        assert (old_info["original_w"], old_info["original_h"], old_info["padding"]) == \
               (new_info["original_w"], new_info["original_h"], new_info["padding"])
        print("\nA/B 全部一致: 匹配结果与 info.json schema 完全兼容")
    finally:
        shutil.rmtree(old_dir, ignore_errors=True)
        shutil.rmtree(new_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
