# -*- coding: utf-8 -*-
"""PyMatcher 端到端冒烟测试: 训练(中文目录) -> 加载 -> 匹配 -> 几何校验。

验证 C++ 化后的 matcher.py 门面:
  1. train() 在中文保存目录正确产出 4 份文件;
  2. add_template_class() 从 .yaml / .info.json 两条路径都能加载;
  3. match() 结果字段齐全, 旋转外框/特征点/抓取点落在真实目标上;
  4. masks 场景掩码生效;
  5. info.json schema 与旧版兼容 (用 json 模块可直接解析)。
"""
import os
import sys
import json
import shutil
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))            # py_tests/
ROOT = os.path.dirname(HERE)                                  # 项目根目录
sys.path.insert(0, ROOT)          # 让根目录的 matcher.py 可导入
from matcher import Matcher, safe_file_name


def pad16(img):
    ph = (16 - img.shape[0] % 16) % 16
    pw = (16 - img.shape[1] % 16) % 16
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def main():
    train_img = cv2.imread(os.path.join(ROOT, "test", "case1", "train.png"))
    assert train_img is not None
    th, tw = train_img.shape[:2]

    # ---------- 1. 训练: 中文目录 + 中文类别名 + 排除区 + 角度范围 ----------
    save_dir = os.path.join(HERE, "输出_测试目录")   # 中文目录, 验证 C++ 侧路径处理
    class_id = "夹具A"
    m = Matcher()
    result = m.train(
        train_img, (0, 0, tw, th), class_id,
        {
            "pyramid_levels": [4, 8], "feature_num": 100,
            "weak_thresh": 30.0, "strong_thresh": 60.0,
            "angle_start": -15.0, "angle_extent": 30.0, "angle_step": 10.0,
            "scale_start": 0.9, "scale_end": 1.1, "scale_step": 0.1,
        },
        save_dir,
        exclusion_zones=[{"type": "exclude_rect", "rect": [tw - 50, th - 50, 50, 50]}],
    )
    print("[1] 训练结果:", {k: v for k, v in result.items() if k != "features_image"})
    assert result["template_count"] > 0, "应有模板生成"
    assert result["base_name"] == safe_file_name("夹具A") == "夹具A"
    for key in ("yaml_path", "info_path", "preview_path"):
        assert result[key] and os.path.isfile(result[key]), f"{key} 未落地: {result[key]}"
    assert os.path.isfile(os.path.join(save_dir, "夹具A.jpg"))

    # info.json 可被标准 json 模块解析, schema 与旧版兼容
    with open(result["info_path"], "r", encoding="utf-8") as f:
        info = json.load(f)
    assert set(info.keys()) == {"training_params", "templates", "base_template_features",
                                "original_w", "original_h", "padding"}, info.keys()
    assert info["original_w"] == tw and info["original_h"] == th
    assert info["templates"], "templates 不应为空"
    print("[1] 通过: 中文目录训练产物齐全, info.json schema 兼容")

    # ---------- 2. 重新实例化, 从 info.json 加载 ----------
    m2 = Matcher()
    loaded_id, params = m2.add_template_class(result["info_path"])
    assert loaded_id == "夹具A"
    assert params["feature_num"] == 100 and params["pyramid_levels"] == [4, 8]
    assert m2.get_loaded_class_ids() == ["夹具A"]

    m3 = Matcher()
    loaded_id3, _ = m3.add_template_class(result["yaml_path"])   # 从 yaml 加载
    assert loaded_id3 == "夹具A"
    print("[2] 通过: info.json / yaml 两种入口都能加载")

    # ---------- 3. 匹配: 把训练图贴进场景 (含旋转), 验证几何 ----------
    # 用"单角度 + 单尺度"训练, 此时模板即 ROI 本身, 外框角点应与真实目标重合
    result_geo = m.train(
        train_img, (0, 0, tw, th), "夹具B",
        {
            "pyramid_levels": [4, 8], "feature_num": 100,
            "weak_thresh": 30.0, "strong_thresh": 60.0,
            "angle_start": 0.0, "angle_extent": 0.0, "angle_step": 10.0,
            "scale_start": 1.0, "scale_end": 1.0, "scale_step": 0.1,
        },
        save_dir,
    )
    m4 = Matcher()
    m4.add_template_class(result_geo["info_path"])
    scene = np.zeros((th + 300, tw + 300, 3), np.uint8)
    scene = pad16(scene)
    OX, OY = 101, 97
    scene[OY:OY + th, OX:OX + tw] = train_img

    results = m4.match(scene, 50.0, use_refine=True)
    assert results, "应至少匹配到 1 个目标"
    r = results[0]
    print("[3] 首个结果:", {k: r[k] for k in
          ("class_id", "score", "refined_x", "refined_y", "refined_angle", "fitness", "overlap",
           "angle", "scale", "refined_scale")})
    # scale / angle 必须与 info.json 里同一 template_id 的记录一致 (旧版这里会丢)
    with open(result_geo["info_path"], "r", encoding="utf-8") as f:
        info_b = json.load(f)
    tpl = info_b["templates"].get(str(r["template_id"]))
    assert tpl is not None, f"info.json 里查不到 template_id={r['template_id']}"
    assert abs(r["scale"] - tpl["scale"]) < 1e-6, f"scale 与 info.json 不符: {r['scale']} != {tpl['scale']}"
    assert abs(r["angle"] - tpl["angle"]) < 1e-6, f"angle 与 info.json 不符: {r['angle']} != {tpl['angle']}"
    # 自匹配场景, ICP 额外缩放应接近 1
    assert 0.2 < r["refined_scale"] < 5.0, f"refined_scale 离谱: {r['refined_scale']}"
    assert r["class_id"] == "夹具B"
    assert abs(r["refined_x"] - (OX + tw / 2)) < 15 and abs(r["refined_y"] - (OY + th / 2)) < 15, \
        f"中心位置偏差过大: ({r['refined_x']:.1f}, {r['refined_y']:.1f})"
    assert abs(r["refined_angle"]) < 8, f"角度偏差过大: {r['refined_angle']}"
    assert r["fitness"] >= 0, "精修后 fitness 应 >= 0"
    assert len(r["refined_box_points"]) == 4 and len(r["matched_features"]) > 0
    assert len(r["refined_box_points"][0]) == 2
    # 旋转外框角点: 预期角点按精修角度绕目标中心旋转后, 应与结果角点重合
    corners = np.array(r["refined_box_points"], np.float64)
    expect = np.array([[OX, OY], [OX + tw, OY], [OX + tw, OY + th], [OX, OY + th]], np.float64)
    ang = np.radians(r["refined_angle"])
    c, s = np.cos(ang), np.sin(ang)
    ctr = expect.mean(axis=0)
    rot = np.array([[c, -s], [s, c]]) @ (expect - ctr).T
    expect_rot = (rot.T + ctr)
    dists = np.linalg.norm(corners[:, None, :] - expect_rot[None, :, :], axis=2).min(axis=1)
    assert dists.mean() < 5.0, f"外框角点偏差过大: {dists}"
    assert "grasp_point" in r and len(r["grasp_point"]) == 2

    # 3b. scale 必须随命中的模板而变, 不能恒为 1.0:
    #     训练一个 scale 0.8~1.2 的类别, 把训练图缩小 0.9 贴进场景,
    #     命中模板的 scale 应落在训练范围内且与 info.json 记录一致。
    result_ms = m.train(
        train_img, (0, 0, tw, th), "夹具C",
        {
            "pyramid_levels": [4, 8], "feature_num": 100,
            "weak_thresh": 30.0, "strong_thresh": 60.0,
            "angle_start": 0.0, "angle_extent": 0.0, "angle_step": 10.0,
            "scale_start": 0.8, "scale_end": 1.2, "scale_step": 0.1,
        },
        save_dir,
    )
    m5 = Matcher()
    m5.add_template_class(result_ms["info_path"])
    small = cv2.resize(train_img, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
    small = pad16(small)
    sh, sw = small.shape[:2]
    scene_s = pad16(np.zeros((sh + 300, sw + 300, 3), np.uint8))
    scene_s[200:200 + sh, 150:150 + sw] = small
    res_s = m5.match(scene_s, 50.0, use_refine=True)
    assert res_s, "缩小 0.9 的场景应能匹配到目标"
    rs = res_s[0]
    with open(result_ms["info_path"], "r", encoding="utf-8") as f:
        info_c = json.load(f)
    tpl_c = info_c["templates"][str(rs["template_id"])]
    print("[3b] 缩小场景命中: angle=%.3f scale=%.4f refined_scale=%.4f (info.json: angle=%.3f scale=%.4f)"
          % (rs["angle"], rs["scale"], rs["refined_scale"], tpl_c["angle"], tpl_c["scale"]))
    assert abs(rs["scale"] - tpl_c["scale"]) < 1e-6, f"scale 与 info.json 不符: {rs['scale']} != {tpl_c['scale']}"
    assert 0.75 <= rs["scale"] <= 1.25, f"scale 不在训练范围 0.8~1.2 内: {rs['scale']}"
    assert abs(rs["refined_scale"] - 1.0) < 0.2, f"自匹配场景的 refined_scale 应贴近 1: {rs['refined_scale']}"

    # 自定义抓取点: ROI 坐标 (10, 20) 应被变换到场景中 OX+10, OY+20 附近
    results_g = m4.match(scene, 50.0, grasp_points_config={"夹具B": [[10, 20]]})
    gp = results_g[0]["grasp_point"]
    assert abs(gp[0] - (OX + 10)) < 15 and abs(gp[1] - (OY + 20)) < 15, f"抓取点偏差: {gp}"
    print("[3] 通过: 匹配几何 / 特征点 / 抓取点全部落位")

    # ---------- 4. 场景 mask: 只在目标区域搜索 ----------
    mask = np.zeros(scene.shape[:2], np.uint8)
    mask[OY:OY + th, OX:OX + tw] = 255
    results_masked = m4.match(scene, 50.0, masks=mask)
    assert results_masked, "mask 覆盖目标时应有结果"
    mask_far = np.zeros(scene.shape[:2], np.uint8)
    mask_far[0:50, 0:50] = 255   # 远离目标的区域
    results_empty = m4.match(scene, 50.0, masks=mask_far)
    assert not results_empty, "mask 不覆盖目标时应无结果"
    print("[4] 通过: 场景 mask 生效")

    # ---------- 5. 覆盖训练 + 清空 ----------
    m2.clear()
    assert m2.get_loaded_class_ids() == []
    print("[5] 通过: clear 生效")

    shutil.rmtree(save_dir, ignore_errors=True)
    print("\n全部通过")


if __name__ == "__main__":
    main()
