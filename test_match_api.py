# -*- coding: utf-8 -*-
"""
验证新增的匹配接口:
  置信度 confidence / 重叠度 overlap / 类别过滤 / NMS / 最大数量 / ICP 开关 / 仿射变换矩阵

术语:
  置信度 = 匹配上的特征点数 / 总特征点数   (库里 similarity 是它的百分制)
  重叠度 = 与其它目标矩形 mask 的重叠面积 / 自身面积  (不是 IoU)
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "build", "Release"))
import shape_based_matching_py as sb


def pad16(img):
    ph = (16 - img.shape[0] % 16) % 16
    pw = (16 - img.shape[1] % 16) % 16
    if ph == 0 and pw == 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def load_gray(name):
    p = os.path.join(HERE, "test", "case1", name)
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    assert img is not None, p
    return pad16(img)


def show(tag, ms):
    print(f"  {tag}: {len(ms)} 个")
    for m in ms:
        print(f"    cls={m.class_id} conf={m.confidence:.3f} overlap={m.overlap:.3f} "
              f"pos=({m.x},{m.y}) size=({m.width}x{m.height}) "
              f"fit={m.fitness:.3f} angle={m.angle:.2f} scale={m.scale:.4f}")


def main():
    src_a = load_gray("train.png")   # 类别 A
    src_b = load_gray("test.png")    # 类别 B(只用来验证类别过滤)
    h, w = src_a.shape
    print(f"模板 A: {w}x{h}, 模板 B: {src_b.shape[1]}x{src_b.shape[0]}")

    full = np.full(src_a.shape, 255, np.uint8)

    # ---------- 场景 1: 单个实例, 用来验证变换矩阵 ----------
    # 场景图本身也必须是 16 的倍数, 否则 linearize 会断言失败
    OX, OY = 61, 97
    scene1 = np.zeros((h + 200, w + 200), np.uint8)
    scene1 = pad16(scene1)
    scene1[OY:OY + h, OX:OX + w] = src_a

    det = sb.Detector(128, [4, 8])
    det.addTemplate(src_a, "A", full)
    det.addTemplate(src_b, "B", np.full(src_b.shape, 255, np.uint8))
    print("模板数:", det.numTemplates())

    print("\n[1] 基本匹配(默认参数)")
    p = sb.MatchParams()
    p.min_confidence = 85
    ms = det.match(scene1, p)
    show("默认", ms)
    assert len(ms) >= 1
    m = ms[0]
    print(f"    真实位置 ({OX},{OY}), 匹配到 ({m.x},{m.y})")

    print("\n[2] 仿射变换矩阵: 用 transform 把模板 warp 到场景, 与真实位置比对")
    tm = m.transform
    print(f"    transform =\n{tm}")
    warped = cv2.warpAffine(full, tm, (scene1.shape[1], scene1.shape[0]))
    gt = np.zeros(scene1.shape, np.uint8)
    gt[OY:OY + h, OX:OX + w] = 255
    inter = cv2.countNonZero(cv2.bitwise_and(warped, gt))
    union = cv2.countNonZero(cv2.bitwise_or(warped, gt))
    print(f"    warp 后与真实位置的 IoU = {inter/union:.4f}  (inter={inter} union={union})")
    assert inter / union > 0.9, "变换矩阵不对, warp 结果没落到目标上"

    print("\n[3] ICP 开关")
    p_off = sb.MatchParams(); p_off.min_confidence = 85; p_off.use_refine = False
    m_off = det.match(scene1, p_off)[0]
    p_on = sb.MatchParams(); p_on.min_confidence = 85; p_on.use_refine = True
    m_on = det.match(scene1, p_on)[0]
    print(f"    关闭: transform[0,2]={m_off.transform[0][2]:.2f} fitness={m_off.fitness} "
          f"angle={m_off.angle:.3f} scale={m_off.scale:.4f}")
    print(f"    开启: transform[0,2]={m_on.transform[0][2]:.2f} fitness={m_on.fitness:.4f} "
          f"rmse={m_on.inlier_rmse:.4f} angle={m_on.angle:.3f} scale={m_on.scale:.4f}")
    assert m_off.fitness < 0, "未精修时 fitness 应为 -1"
    assert 0 <= m_on.fitness <= 1, "精修后 fitness 应在 0~1"

    # 精修后的 transform 也拿来 warp 一次, 应该同样(或更好)地落在目标上
    for tag, mm in (("关闭", m_off), ("开启", m_on)):
        wd = cv2.warpAffine(full, mm.transform, (scene1.shape[1], scene1.shape[0]))
        it = cv2.countNonZero(cv2.bitwise_and(wd, gt))
        un = cv2.countNonZero(cv2.bitwise_or(wd, gt))
        print(f"    {tag} ICP 时 transform 的 warp IoU = {it/un:.4f}")
        assert it / un > 0.9, f"{tag} ICP 的变换矩阵不对"

    print("\n[4] 类别过滤")
    for ids in (["A"], ["B"], ["A", "B"], []):
        p2 = sb.MatchParams(); p2.min_confidence = 85; p2.class_ids = ids
        got = det.match(scene1, p2)
        print(f"    class_ids={ids or '(全部)'} -> {len(got)} 个, 类别={[x.class_id for x in got]}")

    # ---------- 场景 2: 两个重叠实例, 验证重叠度与 NMS ----------
    print("\n[5] 重叠度与 NMS (两个实例水平重叠 200 像素)")
    OVER = 200
    scene2 = np.zeros((h + 200, w + 800), np.uint8)
    scene2 = pad16(scene2)
    x1, x2 = 61, 61 + w - OVER
    scene2[OY:OY + h, x1:x1 + w] = src_a
    scene2[OY:OY + h, x2:x2 + w] = np.maximum(scene2[OY:OY + h, x2:x2 + w], src_a)
    expect_ov = OVER / w
    print(f"    理论重叠度 = {OVER}/{w} = {expect_ov:.3f}")

    # 注意: 两个实例在重叠区互相覆盖了像素, 所以各自的置信度都会下降, 阈值要放低才都能匹配到
    p3 = sb.MatchParams(); p3.min_confidence = 70; p3.nms = False
    ms3 = det.match(scene2, p3)
    show("nms=False", ms3)
    if len(ms3) >= 2:
        got_ov = max(m.overlap for m in ms3)
        print(f"    实测最大重叠度 = {got_ov:.3f} (理论 {expect_ov:.3f}, 分母是包围盒宽 {ms3[0].width})")
        assert abs(got_ov - expect_ov) < 0.08, "重叠度数值偏离理论值"

    print("\n[6] NMS 开关对比")
    for flag, thr in ((False, 0.5), (True, 0.5), (True, 0.2), (True, 0.9)):
        p4 = sb.MatchParams(); p4.min_confidence = 70; p4.nms = flag; p4.nms_overlap = thr
        print(f"    nms={flag}, nms_overlap={thr} -> {len(det.match(scene2, p4))} 个")

    print("\n[7] 最大匹配数量 / 重叠度上限")
    p5 = sb.MatchParams(); p5.min_confidence = 70; p5.nms = False; p5.max_matches = 1
    print(f"    max_matches=1 -> {len(det.match(scene2, p5))} 个")
    p6 = sb.MatchParams(); p6.min_confidence = 70; p6.nms = False; p6.max_overlap = 0.1
    r6 = det.match(scene2, p6)
    print(f"    max_overlap=0.1 -> {len(r6)} 个 (重叠度 {[round(m.overlap,3) for m in r6]})")

    print("\n[8] 置信度阈值")
    for th in (70, 85, 95, 99):
        p7 = sb.MatchParams(); p7.min_confidence = th; p7.nms = False
        got = det.match(scene2, p7)
        print(f"    min_confidence={th} -> {len(got)} 个")

    print("\n全部通过")


if __name__ == "__main__":
    main()
