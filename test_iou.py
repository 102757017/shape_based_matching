# -*- coding: utf-8 -*-
"""
computeIoU 自测：把模板贴到场景图的已知位置，GT mask 由构造过程精确生成，
再用 detector.match + detector.computeIoU 反算，IoU 应接近 1。
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "build", "Release"))
import shape_based_matching_py as sb

TRAIN = os.path.join(HERE, "test", "case1", "train.png")


def pad16(img):
    """库要求输入尺寸为 16 的倍数"""
    h, w = img.shape[:2]
    ph = (-h) % 16
    pw = (-w) % 16
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


def main():
    src = cv2.imread(TRAIN, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise SystemExit("cannot read train.png")
    src = pad16(src)

    # ---- 训练用的 mask：整幅 ROI ----
    train_mask = np.full(src.shape, 255, np.uint8)

    # ---- 构造场景：把 src 贴到 (OX, OY) ----
    OX, OY = 97, 61
    scene = np.zeros((src.shape[0] + 160, src.shape[1] + 160), np.uint8)
    scene[OY:OY + src.shape[0], OX:OX + src.shape[1]] = src
    gt_mask = np.zeros(scene.shape, np.uint8)
    gt_mask[OY:OY + src.shape[0], OX:OX + src.shape[1]] = train_mask

    # ---- 训练 ----
    det = sb.Detector(100, [4, 8])
    infos = []
    for ang in (0, 15, 30):
        infos.append(sb.Info(ang, 1.0))

    producer = sb.shapeInfo_producer(src, train_mask)
    producer.infos = infos
    tid0 = None
    for i, info in enumerate(infos):
        s = producer.src_of(info)
        m = producer.mask_of(info)
        assert s.shape == src.shape, (s.shape, src.shape)
        t = det.addTemplate(pad16(s), "obj", pad16(m))
        if i == 0:
            tid0 = t
    print("templates:", det.numTemplates())

    # ---- 匹配 ----
    matches = det.match(pad16(scene), 80)
    print("matches:", len(matches))
    if not matches:
        raise SystemExit("no match")

    # 找与构造位置最接近的那个 match
    m = min(matches, key=lambda mm: (mm.x - OX) ** 2 + (mm.y - OY) ** 2)
    print(f"best match: x={m.x} y={m.y} (gt {OX},{OY}) sim={m.similarity:.2f} tid={m.template_id}")

    # ---- IoU（用内部缓存的训练 mask，走 icp 精修）----
    tm_chk = det.getTemplateMask("obj", 0)
    print(f"diag: src={src.shape} scene={scene.shape} templ_mask={tm_chk.shape} nonzero={cv2.countNonZero(tm_chk)}")
    r = det.computeIoU(m, gt_mask)
    print(f"[refine ] iou={r.iou:.4f} inter={r.inter_area:.0f} union={r.union_area:.0f} "
          f"pred={r.pred_area:.0f} gt={r.gt_area:.0f} precision={r.precision:.4f} recall={r.recall:.4f}")

    # ---- IoU（不精修，只平移）----
    r2 = det.computeIoU(m, gt_mask, use_refine=False)
    print(f"[no refine] iou={r2.iou:.4f} pred={r2.pred_area:.0f} gt={r2.gt_area:.0f}")

    # ---- 对照：故意错开 20 像素，IoU 应明显下降 ----
    shifted = np.roll(gt_mask, 20, axis=1)
    r3 = det.computeIoU(m, shifted, use_refine=False)
    print(f"[shifted gt] iou={r3.iou:.4f}")

    # ---- fitness 对照 ----
    reg = det.refine(m)
    print("T (model->scene):", [[round(v, 4) for v in row] for row in reg.transformation])
    print(f"refine: fitness={reg.fitness:.4f} inlier_rmse={reg.inlier_rmse:.4f}")

    # ---- 显式传 templ_mask 的路径 ----
    tm = det.getTemplateMask("obj", m.template_id)
    print("cached templ mask:", tm.shape, tm.dtype)
    r4 = det.computeIoU(m, gt_mask, tm)
    print(f"[explicit mask] iou={r4.iou:.4f}")


if __name__ == "__main__":
    main()
