# ICFIE-YOLO 目标检测评价报告

## 评价概述

| 项目 | 值 |
|------|----|
| 评价时间 | 2026-05-15 19:59:19 |
| Checkpoint | runs/icfie_yolo-5-15-a1-b10-c10/stage_c_epoch_10.pt |
| ENABLE_MSICN | True |
| ENABLE_FIE | True |
| 数据集 | data/Exdark/val.txt |
| 图像数量 | 1472 |
| 图像尺寸 | 416 |
| CONF_THRESHOLD | 0.25 |
| NMS_IOU_THRESHOLD | 0.45 |
| 设备 | cuda:0 |

## 总体检测指标

| 指标 | 值 |
|------|----|
| mAP@0.5 | 0.7252 |
| mAP@0.5:0.95 | 0.4491 |
| Precision | 0.7820 |
| Recall | 0.7091 |
| F1 | 0.7407 |

## 各类别 AP@0.5 详细结果

(按 AP@0.5 降序排列)

| 排名 | 类别 | AP@0.5 |
|------|------|--------|
| 1 | Bus | 0.8742 |
| 2 | Car | 0.8060 |
| 3 | People | 0.7759 |
| 4 | Bicycle | 0.7753 |
| 5 | Motorbike | 0.7647 |
| 6 | Boat | 0.7242 |
| 7 | Chair | 0.7033 |
| 8 | Cup | 0.7003 |
| 9 | Dog | 0.6967 |
| 10 | Bottle | 0.6493 |
| 11 | Cat | 0.6338 |
| 12 | Table | 0.5987 |

## 统计摘要

| 统计量 | 值 |
|--------|----|
| AP@0.5 均值 | 0.7252 |
| AP@0.5 中位数 | 0.7138 |
| AP@0.5 标准差 | 0.0749 |
| 推理耗时均值 | 24.8 ms |
| 推理耗时中位数 | 21.6 ms |

## 附注

评价使用 YOLOv7 标准推理链路: letterbox -> 模型前向 -> NMS -> scale_coords.
AP 计算复用 yolov7/utils/metrics.py 中的 ap_per_class 函数.
EVAL_IOU_RANGE: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
