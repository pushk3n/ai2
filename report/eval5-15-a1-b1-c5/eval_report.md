# ICFIE-YOLO 目标检测评价报告

## 评价概述

| 项目 | 值 |
|------|----|
| 评价时间 | 2026-05-15 20:37:51 |
| Checkpoint | runs/icfie_yolo-5-15-a1-b1-c5/stage_c_epoch_5.pt |
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
| mAP@0.5 | 0.7008 |
| mAP@0.5:0.95 | 0.4199 |
| Precision | 0.7441 |
| Recall | 0.7170 |
| F1 | 0.7273 |

## 各类别 AP@0.5 详细结果

(按 AP@0.5 降序排列)

| 排名 | 类别 | AP@0.5 |
|------|------|--------|
| 1 | Bus | 0.8382 |
| 2 | Bicycle | 0.7960 |
| 3 | Car | 0.7828 |
| 4 | People | 0.7561 |
| 5 | Motorbike | 0.7037 |
| 6 | Chair | 0.6914 |
| 7 | Boat | 0.6913 |
| 8 | Cup | 0.6604 |
| 9 | Dog | 0.6587 |
| 10 | Bottle | 0.6390 |
| 11 | Cat | 0.6098 |
| 12 | Table | 0.5817 |

## 统计摘要

| 统计量 | 值 |
|--------|----|
| AP@0.5 均值 | 0.7008 |
| AP@0.5 中位数 | 0.6913 |
| AP@0.5 标准差 | 0.0751 |
| 推理耗时均值 | 22.3 ms |
| 推理耗时中位数 | 20.7 ms |

## 附注

评价使用 YOLOv7 标准推理链路: letterbox -> 模型前向 -> NMS -> scale_coords.
AP 计算复用 yolov7/utils/metrics.py 中的 ap_per_class 函数.
EVAL_IOU_RANGE: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
