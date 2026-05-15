# ICFIE-YOLO 目标检测评价报告

## 评价概述

| 项目 | 值 |
|------|----|
| 评价时间 | 2026-05-15 20:02:15 |
| Checkpoint | runs/icfie_yolo-5-15-a1-b10-c10/stage_a_epoch_1.pt |
| ENABLE_MSICN | False |
| ENABLE_FIE | False |
| 数据集 | data/Exdark/val.txt |
| 图像数量 | 1472 |
| 图像尺寸 | 416 |
| CONF_THRESHOLD | 0.25 |
| NMS_IOU_THRESHOLD | 0.45 |
| 设备 | cuda:0 |

## 总体检测指标

| 指标 | 值 |
|------|----|
| mAP@0.5 | 0.4953 |
| mAP@0.5:0.95 | 0.2013 |
| Precision | 0.6016 |
| Recall | 0.5637 |
| F1 | 0.5660 |

## 各类别 AP@0.5 详细结果

(按 AP@0.5 降序排列)

| 排名 | 类别 | AP@0.5 |
|------|------|--------|
| 1 | People | 0.7187 |
| 2 | Car | 0.6380 |
| 3 | Bicycle | 0.5945 |
| 4 | Dog | 0.5814 |
| 5 | Chair | 0.5593 |
| 6 | Motorbike | 0.5438 |
| 7 | Bottle | 0.4934 |
| 8 | Cup | 0.4371 |
| 9 | Boat | 0.4141 |
| 10 | Bus | 0.3988 |
| 11 | Cat | 0.3215 |
| 12 | Table | 0.2426 |

## 统计摘要

| 统计量 | 值 |
|--------|----|
| AP@0.5 均值 | 0.4953 |
| AP@0.5 中位数 | 0.5186 |
| AP@0.5 标准差 | 0.1314 |
| 推理耗时均值 | 9.5 ms |
| 推理耗时中位数 | 8.0 ms |

## 附注

评价使用 YOLOv7 标准推理链路: letterbox -> 模型前向 -> NMS -> scale_coords.
AP 计算复用 yolov7/utils/metrics.py 中的 ap_per_class 函数.
EVAL_IOU_RANGE: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
