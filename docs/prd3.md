# PRD: ICFIE-YOLO 评价脚本与报告生成

> **依据**: 本文档在 [prd2.md](prd2.md) 的架构与推理约束基础上制定,
> 专注于量化评价已训练模型的检测能力, 并自动生成中文 Markdown 报告.
> 一切推理流水线约束（letterbox/NMS/scale_coords/通道断言）均继承自 prd2.md,
> 本文档不重复定义, 仅补充评价专属规范.

---

## 1. 背景与目标

### 1.1 问题背景

当前训练脚本 `src/train.py` 仅记录各阶段的 box/obj/cls/total 训练损失, 不计算
mAP、Recall 等目标检测评价指标. 无法判断模型实际检测能力, 也无法对比 MSICN/FIE
各模块的贡献.

本 PRD 指导实现 `src/eval.py`, 满足以下目标:

1. **单图评价模式**: 对指定单张图像完成推理可视化, 若提供对应标签则计算
   TP/FP/FN 和单图 AP.
2. **批量评价模式**: 读取 `val.txt` 列表, 对整个验证集计算 mAP@0.5、
   mAP@0.5:0.95、Precision、Recall、F1 等标准指标.
3. **报告生成**: 将评价结果写入指定目录下的中文 Markdown 文件, 含指标表格、
   统计摘要和推理性能.

### 1.2 数据集策略

**无需修改 `data-process.py`**.

`data-process.py` 已按 80/20 比例将 ExDark 分为训练集（`data/Exdark/train.txt`,
约 5890 张）和验证集（`data/Exdark/val.txt`, 约 1473 张）.

经确认, `src/train.py` 在训练循环期间**从不读取**验证集做评价:
`val_path` 仅供训练启动前的 PNG 元数据清洗（`normalize_png_before_train`）.
因此 `data/Exdark/images/val/` 中的图像从未被模型"见过", 可直接作为公正评价集.

标签路径推导规则（无需额外文件列表）:

```
图像路径:  .../data/Exdark/images/val/Dog_2015_05395.jpg
标签路径:  .../data/Exdark/labels/val/Dog_2015_05395.txt
           规则: images/val/ -> labels/val/, 扩展名 -> .txt
```

### 1.3 与现有脚本的关系

| 脚本 | 用途 | 是否计算指标 |
|------|------|------------|
| `src/test.py` | 单图调试, 可视化四层流水线中间结果 | 无 |
| `src/test-compri.py` | 对比四种开关组合的可视化 | 无 |
| `src/eval.py` (本 PRD) | 量化评价, 可选报告生成 | Yes |
| `yolov7/test.py` | 原生 YOLOv7 格式 checkpoint 评价 | Yes, 但不支持 MSICN/FIE 开关 |

---

## 2. 宏参数完整规范

`src/eval.py` 顶部集中定义所有可配置参数, 分 5 组.
**不允许**将任何关键参数隐藏在函数默认值或 `**kwargs` 中.

### 2.1 模型参数

```python
# ─────────────────────────────────────────────
# 模型参数
# ─────────────────────────────────────────────

# 待评价的 checkpoint 路径（相对工作目录 github-ai2/）
# ENABLE_FIE=True  -> 应指向 stage_c_epoch_N.pt（含 FIE 联合微调）
# ENABLE_FIE=False -> 应指向 stage_a_epoch_N.pt（纯 YOLO 基线）
CHECKPOINT_PATH = "runs/icfie_yolo-5-13-a1-b1-c1/stage_c_epoch_1.pt"

# YOLOv7 原始权重路径（用于构建模型结构, 不影响评价权重）
YOLOV7_WEIGHTS_PATH = "yolov7/yolov7.pt"

# YOLOv7 网络结构配置文件
YOLOV7_CFG_PATH = "yolov7/cfg/training/yolov7.yaml"

# 是否启用 MSICN 光照矫正模块
ENABLE_MSICN = True

# 是否启用 FIE 特征交互增强模块
# 注意: ENABLE_FIE=False 时 CHECKPOINT_PATH 必须指向 stage_a checkpoint
ENABLE_FIE = True

# 推理设备（"cuda:0" / "cpu"）
DEVICE = "cuda:0"

# 推理图像尺寸（必须与训练时一致）
IMAGE_SIZE = 416
```

### 2.2 推理参数

```python
# ─────────────────────────────────────────────
# 推理参数
# ─────────────────────────────────────────────

# NMS 置信度阈值（低于此值的预测框在 NMS 前被滤除）
CONF_THRESHOLD = 0.25

# NMS IoU 阈值（用于抑制重叠框）
NMS_IOU_THRESHOLD = 0.45

# 是否评价时使用半精度（FP16）推理; 需要 GPU 支持, CPU 下自动降级为 FP32
USE_HALF_PRECISION = False
```

### 2.3 数据集参数

```python
# ─────────────────────────────────────────────
# 数据集参数
# ─────────────────────────────────────────────

# 评价模式: "batch"=批量评价, "single"=单图评价
MODE = "batch"

# 批量模式: val.txt 文件路径（每行一个绝对图像路径）
VAL_TXT = "data/Exdark/val.txt"

# 单图模式: 待评价图像路径
SINGLE_IMAGE_PATH = "test_sample/your_image.jpg"

# 单图模式: 对应标签路径（None=自动推导; 若推导失败则只做可视化, 不计算指标）
SINGLE_LABEL_PATH = None

# 类别数量
NUM_CLASSES = 12

# 类别名称列表（顺序必须与训练标签 class_id 严格一致）
CLASS_NAMES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike",
    "People", "Table",
]

# 批量模式 DataLoader 并行进程数
NUM_WORKERS = 4

# 批量模式每批图像数量（增大可提升吞吐量, 需有足够显存）
BATCH_SIZE = 8
```

### 2.4 指标参数

```python
# ─────────────────────────────────────────────
# 指标参数
# ─────────────────────────────────────────────

# mAP@0.5 使用的 IoU 阈值（主要指标）
EVAL_IOU_50 = 0.5

# mAP@0.5:0.95 使用的 IoU 阈值范围（COCO 风格, 共 10 个阈值）
EVAL_IOU_RANGE = [round(x * 0.05, 2) for x in range(10, 20)]  # [0.5, 0.55, ..., 0.95]
```

### 2.5 报告参数

```python
# ─────────────────────────────────────────────
# 报告参数
# ─────────────────────────────────────────────

# 是否生成 Markdown 报告
GENERATE_REPORT = True

# 报告输出目录（相对工作目录 github-ai2/）
REPORT_DIR = "report/eval_stage_c"

# 报告文件名
REPORT_FILENAME = "eval_report.md"

# 是否将带预测框的图像保存到报告目录（batch 模式下文件较多, 默认关闭）
# 单图模式下固定保存, 不受此宏控制
SAVE_PRED_IMAGES = False

# 单图模式下: 可视化图的文件名（保存在 REPORT_DIR/ 下）
SINGLE_PRED_IMAGE_FILENAME = "pred_visualization.jpg"
```

---

## 3. 推理流水线规范

### 3.1 继承约束（来自 prd2.md）

以下约束**强制执行**, 不得为"跑通"而绕过:

1. **图像预处理**: 必须使用 `yolov7/utils/datasets.py` 中的 `letterbox` 函数对图像
   进行缩放和灰边填充, 将输入图像等比例缩放到 `IMAGE_SIZE x IMAGE_SIZE`, 不得
   直接 `resize` (会改变目标宽高比).
2. **NMS**: 必须使用 `yolov7/utils/general.py` 中的 `non_max_suppression`, 参数
   `conf_thres=CONF_THRESHOLD, iou_thres=NMS_IOU_THRESHOLD`.
3. **坐标回投**: NMS 输出坐标基于 letterbox 后的图像尺寸, 须调用 `scale_coords`
   回投到**原始图像像素坐标**后再与 YOLO 标签对比.
4. **MSICN 输出值域**: 断言矫正图像 `corrected_image` 所有像素 ∈ [0, 1].
5. **FIE 输出通道**: 若 `project_after_fusion=True`, 断言 FIE 输出各尺度通道数
   分别等于 backbone 输出的 C（256/512/1024）.

### 3.2 模型加载流程

```python
# 伪代码示意（src/eval.py 须严格遵循此流程）

def build_eval_model(device):
    yolo_cfg = YOLOv7WrapperConfig(
        cfg_path=YOLOV7_CFG_PATH,
        weights_path=YOLOV7_WEIGHTS_PATH,
        num_classes=NUM_CLASSES,
    )
    backbone_neck, detect_head, stride = build_yolov7_components(yolo_cfg, device)

    model_config = ICFIEYOLOConfig(
        enable_msicn=ENABLE_MSICN,
        enable_fie=ENABLE_FIE,
    )
    model = ICFIEYOLO(backbone_neck=backbone_neck, detect_head=detect_head,
                      config=model_config)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), stride
```

加载后必须替换 `class_names` 为 `CLASS_NAMES`（`build_yolov7_components` 返回的
`class_names` 是 COCO 80 类, 直接用于标签打印会显示错误类别名）.

### 3.3 单张图像推理步骤

```
1. 原始图像（HxWx3, BGR, uint8）
2. letterbox(img, new_shape=(IMAGE_SIZE, IMAGE_SIZE), stride=stride)
   -> padded_img（uint8）, ratio, pad
3. padded_img / 255.0  # 归一化到 [0, 1]
4. torch.from_numpy(padded_img).permute(2,0,1).unsqueeze(0).float()
   -> tensor（1,3,IMAGE_SIZE,IMAGE_SIZE）
5. model(tensor)  # 返回 decoded_predictions（经 IDetect 解码后的 anchor-based 预测）
6. non_max_suppression(decoded_predictions, CONF_THRESHOLD, NMS_IOU_THRESHOLD)
   -> detections: list[Tensor(N,6)]  # x1,y1,x2,y2,conf,class_id（letterbox 坐标）
7. scale_coords((IMAGE_SIZE,IMAGE_SIZE), detections[0][:,:4], original_img.shape[:2])
   -> 回投到原图坐标
```

---

## 4. 指标计算规范

### 4.1 指标体系

| 指标 | 含义 | 来源 |
|------|------|------|
| **mAP@0.5** | IoU=0.5 下各类 AP 均值（主要指标, 与论文直接对比） | 论文 Table 2 |
| **Recall@0.5** | IoU=0.5 下全类平均召回率 | 论文 Table 2 |
| **mAP@0.5:0.95** | IoU 0.5~0.95 步进 0.05 的平均 mAP（COCO 风格, 更严格） | 补充指标 |
| **Precision@0.5** | IoU=0.5 下全类平均精确率 | 补充指标 |
| **F1@0.5** | Precision 与 Recall 的调和均值 | 补充指标 |
| **各类 AP@0.5** | 12 类各自的 AP 值 | 用于分析哪类最难检测 |
| **AP 中位数@0.5** | 12 类 AP 的中位数（抵抗极端值影响, 反映典型类别表现） | 统计摘要 |
| **AP 标准差@0.5** | 12 类 AP 的标准差（反映类别间差距） | 统计摘要 |
| **推理耗时均值** | 每张图从输入张量到 NMS 输出的平均耗时 (ms) | 性能摘要 |
| **推理耗时中位数** | 同上中位数（比均值更抗离群值干扰） | 性能摘要 |

### 4.2 TP/FP/FN 判定规则

1. 以置信度降序排列所有预测框.
2. 对每个预测框, 查找同类未匹配 GT 框中 IoU 最大者.
3. 若最大 IoU ≥ `EVAL_IOU_50`, 标记为 TP（该 GT 框标为已匹配, 不可再用）;
   否则标为 FP.
4. 未被任何预测框匹配的 GT 框计为 FN.

`mAP@0.5:0.95` 在 `EVAL_IOU_RANGE` 的每个阈值下独立执行上述判定, 取 10 次 AP 均值.

### 4.3 AP 计算实现

**复用** `yolov7/utils/metrics.py` 中的 `ap_per_class` 函数, 不自行实现 AP 计算:

```python
from utils.metrics import ap_per_class

# tp: shape (N, len(EVAL_IOU_RANGE)), N = 所有预测框总数
# conf: shape (N,)
# pred_cls: shape (N,)
# target_cls: shape (M,)  M = 所有 GT 框总数
p, r, ap, f1, unique_classes = ap_per_class(
    tp, conf, pred_cls, target_cls,
    v5_metric=False, plot=False,
)
# ap shape: (nc, len(EVAL_IOU_RANGE))
# ap[:,0] -> 各类 AP@0.5（EVAL_IOU_RANGE[0] = 0.5）
# ap.mean(1) -> 各类 mAP@0.5:0.95
```

`ap_per_class` 的 `tp` 参数支持多 IoU 阈值（列数 = 阈值数）, 直接传入
`len(EVAL_IOU_RANGE)` 列, 一次调用同时获得 mAP@0.5 和 mAP@0.5:0.95.

### 4.4 统计摘要计算

```python
import numpy as np

ap_50 = ap[:, 0]  # 各类 AP@0.5, shape (nc,)

summary = {
    "mAP@0.5":       float(ap_50.mean()),
    "mAP@0.5:0.95":  float(ap.mean()),
    "AP_median@0.5": float(np.median(ap_50)),
    "AP_std@0.5":    float(np.std(ap_50)),
    "Precision":     float(p.mean()),  # p: shape (nc,)
    "Recall":        float(r.mean()),  # r: shape (nc,)
    "F1":            float(f1.mean()), # f1: shape (nc,)
}
```

推理耗时统计（批量模式）:

```python
latencies_ms = [...]  # 每张图的推理耗时（仅含模型前向+NMS, 不含数据加载）
latency_summary = {
    "mean_ms":   float(np.mean(latencies_ms)),
    "median_ms": float(np.median(latencies_ms)),
}
```

---

## 5. 中文 Markdown 报告规范

### 5.1 格式约束

- 内容使用**中文**, 标点符号使用**英文符号**（`,` `.` `:` `(` `)` 等）.
  不使用中文标点（不出现 `，` `。` `：` `（` `）`）.
- 报告路径: `{REPORT_DIR}/{REPORT_FILENAME}`, 目录不存在时自动创建.
- 时间格式: `YYYY-MM-DD HH:MM:SS`（本地时间）.

### 5.2 报告结构模板

```markdown
# ICFIE-YOLO 目标检测评价报告

## 评价概述

| 项目 | 值 |
|------|----|
| 评价时间 | 2026-05-15 12:00:00 |
| Checkpoint | runs/.../stage_c_epoch_1.pt |
| ENABLE_MSICN | True |
| ENABLE_FIE | True |
| 数据集 | data/Exdark/val.txt |
| 图像数量 | 1473 |
| 图像尺寸 | 416 |
| CONF_THRESHOLD | 0.25 |
| NMS_IOU_THRESHOLD | 0.45 |
| 设备 | cuda:0 |

## 总体检测指标

| 指标 | 值 |
|------|----|
| mAP@0.5 | 0.XXXX |
| mAP@0.5:0.95 | 0.XXXX |
| Precision | 0.XXXX |
| Recall | 0.XXXX |
| F1 | 0.XXXX |

## 各类别 AP@0.5 详细结果

| 排名 | 类别 | AP@0.5 |
|------|------|--------|
| 1 | Car | 0.XXXX |
| ... | ... | ... |
| 12 | Boat | 0.XXXX |

（按 AP@0.5 降序排列）

## 统计摘要

| 统计量 | 值 |
|--------|----|
| AP@0.5 均值 | 0.XXXX |
| AP@0.5 中位数 | 0.XXXX |
| AP@0.5 标准差 | 0.XXXX |
| 推理耗时均值 | XX.X ms |
| 推理耗时中位数 | XX.X ms |

## 附注

评价使用 YOLOv7 标准推理链路: letterbox -> 模型前向 -> NMS -> scale_coords.
AP 计算复用 yolov7/utils/metrics.py 中的 ap_per_class 函数.
```

批量模式下若 `SAVE_PRED_IMAGES=True`, 在报告末尾追加可视化图路径列表（相对报告目录）.

### 5.3 单图模式额外内容

单图模式在报告最后追加以下节：

```markdown
## 单图评价结果

| 项目 | 值 |
|------|----|
| 图像路径 | test_sample/your_image.jpg |
| 原始尺寸 | 640x480 |
| 预测框数量 | N |
| GT 框数量 | M |
| TP | X |
| FP | Y |
| FN | Z |
| 可视化图 | pred_visualization.jpg |
```

---

## 6. 输出文件结构

```
report/{REPORT_DIR}/
    eval_report.md                   # 主报告（中文 Markdown）
    pred_images/                     # 仅 SAVE_PRED_IMAGES=True 时生成
        Dog_2015_05395.jpg
        ...
    pred_visualization.jpg           # 仅 single 模式生成
```

---

## 7. 数值安全约束

以下约束继承自 prd2.md 中的 "强制校验项", **不得为了跑通而删除**:

1. `EVAL_IOU_RANGE` 长度必须 ≥ 1 且所有值 ∈ [0, 1].
2. `tp` 数组传入 `ap_per_class` 前, 其列数必须等于 `len(EVAL_IOU_RANGE)`.
3. `scale_coords` 回投后的坐标须 clamp 到原图尺寸范围内（`x ∈ [0, img_w], y ∈ [0, img_h]`）.
4. `corrected_image` 推断后断言值域 ∈ [0, 1]（MSICN 的硬约束, 见 prd2.md 3.1 节）.
5. 若验证集为空（`val.txt` 包含 0 条有效路径）, 脚本须抛出 `ValueError` 而非静默
   返回全零指标.
6. 若某类在验证集中没有任何 GT 框, 该类 AP 记为 0.0 并在终端打印警告, 不影响其他类计算.

---

## 8. 快速验证方法

### 8.1 冒烟验证（单图模式, 不需要 GPU）

```bash
cd /home/pushk3n/github-ai2
# 修改 src/eval.py 顶部宏: MODE="single", DEVICE="cpu"
python src/eval.py
```

预期输出: 终端打印单图推理结果, `report/eval_stage_c/eval_report.md` 文件生成.

### 8.2 小批量验证（批量模式先跑 10 张）

修改 `src/eval.py` 中 `BATCH_SIZE=1`, 在 `val.txt` 路径列表读取后截断为前 10 行,
确认指标字典格式正确后再开放全量评价.

### 8.3 与 yolov7/test.py 对比验证

使用纯 YOLO 基线 checkpoint 时（`ENABLE_MSICN=False, ENABLE_FIE=False`）,
`src/eval.py` 的 mAP@0.5 应与在相同 checkpoint 和相同验证集上运行 `yolov7/test.py`
的结果高度接近（差异 ≤ 0.5pp, 误差来自浮点精度和图像加载实现差异）.
若差异过大, 优先排查 letterbox 参数（`auto=False, stride=32`）是否与 yolov7 原版一致.

---

## 9. 依赖关系

| 依赖 | 来源 | 作用 |
|------|------|------|
| `src/icfie_yolo.py` | 本项目 | `ICFIEYOLO`, `ICFIEYOLOConfig` |
| `src/yolo_wrapper.py` | 本项目 | `build_yolov7_components`, `YOLOv7WrapperConfig` |
| `src/detect.py` | 本项目 | `YOLOv7DetectHeadAdapter` |
| `src/msicn.py` | 本项目 | `MSICNConfig` |
| `src/fie.py` | 本项目 | `MultiScaleFIEConfig`, `FIEBlockConfig` |
| `yolov7/utils/datasets.py` | YOLOv7 | `letterbox` |
| `yolov7/utils/general.py` | YOLOv7 | `non_max_suppression`, `scale_coords` |
| `yolov7/utils/metrics.py` | YOLOv7 | `ap_per_class` |
| `yolov7/utils/plots.py` | YOLOv7 | `plot_one_box`（可视化预测框） |

从 `src/` 目录运行时, YOLOv7 工具通过 `sys.path.append(YOLOV7_ROOT)` 引入（与
`src/train.py` 的 import 惯例保持一致, 见 [import-paths.md](../memories/repo/import-paths.md)）.

---

## 10. 待实现项

| 优先级 | 文件 | 说明 |
|--------|------|------|
| P0 | `src/eval.py` | 本 PRD 对应的评价脚本实现 |
| P1 | `report/` | 报告输出目录（首次运行时自动创建） |
