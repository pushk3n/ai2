# ICFIE-YOLO 训练说明

本文档完整说明 ICFIE-YOLO 的训练原理、参数体系、可训练参数规模以及训练效果评判方法，对应论文《基于ICFIE-YOLO的低照度图像目标检测方法》第 3、4 节。

---

## 1. 训练依据

### 1.1 问题背景

低照度图像存在三个核心困难：全局亮度不足、局部光照不均、特征空间噪声偏多。直接将正常照度下训练的检测器迁移到低照度场景，mAP 会出现显著下滑。

### 1.2 论文方法论

ICFIE-YOLO 采用**图 1(c) 策略**：将光照矫正网络（MSICN）与检测器（YOLOv7）集成为一个整体，MSICN 只用检测损失（无图像重建损失）进行优化，保证优化方向完全由检测任务决定。  
这与"先增强再检测"或"增强+检测联合损失"的方案有本质区别：  
- 无需成对正常/低照度图像数据集  
- MSICN 的优化目标与检测精度完全对齐  
- 光照矫正的"方式"由检测损失自动决定，不是人工设计的增强目标

### 1.3 损失函数

训练全程只使用 **YOLOv7 标准检测损失**，分三项：

| 损失项 | 含义 | 对应字段 |
|--------|------|----------|
| Box Loss | 边界框定位损失，基于 CIoU | `box_loss` |
| Objectness Loss | 目标置信度损失，BCE | `obj_loss` |
| Classification Loss | 类别分类损失，BCE | `cls_loss` |
| Total Loss | 三项加权求和，优化目标 | `total_loss` |

损失计算由 `yolov7/utils/loss.py` 中的 `ComputeLoss`（或 `ComputeLossOTA`，对应 `use_ota_loss: true`）完成。

### 1.4 训练数据集

| 数据集 | 类别数 | 图片数 | 训练/测试划分 |
|--------|--------|--------|--------------|
| ExDark | 12 | 7363 | 5890 / 1473（80:20） |
| DarkFace | 1（人脸） | 6000 | 5400 / 600 |

ExDark 12 个类别：Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table。  
当前仓库仅集成了 ExDark，对应 `configs/train.yaml`。

---

## 2. 三阶段训练流程

ICFIE-YOLO 的训练实际包含一个初始化快照步骤和三个训练阶段。训练过程中对不同模块的 `requires_grad` 进行**显式设置**（不允许仅靠 optimizer 参数列表模拟冻结）。

### 初始化快照：加载预训练权重（无训练）

- 直接加载 `yolov7/yolov7.pt`（在 MS COCO 上预训练的 YOLOv7 权重），视为正常照度下的检测基线已具备。
- 保存快照 `runs/<run_dir>/stage_a_loaded.pt` 作为起点记录。

这一步只记录迁移学习起点，不参与梯度更新。

### 阶段 A：纯 YOLO 基线训练

**目标**：在 ExDark 上先训练一个不经过 MSICN、也不经过 FIE 的纯 YOLO 基线，使 backbone 和 detect head 先适配 12 类低照度检测任务。

- 前向路径为 `image -> backbone_neck -> detect_head`。
- MSICN 与 FIE 均跳过，不参与前向，也不参与优化。
- 该阶段生成的 `stage_a_epoch_N.pt` 是 `ENABLE_FIE=False` 时应使用的 checkpoint。

| 模块 | requires_grad | 模式 |
|------|--------------|------|
| MSICN | False | eval |
| YOLOv7 BackboneNeck | True | train |
| FIE | False | eval |
| Detect Head | True | train |

### 阶段 B：基于检测损失的光照对齐训练

**目标**：用低照度训练集图像，通过检测损失反传，让 MSICN 学习"让后续检测更容易"的光照映射。

- 冻结阶段 A 已适配好的 BackboneNeck 与 DetectHead，仅优化 MSICN。
- 仅 MSICN 的参数参与梯度更新。
- 本阶段不需要成对正常照度图像，**纯检测损失驱动**。

| 模块 | requires_grad | 模式 |
|------|--------------|------|
| MSICN | **True** | train |
| YOLOv7 BackboneNeck | False | eval |
| FIE | False | eval |
| Detect Head | False | eval |

### 阶段 C：端到端联合微调

**目标**：在低照度训练集上对全模型（含 FIE）进行联合微调，进一步提升 mAP 和 Recall。

- 解冻所有模块，所有参数均参与梯度更新。
- FIE 在此阶段**首次**真正参与训练（阶段 B 时 FIE 冻结）。
- 学习率切换为较小的 `stage_c_lr`（默认 `1e-4`），防止预训练权重被破坏。

| 模块 | requires_grad | 模式 |
|------|--------------|------|
| MSICN | True | train |
| YOLOv7 BackboneNeck | True | train |
| FIE | **True** | train |
| Detect Head | True | train |

---

## 3. 训练参数说明

所有参数集中在 `configs/train.yaml` 中，通过 `src/train_config.py` 中的 dataclass 显式注入，不依赖隐式默认值。

### 3.1 顶层参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `seed` | 42 | 随机种子，固定 torch/numpy/random，保证可复现 |
| `run_dir` | `runs/icfie_yolo-5-13` | 训练产物保存目录 |
| `hyp_path` | `yolov7/data/hyp.scratch.p5.yaml` | YOLOv7 损失函数超参文件（含 anchor、损失权重等） |
| `project_after_fusion` | `true` | FIE 输出是否经 1×1 Conv 投影回原始通道数（对接标准检测头） |
| `resume_from` | `null` | 断点续训 checkpoint 路径；为 `null` 时默认从头训练 |
| `normalize_png_before_train` | `true` | 训练启动前是否按需清洗带异常元数据的 PNG，规避 libpng 崩溃 |
| `save_every` | 1 | 每隔 N 个 epoch 保存一次检查点 |
| `use_ota_loss` | `false` | 是否使用 YOLOv7 OTA（最优传输分配）损失 |
| `cache_images` | `false` | 是否缓存图像到内存 |
| `rect` | `false` | 是否使用矩形推理（按长宽比分 batch） |

`normalize_png_before_train=true` 时，训练入口会在创建 DataLoader 前扫描 `train_path/val_path` 中引用到的 PNG，只有当文件含有 `icc_profile/chromaticity/srgb/gamma` 等可疑元数据时才执行重编码，并在终端打印处理进度。

### 3.2 数据集参数（`dataset`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `train_path` | ExDark train.txt | 训练集路径（图片目录或包含图片路径的 .txt 文件） |
| `val_path` | ExDark val.txt  | 验证集路径（当前 train.py 保留字段，供 eval 复用） |
| `num_classes` | 12 | 类别数量 |
| `class_names` | Bicycle 等 12 类 | 类别名列表，不填时自动生成 class_0...class_N |
| `image_size` | 416 | 输入图像尺寸（论文统一使用 416×416） |
| `single_cls` | `false` | 是否将所有类别合并为单类 |

### 3.3 YOLOv7 模型参数（`yolo`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `cfg_path` | `yolov7/cfg/training/yolov7.yaml` | YOLOv7 网络结构定义文件 |
| `weights_path` | `yolov7/yolov7.pt` | 预训练权重路径（初始化快照步骤加载） |
| `num_classes` | 12 | 检测类别数，需与 `dataset.num_classes` 一致 |

### 3.4 硬件参数（`hardware`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `device` | `cuda:0` | 训练设备（单卡用 `cuda:0`，DDP 用 `cuda`） |
| `use_amp` | `true` | 自动混合精度（FP16），节省显存约 30~40% |
| `batch_size` | 4 | 单卡每次处理的图片数量 |
| `accumulate_steps` | 4 | 梯度累积步数，等效 batch = batch_size × accumulate_steps = 16 |
| `num_workers` | 4 | DataLoader 的并行读取进程数 |
| `pin_memory` | `true` | 锁页内存，加速 CPU→GPU 数据传输 |
| `ddp` | `false` | 是否启用 DistributedDataParallel（多卡训练） |
| `use_grad_checkpoint` | `true` | 对 BackboneNeck 启用梯度检查点，阶段 C 显存受限时使用 |

### 3.5 优化器参数（`optimizer`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `stage_a_lr` | 0.0001 | 阶段 A 的初始学习率；用于纯 YOLO 基线训练，默认采用保守值避免破坏预训练 backbone |
| `lr` | 0.01 | Adam 初始学习率，用于阶段 A 和阶段 B |
| `stage_c_lr` | 0.0001 | 阶段 C 的初始学习率（全模型微调，应较小） |
| `min_lr` | 0.00001 | CosineAnnealingLR 退火终点学习率 |
| `weight_decay` | 0.0005 | Adam 权重衰减（L2 正则化系数） |
| `beta1` | 0.937 | Adam 一阶矩参数，YOLOv7 默认值 |
| `beta2` | 0.999 | Adam 二阶矩参数 |

调度策略：每个阶段内使用 **CosineAnnealingLR**，从 `lr`（或 `stage_c_lr`）余弦衰减到 `min_lr`。

说明：

- 阶段 A 当前默认使用 `stage_a_lr=1e-4`，原因是直接用 `1e-2` 级别学习率去微调整个 YOLOv7 backbone，极易在第一轮就把特征分布训塌，表现为不同输入得到几乎相同的 P3/P4/P5 特征图与极低的 objectness 分数。
- 阶段 B 仍沿用 `lr`，因为此时只训练 MSICN，参数规模小且不会直接破坏预训练 backbone。

### 3.6 训练阶段轮数（`schedule`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `stage_a_epochs` | 5 | 阶段 A 训练的 epoch 数，用于建立纯 YOLO 基线 |
| `stage_b_epochs` | 5 | 阶段 B 训练的 epoch 数（当前配置值，正式训练建议 10~30） |
| `stage_c_epochs` | 5 | 阶段 C 训练的 epoch 数（当前配置值，正式训练建议 10~30） |

### 3.7 可视化参数（`visualization`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `batch_log_interval` | 25 | 每隔 N 个 batch 写入一次 batch-level 损失记录 |

---

## 4. 可训练参数规模

ICFIE-YOLO 的参数可分为两部分：预训练的 YOLOv7 部分和新增的 MSICN/FIE 部分。

### 4.1 YOLOv7 BackboneNeck + Detect Head

YOLOv7 标准版本参数量约 **3690 万（36.9M）**，来自官方预训练权重 `yolov7.pt`。  
这部分参数在阶段 A 和阶段 C 参与更新，在阶段 B 冻结。

### 4.2 MSICN（新增模块）

MSICN 是本文新设计的核心模块，全参数均随机初始化后由检测损失驱动学习。

| 子模块 | 结构 | 参数量估算 |
|--------|------|-----------|
| **IFE** | 6 层 3×3 Conv（通道 3→4→4→4→4→8→8，级联输入） | ~6K |
| **GIC** | 2 层 5×5 空洞 Conv（32→32，dilation=3）+ 1×1 Conv（→3）+ AvgPool | ~16K |
| **LIC** | 精细分支：2 层 3×3 Conv（32→32→15）；上下文分支：3 层 3×3 Conv + AvgPool（32→32→15） | ~30K |
| **NLIS** | **零参数**，纯数值迭代 | 0 |
| **MSICN 合计** | — | **约 52K** |

> IFE 每层输入通道为前一层输出通道，其通道依次为 (3→4), (4→4), (4→4), (4→4), (4→8), (8→8)，6 层输出在通道维度拼接为 32 通道。NLIS 不含任何可训练参数，这是论文明确规定的设计。

### 4.3 FIE（新增模块）

FIE 对 3 个尺度（P3/P4/P5）分别独立应用，每个尺度含空间分支和通道分支。

| 分支 | 结构（以 P3，C=256 为例） | 参数量（单尺度，C=256） |
|------|--------------------------|----------------------|
| 空间分支 Q/K | 1×1 Conv（256→128），×2 | 2 × (256×128 + 128) ≈ 66K |
| 空间分支 V | 1×1 Conv（256→256） | 256×256 + 256 ≈ 66K |
| 通道分支 Q/K/V | 3×3 Conv，stride=2（256→256），×3 | 3 × (256×256×9 + 256) ≈ 1.77M |
| 投影层（`project_after_fusion=True`） | 1×1 Conv（768→256） | 768×256 + 256 ≈ 197K |

三个尺度（C = 256/512/1024）的 FIE 总参数量约 **20~25M**（主要来自通道分支的 3×3 Conv，通道数越大贡献越大）。

### 4.4 全模型参数量汇总

| 模块 | 参数量 | 阶段 A 可训练 | 阶段 B 可训练 | 阶段 C 可训练 |
|------|--------|--------------|--------------|--------------|
| YOLOv7 BackboneNeck | ~36.9M | ✓ | ✗ | ✓ |
| Detect Head | ~0.5M | ✓ | ✗ | ✓ |
| MSICN | ~52K | ✗ | **✓** | ✓ |
| FIE | ~20~25M | ✗ | ✗ | ✓ |
| **合计** | **~58~63M** | ~37.4M | ~52K | 全部 |

> 阶段 B 只训练约 52K 参数，其余约 58M 参数完全冻结，这使得阶段 B 的显存占用极低、收敛速度快。

---

## 5. 训练输出文件

训练产物保存在 `configs/train.yaml` 中 `run_dir` 指定的目录下（默认 `runs/icfie_yolo-5-13/`）：

| 文件 | 含义 |
|------|------|
| `stage_a_loaded.pt` | 初始化快照，即加载完 `yolov7.pt` 后尚未开始训练时的完整模型状态 |
| `stage_a_epoch_N.pt` | 阶段 A 第 N 个 epoch 结束时的纯 YOLO 基线检查点，含模型/优化器/调度器状态 |
| `stage_b_epoch_N.pt` | 阶段 B 第 N 个 epoch 结束时的检查点，含模型/优化器/调度器状态 |
| `stage_c_epoch_N.pt` | 阶段 C 第 N 个 epoch 结束时的检查点，含模型/优化器/调度器状态 |
| `training_metrics.csv` | 每个 epoch 的四项损失 + 学习率 + 耗时 |
| `training_batch_metrics.csv` | batch 粒度的损失记录（每 `batch_log_interval` 个 batch 写一次） |
| `training_metrics.png` | 训练过程可视化图（box/obj/cls/total 四图） |
| `train_config_snapshot.yaml` | 本次训练的配置文件快照，便于后续复现 |

补充说明：

- `stage_a_loaded.pt` 只保存初始化模型权重，不包含真正训练过的 optimizer/scheduler 状态，更适合作为实验起点记录。
- 真正用于断点续训的 checkpoint 应优先选择 `stage_a_epoch_N.pt`、`stage_b_epoch_N.pt` 或 `stage_c_epoch_N.pt`。

---

## 6. 评判训练效果的指标

### 6.1 训练过程中：损失曲线

训练的直接可观测信号是四个损失项的下降趋势。合格的训练曲线应具备：

- `total_loss` 在阶段 B 中持续下降并趋于平稳。
- 阶段 C 开始时损失可能出现短暂波动（全模型解冻初期适应），随后应继续下降。
- 阶段 B 与阶段 C 之间的`box_loss`不应出现大幅跳升，否则说明光照矫正对定位特征有破坏。

读取方式：

```bash
# 终端实时观测
python src/train.py --config configs/train.yaml

# 查看 CSV
cat runs/icfie_yolo-5-13/training_metrics.csv
```

### 6.2 训练结束后：目标检测指标

训练完成后，用以下命令在 ExDark 验证集上计算标准目标检测评估指标：

```bash
# 使用 YOLOv7 官方 test.py 评估（推荐）
cd yolov7
python test.py \
  --data ../configs/train.yaml \
  --weights ../runs/icfie_yolo-5-13/stage_c_epoch_N.pt \
  --img-size 416 \
  --batch-size 4 \
  --task val \
  --name icfie_eval
```

核心评估指标说明：

| 指标 | 含义 | 论文基准（ExDark） |
|------|------|-----------------|
| **mAP@0.5** | IoU 阈值 0.5 下各类平均精度均值，最常用主指标 | 较 YOLOv7 基线 +2.1pp 以上 |
| **mAP@0.5:0.95** | IoU 从 0.5 到 0.95 步长 0.05 的平均 mAP，更严格 | — |
| **Recall** | 召回率，反映对低照度目标的检出能力 | 较基线 +2.6pp，较现有低照度方法 +4.2pp |
| **Precision** | 精确率，反映误检率 | — |

### 6.3 消融实验参考

通过修改 `configs/train.yaml` 中的两个开关字段可以独立评估各模块的贡献：

| 实验配置 | 含义 |
|----------|------|
| `enable_msicn: false`（在 `ICFIEYOLOConfig` 中关闭） | 移除 MSICN，验证光照矫正的贡献 |
| `enable_fie: false` | 移除 FIE，验证特征交互增强的贡献 |
| `use_ota_loss: true` | 使用 OTA 损失替换标准损失，对比效果 |

> 当前 `src/icfie_yolo.py` 中的 `ICFIEYOLOConfig` 支持 `enable_msicn` 和 `enable_fie` 两个开关，直接在代码中修改或将其暴露到 YAML 即可进行消融测试。

补充说明：

- `enable_fie: false` 的推理/测试场景应使用 `stage_a_epoch_N.pt`，而不是 `stage_c_epoch_N.pt`。
- `stage_c_epoch_N.pt` 的 detect head 已经在 FIE 输出特征上联合微调，关闭 FIE 后其输入分布会发生变化，常见现象是 objectness 与分类分数整体偏低，最终被 NMS 全部滤掉。

### 6.4 判断训练质量的综合标准

建议按以下顺序评判一次训练是否成功：

1. **无 NaN / Inf 损失**：训练脚本在出现非有限损失时会立即抛出 RuntimeError，若训练正常结束说明数值稳定。
2. **阶段 A 基线可用，阶段 B 继续收敛**：阶段 A 训练后纯 YOLO 基线应能正常输出检测结果；进入阶段 B 后，`total_loss` 相比阶段 A 后期不应明显恶化，通常会继续下降。
3. **阶段 C mAP@0.5 ≥ 基线**：用 YOLOv7 官方评估脚本，确认 `stage_c_epoch_N.pt` 的 mAP@0.5 不低于直接在 ExDark 上微调的 YOLOv7 基线。
4. **Recall 提升**：低照度目标检测中 Recall 是比 Precision 更重要的指标（漏检代价高），论文目标是 Recall 比现有方法提升 ≥4.2pp。

---

## 7. 启动训练

### 单卡训练（4060 等单卡环境）

```bash
cd /home/pushk3n/github-ai2
python src/train.py --config configs/train.yaml
```

### 断点续训

训练入口支持两种断点续训方式。

方式 1：命令行显式指定 checkpoint。

```bash
cd /home/pushk3n/github-ai2
python src/train.py --config configs/train.yaml --resume runs/your_run_dir/stage_c_epoch_2.pt
```

方式 2：自动从当前 `run_dir` 中选择最新 checkpoint。

```bash
cd /home/pushk3n/github-ai2
python src/train.py --config configs/train.yaml --resume
```

也可以在 YAML 顶层配置：

```yaml
resume_from: runs/your_run_dir/stage_c_epoch_2.pt
```

续训行为说明：

1. `--resume` 优先级高于 YAML 中的 `resume_from`。
2. `--resume` 不带路径时，会自动在 `run_dir` 中查找最新的 `stage_*_epoch_*.pt`。
3. 续训时会恢复模型参数、优化器状态、学习率调度器状态以及 GradScaler 状态。
4. 训练入口会根据 checkpoint 中记录的 `stage/epoch` 自动跳过已经完成的前置阶段，只从当前阶段的下一轮继续训练。
5. 如果 checkpoint 中缺少优化器或调度器状态（例如 `stage_a_loaded.pt`），训练仍可启动，但会以当前配置重新初始化这些运行时状态。

### 多卡 DDP 训练（双 3090 等多卡环境）

```bash
cd /home/pushk3n/github-ai2
torchrun --nproc_per_node=2 src/train.py --config configs/train.yaml
```

DDP 模式需同时在 `configs/train.yaml` 中设置 `hardware.ddp: true`，并将 `hardware.device` 改为 `cuda`。

### 正式训练建议参数

将 `configs/train.yaml` 中以下字段从冒烟验证值调整为正式训练值：

```yaml
schedule:
  stage_a_epochs: 20   # 建议 10~30，先把纯 YOLO baseline 训稳
  stage_b_epochs: 30   # 建议 10~30，视收敛曲线决定是否提前停止
  stage_c_epochs: 20   # 建议 10~30
hardware:
  batch_size: 4        # 4060 单卡保持 4；3090 单卡可提升到 16
  accumulate_steps: 4  # 等效 batch=16，3090 可降为 1
```
