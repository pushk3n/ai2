# ICFIE-YOLO 低照度图像目标检测

本项目复现论文《基于ICFIE-YOLO的低照度图像目标检测方法》（秦嘉奇、江泽涛、雷晓春，电子学报 2025）。  
论文提出将多尺度光照矫正网络（MSICN）与特征交互增强检测头（FIE）集成到 YOLOv7 框架，在低照度数据集 ExDark 上实现端到端训练，mAP 较现有方法提升 2.1 个百分点以上。

**核心模块：**
- **MSICN**（`src/msicn.py`）：IFE → GIC/LIC → NLIS，无参考图的多尺度光照矫正
- **YOLOv7 BackboneNeck**（`src/yolo_wrapper.py`）：加载官方预训练权重，提取 P3/P4/P5 多尺度特征
- **FIE**（`src/fie.py`）：空间交互增强 + 通道交互增强，抑制低照度特征噪声
- **Detect Head**（`src/detect.py`）：适配 FIE 输出通道的检测头

**训练策略：**　三阶段对齐训练 — 保存预训练初始化快照 → 纯 YOLO 基线训练（阶段 A）→ 仅优化 MSICN（阶段 B）→ 全模型微调（阶段 C），详细说明见 [docs/train.md](docs/train.md)。

---

## 环境配置

### 配置虚拟环境
#### 这里以conda为例

```bash 
conda create -n ai2 python=3.10 -y
conda activate ai2

```

### 安装 PyTorch

请不要直接在本项目里无脑安装最新版 `torch`，需要和本机 NVIDIA 驱动版本匹配。

如果你是单卡 4060，且当前驱动较旧、无法支持 CUDA 13，对应推荐先安装 `cu121` 版 PyTorch：

```bash
conda activate ai2
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果你已经安装了与驱动匹配的 PyTorch，可以跳过这一步。

### 克隆 yolov7 仓库
```bash
# 请在项目根目录下克隆 yolov7 仓库, yolov7/ 已经被添加到 .gitignore 中，确保不会被提交到版本控制系统
git clone https://github.com/WongKinYiu/yolov7.git
```

### 下载 yolov7 权重文件
```bash
# 请将 yolov7 权重文件 yolov7.pt 下载到 yolov7/ 目录下， 以便后续推理使用
# 下载链接: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

### 安装依赖
```bash     
conda activate ai2
pip install -r yolov7/requirements.txt
pip install -r requirements.txt
```

说明：

1. 根目录 [requirements.txt](requirements.txt) 不再直接声明 `torch`，避免把你已经安装好的 CUDA 匹配版本覆盖掉。
2. 训练依赖仍然以 [yolov7/requirements.txt](yolov7/requirements.txt) 为主，根目录依赖补充本项目脚本直接使用的库。

### 解压测试集
```bash
unzip test_images.zip -d test_images
```

### ExDark 数据预处理

仓库根目录提供了 [data-process.py](data-process.py)，用于把原始 ExDark 数据集转换为 YOLO 训练格式，并自动回写 [configs/train.yaml](configs/train.yaml) 的数据集路径。

先修改脚本顶部宏定义：

1. `EXDARK_IMG_DIR`：ExDark 图像目录。
2. `EXDARK_ANNO_DIR`：ExDark 标注目录。

ExDark 原始标注不是 YOLO 格式，实际格式为：

```text
% bbGt version=3
ClassName x y w h 0 0 0 0 0 0 0
```

其中：

1. `ClassName` 是类别名。
2. `x y` 是左上角绝对像素坐标。
3. `w h` 是绝对像素宽高。

脚本会自动转换为 YOLO 需要的归一化格式：

```text
class_id x_center y_center width height
```

运行命令：

```bash
cd /home/pushk3n/github-ai2
python data-process.py
```

处理完成后会生成：

1. `data/Exdark/images/train|val`
2. `data/Exdark/labels/train|val`
3. `data/Exdark/train.txt`
4. `data/Exdark/val.txt`

并自动更新 [configs/train.yaml](configs/train.yaml) 中的 `dataset.train_path` 和 `dataset.val_path`。

### 训练介绍

完整的训练原理、参数说明与结果评判方法见 **[docs/train.md](docs/train.md)**，这里仅列出快速参考。

`src/train.py` 是 ICFIE-YOLO 三阶段训练的唯一入口，训练参数统一通过 `configs/train.yaml` 管理。

**三阶段概述：**

| 阶段 | 可训练模块 | 目标 |
|------|-----------|------|
| 初始化快照 | — | 加载 YOLOv7 `yolov7.pt` 预训练权重，保存 `stage_a_loaded.pt` |
| 阶段 A | YOLO BackboneNeck + Detect Head | 在 ExDark 上训练纯 YOLO 基线，生成 `stage_a_epoch_N.pt`，供 `ENABLE_FIE=False` 使用 |
| 阶段 B | MSICN | 冻结阶段 A 训练好的纯 YOLO 检测器，仅用检测损失驱动光照矫正网络 |
| 阶段 C | 全模型 | 解冻所有模块，端到端联合微调 |

**训练过程中监控指标：**

每个 epoch 在终端和 `runs/<run_dir>/training_metrics.csv` 中记录四个损失项：

| 字段 | 含义 |
|------|------|
| `box_loss` | 边界框定位损失（IoU-based） |
| `obj_loss` | 目标置信度损失（objectness） |
| `cls_loss` | 分类损失（cross entropy） |
| `total_loss` | 三项之和，是优化目标 |

训练结束后用 `src/yolo_test.py` 或 YOLOv7 官方 `test.py` 评估 **mAP@0.5**、**mAP@0.5:0.95** 和 **Recall**。

**关键配置项（`configs/train.yaml`）：**

```yaml
dataset:
  image_size: 416          # 输入分辨率，论文统一 416×416
hardware:
  batch_size: 4            # 单卡 4060 推荐值
  accumulate_steps: 4      # 梯度累积，等效 batch=16
  use_amp: true            # 自动混合精度节省显存
optimizer:
	lr: 0.01                 # Adam 初始学习率（阶段 A/B 通用）
  stage_c_lr: 0.0001       # 阶段 C 学习率（全模型微调用较小值）
  min_lr: 0.00001          # CosineAnnealingLR 最低学习率
schedule:
	stage_a_epochs: 5        # 阶段 A 轮数；用于建立纯 YOLO baseline
	stage_b_epochs: 5        # 阶段 B 轮数；当前配置值
	stage_c_epochs: 5        # 阶段 C 轮数；当前配置值
```

当前 `configs/train.yaml` 已按单卡 4060 调整为 `cuda:0 + AMP + grad checkpoint`。

### 训练数据准备

当前训练入口直接复用 YOLOv7 的 create_dataloader，因此配置文件中的 dataset.train_path 需要满足 YOLOv7 数据读取约定。

支持两种常见形式：

1. 传入图片目录。
2. 传入 txt 文件，文件内每行是一张训练图片的绝对路径或相对路径。

标签文件需保持 YOLO 格式，与图片同名，内容为：

```text
class x_center y_center width height
```

所有坐标均为归一化到 [0, 1] 的浮点数。

### YAML 配置示例

训练配置采用分层结构：

```yaml
seed: 42
run_dir: runs/icfie_yolo
hyp_path: yolov7/data/hyp.scratch.p5.yaml
project_after_fusion: true

dataset:
	train_path: data/train.txt
	val_path: null
	num_classes: 12
	class_names: []
	image_size: 416

yolo:
	cfg_path: yolov7/cfg/training/yolov7.yaml
	weights_path: yolov7/yolov7.pt

hardware:
	device: cuda:0
	use_amp: true
	batch_size: 4
	accumulate_steps: 4

optimizer:
	lr: 0.01
	min_lr: 0.00001

schedule:
	stage_a_epochs: 1
	stage_b_epochs: 1
	stage_c_epochs: 1
```

建议先复制或直接编辑 configs/train.yaml，再启动训练。

### 单卡训练

4060 或其他单卡环境可直接运行：

```bash
cd /home/pushk3n/github-ai2
python src/train.py --config configs/train.yaml
```

如果要做正式训练，建议先把 [configs/train.yaml](configs/train.yaml) 中的 `schedule.stage_a_epochs`、`schedule.stage_b_epochs` 和 `schedule.stage_c_epochs` 一并调大，再启动训练：

```bash
cd /home/pushk3n/github-ai2
python src/train.py --config configs/train.yaml
```

### 双卡 DDP 训练

双 3090 环境可以使用 torchrun：

```bash
cd /home/pushk3n/github-ai2
torchrun --nproc_per_node=2 src/train.py --config configs/train.yaml
```

对应地，把 configs/train.yaml 里的 hardware.ddp 改为 true，并将 hardware.device 改为 cuda。

### 常见环境问题

如果运行训练时出现：

1. `CUDA initialization: The NVIDIA driver on your system is too old`
2. `torch.cuda.is_available() == False`

说明当前安装的 PyTorch CUDA 版本高于本机驱动可支持的版本。

解决方式二选一：

1. 安装与当前驱动匹配的 PyTorch 版本，例如上面的 `cu121`。
2. 升级 NVIDIA 驱动后，再安装更高 CUDA 版本的 PyTorch。

另外，训练入口已经兼容了 PyTorch 2.6+ 对 YOLOv7 `.cache` 文件加载行为的变更，不需要手工删除 `train.cache` / `val.cache` 才能启动。

### 常用 YAML 字段

1. hyp_path: YOLOv7 损失超参文件。
2. dataset.train_path: 训练集目录或 txt。
3. dataset.num_classes: 类别数。
4. dataset.class_names: 类别名列表，可留空。
5. hardware.use_amp: 是否启用混合精度。
6. hardware.ddp: 是否启用 DDP。
7. optimizer.lr / optimizer.min_lr: 优化器学习率配置。
8. schedule.stage_a_epochs / schedule.stage_b_epochs / schedule.stage_c_epochs: 三个训练阶段的轮数。
9. use_ota_loss: 是否启用 YOLOv7 的 OTA loss。
10. project_after_fusion: 是否对 FIE 输出做 1x1 投影回原始检测头通道。

说明：

1. dataset.class_names 可选；如果传入，数量必须与 dataset.num_classes 一致。
2. yolo.num_classes 如果显式写出，必须与 dataset.num_classes 一致。
3. project_after_fusion 关闭时，前提是你已经同步修改检测头输入通道。
4. dataset.val_path 字段已预留，后续 eval.py 会复用，但当前 train.py 还没有在训练后自动执行验证。

纯 YOLO / FIE 关闭模式注意事项：

1. `stage_a_loaded.pt` 只是加载 `yolov7.pt` 后的初始化快照，不是训练完成的纯 YOLO 权重。
2. 如果在 [src/test.py](src/test.py) 中设置 `ENABLE_FIE=False`，应把 `YOLO_ONLY_CHECKPOINT_PATH` 指向 `stage_a_epoch_N.pt`。
3. 不要在 `ENABLE_FIE=False` 时加载 `stage_c_epoch_N.pt`，因为阶段 C 的 detect head 已经适配了 FIE 输出特征分布，关闭 FIE 后通常会导致 NMS 置信度整体过低。

### 训练输出

训练过程会在 run-dir 下保存以下文件：

1. stage_a_loaded.pt
2. stage_a_epoch_N.pt
3. stage_b_epoch_N.pt
4. stage_c_epoch_N.pt

默认输出目录：

```bash
runs/icfie_yolo/
```

### 当前进度
1. 已经跑通MSICN模块 (infer.py)
2. 已经通过mock的方式验证了 MSICN yolo FIE 三模块集成的整体流程 (mock.py) 注意mock.py已经废弃
3. 增加单图全流程测试 (test.py) 目前可以做到单图全流程测试，但输出的检测结果仍待调试
4. 增加了 yolo_test.py 用于验证yolo检测头的输入输出，目前可以做到单图检测头测试
5. 已新增 yolo_wrapper.py，统一封装真实 YOLOv7 backbone_neck 和 detect_head 适配逻辑
6. 已新增 train.py 和 train_config.py，支持 ICFIE-YOLO 三阶段训练入口
7. 完成四层解耦重构：推理流水线四层模块分别对应独立文件，icfie_yolo.py 仅负责串联

### 代码架构（四层流水线）

推理链路按职责拆分到四个独立文件：

| 层级 | 文件 | 职责 |
|------|------|------|
| 第一层 | src/msicn.py | MSICN 多尺度光照矫正网络 |
| 第二层 | src/yolo_wrapper.py | YOLOv7 Backbone+Neck 特征提取适配器（含冒烟测试用 MockYOLOv7BackboneNeck）|
| 第三层 | src/fie.py | FIE 特征交互增强模块 |
| 第四层 | src/detect.py | Detect Head 检测头适配器（YOLOv7DetectHeadAdapter + MockYOLOv7DetectHead）|
| 串联层 | src/icfie_yolo.py | ICFIEYOLO 顶层适配器，仅负责连接四层 |



