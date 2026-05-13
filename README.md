### 配置虚拟环境
#### 这里以conda为例

```bash 
conda create -n ai2 python=3.10 -y
conda activate ai2

```

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

### 解压测试集
```bash
unzip test_images.zip -d test_images
```

### 训练介绍

当前仓库已经提供 ICFIE-YOLO 三阶段训练入口: src/train.py。

训练参数现在统一通过 YAML 文件管理，默认配置文件为 configs/train.yaml。

三阶段与 PRD 的对应关系如下：

1. 阶段 A：直接加载 yolov7.pt，视为正常光照预训练已完成，并保存 stage_a_loaded.pt。
2. 阶段 B：只训练 MSICN，显式冻结 YOLO backbone_neck 和 detect_head，且冻结部分前向使用 no_grad。
3. 阶段 C：解冻 MSICN、YOLO backbone_neck、FIE 和 detect_head，进行端到端联合微调。

训练脚本默认行为：

1. 输入尺寸默认 416。
2. 优化器默认 Adam，lr=1e-2，min_lr=1e-5。
3. 默认开启 FIE 后投影，对接原始 YOLOv7 检测头输入通道。
4. 默认输出目录为 runs/icfie_yolo/。

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
	stage_b_epochs: 30
	stage_c_epochs: 30
```

建议先复制或直接编辑 configs/train.yaml，再启动训练。

### 单卡训练

4060 或其他单卡环境可直接运行：

```bash
cd /home/pushk3n/ai2
python src/train.py --config configs/train.yaml
```

如果只想快速冒烟验证训练链路，可以先把 epoch 改成 1：

```bash
cd /home/pushk3n/ai2
python src/train.py --config configs/train.yaml
```

然后把 configs/train.yaml 中的 schedule.stage_b_epochs 和 schedule.stage_c_epochs 改成 1。

### 双卡 DDP 训练

双 3090 环境可以使用 torchrun：

```bash
cd /home/pushk3n/ai2
torchrun --nproc_per_node=2 src/train.py --config configs/train.yaml
```

对应地，把 configs/train.yaml 里的 hardware.ddp 改为 true，并将 hardware.device 改为 cuda。

### 常用 YAML 字段

1. hyp_path: YOLOv7 损失超参文件。
2. dataset.train_path: 训练集目录或 txt。
3. dataset.num_classes: 类别数。
4. dataset.class_names: 类别名列表，可留空。
5. hardware.use_amp: 是否启用混合精度。
6. hardware.ddp: 是否启用 DDP。
7. optimizer.lr / optimizer.min_lr: 优化器学习率配置。
8. schedule.stage_b_epochs / schedule.stage_c_epochs: 两个训练阶段的轮数。
9. use_ota_loss: 是否启用 YOLOv7 的 OTA loss。
10. project_after_fusion: 是否对 FIE 输出做 1x1 投影回原始检测头通道。

说明：

1. dataset.class_names 可选；如果传入，数量必须与 dataset.num_classes 一致。
2. yolo.num_classes 如果显式写出，必须与 dataset.num_classes 一致。
3. project_after_fusion 关闭时，前提是你已经同步修改检测头输入通道。
4. dataset.val_path 字段已预留，后续 eval.py 会复用，但当前 train.py 还没有在训练后自动执行验证。

### 训练输出

训练过程会在 run-dir 下保存以下文件：

1. stage_a_loaded.pt
2. stage_b_epoch_N.pt
3. stage_c_epoch_N.pt

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



