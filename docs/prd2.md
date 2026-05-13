# PRD: ICFIE-YOLO 架构重构与训练流程

> **依据**: 本文档以论文《基于ICFIE-YOLO的低照度图像目标检测方法》（秦嘉奇等，电子学报 2025）
> 为唯一权威来源。所有架构细节、公式、超参数均须与论文第 3 节保持一致。
> 工程目标：在 RTX 4060 (8 GB VRAM, 单卡) 和 双 RTX 3090 (24 GB × 2, DDP) 上均能正常训练。

## 1. 核心目标
复现 ICFIE-YOLO 并在 ExDark / DarkFace 数据集上完成三阶段训练，最终输出可部署权重文件。
具体要求：
- 前向推理流程显式解耦为四大模块，不允许跨模块偷传状态。
- 全局配置通过 `dataclass` 透明注入，禁止 `**kwargs` 隐藏超参。
- 三阶段训练中，冻结/解冻操作必须显式遍历 `param.requires_grad`，不得仅靠 optimizer 参数列表。

## 2. 编码与架构规范
- **语言**：代码注释和文档描述均使用中文。
- **框架**：PyTorch（论文原代码使用 TensorFlow，本项目统一用 PyTorch 复现）。
- **架构设计**：显式依赖注入（Dependency Injection）。严禁将配置隐藏在函数的 `**kwargs` 或默认参数中，必须将配置对象显式传入构造函数。
- **输入尺寸**：论文统一使用 **416×416**（EfficientDET 用 512×512，其余全部 416×416）。

## 3. 模块解耦 (四大前向模块)

前向数据流（忠于论文图3）：
```
低照度图像 (B,3,416,416)
  └─> MSICN        → 矫正图像 (B,3,416,416)  值域 [0,1]
        └─> YOLO_BackboneNeck  → P3(B,256,52,52) / P4(B,512,26,26) / P5(B,1024,13,13)
              └─> FIE          → P3'(B,768,...) / P4'(B,1536,...) / P5'(B,3072,...)
                    └─> Detect Head → 检测输出
```
> P3/P4/P5 通道数为 YOLOv7 标准值，FIE 输出通道 = 3C（论文原始设计，不做投影）。
> 若需对接不修改检测头的场景，可开启 `project_after_fusion=True` 投影回 C 通道（工程扩展）。

### 3.1 MSICN (多尺度光照矫正网络)

**论文第 3.2 节**

- **输入**: 低照度图像张量 `(B, 3, H, W)`，像素范围 `[0, 1]`。
- **子模块及数据流（并行，非串行）**:
  ```
  image → IFE → illumination_features (B, 32, H, W)
                   ├─ GIC → global_coefficients (B, 3, 1, 1)   ∈ (0,1)
                   └─ LIC → local_coefficients  (B, 30, H, W)  ∈ (-1,1)
  image + global_coefficients + local_coefficients → NLIS → corrected_image
  ```
  GIC 和 LIC **共享同一份 IFE 输出**，二者并行计算，均以 illumination_features 为输入。
- **输出**: clamp 至 `[0, 1]` 的矫正图像张量 `(B, 3, H, W)`。

#### 3.1.1 IFE (光照特征提取)
- 6 层 3×3 卷积，stride=1，ReLU，各层输出通道依次为 (4, 4, 4, 4, 8, 8)。
- **每层输出均保留**，最终在通道维度 concat，输出 32 = 4+4+4+4+8+8 通道特征图。
  这种逐层 concat 保留了从小感受野到大感受野的多尺度光照特征，是"多尺度"的核心设计。
- IFE 各层输出通道之和必须严格等于 32（构造函数中硬断言），以对齐 GIC/LIC 的输入通道。

#### 3.1.2 GIC (全局光照矫正)
- 输入：illumination_features `(B, 32, H, W)`。
- 结构：2 层 5×5 空洞卷积（dilation=3，感受野 = 5+(5-1)×(3-1) = 13）→ AdaptiveAvgPool2d(1) → 1×1 Conv → Sigmoid。
- 输出：全局矫正系数 k，shape `(B, 3, 1, 1)`，值域 `(0, 1)`，R/G/B 三通道独立。
- 应用公式（论文公式 1）：
  $$I_{\text{g\_out}} = I_{\text{in}} \cdot (-k \cdot I_{\text{in}} + (1+k)), \quad k \in (0,1)$$

#### 3.1.3 LIC (局部光照矫正)
- 输入：illumination_features `(B, 32, H, W)`。
- 结构（双分支并行）：
  - **非下采样精细分支**（论文"非下采样光照矫正"）：连续 3×3 卷积，感受野较小，拟合小范围局部光源。输出 15 通道。
  - **上下文分支**（论文"下采样光照矫正"）：3×3 卷积 + 空洞 3×3 卷积（dilation=2）+ AvgPool（kernel=3, stride=1）→ 3×3 卷积降维。感受野较大，拟合大范围局部光源。输出 15 通道。
  - 两分支 concat → 原始系数 30 通道。
- 激活：`2·Sigmoid(x) − 1`，输出值域 `(-1, 1)`，允许局部压暗（负值）或增亮（正值）。
- 输出：局部矫正系数 `(B, 30, H, W)`，含 10 组 × 3 通道。

#### 3.1.4 NLIS (非线性光照堆叠)
- **零可训练参数**，纯数学迭代。
- 先做全局矫正（公式 1），再做 10 次局部迭代矫正（论文公式 2）：
  $$I_{\text{out}_{i}} = I_{\text{out}_{i-1}} \cdot (k_i \cdot I_{\text{out}_{i-1}} + (1-k_i)), \quad i=1..10, \quad k_i \in (-1,1)$$
- 输出前做 `clamp(0, 1)` 保证数值稳定。

### 3.2 YOLO_BackboneNeck (主干与颈部网络)

- **输入**: MSICN 输出的矫正图像张量 `(B, 3, H, W)`。
- **输出**: 两种格式均需支持（由 `ICFIEYOLOConfig.feature_names` 控制键名映射）：
  - `dict{"p3": P3_tensor, "p4": P4_tensor, "p5": P5_tensor}`
  - `list/tuple [P3_tensor, P4_tensor, P5_tensor]`
- **约束**: 严格作为特征提取器。与 MSICN、FIE 完全解耦，便于将来替换为真实 YOLOv7 权重。
- **真实接入方式**：加载 `yolov7/yolov7.pt`，截取主干+颈部网络输出 P3/P4/P5，包装成满足上述接口的 `nn.Module`。

### 3.3 FIE (特征交互增强模块)

**论文第 3.3 节**

- **输入**: `[P3, P4, P5]`，各尺度 shape `(B, C, H, W)`，C ∈ {256, 512, 1024}。
- **结构（每个尺度独立，论文图 5）**:

  **空间交互增强分支**（像素间关联）：
  - 3 个 1×1 Conv 生成 b、c（通道数 = C/2）和 d（通道数 = C）。
  - b 展平 → q `(B, HW, C/2)`，c 展平 → `(B, C/2, HW)`。
  - `u = Softmax(q ⊗ cᵀ)` `(B, HW, HW)`，空间关联矩阵。
  - `f_s = Re(u ⊗ Flatten(d))` `(B, C, H, W)`，式（3）。

  **通道交互增强分支**（通道间语义关联）：
  - 3 个 3×3 Conv（stride=2，空间压缩 H/2 × W/2）生成 x、y、z，通道数 = C。
  - `n = Softmax(yᵀ ⊗ p)` `(B, C, C)`，通道关联矩阵（p 为 x 展平）。
  - `f_c = Re(Flatten(z) ⊗ n)` 上采样回 `(B, C, H, W)`，式（4）。

  **融合**：`cat([f, f_s, f_c], dim=1)` → 输出通道 = 3C（论文原始设计）。

- **输出通道**（`project_after_fusion=False`，论文忠实）：
  - P3_out: `(B, 768, 52, 52)`, P4_out: `(B, 1536, 26, 26)`, P5_out: `(B, 3072, 13, 13)`。
- **工程扩展**（`project_after_fusion=True`）：额外 1×1 Conv 将 3C 投影回 C，供对接不修改检测头的场景，属于消融/工程选项，**非论文原始设计**。

### 3.4 Detect Head (检测头)
- **输入**: FIE 输出的增强特征图列表。
- 当 `project_after_fusion=False` 时，输入通道为 (768, 1536, 3072)；当 `project_after_fusion=True` 时为原始 (256, 512, 1024)。
- **输出**: 预测结果张量（类别、边界框、置信度），接口与 YOLOv7 检测头一致。

## 4. 显式依赖注入与参数管理

配置数据必须使用 `dataclass` 定义，并显式注入到类的 `__init__` 中。

```python
@dataclass(frozen=True)
class MSICNConfig:
    input_channels: int = 3
    ife_channels: Tuple[int, ...] = (4, 4, 4, 4, 8, 8)  # 各层必须恰好 6 个，总和=32
    gic_hidden_channels: int = 32
    gic_dilation: int = 3
    lic_hidden_channels: int = 32
    lic_context_dilation: int = 2
    lic_context_pool_kernel: int = 3
    local_groups: int = 10   # NLIS 迭代次数，论文固定为 10
    clamp_output: bool = True

class MSICN(nn.Module):
    def __init__(self, config: MSICNConfig):
        super().__init__()
        self.config = config
        self.ife = IFE(config)
        self.gic = GIC(config)   # GIC 和 LIC 并行，均以 IFE 输出为输入
        self.lic = LIC(config)
        self.nlis = NLIS(config)
```

## 5. 可训练参数粒度与对齐要求

### 5.1 MSICN 可训练参数

| 子模块 | 可训练层 | 规格 |
|--------|---------|------|
| IFE | 6 层 `nn.Conv2d` | 3×3 卷积核，含 bias |
| GIC | 2 层 `nn.Conv2d` + 1 层 `nn.Conv2d` | 5×5 空洞(dilation=3) + 1×1 proj，含 bias |
| LIC | 精细分支: 2 层 `nn.Conv2d`；上下文分支: 3 层 `nn.Conv2d` | 3×3 卷积核，含 bias |
| NLIS | **无可训练参数** | 纯数值迭代（论文明确规定） |

### 5.2 FIE 可训练参数

| 分支 | 可训练层 | 规格 |
|------|---------|------|
| 空间分支 | Q(b)、K(c)、V(d) 共 3 个 `nn.Conv2d` | 1×1 卷积核；Q/K 输出通道 = C/2，V = C |
| 通道分支 | Q(x)、K(y)、V(z) 共 3 个 `nn.Conv2d` | 3×3 卷积核，stride=2 |
| 融合投影层 | `project_after_fusion=True` 时：1 个 `nn.Conv2d` per scale | 1×1 卷积核，将 3C→C |

## 6. 三阶段训练流程

> **数据集规格（논문第 4.1 节）**
> - ExDark：7363 张，12 类，训练/测试 = 5890/1473（8:2）
> - DarkFace：6000 张，训练/测试 = 5400/600
> - 正常光照预训练：MS COCO（YOLOv7 官方预训练权重 `yolov7.pt` 直接加载，无须从头训练 COCO）

> **优化器（论文第 4.1 节）**：Adam，初始 lr = 1e-2，最低 lr = 1e-5；所有阶段统一。

训练脚本入口：`src/train.py`（**待实现**，为当前最高优先级）。

### 阶段 A：正常光照预训练（加载 YOLOv7 预训练权重，跳过重训）

目标：获得在 MS COCO 上具有基线检测性能的 YOLO 检测器。

**实际操作**：直接加载官方 `yolov7.pt`，无须从头训练 COCO，视为阶段 A 已完成。

参数状态：

| 模块 | requires_grad |
|------|--------------|
| YOLO_BackboneNeck | True（预训练权重初始化后用于阶段 B 冻结） |
| Detect Head | True |
| MSICN | False |
| FIE | False |

### 阶段 B：基于检测损失的光照对齐训练

目标：冻结预训练好的 YOLO 检测器（BackboneNeck + DetectHead），仅用检测损失反向传播优化 MSICN，使其学习从低照度到"利于目标检测"的光照映射。

**关键约束**：
- 必须显式遍历参数并设置 `requires_grad`，禁止仅靠 optimizer 参数列表模拟冻结。
- 对冻结的 YOLO 前向必须用 `with torch.no_grad():` 包裹以节省显存。
- 数据集：低照度训练集（ExDark train 或 DarkFace train），**无需成对正常光照图像**。
- 损失函数：仅使用 YOLOv7 的检测损失（分类 + 定位 + 置信度），不加任何图像重建损失。

参数状态：

| 模块 | requires_grad |
|------|--------------|
| MSICN | **True** |
| YOLO_BackboneNeck | False |
| FIE | False |
| Detect Head | False |

### 阶段 C：端到端联合微调

目标：在低照度数据集上对全模型进行微调，论文明确"解冻目标检测器部分参数，对全模型参数进行微调训练"。

参数状态：

| 模块 | requires_grad |
|------|--------------|
| MSICN | True |
| YOLO_BackboneNeck | True |
| FIE | **True**（阶段 C 首次激活） |
| Detect Head | True |

## 7. 硬件适配规范

### 7.1 RTX 4060（8 GB VRAM，单卡）

- **AMP（自动混合精度）**：全程启用 `torch.autocast("cuda", dtype=torch.float16)` + `torch.cuda.amp.GradScaler`。
- **批量大小**：阶段 A/B/C 推荐 `batch_size=4`（416×416 输入），若显存溢出降至 2。
- **梯度累积**：设 `accumulate_steps=4`，等效 batch=16，弥补小 batch 的梯度估计噪声。
- **梯度检查点**：若阶段 C 全模型微调时显存仍溢出，对 YOLOv7 BackboneNeck 启用 `torch.utils.checkpoint`。
- **数据加载**：`num_workers=4`，`pin_memory=True`。
- **配置字段**：`HardwareConfig(device="cuda:0", use_amp=True, batch_size=4, accumulate_steps=4)`。

### 7.2 双 RTX 3090（24 GB × 2，DDP）

- **启动方式**：`torchrun --nproc_per_node=2 src/train.py`。
- **分布式**：`torch.nn.parallel.DistributedDataParallel`（DDP），`DistributedSampler` 保证数据不重叠。
- **批量大小**：每卡 `batch_size=16`，等效全局 batch=32；`accumulate_steps=1`。
- **AMP**：建议开启（可选），3090 VRAM 充足时可关闭 AMP 使用 FP32 保证精度。
- **同步 BN**：`nn.SyncBatchNorm.convert_sync_batchnorm(model)` 在 DDP 初始化前调用。
- **冻结阶段注意**：DDP 会对所有参数（含冻结参数）构建 bucket，需在 DDP 包装前完成 `requires_grad` 设置，或使用 `find_unused_parameters=True`（性能损失可接受）。
- **配置字段**：`HardwareConfig(device="cuda", use_amp=False, batch_size=16, accumulate_steps=1, ddp=True)`。

### 7.3 HardwareConfig dataclass

```python
@dataclass
class HardwareConfig:
    device: str = "cuda"          # "cuda:0" 单卡 / "cuda" DDP
    use_amp: bool = True          # 混合精度
    batch_size: int = 4           # 单卡 batch
    accumulate_steps: int = 4     # 梯度累积步数
    num_workers: int = 4
    pin_memory: bool = True
    ddp: bool = False             # 是否启用 DDP
    use_grad_checkpoint: bool = False  # BackboneNeck 梯度检查点
```

## 8. 强制性架构约束 (必选项)

**通道对齐断言**：位于 `ICFIEYOLO.__init__` 中。在 FIE 的 `output_channels()` 与 detect_head 期望通道数之间进行显式 `assert`，禁止依赖框架隐式报错。

```python
fie_out_ch = self.fie.output_channels()
assert list(fie_out_ch) == list(detect_input_channels), (
    f"FIE 输出通道 {fie_out_ch} 与检测头期望通道 {detect_input_channels} 不匹配"
)
```

**绝对的梯度截断**：在阶段 B 冻结参数时，必须显式遍历：
```python
for param in model.backbone_neck.parameters():
    param.requires_grad = False
for param in model.detect_head.parameters():
    param.requires_grad = False
```
严禁仅通过在 Optimizer 中剔除参数来实现。

**精度转换显式化**：MSICN 输出严格在 `[0, 1]` 浮点区间。在 `ICFIEYOLO.forward` Step 1→Step 2 之间，显式对齐精度：
```python
corrected_image = corrected_image.to(dtype=next(self.backbone_neck.parameters()).dtype)
```

**全局随机种子**：提供 `set_seed(seed: int)` 函数，统一固定 `torch`、`numpy`、`random` 的随机种子，确保完全可复现。

**IFE 通道约束**：构造函数中硬断言 `len(ife_channels) == 6` 且 `sum(ife_channels) == 32`。

## 9. 待实现项

| 优先级 | 文件 | 说明 |
|--------|------|------|
| P0 | `src/train.py` | 三阶段训练入口，含 HardwareConfig 分支（单卡 AMP / DDP） |
| P0 | `src/yolo_wrapper.py` | 将真实 YOLOv7（`yolov7.pt`）包装为符合 backbone_neck/detect_head 接口的 `nn.Module` |
| P1 | `src/datasets.py` | ExDark 和 DarkFace 的 `torch.utils.data.Dataset` 实现，含 letterbox 预处理 |
| P1 | `src/train_config.py` | `TrainConfig` dataclass，统一管理 epochs、lr、weight_decay、数据路径等 |
| P2 | `src/eval.py` | 在测试集上计算 mAP / Recall，与论文表 1 对比 |