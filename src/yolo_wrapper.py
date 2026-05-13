from __future__ import annotations

# --------------------------------------------------------
# yolo_wrapper.py — 第二层: YOLOv7 Backbone + Neck
#
# 职责:
#   封装真实 YOLOv7 的 backbone+neck 部分，向下游输出
#   三尺度特征 [P3, P4, P5]。
#
# 架构位置（四层流水线）:
#   1. msicn.py        → MSICN   光照矫正
#   2. yolo_wrapper.py → YOLO Backbone+Neck  特征提取  ← 本文件
#   3. fie.py          → FIE     特征增强
#   4. detect.py       → Detect Head  检测输出
#
# 说明:
#   - YOLOv7BackboneNeckAdapter  生产路径，截取 Detect 前所有层
#   - MockYOLOv7BackboneNeck     冒烟测试专用轻量替代品
#   - 检测头适配器已拆分至 detect.py
# --------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn

from detect import YOLOv7DetectHeadAdapter


@dataclass(frozen=True)
class YOLOv7WrapperConfig:
    # 真实 YOLOv7 包装配置
    # cfg_path     - YOLOv7 yaml 配置文件路径
    # weights_path - 预训练权重路径  为 None 时仅加载随机初始化权重
    # num_classes  - 类别数  默认沿用 COCO 80 类

    cfg_path: Path
    weights_path: Path | None
    num_classes: int = 80


class YOLOv7BackboneNeckAdapter(nn.Module):
    # 真实 YOLOv7 Backbone + Neck 适配器
    # 复用原始 layer 图结构  运行到 Detect 前一层并取检测头对应的三尺度输入特征
    #
    # use_grad_checkpoint:
    #   False（默认）: 正常前向
    #   True          : 将整个 backbone+neck 包进 torch.utils.checkpoint.checkpoint
    #                   以时间换空间  显著降低显存  适合 4060 等 VRAM 受限显卡

    def __init__(self, yolo_model: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(yolo_model.model[:-1]))
        self.saved_layer_indices = set(yolo_model.save)
        detect_layer = yolo_model.model[-1]
        self.feature_indices = tuple(int(index) for index in detect_layer.f)
        # 由 build_model 根据 HardwareConfig.use_grad_checkpoint 显式设置
        self.use_grad_checkpoint: bool = False

    def _forward_impl(self, image: Tensor) -> tuple[Tensor, ...]:
        # 实际前向逻辑  返回 tuple 方便 torch.utils.checkpoint 透传
        saved_outputs: list[Tensor | Sequence[Tensor] | None] = []
        current: Tensor | list[Tensor] = image
        for layer in self.layers:
            if layer.f != -1:
                if isinstance(layer.f, int):
                    current = saved_outputs[layer.f]
                else:
                    current = [current if index == -1 else saved_outputs[index] for index in layer.f]

            current = layer(current)
            should_save = layer.i in self.saved_layer_indices or layer.i in self.feature_indices
            saved_outputs.append(current if should_save else None)

        features = [saved_outputs[index] for index in self.feature_indices]
        if any(feature is None for feature in features):
            raise RuntimeError(f"未能从 YOLOv7 中提取完整的检测特征层: {self.feature_indices}")
        return tuple(feature for feature in features if isinstance(feature, Tensor))

    def forward(self, image: Tensor) -> list[Tensor]:
        if self.use_grad_checkpoint and image.requires_grad:
            # 梯度检查点模式: 用时间换显存
            # use_reentrant=False 为推荐写法  兼容更多情景
            from torch.utils.checkpoint import checkpoint as cp
            return list(cp(self._forward_impl, image, use_reentrant=False))
        return list(self._forward_impl(image))


# ================================================================
# 冒烟测试专用 Mock Backbone+Neck
# ================================================================

@dataclass(frozen=True)
class MockYOLOv7BackboneNeckConfig:
    # 轻量 Mock Backbone/Neck 配置，仅用于本地联调验证。
    # 使用 5 个步长为 2 的卷积模拟 YOLOv7 的下采样过程。
    # 输出三个尺度特征: P3(1/8), P4(1/16), P5(1/32)。
    # 通道数与真实 YOLOv7 保持一致以确保 FIE 接口对齐。

    input_channels: int = 3
    stem_channels: int = 32            # 第一个下采样卷积的输出通道
    feature_channels: Tuple[int, ...] = (256, 512, 1024)    # P3/P4/P5 的输出通道


class ConvBNAct(nn.Module):
    # 基础卷积块: Conv + BN + SiLU，仅用于 Mock 模块。
    # SiLU (Sigmoid Linear Unit) 是 YOLOv7 中广泛使用的激活函数。

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),    # SiLU: x * sigmoid(x)，比 ReLU 更平滑的激活
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MockYOLOv7BackboneNeck(nn.Module):
    # Mock YOLOv7 风格主干 + Neck，用于 smoke test。
    #
    # 5 阶段串行下采样:
    #   stage0: stride=2  (B,3,H,W)    -> (B, 32, H/2 , W/2 )
    #   stage1: stride=2  (B,32,H/2,.) -> (B, 64, H/4 , W/4 )
    #   stage2: stride=2  -> P3: (B, 256, H/8 , W/8 )
    #   stage3: stride=2  -> P4: (B, 512, H/16, W/16)
    #   stage4: stride=2  -> P5: (B,1024, H/32, W/32)
    #
    # 输出为 dict，键名与 ICFIEYOLOConfig.feature_names 对应。

    def __init__(self, config: MockYOLOv7BackboneNeckConfig | None = None) -> None:
        super().__init__()
        self.config = config or MockYOLOv7BackboneNeckConfig()
        c1, c2, c3 = self.config.feature_channels    # 解包 P3/P4/P5 通道数
        stem = self.config.stem_channels
        # 5 个步长为 2 的卷积，每个使空间尺寸减半
        self.stage0 = ConvBNAct(self.config.input_channels, stem, stride=2)
        self.stage1 = ConvBNAct(stem, stem * 2, stride=2)
        self.stage2 = ConvBNAct(stem * 2, c1, stride=2)    # P3 输出
        self.stage3 = ConvBNAct(c1, c2, stride=2)          # P4 输出
        self.stage4 = ConvBNAct(c2, c3, stride=2)          # P5 输出

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        x = self.stage0(image)    # (B, 32,  H/2,  W/2 )
        x = self.stage1(x)        # (B, 64,  H/4,  W/4 )
        p3 = self.stage2(x)       # (B, 256, H/8,  W/8 )
        p4 = self.stage3(p3)      # (B, 512, H/16, W/16)
        p5 = self.stage4(p4)      # (B,1024, H/32, W/32)
        # 以 dict 形式返回，键与 ICFIEYOLOConfig.feature_names 保持一致
        return {"p3": p3, "p4": p4, "p5": p5}


def build_yolov7_components(
    config: YOLOv7WrapperConfig,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, YOLOv7BackboneNeckAdapter, YOLOv7DetectHeadAdapter, int, list[str]]:
    # 加载真实 YOLOv7 并拆成 backbone_neck / detect_head 两个显式模块
    # 返回值:
    #   yolo_model     - 原始 YOLOv7 Model 实例
    #   backbone_neck  - 三尺度特征提取适配器
    #   detect_head    - 检测头适配器
    #   stride         - 最大步长  用于 letterbox 对齐
    #   class_names    - 类别名列表

    from models.yolo import Model
    from utils.torch_utils import intersect_dicts

    yolo_model = Model(str(config.cfg_path), ch=3, nc=config.num_classes)
    class_names = [str(name) for name in yolo_model.names]

    if config.weights_path is not None:
        checkpoint = torch.load(config.weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
            state_dict = checkpoint_model.float().state_dict()
            class_names = [str(name) for name in getattr(checkpoint_model, "names", class_names)]
        else:
            state_dict = checkpoint

        # 允许用 COCO 预训练权重初始化自定义类别模型。
        # 检测头输出维度不一致时，只加载键名和形状都匹配的参数，
        # 其余层（主要是 detect head）保留当前模型随机初始化结果。
        compatible_state_dict = intersect_dicts(state_dict, yolo_model.state_dict())
        yolo_model.load_state_dict(compatible_state_dict, strict=False)

    yolo_model = yolo_model.to(device).eval()
    detect_layer = yolo_model.model[-1]
    backbone_neck = YOLOv7BackboneNeckAdapter(yolo_model)
    detect_head = YOLOv7DetectHeadAdapter(detect_layer)
    stride = int(yolo_model.stride.max().item())
    return yolo_model, backbone_neck, detect_head, stride, class_names