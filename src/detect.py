from __future__ import annotations

# --------------------------------------------------------
# detect.py — 第四层: 检测头
#
# 职责:
#   封装 YOLOv7 IDetect 层，对外暴露统一的 forward 接口。
#   同时提供本地冒烟测试用的 Mock 检测头。
#
# 架构位置（四层流水线）:
#   1. msicn.py        → MSICN   光照矫正
#   2. yolo_wrapper.py → YOLO Backbone+Neck  特征提取
#   3. fie.py          → FIE     特征增强
#   4. detect.py       → Detect Head  检测输出  ← 本文件
#
# 设计原则:
#   - 只依赖标准 PyTorch 和本地模块，不依赖 icfie_yolo.py
#   - YOLOv7DetectHeadAdapter 是生产路径，直接包装 IDetect 层
#   - MockYOLOv7DetectHead 仅用于本地联调/冒烟测试
# --------------------------------------------------------

from dataclasses import dataclass
from typing import Sequence, Tuple

from torch import Tensor, nn


# ================================================================
# 生产路径: YOLOv7 真实检测头适配器
# ================================================================

class YOLOv7DetectHeadAdapter(nn.Module):
    # 真实 YOLOv7 Detect 头适配器
    # 直接复用 IDetect 层；eval 模式下返回 (解码后预测, 原始三尺度输出) 元组。
    #
    # expected_in_channels:
    #   从 IDetect.m 每个子模块的 in_channels 读取，
    #   供 ICFIEYOLO._validate_detect_head_channels 做通道对齐断言。

    def __init__(self, detect_layer: nn.Module) -> None:
        super().__init__()
        self.detect = detect_layer
        self.expected_in_channels = tuple(int(head.in_channels) for head in detect_layer.m)

    def forward(self, features: Sequence[Tensor]) -> object:
        return self.detect(list(features))


# ================================================================
# 冒烟测试专用 Mock 模块
# ================================================================

@dataclass(frozen=True)
class MockYOLOv7DetectHeadConfig:
    # Mock 检测头配置
    # in_channels:          FIE 输出后每个尺度的通道数
    #   project_after_fusion=False 时为 3C，即 (768, 1536, 3072)
    #   project_after_fusion=True  时为原始通道，即 (256, 512, 1024)
    # num_classes:          目标类别数（ExDark=12，DarkFace=1）
    # bbox_channels:        坐标回归输出通道，固定 4（cx, cy, w, h）
    # objectness_channels:  目标存在置信度通道，固定 1

    in_channels: Tuple[int, ...] = (256, 512, 1024)
    num_classes: int = 12
    bbox_channels: int = 4
    objectness_channels: int = 1

    @property
    def output_channels(self) -> int:
        # 每个锚点的输出向量长度 = 类别数 + bbox 4 维 + 置信度 1 维
        return self.num_classes + self.bbox_channels + self.objectness_channels


class MockYOLOv7DetectHead(nn.Module):
    # Mock 检测头，对每个尺度特征做 1×1 Conv 输出检测向量。
    # 真实 YOLOv7 的检测头远比此复杂，这里仅用于验证通道数对齐。
    #
    # 输出每个尺度的 shape: (B, output_channels, H_i, W_i)
    #   output_channels = num_classes + 4 + 1

    def __init__(self, config: MockYOLOv7DetectHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or MockYOLOv7DetectHeadConfig()
        # 为每个尺度建立独立的 1×1 卷积检测头
        self.heads = nn.ModuleList(
            [nn.Conv2d(in_channels, self.config.output_channels, kernel_size=1)
             for in_channels in self.config.in_channels]
        )

    def forward(self, features: Sequence[Tensor]) -> Tuple[Tensor, ...]:
        if len(features) != len(self.heads):
            raise ValueError(f"DetectHead 期望 {len(self.heads)} 个尺度，实际得到 {len(features)} 个")
        # 逐尺度做 1×1 Conv 变换，输出通道为 num_classes+4+1
        return tuple(head(feature) for head, feature in zip(self.heads, features))
