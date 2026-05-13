from __future__ import annotations

# --------------------------------------------------------
# FIE - Feature Interacted Enhancement
# 特征交互增强模块
#
# 论文出处: 基于ICFIE-YOLO的低照度图像目标检测方法
#
# 核心思路:
#   低照度图像经过主干网络提取的特征图中存在大量背景噪声和弱目标特征
#   FIE 通过建立像素级空间关联矩阵和通道级语义关联矩阵
#   对有效目标特征进行增强  同时抑制噪声干扰
#
# 整体结构 (对每个尺度特征独立处理):
#   输入特征 f  shape: (B, C, H, W)
#       --> 空间交互增强分支 (SpatialInteractionEnhancement)
#               建立像素对像素的关系矩阵  shape: (B, HW, HW)
#               输出空间增强特征 f_s  shape: (B, C, H, W)
#       --> 通道交互增强分支 (ChannelInteractionEnhancement)
#               压缩空间后建立通道对通道的关系矩阵  shape: (B, C, C)
#               输出通道增强特征 f_c  shape: (B, C, H, W)
#       --> 拼接融合: cat([f, f_s, f_c], dim=1)
#               输出通道数 = 3C  (原始 + 空间增强 + 通道增强)
#
# 多尺度:
#   MultiScaleFIE 对 P3/P4/P5 分别独立应用 FIEBlock
#   特征通道数分别为 (256, 512, 1024)  输出变为 (768, 1536, 3072)
#   若设置 project_after_fusion=True 则再 1x1 Conv 投影回原通道数
#
# 可训练参数:
#   空间交互分支 (Spatial):     Q, K, V 的 1x1 投影卷积
#   通道交互分支 (Channel):     Q, K, V 的 3x3 步长为 2 的空间压缩卷积
#   投影层 (Projection):       如果开启了融合后投影，对应的 1x1 降维卷积
#
# --------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SpatialInteractionConfig:
    # 空间交互分支的超参数配置
    #
    # reduction_ratio      - Q/K 中间通道压缩比  inter = max(min_inter, C // ratio)
    #                        压缩减少 (HW x HW) 关系矩阵的计算量
    # min_inter_channels   - 中间通道下限  防止对小通道数特征过度压缩
    # use_bias             - Q/K/V 的 1x1 投影卷积是否带偏置
    # attention_temperature- 注意力 softmax 的温度系数  1.0 表示不缩放

    reduction_ratio: int = 2
    min_inter_channels: int = 16
    use_bias: bool = True
    attention_temperature: float = 1.0


@dataclass(frozen=True)
class ChannelInteractionConfig:
    # 通道交互分支的超参数配置
    #
    # compression_kernel_size/stride/padding - 空间压缩卷积的参数
    #   stride=2 将 HxW 压缩为 (H/2)x(W/2)  减少后续通道注意力矩阵的计算量
    # use_bias             - Q/K/V 卷积是否带偏置
    # attention_temperature- 通道注意力温度系数
    # upsample_mode        - 压缩后恢复原空间尺寸的插值模式  默认 bilinear
    # align_corners        - bilinear/bicubic 上采样时是否对齐角点坐标
    #   注意: 只有 bilinear/bicubic/trilinear 模式才接受该参数  nearest 模式不支持

    compression_kernel_size: int = 3
    compression_stride: int = 2
    compression_padding: int = 1
    use_bias: bool = True
    attention_temperature: float = 1.0
    upsample_mode: str = "bilinear"
    align_corners: bool = False


@dataclass(frozen=True)
class FIEBlockConfig:
    # 单尺度 FIE 块配置
    #
    # enable_spatial_branch - False 时跳过空间分支 (消融实验: 验证空间分支贡献)
    # enable_channel_branch - False 时跳过通道分支 (消融实验: 验证通道分支贡献)
    # keep_original_feature - True 时 cat 包含原始特征 f  输出 3C 通道 (论文默认)
    #                         False 时只 cat 增强特征   输出 2C 或 1C
    # spatial / channel     - 各分支详细配置

    enable_spatial_branch: bool = True
    enable_channel_branch: bool = True
    keep_original_feature: bool = True
    spatial: SpatialInteractionConfig = field(default_factory=SpatialInteractionConfig)
    channel: ChannelInteractionConfig = field(default_factory=ChannelInteractionConfig)


@dataclass(frozen=True)
class MultiScaleFIEConfig:
    # 多尺度 FIE 总配置
    #
    # feature_channels     - P3/P4/P5 的输入通道数  YOLOv7 标准: (256, 512, 1024)
    # per_scale            - 每个尺度对应的 FIEBlockConfig  长度必须与 feature_channels 一致
    # project_after_fusion - True  : cat 后接 1x1 Conv 投影回指定通道  适合不修改检测头的场景
    #                        False : 直接输出 3C 通道 (论文忠实实现)  需修改检测头输入通道
    # projection_channels  - project_after_fusion=True 时的目标通道数  None 则默认回原通道
    # projection_use_bias  - 投影卷积是否带偏置

    feature_channels: Tuple[int, ...] = (256, 512, 1024)
    per_scale: Tuple[FIEBlockConfig, ...] = field(
        default_factory=lambda: (FIEBlockConfig(), FIEBlockConfig(), FIEBlockConfig())
    )
    project_after_fusion: bool = False
    projection_channels: Tuple[int, ...] | None = None
    projection_use_bias: bool = True


class SpatialInteractionEnhancement(nn.Module):
    # 空间交互增强分支
    #
    # 作用: 建立特征图上各像素位置之间的长程空间依赖关系
    #       让每个像素的特征都能感知到来自其他位置的上下文信息
    #       对低照度图像: 目标像素可以从周围较亮区域借取特征  增强可见性
    #
    # 实现方式 (Non-local self-attention):
    #   Q = 1x1 Conv(f)  shape: (B,C,H,W) -> (B, inter_C, H, W) -> (B, HW, inter_C)
    #   K = 1x1 Conv(f)  同 Q  保持 (B, inter_C, HW)
    #   V = 1x1 Conv(f)  shape: (B, C, H, W) -> (B, HW, C)
    #   Attention = softmax(Q @ K^T / temperature)  shape: (B, HW, HW)
    #   Output = (Attention @ V) reshape -> (B, C, H, W)
    #
    # 计算量提醒:
    #   Q @ K^T 矩阵: O(B * (HW)^2 * inter_C)
    #   P3 尺度 H=W=52 时 HW=2704  矩阵 2704x2704  显存开销大
    #   reduction_ratio 默认 2  将 inter_C 压为 C//2  减少约 50% 计算
    #
    # 输入 shape:  (B, C, H, W)
    # 输出 shape:  (B, C, H, W)

    def __init__(self, in_channels: int, config: SpatialInteractionConfig) -> None:
        """
        # 可训练参数:
        # Q, K, V 的 1x1 投影卷积:
        # query_proj, key_proj, value_proj
        """
        super().__init__()
        inter_channels = max(config.min_inter_channels, in_channels // config.reduction_ratio)
        self.temperature = config.attention_temperature
        # Q: 查询投影  通道压缩到 inter_channels
        self.query_proj = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=config.use_bias)
        # K: 键投影  与 Q 相同维度  用于被查询匹配
        self.key_proj = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=config.use_bias)
        # V: 值投影  保持原通道  用于加权聚合最终输出
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=config.use_bias)

    def forward(self, feature: Tensor, return_details: bool = False) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        batch_size, channels, height, width = feature.shape

        # (B, inter_C, H, W) -> flatten(2) -> (B, inter_C, HW) -> transpose -> (B, HW, inter_C)
        query = self.query_proj(feature).flatten(2).transpose(1, 2)    # (B, HW, inter_C)

        # (B, inter_C, H, W) -> flatten(2) -> (B, inter_C, HW)  不转置  便于 bmm
        key = self.key_proj(feature).flatten(2)    # (B, inter_C, HW)

        # bmm: (B, HW, inter_C) @ (B, inter_C, HW) = (B, HW, HW)
        # 元素 attention[b,i,j] = 位置 i 对位置 j 的注意力权重 (归一化后)
        attention_logits = torch.bmm(query, key) / self.temperature    # (B, HW, HW)
        attention = torch.softmax(attention_logits, dim=-1)    # (B, HW, HW)

        # (B, C, H, W) -> flatten(2) -> (B, C, HW) -> transpose -> (B, HW, C)
        value = self.value_proj(feature).flatten(2).transpose(1, 2)    # (B, HW, C)

        # bmm: (B, HW, HW) @ (B, HW, C) = (B, HW, C)
        # 每个位置的输出 = 对全图所有位置的 value 做加权聚合
        enhanced = torch.bmm(attention, value)    # (B, HW, C)
        # 转置 + reshape 恢复空间结构
        enhanced = enhanced.transpose(1, 2).reshape(batch_size, channels, height, width)    # (B, C, H, W)

        if not return_details:
            return enhanced
        return enhanced, {"spatial_attention": attention}    # 返回注意力图供可视化


class ChannelInteractionEnhancement(nn.Module):
    # 通道交互增强分支
    #
    # 作用: 建立特征通道之间的语义关联
    #       让每个通道感知其他语义相关通道的信息  同时抑制无关 (噪声) 通道
    #
    # 实现方式 (Channel self-attention with spatial compression):
    #   stride-2 Conv 将 HxW 压为 (H/2)x(W/2)  减少矩阵乘法量
    #   Q = stride-2 Conv(f) -> flatten -> (B, C, HW/4)
    #   K = stride-2 Conv(f) -> flatten -> (B, C, HW/4)
    #   V = stride-2 Conv(f) -> flatten -> (B, C, HW/4)
    #   Channel_Attn = softmax(K @ Q^T / temp)  shape: (B, C, C)
    #     注意乘法顺序: K @ Q^T  对应论文通道注意力的标准写法
    #   Enhanced_half = Channel_Attn @ V_flat  -> reshape -> (B, C, H/2, W/2)
    #   Enhanced = upsample(Enhanced_half, size=(H,W))  -> (B, C, H, W)
    #
    # 输入 shape:  (B, C, H, W)
    # 输出 shape:  (B, C, H, W)

    def __init__(self, in_channels: int, config: ChannelInteractionConfig) -> None:
        """
        # 可训练参数:  
        # Q, K, V 的 3x3 步长为 2 的空间压缩卷积：
        # query_proj, key_proj, value_proj
        """
        super().__init__()
        self.temperature = config.attention_temperature
        self.upsample_mode = config.upsample_mode
        # align_corners 只传给支持它的插值模式  nearest 等模式传 None
        self.align_corners = config.align_corners if config.upsample_mode in {"bilinear", "bicubic", "trilinear"} else None

        conv_kwargs = {
            "kernel_size": config.compression_kernel_size,
            "stride": config.compression_stride,       # stride=2  核心: 压缩空间
            "padding": config.compression_padding,
            "bias": config.use_bias,
        }
        # Q/K/V 三路并联压缩卷积  参数独立  允许各自学习不同投影方向
        self.query_proj = nn.Conv2d(in_channels, in_channels, **conv_kwargs)
        self.key_proj = nn.Conv2d(in_channels, in_channels, **conv_kwargs)
        self.value_proj = nn.Conv2d(in_channels, in_channels, **conv_kwargs)

    def forward(self, feature: Tensor, return_details: bool = False) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        original_size = feature.shape[-2:]    # 记录原始 (H, W)  用于最后上采样

        # stride-2 卷积压缩:  (B, C, H, W) -> (B, C, H/2, W/2)
        query = self.query_proj(feature)    # (B, C, H/2, W/2)
        key = self.key_proj(feature)        # (B, C, H/2, W/2)
        value = self.value_proj(feature)    # (B, C, H/2, W/2)

        # 展平空间维  hw = H/2 * W/2
        query_flat = query.flatten(2)    # (B, C, hw)
        key_flat = key.flatten(2)        # (B, C, hw)

        # bmm: (B, C, hw) @ (B, hw, C) = (B, C, C)
        # 元素 channel_attention[b,i,j] = 通道 i 与通道 j 的相关性
        channel_attention_logits = torch.bmm(key_flat, query_flat.transpose(1, 2)) / self.temperature    # (B, C, C)
        channel_attention = torch.softmax(channel_attention_logits, dim=-1)    # (B, C, C)

        value_flat = value.flatten(2)    # (B, C, hw)
        # bmm: (B, C, C) @ (B, C, hw) = (B, C, hw)
        # 每个通道的输出 = 对所有通道的 value 做加权融合
        enhanced_half = torch.bmm(channel_attention, value_flat).reshape_as(value)    # (B, C, H/2, W/2)

        # 上采样恢复至原始空间尺寸  使输出能与原始特征 cat
        enhanced = F.interpolate(
            enhanced_half,
            size=original_size,
            mode=self.upsample_mode,
            align_corners=self.align_corners,
        )    # (B, C, H, W)

        if not return_details:
            return enhanced
        details = {
            "channel_attention": channel_attention,     # (B, C, C) 通道注意力矩阵  可视化用
            "compressed_feature": enhanced_half,        # (B, C, H/2, W/2) 压缩态增强特征
        }
        return enhanced, details


class FIEBlock(nn.Module):
    # 单尺度 FIE 模块
    #
    # 整合空间分支和通道分支  最后 cat 得到融合特征
    # fused_channels = N * in_channels
    #   N = int(keep_original) + int(enable_spatial) + int(enable_channel)
    # 默认配置: N=3  fused_channels = 3 * in_channels
    #
    # 输入 shape:  (B, C, H, W)
    # 输出 shape:  (B, N*C, H, W)

    def __init__(self, in_channels: int, config: FIEBlockConfig) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.config = config
        self.spatial_branch = (
            SpatialInteractionEnhancement(in_channels, config.spatial) if config.enable_spatial_branch else None
        )
        self.channel_branch = (
            ChannelInteractionEnhancement(in_channels, config.channel) if config.enable_channel_branch else None
        )

    @property
    def fused_channels(self) -> int:
        # 动态计算融合后通道数  供检测头实例化时调用
        branch_count = int(self.config.keep_original_feature)
        branch_count += int(self.config.enable_spatial_branch)
        branch_count += int(self.config.enable_channel_branch)
        return branch_count * self.in_channels

    def forward(self, feature: Tensor, return_details: bool = False) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        fused_parts: List[Tensor] = []    # 待 cat 的各部分特征
        details: Dict[str, Tensor] = {}

        # 原始特征 f  保留以确保梯度直接流回主干
        if self.config.keep_original_feature:
            fused_parts.append(feature)

        # 空间增强特征 f_s
        if self.spatial_branch is not None:
            spatial_result = self.spatial_branch(feature, return_details=return_details)
            if return_details:
                spatial_feature, spatial_details = spatial_result
                details.update(spatial_details)
            else:
                spatial_feature = spatial_result
            fused_parts.append(spatial_feature)

        # 通道增强特征 f_c
        if self.channel_branch is not None:
            channel_result = self.channel_branch(feature, return_details=return_details)
            if return_details:
                channel_feature, channel_details = channel_result
                details.update(channel_details)
            else:
                channel_feature = channel_result
            fused_parts.append(channel_feature)

        if not fused_parts:
            raise ValueError("FIEBlock 至少需要保留原始特征或开启一个增强分支")

        # 通道维度拼接  (B, N*C, H, W)
        fused_feature = torch.cat(fused_parts, dim=1)

        if not return_details:
            return fused_feature
        details["fused_feature"] = fused_feature
        return fused_feature, details


class MultiScaleFIE(nn.Module):
    # 多尺度 FIE 模块
    #
    # 对 YOLOv7 的 P3/P4/P5 三个尺度特征分别独立应用 FIEBlock
    # 各尺度参数不共享  允许各自学习适合当前尺度的注意力模式
    #
    # 可选投影层:
    #   project_after_fusion=True:  1x1 Conv 投影  适配不修改检测头的场景
    #   project_after_fusion=False: 直接输出 3C  论文原始设计  需修改检测头
    #
    # 输入 shapes:  [(B,C1,H1,W1), (B,C2,H2,W2), (B,C3,H3,W3)]
    # 输出 shapes:  [(B,3C1,...), (B,3C2,...), (B,3C3,...)] 或投影后通道

    def __init__(self, config: MultiScaleFIEConfig | None = None) -> None:
        """
        # 可训练参数:
        - projections: 如果 project_after_fusion=True 则每个尺度对应一个 1x1 Conv 投影层  否则为 Identity 无参数
        """
        super().__init__()
        self.config = config or MultiScaleFIEConfig()
        if len(self.config.feature_channels) != len(self.config.per_scale):
            raise ValueError("feature_channels 与 per_scale 的长度必须一致")

        # 为每个尺度创建独立的 FIEBlock  参数相互独立
        self.blocks = nn.ModuleList(
            [FIEBlock(in_channels, block_config)
             for in_channels, block_config in zip(self.config.feature_channels, self.config.per_scale)]
        )

        projection_channels = self.config.projection_channels or self.config.feature_channels
        if self.config.project_after_fusion:
            # 投影层: 将融合后的 3C 通道压回 projection_channels
            self.projections = nn.ModuleList(
                [
                    nn.Conv2d(block.fused_channels, out_channels, kernel_size=1, bias=self.config.projection_use_bias)
                    for block, out_channels in zip(self.blocks, projection_channels)
                ]
            )
        else:
            # 恒等映射: 无额外参数  直接透传
            self.projections = nn.ModuleList([nn.Identity() for _ in self.blocks])

    def output_channels(self) -> Tuple[int, ...]:
        # 返回每个尺度的输出通道数  用于外部查询 (如检测头初始化)
        if self.config.project_after_fusion:
            return self.config.projection_channels or self.config.feature_channels
        return tuple(block.fused_channels for block in self.blocks)

    def forward(
        self,
        features: Sequence[Tensor],
        return_details: bool = False,
    ) -> Sequence[Tensor] | Tuple[Sequence[Tensor], List[Dict[str, Tensor]]]:
        if len(features) != len(self.blocks):
            raise ValueError(f"MultiScaleFIE 期望 {len(self.blocks)} 个尺度特征  实际得到 {len(features)} 个")

        enhanced_features: List[Tensor] = []
        all_details: List[Dict[str, Tensor]] = []

        # 逐尺度处理 P3/P4/P5
        for feature, block, projection in zip(features, self.blocks, self.projections):
            block_result = block(feature, return_details=return_details)
            if return_details:
                fused_feature, block_details = block_result
            else:
                fused_feature = block_result
                block_details = {}

            # 可选: 1x1 Conv 投影降通道  或 Identity 直接透传
            projected_feature = projection(fused_feature)
            enhanced_features.append(projected_feature)

            if return_details:
                block_details["projected_feature"] = projected_feature
                all_details.append(block_details)

        if not return_details:
            return enhanced_features
        return enhanced_features, all_details
