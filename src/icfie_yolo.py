from __future__ import annotations

# --------------------------------------------------------
# icfie_yolo.py — 顶层串联适配器
#
# 职责: 将四层模块串联成完整的端到端检测模型
#
# 四层对应关系:
#   1. msicn.py        → MSICN            光照矫正
#   2. yolo_wrapper.py → YOLO Backbone+Neck  特征提取
#   3. fie.py          → MultiScaleFIE    特征增强
#   4. detect.py       → Detect Head      检测输出
#
# 设计原则:
#   - 四层模块均由外部传入，本模块只做串联适配，不持有任何业务逻辑
#   - backbone_neck / detect_head 由调用方（如 yolo_wrapper.build_yolov7_components）构建
#   - 冒烟测试用 Mock 类已迁移至 yolo_wrapper.py (backbone) 和 detect.py (head)
#
# 正向数据流:
#   image (B,3,H,W)
#     -> MSICN (可选)       光照矫正  -> corrected_image (B,3,H,W)
#     -> backbone_neck      特征提取  -> [P3, P4, P5] 多尺度特征
#     -> MultiScaleFIE (可选)  特征增强  -> [P3', P4', P5']
#     -> detect_head        检测输出  -> predictions
# --------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

# 本地依赖
from fie import MultiScaleFIE, MultiScaleFIEConfig
from msicn import MSICN, MSICNConfig


@dataclass(frozen=True)
class ICFIEYOLOConfig:
    """
    ICFIE-YOLO 顶层配置
    Args:
        enable_msicn     - True  : 在主干前执行光照矫正 (完整 ICFIE-YOLO)
                                False : 跳过 MSICN  直接送主干 (消融: 验证 MSICN 贡献)
        enable_fie       - True  : 在检测头前执行特征增强 (完整 ICFIE-YOLO)
                                False : 跳过 FIE  直接送检测头 (消融: 验证 FIE 贡献)
        feature_names    - backbone_neck 以 dict 返回时  按此顺序取 P3/P4/P5
                            backbone_neck 以 list/tuple 返回时  顺序即为 P3->P4->P5
        msicn            - MSICN 子模块配置
        fie              - MultiScaleFIE 子模块配置
    """
    enable_msicn: bool = True
    enable_fie: bool = True
    feature_names: Tuple[str, ...] = ("p3", "p4", "p5")
    msicn: MSICNConfig = field(default_factory=MSICNConfig)
    fie: MultiScaleFIEConfig = field(default_factory=MultiScaleFIEConfig)


class ICFIEYOLO(nn.Module):
    # ICFIE-YOLO 顶层模型包装器
    #
    # 使用方式:
    #   1. 传入满足接口的 backbone_neck 和 detect_head
    #   2. 配置 ICFIEYOLOConfig 控制模块开关
    #   3. 调用 forward(image) 或 forward(image, return_details=True)
    #
    # backbone_neck 接口约定:
    #   输入:  (B, 3, H, W) 的图像张量
    #   输出:  dict{"p3":..., "p4":..., "p5":...}
    #          或 list/tuple [P3_tensor, P4_tensor, P5_tensor]
    #
    # detect_head 接口约定:
    #   输入:  [P3, P4, P5] 的列表  每个元素为 (B, C, H, W)
    #          注意: 若 FIE 未投影  通道 C 为 3 * 原始通道
    #   输出:  任意形式的预测结果 (Tensor 或 Tensor tuple 均可)
    #
    # 论文主线映射:
    #   Step 1: I_c = MSICN(I_in)                                  对应论文第 3.2 节
    #   Step 2: [P3,P4,P5] = YOLOv7_BackboneNeck(I_c)             对应检测骨干特征抽取
    #   Step 3: [P3',P4',P5'] = FIE([P3,P4,P5])                   对应论文第 3.3 节
    #   Step 4: pred = DetectHead([P3',P4',P5'])                  对应检测输出

    def __init__(
        self,
        backbone_neck: nn.Module,
        detect_head: nn.Module,
        config: ICFIEYOLOConfig | None = None,
        msicn_module: MSICN | None = None,
        fie_module: MultiScaleFIE | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ICFIEYOLOConfig()
        self.backbone_neck = backbone_neck    # 外部传入的主干网络
        self.detect_head = detect_head        # 外部传入的检测头
        # 允许直接传入预构建的 MSICN/FIE  便于权重共享或消融测试
        self.msicn = msicn_module or MSICN(self.config.msicn)
        self.fie = fie_module or MultiScaleFIE(self.config.fie)
        self._validate_detect_head_channels()

    def _detect_input_channels(self) -> Tuple[int, ...] | None:
        # 优先读取检测头适配器显式暴露的输入通道约束
        expected_in_channels = getattr(self.detect_head, "expected_in_channels", None)
        if expected_in_channels is not None:
            return tuple(int(channel) for channel in expected_in_channels)

        detect_config = getattr(self.detect_head, "config", None)
        if detect_config is not None and hasattr(detect_config, "in_channels"):
            return tuple(int(channel) for channel in detect_config.in_channels)

        return None

    def _validate_detect_head_channels(self) -> None:
        detect_input_channels = self._detect_input_channels()
        if detect_input_channels is None:
            return

        # 显式通道断言是当前工程的重要安全检查。
        # 论文原始 FIE 输出为 3C；若启用了 project_after_fusion，则输出回到 C。
        # 这里必须在模型构造期就校验清楚，避免把通道不匹配的问题拖到训练时才暴露。
        expected_channels = self.fie.output_channels() if self.config.enable_fie else self.config.fie.feature_channels
        assert list(expected_channels) == list(detect_input_channels), (
            f"FIE 输出通道 {expected_channels} 与检测头期望通道 {detect_input_channels} 不匹配"
        )

    def _align_corrected_image_dtype(self, corrected_image: Tensor) -> Tensor:
        # PRD 要求在 MSICN 与主干之间显式对齐 dtype
        # 原因: Stage B / Stage C 中 backbone 可能处于 fp16 或混合精度状态，
        # 而 MSICN 默认输出常为 fp32；若不显式对齐，后续卷积会发生 dtype mismatch。
        first_parameter = next(self.backbone_neck.parameters(), None)
        if first_parameter is None or corrected_image.dtype == first_parameter.dtype:
            return corrected_image
        return corrected_image.to(dtype=first_parameter.dtype)

    def forward_stage_a(self, image: Tensor) -> object:
        # PRD 阶段 A 专用前向
        #
        # 数据流:
        #   backbone_neck (梯度流通) -> detect_head (梯度流通)
        #   MSICN 和 FIE 均不参与前向  目的是预训练纯 YOLO 基线 (ExDark 12 类)
        #   stage_a 权重可在推理时搭配 ENABLE_FIE=False 使用，实现独立模块验证
        #
        # 调用方在切换到阶段 A 之前必须确保:
        #   1. apply_training_stage(model, STAGE_A) 已调用  确保 requires_grad 设置正确
        #   2. backbone_neck / detect_head 已切换到 train 模式  msicn / fie 已切换到 eval 模式
        raw_features = self._normalize_features(self.backbone_neck(image))
        predictions = self.detect_head(raw_features)
        return predictions

    def forward_stage_b(self, image: Tensor) -> object:
        # PRD 阶段 B 专用前向
        #
        # 数据流:
        #   MSICN (梯度流通) -> backbone_neck (no_grad) -> detect_head (no_grad)
        #   FIE 在阶段 B 不参与前向  因为阶段 B 对齐的基准是阶段 A 训练好的纯 YOLO
        #
        # 调用方在切换到阶段 B 之前必须确保:
        #   1. apply_training_stage(model, STAGE_B) 已调用  确保 requires_grad 设置正确
        #   2. backbone_neck / detect_head 已切换到 eval 模式
        corrected_image = self.msicn(image)
        corrected_image = self._align_corrected_image_dtype(corrected_image)
        # 注意: 不使用 torch.no_grad()。
        # backbone_neck / detect_head 的参数已在 apply_training_stage 中设置 requires_grad=False，
        # 不会被优化器更新；但计算图必须保留，以确保梯度能从 Loss 经 backbone 反传到 MSICN 输出。
        # 若用 no_grad 包裹，backbone 的前向不记录图，corrected_image 的梯度链断开，
        # MSICN 将永远得不到任何梯度，阶段 B 训练失效。
        raw_features = self._normalize_features(self.backbone_neck(corrected_image))
        predictions = self.detect_head(raw_features)
        return predictions

    def _normalize_features(self, features: object) -> list[Tensor]:
        # 将 backbone_neck 的输出统一为 [P3, P4, P5] 列表格式
        # 兼容两种返回形式:
        #   dict  : {"p3": tensor, "p4": tensor, "p5": tensor}  按 feature_names 顺序取
        #   list/tuple: [P3_tensor, P4_tensor, P5_tensor]  直接使用  要求长度一致

        if isinstance(features, dict):
            # 检查必要的 key 是否都存在
            missing_keys = [key for key in self.config.feature_names if key not in features]
            if missing_keys:
                raise KeyError(f"特征字典缺少键: {missing_keys}")
            return [features[key] for key in self.config.feature_names]

        if isinstance(features, (list, tuple)):
            if len(features) != len(self.config.feature_names):
                raise ValueError(
                    f"ICFIEYOLO 期望 {len(self.config.feature_names)} 个特征尺度  实际得到 {len(features)} 个"
                )
            return list(features)

        raise TypeError("backbone_neck 必须返回 list/tuple 或 dict 形式的多尺度特征")

    def forward(self, image: Tensor, return_details: bool = False) -> Tensor | Dict[str, object]:
        # return_details=False: 只返回检测头的原始输出  用于正常训练/推理
        # return_details=True : 返回包含所有中间结果的字典  用于调试和可视化
        # 这条 forward 严格保持论文推理主线顺序，不在串联层引入任何额外业务分支。
        details: Dict[str, object] = {}

        # ---- Step 1: MSICN 光照矫正 ----
        if self.config.enable_msicn:
            msicn_result = self.msicn(image, return_details=return_details)
            if return_details:
                corrected_image, msicn_details = msicn_result
                details["msicn"] = msicn_details    # 包含 illumination_features/global_coeff/local_coeff
            else:
                corrected_image = msicn_result
        else:
            # 消融: 跳过 MSICN  直接用原始低照度图像
            corrected_image = image

        corrected_image = self._align_corrected_image_dtype(corrected_image)

        # ---- Step 2: Backbone + Neck 提取多尺度特征 ----
        # corrected_image 经主干提取: P3(1/8 尺度) P4(1/16) P5(1/32)
        raw_features = self.backbone_neck(corrected_image)
        # 统一格式: dict 或 list/tuple -> [P3_tensor, P4_tensor, P5_tensor]
        multi_scale_features = self._normalize_features(raw_features)

        # ---- Step 3: FIE 特征交互增强 ----
        if self.config.enable_fie:
            fie_result = self.fie(multi_scale_features, return_details=return_details)
            if return_details:
                enhanced_features, fie_details = fie_result
                details["fie"] = fie_details    # 包含各尺度的空间/通道注意力
            else:
                enhanced_features = fie_result
        else:
            # 消融: 跳过 FIE  直接用主干特征
            enhanced_features = multi_scale_features

        # ---- Step 4: 检测头输出 ----
        predictions = self.detect_head(enhanced_features)

        if not return_details:
            return predictions    # 正常推理路径

        # 调试路径: 返回完整中间结果
        details["corrected_image"] = corrected_image
        details["raw_features"] = multi_scale_features
        details["enhanced_features"] = enhanced_features
        details["predictions"] = predictions
        return details


# ================================================================
# Mock 类已迁移:
#   MockYOLOv7BackboneNeck / MockYOLOv7BackboneNeckConfig / ConvBNAct
#       → yolo_wrapper.py
#   MockYOLOv7DetectHead / MockYOLOv7DetectHeadConfig
#       → detect.py
# 冒烟测试请从对应模块导入。
# ================================================================
