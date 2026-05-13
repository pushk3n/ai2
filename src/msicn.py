from __future__ import annotations

# --------------------------------------------------------
# MSICN - Multi Scale Illumination Correction Network
# 多尺度光照矫正网络
#
# 论文出处: 基于ICFIE-YOLO的低照度图像目标检测方法
# 作者: 秦嘉奇 江泽涛 雷晓春
#
# 整体数据流:
#   低照度图像 I_in
#       -> IFE  (Illumination Feature Extraction)  提取32通道光照特征 F_ill
#       -> GIC  (Global  Illumination Correction)  输出全局矫正系数 k  in (0,1)
#       -> LIC  (Local   Illumination Correction)  输出10组局部矫正系数 in (-1,1)
#       -> NLIS (Non-Linear Illumination Stacking) 按公式迭代矫正图像像素
#   -> 矫正后图像 I_c  值域 [0,1]  尺寸与输入完全一致
#
# 设计关键:
#   1. 全程只使用检测损失驱动 MSICN 优化  不依赖成对增强数据集
#   2. NLIS 不含可训练参数  纯数值迭代组合 GIC/LIC 输出
#   3. 全局矫正消除整体亮度偏低  局部矫正处理不均匀光源
#
# 可训练参数:
#   IFE: 6 层卷积的权重和偏置                                   -- IFE.__init__ 中 blocks 列表
#   GIC: 2 层 5x5 空洞卷积和末端 1x1 卷积的权重和偏置              -- GIC.__init__ 中 features 和 proj
#   LIC: 局部精细分支和上下文分支的所有 3x3 卷积的权重和偏置          -- LIC: 局部精细分支和上下文分支的所有 3x3 卷积的权重和偏置
# --------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


def _conv3x3(in_channels: int, out_channels: int, *, dilation: int = 1) -> nn.Conv2d:
    # 构建标准 3x3 卷积层
    # 参数说明:
    #   in_channels  - 输入通道数
    #   out_channels - 输出通道数
    #   dilation     - 空洞卷积膨胀率  默认 1 即普通卷积
    # padding 的计算方式保证了输出的空间尺寸与输入完全相同 (same padding)
    # 对于 kernel=3 stride=1: padding = dilation  可以手动推导验证
    padding = dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=padding,
        dilation=dilation,
        bias=True,     # 保留偏置项  便于学习光照偏移量
    )


def _conv5x5(in_channels: int, out_channels: int, *, dilation: int = 1) -> nn.Conv2d:
    # 构建 5x5 卷积层
    # GIC 分支需要比 LIC 更大的感受野来捕捉全局光照分布
    # 因此单独封装 5x5 版本  其余设计思路与 _conv3x3 相同
    # same padding 公式: padding = ((kernel - 1) * dilation) // 2
    padding = ((5 - 1) * dilation) // 2
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=padding,
        dilation=dilation,
        bias=True,
    )




@dataclass(frozen=True)
class MSICNConfig:
    # MSICN 全局配置  使用 frozen=True 固化超参防止运行时意外修改
    # 所有通道数和结构参数均集中在此  便于消融实验时统一切换
    #
    # 字段含义:
    #   input_channels         - 输入图像通道数  RGB=3
    #   ife_channels           - IFE 六层卷积各层输出通道数  必须恰好 6 个值且总和为 32
    #   gic_hidden_channels    - GIC 特征变换的中间通道数
    #   gic_dilation           - GIC 5x5 卷积的空洞率  增大感受野覆盖全局
    #   lic_hidden_channels    - LIC 两条分支的中间通道数
    #   lic_context_dilation   - LIC 上下文分支的空洞率  覆盖更大局部区域
    #   lic_context_pool_kernel- LIC 上下文分支 AvgPool 的 kernel 大小
    #   local_groups           - NLIS 迭代次数 / LIC 输出的局部系数组数  论文中固定为 10
    #   clamp_output           - True 时在 NLIS 最后做 clamp(0,1)  保证输出数值稳定

    input_channels: int = 3
    ife_channels: Tuple[int, ...] = (4, 4, 4, 4, 8, 8)
    gic_hidden_channels: int = 32
    gic_dilation: int = 3
    lic_hidden_channels: int = 32
    lic_context_dilation: int = 2
    lic_context_pool_kernel: int = 3
    local_groups: int = 10
    clamp_output: bool = True

    @property
    def illumination_channels(self) -> int:
        # IFE 把每层输出特征在通道维度 cat  总通道数 = sum(ife_channels) = 32
        return sum(self.ife_channels)

    @property
    def local_channels(self) -> int:
        # LIC 最终输出的通道数 = 局部组数 x 每组通道数(=input_channels=3)
        # 即 10 x 3 = 30  对应 NLIS 中 10 次迭代每次 3 通道的局部系数
        return self.local_groups * self.input_channels




class IFE(nn.Module):
    # Illumination Feature Extraction - 光照特征提取模块
    #
    # 作用: 从低照度 RGB 图像中提取包含全局和局部光照信息的多尺度特征
    #
    # 结构设计:
    #   - 6 层串行 3x3 卷积  每层通道数按 ife_channels 配置
    #   - 每层输出都保存下来  最后在 channel 维度 concat
    #   - 之所以把每层都 concat 而不只取最后一层:
    #     浅层特征感受野小  捕捉局部像素级光照细节
    #     深层特征感受野大  捕捉区域级光照分布
    #     多尺度 concat 保留了从细到粗的光照信息  对应论文"多尺度"设计
    #
    # 输入 shape:  (B, 3, H, W)
    # 输出 shape:  (B, 32, H, W)   32 = sum(4+4+4+4+8+8)
    # 输出与输入的空间尺寸完全相同  每个 3x3 卷积均为 same padding

    def __init__(self, config: MSICNConfig) -> None:
        """
        # 可训练参数：
        # blocks: IFE 6 层卷积的权重和偏置
        """

        super().__init__()
        channels = tuple(config.ife_channels)
        if len(channels) != 6:
            raise ValueError("IFE 必须严格使用 6 层卷积")
        if sum(channels) != 32:
            raise ValueError("IFE 的各层输出通道之和必须为 32")


        
        blocks = []
        current_in_channels = config.input_channels
        for current_out_channels in channels:
            # 每个 block = Conv3x3 + ReLU
            # ReLU 引入非线性  使网络能学习非线性光照响应曲线
            blocks.append(
                nn.Sequential(
                    _conv3x3(current_in_channels, current_out_channels),
                    nn.ReLU(inplace=True),   # inplace 节省显存
                )
            )
            current_in_channels = current_out_channels    # 下一层的输入通道 = 当前层的输出通道
        self.blocks = nn.ModuleList(blocks)    # 注册为 Module  确保权重被 optimizer 管理

    def forward(self, image: Tensor) -> Tensor:
        # 记录每一层的输出  最后统一 cat
        features = []
        current = image          # current shape: (B, 3, H, W)
        for block in self.blocks:
            current = block(current)          # shape 变化: (B, 3, H, W) -> (B, 4, H, W) -> ... -> (B, 8, H, W)
            features.append(current)          # 每层输出都加入列表
        # torch.cat 在 dim=1(通道维度) 拼接  shape: (B, 32, H, W)
        return torch.cat(features, dim=1)




class GIC(nn.Module):
    # Global Illumination Correction - 全局光照矫正模块
    #
    # 作用: 基于 IFE 提取的光照特征  输出 R/G/B 三通道各自独立的全局矫正系数 k
    #
    # 输出值域:
    #   k in (0, 1)  由 Sigmoid 保证
    #   论文公式(1): I_global = I_in * (-k * I_in + (1 + k))
    #   当 k > 0 时  该变换对暗区亮度提升大于对亮区的提升  实现非线性全局曝光
    #
    # 结构设计:
    #   两层 5x5 空洞卷积 (dilation=3) -> AdaptiveAvgPool2d(1) -> 1x1 Conv -> Sigmoid
    #   使用 5x5 空洞卷积: 感受野 = 5 + (5-1)*(3-1) = 13  能覆盖较大区域的全局亮度分布
    #   AdaptiveAvgPool2d(1) 全局平均池化把特征图压缩到 1x1  得到全图的光照统计量
    #   1x1 Conv 将 gic_hidden_channels 映射到 3 通道 (RGB 各一个系数)
    #   最终 k 的 shape: (B, 3, 1, 1)  在 NLIS 中会被广播到全图
    #
    # 输入 shape:  (B, 32, H, W)
    # 输出 shape:  (B,  3, 1, 1)

    def __init__(self, config: MSICNConfig) -> None:
        """
        # 可训练参数：
        # futures :   GIC 2 层 5x5 空洞卷积
        # proj    :   GIC 末端 1x1 卷积
        """ 
        super().__init__()
        # 两层 5x5 空洞卷积 + ReLU  扩大感受野感知全局光照分布
        self.features = nn.Sequential(
            _conv5x5(config.illumination_channels, config.gic_hidden_channels, dilation=config.gic_dilation),
            nn.ReLU(inplace=True),
            _conv5x5(config.gic_hidden_channels, config.gic_hidden_channels, dilation=config.gic_dilation),
            nn.ReLU(inplace=True),
        )
        # 全局平均池化: 把 (B, C, H, W) 压成 (B, C, 1, 1)  提取全局光照统计量
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 1x1 卷积把 hidden_channels 映射到 input_channels(=3)  对应 RGB 三通道
        self.proj = nn.Conv2d(
            config.gic_hidden_channels,
            config.input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        # Sigmoid 确保 k 严格在 (0, 1)  避免矫正系数出现负值或大于 1 导致像素溢出
        self.activation = nn.Sigmoid()

    def forward(self, illumination_features: Tensor) -> Tensor:
        # illumination_features shape: (B, 32, H, W)
        global_features = self.features(illumination_features)   # (B, 32, H, W)
        pooled = self.pool(global_features)                       # (B, 32,  1, 1)
        # proj + sigmoid -> k shape: (B, 3, 1, 1)
        # 这里 1x1 的空间尺寸在 NLIS 中会被广播到全图 (H, W)
        return self.activation(self.proj(pooled))




class LIC(nn.Module):
    # Local Illumination Correction - 局部光照矫正模块
    #
    # 作用: 针对图像中光照不均匀区域  输出 10 组 x 3 通道的逐像素局部矫正系数
    #       每个像素位置拥有独立的矫正系数  因此可以处理复杂局部光源场景
    #
    # 输出值域:
    #   系数 in (-1, 1)  由 2*Sigmoid(x) - 1 保证
    #   注意: 这里不是标准 Sigmoid 输出 (0,1)  而是双极性输出  允许负向矫正
    #
    # 结构设计 (双分支并行):
    #   [局部精细分支] 两层 3x3 普通卷积  感受野较小  捕捉精细局部细节
    #   [上下文分支]   两层 3x3 卷积 + AvgPool + 一层 3x3 空洞卷积
    #                  感受野较大  捕捉周围环境的光照上下文
    #   两条分支输出在 channel 维度 cat -> 得到 30 通道 = 10组 x 3通道
    #
    # 为什么用双分支而不是单分支:
    #   低照度图像的局部光照具有两个层次的变化:
    #   1. 精细层次: 相邻像素间的微小差异 (如边缘处的高频光照变化)
    #   2. 上下文层次: 区域内的整体光照趋势 (如灯光形成的渐变区域)
    #   双分支分别建模两个层次  concat 后兼顾两种尺度
    #
    # 输入 shape:  (B, 32, H, W)
    # 输出 shape:  (B, 30, H, W)   30 = local_groups * input_channels = 10 * 3

    def __init__(self, config: MSICNConfig) -> None:
        """
        # 可训练参数：
        # local_branch:     局部精细分支    :       2 层 3x3 卷积
        # context_branch:   上下文分支      :       1 层普通 3x3 卷积 + 1 层空洞 3x3 卷积 + 1 层降维 3x3 卷积
        """

        super().__init__()
        # 把 30 通道均分给两条分支  若总数为奇数则 local_branch 取少的那半
        branch_channels = config.local_channels // 2          # 15 通道给局部精细分支
        remaining_channels = config.local_channels - branch_channels    # 15 通道给上下文分支

        # [局部精细分支]: 两层 3x3 普通卷积
        # 输入 32 通道  先变换到 lic_hidden_channels(32)  再压到 branch_channels(15)
        self.local_branch = nn.Sequential(
            _conv3x3(config.illumination_channels, config.lic_hidden_channels),
            nn.ReLU(inplace=True),
            _conv3x3(config.lic_hidden_channels, branch_channels),
            # 注意: 这里没有激活函数  最终激活统一在 forward 中的 Sigmoid 完成
        )

        # [上下文分支]: 更大感受野捕捉局部光照趋势
        # Conv3x3 -> ReLU -> 空洞Conv3x3(dilation=2) -> ReLU -> AvgPool -> Conv3x3
        # AvgPool 做局部平均  相当于对光照系数做低通滤波  让系数在空间上更平滑
        self.context_branch = nn.Sequential(
            _conv3x3(config.illumination_channels, config.lic_hidden_channels),
            nn.ReLU(inplace=True),
            _conv3x3(
                config.lic_hidden_channels,
                config.lic_hidden_channels,
                dilation=config.lic_context_dilation,    # dilation=2  感受野 = 3+(3-1)*1 = 5
            ),
            nn.ReLU(inplace=True),
            # AvgPool 平滑局部光照系数  kernel=3 stride=1 padding=1 保证尺寸不变
            nn.AvgPool2d(
                kernel_size=config.lic_context_pool_kernel,
                stride=1,
                padding=config.lic_context_pool_kernel // 2,     # same padding
            ),
            _conv3x3(config.lic_hidden_channels, remaining_channels),
        )
        # 激活函数: 标准 Sigmoid  之后在 forward 中做 2*sigmoid(x)-1 变换
        self.activation = nn.Sigmoid()

    def forward(self, illumination_features: Tensor) -> Tensor:
        # illumination_features shape: (B, 32, H, W)
        local_features = self.local_branch(illumination_features)       # (B, 15, H, W)
        context_features = self.context_branch(illumination_features)   # (B, 15, H, W)

        # 在通道维度 cat  得到 30 通道原始系数
        raw_coefficients = torch.cat([local_features, context_features], dim=1)   # (B, 30, H, W)

        # 论文规定 LIC 输出必须在 (-1, 1)
        # 变换: 2 * sigmoid(x) - 1
        #   sigmoid(x) in (0, 1)
        #   2*sigmoid(x) in (0, 2)
        #   2*sigmoid(x) - 1 in (-1, 1)
        # 这样允许局部系数为负值  即允许局部压暗来平衡过亮区域
        return 2.0 * self.activation(raw_coefficients) - 1.0




class NLIS(nn.Module):
    # Non-Linear Illumination Stacking - 非线性光照迭代堆叠模块
    #
    # 作用: 将 GIC 的全局系数 k 和 LIC 的局部系数 c_i 依次作用于输入图像
    #       实现从全局到局部的多层次光照矫正
    #
    # 重要: NLIS 不含任何可训练参数  是纯粹的数值计算模块
    #       所有可学习参数都在 GIC 和 LIC 中
    #
    # 论文公式(1) - 全局矫正:
    #   I_global = I_in * (-k * I_in + (1 + k))
    #   其中 k in (0,1)  I_in in [0,1]
    #   展开: I_global = I_in + k * I_in * (1 - I_in)
    #         注意 I_in*(1-I_in) 在 I_in=0 和 I_in=1 时为 0  在 I_in=0.5 时最大
    #         所以 k 控制的是中间亮度像素的提升幅度  天然保留纯黑和纯白不变
    #
    # 论文公式(2) - 局部迭代矫正:
    #   I_{i+1} = I_i * (c_i * I_i + (1 - c_i))   for i = 0..9
    #   其中 c_i in (-1,1)
    #   当 c_i > 0 时: 系数 > 0  对亮区增益更大  让亮区更亮
    #   当 c_i < 0 时: 系数 < 0  对暗区增益更大  让暗区更亮 (提亮压暗区)
    #   10 次迭代形成非线性曲线叠加  逐步细化局部光照调整
    #
    # 输入:
    #   image              (B, 3, H, W)   原始低照度图像  像素范围 [0,1]
    #   global_coefficients(B, 3, 1, 1)   GIC 输出的全局系数 k
    #   local_coefficients (B, 30, H, W)  LIC 输出的局部系数  每组 3 通道共 10 组
    # 输出:
    #   corrected image    (B, 3, H, W)   矫正后图像  值域 [0,1] (clamp 保证)

    def __init__(self, config: MSICNConfig) -> None:
        super().__init__()
        self.local_groups = config.local_groups        # = 10  迭代次数
        self.input_channels = config.input_channels    # = 3   RGB
        self.clamp_output = config.clamp_output        # True  开启输出 clamp

    def forward(self, image: Tensor, global_coefficients: Tensor, local_coefficients: Tensor) -> Tensor:
        batch_size, channels, height, width = image.shape
        if channels != self.input_channels:
            raise ValueError(f"MSICN 期望输入通道为 {self.input_channels} 实际得到 {channels}")

        # 先对输入图像做一次 clamp  防止输入本身存在浮点精度越界
        current = image.clamp(0.0, 1.0)    # (B, 3, H, W)

        # ---- 步骤1: 全局矫正 对应公式(1) ----
        # global_coefficients shape: (B, 3, 1, 1)
        # current shape:             (B, 3, H, W)
        # 广播规则: (B,3,1,1) 自动扩展到 (B,3,H,W)  逐像素相乘
        # I_global = I * (-k * I + (1 + k))
        #          = I * (1 + k * (1 - I))
        current = current * (-global_coefficients * current + (1.0 + global_coefficients))

        # ---- 步骤2: 局部迭代矫正 对应公式(2) ----
        # 把 (B, 30, H, W) reshape 成 (B, 10, 3, H, W)  便于逐组遍历
        local_coefficients = local_coefficients.view(
            batch_size,
            self.local_groups,      # 10 组
            self.input_channels,    # 每组 3 通道对应 RGB
            height,
            width,
        )
        for group_index in range(self.local_groups):
            # 取第 group_index 组系数  shape: (B, 3, H, W)
            group_coefficients = local_coefficients[:, group_index]
            # 公式(2): I_{i+1} = I_i * (c_i * I_i + (1 - c_i))
            #                   = I_i * (1 + c_i * (I_i - 1))
            # c_i in (-1,1)  所以 c_i*(I_i-1) 的范围是有界的  不会出现无穷大
            current = current * (group_coefficients * current + (1.0 - group_coefficients))

        # 最终 clamp 确保数值稳定  防止浮点运算累积导致轻微越界
        if self.clamp_output:
            current = current.clamp(0.0, 1.0)
        return current




class MSICN(nn.Module):
    # Multi Scale Illumination Correction Network - 多尺度光照矫正网络
    #
    # 整合 IFE / GIC / LIC / NLIS 的顶层模块
    #
    # 调用顺序:
    #   image -> IFE -> illumination_features (32ch)
    #                -> GIC -> global_coefficients   (3ch 1x1)
    #                -> LIC -> local_coefficients    (30ch HxW)
    #   image + global_coefficients + local_coefficients -> NLIS -> corrected_image
    #
    # return_details=True 时额外返回中间特征  方便调试和可视化检验矫正效果
    # 正常推理/训练时传 False  节省显存不保留中间张量

    def __init__(self, config: MSICNConfig | None = None) -> None:
        super().__init__()
        self.config = config or MSICNConfig()    # 使用默认配置  或传入自定义配置
        self.ife = IFE(self.config)
        self.gic = GIC(self.config)
        self.lic = LIC(self.config)
        self.nlis = NLIS(self.config)

    def forward(self, image: Tensor, return_details: bool = False) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        # ---- 第一步: IFE 提取光照特征 ----
        # image shape: (B, 3, H, W)
        illumination_features = self.ife(image)    # (B, 32, H, W)

        # ---- 第二步: 并行计算 GIC 和 LIC 矫正系数 ----
        # 两条分支共享同一份 IFE 特征作为输入  避免重复计算
        global_coefficients = self.gic(illumination_features)    # (B,  3, 1, 1)  in (0,1)
        local_coefficients = self.lic(illumination_features)     # (B, 30, H, W)  in (-1,1)

        # ---- 第三步: NLIS 非线性叠加矫正 ----
        corrected = self.nlis(image, global_coefficients, local_coefficients)    # (B, 3, H, W)

        if not return_details:
            return corrected    # 正常推理路径  只返回矫正图像

        # 调试路径: 返回矫正图像 + 中间特征字典
        details = {
            "illumination_features": illumination_features,    # IFE 输出的光照特征图
            "global_coefficients": global_coefficients,        # GIC 输出的全局矫正系数
            "local_coefficients": local_coefficients,          # LIC 输出的局部矫正系数
        }
        return corrected, details