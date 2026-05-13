from __future__ import annotations

# --------------------------------------------------------
# infer.py - ICFIE-YOLO 推理与可视化脚本
#
# 功能:
#   1. 从 test_sample/ 批量读取 DarkFace 测试图像
#   2. 对每张图像执行 MSICN 光照矫正
#   3. 生成对比可视化图 (原图 | MSICN矫正图 | 差分图)
#   4. 保存结果到 results/ 目录
#   5. 若提供了权重文件  则同时执行完整检测并标注人脸框
#
# 基本用法 (仅 MSICN 可视化  无需权重):
#   conda activate ai2
#   python infer.py
#
# 完整检测用法 (需权重文件):
#   python infer.py --weights runs/train/darkface/weights/best.pt --detect
#
# 参数说明:
#   --input    测试图像目录  默认 test_sample/
#   --output   结果保存目录  默认 results/
#   --weights  模型权重路径  不传则只做 MSICN 可视化
#   --detect   传入此flag则执行完整检测 (需要 --weights)
#   --img-size 推理时缩放到的尺寸  默认 416
#   --conf     检测置信度阈值  默认 0.25
#   --iou      NMS IoU 阈值  默认 0.45
#   --device   推理设备  cpu 或 cuda 序号  默认 cpu
#   --batch    批量推理的 batch size  默认 1
# --------------------------------------------------------

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")    # 使用无头后端  服务器/无显示器环境也能保存图片
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from msicn import MSICN, MSICNConfig


def configure_matplotlib_cjk_font() -> None:
    # 为 Matplotlib 选择可用的中文字体  避免保存图片时标题出现乱码
    preferred_fonts = (
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Droid Sans Fallback",
    )
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    selected_font = None
    for preferred_font in preferred_fonts:
        selected_font = next((name for name in available_fonts if preferred_font in name), None)
        if selected_font is not None:
            break

    if selected_font is None:
        print("[警告] 未找到可用中文字体  Matplotlib 输出标题可能出现乱码")
        return

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
# 配置 Matplotlib 字体  确保中文标题正常显示  只需执行一次
configure_matplotlib_cjk_font()


# ================================================================
# 图像预处理工具函数
# ================================================================

def load_image_as_tensor(path: str, img_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """# 加载单张图像  缩放到指定尺寸  转换为 PyTorch 张量
    
     参数:
       path     - 图像文件路径
       img_size - 目标短边或正方形尺寸  (img_size, img_size)
     返回:
       tensor   - shape (1, 3, img_size, img_size)  float32  像素范围 [0, 1]
       orig_size- 原始图像 (H, W)  用于反变换和可视化
     """

    img = Image.open(path).convert("RGB")    # 统一转 RGB  避免 RGBA/灰度图问题
    orig_size = (img.height, img.width)      # 记录原始尺寸

    # 缩放到 (img_size, img_size)  LANCZOS 抗锯齿质量最好  比 BILINEAR 稍慢
    img_resized = img.resize((img_size, img_size), Image.LANCZOS)

    # PIL Image -> numpy (H, W, C) uint8 -> float32 归一化到 [0,1]
    img_np = np.array(img_resized, dtype=np.float32) / 255.0    # (H, W, 3)

    # HWC -> CHW -> 增加 batch 维度 -> (1, 3, H, W)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)    # (1,3,H,W)
    return tensor, orig_size


def tensor_to_numpy_image(tensor: Tensor) -> np.ndarray:
    """
    将推理结果张量转换为 numpy 图像数组  用于 matplotlib 可视化
    
    输入: (1, 3, H, W) float32 张量  值域 [0,1] 
    输出: (H, W, 3) float32 数组   值域 [0,1]
    """

    # squeeze 移除 batch 维  permute CHW -> HWC  clamp 确保值域合法
    return tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()


def get_image_paths(input_dir: str) -> List[str]:
    """递归扫描目录  返回所有图像文件路径  按文件名排序"""
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    all_paths = []
    for p in sorted(Path(input_dir).rglob("*")):
        if p.suffix.lower() in supported_ext:
            all_paths.append(str(p))
    return all_paths


# ================================================================
# MSICN 可视化函数
# ================================================================

def save_msicn_comparison(
    orig_tensor: Tensor,
    corrected_tensor: Tensor,
    save_path: str,
    filename: str,
    show_diff: bool = True,
) -> None:
    """
     保存 MSICN 前后对比图
    #
    # 布局:
    #   左:  原始低照度图像
    #   中:  MSICN 矫正后图像
    #   右:  像素差分图 (放大 5 倍以便观察)  显示矫正了哪些区域
    #
    # 参数:
    #   orig_tensor      - 原始图像张量 (1,3,H,W)
    #   corrected_tensor - 矫正后图像张量 (1,3,H,W)
    #   save_path        - 保存目录
    #   filename         - 保存文件名 (不含扩展名)
    #   show_diff        - True 时显示三列  False 时只显示两列
    # """

    orig_np = tensor_to_numpy_image(orig_tensor)           # (H,W,3) [0,1]
    corrected_np = tensor_to_numpy_image(corrected_tensor) # (H,W,3) [0,1]

    ncols = 3 if show_diff else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")    # 深色背景  贴合低照度主题

    # ---- 列1: 原始图像 ----
    axes[0].imshow(orig_np)
    axes[0].set_title("原始低照度图像", fontsize=12, color="white", pad=8)
    axes[0].axis("off")

    # ---- 列2: MSICN 矫正图 ----
    axes[1].imshow(corrected_np)
    axes[1].set_title("MSICN 矫正后", fontsize=12, color="white", pad=8)
    axes[1].axis("off")

    if show_diff:
        # ---- 列3: 差分图 ----
        # diff = corrected - orig  正值表示变亮  负值表示变暗
        # 乘以 5 放大差异  加 0.5 平移到 [0,1] 左右以便用灰色底图显示
        diff_np = np.clip((corrected_np - orig_np) * 5.0 + 0.5, 0.0, 1.0)
        axes[2].imshow(diff_np)
        axes[2].set_title("差分图 (x5倍放大  亮=变亮 暗=变暗)", fontsize=10, color="white", pad=8)
        axes[2].axis("off")

    plt.tight_layout(pad=0.5)
    out_file = os.path.join(save_path, f"{filename}_msicn.png")
    plt.savefig(out_file, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_global_coeff_visualization(
    orig_tensor: Tensor,
    global_coeffs: Tensor,
    save_path: str,
    filename: str,
) -> None:
    # 可视化 GIC 输出的全局光照矫正系数
    #
    # global_coeffs shape: (1, 3, 1, 1)  RGB 各一个标量系数
    # 系数越大  对应通道被提亮幅度越大
    # 可以直观看出 MSICN 对哪个颜色通道做了更多矫正

    orig_np = tensor_to_numpy_image(orig_tensor)
    # squeeze 到 (3,)  三个浮点数分别对应 R/G/B
    k = global_coeffs.squeeze().cpu().numpy()    # shape (3,)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")

    # 原图展示
    axes[0].imshow(orig_np)
    axes[0].set_title("原始图像", fontsize=12, color="white", pad=8)
    axes[0].axis("off")

    # RGB 全局系数柱状图
    channels = ["R", "G", "B"]
    colors = ["#ff4444", "#44ff44", "#4488ff"]
    bars = axes[1].bar(channels, k, color=colors, width=0.4, edgecolor="white", linewidth=1.2)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("GIC 全局矫正系数 k (越大=提亮幅度越大)", fontsize=11, color="white", pad=8)
    axes[1].set_facecolor("#16213e")
    axes[1].tick_params(colors="white")
    axes[1].spines["bottom"].set_color("white")
    axes[1].spines["left"].set_color("white")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].yaxis.label.set_color("white")
    # 在每个柱子顶部标注数值
    for bar, val in zip(bars, k):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=11,
        )

    plt.tight_layout(pad=0.5)
    out_file = os.path.join(save_path, f"{filename}_gic_coeff.png")
    plt.savefig(out_file, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ================================================================
# 推理主函数
# ================================================================

def run_msicn_inference(
    input_dir: str,
    output_dir: str,
    img_size: int,
    device: torch.device,
    max_images: Optional[int] = None,
    save_gic_vis: bool = True,
) -> None:
    # 批量执行 MSICN 光照矫正并保存可视化结果
    #
    # 参数:
    #   input_dir   - 输入图像目录
    #   output_dir  - 输出结果目录  自动创建子目录
    #   img_size    - 推理分辨率  正方形 (img_size x img_size)
    #   device      - 推理设备  cpu 或 cuda
    #   max_images  - 最多处理前 N 张图像  None 表示全部处理
    #   save_gic_vis- 是否保存 GIC 系数可视化图

    # ---- 初始化 MSICN ----
    # 使用默认配置: 6 层 IFE + GIC + LIC + 10 次 NLIS 迭代
    msicn = MSICN(MSICNConfig()).to(device)     # 初始化 MSICN 模型， 并移动到指定设备(device)
    msicn.eval()                                # eval 模式， 即推理摸式， 关闭 BN 统计更新， 确保结果确定性

    # 1. ---- 准备目录 ----
    msicn_dir = os.path.join(output_dir, "msicn_comparison")        # 对比图保存目录
    gic_dir = os.path.join(output_dir, "gic_visualization")         # GIC 系数可视化目录
    corrected_dir = os.path.join(output_dir, "corrected_images")    # 纯矫正图保存目录
    # 创建目录  若已存在则保持不变
    for d in [msicn_dir, gic_dir, corrected_dir]:
        os.makedirs(d, exist_ok=True)


    # 2. ---- 扫描图像文件 ----
    image_paths = get_image_paths(input_dir)
    if not image_paths:
        print(f"[警告] 在 {input_dir} 中未找到图像文件")
        return
    if max_images is not None:
        image_paths = image_paths[:max_images]

    total = len(image_paths)
    print(f"[INFO] 共找到 {total} 张图像  推理尺寸: {img_size}x{img_size}  设备: {device}")


    # 3. ---- 逐图推理 ----
    total_time = 0.0
    for i, img_path in enumerate(image_paths):
        stem = Path(img_path).stem    # 文件名不含扩展名  用于保存结果

        # 加载并预处理图像
        try:
            tensor, orig_size = load_image_as_tensor(img_path, img_size)
        except Exception as e:
            print(f"[跳过] {img_path}: {e}")
            continue

        tensor = tensor.to(device)    # 移动到指定设备

        # MSICN 前向传播
        t0 = time.perf_counter()
        with torch.no_grad():
            corrected, msicn_details = msicn(tensor, return_details=True)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # 保存 MSICN 前后对比图 默认在  results/msicn_comparison/  目录下  文件名格式: 原文件名_msicn.png
        save_msicn_comparison(
            orig_tensor=tensor,
            corrected_tensor=corrected,
            save_path=msicn_dir,
            filename=stem,
            show_diff=True,
        )

        # 保存 GIC 系数可视化 (可选  节省时间时可关闭)
        if save_gic_vis:
            save_global_coeff_visualization(
                orig_tensor=tensor,
                global_coeffs=msicn_details["global_coefficients"],
                save_path=gic_dir,
                filename=stem,
            )

        # 保存纯矫正图 (numpy uint8 格式  直接用于后续处理)
        corrected_np_uint8 = (tensor_to_numpy_image(corrected) * 255).astype(np.uint8)
        corrected_pil = Image.fromarray(corrected_np_uint8)
        corrected_pil.save(os.path.join(corrected_dir, f"{stem}_corrected.png"))

        # 进度打印
        fps = 1.0 / elapsed if elapsed > 0 else float("inf")
        print(f"[{i+1:4d}/{total}] {stem:<20s}  {elapsed*1000:.1f}ms  {fps:.1f}fps")

    # ---- 统计汇总 ----
    avg_ms = total_time / total * 1000 if total > 0 else 0.0
    print(f"\n[完成] 已处理 {total} 张图像")
    print(f"       平均推理时间: {avg_ms:.1f}ms/张")
    print(f"       对比图保存至: {msicn_dir}")
    print(f"       矫正图保存至: {corrected_dir}")
    if save_gic_vis:
        print(f"       GIC可视化至: {gic_dir}")


# ================================================================
# 命令行入口
# ================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICFIE-YOLO 推理与 MSICN 可视化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅执行 MSICN 可视化 (无需权重)
  python infer.py

  # 指定输入目录和输出目录
  python infer.py --input test_sample/ --output results/

  # 限定只处理前 10 张
  python infer.py --max-images 10

  # 使用 GPU 加速
  python infer.py --device cuda:0
        """,
    )
    parser.add_argument("--input",      type=str,   default="test_sample/",
                        help="输入图像目录  默认 test_sample/")
    parser.add_argument("--output",     type=str,   default="results/",
                        help="结果输出目录  默认 results/  自动创建")
    parser.add_argument("--img-size",   type=int,   default=416,
                        help="推理时图像缩放尺寸  默认 416")
    parser.add_argument("--device",     type=str,   default="cpu",
                        help="推理设备  如 cpu 或 cuda:0  默认 cpu")
    parser.add_argument("--max-images", type=int,   default=None,
                        help="最多处理前 N 张图像  默认全部处理")
    parser.add_argument("--no-gic-vis", action="store_true",
                        help="跳过 GIC 系数可视化  加快速度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 解析设备
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[警告] 指定了 cuda 但 CUDA 不可用  自动回退到 cpu")
        args.device = "cpu"
    device = torch.device(args.device)

    print("=" * 60)
    print(" ICFIE-YOLO 推理脚本")
    print("=" * 60)
    print(f" 输入目录: {args.input}")
    print(f" 输出目录: {args.output}")
    print(f" 推理尺寸: {args.img_size}x{args.img_size}")
    print(f" 推理设备: {device}")
    if args.max_images:
        print(f" 最多处理: {args.max_images} 张")
    print("=" * 60)

    run_msicn_inference(
        input_dir=args.input,               # 输入图像目录
        output_dir=args.output,             # 结果保存目录
        img_size=args.img_size,             # 推理时缩放尺寸
        device=device,                      # 推理设备
        max_images=args.max_images,         # 最多处理前 N 张图像
        save_gic_vis=not args.no_gic_vis,   # 是否保存 GIC 可视化图  默认保存  --no-gic-vis 可关闭以节省时间
    )


if __name__ == "__main__":
    main()
