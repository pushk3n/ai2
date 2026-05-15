"""
ExDark 数据集预处理脚本
将原始 ExDark 数据集转换为 YOLO 格式，并更新训练配置，之后可直接启动训练。

使用方法：
    在项目根目录执行
    python data-process.py

前提条件：
    1. 下载 ExDark 图像集并解压，确保 EXDARK_IMG_DIR 路径正确。
    2. 下载 ExDark 标注文件（ExDark_Annno），确保 EXDARK_ANNO_DIR 路径正确。
    3. 原始 ExDark 数据不要求放在项目目录内；这里只需要把宏定义改成你的真实数据路径。

ExDark 标注文件格式（bbGt version=3，文件名 = 图像文件名 + .txt）：
    % bbGt version=3
    ClassName x y w h 0 0 0 0 0 0 0
    ClassName x y w h 0 0 0 0 0 0 0
    ...
    其中 x, y 为左上角像素坐标（绝对值），w, h 为像素宽高（绝对值）。
    同一文件内可包含多个不同类别的 bbox，每行开头为类别名。

输出结构（OUTPUT_DIR）：
    data/Exdark/
        images/
            train/   ← 训练集图像
            val/     ← 验证集图像
        labels/
            train/   ← 训练集 YOLO 格式标注 (.txt)
            val/     ← 验证集 YOLO 格式标注 (.txt)
        train.txt    ← 训练集图像路径列表
        val.txt      ← 验证集图像路径列表

脚本会自动同步 configs/train.yaml 中以下字段：
    1. dataset.train_path / val_path
    2. dataset.num_classes / class_names
    3. yolo.num_classes
    4. nc / names（兼容 YOLOv7 常见数据集配置字段）
"""

from __future__ import annotations

# ============================================================
# 宏定义区 —— 根据本机路径修改以下配置
# ============================================================

# ExDark 图像根目录（内含 12 个类别子目录，每目录下存放 .jpg/.png 图像）
# 说明: 原始数据集可以放在项目外部任意位置，这里写你的真实绝对路径即可。
EXDARK_IMG_DIR = "/path/to/your/ExDark/images"

# ExDark 标注根目录（内含 12 个类别子目录，每目录下存放与图像同名的 .txt 标注文件）
# 标注文件命名规则：<原始图像文件名>.txt（保留图像扩展名），如 2015_00001.jpg.txt
EXDARK_ANNO_DIR = "/path/to/your/ExDark_Annno"

# 预处理输出目录（相对工作目录 github-ai2/）
OUTPUT_DIR = "data/Exdark"

# 训练配置文件路径（脚本末尾会自动更新 dataset.train_path / val_path）
TRAIN_CONFIG_PATH = "configs/train.yaml"

# 是否在生成前清空 OUTPUT_DIR 下的 images/train|val 与 labels/train|val
# 这样可避免重新划分数据集后残留旧文件，保证 images/labels 与 train.txt/val.txt 完全一致。
CLEAN_OUTPUT_SPLIT_DIRS = True

# 是否在导出阶段重编码 PNG 图像。
# ExDark 中部分 PNG 自带错误的 ICC / cHRM 元数据，训练读取时会触发 libpng warning。
# 该操作会增加 CPU / IO 开销，默认关闭；只有在确实需要清理 libpng warning 时再手动打开。
NORMALIZE_PNG_OUTPUT = True

# 仅当 PNG 包含以下可疑元数据时才执行清洗，避免对所有 PNG 做不必要的重编码。
PNG_METADATA_KEYS_TO_STRIP = ("icc_profile", "chromaticity", "srgb", "gamma")

# 验证集占比（0.0~1.0）
VAL_SPLIT_RATIO = 0.2

# 随机种子（影响训练/验证集划分）
SPLIT_SEED = 42

# ============================================================
# 以下内容无需修改
# ============================================================

import os
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# ExDark 12 类别（顺序与 train.yaml 保持一致）
CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike",
    "People", "Table",
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# 支持的图像扩展名（统一为小写再比较）
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def find_anno_file(img_path: Path, anno_root: Path) -> Path | None:
    """根据图像路径，在 anno_root 中寻找对应的标注 .txt 文件。

    ExDark 标注文件命名规则：<原始图像文件名>.txt（包含图像扩展名）
    例如：图像 2015_00001.jpg → 标注 2015_00001.jpg.txt

    支持两种目录布局：
    1. anno_root/<ClassName>/<imagename>.<ext>.txt （按类别分子目录）
    2. anno_root/<imagename>.<ext>.txt             （平铺）
    """
    # 标注文件名 = 图像文件名 + ".txt"（保留原始扩展名）
    anno_name = img_path.name + ".txt"
    cls_name = img_path.parent.name
    # 布局 1：按类别子目录
    candidate1 = anno_root / cls_name / anno_name
    if candidate1.exists():
        return candidate1
    # 布局 2：平铺
    candidate2 = anno_root / anno_name
    if candidate2.exists():
        return candidate2
    return None


def parse_exdark_anno(anno_path: Path, img_w: int, img_h: int) -> list[str]:
    """解析 ExDark 原始标注文件，返回 YOLO 格式字符串列表。

    ExDark 标注格式（bbGt version=3）：
        % bbGt version=3
        ClassName x y w h 0 0 0 0 0 0 0
        ...
        其中 x, y 为左上角像素坐标，w, h 为像素宽高（绝对值）。
        同一文件内可包含多个不同类别的 bbox，每行带类别名。

    YOLO 格式（归一化）：
        class_id cx cy w h
        其中 cx, cy 为归一化中心坐标，w, h 为归一化宽高。
    """
    yolo_lines: list[str] = []
    lines = anno_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%"):
            # 跳过空行和注释行（如 % bbGt version=3）
            continue

        parts = line.split()
        # 格式：ClassName x y w h [其余字段...]
        if len(parts) < 5:
            continue

        cls_name = parts[0]
        class_idx = CLASS_TO_IDX.get(cls_name)
        if class_idx is None:
            # 类别名不在 12 类列表中，跳过
            continue

        try:
            x, y, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue

        # 防止零宽高
        if bw <= 0 or bh <= 0 or img_w <= 0 or img_h <= 0:
            continue

        # 绝对像素坐标（左上角 + 宽高）→ 归一化中心坐标
        cx = (x + bw / 2.0) / img_w
        cy = (y + bh / 2.0) / img_h
        nw = bw / img_w
        nh = bh / img_h

        # 裁剪至 [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        yolo_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines


def collect_images(img_root: Path) -> list[Path]:
    """收集 img_root 下所有图像文件，忽略 macOS 元数据目录。"""
    images: list[Path] = []
    for p in img_root.rglob("*"):
        if "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            images.append(p)
    return images


def prepare_output_dirs(out_root: Path) -> None:
    """准备输出目录，必要时先清理旧的 train/val 切分结果。"""
    split_dirs = [
        out_root / "images" / "train",
        out_root / "images" / "val",
        out_root / "labels" / "train",
        out_root / "labels" / "val",
    ]

    if CLEAN_OUTPUT_SPLIT_DIRS:
        for split_dir in split_dirs:
            if split_dir.exists():
                shutil.rmtree(split_dir)

    for split_dir in split_dirs:
        split_dir.mkdir(parents=True, exist_ok=True)


def read_valid_image_size(img_path: Path) -> tuple[int, int] | None:
    """读取图像尺寸，并尽早识别损坏图像。"""
    try:
        # verify() 会检查文件完整性；检查后需重新打开才能继续读取属性。
        with Image.open(img_path) as image:
            image.verify()

        with Image.open(img_path) as image:
            img_w, img_h = image.size
    except Exception as exc:
        print(f"  [警告] 图像损坏或无法读取: {img_path} ({exc})，已跳过。")
        return None

    if img_w <= 0 or img_h <= 0:
        print(f"  [警告] 图像尺寸非法: {img_path} ({img_w}x{img_h})，已跳过。")
        return None

    return img_w, img_h


def export_image_file(src_img: Path, dst_img: Path) -> bool:
    """导出图像文件；必要时重编码 PNG 以清理异常元数据。"""
    if NORMALIZE_PNG_OUTPUT and src_img.suffix.lower() == ".png":
        try:
            with Image.open(src_img) as image:
                image.load()
                if not any(key in image.info for key in PNG_METADATA_KEYS_TO_STRIP):
                    shutil.copy2(src_img, dst_img)
                    return False

                image.save(dst_img, format="PNG", pnginfo=PngInfo())
            return True
        except Exception as exc:
            print(f"  [警告] PNG 重编码失败: {src_img} ({exc})，将退回原样拷贝。")

    shutil.copy2(src_img, dst_img)
    return False


def update_train_config(config_path: Path, train_txt: Path, val_txt: Path) -> None:
    """同步训练配置，兼容当前工程配置与 YOLOv7 常见字段。"""
    if not config_path.exists():
        print(f"  [警告] 配置文件 {config_path} 不存在，跳过更新。")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件 {config_path} 顶层必须是 YAML 映射。")

    cfg.setdefault("dataset", {})
    cfg["dataset"]["train_path"] = str(train_txt.resolve())
    cfg["dataset"]["val_path"] = str(val_txt.resolve())
    cfg["dataset"]["num_classes"] = len(CLASSES)
    cfg["dataset"]["class_names"] = list(CLASSES)

    cfg.setdefault("yolo", {})
    cfg["yolo"]["num_classes"] = len(CLASSES)

    cfg["nc"] = len(CLASSES)
    cfg["names"] = list(CLASSES)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"      已同步数据集路径、类别数量和类别名称到 {config_path}。")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main() -> None:
    img_root = Path(EXDARK_IMG_DIR)
    anno_root = Path(EXDARK_ANNO_DIR)
    out_root = Path(OUTPUT_DIR)
    config_path = Path(TRAIN_CONFIG_PATH)

    # ── 路径校验 ──────────────────────────────
    if not img_root.exists():
        print(f"[错误] 图像目录不存在: {img_root}")
        print("  请修改脚本顶部的 EXDARK_IMG_DIR 宏定义后重试。")
        sys.exit(1)

    if not anno_root.exists():
        print(f"[错误] 标注目录不存在: {anno_root}")
        print("  请下载 ExDark 标注文件（ExDark_Annno）后，修改脚本顶部的 EXDARK_ANNO_DIR 宏定义。")
        print("  下载地址: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/releases")
        sys.exit(1)

    # ── 创建输出目录 ──────────────────────────
    prepare_output_dirs(out_root)

    # ── 收集图像并划分训练/验证集 ─────────────
    print("[1/4] 扫描图像文件...")
    all_images = collect_images(img_root)
    if not all_images:
        print(f"[错误] 在 {img_root} 下未找到任何图像文件。")
        sys.exit(1)
    print(f"      共找到 {len(all_images)} 张图像。")

    random.seed(SPLIT_SEED)
    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * VAL_SPLIT_RATIO))
    val_set = set(str(p) for p in all_images[:n_val])

    # ── 拷贝图像 + 转换标注 ───────────────────
    print("[2/4] 拷贝图像并转换标注为 YOLO 格式...")

    skipped_no_anno = 0
    skipped_no_bbox = 0
    skipped_bad_image = 0
    normalized_png = 0
    processed = 0
    train_paths: list[str] = []
    val_paths: list[str] = []

    for img_path in all_images:
        split = "val" if str(img_path) in val_set else "train"

        # 目标路径（保留 <ClassName>_<stem> 防止不同类别同名文件冲突）
        cls_prefix = img_path.parent.name
        new_name = f"{cls_prefix}_{img_path.name}"
        dst_img = out_root / "images" / split / new_name
        dst_lbl = out_root / "labels" / split / (Path(new_name).stem + ".txt")

        # 寻找标注文件
        anno_path = find_anno_file(img_path, anno_root)
        if anno_path is None:
            skipped_no_anno += 1
            continue

        # 读取图像尺寸，并在复制前识别损坏图像
        image_size = read_valid_image_size(img_path)
        if image_size is None:
            skipped_bad_image += 1
            continue
        img_w, img_h = image_size

        # 解析标注
        yolo_lines = parse_exdark_anno(anno_path, img_w, img_h)
        if not yolo_lines:
            skipped_no_bbox += 1
            continue

        # 导出图像；PNG 会按需重编码以清理异常元数据
        png_was_normalized = export_image_file(img_path, dst_img)
        if png_was_normalized:
            normalized_png += 1

        # 写入 YOLO 标注
        dst_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        # 记录绝对路径（train.txt 使用绝对路径，便于从任意目录启动训练）
        abs_img_path = str(dst_img.resolve())
        if split == "train":
            train_paths.append(abs_img_path)
        else:
            val_paths.append(abs_img_path)

        processed += 1

    print(
        f"      处理完成: {processed} 张，"
        f"跳过（无标注）: {skipped_no_anno}，"
        f"跳过（损坏图像）: {skipped_bad_image}，"
        f"跳过（空 bbox）: {skipped_no_bbox}，"
        f"PNG 重编码: {normalized_png}。"
    )

    if processed == 0:
        print("[错误] 没有任何图像被成功处理，请检查标注目录和格式。")
        sys.exit(1)

    # ── 写入索引文件 ──────────────────────────
    print("[3/4] 写入 train.txt / val.txt 索引文件...")

    train_txt = out_root / "train.txt"
    val_txt = out_root / "val.txt"
    train_txt.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_paths) + "\n", encoding="utf-8")

    print(f"      训练集: {len(train_paths)} 张 → {train_txt}")
    print(f"      验证集: {len(val_paths)} 张 → {val_txt}")

    # ── 更新 configs/train.yaml ───────────────
    print("[4/4] 更新训练配置文件...")
    update_train_config(config_path, train_txt, val_txt)

    print()
    print("=" * 60)
    print("预处理完成！可通过以下命令开始训练：")
    print()
    print("    cd src")
    print("    python train.py --config ../configs/train.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
