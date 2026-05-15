from __future__ import annotations

# --------------------------------------------------------
# eval.py — ICFIE-YOLO 量化评价脚本
#
# 功能:
#   单图模式 (MODE="single"): 推理单张图像, 可视化检测结果,
#     若提供对应标签则计算 TP/FP/FN 和单图 AP.
#   批量模式 (MODE="batch"): 遍历 val.txt, 计算完整验证集的
#     mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1.
#   可选生成中文 Markdown 评价报告.
#
# 运行方式 (从项目根目录):
#   python src/eval.py
#
# 修改顶部宏参数以切换模式/数据集/模型路径, 不修改函数内部逻辑.
#
# 权重选择规则:
#   ENABLE_FIE=True  时  CHECKPOINT_PATH 应指向 stage_c_epoch_N.pt
#   ENABLE_FIE=False 时  CHECKPOINT_PATH 应指向 stage_a_epoch_N.pt
# 这是因为阶段 C 的 detect head 已在 FIE 输出分布上联合微调，不能直接拿来评估纯 YOLO 路径。
# --------------------------------------------------------

from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 仓库根目录 (github-ai2/)
YOLOV7_ROOT = PROJECT_ROOT / "yolov7"   # YOLOv7 代码根目录 (github-ai2/yolov7/)
if str(YOLOV7_ROOT) not in sys.path:
    sys.path.append(str(YOLOV7_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ================================================================
# 宏定义区: 所有可配置参数集中在此处, 不允许隐藏在函数默认值中
# ================================================================

# ─────────────────────────────────────────────
# 模型参数
# ─────────────────────────────────────────────

# 待评价的 checkpoint 路径（建议写相对项目根目录的路径）
# ENABLE_FIE=True  -> 应指向 stage_c_epoch_N.pt (含 FIE 联合微调)
# ENABLE_FIE=False -> 应指向 stage_a_epoch_N.pt (纯 YOLO 基线)
CHECKPOINT_PATH = "runs/icfie_yolo-5-15-a1-b1-c5/stage_c_epoch_5.pt"

# YOLOv7 原始权重路径 (用于构建模型结构, 不影响评价权重)
YOLOV7_WEIGHTS_PATH = "yolov7/yolov7.pt"

# YOLOv7 网络结构配置文件
YOLOV7_CFG_PATH = "yolov7/cfg/training/yolov7.yaml"

# 是否启用 MSICN 光照矫正模块
ENABLE_MSICN = True

# 是否启用 FIE 特征交互增强模块
# 注意: ENABLE_FIE=False 时 CHECKPOINT_PATH 必须指向 stage_a checkpoint
ENABLE_FIE = True

# 推理设备 ("cuda:0" / "cpu")
DEVICE = "cuda:0"

# 推理图像尺寸 (必须与训练时一致)
IMAGE_SIZE = 416

# ─────────────────────────────────────────────
# 推理参数
# ─────────────────────────────────────────────

# NMS 置信度阈值 (低于此值的预测框在 NMS 前被滤除)
CONF_THRESHOLD = 0.25

# NMS IoU 阈值 (用于抑制重叠框)
NMS_IOU_THRESHOLD = 0.45

# 是否使用半精度 (FP16) 推理; GPU 支持时有效, CPU 自动回退 FP32
USE_HALF_PRECISION = False

# ─────────────────────────────────────────────
# 数据集参数
# ─────────────────────────────────────────────

# 评价模式: "batch"=批量评价, "single"=单图评价
MODE = "batch"

# 批量模式: val.txt 文件路径
# 文件内既可写绝对路径，也可写相对项目根目录的路径；推荐统一维护为项目根目录相对路径。
VAL_TXT = "data/Exdark/val.txt"

# 单图模式: 待评价图像路径
SINGLE_IMAGE_PATH = "test_sample/mc.jpg"

# 单图模式: 对应标签路径 (None=自动推导; 推导失败则只做可视化, 不计算指标)
# 自动推导规则: images/val/ -> labels/val/, 扩展名 -> .txt
SINGLE_LABEL_PATH = None

# 类别数量
NUM_CLASSES = 12

# 类别名称列表 (顺序必须与训练标签 class_id 严格一致)
CLASS_NAMES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike",
    "People", "Table",
]

# 批量模式 DataLoader 并行进程数
NUM_WORKERS = 4

# 批量模式每批图像数量 (增大可提升吞吐量, 需要足够显存)
BATCH_SIZE = 8

# ─────────────────────────────────────────────
# 指标参数
# ─────────────────────────────────────────────

# mAP@0.5 使用的 IoU 阈值 (主要指标)
EVAL_IOU_50 = 0.5

# mAP@0.5:0.95 使用的 IoU 阈值范围 (COCO 风格, 共 10 个阈值)
EVAL_IOU_RANGE = [round(x * 0.05, 2) for x in range(10, 20)]  # [0.5, 0.55, ..., 0.95]

# ─────────────────────────────────────────────
# 报告参数
# ─────────────────────────────────────────────

# 是否生成 Markdown 报告
GENERATE_REPORT = True

# 报告输出目录 (相对工作目录 github-ai2/)
REPORT_DIR = "report/eval5-15-a1-b1-c5"

# 报告文件名
REPORT_FILENAME = "eval_report.md"

# 是否将带预测框的图像保存到报告目录 (batch 模式文件较多, 默认关闭)
# 单图模式下固定保存, 不受此宏控制
SAVE_PRED_IMAGES = False

# 单图模式下可视化图的文件名 (保存在 REPORT_DIR/ 下)
SINGLE_PRED_IMAGE_FILENAME = "pred_visualization.jpg"

# ================================================================
# 以下为实现代码, 通常无需修改
# ================================================================

from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from fie import FIEBlockConfig, MultiScaleFIEConfig
# ===== FIE 配置修正：强制输出通道对齐检测头 =====
# 说明:
#   论文原始 FIE 输出是 3C 通道。
#   评价脚本这里固定 project_after_fusion=True，是为了与当前训练产物中的检测头输入通道保持一致。
FIE_CONFIG = MultiScaleFIEConfig(
    project_after_fusion=True,
    projection_channels=(256, 512, 1024),
)
from icfie_yolo import ICFIEYOLO, ICFIEYOLOConfig
from yolo_wrapper import YOLOv7WrapperConfig, build_yolov7_components
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.metrics import ap_per_class


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def configure_matplotlib_cjk_font() -> None:
    """尝试配置 CJK 字体, 用于报告中的 Matplotlib 图像标题."""
    preferred = (
        "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Serif CJK SC",
        "Source Han Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei",
        "SimHei", "PingFang SC", "Droid Sans Fallback",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = None
    for name in preferred:
        selected = next((n for n in available if name in n), None)
        if selected:
            break
    if selected:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False


def resolve_path(path_str: str) -> Path:
    """将相对路径解析为绝对路径 (相对工作目录或项目根目录均可)."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    # 优先相对当前工作目录, 再尝试项目根目录
    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return cwd_path
    return PROJECT_ROOT / p


def derive_label_path(image_path: Path) -> Path | None:
    """根据图像路径推导 YOLO 格式标签路径.

    规则: .../images/... -> .../labels/..., 扩展名替换为 .txt
    例如: data/Exdark/images/val/Dog_xxx.jpg
       -> data/Exdark/labels/val/Dog_xxx.txt
    """
    parts = image_path.parts
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        new_parts = parts[:idx] + ("labels",) + parts[idx + 1:]
        label_path = Path(*new_parts).with_suffix(".txt")
        if label_path.exists():
            return label_path
    return None


def load_image_bgr(image_path: Path) -> np.ndarray:
    """加载 BGR 图像, 失败时抛出有意义的错误."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return img


def preprocess_image(img_bgr: np.ndarray, img_size: int, stride: int) -> tuple[Tensor, tuple, tuple]:
    """letterbox 预处理, 返回 (tensor, letterbox_ratio, letterbox_pad).

    tensor shape: (1, 3, img_size, img_size), float32, 值域 [0,1]
    """
    padded, ratio, pad = letterbox(img_bgr, new_shape=(img_size, img_size),
                                   stride=stride, auto=False)
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_np = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
    return tensor, ratio, pad


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """解析 YOLO 格式标签文件, 返回 shape (N, 5) 的数组 [class_id, x1, y1, x2, y2].

    标签文件每行: class_id  cx  cy  nw  nh (归一化)
    输出坐标为像素绝对坐标 (xyxy 格式).
    """
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - nw / 2.0) * img_w
        y1 = (cy - nh / 2.0) * img_h
        x2 = (cx + nw / 2.0) * img_w
        y2 = (cy + nh / 2.0) * img_h
        boxes.append([cls_id, x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def box_iou_single(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个框的 IoU. box 格式: [x1, y1, x2, y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_tp_for_image(
    pred_boxes: np.ndarray,   # (N, 4) xyxy 原图坐标
    pred_cls: np.ndarray,     # (N,)
    pred_conf: np.ndarray,    # (N,)
    gt_boxes: np.ndarray,     # (M, 4) xyxy 原图坐标
    gt_cls: np.ndarray,       # (M,)
    iou_thresholds: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算单张图像的 TP 矩阵, 用于 ap_per_class.

    返回:
        tp        shape (N, len(iou_thresholds)), 每列对应一个 IoU 阈值
        conf      shape (N,)
        pred_cls_ shape (N,)
    """
    n_pred = len(pred_boxes)
    n_iou = len(iou_thresholds)
    tp = np.zeros((n_pred, n_iou), dtype=np.float32)

    if n_pred == 0 or len(gt_boxes) == 0:
        return tp, pred_conf, pred_cls

    # 按置信度降序处理
    sort_idx = np.argsort(-pred_conf)
    pred_boxes = pred_boxes[sort_idx]
    pred_cls_sorted = pred_cls[sort_idx]
    pred_conf_sorted = pred_conf[sort_idx]

    for iou_idx, iou_thr in enumerate(iou_thresholds):
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        for pred_i in range(n_pred):
            p_cls = pred_cls_sorted[pred_i]
            best_iou = 0.0
            best_gt_j = -1
            for gt_j in range(len(gt_boxes)):
                if gt_matched[gt_j]:
                    continue
                if gt_cls[gt_j] != p_cls:
                    continue
                iou = box_iou_single(pred_boxes[pred_i], gt_boxes[gt_j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_j = gt_j
            if best_iou >= iou_thr and best_gt_j >= 0:
                tp[pred_i, iou_idx] = 1.0
                gt_matched[best_gt_j] = True

    # 还原排列顺序 (ap_per_class 自行按 conf 排序, 但保持与 conf/pred_cls 对齐)
    restore_idx = np.argsort(sort_idx)
    tp = tp[restore_idx]
    return tp, pred_conf, pred_cls


# ─────────────────────────────────────────────
# 模型构建
# ─────────────────────────────────────────────

def build_eval_model(device: torch.device) -> tuple[ICFIEYOLO, int]:
    """构建 ICFIE-YOLO 评价模型并加载 CHECKPOINT_PATH.

    返回 (model, stride).
    """
    ckpt_path = resolve_path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    yolo_cfg = YOLOv7WrapperConfig(
        cfg_path=resolve_path(YOLOV7_CFG_PATH),
        weights_path=resolve_path(YOLOV7_WEIGHTS_PATH),
        num_classes=NUM_CLASSES,
    )

    # 复用与 PyTorch 版本兼容的 torch.load patch (见 train.py)
    original_torch_load = torch.load

    def compatible_torch_load(path, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(path, *args, **kwargs)

    torch.load = compatible_torch_load
    try:
        _, backbone_neck, detect_head, stride, _ = build_yolov7_components(yolo_cfg, device=device)
    finally:
        torch.load = original_torch_load

    model_config = ICFIEYOLOConfig(
        enable_msicn=ENABLE_MSICN,
        enable_fie=ENABLE_FIE,
        fie=FIE_CONFIG,
    )
    model = ICFIEYOLO(
        backbone_neck=backbone_neck,
        detect_head=detect_head,
        config=model_config,
    )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"checkpoint 缺少 'model' 字段: {ckpt_path}")
    model.load_state_dict(checkpoint["model"], strict=True)

    use_half = USE_HALF_PRECISION and device.type != "cpu"
    if use_half:
        model = model.half()

    return model.to(device).eval(), stride


# ─────────────────────────────────────────────
# 单张图像推理
# ─────────────────────────────────────────────

def infer_single_image(
    model: ICFIEYOLO,
    img_bgr: np.ndarray,
    stride: int,
    device: torch.device,
) -> tuple[Tensor, float]:
    """对单张 BGR 图像执行完整推理流水线.

    返回:
        detections: (N, 6) Tensor [x1,y1,x2,y2,conf,cls], 原图坐标, CPU
        latency_ms: 模型前向 + NMS 耗时 (ms)
    """
    orig_h, orig_w = img_bgr.shape[:2]
    tensor, _, _ = preprocess_image(img_bgr, IMAGE_SIZE, stride)

    use_half = USE_HALF_PRECISION and device.type != "cpu"
    if use_half:
        tensor = tensor.half()
    tensor = tensor.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        predictions = model(tensor)

    # 解析检测头输出。
    # eval 模式下真实 YOLOv7 IDetect 返回 (decoded_predictions, raw_maps)，
    # 其中 decoded_predictions 才是执行 NMS 所需的候选框集合。
    if isinstance(predictions, (tuple, list)) and len(predictions) == 2 and isinstance(predictions[1], (list, tuple)):
        decoded = predictions[0]
    elif isinstance(predictions, Tensor):
        decoded = predictions
    else:
        decoded = predictions[0] if isinstance(predictions, (tuple, list)) else predictions

    nms_result = non_max_suppression(decoded, CONF_THRESHOLD, NMS_IOU_THRESHOLD)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    detections = nms_result[0]  # 单张图
    if detections is not None and detections.numel() > 0:
        detections = detections.clone().float().cpu()
        # 坐标回投: letterbox 尺寸 -> 原图尺寸。
        # 这一步与论文无关，但属于目标检测评价必须遵守的 YOLOv7 官方后处理链路。
        scale_coords((IMAGE_SIZE, IMAGE_SIZE), detections[:, :4], (orig_h, orig_w))
        # clamp 到原图范围 (数值安全约束)
        detections[:, 0].clamp_(0, orig_w)
        detections[:, 1].clamp_(0, orig_h)
        detections[:, 2].clamp_(0, orig_w)
        detections[:, 3].clamp_(0, orig_h)
    else:
        detections = torch.zeros((0, 6), dtype=torch.float32)

    return detections, latency_ms


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────

def draw_boxes_on_image(
    img_bgr: np.ndarray,
    detections: Tensor,
    gt_boxes: np.ndarray | None = None,
    gt_cls: np.ndarray | None = None,
) -> np.ndarray:
    """在图像上绘制预测框 (绿色) 和 GT 框 (红色).

    返回 RGB 格式 numpy 数组 (用于 matplotlib 或 cv2.imwrite).
    """
    img = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)

    # 绘制 GT 框 (红色虚线效果用实线代替)
    if gt_boxes is not None and gt_cls is not None and len(gt_boxes) > 0:
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cls_id = int(gt_cls[i])
            label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (220, 50, 50), 2)
            cv2.putText(img, f"GT:{label}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 50, 50), 1, cv2.LINE_AA)

    # 绘制预测框 (绿色)
    if detections.numel() > 0:
        for det in detections.tolist():
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
            label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 200, 80), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 200, 80), 1, cv2.LINE_AA)

    return img


def save_visualization(
    img_rgb: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """保存可视化图像 (RGB ndarray -> JPEG)."""
    # 转回 BGR 用 cv2 保存
    img_bgr_save = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr_save)


# ─────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────

def _fmt(value: float, decimals: int = 4) -> str:
    """格式化浮点数."""
    return f"{value:.{decimals}f}"


def generate_report(
    report_path: Path,
    summary: dict,
    per_class_ap: dict[str, float],
    latency: dict,
    eval_time: str,
    image_count: int,
    mode: str,
    single_image_info: dict | None = None,
    pred_image_paths: list[str] | None = None,
) -> None:
    """写入中文 Markdown 评价报告 (英文标点)."""
    lines = []

    lines.append("# ICFIE-YOLO 目标检测评价报告\n")

    # 评价概述
    lines.append("## 评价概述\n")
    lines.append("| 项目 | 值 |")
    lines.append("|------|----|")
    lines.append(f"| 评价时间 | {eval_time} |")
    lines.append(f"| Checkpoint | {CHECKPOINT_PATH} |")
    lines.append(f"| ENABLE_MSICN | {ENABLE_MSICN} |")
    lines.append(f"| ENABLE_FIE | {ENABLE_FIE} |")
    lines.append(f"| 数据集 | {VAL_TXT if mode == 'batch' else SINGLE_IMAGE_PATH} |")
    lines.append(f"| 图像数量 | {image_count} |")
    lines.append(f"| 图像尺寸 | {IMAGE_SIZE} |")
    lines.append(f"| CONF_THRESHOLD | {CONF_THRESHOLD} |")
    lines.append(f"| NMS_IOU_THRESHOLD | {NMS_IOU_THRESHOLD} |")
    lines.append(f"| 设备 | {DEVICE} |")
    lines.append("")

    # 总体检测指标
    lines.append("## 总体检测指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|----|")
    lines.append(f"| mAP@0.5 | {_fmt(summary.get('mAP@0.5', 0.0))} |")
    lines.append(f"| mAP@0.5:0.95 | {_fmt(summary.get('mAP@0.5:0.95', 0.0))} |")
    lines.append(f"| Precision | {_fmt(summary.get('Precision', 0.0))} |")
    lines.append(f"| Recall | {_fmt(summary.get('Recall', 0.0))} |")
    lines.append(f"| F1 | {_fmt(summary.get('F1', 0.0))} |")
    lines.append("")

    # 各类别 AP
    if per_class_ap:
        lines.append("## 各类别 AP@0.5 详细结果\n")
        lines.append("(按 AP@0.5 降序排列)\n")
        lines.append("| 排名 | 类别 | AP@0.5 |")
        lines.append("|------|------|--------|")
        sorted_classes = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
        for rank, (cls_name, ap_val) in enumerate(sorted_classes, 1):
            lines.append(f"| {rank} | {cls_name} | {_fmt(ap_val)} |")
        lines.append("")

    # 统计摘要
    lines.append("## 统计摘要\n")
    lines.append("| 统计量 | 值 |")
    lines.append("|--------|----|")
    lines.append(f"| AP@0.5 均值 | {_fmt(summary.get('mAP@0.5', 0.0))} |")
    lines.append(f"| AP@0.5 中位数 | {_fmt(summary.get('AP_median@0.5', 0.0))} |")
    lines.append(f"| AP@0.5 标准差 | {_fmt(summary.get('AP_std@0.5', 0.0))} |")
    lines.append(f"| 推理耗时均值 | {latency.get('mean_ms', 0.0):.1f} ms |")
    lines.append(f"| 推理耗时中位数 | {latency.get('median_ms', 0.0):.1f} ms |")
    lines.append("")

    # 单图模式额外内容
    if mode == "single" and single_image_info:
        lines.append("## 单图评价结果\n")
        lines.append("| 项目 | 值 |")
        lines.append("|------|----|")
        for k, v in single_image_info.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # 批量模式可视化图路径列表
    if mode == "batch" and pred_image_paths:
        lines.append("## 预测图像列表\n")
        for p in pred_image_paths:
            lines.append(f"- {p}")
        lines.append("")

    # 附注
    lines.append("## 附注\n")
    lines.append("评价使用 YOLOv7 标准推理链路: letterbox -> 模型前向 -> NMS -> scale_coords.")
    lines.append("AP 计算复用 yolov7/utils/metrics.py 中的 ap_per_class 函数.")
    lines.append(f"EVAL_IOU_RANGE: {EVAL_IOU_RANGE}")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[报告] 已保存: {report_path}")


# ─────────────────────────────────────────────
# 单图评价模式
# ─────────────────────────────────────────────

def run_single_mode(model: ICFIEYOLO, stride: int, device: torch.device) -> None:
    """单图模式: 推理 + 可视化 + 可选指标计算."""
    image_path = resolve_path(SINGLE_IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")

    img_bgr = load_image_bgr(image_path)
    orig_h, orig_w = img_bgr.shape[:2]

    print(f"[单图] 图像: {image_path}  尺寸: {orig_w}x{orig_h}")

    # 确定标签路径
    if SINGLE_LABEL_PATH is not None:
        label_path: Path | None = resolve_path(SINGLE_LABEL_PATH)
        if not label_path.exists():
            print(f"[警告] 指定标签文件不存在: {label_path}, 跳过指标计算.")
            label_path = None
    else:
        label_path = derive_label_path(image_path)
        if label_path is None:
            print("[信息] 未找到对应标签文件, 只执行推理可视化, 不计算指标.")

    # 推理
    detections, latency_ms = infer_single_image(model, img_bgr, stride, device)
    print(f"[单图] 检出框数量: {len(detections)}  推理耗时: {latency_ms:.1f} ms")

    # 加载 GT 标签
    gt_labels = None
    if label_path is not None:
        gt_labels = parse_yolo_labels(label_path, orig_w, orig_h)
        print(f"[单图] GT 框数量: {len(gt_labels)}")

    # 单图指标计算
    single_info: dict = {}
    summary: dict = {}
    per_class_ap: dict = {}
    tp_count = fp_count = fn_count = 0

    single_info["图像路径"] = str(image_path)
    single_info["原始尺寸"] = f"{orig_w}x{orig_h}"
    single_info["预测框数量"] = len(detections)

    if gt_labels is not None and len(gt_labels) > 0:
        pred_boxes_np = detections[:, :4].numpy() if detections.numel() > 0 else np.zeros((0, 4))
        pred_cls_np = detections[:, 5].numpy().astype(int) if detections.numel() > 0 else np.zeros(0, dtype=int)
        pred_conf_np = detections[:, 4].numpy() if detections.numel() > 0 else np.zeros(0)

        gt_boxes_np = gt_labels[:, 1:5]
        gt_cls_np = gt_labels[:, 0].astype(int)

        tp_arr, conf_arr, pcls_arr = compute_tp_for_image(
            pred_boxes_np, pred_cls_np, pred_conf_np,
            gt_boxes_np, gt_cls_np,
            EVAL_IOU_RANGE,
        )

        single_info["GT 框数量"] = len(gt_labels)

        if len(conf_arr) > 0:
            p_arr, r_arr, ap_arr, f1_arr, unique_cls = ap_per_class(
                tp_arr, conf_arr, pcls_arr, gt_cls_np,
                v5_metric=False, plot=False,
            )
            ap_50 = ap_arr[:, 0]
            summary["mAP@0.5"] = float(ap_50.mean())
            summary["mAP@0.5:0.95"] = float(ap_arr.mean())
            summary["AP_median@0.5"] = float(np.median(ap_50))
            summary["AP_std@0.5"] = float(np.std(ap_50))
            summary["Precision"] = float(p_arr.mean())
            summary["Recall"] = float(r_arr.mean())
            summary["F1"] = float(f1_arr.mean())

            for cls_id, ap_val in zip(unique_cls, ap_50):
                cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                per_class_ap[cls_name] = float(ap_val)

            # TP/FP/FN 统计 (IoU=0.5)
            tp_count = int(tp_arr[:, 0].sum())
            fp_count = int((1 - tp_arr[:, 0]).sum())
            fn_count = max(0, len(gt_labels) - tp_count)

        single_info["TP"] = tp_count
        single_info["FP"] = fp_count
        single_info["FN"] = fn_count

        print(f"[单图] mAP@0.5={summary.get('mAP@0.5', 0):.4f}  "
              f"P={summary.get('Precision', 0):.4f}  "
              f"R={summary.get('Recall', 0):.4f}  "
              f"TP={tp_count}  FP={fp_count}  FN={fn_count}")
    else:
        single_info["GT 框数量"] = "N/A (无标签)"

    # 可视化并保存
    gt_boxes_for_draw = gt_labels[:, 1:5] if gt_labels is not None and len(gt_labels) > 0 else None
    gt_cls_for_draw = gt_labels[:, 0].astype(int) if gt_labels is not None and len(gt_labels) > 0 else None
    vis_img = draw_boxes_on_image(img_bgr, detections, gt_boxes_for_draw, gt_cls_for_draw)

    report_dir = resolve_path(REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    vis_save_path = report_dir / SINGLE_PRED_IMAGE_FILENAME
    save_visualization(vis_img, vis_save_path)
    single_info["可视化图"] = SINGLE_PRED_IMAGE_FILENAME
    print(f"[单图] 可视化图已保存: {vis_save_path}")

    if GENERATE_REPORT:
        report_path = report_dir / REPORT_FILENAME
        latency_stats = {"mean_ms": latency_ms, "median_ms": latency_ms}
        generate_report(
            report_path=report_path,
            summary=summary,
            per_class_ap=per_class_ap,
            latency=latency_stats,
            eval_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_count=1,
            mode="single",
            single_image_info=single_info,
        )


# ─────────────────────────────────────────────
# 批量评价模式
# ─────────────────────────────────────────────

def read_val_txt(val_txt_path: Path) -> list[Path]:
    """读取 val.txt 并返回有效图像路径列表."""
    lines = val_txt_path.read_text(encoding="utf-8").strip().splitlines()
    paths = [Path(line.strip()) for line in lines if line.strip()]
    valid = [p for p in paths if p.exists()]
    missing = len(paths) - len(valid)
    if missing > 0:
        print(f"[警告] val.txt 中 {missing} 条路径不存在, 已跳过.")
    if len(valid) == 0:
        raise ValueError(f"val.txt 中没有有效图像路径: {val_txt_path}")
    return valid


def run_batch_mode(model: ICFIEYOLO, stride: int, device: torch.device) -> None:
    """批量模式: 遍历 val.txt, 计算完整验证集指标."""
    val_txt_path = resolve_path(VAL_TXT)
    if not val_txt_path.exists():
        raise FileNotFoundError(f"val.txt 不存在: {val_txt_path}")

    image_paths = read_val_txt(val_txt_path)
    total = len(image_paths)
    print(f"[批量] 共 {total} 张图像, 开始评价...")

    # 全局累积 TP/conf/pred_cls/target_cls
    all_tp: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_target_cls: list[np.ndarray] = []
    all_latencies: list[float] = []
    pred_image_save_paths: list[str] = []

    report_dir = resolve_path(REPORT_DIR)
    pred_img_dir = report_dir / "pred_images"
    if SAVE_PRED_IMAGES:
        pred_img_dir.mkdir(parents=True, exist_ok=True)

    # 检查各类是否有 GT
    class_gt_count = {name: 0 for name in CLASS_NAMES}

    for img_idx, img_path in enumerate(image_paths):
        if (img_idx + 1) % 100 == 0 or img_idx == 0:
            print(f"[批量] {img_idx + 1}/{total} ...")

        try:
            img_bgr = load_image_bgr(img_path)
        except FileNotFoundError as e:
            print(f"[警告] {e}, 跳过.")
            continue

        orig_h, orig_w = img_bgr.shape[:2]

        # 推理
        detections, latency_ms = infer_single_image(model, img_bgr, stride, device)
        all_latencies.append(latency_ms)

        # 加载 GT
        label_path = derive_label_path(img_path)
        if label_path is None:
            # 尝试标准替换
            label_candidate = Path(str(img_path).replace("/images/", "/labels/"))
            label_candidate = label_candidate.with_suffix(".txt")
            if label_candidate.exists():
                label_path = label_candidate

        if label_path is None or not label_path.exists():
            # 无标签则跳过此图的指标贡献, 但保留推理耗时统计
            continue

        gt_labels = parse_yolo_labels(label_path, orig_w, orig_h)
        if len(gt_labels) == 0:
            continue

        gt_boxes_np = gt_labels[:, 1:5]
        gt_cls_np = gt_labels[:, 0].astype(int)

        # 更新各类 GT 计数
        for cls_id in gt_cls_np:
            if 0 <= cls_id < len(CLASS_NAMES):
                class_gt_count[CLASS_NAMES[cls_id]] += 1

        # 构造预测数组
        if detections.numel() > 0:
            pred_boxes_np = detections[:, :4].numpy()
            pred_cls_np = detections[:, 5].numpy().astype(int)
            pred_conf_np = detections[:, 4].numpy()
        else:
            pred_boxes_np = np.zeros((0, 4), dtype=np.float32)
            pred_cls_np = np.zeros(0, dtype=int)
            pred_conf_np = np.zeros(0, dtype=np.float32)

        tp_img, conf_img, pcls_img = compute_tp_for_image(
            pred_boxes_np, pred_cls_np, pred_conf_np,
            gt_boxes_np, gt_cls_np,
            EVAL_IOU_RANGE,
        )

        if len(conf_img) > 0:
            all_tp.append(tp_img)
            all_conf.append(conf_img)
            all_pred_cls.append(pcls_img)
        all_target_cls.append(gt_cls_np)

        # 可视化保存
        if SAVE_PRED_IMAGES:
            vis_img = draw_boxes_on_image(img_bgr, detections, gt_boxes_np, gt_cls_np)
            save_name = img_path.name
            save_path = pred_img_dir / save_name
            save_visualization(vis_img, save_path)
            pred_image_save_paths.append(f"pred_images/{save_name}")

    # 打印 GT 为零的类别警告
    for cls_name, cnt in class_gt_count.items():
        if cnt == 0:
            print(f"[警告] 类别 '{cls_name}' 在验证集中没有 GT 框, AP 记为 0.0.")

    # 计算全局指标
    summary: dict = {}
    per_class_ap: dict[str, float] = {}

    if not all_conf:
        print("[警告] 没有任何有效预测参与指标计算, 检查 CONF_THRESHOLD 或数据集路径.")
        summary = {k: 0.0 for k in ["mAP@0.5", "mAP@0.5:0.95", "AP_median@0.5",
                                      "AP_std@0.5", "Precision", "Recall", "F1"]}
    else:
        tp_all = np.concatenate(all_tp, axis=0)      # (N_total, n_iou)
        conf_all = np.concatenate(all_conf, axis=0)  # (N_total,)
        pcls_all = np.concatenate(all_pred_cls, axis=0)
        tcls_all = np.concatenate(all_target_cls, axis=0)

        # 数值安全断言
        assert tp_all.shape[1] == len(EVAL_IOU_RANGE), (
            f"tp 列数 {tp_all.shape[1]} != len(EVAL_IOU_RANGE) {len(EVAL_IOU_RANGE)}"
        )

        p_arr, r_arr, ap_arr, f1_arr, unique_cls = ap_per_class(
            tp_all, conf_all, pcls_all, tcls_all,
            v5_metric=False, plot=False,
        )

        ap_50 = ap_arr[:, 0]
        summary["mAP@0.5"] = float(ap_50.mean())
        summary["mAP@0.5:0.95"] = float(ap_arr.mean())
        summary["AP_median@0.5"] = float(np.median(ap_50))
        summary["AP_std@0.5"] = float(np.std(ap_50))
        summary["Precision"] = float(p_arr.mean())
        summary["Recall"] = float(r_arr.mean())
        summary["F1"] = float(f1_arr.mean())

        for cls_id, ap_val in zip(unique_cls, ap_50):
            cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
            per_class_ap[cls_name] = float(ap_val)

    latency_stats = {
        "mean_ms": float(np.mean(all_latencies)) if all_latencies else 0.0,
        "median_ms": float(np.median(all_latencies)) if all_latencies else 0.0,
    }

    # 终端汇总输出
    print("\n" + "=" * 60)
    print(f"  mAP@0.5       = {summary.get('mAP@0.5', 0):.4f}")
    print(f"  mAP@0.5:0.95  = {summary.get('mAP@0.5:0.95', 0):.4f}")
    print(f"  Precision     = {summary.get('Precision', 0):.4f}")
    print(f"  Recall        = {summary.get('Recall', 0):.4f}")
    print(f"  F1            = {summary.get('F1', 0):.4f}")
    print(f"  AP 中位数     = {summary.get('AP_median@0.5', 0):.4f}")
    print(f"  推理耗时均值  = {latency_stats['mean_ms']:.1f} ms")
    print(f"  推理耗时中位数= {latency_stats['median_ms']:.1f} ms")
    print("=" * 60)

    if per_class_ap:
        print("\n各类别 AP@0.5:")
        for cls_name, ap_val in sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls_name:<12} {ap_val:.4f}")

    if GENERATE_REPORT:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / REPORT_FILENAME
        generate_report(
            report_path=report_path,
            summary=summary,
            per_class_ap=per_class_ap,
            latency=latency_stats,
            eval_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_count=total,
            mode="batch",
            pred_image_paths=pred_image_save_paths if SAVE_PRED_IMAGES else None,
        )


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main() -> None:
    configure_matplotlib_cjk_font()

    # 数值安全前置检查
    assert len(EVAL_IOU_RANGE) >= 1, "EVAL_IOU_RANGE 不能为空"
    assert all(0.0 <= v <= 1.0 for v in EVAL_IOU_RANGE), "EVAL_IOU_RANGE 中存在超出 [0,1] 的值"
    assert MODE in ("single", "batch"), f"MODE 必须为 'single' 或 'batch', 当前值: {MODE}"

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print("[警告] 指定了 CUDA 但当前不可用, 自动回退到 CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)

    print(f"[配置] MODE={MODE}  DEVICE={device}  ENABLE_MSICN={ENABLE_MSICN}  ENABLE_FIE={ENABLE_FIE}")
    print(f"[配置] CHECKPOINT={CHECKPOINT_PATH}")

    model, stride = build_eval_model(device)
    print(f"[模型] 加载完成  stride={stride}")

    if MODE == "single":
        run_single_mode(model, stride, device)
    else:
        run_batch_mode(model, stride, device)


if __name__ == "__main__":
    main()
