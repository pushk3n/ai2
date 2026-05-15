from __future__ import annotations

# --------------------------------------------------------
# test.py - 单图完整流程测试脚本
#
# 目的:
#   直接指定 test_sample/ 中的一张图像
#   跑通四层流水线（MSICN → YOLO Backbone+Neck → FIE → Detect Head）的完整前向流程
#   并将可视化结果保存到 results/ 目录下
#
# 四层模块对应:
#   1. msicn.py        → MSICN            光照矫正
#   2. yolo_wrapper.py → YOLO Backbone+Neck  特征提取（真实 YOLOv7 权重）
#   3. fie.py          → MultiScaleFIE    特征增强
#   4. detect.py       → Detect Head      检测输出（YOLOv7DetectHeadAdapter）
#
# 运行方式:
#   python src/test.py
#
# 配置方式:
#   1. 先在项目根目录下创建 test_sample/，放入待测图片。
#   2. 修改 TARGET_IMAGE_NAME 选择图片。
#   3. 修改 YOLO_WEIGHTS_PATH 指向 yolov7/yolov7.pt。
#   4. 按 ENABLE_FIE 选择对应 checkpoint:
#      ENABLE_FIE=True  -> ICFIE_CHECKPOINT_PATH 指向 stage_c_epoch_N.pt
#      ENABLE_FIE=False -> YOLO_ONLY_CHECKPOINT_PATH 指向 stage_a_epoch_N.pt
# --------------------------------------------------------

from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLOV7_ROOT = PROJECT_ROOT / "yolov7"
if str(YOLOV7_ROOT) not in sys.path:
	sys.path.append(str(YOLOV7_ROOT))

from fie import FIEBlockConfig, MultiScaleFIEConfig
from icfie_yolo import (
	ICFIEYOLO,
	ICFIEYOLOConfig,
)
from yolo_wrapper import YOLOv7WrapperConfig, build_yolov7_components
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


# ================================================================
# 宏定义区: 所有与测试图片 / 权重 / 推理行为相关的配置都集中放在这里
# ================================================================

TARGET_IMAGE_NAME = "mc.jpg"
IMG_SIZE = 416
DEVICE = "cpu"
ENABLE_MSICN = True   # False: 跳过光照矫正  纯 YOLO baseline
ENABLE_FIE = True     # False: 跳过特征增强  纯 YOLO baseline
ENABLE_NMS: bool = True   # False: 跳过 NMS  不显示检测框
CONF_THRES: float = 0.25
IOU_THRES: float = 0.45
YOLO_CFG_PATH = YOLOV7_ROOT / "cfg" / "training" / "yolov7.yaml"

YOLO_WEIGHTS_PATH: Path | None = YOLOV7_ROOT / "yolov7.pt"  	# YOLOv7 官方预训练权重  需要放到 yolov7/ 目录下

# 训练出来的权重路径  需要先训练好模型并改成你自己的 run_dir
# 注意: 不同 FIE 开关状态需要使用不同 checkpoint:
#   ENABLE_FIE=True  → 使用 Stage C checkpoint (backbone+FIE+detect 联合微调)
#   ENABLE_FIE=False → 使用 Stage A checkpoint (纯 YOLO baseline，backbone+detect 无 FIE 预训练)
# 若 FIE 禁用时仍加载 Stage C checkpoint，detect head 期望的是 FIE 输出特征分布，
# 收到原始 backbone 特征后置信度全部偏低，NMS 将滤除所有框。
ICFIE_CHECKPOINT_PATH: Path | None = PROJECT_ROOT / "runs" / "icfie_yolo-5-15-a1-b10-c10" / "stage_c_epoch_10.pt"

# 纯 YOLO 模式（ENABLE_FIE=False）专用 checkpoint
# 重新训练后指向 stage_a_epoch_<N>.pt（Stage A 训练完成后生成）
# 若设为 None 且 ENABLE_FIE=False，只保留 yolov7.pt 主干权重（检测头为随机初始化，12 类可能无法检出）
YOLO_ONLY_CHECKPOINT_PATH: Path | None = PROJECT_ROOT / "runs" / "icfie_yolo-5-15-a1-b10-c10" / "stage_a_epoch_1.pt"

YOLO_NUM_CLASSES: int | None = 12	# 这里需要与训练配置中的 dataset.num_classes 保持一致  否则会导致模型加载失败或推理错误

# ExDark 12 类别名  与 data-process.py/train.yaml 保持一致
# 注意: build_yolov7_components 加载 yolov7.pt 时会把 class_names 覆盖为 COCO 80 类名
# 必须在 build_model 里用下面的列表替换  否则 class_id=10 会被映射成 COCO 的 "fire hydrant"
EXDARK_CLASS_NAMES: list[str] = [
	"Bicycle", "Boat", "Bottle", "Bus", "Car",
	"Cat", "Chair", "Cup", "Dog", "Motorbike",
	"People", "Table",
]

INPUT_DIR = PROJECT_ROOT / "test_sample"
OUTPUT_DIR = PROJECT_ROOT / "results" / "single_image_pipeline"


def configure_matplotlib_cjk_font() -> None:
	# infer.py 已被移除，测试脚本直接内聚最小字体配置逻辑，避免再依赖已删除模块。
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
		print("[警告 Warning] 未找到可用中文字体 / no CJK font found for Matplotlib")
		return

	plt.rcParams["font.family"] = "sans-serif"
	plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
	plt.rcParams["axes.unicode_minus"] = False


def tensor_to_numpy_image(image_tensor: Tensor) -> np.ndarray:
	# 把模型中的图像张量还原为 matplotlib 可显示的 RGB numpy 图像。
	# 约束与 MSICN 输出一致：输入 shape 为 (1,3,H,W) 或 (3,H,W)，数值范围期望在 [0,1]。
	if image_tensor.ndim == 4:
		if image_tensor.size(0) != 1:
			raise ValueError(f"仅支持 batch=1 的图像可视化，实际得到 shape={tuple(image_tensor.shape)}")
		image_tensor = image_tensor.squeeze(0)
	if image_tensor.ndim != 3 or image_tensor.size(0) != 3:
		raise ValueError(f"期望图像张量 shape 为 (3,H,W)，实际得到 {tuple(image_tensor.shape)}")

	image = image_tensor.detach().float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
	return image


def build_output_stem(image_stem: str, enable_msicn: bool, enable_fie: bool) -> str:
	# 输出文件名显式带上模块开关状态，便于对比不同组合的结果图
	msicn_tag = "msicn_on" if enable_msicn else "msicn_off"
	fie_tag = "fie_on" if enable_fie else "fie_off"
	return f"{image_stem}_{msicn_tag}_{fie_tag}"


def feature_map_to_heatmap(feature: Tensor) -> np.ndarray:
	# 将特征图压成单通道热力图便于可视化
	# 使用通道绝对值均值  可以稳定反映该层的整体响应强度
	feature_2d = feature.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
	feature_2d = feature_2d - feature_2d.min()
	denominator = feature_2d.max()
	if denominator > 0:
		feature_2d = feature_2d / denominator
	return feature_2d


def prediction_to_heatmap(prediction: Tensor) -> np.ndarray:
	# 兼容两种检测输出:
	#   4D: Mock 检测头输出 (B, C, H, W)
	#   5D: YOLOv7 IDetect 原始输出 (B, A, H, W, no)
	# 统一压成 2D 热力图观察响应分布
	if prediction.ndim == 5:
		prediction_2d = prediction.detach().abs().mean(dim=(1, 4)).squeeze(0).cpu().numpy()
	elif prediction.ndim == 4:
		prediction_2d = prediction.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
	else:
		raise ValueError(f"不支持的检测输出维度: {tuple(prediction.shape)}")
	prediction_2d = prediction_2d - prediction_2d.min()
	denominator = prediction_2d.max()
	if denominator > 0:
		prediction_2d = prediction_2d / denominator
	return prediction_2d


def split_prediction_outputs(predictions: object) -> tuple[Tensor | None, list[Tensor]]:
	# 统一解析检测头输出
	# 真实 YOLOv7 eval: (decoded_predictions, raw_prediction_maps)
	# 训练态或简化头: raw_prediction_maps
	if isinstance(predictions, tuple) and len(predictions) == 2 and isinstance(predictions[1], (list, tuple)):
		decoded_predictions = predictions[0] if isinstance(predictions[0], Tensor) else None
		return decoded_predictions, list(predictions[1])
	if isinstance(predictions, (list, tuple)) and all(isinstance(item, Tensor) for item in predictions):
		return None, list(predictions)
	raise TypeError("检测头输出格式不符合预期")


def load_yolo_image_as_tensor(path: Path, img_size: int, stride: int) -> tuple[Tensor, np.ndarray]:
	"""
	加载 YOLO 模型输入图像为张量。

	这里沿用 YOLOv7 标准 letterbox 预处理，保证 test.py 的单图测试
	与 eval.py 和官方推理链路的输入分布一致。
	"""
	image_bgr = cv2.imread(str(path))
	if image_bgr is None:
		raise FileNotFoundError(f"无法读取测试图像: {path}")
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	letterboxed = letterbox(image_bgr, new_shape=img_size, stride=stride)[0]
	letterboxed_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
	image_np = np.ascontiguousarray(letterboxed_rgb.transpose(2, 0, 1))
	image_tensor = torch.from_numpy(image_np).float().unsqueeze(0) / 255.0
	return image_tensor, image_rgb


def decode_single_image_detections(
	decoded_predictions: Tensor,
	input_shape: tuple[int, int],
	original_shape: tuple[int, int],
	conf_thres: float,
	iou_thres: float,
) -> Tensor:
	# 对单张图像执行 YOLOv7 标准后处理  并把框从 letterbox 输入尺寸映射回原图尺寸
	detections = non_max_suppression(decoded_predictions, conf_thres, iou_thres)[0]
	if detections.numel() == 0:
		return detections.cpu()
	detections = detections.clone()
	scale_coords(input_shape, detections[:, :4], original_shape)
	return detections.cpu()


def draw_detections(axis: plt.Axes, image: np.ndarray, detections: Tensor, class_names: list[str]) -> None:
	# 在原图上叠加 NMS 后的检测框  这是最终可解释的检测结果
	axis.imshow(image)
	axis.axis("off")
	if detections.numel() == 0:
		axis.text(
			0.5,
			0.5,
			"未检出目标",
			color="white",
			fontsize=14,
			ha="center",
			va="center",
			transform=axis.transAxes,
			bbox={"facecolor": "#17212b", "edgecolor": "white", "alpha": 0.8, "pad": 6},
		)
		return
	color_map = plt.cm.get_cmap("tab10", max(len(class_names), 10))
	for detection in detections.tolist():
		x1, y1, x2, y2, confidence, class_id = detection
		class_index = int(class_id)
		color = color_map(class_index % color_map.N)
		axis.add_patch(
			Rectangle(
				(x1, y1),
				x2 - x1,
				y2 - y1,
				fill=False,
				linewidth=2,
				edgecolor=color,
			)
		)
		label = class_names[class_index] if 0 <= class_index < len(class_names) else f"cls {class_index}"
		axis.text(
			x1,
			max(y1 - 4, 4),
			f"{label} {confidence:.2f}",
			color="white",
			fontsize=9,
			va="bottom",
			bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
		)
	axis.set_title(f"NMS 检测结果 ({len(detections)} boxes)", color="white", fontsize=12)


def save_pipeline_visualization(
	original_image: np.ndarray,
	corrected_tensor: Tensor,
	features: list[Tensor],
	detections: Tensor,
	class_names: list[str],
	enable_msicn: bool,
	enable_fie: bool,
	enable_nms: bool,
	save_path: Path,
	filename: str,
) -> None:
	# 汇总保存完整流程可视化 (布局与 yolo_test.py 对齐):
	#   第一行: 原始输入 / MSICN矫正后(或已关闭) / NMS检测结果(或已关闭)
	#   第二行: P3/P4/P5 特征热力图 (FIE增强后 或 主干原始)
	corrected_np = tensor_to_numpy_image(corrected_tensor)

	fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=120)
	fig.patch.set_facecolor("#101820")

	for row in axes:
		for ax in row:
			ax.set_facecolor("#17212b")
			ax.axis("off")

	axes[0, 0].imshow(original_image)
	axes[0, 0].set_title("原始输入", color="white", fontsize=12)

	axes[0, 1].imshow(corrected_np)
	msicn_title = "MSICN 矫正后" if enable_msicn else "MSICN 已关闭 (原始输入)"
	axes[0, 1].set_title(msicn_title, color="white" if enable_msicn else "#aaaaaa", fontsize=12)

	if enable_nms:
		draw_detections(axes[0, 2], original_image, detections, class_names)
	else:
		axes[0, 2].imshow(original_image)
		axes[0, 2].set_title("NMS 已关闭", color="#aaaaaa", fontsize=12)

	scale_names = ("P3", "P4", "P5")
	feat_suffix = " FIE" if enable_fie else " Feat"
	for col, (scale_name, feature) in enumerate(zip(scale_names, features)):
		axes[1, col].imshow(feature_map_to_heatmap(feature), cmap="magma")
		axes[1, col].set_title(f"{scale_name}{feat_suffix}", color="white", fontsize=12)

	plt.tight_layout(pad=1.0)
	plt.savefig(save_path / f"{filename}_pipeline.png", bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.close(fig)


def build_model(device: torch.device) -> tuple[ICFIEYOLO, int, list[str]]:
	# 使用真实 YOLOv7 Backbone/Neck + Detect 头接入 ICFIEYOLO
	# 返回 (model, stride, class_names)  stride 用于 letterbox 预处理
	# 这里默认把 FIE 输出投影回检测头所需通道，便于直接复用当前训练好的检测头参数。
	_, backbone, detect_head, stride, class_names = build_yolov7_components(
		YOLOv7WrapperConfig(
			cfg_path=YOLO_CFG_PATH,
			weights_path=YOLO_WEIGHTS_PATH,
			num_classes=YOLO_NUM_CLASSES,
		),
		device=device,
	)
	feature_channels = tuple(detect_head.expected_in_channels)
	fie_config = MultiScaleFIEConfig(
		feature_channels=feature_channels,
		per_scale=(FIEBlockConfig(), FIEBlockConfig(), FIEBlockConfig()),
		project_after_fusion=True,
	)

	model_config = ICFIEYOLOConfig(
		enable_msicn=ENABLE_MSICN,
		enable_fie=ENABLE_FIE,
		fie=fie_config,
	)
	model = ICFIEYOLO(backbone_neck=backbone, detect_head=detect_head, config=model_config)

	# 根据 FIE 状态选择对应 checkpoint:
	#   FIE 开启 → Stage C checkpoint（全模型含 FIE 联合微调）
	#   FIE 关闭 → Stage A checkpoint（纯 YOLO 预训练，detect head 针对原始 backbone 特征）
	# 两者不可混用：Stage C 的 detect head 仅对 FIE 输出有效，关闭 FIE 后置信度会全部偏低。
	if ENABLE_FIE:
		checkpoint_path = ICFIE_CHECKPOINT_PATH
	else:
		checkpoint_path = YOLO_ONLY_CHECKPOINT_PATH
		if checkpoint_path is None and ICFIE_CHECKPOINT_PATH is not None:
			print(
				"[警告] ENABLE_FIE=False 但未设置 YOLO_ONLY_CHECKPOINT_PATH。\n"
				"  Stage C checkpoint 的 detect head 是针对 FIE 特征训练的，关闭 FIE 后无法正常检测。\n"
				"  请重新训练并将 YOLO_ONLY_CHECKPOINT_PATH 指向 stage_a_epoch_<N>.pt。"
			)

	if checkpoint_path is not None:
		checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
		model.load_state_dict(checkpoint["model"], strict=True)

	# build_yolov7_components 从 yolov7.pt 拿到的 class_names 是 COCO 80 类名
	# 必须替换为 ExDark 12 类名  否则 class_id=10 → "fire hydrant" 而非 "People"
	return model.to(device).eval(), stride, EXDARK_CLASS_NAMES


def run_single_image_pipeline() -> None:
	configure_matplotlib_cjk_font()
	torch.manual_seed(42)

	image_path = INPUT_DIR / TARGET_IMAGE_NAME
	if not image_path.exists():
		raise FileNotFoundError(
			f"未找到测试图像: {image_path}\n"
			"请先在项目根目录下创建 test_sample/，再把待测图片放进去。"
		)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	pipeline_dir = OUTPUT_DIR / "pipeline_visualization"
	pipeline_dir.mkdir(parents=True, exist_ok=True)

	if DEVICE.startswith("cuda") and not torch.cuda.is_available():
		print("[警告] 指定了 CUDA 但当前不可用  自动回退到 CPU")
		runtime_device = torch.device("cpu")
	else:
		runtime_device = torch.device(DEVICE)

	model, stride, class_names = build_model(runtime_device)
	image_tensor, original_image = load_yolo_image_as_tensor(image_path, IMG_SIZE, stride)
	image_tensor = image_tensor.to(runtime_device)

	with torch.no_grad():
		outputs = model(image_tensor, return_details=True)

	corrected_image = outputs["corrected_image"]
	raw_features = outputs["raw_features"]
	enhanced_features = outputs["enhanced_features"]
	decoded_predictions, raw_prediction_maps = split_prediction_outputs(outputs["predictions"])

	assert tuple(corrected_image.shape) == tuple(image_tensor.shape), "矫正图尺寸与输入不一致"
	assert len(raw_features) == 3, "主干应输出 3 个尺度特征"
	assert len(enhanced_features) == 3, "FIE 应输出 3 个尺度增强特征"
	assert len(raw_prediction_maps) == 3, "检测头应输出 3 个尺度预测"

	features_to_show = list(enhanced_features) if ENABLE_FIE else list(raw_features)

	if ENABLE_NMS:
		if decoded_predictions is None:
			raise RuntimeError("YOLOv7 eval 模式未返回解码后的预测结果  无法执行 NMS")
		detections = decode_single_image_detections(
			decoded_predictions=decoded_predictions,
			input_shape=tuple(image_tensor.shape[2:]),
			original_shape=original_image.shape[:2],
			conf_thres=CONF_THRES,
			iou_thres=IOU_THRES,
		)
	else:
		detections = torch.zeros((0, 6))

	stem = build_output_stem(
		image_stem=image_path.stem,
		enable_msicn=ENABLE_MSICN,
		enable_fie=ENABLE_FIE,
	)
	save_pipeline_visualization(
		original_image=original_image,
		corrected_tensor=corrected_image,
		features=features_to_show,
		detections=detections,
		class_names=class_names,
		enable_msicn=ENABLE_MSICN,
		enable_fie=ENABLE_FIE,
		enable_nms=ENABLE_NMS,
		save_path=pipeline_dir,
		filename=stem,
	)

	print("=" * 60)
	print(" 单图完整流程测试完成")
	print("=" * 60)
	print(f" 测试图片: {image_path}")
	print(f" 推理设备: {runtime_device}")
	print(f" YOLO配置: {YOLO_CFG_PATH}")
	print(f" YOLO权重: {YOLO_WEIGHTS_PATH if YOLO_WEIGHTS_PATH is not None else '未加载  使用随机初始化权重'}")
	print(f" 输入张量: {tuple(image_tensor.shape)}")
	print(f" 矫正张量: {tuple(corrected_image.shape)}")
	print(f" 原始特征: {[tuple(f.shape) for f in raw_features]}")
	print(f" 增强特征: {[tuple(f.shape) for f in enhanced_features]}")
	print(f" 检测原始输出: {[tuple(p.shape) for p in raw_prediction_maps]}")
	if decoded_predictions is not None:
		print(f" 解码输出: {tuple(decoded_predictions.shape)}")
	if ENABLE_NMS:
		print(f" NMS框数量: {len(detections)}  (conf={CONF_THRES}, iou={IOU_THRES})")
	print(f" 流程图目录: {pipeline_dir}")
	print(f" MSICN: {'开启' if ENABLE_MSICN else '关闭'}  FIE: {'开启' if ENABLE_FIE else '关闭'}  NMS: {'开启' if ENABLE_NMS else '关闭'}")


if __name__ == "__main__":
	run_single_image_pipeline()
