from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLOV7_ROOT = PROJECT_ROOT / "yolov7"
if str(YOLOV7_ROOT) not in sys.path:
    sys.path.append(str(YOLOV7_ROOT))

from fie import FIEBlockConfig, MultiScaleFIEConfig
from icfie_yolo import ICFIEYOLO, ICFIEYOLOConfig
from train_config import DatasetConfig, HardwareConfig, TrainConfig, TrainingStage, load_train_config
from utils.datasets import create_dataloader
from utils.general import check_img_size
from utils.loss import ComputeLoss, ComputeLossOTA
from yolo_wrapper import YOLOv7WrapperConfig, build_yolov7_components


def configure_matplotlib_cjk_font() -> None:
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

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_matplotlib_cjk_font()


@dataclass(frozen=True)
class EpochMetric:
    stage: str
    stage_index: int
    epoch: int
    epochs: int
    box_loss: float
    obj_loss: float
    cls_loss: float
    total_loss: float
    learning_rate: float
    elapsed_seconds: float


@dataclass(frozen=True)
class BatchMetric:
    stage: str
    stage_index: int
    epoch: int
    batch: int
    total_batches: int
    box_loss: float
    obj_loss: float
    cls_loss: float
    total_loss: float
    learning_rate: float
    elapsed_seconds: float


def stage_display_name(stage: TrainingStage) -> str:
    if stage == TrainingStage.STAGE_A:
        return "阶段A / Stage A"
    if stage == TrainingStage.STAGE_B:
        return "阶段B / Stage B"
    if stage == TrainingStage.STAGE_C:
        return "阶段C / Stage C"
    return f"{stage.value} / {stage.value.upper()}"


def metrics_csv_path(run_dir: Path) -> Path:
    return run_dir / "training_metrics.csv"


def metrics_plot_path(run_dir: Path) -> Path:
    return run_dir / "training_metrics.png"


def batch_metrics_csv_path(run_dir: Path) -> Path:
    return run_dir / "training_batch_metrics.csv"


def reset_visualization_artifacts(run_dir: Path) -> None:
    for artifact_path in (metrics_csv_path(run_dir), batch_metrics_csv_path(run_dir), metrics_plot_path(run_dir)):
        if artifact_path.exists():
            artifact_path.unlink()


def append_epoch_metrics(run_dir: Path, metric: EpochMetric) -> None:
    output_path = metrics_csv_path(run_dir)
    should_write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "stage_index",
                "epoch",
                "epochs",
                "box_loss",
                "obj_loss",
                "cls_loss",
                "total_loss",
                "learning_rate",
                "elapsed_seconds",
            ),
        )
        if should_write_header:
            writer.writeheader()
        writer.writerow(asdict(metric))


def append_batch_metrics(run_dir: Path, metric: BatchMetric) -> None:
    output_path = batch_metrics_csv_path(run_dir)
    should_write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "stage_index",
                "epoch",
                "batch",
                "total_batches",
                "box_loss",
                "obj_loss",
                "cls_loss",
                "total_loss",
                "learning_rate",
                "elapsed_seconds",
            ),
        )
        if should_write_header:
            writer.writeheader()
        writer.writerow(asdict(metric))


def load_epoch_metrics(run_dir: Path) -> list[EpochMetric]:
    input_path = metrics_csv_path(run_dir)
    if not input_path.exists():
        return []

    metrics: list[EpochMetric] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics.append(
                EpochMetric(
                    stage=str(row["stage"]),
                    stage_index=int(row["stage_index"]),
                    epoch=int(row["epoch"]),
                    epochs=int(row["epochs"]),
                    box_loss=float(row["box_loss"]),
                    obj_loss=float(row["obj_loss"]),
                    cls_loss=float(row["cls_loss"]),
                    total_loss=float(row["total_loss"]),
                    learning_rate=float(row["learning_rate"]),
                    elapsed_seconds=float(row["elapsed_seconds"]),
                )
            )
    return metrics


def load_batch_metrics(run_dir: Path) -> list[BatchMetric]:
    input_path = batch_metrics_csv_path(run_dir)
    if not input_path.exists():
        return []

    metrics: list[BatchMetric] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics.append(
                BatchMetric(
                    stage=str(row["stage"]),
                    stage_index=int(row["stage_index"]),
                    epoch=int(row["epoch"]),
                    batch=int(row["batch"]),
                    total_batches=int(row["total_batches"]),
                    box_loss=float(row["box_loss"]),
                    obj_loss=float(row["obj_loss"]),
                    cls_loss=float(row["cls_loss"]),
                    total_loss=float(row["total_loss"]),
                    learning_rate=float(row["learning_rate"]),
                    elapsed_seconds=float(row["elapsed_seconds"]),
                )
            )
    return metrics


def should_log_batch_metric(batch_index: int, total_batches: int, interval: int) -> bool:
    current_batch = batch_index + 1
    if current_batch == 1 or current_batch == total_batches:
        return True
    return current_batch % max(interval, 1) == 0


def render_training_metrics(run_dir: Path) -> None:
    metrics = load_epoch_metrics(run_dir)
    batch_metrics = load_batch_metrics(run_dir)
    if not metrics and not batch_metrics:
        return

    stage_colors = {
        TrainingStage.STAGE_A.value: "#2ca02c",
        TrainingStage.STAGE_B.value: "#2f6fed",
        TrainingStage.STAGE_C.value: "#dd8452",
    }
    stage_labels = {
        TrainingStage.STAGE_A.value: "阶段A / Stage A",
        TrainingStage.STAGE_B.value: "阶段B / Stage B",
        TrainingStage.STAGE_C.value: "阶段C / Stage C",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor("#f7f5ef")
    line_specs = (
        ("box_loss", "框损失 / Box Loss", axes[0, 0]),
        ("obj_loss", "目标损失 / Objectness Loss", axes[0, 1]),
        ("cls_loss", "分类损失 / Classification Loss", axes[1, 0]),
        ("total_loss", "总损失 / Total Loss", axes[1, 1]),
    )

    last_batch_position_by_epoch: dict[tuple[str, int], int] = {}
    for position, metric in enumerate(batch_metrics, start=1):
        last_batch_position_by_epoch[(metric.stage, metric.epoch)] = position

    epoch_positions = []
    for index, metric in enumerate(metrics, start=1):
        epoch_positions.append(last_batch_position_by_epoch.get((metric.stage, metric.epoch), index))

    for field_name, title, axis in line_specs:
        axis.set_facecolor("#fffdf8")
        for stage_value in (TrainingStage.STAGE_A.value, TrainingStage.STAGE_B.value, TrainingStage.STAGE_C.value):
            stage_batch_points = [
                (position, metric)
                for position, metric in enumerate(batch_metrics, start=1)
                if metric.stage == stage_value
            ]
            if stage_batch_points:
                axis.plot(
                    [position for position, _ in stage_batch_points],
                    [getattr(metric, field_name) for _, metric in stage_batch_points],
                    linewidth=1.3,
                    alpha=0.35,
                    color=stage_colors.get(stage_value, "#333333"),
                )

            stage_epoch_points = [
                (position, metric)
                for position, metric in zip(epoch_positions, metrics)
                if metric.stage == stage_value
            ]
            if stage_epoch_points:
                axis.plot(
                    [position for position, _ in stage_epoch_points],
                    [getattr(metric, field_name) for _, metric in stage_epoch_points],
                    marker="o",
                    linewidth=2.2,
                    markersize=6.0,
                    color=stage_colors.get(stage_value, "#333333"),
                    label=stage_labels.get(stage_value, stage_value),
                )
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("批次记录 / Logged Batch Step")
        axis.set_ylabel(field_name)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        axis.legend(loc="best")

    fig.suptitle(
        "ICFIE-YOLO 训练过程可视化 / Training Overview\n浅色曲线=批次趋势 深色圆点=轮次均值 / light line=batch trend, dark marker=epoch mean",
        fontsize=15,
        y=0.99,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(metrics_plot_path(run_dir), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def summarize_trainable_modules(model: ICFIEYOLO) -> str:
    module_pairs = (
        ("MSICN", model.msicn),
        ("BackboneNeck", model.backbone_neck),
        ("FIE", model.fie),
        ("DetectHead", model.detect_head),
    )
    status_parts = []
    for module_name, module in module_pairs:
        is_trainable = any(parameter.requires_grad for parameter in module.parameters())
        status_parts.append(f"{module_name}={'Train' if is_trainable else 'Frozen'}")
    return " | ".join(status_parts)


def stage_order(stage: TrainingStage) -> int:
    if stage == TrainingStage.STAGE_B:
        return 2
    if stage == TrainingStage.STAGE_C:
        return 3
    return 1


def set_seed(seed: int) -> None:
    # 显式固定 torch / numpy / random 的随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def is_main_process(rank: int) -> bool:
    return rank in (-1, 0)


def unwrap_model(model: nn.Module) -> ICFIEYOLO:
    return model.module if isinstance(model, DDP) else model


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def extract_prediction_maps(predictions: object) -> list[Tensor]:
    # 统一抽取可用于 YOLOv7 loss 的原始多尺度输出
    if isinstance(predictions, tuple) and len(predictions) == 2 and isinstance(predictions[1], (list, tuple)):
        return list(predictions[1])
    if isinstance(predictions, (list, tuple)) and all(isinstance(item, Tensor) for item in predictions):
        return list(predictions)
    raise TypeError("检测头输出格式不符合 ComputeLoss 预期")


def init_distributed_mode(hardware: HardwareConfig) -> tuple[torch.device, int, int, int]:
    if hardware.ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return device, rank, local_rank, world_size

    requested_device = hardware.device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "当前配置请求使用 CUDA，但当前环境中的 CUDA 不可用。"
            "请更新 NVIDIA 驱动/匹配的 PyTorch CUDA 版本，或把 hardware.device 改为 cpu。"
        )

    return torch.device(hardware.device), -1, 0, 1


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_dataloader_option(dataset_config: DatasetConfig) -> SimpleNamespace:
    # create_dataloader 只依赖 single_cls 这一项
    return SimpleNamespace(single_cls=dataset_config.single_cls)


@contextmanager
def yolov7_cache_torch_load_compat():
    # PyTorch 2.6+ 将 torch.load 的 weights_only 默认值改为 True。
    # YOLOv7 的数据集 .cache 文件存的是普通 Python / NumPy 对象，
    # 不是纯权重张量，直接按默认行为加载会触发 UnpicklingError。
    # 这里只在 create_dataloader 调用期间临时改回 weights_only=False，
    # 作用范围限定在本地受信任的数据缓存读取，不影响其他调用点。
    original_torch_load = torch.load

    def compatible_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = compatible_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_hyperparameters(config: TrainConfig) -> dict[str, float]:
    with open(config.hyp_path, encoding="utf-8") as handle:
        hyp = yaml.safe_load(handle)
    hyp["lr0"] = config.optimizer.lr
    hyp["lrf"] = config.optimizer.min_lr / config.optimizer.lr
    return hyp


def build_model(config: TrainConfig, device: torch.device, hyp: dict[str, float]) -> tuple[ICFIEYOLO, nn.Module, int, tuple[str, ...]]:
    yolo_model, backbone_neck, detect_head, stride, _ = build_yolov7_components(config.yolo, device=device)

    # PRD 7.1: 若 use_grad_checkpoint=True，在 backbone_neck 上启用梯度检查点以节省显存
    # 注意: 梯度检查点仅在 image.requires_grad=True 时生效
    # 使用场景: 4060 等 VRAM 受限显卡的阶段 C 全模型微调
    backbone_neck.use_grad_checkpoint = config.hardware.use_grad_checkpoint

    feature_channels = tuple(detect_head.expected_in_channels)
    fie_config = MultiScaleFIEConfig(
        feature_channels=feature_channels,
        per_scale=tuple(FIEBlockConfig() for _ in feature_channels),
        project_after_fusion=config.project_after_fusion,
        projection_channels=feature_channels if config.project_after_fusion else None,
    )
    model_config = ICFIEYOLOConfig(enable_msicn=True, enable_fie=True, fie=fie_config)
    model = ICFIEYOLO(backbone_neck=backbone_neck, detect_head=detect_head, config=model_config).to(device)

    yolo_model.hyp = hyp
    yolo_model.gr = 1.0

    class_names = config.dataset.class_names or tuple(f"class_{index}" for index in range(config.dataset.num_classes))
    return model, yolo_model, stride, class_names


def build_train_dataloader(
    config: TrainConfig,
    hyp: dict[str, float],
    stride: int,
    rank: int,
    world_size: int,
):
    train_path = config.dataset.train_path
    if not train_path.exists():
        raise FileNotFoundError(f"未找到训练集路径: {train_path}")

    image_size = check_img_size(config.dataset.image_size, stride)
    with yolov7_cache_torch_load_compat():
        dataloader, dataset = create_dataloader(
            str(train_path),
            image_size,
            config.hardware.batch_size,
            stride,
            build_dataloader_option(config.dataset),
            hyp=hyp,
            augment=True,
            cache=config.cache_images,
            pad=0.0,
            rect=config.rect,
            rank=rank,
            world_size=world_size,
            workers=config.hardware.num_workers,
            image_weights=False,
            quad=False,
            prefix="train: ",
        )
    return dataloader, dataset, image_size


def apply_training_stage(model: ICFIEYOLO, stage: TrainingStage) -> None:
    # 显式设置每个模块的 requires_grad  状态符合 PRD 约束
    if stage == TrainingStage.STAGE_A:
        set_module_requires_grad(model.backbone_neck, True)
        set_module_requires_grad(model.detect_head, True)
        set_module_requires_grad(model.msicn, False)
        set_module_requires_grad(model.fie, False)

        model.backbone_neck.train()
        model.detect_head.train()
        model.msicn.eval()
        model.fie.eval()
        return

    if stage == TrainingStage.STAGE_B:
        set_module_requires_grad(model.msicn, True)
        set_module_requires_grad(model.backbone_neck, False)
        set_module_requires_grad(model.fie, False)
        set_module_requires_grad(model.detect_head, False)

        model.msicn.train()
        model.backbone_neck.eval()
        model.fie.eval()
        model.detect_head.eval()
        return

    if stage == TrainingStage.STAGE_C:
        set_module_requires_grad(model.msicn, True)
        set_module_requires_grad(model.backbone_neck, True)
        set_module_requires_grad(model.fie, True)
        set_module_requires_grad(model.detect_head, True)

        model.msicn.train()
        model.backbone_neck.train()
        model.fie.train()
        model.detect_head.train()
        return

    raise ValueError(f"不支持的训练阶段: {stage}")


def build_optimizer(model: ICFIEYOLO, config: TrainConfig, stage: TrainingStage) -> Adam:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("当前阶段没有可训练参数")
    learning_rate = config.optimizer.lr_for_stage(stage)
    return Adam(
        trainable_parameters,
        lr=learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )


def forward_for_stage(model: ICFIEYOLO, images: Tensor, stage: TrainingStage) -> list[Tensor]:
    # 阶段 A: 通过 ICFIEYOLO.forward_stage_a 保持四层边界
    #         backbone_neck 和 detect_head 在 train 模式  msicn / fie 在 eval 模式
    #         不经过 MSICN 和 FIE，建立纯 YOLO 基线
    # 阶段 B: 通过 ICFIEYOLO.forward_stage_b 保持四层边界
    #         backbone_neck 和 detect_head 已在 apply_training_stage 中切换到 eval
    #         no_grad 包裹在 forward_stage_b 内部处理
    # 阶段 C: 正常全模型前向  四层均在 train 模式且梯度开放
    if stage == TrainingStage.STAGE_A:
        predictions = model.forward_stage_a(images)
        return extract_prediction_maps(predictions)

    if stage == TrainingStage.STAGE_B:
        predictions = model.forward_stage_b(images)
        return extract_prediction_maps(predictions)

    predictions = model(images)
    return extract_prediction_maps(predictions)


def maybe_wrap_model(model: ICFIEYOLO, device: torch.device, hardware: HardwareConfig) -> nn.Module:
    wrapped_model: nn.Module = model
    if hardware.ddp and device.type == "cuda":
        wrapped_model = nn.SyncBatchNorm.convert_sync_batchnorm(wrapped_model)
        wrapped_model = DDP(
            wrapped_model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
    elif hardware.ddp:
        wrapped_model = DDP(wrapped_model, find_unused_parameters=True)
    return wrapped_model


def save_checkpoint(
    model: nn.Module,
    optimizer: Adam,
    config: TrainConfig,
    stage: TrainingStage,
    epoch: int,
) -> None:
    checkpoint_path = config.run_dir / f"{stage.value}_epoch_{epoch + 1}.pt"
    checkpoint = {
        "stage": stage.value,
        "epoch": epoch + 1,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
    }
    torch.save(checkpoint, checkpoint_path)


def train_stage(
    model: nn.Module,
    loss_owner: nn.Module,
    dataloader,
    config: TrainConfig,
    stage: TrainingStage,
    epochs: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    if epochs == 0:
        if is_main_process(rank):
            print(f"[跳过 Skip] {stage_display_name(stage)} | epochs=0")
        return

    base_model = unwrap_model(model)
    apply_training_stage(base_model, stage)
    optimizer = build_optimizer(base_model, config, stage)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=config.optimizer.min_lr)
    # torch.cuda.amp.GradScaler 已废弃，改用 torch.amp.GradScaler
    scaler = torch.amp.GradScaler(device.type, enabled=config.hardware.use_amp and device.type == "cuda")
    compute_loss = ComputeLossOTA(loss_owner) if config.use_ota_loss else ComputeLoss(loss_owner)
    accumulate_steps = max(config.hardware.accumulate_steps, 1)

    if is_main_process(rank):
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[开始 Start] {stage_display_name(stage)} | epochs={epochs} | batch_size={config.hardware.batch_size} "
            f"| accumulate={accumulate_steps} | lr={current_lr:.6g}"
        )
        print(f"[阶段配置 Stage Setup] {summarize_trainable_modules(base_model)}")
        print(
            f"[可视化 Visualization] epoch_csv={metrics_csv_path(config.run_dir)} | batch_csv={batch_metrics_csv_path(config.run_dir)} "
            f"| png={metrics_plot_path(config.run_dir)} | interval={config.visualization.batch_log_interval}"
        )

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        if config.hardware.ddp and hasattr(dataloader, "sampler") and dataloader.sampler is not None:
            dataloader.sampler.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)
        progress = enumerate(dataloader)
        if is_main_process(rank):
            progress = tqdm(
                progress,
                total=len(dataloader),
                desc=f"{stage_display_name(stage)} | epoch {epoch + 1}/{epochs}",
            )

        mean_loss = torch.zeros(4, device=device)
        for batch_index, (images, targets, _, _) in progress:
            images = images.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.hardware.use_amp and device.type == "cuda"):
                predictions = forward_for_stage(base_model, images, stage)
                loss, loss_items = compute_loss(predictions, targets)
                if world_size > 1:
                    loss = loss * world_size
                loss = loss / accumulate_steps

            loss_is_finite = torch.isfinite(loss.detach())
            loss_items_are_finite = torch.isfinite(loss_items.detach()).all()
            if not bool(loss_is_finite and loss_items_are_finite):
                current_lr = optimizer.param_groups[0]["lr"]
                raise RuntimeError(
                    f"{stage_display_name(stage)} 在 epoch {epoch + 1} batch {batch_index + 1} 出现非有限损失 / non-finite loss: "
                    f"loss={loss.detach().item()} loss_items={loss_items.detach().tolist()} lr={current_lr:.6g}"
                )

            scaler.scale(loss).backward()

            should_step = (batch_index + 1) % accumulate_steps == 0 or (batch_index + 1) == len(dataloader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if is_main_process(rank):
                mean_loss = (mean_loss * batch_index + loss_items.detach()) / (batch_index + 1)
                progress.set_postfix(
                    box=f"{mean_loss[0].item():.4f}",
                    obj=f"{mean_loss[1].item():.4f}",
                    cls=f"{mean_loss[2].item():.4f}",
                    total=f"{mean_loss[3].item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.3e}",
                )
                if should_log_batch_metric(batch_index, len(dataloader), config.visualization.batch_log_interval):
                    append_batch_metrics(
                        config.run_dir,
                        BatchMetric(
                            stage=stage.value,
                            stage_index=stage_order(stage),
                            epoch=epoch + 1,
                            batch=batch_index + 1,
                            total_batches=len(dataloader),
                            box_loss=float(mean_loss[0].item()),
                            obj_loss=float(mean_loss[1].item()),
                            cls_loss=float(mean_loss[2].item()),
                            total_loss=float(mean_loss[3].item()),
                            learning_rate=float(optimizer.param_groups[0]["lr"]),
                            elapsed_seconds=float(time.perf_counter() - epoch_start_time),
                        ),
                    )

        epoch_elapsed = time.perf_counter() - epoch_start_time
        metric = EpochMetric(
            stage=stage.value,
            stage_index=stage_order(stage),
            epoch=epoch + 1,
            epochs=epochs,
            box_loss=float(mean_loss[0].item()),
            obj_loss=float(mean_loss[1].item()),
            cls_loss=float(mean_loss[2].item()),
            total_loss=float(mean_loss[3].item()),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            elapsed_seconds=epoch_elapsed,
        )
        if is_main_process(rank):
            append_epoch_metrics(config.run_dir, metric)
            render_training_metrics(config.run_dir)
            print(
                f"[轮次完成 Epoch End] {stage_display_name(stage)} | epoch={epoch + 1}/{epochs} "
                f"| box损失 box={metric.box_loss:.4f} | 目标损失 obj={metric.obj_loss:.4f} "
                f"| 分类损失 cls={metric.cls_loss:.4f} | 总损失 total={metric.total_loss:.4f} "
                f"| 学习率 lr={metric.learning_rate:.6g} | 用时 elapsed={metric.elapsed_seconds:.1f}s"
            )

        scheduler.step()
        if is_main_process(rank) and (epoch + 1) % max(config.save_every, 1) == 0:
            save_checkpoint(model, optimizer, config, stage, epoch)
            print(
                f"[检查点 Checkpoint] {stage_display_name(stage)} | saved={config.run_dir / f'{stage.value}_epoch_{epoch + 1}.pt'}"
            )


def save_stage_a_checkpoint(model: nn.Module, config: TrainConfig, rank: int) -> None:
    if not is_main_process(rank):
        return
    checkpoint_path = config.run_dir / "stage_a_loaded.pt"
    torch.save({"stage": TrainingStage.STAGE_A.value, "model": unwrap_model(model).state_dict()}, checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICFIE-YOLO 三阶段训练入口")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train.yaml",
        help="训练 YAML 配置文件路径",
    )
    return parser.parse_args()


def resolve_cli_config_path(config_path: Path, *, project_root: Path) -> Path:
    candidate = config_path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = (
        Path.cwd(),
        project_root,
        Path(__file__).resolve().parent,
    )
    for base_dir in search_roots:
        resolved_path = (base_dir / candidate).resolve()
        if resolved_path.exists():
            return resolved_path

    # 若所有候选都不存在，则保留基于当前工作目录的解析结果，交给后续 open 抛出明确错误。
    return (Path.cwd() / candidate).resolve()


def main() -> None:
    args = parse_args()
    resolved_config_path = resolve_cli_config_path(args.config, project_root=PROJECT_ROOT)
    config = load_train_config(resolved_config_path, project_root=PROJECT_ROOT)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    device, rank, _, world_size = init_distributed_mode(config.hardware)
    try:
        if is_main_process(rank):
            reset_visualization_artifacts(config.run_dir)
        set_seed(config.seed)
        hyp = load_hyperparameters(config)
        base_model, loss_owner, stride, class_names = build_model(config, device, hyp)
        model = maybe_wrap_model(base_model, device, config.hardware)
        dataloader, _, image_size = build_train_dataloader(config, hyp, stride, rank, world_size)

        if is_main_process(rank):
            print(f"[配置文件 Config] {resolved_config_path}")
            print(f"[训练配置 Train Setup] img_size={image_size} | classes={len(class_names)} | stride={stride}")
            # 将 YAML 配置快照写入 run_dir  便于后续复现实验
            import shutil
            shutil.copy2(resolved_config_path, config.run_dir / "train_config_snapshot.yaml")
            print(f"[运行目录 Run Dir] {config.run_dir}")

        # 阶段 A: 预训练 backbone + detect head（纯 YOLO，不经过 MSICN/FIE）
        # 保存初始权重快照（yolov7.pt 迁移状态，阶段 A 训练前）
        save_stage_a_checkpoint(model, config, rank)
        train_stage(
            model=model,
            loss_owner=loss_owner,
            dataloader=dataloader,
            config=config,
            stage=TrainingStage.STAGE_A,
            epochs=config.schedule.stage_a_epochs,
            device=device,
            rank=rank,
            world_size=world_size,
        )

        train_stage(
            model=model,
            loss_owner=loss_owner,
            dataloader=dataloader,
            config=config,
            stage=TrainingStage.STAGE_B,
            epochs=config.schedule.stage_b_epochs,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        train_stage(
            model=model,
            loss_owner=loss_owner,
            dataloader=dataloader,
            config=config,
            stage=TrainingStage.STAGE_C,
            epochs=config.schedule.stage_c_epochs,
            device=device,
            rank=rank,
            world_size=world_size,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()