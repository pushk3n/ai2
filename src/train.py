from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import Tensor, nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLOV7_ROOT = PROJECT_ROOT / "yolov7"
if str(YOLOV7_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV7_ROOT))

from fie import FIEBlockConfig, MultiScaleFIEConfig
from icfie_yolo import ICFIEYOLO, ICFIEYOLOConfig
from train_config import DatasetConfig, HardwareConfig, TrainConfig, TrainingStage, load_train_config
from utils.datasets import create_dataloader
from utils.general import check_img_size
from utils.loss import ComputeLoss, ComputeLossOTA
from yolo_wrapper import YOLOv7WrapperConfig, build_yolov7_components


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


def build_optimizer(model: ICFIEYOLO, config: TrainConfig) -> Adam:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("当前阶段没有可训练参数")
    return Adam(
        trainable_parameters,
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )


def forward_for_stage(model: ICFIEYOLO, images: Tensor, stage: TrainingStage) -> list[Tensor]:
    # 阶段 B: 通过 ICFIEYOLO.forward_stage_b 保持四层边界
    #         backbone_neck 和 detect_head 已在 apply_training_stage 中切换到 eval
    #         no_grad 包裹在 forward_stage_b 内部处理
    # 阶段 C: 正常全模型前向  四层均在 train 模式且梯度开放
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
            print(f"[跳过] {stage.value} epochs=0")
        return

    base_model = unwrap_model(model)
    apply_training_stage(base_model, stage)
    optimizer = build_optimizer(base_model, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=config.optimizer.min_lr)
    # torch.cuda.amp.GradScaler 已废弃，改用 torch.amp.GradScaler
    scaler = torch.amp.GradScaler(device.type, enabled=config.hardware.use_amp and device.type == "cuda")
    compute_loss = ComputeLossOTA(loss_owner) if config.use_ota_loss else ComputeLoss(loss_owner)
    accumulate_steps = max(config.hardware.accumulate_steps, 1)

    if is_main_process(rank):
        print(f"[开始] {stage.value} epochs={epochs} batch_size={config.hardware.batch_size} accumulate={accumulate_steps}")

    for epoch in range(epochs):
        if config.hardware.ddp and hasattr(dataloader, "sampler") and dataloader.sampler is not None:
            dataloader.sampler.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)
        progress = enumerate(dataloader)
        if is_main_process(rank):
            progress = tqdm(progress, total=len(dataloader), desc=f"{stage.value} {epoch + 1}/{epochs}")

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
                )

        scheduler.step()
        if is_main_process(rank) and (epoch + 1) % max(config.save_every, 1) == 0:
            save_checkpoint(model, optimizer, config, stage, epoch)


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


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, project_root=PROJECT_ROOT)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    device, rank, _, world_size = init_distributed_mode(config.hardware)
    try:
        set_seed(config.seed)
        hyp = load_hyperparameters(config)
        base_model, loss_owner, stride, class_names = build_model(config, device, hyp)
        model = maybe_wrap_model(base_model, device, config.hardware)
        dataloader, _, image_size = build_train_dataloader(config, hyp, stride, rank, world_size)

        save_stage_a_checkpoint(model, config, rank)
        if is_main_process(rank):
            print(f"[配置文件] {args.config.expanduser().resolve()}")
            print(f"[阶段A] 已加载 YOLOv7 预训练权重  视为完成")
            print(f"[配置] img_size={image_size} classes={len(class_names)} stride={stride}")
            # 将 YAML 配置快照写入 run_dir  便于后续复现实验
            import shutil
            shutil.copy2(args.config, config.run_dir / "train_config_snapshot.yaml")

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