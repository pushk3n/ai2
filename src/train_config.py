from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Tuple

import yaml

from yolo_wrapper import YOLOv7WrapperConfig


class TrainingStage(str, Enum):
    # 训练阶段枚举

    STAGE_A = "stage_a"
    STAGE_B = "stage_b"
    STAGE_C = "stage_c"


@dataclass(frozen=True)
class DatasetConfig:
    # 数据集配置
    # train_path  - 训练集路径  直接传给 YOLOv7 create_dataloader
    # val_path    - 验证集路径  当前训练入口先保留字段  后续 eval.py 复用
    # num_classes - 类别数
    # class_names - 类别名  可为空  为空时由训练入口自动补成 class_0...class_n

    train_path: Path
    val_path: Path | None
    num_classes: int
    class_names: Tuple[str, ...] = ()
    image_size: int = 416
    single_cls: bool = False

    def __post_init__(self) -> None:
        if self.train_path is None:
            raise ValueError("dataset.train_path 不可为空，请在配置文件中指定训练集路径")
        if self.num_classes <= 0:
            raise ValueError("num_classes 必须大于 0")
        if self.image_size <= 0:
            raise ValueError("image_size 必须大于 0")
        if self.class_names and len(self.class_names) != self.num_classes:
            raise ValueError("class_names 长度必须与 num_classes 一致")


@dataclass(frozen=True)
class OptimizerConfig:
    # 优化器配置

    lr: float = 1e-2
    stage_c_lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 5e-4
    beta1: float = 0.937
    beta2: float = 0.999

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("学习率必须大于 0")
        if self.stage_c_lr <= 0:
            raise ValueError("阶段 C 学习率必须大于 0")
        if self.min_lr <= 0:
            raise ValueError("最小学习率必须大于 0")
        if self.min_lr > self.lr:
            raise ValueError("最小学习率不能大于初始学习率")
        if self.min_lr > self.stage_c_lr:
            raise ValueError("最小学习率不能大于阶段 C 学习率")

    def lr_for_stage(self, stage: TrainingStage) -> float:
        if stage == TrainingStage.STAGE_C:
            return self.stage_c_lr
        return self.lr


@dataclass(frozen=True)
class StageScheduleConfig:
    # 分阶段训练轮数配置

    stage_a_epochs: int = 3
    stage_b_epochs: int = 1
    stage_c_epochs: int = 1

    def __post_init__(self) -> None:
        if self.stage_a_epochs < 0 or self.stage_b_epochs < 0 or self.stage_c_epochs < 0:
            raise ValueError("阶段训练轮数不能为负数")


@dataclass(frozen=True)
class VisualizationConfig:
    # 训练可视化配置

    batch_log_interval: int = 25

    def __post_init__(self) -> None:
        if self.batch_log_interval <= 0:
            raise ValueError("batch_log_interval 必须大于 0")


@dataclass
class HardwareConfig:
    # 硬件配置

    device: str = "cuda:0"
    use_amp: bool = True
    batch_size: int = 4
    accumulate_steps: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    ddp: bool = False
    use_grad_checkpoint: bool = False


@dataclass(frozen=True)
class TrainConfig:
    # 训练总配置

    seed: int
    run_dir: Path
    hyp_path: Path
    yolo: YOLOv7WrapperConfig
    dataset: DatasetConfig
    hardware: HardwareConfig
    optimizer: OptimizerConfig
    schedule: StageScheduleConfig
    visualization: VisualizationConfig
    project_after_fusion: bool = True
    save_every: int = 1
    use_ota_loss: bool = False
    cache_images: bool = False
    rect: bool = False


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 YAML 映射对象")
    return dict(value)


def _resolve_path(raw_value: str | None, *, project_root: Path, config_dir: Path) -> Path | None:
    if raw_value is None:
        return None

    path = Path(str(raw_value)).expanduser()
    if path.is_absolute():
        return path

    project_path = (project_root / path).resolve()
    config_path = (config_dir / path).resolve()

    if project_path.exists() or not config_path.exists():
        return project_path
    return config_path


def _parse_class_names(raw_value: Any) -> Tuple[str, ...]:
    if raw_value in (None, ""):
        return ()
    if isinstance(raw_value, str):
        return tuple(name.strip() for name in raw_value.split(",") if name.strip())
    if isinstance(raw_value, (list, tuple)):
        return tuple(str(name) for name in raw_value)
    raise TypeError("dataset.class_names 必须是字符串或字符串列表")


def load_train_config(config_path: Path, *, project_root: Path) -> TrainConfig:
    resolved_config_path = config_path.expanduser().resolve()
    with open(resolved_config_path, encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    root = _require_mapping("train_config", raw_config)
    config_dir = resolved_config_path.parent

    dataset_section = _require_mapping("dataset", root.get("dataset"))
    dataset = DatasetConfig(
        train_path=_resolve_path(dataset_section["train_path"], project_root=project_root, config_dir=config_dir),
        val_path=_resolve_path(dataset_section.get("val_path"), project_root=project_root, config_dir=config_dir),
        num_classes=int(dataset_section["num_classes"]),
        class_names=_parse_class_names(dataset_section.get("class_names")),
        image_size=int(dataset_section.get("image_size", 416)),
        single_cls=bool(dataset_section.get("single_cls", False)),
    )

    hardware_section = _require_mapping("hardware", root.get("hardware"))
    hardware = HardwareConfig(
        device=str(hardware_section.get("device", "cuda:0")),
        use_amp=bool(hardware_section.get("use_amp", True)),
        batch_size=int(hardware_section.get("batch_size", 4)),
        accumulate_steps=int(hardware_section.get("accumulate_steps", 4)),
        num_workers=int(hardware_section.get("num_workers", 4)),
        pin_memory=bool(hardware_section.get("pin_memory", True)),
        ddp=bool(hardware_section.get("ddp", False)),
        use_grad_checkpoint=bool(hardware_section.get("use_grad_checkpoint", False)),
    )

    optimizer_section = _require_mapping("optimizer", root.get("optimizer"))
    optimizer = OptimizerConfig(
        lr=float(optimizer_section.get("lr", 1e-2)),
        stage_c_lr=float(optimizer_section.get("stage_c_lr", 1e-4)),
        min_lr=float(optimizer_section.get("min_lr", 1e-5)),
        weight_decay=float(optimizer_section.get("weight_decay", 5e-4)),
        beta1=float(optimizer_section.get("beta1", 0.937)),
        beta2=float(optimizer_section.get("beta2", 0.999)),
    )

    schedule_section = _require_mapping("schedule", root.get("schedule"))
    schedule = StageScheduleConfig(
        stage_a_epochs=int(schedule_section.get("stage_a_epochs", 3)),
        stage_b_epochs=int(schedule_section.get("stage_b_epochs", 1)),
        stage_c_epochs=int(schedule_section.get("stage_c_epochs", 1)),
    )

    visualization_section = _require_mapping("visualization", root.get("visualization"))
    visualization = VisualizationConfig(
        batch_log_interval=int(visualization_section.get("batch_log_interval", 25)),
    )

    yolo_section = _require_mapping("yolo", root.get("yolo"))
    yolo_num_classes = int(yolo_section.get("num_classes", dataset.num_classes))
    if yolo_num_classes != dataset.num_classes:
        raise ValueError("yolo.num_classes 必须与 dataset.num_classes 一致")
    yolo = YOLOv7WrapperConfig(
        cfg_path=_resolve_path(yolo_section.get("cfg_path", "yolov7/cfg/training/yolov7.yaml"), project_root=project_root, config_dir=config_dir),
        weights_path=_resolve_path(yolo_section.get("weights_path", "yolov7/yolov7.pt"), project_root=project_root, config_dir=config_dir),
        num_classes=yolo_num_classes,
    )

    return TrainConfig(
        seed=int(root.get("seed", 42)),
        run_dir=_resolve_path(root.get("run_dir", "runs/icfie_yolo"), project_root=project_root, config_dir=config_dir),
        hyp_path=_resolve_path(root.get("hyp_path", "yolov7/data/hyp.scratch.p5.yaml"), project_root=project_root, config_dir=config_dir),
        yolo=yolo,
        dataset=dataset,
        hardware=hardware,
        optimizer=optimizer,
        schedule=schedule,
        visualization=visualization,
        project_after_fusion=bool(root.get("project_after_fusion", True)),
        save_every=int(root.get("save_every", 1)),
        use_ota_loss=bool(root.get("use_ota_loss", False)),
        cache_images=bool(root.get("cache_images", False)),
        rect=bool(root.get("rect", False)),
    )