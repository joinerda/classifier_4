from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "runs"
    arch: str = "resnet18"
    pretrained: bool = True
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw or sgd
    num_workers: int = 2
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1)
    vertical_flip: bool = True
    image_size: int = 224
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    check_split_classes: bool = True
    eval_report: bool = False
    eval_confusion_matrix: bool = False
    eval_per_class: bool = False
    run_comment: Optional[str] = None
    seed: Optional[int] = 1337
    deterministic: bool = True
    device: Optional[str] = None
    aug_crop_scale_min: float = 0.7
    aug_rotation: float = 12.0
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_saturation: float = 0.2
    aug_hue: float = 0.05
    aug_erasing_prob: float = 0.2
    aug_erasing_scale_max: float = 0.2
    aug_blur_prob: float = 0.0
    aug_grayscale_prob: float = 0.0
    aug_perspective_prob: float = 0.0


@dataclass
class ManifestTrainConfig:
    manifest_path: str
    image_root: str
    train_domains: str          # comma-separated: "lab" | "wild" | "lab,wild"
    test_domains: str           # comma-separated: "lab" | "wild"
    output_dir: str = "runs"
    arch: str = "resnet34"
    pretrained: bool = True
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    num_workers: int = 8
    val_fraction: float = 0.15
    test_fraction: Optional[float] = None   # if set, overrides manifest split column
    max_per_class: Optional[int] = 400
    vertical_flip: bool = True
    image_size: int = 224
    early_stopping_patience: Optional[int] = 20
    early_stopping_min_delta: float = 0.0001
    use_dann: bool = False
    min_epochs: int = 0
    dann_lambda_max: float = 0.5
    dann_burnin_epochs: int = 0
    check_split_classes: bool = True
    eval_report: bool = True
    eval_confusion_matrix: bool = True
    eval_per_class: bool = True
    run_comment: Optional[str] = None
    seed: Optional[int] = 1337
    deterministic: bool = False
    device: Optional[str] = None
    labels_path: Optional[str] = None
    # aug params
    aug_crop_scale_min: float = 0.7
    aug_rotation: float = 12.0
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_saturation: float = 0.2
    aug_hue: float = 0.05
    aug_erasing_prob: float = 0.2
    aug_erasing_scale_max: float = 0.2
    aug_blur_prob: float = 0.0
    aug_grayscale_prob: float = 0.0
    aug_perspective_prob: float = 0.0


@dataclass
class DANNConfig:
    manifest_path: str
    image_root: str
    source_domain: str          # "lab" or "wild"
    target_domain: str          # "lab" or "wild"
    output_dir: str = "runs"
    arch: str = "resnet34"
    pretrained: bool = True
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    num_workers: int = 8
    val_fraction: float = 0.15
    max_per_class: Optional[int] = 400
    vertical_flip: bool = True
    image_size: int = 224
    early_stopping_patience: Optional[int] = 20
    early_stopping_min_delta: float = 0.0001
    min_epochs: int = 0
    dann_lambda_max: float = 0.5
    dann_burnin_epochs: int = 0
    eval_report: bool = True
    eval_confusion_matrix: bool = True
    eval_per_class: bool = True
    run_comment: Optional[str] = None
    seed: Optional[int] = 1337
    deterministic: bool = False
    device: Optional[str] = None
    labels_path: Optional[str] = None
    aug_crop_scale_min: float = 0.7
    aug_rotation: float = 12.0
    aug_brightness: float = 0.5
    aug_contrast: float = 0.5
    aug_saturation: float = 0.5
    aug_hue: float = 0.15
    aug_erasing_prob: float = 0.2
    aug_erasing_scale_max: float = 0.2
    aug_blur_prob: float = 0.3
    aug_grayscale_prob: float = 0.1
    aug_perspective_prob: float = 0.2


@dataclass
class EvalConfig:
    data_dir: str
    checkpoint_path: str
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 1337
    image_size: Optional[int] = None
    confusion_matrix: bool = False
    per_class: bool = False
    report_path: Optional[str] = None
    device: Optional[str] = None
    labels_path: Optional[str] = None


@dataclass
class PredictConfig:
    checkpoint_path: str
    image_paths: Sequence[str]
    top_k: int = 3
    image_size: Optional[int] = None
    device: Optional[str] = None
