from __future__ import annotations

import argparse
import json

from .config import DANNConfig, EvalConfig, ManifestTrainConfig, PredictConfig, TrainConfig
from .eval import evaluate_model
from .predict import predict_images
from .train import train_model, train_model_dann, train_model_manifest
from .tune import tune_hyperparams


def _require_single_domain(name: str, value: str) -> str:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 1:
        raise ValueError(f"{name} must name exactly one domain when DANN is enabled; got {value!r}")
    return parts[0]


def _cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arch=args.arch,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        num_workers=args.num_workers,
        split_ratios=(args.split_train, args.split_val, args.split_test),
        vertical_flip=args.vertical_flip,
        image_size=args.image_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        check_split_classes=args.check_split_classes,
        eval_report=args.eval_report,
        eval_confusion_matrix=args.eval_confusion_matrix,
        eval_per_class=args.eval_per_class,
        run_comment=args.run_comment,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        aug_crop_scale_min=args.aug_crop_scale_min,
        aug_rotation=args.aug_rotation,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_saturation=args.aug_saturation,
        aug_hue=args.aug_hue,
        aug_erasing_prob=args.aug_erasing_prob,
        aug_erasing_scale_max=args.aug_erasing_scale_max,
        aug_blur_prob=args.aug_blur_prob,
        aug_grayscale_prob=args.aug_grayscale_prob,
        aug_perspective_prob=args.aug_perspective_prob,
    )
    metrics = train_model(cfg)
    print(json.dumps(metrics, indent=2), flush=True)


def _cmd_eval(args: argparse.Namespace) -> None:
    cfg = EvalConfig(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        confusion_matrix=args.confusion_matrix,
        per_class=args.per_class,
        report_path=args.report_path,
        labels_path=args.labels_path,
        device=args.device,
    )
    metrics = evaluate_model(cfg)
    print(json.dumps(metrics, indent=2), flush=True)


def _cmd_predict(args: argparse.Namespace) -> None:
    cfg = PredictConfig(
        checkpoint_path=args.checkpoint,
        image_paths=args.images,
        top_k=args.top_k,
        image_size=args.image_size,
        device=args.device,
    )
    preds = predict_images(cfg)
    print(json.dumps(preds, indent=2), flush=True)


def _cmd_tune(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arch=args.arch,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        num_workers=args.num_workers,
        split_ratios=(args.split_train, args.split_val, args.split_test),
        vertical_flip=args.vertical_flip,
        image_size=args.image_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        check_split_classes=args.check_split_classes,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        aug_crop_scale_min=args.aug_crop_scale_min,
        aug_rotation=args.aug_rotation,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_saturation=args.aug_saturation,
        aug_hue=args.aug_hue,
        aug_erasing_prob=args.aug_erasing_prob,
        aug_erasing_scale_max=args.aug_erasing_scale_max,
        aug_blur_prob=args.aug_blur_prob,
        aug_grayscale_prob=args.aug_grayscale_prob,
        aug_perspective_prob=args.aug_perspective_prob,
    )

    grid = {
        "lr": args.tune_lr,
        "batch_size": args.tune_batch,
        "weight_decay": args.tune_wd,
    }
    results = tune_hyperparams(cfg, grid=grid, max_trials=args.max_trials)
    print(json.dumps(results, indent=2), flush=True)


def _cmd_train_manifest(args: argparse.Namespace) -> None:
    if args.use_dann:
        cfg = DANNConfig(
            manifest_path=args.manifest,
            image_root=args.image_root,
            source_domain=_require_single_domain("--train-domains", args.train_domains),
            target_domain=_require_single_domain("--test-domains", args.test_domains),
            output_dir=args.output_dir,
            arch=args.arch,
            pretrained=not args.no_pretrained,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            num_workers=args.num_workers,
            val_fraction=args.val_fraction,
            max_per_class=args.max_per_class,
            vertical_flip=args.vertical_flip,
            image_size=args.image_size,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            min_epochs=args.min_epochs,
            dann_lambda_max=args.dann_lambda_max,
            dann_burnin_epochs=args.dann_burnin_epochs,
            eval_report=args.eval_report,
            eval_confusion_matrix=args.eval_confusion_matrix,
            eval_per_class=args.eval_per_class,
            run_comment=args.run_comment,
            seed=args.seed,
            deterministic=args.deterministic,
            device=args.device,
            labels_path=args.labels_path,
            aug_crop_scale_min=args.aug_crop_scale_min,
            aug_rotation=args.aug_rotation,
            aug_brightness=args.aug_brightness,
            aug_contrast=args.aug_contrast,
            aug_saturation=args.aug_saturation,
            aug_hue=args.aug_hue,
            aug_erasing_prob=args.aug_erasing_prob,
            aug_erasing_scale_max=args.aug_erasing_scale_max,
            aug_blur_prob=args.aug_blur_prob,
            aug_grayscale_prob=args.aug_grayscale_prob,
            aug_perspective_prob=args.aug_perspective_prob,
        )
        metrics = train_model_dann(cfg)
        print(json.dumps(metrics, indent=2), flush=True)
        return

    cfg = ManifestTrainConfig(
        manifest_path=args.manifest,
        image_root=args.image_root,
        train_domains=args.train_domains,
        test_domains=args.test_domains,
        output_dir=args.output_dir,
        arch=args.arch,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        max_per_class=args.max_per_class,
        vertical_flip=args.vertical_flip,
        image_size=args.image_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_dann=args.use_dann,
        min_epochs=args.min_epochs,
        dann_lambda_max=args.dann_lambda_max,
        dann_burnin_epochs=args.dann_burnin_epochs,
        eval_report=args.eval_report,
        eval_confusion_matrix=args.eval_confusion_matrix,
        eval_per_class=args.eval_per_class,
        run_comment=args.run_comment,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        labels_path=args.labels_path,
        aug_crop_scale_min=args.aug_crop_scale_min,
        aug_rotation=args.aug_rotation,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_saturation=args.aug_saturation,
        aug_hue=args.aug_hue,
        aug_erasing_prob=args.aug_erasing_prob,
        aug_erasing_scale_max=args.aug_erasing_scale_max,
        aug_blur_prob=args.aug_blur_prob,
        aug_grayscale_prob=args.aug_grayscale_prob,
        aug_perspective_prob=args.aug_perspective_prob,
    )
    metrics = train_model_manifest(cfg)
    print(json.dumps(metrics, indent=2), flush=True)


def _cmd_train_dann(args: argparse.Namespace) -> None:
    cfg = DANNConfig(
        manifest_path=args.manifest,
        image_root=args.image_root,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        output_dir=args.output_dir,
        arch=args.arch,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        max_per_class=args.max_per_class,
        vertical_flip=args.vertical_flip,
        image_size=args.image_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        min_epochs=args.min_epochs,
        dann_lambda_max=args.dann_lambda_max,
        dann_burnin_epochs=args.dann_burnin_epochs,
        eval_report=args.eval_report,
        eval_confusion_matrix=args.eval_confusion_matrix,
        eval_per_class=args.eval_per_class,
        run_comment=args.run_comment,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        labels_path=args.labels_path,
        aug_crop_scale_min=args.aug_crop_scale_min,
        aug_rotation=args.aug_rotation,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_saturation=args.aug_saturation,
        aug_hue=args.aug_hue,
        aug_erasing_prob=args.aug_erasing_prob,
        aug_erasing_scale_max=args.aug_erasing_scale_max,
        aug_blur_prob=args.aug_blur_prob,
        aug_grayscale_prob=args.aug_grayscale_prob,
        aug_perspective_prob=args.aug_perspective_prob,
    )
    metrics = train_model_dann(cfg)
    print(json.dumps(metrics, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image classifier with ResNet")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", required=True)
        p.add_argument("--output-dir", default="runs")
        p.add_argument("--arch", default="resnet18")
        p.add_argument("--no-pretrained", action="store_true")
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--weight-decay", type=float, default=1e-4)
        p.add_argument("--optimizer", default="adamw")
        p.add_argument("--num-workers", type=int, default=2)
        p.add_argument("--split-train", type=float, default=0.8)
        p.add_argument("--split-val", type=float, default=0.1)
        p.add_argument("--split-test", type=float, default=0.1)
        p.add_argument("--image-size", type=int, default=224, help="Input size for resize/crop.")
        flip = p.add_mutually_exclusive_group()
        flip.add_argument("--vertical-flip", dest="vertical_flip", action="store_true")
        flip.add_argument("--no-vertical-flip", dest="vertical_flip", action="store_false")
        p.set_defaults(vertical_flip=True)
        p.add_argument(
            "--early-stopping-patience",
            type=int,
            default=None,
            help="Stop after N epochs without val_acc improvement (disabled if unset).",
        )
        p.add_argument(
            "--early-stopping-min-delta",
            type=float,
            default=0.0,
            help="Minimum val_acc improvement to reset early stopping.",
        )
        p.add_argument("--eval-report", action="store_true", help="Write eval_report.json after training.")
        p.add_argument(
            "--eval-confusion-matrix",
            action="store_true",
            help="Include confusion matrix in eval report.",
        )
        p.add_argument("--eval-per-class", action="store_true", help="Include per-class stats in eval report.")
        p.add_argument("--run-comment", default=None, help="Freeform run note stored in history/logs.")
        split_check = p.add_mutually_exclusive_group()
        split_check.add_argument(
            "--check-split-classes",
            dest="check_split_classes",
            action="store_true",
            help="Verify and remap train/val/test class indices when split folders exist.",
        )
        split_check.add_argument(
            "--no-check-split-classes",
            dest="check_split_classes",
            action="store_false",
            help="Skip train/val/test class consistency checks/remapping.",
        )
        p.set_defaults(check_split_classes=True)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--deterministic", action="store_true")
        p.add_argument("--device", default=None)
        p.add_argument("--aug-crop-scale-min", type=float, default=0.7)
        p.add_argument("--aug-rotation", type=float, default=12.0)
        p.add_argument("--aug-brightness", type=float, default=0.2)
        p.add_argument("--aug-contrast", type=float, default=0.2)
        p.add_argument("--aug-saturation", type=float, default=0.2)
        p.add_argument("--aug-hue", type=float, default=0.05)
        p.add_argument("--aug-erasing-prob", type=float, default=0.2)
        p.add_argument("--aug-erasing-scale-max", type=float, default=0.2)
        p.add_argument("--aug-blur-prob", type=float, default=0.0)
        p.add_argument("--aug-grayscale-prob", type=float, default=0.0)
        p.add_argument("--aug-perspective-prob", type=float, default=0.0)

    p_train = sub.add_parser("train", help="Train a model")
    add_shared(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_train_mf = sub.add_parser("train-manifest", help="Train a model using a manifest CSV for domain splits")
    p_train_mf.add_argument("--manifest", required=True, help="Path to manifest.csv")
    p_train_mf.add_argument("--image-root", required=True, help="Path to lab_wild_combined_feb_26/ directory")
    p_train_mf.add_argument(
        "--train-domains",
        required=True,
        help='Comma-separated domains for training (e.g. "lab" or "lab,wild")',
    )
    p_train_mf.add_argument(
        "--test-domains",
        required=True,
        help='Comma-separated domains for test evaluation (e.g. "wild")',
    )
    p_train_mf.add_argument("--output-dir", default="runs")
    p_train_mf.add_argument("--arch", default="resnet34")
    p_train_mf.add_argument("--no-pretrained", action="store_true")
    p_train_mf.add_argument("--epochs", type=int, default=100)
    p_train_mf.add_argument("--batch-size", type=int, default=64)
    p_train_mf.add_argument("--lr", type=float, default=3e-5)
    p_train_mf.add_argument("--weight-decay", type=float, default=1e-4)
    p_train_mf.add_argument("--optimizer", default="adamw")
    p_train_mf.add_argument("--num-workers", type=int, default=8)
    p_train_mf.add_argument("--val-fraction", type=float, default=0.15)
    p_train_mf.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="If set, ignore manifest split column and carve test set randomly at this fraction.",
    )
    p_train_mf.add_argument("--max-per-class", type=int, default=400)
    p_train_mf.add_argument("--image-size", type=int, default=224)
    mf_flip = p_train_mf.add_mutually_exclusive_group()
    mf_flip.add_argument("--vertical-flip", dest="vertical_flip", action="store_true")
    mf_flip.add_argument("--no-vertical-flip", dest="vertical_flip", action="store_false")
    p_train_mf.set_defaults(vertical_flip=True)
    p_train_mf.add_argument("--early-stopping-patience", type=int, default=20)
    p_train_mf.add_argument("--early-stopping-min-delta", type=float, default=0.0001)
    p_train_mf.add_argument("--use-dann", action="store_true", help="Use DANN instead of standard supervised manifest training.")
    p_train_mf.add_argument("--dann-lambda-max", type=float, default=0.5, help="Max domain loss weight when --use-dann is enabled.")
    p_train_mf.add_argument("--dann-burnin-epochs", type=int, default=0, help="Burn-in epochs with lambda=0 when --use-dann is enabled.")
    p_train_mf.add_argument("--min-epochs", type=int, default=0, help="Do not allow early stopping before this total epoch count.")
    p_train_mf.add_argument("--eval-report", action="store_true", default=True)
    p_train_mf.add_argument("--no-eval-report", dest="eval_report", action="store_false")
    p_train_mf.add_argument("--eval-confusion-matrix", action="store_true", default=True)
    p_train_mf.add_argument("--no-eval-confusion-matrix", dest="eval_confusion_matrix", action="store_false")
    p_train_mf.add_argument("--eval-per-class", action="store_true", default=True)
    p_train_mf.add_argument("--no-eval-per-class", dest="eval_per_class", action="store_false")
    p_train_mf.add_argument("--run-comment", default=None)
    p_train_mf.add_argument("--seed", type=int, default=1337)
    p_train_mf.add_argument("--deterministic", action="store_true")
    p_train_mf.add_argument("--device", default=None)
    p_train_mf.add_argument("--labels-path", default=None)
    p_train_mf.add_argument("--aug-crop-scale-min", type=float, default=0.7)
    p_train_mf.add_argument("--aug-rotation", type=float, default=12.0)
    p_train_mf.add_argument("--aug-brightness", type=float, default=0.2)
    p_train_mf.add_argument("--aug-contrast", type=float, default=0.2)
    p_train_mf.add_argument("--aug-saturation", type=float, default=0.2)
    p_train_mf.add_argument("--aug-hue", type=float, default=0.05)
    p_train_mf.add_argument("--aug-erasing-prob", type=float, default=0.2)
    p_train_mf.add_argument("--aug-erasing-scale-max", type=float, default=0.2)
    p_train_mf.add_argument("--aug-blur-prob", type=float, default=0.0)
    p_train_mf.add_argument("--aug-grayscale-prob", type=float, default=0.0)
    p_train_mf.add_argument("--aug-perspective-prob", type=float, default=0.0)
    p_train_mf.set_defaults(func=_cmd_train_manifest)

    p_dann = sub.add_parser("train-dann", help="Train a DANN model for cross-domain adaptation")
    p_dann.add_argument("--manifest", required=True, help="Path to manifest.csv")
    p_dann.add_argument("--image-root", required=True, help="Path to lab_wild_combined_feb_26/ directory")
    p_dann.add_argument("--source-domain", required=True, help='Domain to train species classifier on (e.g. "lab")')
    p_dann.add_argument("--target-domain", required=True, help='Domain to adapt toward and evaluate on (e.g. "wild")')
    p_dann.add_argument("--output-dir", default="runs")
    p_dann.add_argument("--arch", default="resnet34")
    p_dann.add_argument("--no-pretrained", action="store_true")
    p_dann.add_argument("--epochs", type=int, default=100)
    p_dann.add_argument("--batch-size", type=int, default=64)
    p_dann.add_argument("--lr", type=float, default=3e-5)
    p_dann.add_argument("--weight-decay", type=float, default=1e-4)
    p_dann.add_argument("--optimizer", default="adamw")
    p_dann.add_argument("--num-workers", type=int, default=8)
    p_dann.add_argument("--val-fraction", type=float, default=0.15)
    p_dann.add_argument("--max-per-class", type=int, default=400)
    p_dann.add_argument("--image-size", type=int, default=224)
    dann_flip = p_dann.add_mutually_exclusive_group()
    dann_flip.add_argument("--vertical-flip", dest="vertical_flip", action="store_true")
    dann_flip.add_argument("--no-vertical-flip", dest="vertical_flip", action="store_false")
    p_dann.set_defaults(vertical_flip=True)
    p_dann.add_argument("--early-stopping-patience", type=int, default=20)
    p_dann.add_argument("--early-stopping-min-delta", type=float, default=0.0001)
    p_dann.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="Do not allow early stopping before this total epoch count.",
    )
    p_dann.add_argument("--dann-lambda-max", type=float, default=0.5, help="Max domain loss weight (Ganin schedule).")
    p_dann.add_argument(
        "--dann-burnin-epochs",
        type=int,
        default=0,
        help="Train with lambda=0 for the first N epochs, then start DANN annealing.",
    )
    p_dann.add_argument("--eval-report", action="store_true", default=True)
    p_dann.add_argument("--no-eval-report", dest="eval_report", action="store_false")
    p_dann.add_argument("--eval-confusion-matrix", action="store_true", default=True)
    p_dann.add_argument("--no-eval-confusion-matrix", dest="eval_confusion_matrix", action="store_false")
    p_dann.add_argument("--eval-per-class", action="store_true", default=True)
    p_dann.add_argument("--no-eval-per-class", dest="eval_per_class", action="store_false")
    p_dann.add_argument("--run-comment", default=None)
    p_dann.add_argument("--seed", type=int, default=1337)
    p_dann.add_argument("--deterministic", action="store_true")
    p_dann.add_argument("--device", default=None)
    p_dann.add_argument("--labels-path", default=None)
    p_dann.add_argument("--aug-crop-scale-min", type=float, default=0.7)
    p_dann.add_argument("--aug-rotation", type=float, default=12.0)
    p_dann.add_argument("--aug-brightness", type=float, default=0.5)
    p_dann.add_argument("--aug-contrast", type=float, default=0.5)
    p_dann.add_argument("--aug-saturation", type=float, default=0.5)
    p_dann.add_argument("--aug-hue", type=float, default=0.15)
    p_dann.add_argument("--aug-erasing-prob", type=float, default=0.2)
    p_dann.add_argument("--aug-erasing-scale-max", type=float, default=0.2)
    p_dann.add_argument("--aug-blur-prob", type=float, default=0.3)
    p_dann.add_argument("--aug-grayscale-prob", type=float, default=0.1)
    p_dann.add_argument("--aug-perspective-prob", type=float, default=0.2)
    p_dann.set_defaults(func=_cmd_train_dann)

    p_eval = sub.add_parser("test", help="Evaluate a model")
    p_eval.add_argument("--data-dir", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--batch-size", type=int, default=32)
    p_eval.add_argument("--num-workers", type=int, default=2)
    p_eval.add_argument("--seed", type=int, default=1337)
    p_eval.add_argument("--image-size", type=int, default=None, help="Override input size (default: from checkpoint).")
    p_eval.add_argument("--confusion-matrix", action="store_true", help="Include confusion matrix in output.")
    p_eval.add_argument("--per-class", action="store_true", help="Include per-class accuracy in output.")
    p_eval.add_argument("--report-path", default=None, help="Write full report JSON to this path.")
    p_eval.add_argument("--labels-path", default=None, help="Path to labels.txt for species name lookup.")
    p_eval.add_argument("--device", default=None)
    p_eval.set_defaults(func=_cmd_eval)

    p_pred = sub.add_parser("predict", help="Predict images")
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--images", nargs="+", required=True)
    p_pred.add_argument("--top-k", type=int, default=3)
    p_pred.add_argument("--image-size", type=int, default=None, help="Override input size (default: from checkpoint).")
    p_pred.add_argument("--device", default=None)
    p_pred.set_defaults(func=_cmd_predict)

    p_tune = sub.add_parser("tune", help="Hyperparameter tuning")
    add_shared(p_tune)
    p_tune.add_argument("--tune-lr", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    p_tune.add_argument("--tune-batch", nargs="+", type=int, default=[16, 32, 64])
    p_tune.add_argument("--tune-wd", nargs="+", type=float, default=[0.0, 1e-4, 1e-3])
    p_tune.add_argument("--max-trials", type=int, default=None)
    p_tune.set_defaults(func=_cmd_tune)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
