from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import SAM, YOLO

DEFAULT_MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")
DEFAULT_SAM_MODEL = "sam2.1_b.pt"


def resolve_device(value: str) -> str | int:
    if value.lower() != "auto":
        return int(value) if value.isdigit() else value
    return 0 if torch.cuda.is_available() else "cpu"


def environment_info(device: str | int) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        index = device if isinstance(device, int) else 0
        gpu = torch.cuda.get_device_name(index)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_source(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")
    return path


def prepare_project(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "run"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    return {
        "precision": as_float(getattr(box, "mp", None)),
        "recall": as_float(getattr(box, "mr", None)),
        "map50": as_float(getattr(box, "map50", None)),
        "map75": as_float(getattr(box, "map75", None)),
        "map50_95": as_float(getattr(box, "map", None)),
        "preprocess_ms": as_float(speed.get("preprocess")),
        "inference_ms": as_float(speed.get("inference")),
        "loss_ms": as_float(speed.get("loss")),
        "postprocess_ms": as_float(speed.get("postprocess")),
    }


def locate_best_weights(model: YOLO, train_result: Any) -> Path:
    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None)
    if best:
        candidate = Path(best)
        if candidate.is_file():
            return candidate.resolve()
    save_dir = getattr(train_result, "save_dir", None) or getattr(trainer, "save_dir", None)
    if save_dir:
        candidate = Path(save_dir) / "weights" / "best.pt"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Training completed but best.pt could not be located")


def train_model(
    model_reference: str,
    data: Path,
    project: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | int,
    workers: int,
    patience: int,
    optimizer: str,
    seed: int,
    cache: bool,
) -> tuple[Path, float]:
    seed_everything(seed)
    model = YOLO(model_reference)
    started = time.perf_counter()
    result = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        optimizer=optimizer,
        seed=seed,
        deterministic=True,
        cache=cache,
        plots=True,
        save=True,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    duration = time.perf_counter() - started
    return locate_best_weights(model, result), duration


def evaluate_model(
    weights: Path,
    data: Path,
    project: Path,
    name: str,
    imgsz: int,
    batch: int,
    device: str | int,
    workers: int,
    split: str,
) -> tuple[dict[str, float | None], float]:
    model = YOLO(str(weights))
    started = time.perf_counter()
    metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        split=split,
        plots=True,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    duration = time.perf_counter() - started
    return extract_metrics(metrics), duration


def command_train(args: argparse.Namespace) -> None:
    data = require_file(args.data, "Dataset YAML")
    project = prepare_project(args.project)
    device = resolve_device(args.device)
    name = safe_name(args.name or f"train_{Path(args.model).stem}_{timestamp()}")
    weights, train_seconds = train_model(
        model_reference=args.model,
        data=data,
        project=project,
        name=name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        seed=args.seed,
        cache=args.cache,
    )
    payload = {
        "mode": "train",
        "model": args.model,
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(device),
        "seed": args.seed,
        "train_seconds": round(train_seconds, 3),
        "best_weights": str(weights),
        "environment": environment_info(device),
    }
    write_json(project / name / "run_summary.json", payload)
    print(json.dumps(payload, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    weights = require_file(args.weights, "Weights")
    data = require_file(args.data, "Dataset YAML")
    project = prepare_project(args.project)
    device = resolve_device(args.device)
    name = safe_name(args.name or f"evaluate_{weights.stem}_{timestamp()}")
    metrics, validation_seconds = evaluate_model(
        weights=weights,
        data=data,
        project=project,
        name=name,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        split=args.split,
    )
    payload = {
        "mode": "evaluate",
        "weights": str(weights),
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(device),
        "validation_seconds": round(validation_seconds, 3),
        **metrics,
        "environment": environment_info(device),
    }
    write_json(project / name / "evaluation_summary.json", payload)
    print(json.dumps(payload, indent=2))


def command_experiment(args: argparse.Namespace) -> None:
    data = require_file(args.data, "Dataset YAML")
    project = prepare_project(args.project)
    device = resolve_device(args.device)
    experiment_dir = project / safe_name(args.name or f"research_{timestamp()}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "data": str(data),
        "models": args.models,
        "image_sizes": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "environment": environment_info(device),
    }
    write_json(experiment_dir / "experiment_config.json", config)
    rows: list[dict[str, Any]] = []
    for model_reference in args.models:
        for image_size in args.imgsz:
            run_name = safe_name(f"{Path(model_reference).stem}_img{image_size}")
            weights, train_seconds = train_model(
                model_reference=model_reference,
                data=data,
                project=experiment_dir,
                name=run_name,
                epochs=args.epochs,
                imgsz=image_size,
                batch=args.batch,
                device=device,
                workers=args.workers,
                patience=args.patience,
                optimizer=args.optimizer,
                seed=args.seed,
                cache=args.cache,
            )
            metrics, validation_seconds = evaluate_model(
                weights=weights,
                data=data,
                project=experiment_dir,
                name=f"{run_name}_validation",
                imgsz=image_size,
                batch=args.batch,
                device=device,
                workers=args.workers,
                split=args.split,
            )
            row = {
                "model": model_reference,
                "imgsz": image_size,
                "epochs": args.epochs,
                "batch": args.batch,
                "seed": args.seed,
                "device": str(device),
                "train_seconds": round(train_seconds, 3),
                "validation_seconds": round(validation_seconds, 3),
                "best_weights": str(weights),
                **metrics,
            }
            rows.append(row)
            write_json(experiment_dir / "experiment_results.json", rows)
            write_csv(experiment_dir / "experiment_results.csv", rows)
            print(json.dumps(row, indent=2))
    ranked = sorted(
        rows,
        key=lambda item: item["map50_95"] if item["map50_95"] is not None else -1.0,
        reverse=True,
    )
    write_json(experiment_dir / "experiment_ranking.json", ranked)
    print(f"Experiment results saved to {experiment_dir}")


def prediction_summary(results: list[Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}
        class_ids = boxes.cls.detach().cpu().tolist() if boxes is not None and boxes.cls is not None else []
        confidences = boxes.conf.detach().cpu().tolist() if boxes is not None and boxes.conf is not None else []
        classes: dict[str, int] = {}
        for class_id in class_ids:
            key = str(names.get(int(class_id), int(class_id)))
            classes[key] = classes.get(key, 0) + 1
        summary.append(
            {
                "image": str(getattr(result, "path", "unknown")),
                "detections": len(class_ids),
                "classes": classes,
                "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else None,
                "max_confidence": round(max(confidences), 6) if confidences else None,
            }
        )
    return summary


def command_predict(args: argparse.Namespace) -> None:
    weights = require_file(args.weights, "Weights")
    source = require_source(args.source)
    project = prepare_project(args.project)
    device = resolve_device(args.device)
    name = safe_name(args.name or f"predict_{timestamp()}")
    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    summary = prediction_summary(results)
    write_json(project / name / "prediction_summary.json", summary)
    print(json.dumps(summary, indent=2))


def command_segment(args: argparse.Namespace) -> None:
    weights = require_file(args.weights, "Weights")
    source = require_source(args.source)
    project = prepare_project(args.project)
    device = resolve_device(args.device)
    name = safe_name(args.name or f"segment_{timestamp()}")
    detection_dir = project / f"{name}_detections"
    segmentation_dir = project / f"{name}_segments"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(weights))
    detector_results = detector.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=True,
        project=str(project),
        name=detection_dir.name,
        exist_ok=True,
        verbose=False,
    )
    segmenter = SAM(args.sam_model)
    records: list[dict[str, Any]] = []
    for index, result in enumerate(detector_results):
        boxes = getattr(result, "boxes", None)
        box_tensor = getattr(boxes, "xyxy", None) if boxes is not None else None
        box_list = box_tensor.detach().cpu().tolist() if box_tensor is not None else []
        source_name = Path(str(getattr(result, "path", f"image_{index}.jpg"))).name
        output_name = f"{index:04d}_{source_name}"
        record = {
            "image": str(getattr(result, "path", source_name)),
            "detections": len(box_list),
            "segments": 0,
            "output": None,
        }
        if box_list:
            sam_results = segmenter.predict(
                source=result.orig_img,
                bboxes=box_list,
                device=device,
                verbose=False,
            )
            output_path = segmentation_dir / output_name
            if sam_results:
                sam_results[0].save(filename=str(output_path))
                masks = getattr(sam_results[0], "masks", None)
                mask_data = getattr(masks, "data", None) if masks is not None else None
                record["segments"] = int(len(mask_data)) if mask_data is not None else 0
                record["output"] = str(output_path)
        records.append(record)
    write_json(segmentation_dir / "segmentation_summary.json", records)
    print(json.dumps(records, indent=2))


def command_export(args: argparse.Namespace) -> None:
    weights = require_file(args.weights, "Weights")
    device = resolve_device(args.device)
    model = YOLO(str(weights))
    exported = model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=device,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    print(str(exported))


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    add_runtime_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tumor-detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--project", default="runs/tumor_detection")
    train_parser.add_argument("--model", default="yolo11n.pt")
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--name")
    add_training_arguments(train_parser)
    train_parser.set_defaults(handler=command_train)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--project", default="runs/tumor_detection")
    evaluate_parser.add_argument("--weights", required=True)
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--imgsz", type=int, default=640)
    evaluate_parser.add_argument("--batch", type=int, default=16)
    evaluate_parser.add_argument("--split", choices=("val", "test"), default="val")
    evaluate_parser.add_argument("--name")
    add_runtime_arguments(evaluate_parser)
    evaluate_parser.set_defaults(handler=command_evaluate)

    experiment_parser = subparsers.add_parser("experiment")
    experiment_parser.add_argument("--project", default="runs/tumor_detection")
    experiment_parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    experiment_parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    experiment_parser.add_argument("--split", choices=("val", "test"), default="val")
    experiment_parser.add_argument("--name")
    add_training_arguments(experiment_parser)
    experiment_parser.set_defaults(handler=command_experiment)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--project", default="runs/tumor_detection")
    predict_parser.add_argument("--weights", required=True)
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--imgsz", type=int, default=640)
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--iou", type=float, default=0.7)
    predict_parser.add_argument("--name")
    predict_parser.add_argument("--device", default="auto")
    predict_parser.set_defaults(handler=command_predict)

    segment_parser = subparsers.add_parser("segment")
    segment_parser.add_argument("--project", default="runs/tumor_detection")
    segment_parser.add_argument("--weights", required=True)
    segment_parser.add_argument("--source", required=True)
    segment_parser.add_argument("--sam-model", default=DEFAULT_SAM_MODEL)
    segment_parser.add_argument("--imgsz", type=int, default=640)
    segment_parser.add_argument("--conf", type=float, default=0.25)
    segment_parser.add_argument("--iou", type=float, default=0.7)
    segment_parser.add_argument("--name")
    segment_parser.add_argument("--device", default="auto")
    segment_parser.set_defaults(handler=command_segment)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--project", default="runs/tumor_detection")
    export_parser.add_argument("--weights", required=True)
    export_parser.add_argument("--format", default="onnx")
    export_parser.add_argument("--imgsz", type=int, default=640)
    export_parser.add_argument("--device", default="auto")
    export_parser.add_argument("--dynamic", action="store_true")
    export_parser.add_argument("--simplify", action="store_true")
    export_parser.set_defaults(handler=command_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.project = str(prepare_project(args.project))
    args.handler(args)


if __name__ == "__main__":
    main()
