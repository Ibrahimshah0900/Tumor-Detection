from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TUMOR_CLASSES = ("glioma", "meningioma", "pituitary")
CLASS_TO_ID = {name: index for index, name in enumerate(TUMOR_CLASSES)}
CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
    "no-tumor": "no_tumor",
}
PLANE_TOKENS = {"ax": "axial", "co": "coronal", "sa": "sagittal"}
DEFAULT_MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")
DEFAULT_SAM_MODEL = "sam2.1_b.pt"


def ml_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import ultralytics
        from ultralytics import SAM, YOLO
    except ImportError as exc:
        raise RuntimeError("Install torch and ultralytics for training and inference") from exc
    return torch, ultralytics, YOLO, SAM


def device(value: str) -> str | int:
    if value.lower() != "auto":
        return int(value) if value.isdigit() else value
    torch, _, _, _ = ml_stack()
    return 0 if torch.cuda.is_available() else "cpu"


def environment(selected: str | int) -> dict[str, Any]:
    torch, ultralytics, _, _ = ml_stack()
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(selected if isinstance(selected, int) else 0)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def seed_all(seed: int) -> None:
    torch, _, _, _ = ml_stack()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def path_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def path_dir(value: str, label: str, create: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise NotADirectoryError(f"{label} not found: {path}")
    return path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "run"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def image_files(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def class_name(value: str) -> str:
    key = value.strip().lower().replace(" ", "_")
    if key not in CLASS_ALIASES:
        raise ValueError(f"Unsupported BRISC class folder: {value}")
    return CLASS_ALIASES[key]


def plane(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return next((name for token, name in PLANE_TOKENS.items() if f"_{token}_" in stem), "unknown")


def class_directories(split_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for item in split_dir.iterdir():
        if item.is_dir():
            try:
                found[class_name(item.name)] = item
            except ValueError:
                pass
    required = (*TUMOR_CLASSES, "no_tumor")
    missing = [name for name in required if name not in found]
    if missing:
        raise FileNotFoundError(f"Missing BRISC classes in {split_dir}: {', '.join(missing)}")
    return found


def brisc_records(root: Path, split: str) -> list[dict[str, Any]]:
    classification = root / "classification_task" / split
    segmentation_images = root / "segmentation_task" / split / "images"
    segmentation_masks = root / "segmentation_task" / split / "masks"
    if not classification.is_dir() or not segmentation_images.is_dir() or not segmentation_masks.is_dir():
        raise FileNotFoundError(f"Incomplete BRISC '{split}' split under {root}")
    seg_images = {path.stem: path for path in image_files(segmentation_images)}
    seg_masks = {path.stem: path for path in image_files(segmentation_masks)}
    records: list[dict[str, Any]] = []
    for label, directory in class_directories(classification).items():
        for original in image_files(directory):
            if label == "no_tumor":
                image, mask = original, None
            else:
                image, mask = seg_images.get(original.stem), seg_masks.get(original.stem)
                if image is None or mask is None:
                    raise FileNotFoundError(f"Missing segmentation pair for {original.name}")
            records.append({"source_split": split, "class": label, "plane": plane(original.name), "image": image, "mask": mask})
    return records


def split_training(records: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["class"], record["plane"])].append(record)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group in groups.values():
        group = group.copy()
        rng.shuffle(group)
        count = max(1, round(len(group) * fraction)) if len(group) > 1 else 0
        validation.extend(group[:count])
        train.extend(group[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def mask_box(path: Path) -> tuple[float, float, float, float, int]:
    mask = binary_mask(path)
    ys, xs = np.where(mask)
    if not xs.size:
        raise ValueError(f"Empty tumor mask: {path}")
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height, int(mask.sum()))


def prepare_record(record: dict[str, Any], split: str, output: Path) -> dict[str, Any]:
    source = Path(record["image"])
    image = output / "images" / split / source.name
    label = output / "labels" / split / f"{source.stem}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, image)
    pixels = 0
    if record["class"] == "no_tumor":
        text = ""
    else:
        x, y, w, h, pixels = mask_box(Path(record["mask"]))
        text = f"{CLASS_TO_ID[record['class']]} {x:.8f} {y:.8f} {w:.8f} {h:.8f}"
    label.write_text(text, encoding="utf-8")
    return {
        "split": split,
        "source_split": record["source_split"],
        "class": record["class"],
        "plane": record["plane"],
        "image": str(image),
        "label": str(label),
        "mask": str(record["mask"]) if record["mask"] else "",
        "mask_pixels": pixels,
    }


def dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_images": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "class_counts": dict(sorted(Counter(f"{row['split']}:{row['class']}" for row in rows).items())),
        "plane_counts": dict(sorted(Counter(f"{row['split']}:{row['plane']}" for row in rows).items())),
        "negative_images": sum(row["class"] == "no_tumor" for row in rows),
        "tumor_images": sum(row["class"] != "no_tumor" for row in rows),
    }


def command_prepare(args: argparse.Namespace) -> None:
    root = path_dir(args.dataset_root, "BRISC dataset")
    output = path_dir(args.output, "Output", create=True)
    train, validation = split_training(brisc_records(root, "train"), args.validation_fraction, args.seed)
    test = brisc_records(root, "test")
    if args.clean:
        for folder in (output / "images", output / "labels"):
            if folder.exists():
                shutil.rmtree(folder)
    rows = [prepare_record(record, split, output) for split, records in (("train", train), ("val", validation), ("test", test)) for record in records]
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f'path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: glioma\n  1: meningioma\n  2: pituitary\n',
        encoding="utf-8",
    )
    summary = {
        "dataset": "BRISC 2025",
        "source": str(root),
        "output": str(output),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "detection_classes": list(TUMOR_CLASSES),
        "negative_policy": "no_tumor images use empty detection labels",
        "split_policy": "official test remains untouched; validation is drawn only from official training and stratified by class and plane",
        "limitation": "patient identifiers are unavailable, so patient-disjoint splitting cannot be verified",
        "data_yaml": str(data_yaml),
        **dataset_summary(rows),
    }
    write_csv(output / "dataset_manifest.csv", rows)
    write_json(output / "dataset_summary.json", summary)
    print(json.dumps(summary, indent=2))


def command_audit(args: argparse.Namespace) -> None:
    root = path_dir(args.dataset_root, "BRISC dataset")
    records = brisc_records(root, "train") + brisc_records(root, "test")
    rows: list[dict[str, Any]] = []
    for record in records:
        fraction = 0.0
        pixels = 0
        if record["mask"]:
            mask = binary_mask(Path(record["mask"]))
            pixels, fraction = int(mask.sum()), float(mask.mean())
        rows.append({"split": record["source_split"], "class": record["class"], "plane": record["plane"], "image": str(record["image"]), "mask_pixels": pixels, "mask_fraction": round(fraction, 8)})
    tumor_fractions = [row["mask_fraction"] for row in rows if row["class"] != "no_tumor"]
    summary = {
        "dataset": "BRISC 2025",
        "images": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "test": sum(row["split"] == "test" for row in rows),
        "class_counts": dict(sorted(Counter(row["class"] for row in rows).items())),
        "plane_counts": dict(sorted(Counter(row["plane"] for row in rows).items())),
        "mean_tumor_fraction": round(float(np.mean(tumor_fractions)), 8) if tumor_fractions else None,
        "median_tumor_fraction": round(float(np.median(tumor_fractions)), 8) if tumor_fractions else None,
        "empty_tumor_masks": sum(row["class"] != "no_tumor" and row["mask_pixels"] == 0 for row in rows),
    }
    if args.output:
        output = path_dir(args.output, "Output", create=True)
        write_csv(output / "brisc_audit.csv", rows)
        write_json(output / "brisc_audit.json", summary)
    print(json.dumps(summary, indent=2))


def value_float(value: Any) -> float | None:
    try:
        return float(value.item() if hasattr(value, "item") else value) if value is not None else None
    except (TypeError, ValueError):
        return None


def metrics_dict(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    return {
        "precision": value_float(getattr(box, "mp", None)),
        "recall": value_float(getattr(box, "mr", None)),
        "map50": value_float(getattr(box, "map50", None)),
        "map75": value_float(getattr(box, "map75", None)),
        "map50_95": value_float(getattr(box, "map", None)),
        "inference_ms": value_float(speed.get("inference")),
    }


def best_weights(model: Any, result: Any) -> Path:
    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None)
    if best and Path(best).is_file():
        return Path(best).resolve()
    save_dir = getattr(result, "save_dir", None) or getattr(trainer, "save_dir", None)
    candidate = Path(save_dir) / "weights" / "best.pt" if save_dir else None
    if candidate and candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError("Training completed but best.pt could not be located")


def train_once(model_name: str, data: Path, project: Path, name: str, args: argparse.Namespace, imgsz: int) -> tuple[Path, float]:
    seed_all(args.seed)
    _, _, YOLO, _ = ml_stack()
    model = YOLO(model_name)
    started = time.perf_counter()
    result = model.train(
        data=str(data), epochs=args.epochs, imgsz=imgsz, batch=args.batch, device=device(args.device), workers=args.workers,
        patience=args.patience, optimizer=args.optimizer, seed=args.seed, deterministic=True, cache=args.cache, plots=True,
        save=True, project=str(project), name=name, exist_ok=True,
    )
    return best_weights(model, result), time.perf_counter() - started


def evaluate_once(weights: Path, data: Path, project: Path, name: str, args: argparse.Namespace, imgsz: int, split: str) -> tuple[dict[str, float | None], float]:
    _, _, YOLO, _ = ml_stack()
    started = time.perf_counter()
    metrics = YOLO(str(weights)).val(
        data=str(data), imgsz=imgsz, batch=args.batch, device=device(args.device), workers=args.workers, split=split,
        plots=True, project=str(project), name=name, exist_ok=True,
    )
    return metrics_dict(metrics), time.perf_counter() - started


def command_train(args: argparse.Namespace) -> None:
    data = path_file(args.data, "Dataset YAML")
    project = path_dir(args.project, "Project", create=True)
    name = clean_name(args.name or f"train_{Path(args.model).stem}_{stamp()}")
    weights, seconds = train_once(args.model, data, project, name, args, args.imgsz)
    selected = device(args.device)
    result = {"mode": "train", "model": args.model, "data": str(data), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed, "device": str(selected), "train_seconds": round(seconds, 3), "best_weights": str(weights), "environment": environment(selected)}
    write_json(project / name / "run_summary.json", result)
    print(json.dumps(result, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    weights = path_file(args.weights, "Weights")
    data = path_file(args.data, "Dataset YAML")
    project = path_dir(args.project, "Project", create=True)
    name = clean_name(args.name or f"evaluate_{stamp()}")
    metrics, seconds = evaluate_once(weights, data, project, name, args, args.imgsz, args.split)
    selected = device(args.device)
    result = {"mode": "evaluate", "weights": str(weights), "split": args.split, "imgsz": args.imgsz, "validation_seconds": round(seconds, 3), **metrics, "environment": environment(selected)}
    write_json(project / name / "evaluation_summary.json", result)
    print(json.dumps(result, indent=2))


def command_experiment(args: argparse.Namespace) -> None:
    data = path_file(args.data, "Dataset YAML")
    project = path_dir(args.project, "Project", create=True) / clean_name(args.name or f"research_{stamp()}")
    project.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        for imgsz in args.imgsz:
            name = clean_name(f"{Path(model_name).stem}_img{imgsz}")
            weights, train_seconds = train_once(model_name, data, project, name, args, imgsz)
            metrics, val_seconds = evaluate_once(weights, data, project, f"{name}_validation", args, imgsz, args.split)
            row = {"model": model_name, "imgsz": imgsz, "epochs": args.epochs, "batch": args.batch, "seed": args.seed, "train_seconds": round(train_seconds, 3), "validation_seconds": round(val_seconds, 3), "best_weights": str(weights), **metrics}
            rows.append(row)
            write_csv(project / "experiment_results.csv", rows)
            write_json(project / "experiment_results.json", rows)
            print(json.dumps(row, indent=2))
    ranking = sorted(rows, key=lambda row: row["map50_95"] if row["map50_95"] is not None else -1, reverse=True)
    write_json(project / "experiment_ranking.json", ranking)
    write_json(project / "experiment_config.json", {"data": str(data), "models": args.models, "image_sizes": args.imgsz, "split": args.split, "seed": args.seed})


def prediction_rows(results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        ids = boxes.cls.detach().cpu().tolist() if boxes is not None and boxes.cls is not None else []
        conf = boxes.conf.detach().cpu().tolist() if boxes is not None and boxes.conf is not None else []
        names = getattr(result, "names", {}) or {}
        classes = Counter(str(names.get(int(value), int(value))) for value in ids)
        rows.append({"image": str(getattr(result, "path", "unknown")), "detections": len(ids), "classes": dict(classes), "mean_confidence": round(float(np.mean(conf)), 6) if conf else None, "max_confidence": round(max(conf), 6) if conf else None})
    return rows


def command_predict(args: argparse.Namespace) -> None:
    weights = path_file(args.weights, "Weights")
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    project = path_dir(args.project, "Project", create=True)
    name = clean_name(args.name or f"predict_{stamp()}")
    _, _, YOLO, _ = ml_stack()
    results = YOLO(str(weights)).predict(source=str(source), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device(args.device), save=True, save_txt=True, save_conf=True, project=str(project), name=name, exist_ok=True, verbose=False)
    rows = prediction_rows(results)
    write_json(project / name / "prediction_summary.json", rows)
    print(json.dumps(rows, indent=2))


def combined_mask(results: list[Any], shape: tuple[int, int]) -> np.ndarray:
    if not results:
        return np.zeros(shape, dtype=bool)
    masks = getattr(results[0], "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None or len(data) == 0:
        return np.zeros(shape, dtype=bool)
    predicted = np.any(data.detach().cpu().numpy() > 0.5, axis=0)
    if predicted.shape != shape:
        predicted = np.asarray(Image.fromarray(predicted.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
    return predicted


def segmentation_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted, truth = predicted.astype(bool), truth.astype(bool)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def class_lookup(root: Path, split: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for label, directory in class_directories(root / "classification_task" / split).items():
        for image in image_files(directory):
            lookup[image.stem] = label
    return lookup


def command_benchmark(args: argparse.Namespace) -> None:
    weights = path_file(args.weights, "Weights")
    root = path_dir(args.dataset_root, "BRISC dataset")
    project = path_dir(args.project, "Project", create=True)
    images = root / "segmentation_task" / args.split / "images"
    masks = root / "segmentation_task" / args.split / "masks"
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Missing BRISC segmentation split: {args.split}")
    lookup = class_lookup(root, args.split)
    _, _, YOLO, SAM = ml_stack()
    detector, segmenter = YOLO(str(weights)), SAM(args.sam_model)
    selected = device(args.device)
    samples = image_files(images)
    samples = samples[: args.limit] if args.limit else samples
    run = project / clean_name(args.name or f"segmentation_benchmark_{args.split}_{stamp()}")
    run.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, image in enumerate(samples, 1):
        mask = masks / f"{image.stem}.png"
        if not mask.is_file():
            candidates = list(masks.glob(f"{image.stem}.*"))
            if not candidates:
                raise FileNotFoundError(f"Mask not found for {image.name}")
            mask = candidates[0]
        truth = binary_mask(mask)
        result = detector.predict(source=str(image), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=selected, verbose=False)[0]
        boxes = result.boxes.xyxy.detach().cpu().tolist() if getattr(result, "boxes", None) is not None and result.boxes.xyxy is not None else []
        predicted = combined_mask(segmenter.predict(source=result.orig_img, bboxes=boxes, device=selected, verbose=False), truth.shape) if boxes else np.zeros(truth.shape, dtype=bool)
        metrics = segmentation_metrics(predicted, truth)
        rows.append({"image": image.name, "class": lookup.get(image.stem, "unknown"), "plane": plane(image.name), "detections": len(boxes), **{key: round(value, 8) for key, value in metrics.items()}})
        if index % max(1, args.checkpoint_every) == 0:
            write_csv(run / "segmentation_metrics.csv", rows)
            write_json(run / "segmentation_metrics.json", rows)
    write_csv(run / "segmentation_metrics.csv", rows)
    write_json(run / "segmentation_metrics.json", rows)
    per_class = {}
    for label in sorted({row["class"] for row in rows}):
        group = [row for row in rows if row["class"] == label]
        per_class[label] = {"count": len(group), "mean_iou": float(np.mean([row["iou"] for row in group])), "mean_dice": float(np.mean([row["dice"] for row in group]))}
    summary = {
        "mode": "segmentation_benchmark",
        "split": args.split,
        "samples": len(rows),
        "detection_coverage": float(np.mean([row["detections"] > 0 for row in rows])) if rows else None,
        "mean_iou": float(np.mean([row["iou"] for row in rows])) if rows else None,
        "median_iou": float(np.median([row["iou"] for row in rows])) if rows else None,
        "mean_dice": float(np.mean([row["dice"] for row in rows])) if rows else None,
        "median_dice": float(np.median([row["dice"] for row in rows])) if rows else None,
        "mean_pixel_precision": float(np.mean([row["pixel_precision"] for row in rows])) if rows else None,
        "mean_pixel_recall": float(np.mean([row["pixel_recall"] for row in rows])) if rows else None,
        "per_class": per_class,
        "weights": str(weights),
        "sam_model": args.sam_model,
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "seconds": round(time.perf_counter() - started, 3),
        "environment": environment(selected),
    }
    write_json(run / "segmentation_summary.json", summary)
    print(json.dumps(summary, indent=2))


def add_training(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="tumor-detection")
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-brisc")
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--output", default="datasets/brisc_yolo")
    prepare.add_argument("--validation-fraction", type=float, default=0.15)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--clean", action="store_true")
    prepare.set_defaults(handler=command_prepare)

    audit = commands.add_parser("audit-brisc")
    audit.add_argument("--dataset-root", required=True)
    audit.add_argument("--output")
    audit.set_defaults(handler=command_audit)

    train = commands.add_parser("train")
    train.add_argument("--project", default="runs/tumor_detection")
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--name")
    add_training(train)
    train.set_defaults(handler=command_train)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--project", default="runs/tumor_detection")
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=16)
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--name")
    evaluate.set_defaults(handler=command_evaluate)

    experiment = commands.add_parser("experiment")
    experiment.add_argument("--project", default="runs/tumor_detection")
    experiment.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    experiment.add_argument("--imgsz", nargs="+", type=int, default=[512, 640])
    experiment.add_argument("--split", choices=("val", "test"), default="val")
    experiment.add_argument("--name")
    add_training(experiment)
    experiment.set_defaults(handler=command_experiment)

    predict = commands.add_parser("predict")
    predict.add_argument("--project", default="runs/tumor_detection")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.7)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--name")
    predict.set_defaults(handler=command_predict)

    benchmark = commands.add_parser("benchmark-segmentation")
    benchmark.add_argument("--project", default="runs/tumor_detection")
    benchmark.add_argument("--weights", required=True)
    benchmark.add_argument("--dataset-root", required=True)
    benchmark.add_argument("--sam-model", default=DEFAULT_SAM_MODEL)
    benchmark.add_argument("--split", choices=("train", "test"), default="test")
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--conf", type=float, default=0.25)
    benchmark.add_argument("--iou", type=float, default=0.7)
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--checkpoint-every", type=int, default=50)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--name")
    benchmark.set_defaults(handler=command_benchmark)

    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
