#!/usr/bin/env python3
"""End-to-end Iranian ALPR demo for images, folders, and videos.

Pipeline:
  source -> optional COCO vehicle detector -> plate detector -> character detector
  -> existing ResNet18 classifier -> ordering/parsing -> annotated media + JSON/CSV

Run this file from the ALPR project root.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Make imports work when launched from the project root or another directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.char_classifier import CharClassifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

# Classes used on ordinary Iranian plates in the existing project.
ALLOWED_LETTERS = {11, 15, 16, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}

# Province membership is deliberately a soft diagnostic, not a hard rejection.
# The historical list in the project is incomplete for some observed plates.
KNOWN_PROVINCE_CODES = {
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    21, 22, 23, 24, 25, 26, 27, 28, 29,
    31, 32, 33, 34, 35, 36, 37, 38, 39,
    41, 42, 43, 44, 45, 46, 47, 48, 49,
    51, 52, 53, 54, 55, 56, 57, 58, 59,
    61, 62, 63, 64, 65, 66, 67, 68, 69,
    71, 72, 73, 74, 75, 76, 77, 78, 79,
    81, 82, 83, 84, 87, 88,
}


@dataclass
class Reading:
    source: str
    frame_index: int
    vehicle_index: int
    plate_index: int
    vehicle_box: list[int]
    plate_box: list[int]
    character_boxes: list[list[int]]
    class_ids: list[int]
    confidences: list[float]
    plate_ascii: str
    plate_fa: str
    grammar_valid: bool
    province_known: bool
    detection_complete: bool
    vehicle_confidence: float
    plate_confidence: float
    character_detection_confidence: float
    character_classification_confidence: float


class ALPRPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = args.device or ("0" if torch.cuda.is_available() else "cpu")
        self.device_text = str(self.device)

        self.classes = self._load_classes(Path(args.classes))
        self.ids = [str(item.get("id", item["index"])) for item in self.classes]
        self.fa = [str(item.get("fa", item.get("id", item["index"]))) for item in self.classes]

        print(f"Device: {self.device_text}")
        self.vehicle_model = None
        if not args.no_vehicle:
            print(f"Loading vehicle detector: {args.vehicle_model}")
            self.vehicle_model = YOLO(args.vehicle_model)

        print(f"Loading plate detector: {args.plate_model}")
        self.plate_model = YOLO(args.plate_model)
        print(f"Loading character detector: {args.char_model}")
        self.char_model = YOLO(args.char_model)
        print(f"Loading ResNet18 classifier: {args.classifier_model}")
        self.classifier = CharClassifier(
            ckpt=args.classifier_model,
            arch_py=args.classifier_arch,
            device=("cuda" if self.device_text != "cpu" and torch.cuda.is_available() else "cpu"),
            batch_size=args.classifier_batch,
        )

    @staticmethod
    def _load_classes(path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        data = sorted(data, key=lambda item: int(item["index"]))
        if len(data) != 34 or [int(x["index"]) for x in data] != list(range(34)):
            raise ValueError(f"Expected exactly 34 ordered classes in {path}")
        return data

    @staticmethod
    def _clip_box(box: Iterable[float], width: int, height: int) -> list[int]:
        x1, y1, x2, y2 = box
        return [
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(1, min(width, int(round(x2)))),
            max(1, min(height, int(round(y2)))),
        ]

    @staticmethod
    def _expand_box(box: list[int], width: int, height: int, margin: float) -> list[int]:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        return [
            max(0, int(round(x1 - bw * margin))),
            max(0, int(round(y1 - bh * margin))),
            min(width, int(round(x2 + bw * margin))),
            min(height, int(round(y2 + bh * margin))),
        ]

    @staticmethod
    def _inside_iran(char_box: list[float], iran_box: list[float]) -> bool:
        cx = (char_box[0] + char_box[2]) / 2
        cy = (char_box[1] + char_box[3]) / 2
        return iran_box[0] <= cx <= iran_box[2] and iran_box[1] <= cy <= iran_box[3]

    def _vehicle_regions(self, image: np.ndarray) -> list[tuple[list[int], float]]:
        height, width = image.shape[:2]
        if self.vehicle_model is None:
            return [([0, 0, width, height], 1.0)]

        result = self.vehicle_model.predict(
            image,
            imgsz=self.args.vehicle_imgsz,
            conf=self.args.vehicle_conf,
            classes=VEHICLE_CLASSES,
            device=self.device,
            verbose=False,
        )[0]
        regions = []
        for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            clipped = self._clip_box(box, width, height)
            expanded = self._expand_box(clipped, width, height, self.args.vehicle_margin)
            if expanded[2] - expanded[0] >= 24 and expanded[3] - expanded[1] >= 24:
                regions.append((expanded, float(conf)))

        # For car-crop demos, allow direct plate detection when COCO finds no vehicle.
        if not regions and self.args.vehicle_fallback:
            regions.append(([0, 0, width, height], 0.0))
        return regions

    def _plate_regions(
        self, image: np.ndarray, vehicle_regions: list[tuple[list[int], float]]
    ) -> list[tuple[int, list[int], list[int], float, float]]:
        if not vehicle_regions:
            return []
        crops = []
        valid_regions = []
        for vehicle_box, vehicle_conf in vehicle_regions:
            x1, y1, x2, y2 = vehicle_box
            crop = image[y1:y2, x1:x2]
            if crop.size:
                crops.append(crop)
                valid_regions.append((vehicle_box, vehicle_conf))
        if not crops:
            return []

        results = self.plate_model.predict(
            crops,
            imgsz=self.args.plate_imgsz,
            conf=self.args.plate_conf,
            device=self.device,
            verbose=False,
        )
        height, width = image.shape[:2]
        found = []
        for vehicle_index, ((vehicle_box, vehicle_conf), result) in enumerate(zip(valid_regions, results)):
            vx1, vy1, _, _ = vehicle_box
            for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                global_box = [box[0] + vx1, box[1] + vy1, box[2] + vx1, box[3] + vy1]
                global_box = self._clip_box(global_box, width, height)
                expanded = self._expand_box(global_box, width, height, self.args.plate_margin)
                if expanded[2] - expanded[0] >= 20 and expanded[3] - expanded[1] >= 8:
                    found.append((vehicle_index, vehicle_box, expanded, float(vehicle_conf), float(conf)))
        return found

    def _read_plate(
        self,
        plate_crop: np.ndarray,
    ) -> tuple[list[list[int]], list[int], list[float], float, bool]:
        result = self.char_model.predict(
            plate_crop,
            imgsz=self.args.char_imgsz,
            conf=min(self.args.char_conf, self.args.iran_conf),
            device=self.device,
            verbose=False,
        )[0]

        chars: list[tuple[list[int], float]] = []
        irans: list[tuple[list[int], float]] = []
        h, w = plate_crop.shape[:2]
        for box, cls_id, conf in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            item = (self._clip_box(box, w, h), float(conf))
            if int(cls_id) == 0 and conf >= self.args.char_conf:
                chars.append(item)
            elif int(cls_id) == 1 and conf >= self.args.iran_conf:
                irans.append(item)

        if irans:
            iran_box, _ = max(irans, key=lambda item: item[1])
            chars = [(box, conf) for box, conf in chars if not self._inside_iran(box, iran_box)]

        # Keep the eight strongest candidates if the detector emits extras, then restore x order.
        if len(chars) > 8:
            chars = sorted(chars, key=lambda item: item[1], reverse=True)[:8]
        chars.sort(key=lambda item: (item[0][0] + item[0][2]) / 2)

        detection_complete = len(chars) == 8
        if not chars:
            return [], [], [], 0.0, False

        boxes = [item[0] for item in chars]
        det_conf = float(np.mean([item[1] for item in chars]))
        classified = self.classifier.classify_boxes(plate_crop, boxes, pad=0.0)
        class_ids = [class_id for class_id, _ in classified]
        class_conf = [float(conf) for _, conf in classified]
        return boxes, class_ids, class_conf, det_conf, detection_complete

    def _parse(self, class_ids: list[int]) -> tuple[str, str, bool, bool]:
        ascii_items = [self.ids[x] if 0 <= x < len(self.ids) else "?" for x in class_ids]
        fa_items = [self.fa[x] if 0 <= x < len(self.fa) else "؟" for x in class_ids]

        if len(class_ids) == 8:
            plate_ascii = f"{''.join(ascii_items[0:2])} {ascii_items[2]} {''.join(ascii_items[3:6])} | {''.join(ascii_items[6:8])}"
            plate_fa = f"{''.join(fa_items[0:2])} {fa_items[2]} {''.join(fa_items[3:6])} ایران {''.join(fa_items[6:8])}"
            structural = (
                all(x < 10 for x in class_ids[0:2])
                and class_ids[2] in ALLOWED_LETTERS
                and all(x < 10 for x in class_ids[3:6])
                and all(x < 10 for x in class_ids[6:8])
            )
            province = class_ids[6] * 10 + class_ids[7] if all(x < 10 for x in class_ids[6:8]) else -1
            return plate_ascii, plate_fa, structural, province in KNOWN_PROVINCE_CODES

        raw_ascii = " ".join(ascii_items) if ascii_items else "unreadable"
        raw_fa = " ".join(fa_items) if fa_items else "خوانده نشد"
        return raw_ascii, raw_fa, False, False

    def process_frame(self, image: np.ndarray, source: str, frame_index: int = 0):
        clean = image.copy()
        annotated = image.copy()
        vehicle_regions = self._vehicle_regions(clean)
        plate_regions = self._plate_regions(clean, vehicle_regions)
        readings: list[Reading] = []

        # In --no-vehicle mode the full image is only a synthetic processing
        # region, not a detected vehicle; do not draw a misleading 1.00 box.
        if not self.args.no_vehicle:
            for vehicle_index, (vehicle_box, vehicle_conf) in enumerate(vehicle_regions):
                x1, y1, x2, y2 = vehicle_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (70, 200, 70), 2)
                cv2.putText(annotated, f"vehicle {vehicle_conf:.2f}", (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 200, 70), 2, cv2.LINE_AA)

        per_vehicle_plate_index: dict[int, int] = {}
        for detected_vehicle_index, vehicle_box, plate_box, vehicle_conf, plate_conf in plate_regions:
            plate_index = per_vehicle_plate_index.get(detected_vehicle_index, 0)
            per_vehicle_plate_index[detected_vehicle_index] = plate_index + 1
            px1, py1, px2, py2 = plate_box
            plate_crop = clean[py1:py2, px1:px2].copy()
            if plate_crop.size == 0:
                continue

            char_boxes, class_ids, class_conf, char_det_conf, complete = self._read_plate(plate_crop)
            plate_ascii, plate_fa, grammar_valid, province_known = self._parse(class_ids)

            global_char_boxes = []
            for box in char_boxes:
                gx1, gy1, gx2, gy2 = box[0] + px1, box[1] + py1, box[2] + px1, box[3] + py1
                global_char_boxes.append([gx1, gy1, gx2, gy2])
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 180, 30), 1)

            color = (40, 210, 40) if grammar_valid else (0, 170, 255)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)
            label = f"{plate_ascii}  p:{plate_conf:.2f}"
            label_y = max(24, py1 - 9)
            cv2.putText(annotated, label, (px1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)

            readings.append(Reading(
                source=source,
                frame_index=frame_index,
                vehicle_index=detected_vehicle_index,
                plate_index=plate_index,
                vehicle_box=vehicle_box,
                plate_box=plate_box,
                character_boxes=global_char_boxes,
                class_ids=class_ids,
                confidences=class_conf,
                plate_ascii=plate_ascii,
                plate_fa=plate_fa,
                grammar_valid=grammar_valid,
                province_known=province_known,
                detection_complete=complete,
                vehicle_confidence=float(vehicle_conf),
                plate_confidence=float(plate_conf),
                character_detection_confidence=char_det_conf,
                character_classification_confidence=float(np.mean(class_conf)) if class_conf else 0.0,
            ))

        return annotated, readings


def find_sources(source: Path) -> tuple[str, list[Path]]:
    if source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        return "images", images
    if source.suffix.lower() in IMAGE_EXTENSIONS:
        return "images", [source]
    if source.suffix.lower() in VIDEO_EXTENSIONS:
        return "video", [source]
    raise SystemExit(f"Unsupported source: {source}")


def save_reports(output: Path, rows: list[Reading], elapsed: float) -> None:
    payload = {
        "generated_at_unix": time.time(),
        "elapsed_seconds": elapsed,
        "detections": [asdict(row) for row in rows],
    }
    (output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source", "frame_index", "vehicle_index", "plate_index", "plate_ascii", "plate_fa",
            "grammar_valid", "province_known", "detection_complete", "vehicle_confidence",
            "plate_confidence", "character_detection_confidence", "character_classification_confidence",
        ])
        for row in rows:
            writer.writerow([
                row.source, row.frame_index, row.vehicle_index, row.plate_index,
                row.plate_ascii, row.plate_fa, row.grammar_valid, row.province_known,
                row.detection_complete, f"{row.vehicle_confidence:.4f}", f"{row.plate_confidence:.4f}",
                f"{row.character_detection_confidence:.4f}",
                f"{row.character_classification_confidence:.4f}",
            ])


def process_images(pipeline: ALPRPipeline, sources: list[Path], output: Path) -> list[Reading]:
    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    rows: list[Reading] = []
    for index, path in enumerate(sources, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] unreadable image: {path}")
            continue
        annotated, found = pipeline.process_frame(image, str(path), 0)
        rows.extend(found)
        cv2.imwrite(str(image_output / path.name), annotated)
        print(f"[{index}/{len(sources)}] {path.name}: {len(found)} plate(s)")
        for item in found:
            print(f"    {item.plate_ascii} | grammar={item.grammar_valid} | conf={item.character_classification_confidence:.2f}")
    return rows


def process_video(pipeline: ALPRPipeline, source: Path, output: Path) -> list[Reading]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = output / f"{source.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {output_path}")

    rows: list[Reading] = []
    frame_index = 0
    last_annotated: np.ndarray | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % pipeline.args.frame_stride == 0:
            last_annotated, found = pipeline.process_frame(frame, str(source), frame_index)
            rows.extend(found)
        else:
            # Keep the original frame on skipped frames; do not reuse stale coordinates.
            last_annotated = frame
        writer.write(last_annotated)
        frame_index += 1
        if frame_index % 50 == 0:
            print(f"Video: {frame_index}/{total if total > 0 else '?'} frames, readings={len(rows)}")

    capture.release()
    writer.release()
    print(f"Video output: {output_path}")
    return rows


def existing_default(*candidates: str) -> str:
    for item in candidates:
        if (PROJECT_ROOT / item).exists():
            return item
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iranian ALPR end-to-end demo")
    parser.add_argument("source", help="Image, image directory, or video")
    parser.add_argument("--output", default="outputs/alpr_demo")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")

    parser.add_argument("--vehicle-model", default="yolo11s.pt")
    parser.add_argument("--plate-model", default="checkpoints/plate_yolo11s_v1.pt")
    parser.add_argument("--char-model", default="checkpoints/char_yolo11s_ft_v3.pt")
    parser.add_argument("--classifier-model", default="checkpoints/resnet18_char34_v1.pth")
    parser.add_argument("--classifier-arch", default="src/models/resnet_18.py")
    parser.add_argument("--classes", default="configs/char_classes.json")

    parser.add_argument("--no-vehicle", action="store_true",
                        help="Run the plate detector directly on the input (useful for car crops)")
    parser.add_argument("--no-vehicle-fallback", dest="vehicle_fallback", action="store_false",
                        help="Do not try the full image when COCO finds no vehicle")
    parser.set_defaults(vehicle_fallback=True)

    parser.add_argument("--vehicle-conf", type=float, default=0.30)
    parser.add_argument("--plate-conf", type=float, default=0.25)
    parser.add_argument("--char-conf", type=float, default=0.20)
    parser.add_argument("--iran-conf", type=float, default=0.08)
    parser.add_argument("--vehicle-margin", type=float, default=0.10)
    parser.add_argument("--plate-margin", type=float, default=0.04)
    parser.add_argument("--vehicle-imgsz", type=int, default=640)
    parser.add_argument("--plate-imgsz", type=int, default=960)
    parser.add_argument("--char-imgsz", type=int, default=640)
    parser.add_argument("--classifier-batch", type=int, default=64)
    parser.add_argument("--frame-stride", type=int, default=3,
                        help="Process every Nth video frame (default: 3)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.source = str(Path(args.source).expanduser())
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")

    # Resolve project-relative model/config paths while still accepting absolute paths.
    for name in ("vehicle_model", "plate_model", "char_model", "classifier_model", "classifier_arch", "classes"):
        value = Path(getattr(args, name)).expanduser()
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        setattr(args, name, str(value))

    source = Path(args.source).resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    kind, sources = find_sources(source)
    if not sources:
        raise SystemExit(f"No supported media found under: {source}")

    start = time.time()
    pipeline = ALPRPipeline(args)
    if kind == "video":
        rows = process_video(pipeline, sources[0], output)
    else:
        rows = process_images(pipeline, sources, output)
    elapsed = time.time() - start
    save_reports(output, rows, elapsed)
    print(f"Done: {len(rows)} reading(s) in {elapsed:.1f}s")
    print(f"Reports: {output / 'results.json'} and {output / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
