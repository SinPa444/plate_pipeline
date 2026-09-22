#!/usr/bin/env python3
"""Stage 4 — کلاسیفایر کاراکتر (ResNet18، ۳۴ کلاس، قفل‌شده).

فقط استنتاج: هیچ تغییری در معماری، وزن یا ورودی 224×224 اعمال نمی‌شود.
پیش‌پردازش (contract قفل‌شده): Resize(224,224) + ToTensor + Normalize(ImageNet)
ورودی کرپ‌ها: BGR (خروجی cv2) → داخلی به RGB تبدیل می‌شود.
"""
from __future__ import annotations

import importlib.util
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CharClassifier:
    def __init__(self, ckpt="checkpoints/resnet18_char34_v1.pth",
                 arch_py="src/models/resnet_18.py",
                 device=None, batch_size=64):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load(ckpt, arch_py).to(self.device).eval()
        self.batch_size = batch_size
        self.pre = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @staticmethod
    def _load(ckpt, arch_py):
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        spec = importlib.util.spec_from_file_location("resnet18_arch", arch_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        makers = []
        # توابع کارخانه مثل ResNet18(num_classes) — همان الگوی baseline
        for name in dir(mod):
            if "resnet18" in name.lower() or "resnet_18" in name.lower():
                obj = getattr(mod, name)
                if callable(obj) and not isinstance(obj, type):
                    makers.append(obj)
        # کلاس‌ها به‌عنوان fallback (اگر روزی فایل معماری عوض شد)
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj.__module__ == mod.__name__:
                makers.append(obj)

        for maker in makers:
            for make in (lambda m=maker: m(34),
                         lambda m=maker: m(num_classes=34)):
                try:
                    m = make()
                except TypeError:
                    continue
                try:
                    m.load_state_dict(sd, strict=True)
                    return m
                except Exception:
                    continue
        raise RuntimeError(
            "بارگذاری ResNet18 ناموفق بود. خروجی این دستور را بفرست:\n"
            f"  grep -n 'class \|def ' {arch_py}"
        )

    @torch.no_grad()
    def classify(self, crops_bgr: Sequence[np.ndarray]):
        """کرپ‌های BGR → لیست (class_id, conf) به همان ترتیب ورودی."""
        if not crops_bgr:
            return []
        out = []
        for i in range(0, len(crops_bgr), self.batch_size):
            batch = [self.pre(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                     for c in crops_bgr[i:i + self.batch_size]]
            x = torch.stack(batch).to(self.device)
            probs = self.model(x).softmax(dim=1)
            conf, ids = probs.max(dim=1)
            out += [(int(c), float(f)) for c, f in zip(ids.tolist(), conf.tolist())]
        return out

    def classify_boxes(self, img_bgr: np.ndarray, boxes_xyxy, pad: float = 0.10):
        """باکس‌های [x1,y1,x2,y2] پیکسلی روی یک تصویر → (class_id, conf) به همان ترتیب."""
        H, W = img_bgr.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes_xyxy:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
            xa = int(max(0, round(x1 - pw))); ya = int(max(0, round(y1 - ph)))
            xb = int(min(W - 1, round(x2 + pw))); yb = int(min(H - 1, round(y2 + ph)))
            if xb <= xa or yb <= ya:
                xb, yb = min(xa + 1, W - 1), min(ya + 1, H - 1)
            crops.append(img_bgr[ya:yb + 1, xa:xb + 1])
        return self.classify(crops)