
# Usage:
#   python infer_vit_timm_wrapper.py \
#       --model_id CristianR8/vit_large2-model \
#       --input /path/to/image_or_dir_or_list.txt \
#       --out results.csv --batch_size 16 --topk 3
#
# Accepted inputs:
#   - Single image file (jpg/png/...) 
#   - Directory with images (recursive)
#   - A .txt or .csv file with one path per line (first column for .csv)

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def discover_images(p: Path) -> List[Path]:
    if p.is_file():
        if p.suffix.lower() in IMG_EXTS:
            return [p]
        # list file: .txt or .csv
        if p.suffix.lower() in {".txt", ".csv"}:
            out = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # for CSV, take first column
                    line = line.split(",")[0]
                    q = Path(line)
                    if q.suffix.lower() in IMG_EXTS and q.exists():
                        out.append(q)
            return out
        # single non-image file: ignore
        return []
    # directory: recursive glob
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(p.rglob(f"*{ext}"))
    return sorted(imgs)

def load_label_maps(cfg: AutoConfig) -> Tuple[List[str], dict, dict]:
    # Prefer id2label/label2id if present. If missing (common in timm_wrapper),
    # build from cfg.label_names in order.
    id2label = getattr(cfg, "id2label", None)
    label2id = getattr(cfg, "label2id", None)

    if id2label and label2id and len(id2label) == len(label2id) and len(id2label) > 0:
        # Ensure id keys are int
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {k: int(v) for k, v in label2id.items()}
        labels = [id2label[i] for i in range(len(id2label))]
        return labels, id2label, label2id

    label_names = getattr(cfg, "label_names", None)
    if label_names and len(label_names) > 0:
        labels = list(label_names)
        id2label = {i: lab for i, lab in enumerate(labels)}
        label2id = {lab: i for i, lab in enumerate(labels)}
        return labels, id2label, label2id

    # Fallback: numeric labels
    num_classes = getattr(cfg, "num_labels", None) or getattr(cfg, "num_classes", None) or 1
    labels = [f"class_{i}" for i in range(int(num_classes))]
    id2label = {i: labels[i] for i in range(len(labels))}
    label2id = {labels[i]: i for i in range(len(labels))}
    return labels, id2label, label2id

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="outputs/outputs_convnext_xlarge", help="HF model repo or local path")
    parser.add_argument("--input", required=True, help="Image file, directory, or a .txt/.csv list")
    parser.add_argument("--out", default="inference_results.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load processor & model (timm_wrapper friendly)
    processor = AutoImageProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_id)
    model = AutoModelForImageClassification.from_pretrained(args.model_id, trust_remote_code=True, config=config)
    model.eval().to(args.device)

    labels, id2label, _ = load_label_maps(config)
    topk = max(1, min(args.topk, len(labels)))

    # Gather images
    paths = discover_images(Path(args.input))
    if not paths:
        raise SystemExit(f"No images found under: {args.input}")

    def load_rgb(p: Path) -> Image.Image:
        # Ensure consistent 3-channel input
        return Image.open(p).convert("RGB")

    # Batched inference
    rows = []
    batch: List[Tuple[Path, Image.Image]] = []
    for p in paths:
        batch.append((p, load_rgb(p)))
        if len(batch) == args.batch_size:
            rows.extend(run_batch(batch, processor, model, labels, topk, args.device))
            batch = []
    if batch:
        rows.extend(run_batch(batch, processor, model, labels, topk, args.device))

    # Write CSV: path, predicted_label, topk_labels, topk_probs, then probs per label
    prob_headers = [f"prob_{lab}" for lab in labels]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "pred", "topk_labels", "topk_probs"] + prob_headers)
        for r in rows:
            writer.writerow(r)

    print(f"Saved {len(rows)} predictions to {args.out}")

def run_batch(
    batch: List[Tuple[Path, Image.Image]],
    processor: AutoImageProcessor,
    model: AutoModelForImageClassification,
    labels: List[str],
    topk: int,
    device: str,
):
    imgs = [im for _, im in batch]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)

    top_probs, top_idx = torch.topk(probs, k=topk, dim=-1)

    out_rows = []
    for (p, _), prob_vec, tprob, tidx in zip(batch, probs, top_probs, top_idx):
        pred_idx = int(torch.argmax(prob_vec).item())
        pred_label = labels[pred_idx]
        # per-label probabilities (ordered as labels list)
        per_label = prob_vec.detach().cpu().tolist()
        top_labels = [labels[int(i)] for i in tidx.detach().cpu().tolist()]
        top_probs_list = tprob.detach().cpu().tolist()
        out_rows.append([
            str(p),
            pred_label,
            "|".join(top_labels),
            "|".join(f"{x:.6f}" for x in top_probs_list),
            *[f"{x:.6f}" for x in per_label]
        ])
    return out_rows

if __name__ == "__main__":
    main()
