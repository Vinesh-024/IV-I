"""
Phase 2 — Formal Evaluation Framework
======================================
Runs held-out test evaluation for all three classifiers:
  1. ModalityRouter CNN  (binary: RGB vs Scan)
  2. BiomedCLIP Linear Probe  (9-class RGB disease)
  3. DINOv2 Linear Probe  (6-class Scan pathology)

Usage:
    cd backend
    python -m app.scripts.evaluate [--split 0.2] [--seed 42] [--output ./evaluation_results]

Outputs (saved to --output directory):
  - modality_metrics.json      Accuracy, precision, recall, F1 (binary)
  - biomed_clip_metrics.json   Per-class precision, recall, F1, macro averages
  - dinov2_metrics.json        Per-class precision, recall, F1, macro averages
  - confusion_modality.png     Confusion matrix heatmap
  - confusion_biomed_clip.png  Confusion matrix heatmap (9 classes)
  - confusion_dinov2.png       Confusion matrix heatmap (6 classes)
  - evaluation_summary.txt     Human-readable full report
"""

import os
import sys
import json
import argparse
import random
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

# ── Try to import optional visualisation deps ───────────────────────────────
try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False
    logger.error("NumPy is required. pip install numpy")
    sys.exit(1)

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logger.error("PyTorch is required. pip install torch")
    sys.exit(1)

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

MPL_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")   # headless backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_OK = True
except ImportError:
    logger.warning("matplotlib not installed — confusion matrix PNGs will be skipped. pip install matplotlib")


# ── Path helpers ─────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BACKEND_DIR, "app", "data", "supervised")
MODELS_DIR  = os.path.join(BACKEND_DIR, "models")
RGB_DIR     = os.path.join(DATA_DIR, "RGB")
SCAN_DIR    = os.path.join(DATA_DIR, "Scan")
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_class_images(base_dir: str):
    """Returns list of (abs_path, class_idx, class_name) tuples."""
    # Classes temporarily discarded / kept aside (preserved in codebase, but skipped during evaluation)
    TEMPORARILY_DISCARDED_CLASSES = {
        "Nail_Infection",  # Kept aside: empty folder (0 images); specific nail conditions handled by Onychomycosis & Nail_Psoriasis
    }

    candidate_classes = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and d not in TEMPORARILY_DISCARDED_CLASSES
        and not d.startswith(('_', '.'))
    ])
    
    # Filter out classes with no valid image files
    classes = []
    for cls in candidate_classes:
        cls_dir = os.path.join(base_dir, cls)
        has_imgs = any(os.path.splitext(f)[1].lower() in IMAGE_EXTS for f in os.listdir(cls_dir))
        if has_imgs:
            classes.append(cls)
        else:
            logger.info(f"Skipping empty or discarded class folder: {cls}")

    samples = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                samples.append((os.path.join(cls_dir, fname), idx, cls))
    return classes, samples


def _train_test_split(samples, test_ratio=0.20, seed=42):
    """Stratified split by class."""
    rng = random.Random(seed)
    from collections import defaultdict
    by_class = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)

    train, test = [], []
    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)
        n_test = max(1, int(len(cls_samples) * test_ratio))
        test.extend(cls_samples[:n_test])
        train.extend(cls_samples[n_test:])
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, class_names):
    """
    Compute per-class and macro precision, recall, F1, and overall accuracy.
    Pure NumPy — no sklearn dependency.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = len(class_names)
    accuracy = float(np.mean(y_true == y_pred))

    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0

    for i, name in enumerate(class_names):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))
        tn = int(np.sum((y_true != i) & (y_pred != i)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   int(np.sum(y_true == i)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }
        macro_p  += precision
        macro_r  += recall
        macro_f1 += f1

    macro = {
        "accuracy":        round(accuracy, 4),
        "macro_precision": round(macro_p  / n_classes, 4),
        "macro_recall":    round(macro_r  / n_classes, 4),
        "macro_f1":        round(macro_f1 / n_classes, 4),
    }
    return {"per_class": per_class, "macro": macro}


def _confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def _save_confusion_matrix_png(cm, class_names, title, save_path):
    if not MPL_OK:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluators
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_modality_router(test_samples_rgb, test_samples_scan, output_dir):
    """Evaluate SimpleModalityCNN on held-out RGB + Scan test images."""
    logger.info("=" * 60)
    logger.info("Evaluating ModalityRouter (CNN binary classifier)...")

    sys.path.insert(0, BACKEND_DIR)
    from app.services.models import ModalityRouter
    model = ModalityRouter()

    y_true, y_pred = [], []
    errors = []

    for img_path, _, _ in test_samples_rgb:
        try:
            pred = model.predict(img_path)
            y_true.append(1)   # RGB = 1
            y_pred.append(1 if pred == "RGB" else 0)
        except Exception as e:
            errors.append(str(e))

    for img_path, _, _ in test_samples_scan:
        try:
            pred = model.predict(img_path)
            y_true.append(0)   # Scan = 0
            y_pred.append(1 if pred == "RGB" else 0)
        except Exception as e:
            errors.append(str(e))

    class_names = ["Scan", "RGB"]
    metrics = _compute_metrics(y_true, y_pred, class_names)
    metrics["errors"] = len(errors)
    metrics["test_size"] = len(y_true)
    metrics["timestamp"] = datetime.now().isoformat()

    out_json = os.path.join(output_dir, "modality_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = _confusion_matrix(y_true, y_pred, 2)
    _save_confusion_matrix_png(cm, class_names, "Modality Router — Confusion Matrix",
                                os.path.join(output_dir, "confusion_modality.png"))

    logger.info(f"Modality Router Accuracy : {metrics['macro']['accuracy']:.2%}")
    logger.info(f"Macro F1                 : {metrics['macro']['macro_f1']:.4f}")
    return metrics


def evaluate_biomed_clip(test_samples, classes, output_dir):
    """Evaluate BiomedCLIP LinearProbe on held-out RGB test images."""
    logger.info("=" * 60)
    logger.info("Evaluating BiomedCLIP RGB Classifier...")

    sys.path.insert(0, BACKEND_DIR)
    from app.services.models import BiomedCLIPService
    model = BiomedCLIPService()

    y_true, y_pred = [], []
    errors = []

    for img_path, true_idx, true_cls in test_samples:
        try:
            result = model.predict(img_path)
            # Match directly via raw_class if available, avoiding substring collision
            if "raw_class" in result and result["raw_class"] in classes:
                best_idx = classes.index(result["raw_class"])
            else:
                pred_disease = result["disease"].lower().replace(" ", "_")
                best_idx = 0
                best_score = -1
                for i, cls in enumerate(classes):
                    score = sum(w in pred_disease for w in cls.lower().split("_") if len(w) > 3)
                    if score > best_score:
                        best_score, best_idx = score, i
            y_true.append(true_idx)
            y_pred.append(best_idx)
        except Exception as e:
            errors.append(str(e))

    metrics = _compute_metrics(y_true, y_pred, classes)
    metrics["errors"] = len(errors)
    metrics["test_size"] = len(y_true)
    metrics["timestamp"] = datetime.now().isoformat()

    out_json = os.path.join(output_dir, "biomed_clip_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = _confusion_matrix(y_true, y_pred, len(classes))
    _save_confusion_matrix_png(cm, [c.replace("_", "\n") for c in classes],
                                "BiomedCLIP — 9-class RGB Confusion Matrix",
                                os.path.join(output_dir, "confusion_biomed_clip.png"))

    logger.info(f"BiomedCLIP Top-1 Accuracy: {metrics['macro']['accuracy']:.2%}")
    logger.info(f"Macro F1                 : {metrics['macro']['macro_f1']:.4f}")
    return metrics


def evaluate_dinov2(test_samples, classes, output_dir):
    """Evaluate DINOv2 LinearProbe on held-out Scan test images."""
    logger.info("=" * 60)
    logger.info("Evaluating DINOv2 Scan Classifier...")

    sys.path.insert(0, BACKEND_DIR)
    from app.services.models import RadDINOService
    model = RadDINOService()

    y_true, y_pred = [], []
    errors = []

    for img_path, true_idx, true_cls in test_samples:
        try:
            result = model.predict(img_path)
            # Match directly via raw_class if available, avoiding substring collision
            if "raw_class" in result and result["raw_class"] in classes:
                best_idx = classes.index(result["raw_class"])
            else:
                pred_disease = result["disease"].lower().replace(" ", "_")
                best_idx = 0
                best_score = -1
                for i, cls in enumerate(classes):
                    score = sum(w in pred_disease for w in cls.lower().split("_") if len(w) > 3)
                    if score > best_score:
                        best_score, best_idx = score, i
            y_true.append(true_idx)
            y_pred.append(best_idx)
        except Exception as e:
            errors.append(str(e))

    metrics = _compute_metrics(y_true, y_pred, classes)
    metrics["errors"] = len(errors)
    metrics["test_size"] = len(y_true)
    metrics["timestamp"] = datetime.now().isoformat()

    out_json = os.path.join(output_dir, "dinov2_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = _confusion_matrix(y_true, y_pred, len(classes))
    _save_confusion_matrix_png(cm, [c.replace("_", "\n") for c in classes],
                                "DINOv2 — 6-class Scan Confusion Matrix",
                                os.path.join(output_dir, "confusion_dinov2.png"))

    logger.info(f"DINOv2 Top-1 Accuracy    : {metrics['macro']['accuracy']:.2%}")
    logger.info(f"Macro F1                 : {metrics['macro']['macro_f1']:.4f}")
    return metrics


def write_summary(modality_m, clip_m, dino_m, output_dir):
    lines = [
        "=" * 70,
        "IV-I MEDICAL AI — PHASE 2 FORMAL EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "1. MODALITY ROUTER (SimpleModalityCNN — Binary: RGB vs Scan)",
        f"   Test samples : {modality_m.get('test_size', '?')}",
        f"   Accuracy     : {modality_m['macro']['accuracy']:.2%}",
        f"   Macro F1     : {modality_m['macro']['macro_f1']:.4f}",
        "",
        "2. RGB DISEASE CLASSIFIER (BiomedCLIP ViT-B/16 + LinearProbe — 9 classes)",
        f"   Test samples : {clip_m.get('test_size', '?')}",
        f"   Accuracy     : {clip_m['macro']['accuracy']:.2%}",
        f"   Macro F1     : {clip_m['macro']['macro_f1']:.4f}",
        "",
        "   Per-class F1 scores:",
    ]
    for cls, m in clip_m["per_class"].items():
        lines.append(f"     {cls:<30} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  (n={m['support']})")
    lines += [
        "",
        "3. SCAN PATHOLOGY CLASSIFIER (DINOv2 ViT-S/14 + LinearProbe — 6 classes)",
        f"   Test samples : {dino_m.get('test_size', '?')}",
        f"   Accuracy     : {dino_m['macro']['accuracy']:.2%}",
        f"   Macro F1     : {dino_m['macro']['macro_f1']:.4f}",
        "",
        "   Per-class F1 scores:",
    ]
    for cls, m in dino_m["per_class"].items():
        lines.append(f"     {cls:<30} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  (n={m['support']})")
    lines += ["", "=" * 70]

    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    for line in lines:
        print(line)
    logger.info(f"Summary saved: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IV-I Phase 2 Formal Evaluation")
    parser.add_argument("--split",  type=float, default=0.20, help="Test split ratio (default 0.20)")
    parser.add_argument("--seed",   type=int,   default=42,   help="Random seed (default 42)")
    parser.add_argument("--output", type=str,   default=os.path.join(BACKEND_DIR, "..", "evaluation_results"),
                        help="Output directory for metrics and plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {os.path.abspath(args.output)}")

    # Load dataset
    rgb_classes, rgb_samples   = _load_class_images(RGB_DIR)
    scan_classes, scan_samples = _load_class_images(SCAN_DIR)

    _, rgb_test   = _train_test_split(rgb_samples,  test_ratio=args.split, seed=args.seed)
    _, scan_test  = _train_test_split(scan_samples, test_ratio=args.split, seed=args.seed)

    logger.info(f"RGB  test set : {len(rgb_test)} images across {len(rgb_classes)} classes")
    logger.info(f"Scan test set : {len(scan_test)} images across {len(scan_classes)} classes")

    # Run evaluations
    modality_m = evaluate_modality_router(rgb_test, scan_test, args.output)
    clip_m     = evaluate_biomed_clip(rgb_test,  rgb_classes,  args.output)
    dino_m     = evaluate_dinov2(scan_test, scan_classes, args.output)

    write_summary(modality_m, clip_m, dino_m, args.output)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
