"""
mri_checker.py - MRI Image Compatibility Checker
Compares test MRI images against training data characteristics
and shows percentage difference for each metric.

Usage:
    python mri_checker.py --test "C:\project intern\data3\Testing"
    python mri_checker.py --test "C:\path\to\single_image.jpg"
    python mri_checker.py --scan   (auto-scan training data first)
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Training Data Reference Stats ─────────────────────────────────────────────
# Computed from data/Training/ folder (your actual training data)
# Run with --scan to recompute from your training folder
TRAINING_STATS = {
    "brightness_mean": 31.7,      # Average pixel brightness (grayscale)
    "brightness_std":  38.4,      # Brightness std deviation (contrast)
    "width_mean":      512.0,     # Average image width
    "height_mean":     512.0,     # Average image height
    "aspect_ratio":    1.0,       # width / height
    "rgb_ratio":       0.05,      # Fraction of RGB images (most are grayscale L)
    "contrast":        38.4,      # Std of pixel values = contrast
}

# ── Acceptable thresholds (what % difference is "OK") ─────────────────────────
THRESHOLDS = {
    "brightness":   20.0,   # ±20% brightness difference is acceptable
    "contrast":     25.0,   # ±25% contrast difference is acceptable
    "size":         30.0,   # ±30% size difference is acceptable
    "aspect_ratio": 15.0,   # ±15% aspect ratio difference is acceptable
    "mode":         10.0,   # ±10% RGB/grayscale mix difference is acceptable
}


def scan_training_data(training_dir):
    """Scan training folder and compute reference statistics."""
    print(f"\nScanning training data: {training_dir}")
    all_brightness, all_contrast, all_widths, all_heights, rgb_count, total = [], [], [], [], 0, 0

    for cls in os.listdir(training_dir):
        cls_path = os.path.join(training_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {cls}: {len(imgs)} images")

        for img_name in imgs[:100]:  # sample 100 per class for speed
            try:
                img = Image.open(os.path.join(cls_path, img_name))
                if img.mode == 'RGB':
                    rgb_count += 1
                arr = np.array(img.convert('L'))
                all_brightness.append(arr.mean())
                all_contrast.append(arr.std())
                all_widths.append(img.size[0])
                all_heights.append(img.size[1])
                total += 1
            except Exception:
                pass

    stats = {
        "brightness_mean": float(np.mean(all_brightness)),
        "brightness_std":  float(np.std(all_brightness)),
        "contrast":        float(np.mean(all_contrast)),
        "width_mean":      float(np.mean(all_widths)),
        "height_mean":     float(np.mean(all_heights)),
        "aspect_ratio":    float(np.mean(all_widths) / np.mean(all_heights)),
        "rgb_ratio":       float(rgb_count / total) if total > 0 else 0,
    }

    print(f"\nTraining Stats (from {total} sampled images):")
    for k, v in stats.items():
        print(f"  {k:20s}: {v:.2f}")

    return stats


def analyze_image(img_path):
    """Analyze a single image and return its statistics."""
    img = Image.open(img_path)
    mode = img.mode
    arr = np.array(img.convert('L'))
    return {
        "brightness_mean": float(arr.mean()),
        "contrast":        float(arr.std()),
        "width":           float(img.size[0]),
        "height":          float(img.size[1]),
        "aspect_ratio":    float(img.size[0] / img.size[1]),
        "is_rgb":          1.0 if mode == 'RGB' else 0.0,
        "mode":            mode,
        "path":            str(img_path),
    }


def analyze_folder(folder_path):
    """Analyze all images in a folder (with class subfolders or flat)."""
    all_stats = []
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # Check if flat or has subfolders
    has_subfolders = any(
        os.path.isdir(os.path.join(folder_path, d))
        for d in os.listdir(folder_path)
    )

    if has_subfolders:
        for cls in sorted(os.listdir(folder_path)):
            cls_path = os.path.join(folder_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(exts):
                    try:
                        s = analyze_image(os.path.join(cls_path, fname))
                        s['class'] = cls
                        all_stats.append(s)
                    except Exception:
                        pass
    else:
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(exts):
                try:
                    s = analyze_image(os.path.join(folder_path, fname))
                    s['class'] = 'unknown'
                    all_stats.append(s)
                except Exception:
                    pass

    return all_stats


def compute_differences(test_stats_list, train_stats):
    """Compute percentage difference between test and training stats."""
    if not test_stats_list:
        return {}

    test_brightness = np.mean([s["brightness_mean"] for s in test_stats_list])
    test_contrast   = np.mean([s["contrast"]        for s in test_stats_list])
    test_width      = np.mean([s["width"]            for s in test_stats_list])
    test_height     = np.mean([s["height"]           for s in test_stats_list])
    test_aspect     = np.mean([s["aspect_ratio"]     for s in test_stats_list])
    test_rgb_ratio  = np.mean([s["is_rgb"]           for s in test_stats_list])

    def pct_diff(test_val, train_val):
        if train_val == 0:
            return 0.0
        return abs(test_val - train_val) / train_val * 100

    diffs = {
        "Brightness":   {
            "train": train_stats["brightness_mean"],
            "test":  test_brightness,
            "diff":  pct_diff(test_brightness, train_stats["brightness_mean"]),
            "unit":  "pixel value (0-255)",
            "threshold": THRESHOLDS["brightness"],
        },
        "Contrast":     {
            "train": train_stats["contrast"],
            "test":  test_contrast,
            "diff":  pct_diff(test_contrast, train_stats["contrast"]),
            "unit":  "std deviation",
            "threshold": THRESHOLDS["contrast"],
        },
        "Image Width":  {
            "train": train_stats["width_mean"],
            "test":  test_width,
            "diff":  pct_diff(test_width, train_stats["width_mean"]),
            "unit":  "pixels",
            "threshold": THRESHOLDS["size"],
        },
        "Image Height": {
            "train": train_stats["height_mean"],
            "test":  test_height,
            "diff":  pct_diff(test_height, train_stats["height_mean"]),
            "unit":  "pixels",
            "threshold": THRESHOLDS["size"],
        },
        "Aspect Ratio": {
            "train": train_stats["aspect_ratio"],
            "test":  test_aspect,
            "diff":  pct_diff(test_aspect, train_stats["aspect_ratio"]),
            "unit":  "width/height",
            "threshold": THRESHOLDS["aspect_ratio"],
        },
        "RGB vs Gray":  {
            "train": train_stats["rgb_ratio"] * 100,
            "test":  test_rgb_ratio * 100,
            "diff":  abs(test_rgb_ratio * 100 - train_stats["rgb_ratio"] * 100),
            "unit":  "% RGB images",
            "threshold": THRESHOLDS["mode"],
        },
    }

    return diffs


def compatibility_score(diffs):
    """
    Compute overall compatibility score (0-100%).
    100% = identical to training data
    0%   = completely different
    """
    scores = []
    weights = {
        "Brightness":   0.35,  # Most important for model performance
        "Contrast":     0.25,
        "Image Width":  0.10,
        "Image Height": 0.10,
        "Aspect Ratio": 0.10,
        "RGB vs Gray":  0.10,
    }

    for metric, info in diffs.items():
        threshold = info["threshold"]
        diff      = info["diff"]
        # Score: 100% if diff=0, 0% if diff >= 2x threshold
        score = max(0, 100 - (diff / threshold) * 50)
        score = min(100, score)
        scores.append(score * weights.get(metric, 0.1))

    return sum(scores)


def print_report(diffs, overall_score, total_images, source_name):
    """Print detailed compatibility report to console."""
    print(f"\n{'='*65}")
    print(f"  MRI COMPATIBILITY REPORT")
    print(f"  Test Source   : {source_name}")
    print(f"  Total Images  : {total_images}")
    print(f"{'='*65}")

    print(f"\n  {'Metric':<18} {'Train':>10} {'Test':>10} {'Diff %':>10} {'Status'}")
    print(f"  {'-'*60}")

    for metric, info in diffs.items():
        diff       = info["diff"]
        threshold  = info["threshold"]
        status     = "✅ OK" if diff <= threshold else ("⚠️  WARN" if diff <= threshold * 2 else "🔴 HIGH")
        print(f"  {metric:<18} {info['train']:>10.1f} {info['test']:>10.1f} {diff:>9.1f}%  {status}")

    print(f"\n{'─'*65}")
    color = "🟢" if overall_score >= 80 else ("🟡" if overall_score >= 60 else "🔴")
    print(f"  {color} OVERALL COMPATIBILITY SCORE : {overall_score:.1f}%")
    print(f"{'─'*65}")

    if overall_score >= 80:
        print("  ✅ PREDICTION: Model should perform WELL on this data")
        print("     Low domain shift — similar to training distribution")
    elif overall_score >= 60:
        print("  ⚠️  PREDICTION: Model may show MODERATE accuracy drop")
        print("     Some domain shift detected — consider normalization")
    else:
        print("  🔴 PREDICTION: Model likely to show POOR accuracy")
        print("     High domain shift — images differ significantly from training data")
        print("     Tip: Use --normalize flag with evaluate_external.py")

    print(f"{'='*65}\n")


def plot_report(diffs, overall_score, source_name, save_path="mri_compatibility.png"):
    """Generate visual compatibility report."""
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#0a0f1e')

    # Title
    fig.suptitle("MRI Compatibility Report", fontsize=18, fontweight='bold',
                 color='white', y=0.97)
    fig.text(0.5, 0.93, f"Test: {source_name}", ha='center',
             fontsize=11, color='#94a3b8')

    # ── Big score gauge (left) ────────────────────────────────────────────────
    ax_score = fig.add_axes([0.03, 0.55, 0.28, 0.32])
    ax_score.set_facecolor('#111827')
    ax_score.set_xlim(0, 1); ax_score.set_ylim(0, 1)
    ax_score.axis('off')

    score_color = ('#22c55e' if overall_score >= 80
                   else ('#f59e0b' if overall_score >= 60 else '#ef4444'))
    ax_score.text(0.5, 0.72, f"{overall_score:.0f}%",
                  ha='center', va='center', fontsize=48,
                  fontweight='bold', color=score_color)
    ax_score.text(0.5, 0.38, "Compatibility", ha='center',
                  fontsize=13, color='white')
    ax_score.text(0.5, 0.22, "Score", ha='center', fontsize=13, color='white')

    status_text = ("GOOD MATCH" if overall_score >= 80
                   else ("MODERATE" if overall_score >= 60 else "POOR MATCH"))
    ax_score.text(0.5, 0.08, status_text, ha='center',
                  fontsize=11, color=score_color, fontweight='bold')
    for spine in ['top', 'right', 'bottom', 'left']:
        ax_score.spines[spine].set_color('#1e293b')

    # ── Per-metric bars ────────────────────────────────────────────────────────
    ax_bars = fig.add_axes([0.36, 0.55, 0.60, 0.34])
    ax_bars.set_facecolor('#111827')

    metrics    = list(diffs.keys())
    diffs_vals = [diffs[m]["diff"] for m in metrics]
    thresholds = [diffs[m]["threshold"] for m in metrics]
    bar_colors = [('#22c55e' if d <= t else ('#f59e0b' if d <= t * 2 else '#ef4444'))
                  for d, t in zip(diffs_vals, thresholds)]

    y_pos = range(len(metrics))
    bars  = ax_bars.barh(list(y_pos), diffs_vals, color=bar_colors, height=0.55, alpha=0.85)

    # Threshold line
    for i, (t, d) in enumerate(zip(thresholds, diffs_vals)):
        ax_bars.plot([t, t], [i - 0.35, i + 0.35], color='white', alpha=0.4,
                     linewidth=1.5, linestyle='--')
        ax_bars.text(d + 0.5, i, f"{d:.1f}%", va='center', fontsize=9, color='white')

    ax_bars.set_yticks(list(y_pos))
    ax_bars.set_yticklabels(metrics, color='white', fontsize=10)
    ax_bars.set_xlabel("% Difference from Training Data", color='#94a3b8', fontsize=10)
    ax_bars.set_title("Per-Metric Difference", color='white', fontsize=12, pad=8)
    ax_bars.tick_params(colors='#94a3b8')
    ax_bars.set_facecolor('#111827')
    for spine in ax_bars.spines.values():
        spine.set_color('#1e293b')
    ax_bars.set_xlim(0, max(max(diffs_vals) * 1.3, 50))

    legend_items = [
        mpatches.Patch(color='white', alpha=0.4, linestyle='--', label='Threshold (acceptable limit)'),
        mpatches.Patch(color='#22c55e', label='OK (within threshold)'),
        mpatches.Patch(color='#f59e0b', label='Warning (1-2x threshold)'),
        mpatches.Patch(color='#ef4444', label='High (>2x threshold)'),
    ]
    ax_bars.legend(handles=legend_items, loc='lower right', fontsize=8,
                   facecolor='#0d1117', labelcolor='white', framealpha=0.8)

    # ── Train vs Test comparison bars ─────────────────────────────────────────
    ax_comp = fig.add_axes([0.03, 0.08, 0.43, 0.38])
    ax_comp.set_facecolor('#111827')

    x        = np.arange(len(metrics))
    width    = 0.35
    train_v  = [diffs[m]["train"] for m in metrics]
    test_v   = [diffs[m]["test"]  for m in metrics]

    ax_comp.bar(x - width/2, train_v, width, label='Training Data',
                color='#1B5FAD', alpha=0.85)
    ax_comp.bar(x + width/2, test_v,  width, label='Test Data',
                color='#E8A020', alpha=0.85)

    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels([m[:10] for m in metrics], rotation=30,
                            ha='right', color='white', fontsize=8)
    ax_comp.set_title("Training vs Test Values", color='white', fontsize=11, pad=8)
    ax_comp.tick_params(colors='#94a3b8')
    ax_comp.set_facecolor('#111827')
    ax_comp.legend(facecolor='#0d1117', labelcolor='white', fontsize=9)
    for spine in ax_comp.spines.values():
        spine.set_color('#1e293b')

    # ── Recommendation box ────────────────────────────────────────────────────
    ax_rec = fig.add_axes([0.50, 0.08, 0.46, 0.38])
    ax_rec.set_facecolor('#111827')
    ax_rec.axis('off')

    ax_rec.text(0.05, 0.92, "Recommendations", fontsize=12,
                fontweight='bold', color='white', transform=ax_rec.transAxes)

    recs = []
    for metric, info in diffs.items():
        d = info["diff"]; t = info["threshold"]
        if d > t * 2:
            recs.append(f"🔴 {metric}: {d:.1f}% diff — HIGH impact on accuracy")
        elif d > t:
            recs.append(f"🟡 {metric}: {d:.1f}% diff — moderate impact")

    if not recs:
        recs = ["✅ All metrics within acceptable range",
                "   Model should perform well on this dataset"]

    if overall_score < 60:
        recs.append("💡 Try: python evaluate_external.py --normalize")
        recs.append("💡 Or: Fine-tune model on samples of this data")
    elif overall_score < 80:
        recs.append("💡 Try: python evaluate_external.py --normalize")

    for j, rec in enumerate(recs[:7]):
        ax_rec.text(0.05, 0.78 - j * 0.13, rec, fontsize=9,
                    color='#e2e8f0', transform=ax_rec.transAxes, wrap=True)

    for spine in ax_rec.spines.values():
        spine.set_color('#1e293b')

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0f1e')
    print(f"Chart saved → {save_path}")
    plt.show()


def check_single_image(img_path, train_stats):
    """Check a single MRI image compatibility."""
    print(f"\nAnalyzing: {img_path}")
    s = analyze_image(img_path)

    print(f"  Mode       : {s['mode']}")
    print(f"  Size       : {int(s['width'])} x {int(s['height'])} pixels")
    print(f"  Brightness : {s['brightness_mean']:.1f}")
    print(f"  Contrast   : {s['contrast']:.1f}")

    diffs = compute_differences([s], train_stats)
    score = compatibility_score(diffs)
    print_report(diffs, score, 1, Path(img_path).name)
    plot_report(diffs, score, Path(img_path).name,
                save_path="mri_compatibility_single.png")


def check_folder(folder_path, train_stats):
    """Check all MRI images in a folder."""
    print(f"\nAnalyzing folder: {folder_path}")
    stats_list = analyze_folder(folder_path)

    if not stats_list:
        print("No images found!")
        return

    print(f"Found {len(stats_list)} images")

    # Class-by-class breakdown
    classes = sorted(set(s['class'] for s in stats_list))
    if len(classes) > 1:
        print(f"\nPer-class breakdown:")
        for cls in classes:
            cls_stats = [s for s in stats_list if s['class'] == cls]
            cls_diffs = compute_differences(cls_stats, train_stats)
            cls_score = compatibility_score(cls_diffs)
            color     = "🟢" if cls_score >= 80 else ("🟡" if cls_score >= 60 else "🔴")
            bright    = np.mean([s['brightness_mean'] for s in cls_stats])
            print(f"  {color} {cls:15s}: {len(cls_stats):4d} images | "
                  f"score={cls_score:.0f}% | brightness={bright:.1f}")

    # Overall
    diffs = compute_differences(stats_list, train_stats)
    score = compatibility_score(diffs)
    print_report(diffs, score, len(stats_list), folder_path)
    plot_report(diffs, score, Path(folder_path).name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MRI Compatibility Checker — compare test data against training distribution"
    )
    parser.add_argument("--test",     help="Path to test image or folder")
    parser.add_argument("--training", default="data/Training",
                        help="Path to training folder (for --scan)")
    parser.add_argument("--scan",     action="store_true",
                        help="Rescan training folder to update reference stats")
    args = parser.parse_args()

    # Load/scan training stats
    if args.scan:
        train_stats = scan_training_data(args.training)
    else:
        train_stats = TRAINING_STATS
        print(f"Using cached training stats (brightness={train_stats['brightness_mean']:.1f}, "
              f"size={train_stats['width_mean']:.0f}x{train_stats['height_mean']:.0f})")
        print("Tip: Run with --scan to recompute from your actual training folder")

    if not args.test:
        print("\nUsage examples:")
        print('  python mri_checker.py --test "C:\\project intern\\data3\\Testing"')
        print('  python mri_checker.py --test "C:\\project intern\\data3\\Testing\\glioma\\img1.jpg"')
        print('  python mri_checker.py --test "C:\\project intern\\data2\\Testing" --scan')
        raise SystemExit

    test_path = args.test

    if os.path.isfile(test_path):
        check_single_image(test_path, train_stats)
    elif os.path.isdir(test_path):
        check_folder(test_path, train_stats)
    else:
        print(f"Path not found: {test_path}")
