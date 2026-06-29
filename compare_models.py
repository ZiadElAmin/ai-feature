"""
Model Comparison: YOLOv8n (merged_v2) vs YOLOv26n (merged_v4) vs YOLOv11m (merged_v5)
Outputs:
  - comparison_metrics.png  : side-by-side metrics bar chart
  - comparison_detection.png: side-by-side detection on a chosen image
  - prints a metrics summary table to console

Note: merged_v2 and merged_v4 were trained on 4 classes (glove, no_glove, hairnet, no_hairnet).
      merged_v5 was trained on 6 classes (+ mask, no_mask). Each validates on its own dataset.
"""

import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import time

V8_MODEL_PATH  = "runs/detect/runs/ppe/merged_v2/weights/best.pt"
V26_MODEL_PATH = "runs/detect/runs/ppe/merged_v4/weights/best.pt"
V11_MODEL_PATH = "runs/merged_v5/weights/best.pt"

DATA_YAML_V4 = r"C:\Users\ziada\PycharmProjects\CVproject\merged4sets\data.yaml"
DATA_YAML_V5 = r"C:\Users\ziada\PycharmProjects\CVproject\merged5sets\data.yaml"

TEST_IMG = "detection_result.png"


def extract(m):
    return {
        "mAP50":     round(float(m.box.map50), 3),
        "mAP50-95":  round(float(m.box.map),   3),
        "Precision": round(float(m.box.mp),     3),
        "Recall":    round(float(m.box.mr),     3),
    }


def main():
    # 1. Load models
    print("Loading models...")
    v8_model  = YOLO(V8_MODEL_PATH)
    v26_model = YOLO(V26_MODEL_PATH)
    v11_model = YOLO(V11_MODEL_PATH)
    print("Done.\n")

    # 2. Validation metrics
    print("=" * 55)
    print("  Running validation — this may take ~30 seconds each")
    print("=" * 55)

    v8_metrics  = v8_model.val(data=DATA_YAML_V4,  verbose=False, workers=0)
    v26_metrics = v26_model.val(data=DATA_YAML_V4, verbose=False, workers=0)
    v11_metrics = v11_model.val(data=DATA_YAML_V5, verbose=False, workers=0)

    v8_scores  = extract(v8_metrics)
    v26_scores = extract(v26_metrics)
    v11_scores = extract(v11_metrics)

    # 3. Inference speed
    RUNS = 20
    t0 = time.perf_counter()
    for _ in range(RUNS):
        v8_model(TEST_IMG, verbose=False)
    v8_ms = round((time.perf_counter() - t0) / RUNS * 1000, 1)

    t0 = time.perf_counter()
    for _ in range(RUNS):
        v26_model(TEST_IMG, verbose=False)
    v26_ms = round((time.perf_counter() - t0) / RUNS * 1000, 1)

    t0 = time.perf_counter()
    for _ in range(RUNS):
        v11_model(TEST_IMG, verbose=False)
    v11_ms = round((time.perf_counter() - t0) / RUNS * 1000, 1)

    v8_scores["Inference (ms)"]  = v8_ms
    v26_scores["Inference (ms)"] = v26_ms
    v11_scores["Inference (ms)"] = v11_ms

    # 4. Print table
    df = pd.DataFrame({
        "Metric":                list(v8_scores.keys()),
        "YOLOv8n (merged_v2)":   list(v8_scores.values()),
        "YOLOv26n (merged_v4)":  list(v26_scores.values()),
        "YOLOv11m (merged_v5)":  list(v11_scores.values()),
    })
    print("\n" + "=" * 65)
    print("  METRICS COMPARISON")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65 + "\n")
    print("* merged_v2 & merged_v4: 4 classes (glove, no_glove, hairnet, no_hairnet)")
    print("* merged_v5: 6 classes (+ mask, no_mask)\n")

    # 5. Bar chart
    metrics_to_plot = ["mAP50", "mAP50-95", "Precision", "Recall"]
    v8_vals  = [v8_scores[m]  for m in metrics_to_plot]
    v26_vals = [v26_scores[m] for m in metrics_to_plot]
    v11_vals = [v11_scores[m] for m in metrics_to_plot]

    x = range(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width for i in x], v8_vals,  width, label='YOLOv8n (merged_v2)',  color='#4C72B0', alpha=0.85)
    bars2 = ax.bar([i          for i in x], v26_vals, width, label='YOLOv26n (merged_v4)', color='#DD8452', alpha=0.85)
    bars3 = ax.bar([i + width  for i in x], v11_vals, width, label='YOLOv11m (merged_v5)', color='#55A868', alpha=0.85)

    for bar in list(bars1) + list(bars2) + list(bars3):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("PPE Detection — Model Comparison\nYOLOv8n vs YOLOv26n vs YOLOv11m", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_metrics.png', dpi=150)
    plt.close()
    print("Saved: comparison_metrics.png")

    # 6. Side-by-side detection
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a test image for visual comparison",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        print(f"\nRunning detection on: {file_path}")
        r8  = v8_model(file_path,  conf=0.3)[0]
        r26 = v26_model(file_path, conf=0.3)[0]
        r11 = v11_model(file_path, conf=0.3)[0]

        img8  = cv2.cvtColor(r8.plot(),  cv2.COLOR_BGR2RGB)
        img26 = cv2.cvtColor(r26.plot(), cv2.COLOR_BGR2RGB)
        img11 = cv2.cvtColor(r11.plot(), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        axes[0].imshow(img8)
        axes[0].set_title(f"YOLOv8n (merged_v2)\nmAP50={v8_scores['mAP50']}  |  {v8_ms}ms/img",
                          fontsize=11, fontweight='bold', color='#4C72B0')
        axes[0].axis('off')

        axes[1].imshow(img26)
        axes[1].set_title(f"YOLOv26n (merged_v4)\nmAP50={v26_scores['mAP50']}  |  {v26_ms}ms/img",
                          fontsize=11, fontweight='bold', color='#DD8452')
        axes[1].axis('off')

        axes[2].imshow(img11)
        axes[2].set_title(f"YOLOv11m (merged_v5)\nmAP50={v11_scores['mAP50']}  |  {v11_ms}ms/img",
                          fontsize=11, fontweight='bold', color='#55A868')
        axes[2].axis('off')

        plt.suptitle("PPE Detection — Model Comparison", fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig('comparison_detection.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: comparison_detection.png")
    else:
        print("No image selected — skipping visual comparison.")

    print("\nDone! Files saved:")
    print("  comparison_metrics.png   — bar chart for documentation")
    print("  comparison_detection.png — side-by-side detection output")


if __name__ == '__main__':
    main()
