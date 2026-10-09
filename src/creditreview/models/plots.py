"""Sinh đồ thị phục vụ luận văn.

Toàn bộ đồ thị được xuất ở hai định dạng: PDF (vector, dùng chèn vào LaTeX) và
PNG (xem nhanh). Bảng màu và kiểu chữ được thống nhất để các hình trong luận văn
tạo thành một hệ thống trực quan nhất quán, có thể đọc được cả khi in đen trắng
(các đường được phân biệt thêm bằng kiểu nét).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import auc, precision_recall_curve, roc_curve

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hệ thống trực quan
# ---------------------------------------------------------------------------
PALETTE: List[str] = [
    "#1f4e79",  # xanh đậm
    "#c0504d",  # đỏ gạch
    "#4f81bd",  # xanh trung
    "#e8a33d",  # cam vàng
    "#548235",  # xanh lá
    "#7030a0",  # tím
    "#2e9599",  # xanh ngọc
    "#a5a5a5",  # xám
    "#843c0c",  # nâu
    "#255e91",  # xanh thép
    "#bf8f00",  # vàng đất
]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": False,
}


def apply_style() -> None:
    plt.rcParams.update(_RC)


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Lưu hình ở cả hai định dạng PDF và PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("  hình: %s", name)


# ===========================================================================
# 1. Đồ thị đánh giá mô hình
# ===========================================================================
def plot_roc_curves(
    y_true: np.ndarray, preds: Dict[str, np.ndarray], out_dir: Path, name: str = "roc_curves"
) -> None:
    """Đường ROC của toàn bộ mô hình trên cùng một hệ trục."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    order = sorted(preds.items(), key=lambda kv: -auc(*roc_curve(y_true, kv[1])[:2]))
    for i, (label, p) in enumerate(order):
        fpr, tpr, _ = roc_curve(y_true, p)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=1.6,
                label=f"{label} (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.9, linestyle=":",
            label="Phân loại ngẫu nhiên")
    ax.set_xlabel("Tỷ lệ dương tính giả (False Positive Rate)")
    ax.set_ylabel("Tỷ lệ dương tính thật (True Positive Rate)")
    ax.set_title("Đường ROC của các mô hình dự báo rủi ro tín dụng")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.002)
    ax.legend(loc="lower right")
    save(fig, out_dir, name)


def plot_pr_curves(
    y_true: np.ndarray, preds: Dict[str, np.ndarray], out_dir: Path, name: str = "pr_curves"
) -> None:
    """Đường Precision-Recall - phù hợp hơn với dữ liệu mất cân bằng."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    items = sorted(preds.items(), key=lambda kv: -_ap(y_true, kv[1]))
    for i, (label, p) in enumerate(items):
        pr, rc, _ = precision_recall_curve(y_true, p)
        ax.plot(rc, pr, color=PALETTE[i % len(PALETTE)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=1.6,
                label=f"{label} (PR-AUC = {_ap(y_true, p):.4f})")
    base = float(np.mean(y_true))
    ax.axhline(base, color="#999999", linewidth=0.9, linestyle=":",
               label=f"Tỷ lệ vỡ nợ cơ sở = {base:.3f}")
    ax.set_xlabel("Recall (độ phủ hồ sơ xấu)")
    ax.set_ylabel("Precision (độ chính xác dự báo xấu)")
    ax.set_title("Đường Precision-Recall của các mô hình")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    save(fig, out_dir, name)


def _ap(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


def plot_ks_curve(
    y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path, model_name: str,
    name: str = "ks_curve",
) -> None:
    """Đồ thị KS: hàm phân phối tích lũy của nhóm tốt và nhóm xấu."""
    apply_style()
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ks_idx = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(thr, tpr, color=PALETTE[1], linewidth=1.8, label="CDF nhóm vỡ nợ (bad)")
    ax.plot(thr, fpr, color=PALETTE[0], linewidth=1.8, linestyle="--",
            label="CDF nhóm không vỡ nợ (good)")
    ax.vlines(thr[ks_idx], fpr[ks_idx], tpr[ks_idx], color=PALETTE[3], linewidth=2.2)
    ax.annotate(
        f"KS = {tpr[ks_idx] - fpr[ks_idx]:.4f}\ntại PD = {thr[ks_idx]:.3f}",
        xy=(thr[ks_idx], (tpr[ks_idx] + fpr[ks_idx]) / 2),
        xytext=(12, 0), textcoords="offset points", va="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e7", edgecolor="#e8a33d", linewidth=0.8),
    )
    ax.set_xlabel("Ngưỡng xác suất vỡ nợ (PD)")
    ax.set_ylabel("Tỷ lệ tích lũy")
    ax.set_title(f"Đồ thị Kolmogorov-Smirnov - {model_name}")
    ax.set_xlim(0, min(1.0, float(np.percentile(y_prob, 99.5)) * 1.1))
    ax.set_ylim(0, 1.002)
    ax.legend(loc="lower right")
    save(fig, out_dir, name)


def plot_calibration(
    y_true: np.ndarray, preds: Dict[str, np.ndarray], out_dir: Path,
    n_bins: int = 12, name: str = "calibration",
) -> None:
    """Đồ thị hiệu chuẩn: xác suất dự báo so với tần suất thực tế."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1.0,
            label="Hiệu chuẩn hoàn hảo")
    for i, (label, p) in enumerate(preds.items()):
        # chia nhóm theo phân vị để mỗi điểm có cùng số quan sát
        qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
        xs, ys = [], []
        for b in range(len(qs) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            xs.append(p[m].mean()); ys.append(y_true[m].mean())
        ax.plot(xs, ys, marker="o", markersize=3.6, linewidth=1.4,
                color=PALETTE[i % len(PALETTE)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], label=label)
    lim = max(0.05, min(1.0, float(np.percentile(np.concatenate(list(preds.values())), 99)) * 1.15))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Xác suất vỡ nợ dự báo (trung bình theo nhóm)")
    ax.set_ylabel("Tỷ lệ vỡ nợ thực tế")
    ax.set_title("Đồ thị hiệu chuẩn xác suất (Reliability diagram)")
    ax.legend(loc="upper left")
    save(fig, out_dir, name)


def plot_confusion_matrix(
    tn: int, fp: int, fn: int, tp: int, out_dir: Path, model_name: str,
    name: str = "confusion_matrix",
) -> None:
    """Ma trận nhầm lẫn của mô hình tại ngưỡng đã chọn."""
    apply_style()
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(4.9, 4.3))
    im = ax.imshow(cm / total, cmap="Blues", vmin=0, vmax=float((cm / total).max()))
    labels = ["Không vỡ nợ\n(y = 0)", "Vỡ nợ\n(y = 1)"]
    pred_labels = ["Dự báo tốt\n(duyệt)", "Dự báo xấu\n(từ chối/thẩm định)"]
    ax.set_xticks([0, 1], pred_labels)
    ax.set_yticks([0, 1], labels)
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{names[i][j]}\n{int(v):,}\n({100*v/total:.2f}%)",
                    ha="center", va="center", fontsize=9.5,
                    color="white" if (v / total) > 0.4 else "#1f3040", fontweight="medium")
    ax.set_title(f"Ma trận nhầm lẫn - {model_name}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=PercentFormatter(1.0))
    save(fig, out_dir, name)


def plot_metric_comparison(
    comparison: pd.DataFrame, out_dir: Path, name: str = "model_metrics_bar"
) -> None:
    """So sánh AUC / KS / F1 giữa các mô hình bằng đồ thị cột ngang."""
    apply_style()
    d = comparison.sort_values("AUC")
    fig, axes = plt.subplots(1, 3, figsize=(11.6, max(3.4, 0.42 * len(d))), sharey=True)
    for ax, col, title, color in zip(
        axes, ["AUC", "KS", "F1"],
        ["ROC-AUC", "Thống kê KS", "F1-score"],
        [PALETTE[0], PALETTE[2], PALETTE[4]],
    ):
        ax.barh(d["Mô hình"], d[col], color=color, height=0.62, edgecolor="white", linewidth=0.6)
        for i, v in enumerate(d[col]):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8.2)
        ax.set_title(title)
        ax.set_xlim(0, float(d[col].max()) * 1.20)
        ax.grid(axis="y", visible=False)
    axes[0].set_ylabel("")
    fig.suptitle("So sánh năng lực phân biệt của các mô hình trên tập kiểm thử", y=1.01)
    save(fig, out_dir, name)


def plot_auc_vs_latency(
    comparison: pd.DataFrame, out_dir: Path, name: str = "auc_vs_latency"
) -> None:
    """Đánh đổi giữa chất lượng dự báo và thời gian chấm điểm một hồ sơ.

    Mô hình được chọn để triển khai phải đồng thời thỏa mãn yêu cầu về năng lực
    phân biệt VÀ ràng buộc về thời gian phản hồi của quy trình kiểm duyệt trực
    tuyến. Biên Pareto trên đồ thị cho thấy những mô hình bị vượt trên cả hai
    tiêu chí -- tức không có lý do kỹ thuật nào để chọn chúng.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    x = comparison["Độ trễ p95 (ms)"].to_numpy()
    y = comparison["AUC"].to_numpy()
    labels = list(comparison["Mô hình"])

    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.scatter(xi, yi, s=110, color=PALETTE[i % len(PALETTE)], zorder=3,
                   edgecolor="white", linewidth=1.2)

    # biên Pareto: các mô hình không bị mô hình nào vượt trên cả hai tiêu chí
    order = np.argsort(x)
    best = -np.inf
    px, py = [], []
    for i in order:
        if y[i] > best:
            best = y[i]; px.append(x[i]); py.append(y[i])
    ax.step(px, py, where="post", color="#c0504d", linewidth=1.2, linestyle="--",
            alpha=0.85, label="Biên Pareto", zorder=2)
    ax.set_xscale("log")
    # Nới trục để nhãn đặt bên trái điểm ngoài cùng bên trái (và bên phải điểm
    # ngoài cùng bên phải) vẫn nằm trong khung vẽ.
    ax.set_xlim(float(x.min()) * 0.30, float(x.max()) * 3.0)
    pad = 0.12 * (float(y.max()) - float(y.min()))
    ax.set_ylim(float(y.min()) - pad, float(y.max()) + pad)
    ax.set_xlabel("Thời gian chấm điểm một hồ sơ, phân vị 95% (ms, thang log)")
    ax.set_ylabel("ROC-AUC trên tập kiểm thử")
    ax.set_title("Đánh đổi giữa năng lực phân biệt và thời gian chấm điểm")
    ax.legend(loc="lower right")

    # Nhãn tên mô hình: nhóm mô hình tốt nhất nằm sát nhau ở góc trên bên trái,
    # nếu dùng chung một độ dời cố định thì các nhãn chồng lên nhau và không đọc
    # được. Đặt nhãn theo thuật toán tham lam: thử lần lượt các vị trí quanh
    # điểm, chọn vị trí đầu tiên không đè lên nhãn đã đặt, và vẽ đường dẫn mảnh
    # khi nhãn phải dời xa.
    CANDIDATES = [
        (9, 4, "left"), (9, -11, "left"), (-9, 4, "right"), (-9, -11, "right"),
        (9, 14, "left"), (9, -22, "left"), (-9, 14, "right"), (-9, -22, "right"),
        (9, 25, "left"), (-9, 25, "right"), (9, -33, "left"), (-9, -33, "right"),
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed: list = []
    # Đặt nhãn của điểm có AUC cao trước: đó là vùng đông đúc nhất.
    for i in sorted(range(len(labels)), key=lambda k: -y[k]):
        chosen = None
        for dx, dy, ha in CANDIDATES:
            ann = ax.annotate(
                labels[i], (x[i], y[i]), xytext=(dx, dy),
                textcoords="offset points", fontsize=8.2, ha=ha, zorder=4,
            )
            bb = ann.get_window_extent(renderer=renderer).expanded(1.03, 1.12)
            if not any(bb.overlaps(b) for b in placed):
                placed.append(bb)
                chosen = (ann, dx, dy)
                break
            ann.remove()
        if chosen is None:
            dx, dy, ha = CANDIDATES[0]
            ann = ax.annotate(
                labels[i], (x[i], y[i]), xytext=(dx, dy),
                textcoords="offset points", fontsize=8.2, ha=ha, zorder=4,
            )
            placed.append(ann.get_window_extent(renderer=renderer))
            chosen = (ann, dx, dy)
        # đường dẫn mảnh khi nhãn nằm xa điểm
        if abs(chosen[2]) > 14:
            ax.annotate(
                "", (x[i], y[i]), xytext=(chosen[1], chosen[2]),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.6),
                zorder=1,
            )
    save(fig, out_dir, name)


def plot_gains(gains: pd.DataFrame, out_dir: Path, name: str = "gains_lift") -> None:
    """Đồ thị tích lũy nợ xấu (gains) và độ tập trung (lift) theo nhóm điểm."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

    ax = axes[0]
    xs = np.concatenate([[0], gains["cum_pct_pop"].to_numpy()])
    ys = np.concatenate([[0], gains["cum_pct_bad"].to_numpy()])
    ax.plot(xs, ys, marker="o", markersize=4, color=PALETTE[0], linewidth=1.8, label="Mô hình")
    ax.plot([0, 1], [0, 1], linestyle=":", color="#999999", linewidth=1.0, label="Ngẫu nhiên")
    ax.fill_between(xs, xs, ys, alpha=0.12, color=PALETTE[0])
    ax.set_xlabel("Tỷ lệ hồ sơ được sàng lọc (theo rủi ro giảm dần)")
    ax.set_ylabel("Tỷ lệ hồ sơ vỡ nợ phát hiện được")
    ax.set_title("Đường tích lũy phát hiện nợ xấu (Cumulative Gains)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.002)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.bar(gains["bin"], gains["lift"], color=PALETTE[2], edgecolor="white", linewidth=0.6)
    ax.axhline(1.0, color=PALETTE[1], linestyle="--", linewidth=1.1, label="Mức cơ sở (lift = 1)")
    for _, r in gains.iterrows():
        ax.text(r["bin"], r["lift"], f"{r['lift']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Nhóm thập phân vị theo rủi ro (1 = rủi ro cao nhất)")
    ax.set_ylabel("Lift (độ tập trung nợ xấu)")
    ax.set_title("Độ tập trung nợ xấu theo nhóm điểm")
    ax.set_xticks(gains["bin"].to_list())
    ax.legend()
    save(fig, out_dir, name)


def plot_score_distribution(
    y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path, thresholds: Dict[str, float],
    name: str = "score_distribution",
) -> None:
    """Phân phối PD của nhóm tốt và nhóm xấu cùng các ngưỡng quyết định."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(0, min(1.0, float(np.percentile(y_prob, 99.7))), 60)
    ax.hist(y_prob[y_true == 0], bins=bins, density=True, alpha=0.62,
            color=PALETTE[0], label="Không vỡ nợ", edgecolor="white", linewidth=0.3)
    ax.hist(y_prob[y_true == 1], bins=bins, density=True, alpha=0.62,
            color=PALETTE[1], label="Vỡ nợ", edgecolor="white", linewidth=0.3)
    for i, (lab, t) in enumerate(thresholds.items()):
        ax.axvline(t, color=PALETTE[3 + i], linestyle="--", linewidth=1.5)
        ax.text(t, ax.get_ylim()[1] * (0.95 - 0.1 * i), f" {lab} = {t:.3f}",
                fontsize=8.5, color=PALETTE[3 + i], va="top")
    ax.set_xlabel("Xác suất vỡ nợ dự báo (PD)")
    ax.set_ylabel("Mật độ")
    ax.set_title("Phân phối điểm rủi ro theo nhóm khách hàng và ngưỡng quyết định")
    ax.legend(loc="upper right")
    save(fig, out_dir, name)


# ===========================================================================
# 2. Đồ thị về đặc trưng
# ===========================================================================
def plot_iv_ranking(
    iv: pd.DataFrame, out_dir: Path, top_n: int = 25, name: str = "iv_ranking"
) -> None:
    """Xếp hạng đặc trưng theo Information Value, tô màu theo nhóm nguồn dữ liệu."""
    apply_style()
    d = iv.head(top_n).iloc[::-1]
    groups = d["group"].fillna("KHÁC").unique().tolist()
    cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(sorted(groups))}
    fig, ax = plt.subplots(figsize=(7.6, max(4.0, 0.30 * len(d))))
    ax.barh(d["feature"], d["iv"], color=[cmap[g] for g in d["group"].fillna("KHÁC")],
            height=0.68, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(d["iv"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    for x, lab in [(0.02, "0,02"), (0.10, "0,10"), (0.30, "0,30")]:
        if x < d["iv"].max() * 1.1:
            ax.axvline(x, color="#bbbbbb", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Information Value (IV)")
    ax.set_title(f"Top {top_n} đặc trưng theo Information Value")
    ax.set_xlim(0, float(d["iv"].max()) * 1.22)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    ax.legend(handles, list(cmap.keys()), title="Nhóm đặc trưng", loc="lower right", title_fontsize=8.5)
    save(fig, out_dir, name)


def plot_iv_by_group(iv_group: pd.DataFrame, out_dir: Path, name: str = "iv_by_group") -> None:
    """Tổng Information Value theo nhóm nguồn dữ liệu.

    Đồ thị này trả lời câu hỏi trọng tâm của đề tài: nhóm đặc trưng động tính từ
    luồng giao dịch đóng góp bao nhiêu vào năng lực dự báo so với nhóm đặc trưng
    tĩnh và nhóm đặc trưng CIC truyền thống.
    """
    apply_style()
    d = iv_group.sort_values("iv_sum")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    axes[0].barh(d["group"], d["iv_sum"], color=PALETTE[0], height=0.6,
                 edgecolor="white", linewidth=0.6)
    for i, v in enumerate(d["iv_sum"]):
        axes[0].text(v, i, f" {v:.2f}", va="center", fontsize=8.5)
    axes[0].set_xlabel("Tổng IV của nhóm")
    axes[0].set_title("Tổng Information Value theo nhóm đặc trưng")
    axes[0].set_xlim(0, float(d["iv_sum"].max()) * 1.18)
    axes[0].grid(axis="y", visible=False)

    axes[1].barh(d["group"], d["iv_max"], color=PALETTE[3], height=0.6,
                 edgecolor="white", linewidth=0.6)
    for i, v in enumerate(d["iv_max"]):
        axes[1].text(v, i, f" {v:.3f}", va="center", fontsize=8.5)
    axes[1].set_xlabel("IV của đặc trưng mạnh nhất trong nhóm")
    axes[1].set_title("Information Value lớn nhất theo nhóm")
    axes[1].set_xlim(0, float(d["iv_max"].max()) * 1.20)
    axes[1].grid(axis="y", visible=False)
    save(fig, out_dir, name)


def plot_feature_importance(
    importance: pd.DataFrame, out_dir: Path, top_n: int = 25,
    value_col: str = "importance", title: str = "Mức độ quan trọng của đặc trưng",
    name: str = "feature_importance",
) -> None:
    """Đồ thị mức độ quan trọng của đặc trưng (SHAP hoặc permutation)."""
    apply_style()
    d = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, max(4.0, 0.30 * len(d))))
    err = d["std"].to_numpy() if "std" in d.columns else None
    ax.barh(d["feature"], d[value_col], color=PALETTE[0], height=0.68,
            edgecolor="white", linewidth=0.5, xerr=err,
            error_kw=dict(ecolor="#666666", lw=0.8, capsize=2))
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    save(fig, out_dir, name)


# ===========================================================================
# 3. Đồ thị mô tả dữ liệu
# ===========================================================================
def plot_data_overview(
    customers: pd.DataFrame, applications: pd.DataFrame, labels: pd.DataFrame,
    out_dir: Path, name: str = "data_overview",
) -> None:
    """Bốn góc nhìn tổng quan về tập dữ liệu mô phỏng."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))

    # (a) phân phối thu nhập (thang log)
    ax = axes[0, 0]
    inc = customers["monthly_income"].to_numpy() / 1e6
    ax.hist(inc, bins=np.logspace(np.log10(max(inc.min(), 1)), np.log10(inc.max()), 50),
            color=PALETTE[0], edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Thu nhập hàng tháng (triệu VND, thang log)")
    ax.set_ylabel("Số khách hàng")
    ax.set_title("(a) Phân phối thu nhập khách hàng")

    # (b) phân phối tuổi theo nhãn
    ax = axes[0, 1]
    m = applications.merge(labels[["application_id", "default_flag"]], on="application_id")
    m = m.merge(customers[["customer_id", "age"]], on="customer_id")
    bins = np.arange(20, 70, 2)
    ax.hist([m.loc[m["default_flag"] == 0, "age"], m.loc[m["default_flag"] == 1, "age"]],
            bins=bins, density=True, color=[PALETTE[0], PALETTE[1]],
            label=["Không vỡ nợ", "Vỡ nợ"], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Tuổi")
    ax.set_ylabel("Mật độ")
    ax.set_title("(b) Phân phối tuổi theo kết quả trả nợ")
    ax.legend()

    # (c) cơ cấu sản phẩm và tỷ lệ vỡ nợ
    ax = axes[1, 0]
    g = m.groupby("product_code", observed=True).agg(
        n=("application_id", "size"), dr=("default_flag", "mean")
    ).sort_values("n", ascending=True)
    ax.barh(g.index, g["n"], color=PALETTE[2], height=0.62, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Số hồ sơ")
    ax.set_title("(c) Cơ cấu hồ sơ theo sản phẩm")
    ax.grid(axis="y", visible=False)
    ax2 = ax.twiny()
    ax2.plot(g["dr"] * 100, g.index, marker="D", markersize=5, color=PALETTE[1],
             linewidth=0, label="Tỷ lệ vỡ nợ")
    ax2.set_xlabel("Tỷ lệ vỡ nợ (%)", color=PALETTE[1])
    ax2.tick_params(axis="x", colors=PALETTE[1])
    ax2.grid(False)
    ax2.set_xlim(0, float(g["dr"].max()) * 100 * 1.25)

    # (d) số hồ sơ và tỷ lệ vỡ nợ theo tháng
    ax = axes[1, 1]
    m["month"] = pd.to_datetime(m["apply_time"]).dt.to_period("M").dt.to_timestamp()
    t = m.groupby("month", observed=True).agg(n=("application_id", "size"), dr=("default_flag", "mean"))
    ax.bar(t.index, t["n"], width=22, color=PALETTE[7], edgecolor="white", linewidth=0.4,
           label="Số hồ sơ")
    ax.set_ylabel("Số hồ sơ/tháng")
    ax.set_xlabel("Thời gian")
    ax.set_title("(d) Khối lượng hồ sơ và tỷ lệ vỡ nợ theo tháng")
    ax3 = ax.twinx()
    ax3.plot(t.index, t["dr"] * 100, color=PALETTE[1], marker="o", markersize=3.4, linewidth=1.5,
             label="Tỷ lệ vỡ nợ")
    ax3.set_ylabel("Tỷ lệ vỡ nợ (%)", color=PALETTE[1])
    ax3.tick_params(axis="y", colors=PALETTE[1])
    ax3.grid(False)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    save(fig, out_dir, name)


def plot_drift(psi: pd.DataFrame, out_dir: Path, top_n: int = 20, name: str = "drift_psi") -> None:
    """Chỉ số PSI của các đặc trưng giữa tập huấn luyện và tập kiểm thử."""
    apply_style()
    d = psi.head(top_n).iloc[::-1]
    colors = ["#c0504d" if v >= 0.25 else "#e8a33d" if v >= 0.10 else "#548235"
              for v in d["psi"]]
    fig, ax = plt.subplots(figsize=(7.4, max(3.8, 0.30 * len(d))))
    ax.barh(d["feature"], d["psi"], color=colors, height=0.66, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(d["psi"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.axvline(0.10, color="#e8a33d", linestyle="--", linewidth=1.1, label="Ngưỡng cảnh báo (0,10)")
    ax.axvline(0.25, color="#c0504d", linestyle="--", linewidth=1.1, label="Ngưỡng báo động (0,25)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Mức độ trôi dạt phân phối đặc trưng giữa tập huấn luyện và tập kiểm thử")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, out_dir, name)


# ===========================================================================
# 4. Đồ thị kết quả hỗ trợ kiểm duyệt
# ===========================================================================
def plot_decision_mix(
    mix: pd.DataFrame, out_dir: Path, name: str = "decision_mix"
) -> None:
    """Cơ cấu khuyến nghị của mô hình và tỷ lệ nợ xấu thực tế theo từng nhóm.

    Đồ thị bên phải là phần quan trọng về nghiệp vụ: nó cho thấy mô hình có
    thực sự phân tách được hồ sơ tốt và hồ sơ xấu hay không. Một mô hình hoạt
    động đúng phải cho tỷ lệ nợ xấu tăng dần theo thứ tự duyệt -> thẩm định ->
    từ chối.
    """
    apply_style()
    if mix.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    colors = [PALETTE[4], PALETTE[3], PALETTE[1]]
    ax = axes[0]
    ax.pie(
        mix["Tỷ trọng (%)"], labels=mix["Kết quả"], autopct="%1.1f%%", startangle=100,
        colors=colors, textprops=dict(fontsize=9),
        wedgeprops=dict(edgecolor="white", linewidth=1.4),
    )
    auto = 100.0 - float(
        mix.loc[mix["Kết quả"].str.contains("thẩm định"), "Tỷ trọng (%)"].sum()
    )
    ax.set_title(f"(a) Cơ cấu khuyến nghị\n(tỷ lệ xử lý tự động: {auto:.1f}%)")

    ax = axes[1]
    ax.bar(mix["Kết quả"], mix["Tỷ lệ nợ xấu thực tế (%)"], color=colors,
           width=0.58, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(mix["Tỷ lệ nợ xấu thực tế (%)"]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Tỷ lệ nợ xấu thực tế (%)")
    ax.set_title("(b) Tỷ lệ nợ xấu thực tế theo nhóm khuyến nghị")
    ax.set_ylim(0, max(mix["Tỷ lệ nợ xấu thực tế (%)"]) * 1.25 + 1)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, out_dir, name)


def plot_rule_hits(
    hits: pd.DataFrame, out_dir: Path, top_n: int = 15, name: str = "rule_hits"
) -> None:
    """Tần suất vi phạm từng luật nghiệp vụ."""
    apply_style()
    d = hits.head(top_n).iloc[::-1]
    if d.empty:
        return
    cmap = {"HARD_REJECT": PALETTE[1], "REVIEW": PALETTE[3], "WARNING": PALETTE[7]}
    colors = [cmap.get(a, PALETTE[0]) for a in d["Hành động"]]
    fig, ax = plt.subplots(figsize=(8.6, max(3.6, 0.36 * len(d))))
    ax.barh(d["Mã luật"] + " - " + d["Tên luật"].str.slice(0, 46),
            d["Tỷ lệ vi phạm"] * 100, color=colors, height=0.66,
            edgecolor="white", linewidth=0.5)
    for i, v in enumerate(d["Tỷ lệ vi phạm"] * 100):
        ax.text(v, i, f" {v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("Tỷ lệ hồ sơ vi phạm (%)")
    ax.set_title("Tần suất vi phạm luật nghiệp vụ")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    ax.legend(handles, ["Từ chối cứng", "Chuyển thẩm định", "Cảnh báo"],
              loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, out_dir, name)


def plot_cutoff_sensitivity(
    sens: pd.DataFrame, out_dir: Path, zones: Dict[str, float] | None = None,
    name: str = "cutoff_sensitivity",
) -> None:
    """Đánh đổi giữa tỷ lệ duyệt và tỷ lệ nợ xấu khi thay đổi điểm cắt.

    Đây là đồ thị dùng để đối thoại với bộ phận kinh doanh: mỗi điểm trên đường
    cong tương ứng một điểm cắt, cho biết muốn duyệt thêm bao nhiêu hồ sơ thì
    phải chấp nhận tỷ lệ nợ xấu tăng thêm bao nhiêu.
    """
    apply_style()
    if sens.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(sens["Tỷ lệ duyệt (%)"], sens["Nợ xấu nhóm duyệt (%)"],
            marker="o", color=PALETTE[0], linewidth=1.8, markersize=5)
    for _, r in sens.iterrows():
        ax.annotate(f"{r['Điểm cắt PD']:.2f}",
                    (r["Tỷ lệ duyệt (%)"], r["Nợ xấu nhóm duyệt (%)"]),
                    textcoords="offset points", xytext=(0, 7), fontsize=7.5,
                    ha="center", color="#444444")
    ax.set_xlabel("Tỷ lệ hồ sơ được duyệt (%)")
    ax.set_ylabel("Tỷ lệ nợ xấu trong nhóm được duyệt (%)")
    ax.set_title("Đánh đổi giữa quy mô cấp tín dụng và chất lượng danh mục\n"
                 "(nhãn trên mỗi điểm là giá trị điểm cắt PD)")
    if zones:
        for key, label, color in [
            ("pd_approve_max", "Ngưỡng khuyến nghị duyệt", PALETTE[4]),
            ("pd_reject_min", "Ngưỡng khuyến nghị từ chối", PALETTE[1]),
        ]:
            v = zones.get(key)
            if v is None:
                continue
            row = sens.iloc[(sens["Điểm cắt PD"] - v).abs().argmin()]
            ax.scatter([row["Tỷ lệ duyệt (%)"]], [row["Nợ xấu nhóm duyệt (%)"]],
                       s=110, facecolors="none", edgecolors=color, linewidths=1.8,
                       label=f"{label} ({v:.3f})", zorder=5)
        ax.legend(loc="upper left")
    save(fig, out_dir, name)


def plot_traditional_comparison(
    table: pd.DataFrame, out_dir: Path, name: str = "traditional_comparison"
) -> None:
    """So sánh mô hình đề xuất với quy trình thẩm định truyền thống."""
    apply_style()
    if table.empty:
        return
    metrics = [
        "Tỷ lệ duyệt (%)", "Tỷ lệ thẩm định thủ công (%)",
        "Nợ xấu nhóm duyệt (%)", "Tỷ lệ tự động hóa (%)",
    ]
    metrics = [m for m in metrics if m in table.columns]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for i, (_, row) in enumerate(table.iterrows()):
        vals = [float(row[m]) for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      color=PALETTE[7] if i == 0 else PALETTE[0],
                      edgecolor="white", linewidth=0.7,
                      label=str(row["Phương pháp"]))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (%)", "") for m in metrics], fontsize=9)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("So sánh mô hình hỗ trợ kiểm duyệt với quy trình thẩm định truyền thống")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, out_dir, name)


__all__ = [
    "apply_style", "save", "PALETTE",
    "plot_roc_curves", "plot_pr_curves", "plot_ks_curve", "plot_calibration",
    "plot_confusion_matrix", "plot_metric_comparison", "plot_auc_vs_latency",
    "plot_gains", "plot_score_distribution",
    "plot_iv_ranking", "plot_iv_by_group", "plot_feature_importance",
    "plot_data_overview", "plot_drift",
    "plot_decision_mix", "plot_rule_hits", "plot_cutoff_sensitivity",
    "plot_traditional_comparison",
]
