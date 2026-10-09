"""Giám sát trôi dạt dữ liệu và suy giảm chất lượng mô hình.

Mô hình tín dụng suy giảm theo thời gian vì ba nguyên nhân độc lập:

* **Trôi dạt biến đầu vào** (*covariate / data drift*) - phân phối đặc trưng thay
  đổi (ví dụ cơ cấu khách hàng dịch chuyển sang kênh số, thu nhập bình quân
  tăng). Được đo bằng **PSI** (Population Stability Index).
* **Trôi dạt khái niệm** (*concept drift*) - quan hệ giữa đặc trưng và rủi ro
  thay đổi (ví dụ suy thoái kinh tế làm cùng một mức DTI trở nên rủi ro hơn).
  Được phát hiện qua suy giảm AUC/KS trên dữ liệu mới.
* **Trôi dạt hiệu chuẩn** - xác suất dự báo lệch khỏi tần suất thực tế dù thứ tự
  xếp hạng vẫn đúng. Đây là dạng trôi dạt nguy hiểm nhất với quy trình phê duyệt
  tự động vì ngưỡng quyết định được đặt theo giá trị PD tuyệt đối.

PSI được tính theo công thức:

    ``PSI = Σ_i (a_i − e_i) · ln(a_i / e_i)``

trong đó ``e_i``, ``a_i`` là tỷ trọng quan sát trong khoảng ``i`` của tập tham
chiếu (expected) và tập hiện tại (actual). Ngưỡng thực tiễn: ``PSI < 0,10`` ổn
định; ``0,10-0,25`` cần theo dõi; ``> 0,25`` cần huấn luyện lại mô hình.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

EPS = 1e-6


def psi_numeric(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """PSI cho biến liên tục, chia khoảng theo phân vị của tập tham chiếu."""
    e = np.asarray(expected, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    e_fin, a_fin = e[np.isfinite(e)], a[np.isfinite(a)]
    if e_fin.size < n_bins or a_fin.size == 0:
        return 0.0

    edges = np.unique(np.quantile(e_fin, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_cnt, _ = np.histogram(e_fin, bins=edges)
    a_cnt, _ = np.histogram(a_fin, bins=edges)

    # bổ sung một nhóm riêng cho giá trị thiếu nếu có
    e_miss, a_miss = int((~np.isfinite(e)).sum()), int((~np.isfinite(a)).sum())
    if e_miss or a_miss:
        e_cnt = np.append(e_cnt, e_miss)
        a_cnt = np.append(a_cnt, a_miss)

    e_pct = np.maximum(e_cnt / max(e_cnt.sum(), 1), EPS)
    a_pct = np.maximum(a_cnt / max(a_cnt.sum(), 1), EPS)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI cho biến phân loại, mỗi hạng mục là một nhóm."""
    e = expected.astype("object").fillna("__NA__").value_counts(normalize=True)
    a = actual.astype("object").fillna("__NA__").value_counts(normalize=True)
    levels = e.index.union(a.index)
    e_pct = np.maximum(e.reindex(levels).fillna(0.0).to_numpy(), EPS)
    a_pct = np.maximum(a.reindex(levels).fillna(0.0).to_numpy(), EPS)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str] = (),
    n_bins: int = 10,
    warn: float = 0.10,
    alert: float = 0.25,
) -> pd.DataFrame:
    """Báo cáo PSI cho toàn bộ đặc trưng, xếp theo mức độ trôi dạt giảm dần."""
    rows: List[dict] = []
    for c in numeric:
        if c not in reference.columns or c not in current.columns:
            continue
        v = psi_numeric(reference[c].to_numpy(dtype="float64", na_value=np.nan),
                        current[c].to_numpy(dtype="float64", na_value=np.nan), n_bins)
        rows.append({"feature": c, "type": "numeric", "psi": v})
    for c in categorical:
        if c not in reference.columns or c not in current.columns:
            continue
        rows.append({"feature": c, "type": "categorical",
                     "psi": psi_categorical(reference[c], current[c])})

    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    df["status"] = np.where(df["psi"] >= alert, "BÁO ĐỘNG",
                            np.where(df["psi"] >= warn, "CẢNH BÁO", "ỔN ĐỊNH"))
    return df


def performance_over_time(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    timestamps: pd.Series,
    freq: str = "ME",
    min_samples: int = 200,
) -> pd.DataFrame:
    """Theo dõi AUC, KS và độ lệch hiệu chuẩn theo từng kỳ thời gian.

    Đây là cơ chế phát hiện trôi dạt khái niệm trong vận hành: nếu AUC theo kỳ
    giảm quá ``auc_drop_alert`` so với mức tại thời điểm nghiệm thu, mô hình cần
    được đánh giá lại và huấn luyện lại.
    """
    from .. models.metrics import ks_statistic

    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob),
                       "t": pd.to_datetime(timestamps)})
    df["period"] = df["t"].dt.to_period(freq.rstrip("E") if freq.endswith("E") else freq)
    rows = []
    for period, g in df.groupby("period", observed=True):
        if len(g) < min_samples or g["y"].nunique() < 2:
            continue
        auc = float(roc_auc_score(g["y"], g["p"]))
        ks, _ = ks_statistic(g["y"].to_numpy(), g["p"].to_numpy())
        rows.append(
            {
                "period": str(period),
                "n": int(len(g)),
                "actual_default_rate": float(g["y"].mean()),
                "predicted_pd_mean": float(g["p"].mean()),
                "calibration_gap": float(g["p"].mean() - g["y"].mean()),
                "auc": auc,
                "ks": ks,
                "gini": 2 * auc - 1,
            }
        )
    return pd.DataFrame(rows)


def plot_performance_over_time(
    perf: pd.DataFrame, out_dir, auc_alert_drop: float = 0.05,
    name: str = "performance_over_time",
) -> None:
    """Đồ thị AUC và độ lệch hiệu chuẩn theo thời gian."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..models.plots import PALETTE, apply_style, save

    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    x = np.arange(len(perf))

    ax = axes[0]
    ax.plot(x, perf["auc"], marker="o", markersize=4.2, color=PALETTE[0], linewidth=1.7,
            label="ROC-AUC theo kỳ")
    baseline = float(perf["auc"].iloc[0])
    ax.axhline(baseline, color="#999999", linestyle=":", linewidth=1.0,
               label=f"Mức nghiệm thu = {baseline:.4f}")
    ax.axhline(baseline - auc_alert_drop, color=PALETTE[1], linestyle="--", linewidth=1.1,
               label=f"Ngưỡng báo động (−{auc_alert_drop:.2f})")
    ax.fill_between(x, baseline - auc_alert_drop, baseline, alpha=0.07, color=PALETTE[3])
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Giám sát chất lượng mô hình theo thời gian")
    ax.legend(loc="lower left", ncols=3)

    ax = axes[1]
    w = 0.38
    ax.bar(x - w / 2, perf["actual_default_rate"] * 100, width=w, color=PALETTE[1],
           label="Tỷ lệ vỡ nợ thực tế", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, perf["predicted_pd_mean"] * 100, width=w, color=PALETTE[2],
           label="PD dự báo bình quân", edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_xlabel("Kỳ quan sát")
    ax.set_xticks(x, perf["period"], rotation=45, ha="right")
    ax.set_title("So sánh PD dự báo và tỷ lệ vỡ nợ thực tế (kiểm tra hiệu chuẩn)")
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    save(fig, out_dir, name)


__all__ = [
    "psi_numeric",
    "psi_categorical",
    "drift_report",
    "performance_over_time",
    "plot_performance_over_time",
]
