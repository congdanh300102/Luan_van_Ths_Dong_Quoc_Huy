"""Chỉ tiêu đánh giá mô hình chấm điểm tín dụng.

Bên cạnh các chỉ tiêu phân loại phổ quát (Accuracy, Precision, Recall,
F1-score, ROC-AUC), module bổ sung các chỉ tiêu đặc thù của lĩnh vực quản trị
rủi ro tín dụng:

* **KS (Kolmogorov-Smirnov)** - khoảng cách lớn nhất giữa hàm phân phối tích lũy
  của nhóm tốt và nhóm xấu. Đây là chỉ tiêu được sử dụng rộng rãi nhất trong
  thực tiễn xây dựng thẻ điểm (scorecard) tại các tổ chức tín dụng.
* **Gini** - biến đổi đơn điệu của AUC: ``Gini = 2·AUC − 1``.
* **PR-AUC** - diện tích dưới đường Precision-Recall, phù hợp hơn ROC-AUC khi
  dữ liệu mất cân bằng nghiêm trọng.
* **Brier score** và **ECE** - đo mức độ *hiệu chuẩn* của xác suất dự báo. Với
  bài toán phê duyệt tín dụng, xác suất được dùng trực tiếp để định giá và
  phân vùng quyết định, nên độ hiệu chuẩn quan trọng không kém năng lực
  phân biệt.
* **Chi phí kỳ vọng** - hàm mất mát nghiệp vụ có trọng số bất đối xứng: bỏ sót
  một hồ sơ xấu (FN) tốn kém hơn nhiều so với từ chối sai một hồ sơ tốt (FP).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationReport:
    """Tập hợp đầy đủ chỉ tiêu đánh giá tại một ngưỡng quyết định."""

    n: int
    n_positive: int
    default_rate: float
    threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float
    pr_auc: float
    gini: float
    ks: float
    ks_threshold: float
    brier: float
    log_loss: float
    ece: float
    tn: int
    fp: int
    fn: int
    tp: int
    expected_cost: float
    approval_rate: float
    bad_rate_approved: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Các chỉ tiêu thành phần
# ---------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Thống kê KS và ngưỡng điểm đạt giá trị KS lớn nhất.

    ``KS = max_t |F_good(t) − F_bad(t)|`` trong đó ``F`` là hàm phân phối tích
    lũy của điểm số. Giá trị KS từ 0,3 trở lên thường được coi là đạt yêu cầu
    triển khai trong thực tiễn ngân hàng.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ks = float(np.max(tpr - fpr))
    idx = int(np.argmax(tpr - fpr))
    return ks, float(thr[idx])


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Sai số hiệu chuẩn kỳ vọng (Expected Calibration Error).

    Chia miền xác suất thành các khoảng, so sánh xác suất dự báo trung bình với
    tần suất thực tế trong từng khoảng, lấy bình quân theo trọng số số lượng.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


def business_cost(
    y_true: np.ndarray, y_pred: np.ndarray, cost_fn: float, cost_fp: float
) -> float:
    """Chi phí nghiệp vụ bình quân trên mỗi hồ sơ.

    ``y_pred = 1`` nghĩa là mô hình dự báo hồ sơ xấu (từ chối / chuyển thẩm định).
    Bỏ sót hồ sơ xấu (FN) gây tổn thất tín dụng; từ chối sai hồ sơ tốt (FP) gây
    mất cơ hội kinh doanh. Tỷ lệ ``cost_fn / cost_fp`` phản ánh mức độ ưu tiên
    kiểm soát rủi ro của tổ chức tín dụng.
    """
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float((cost_fn * fn + cost_fp * fp) / max(len(y_true), 1))


def optimal_threshold_by_cost(
    y_true: np.ndarray, y_prob: np.ndarray, cost_fn: float, cost_fp: float,
    n_grid: int = 400,
) -> Tuple[float, float]:
    """Tìm ngưỡng quyết định tối thiểu hóa chi phí nghiệp vụ kỳ vọng.

    Ngưỡng lý thuyết cho hàm chi phí tuyến tính là ``t* = c_FP / (c_FP + c_FN)``
    khi xác suất đã được hiệu chuẩn hoàn hảo. Trên thực tế xác suất chỉ hiệu
    chuẩn xấp xỉ, nên ngưỡng được tìm bằng cách quét trên lưới giá trị.
    """
    grid = np.quantile(y_prob, np.linspace(0.001, 0.999, n_grid))
    grid = np.unique(np.round(grid, 6))
    best_t, best_c = 0.5, float("inf")
    for t in grid:
        c = business_cost(y_true, (y_prob >= t).astype(int), cost_fn, cost_fp)
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t, best_c


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    cost_fn: float = 12.0,
    cost_fp: float = 1.0,
) -> ClassificationReport:
    """Tính toàn bộ chỉ tiêu đánh giá cho một mô hình."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if threshold is None:
        threshold, _ = optimal_threshold_by_cost(y_true, y_prob, cost_fn, cost_fp)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_true, y_prob))
    ks, ks_thr = ks_statistic(y_true, y_prob)

    # Tỷ lệ phê duyệt và tỷ lệ nợ xấu trong nhóm được duyệt: hai chỉ tiêu
    # then chốt để đánh giá tác động kinh doanh của mô hình.
    approved = y_pred == 0
    approval_rate = float(approved.mean())
    bad_rate_approved = float(y_true[approved].mean()) if approved.any() else 0.0

    return ClassificationReport(
        n=int(len(y_true)),
        n_positive=int(y_true.sum()),
        default_rate=float(y_true.mean()),
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(tn / max(tn + fp, 1)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        pr_auc=float(average_precision_score(y_true, y_prob)),
        gini=float(2 * auc - 1),
        ks=ks,
        ks_threshold=ks_thr,
        brier=float(brier_score_loss(y_true, y_prob)),
        log_loss=float(log_loss(y_true, np.clip(y_prob, 1e-9, 1 - 1e-9))),
        ece=expected_calibration_error(y_true, y_prob),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        expected_cost=business_cost(y_true, y_pred, cost_fn, cost_fp),
        approval_rate=approval_rate,
        bad_rate_approved=bad_rate_approved,
    )


# ---------------------------------------------------------------------------
# Phân tích theo thập phân vị điểm (gains table) - công cụ chuẩn của scorecard
# ---------------------------------------------------------------------------
def gains_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Bảng phân tích theo nhóm điểm (decile analysis).

    Sắp xếp hồ sơ theo rủi ro giảm dần, chia thành ``n_bins`` nhóm bằng nhau và
    thống kê tỷ lệ vỡ nợ, độ tập trung nợ xấu (lift) theo từng nhóm. Đây là
    công cụ trực quan chuẩn để kiểm tra tính đơn điệu và khả năng phân tách của
    thẻ điểm.
    """
    import pandas as pd

    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["bin"] = np.minimum((np.arange(len(df)) * n_bins) // len(df), n_bins - 1) + 1
    total_bad = max(int(df["y"].sum()), 1)
    g = df.groupby("bin").agg(
        n=("y", "size"), n_bad=("y", "sum"), pd_min=("p", "min"), pd_max=("p", "max"),
        pd_mean=("p", "mean"),
    )
    g["bad_rate"] = g["n_bad"] / g["n"]
    g["pct_bad"] = g["n_bad"] / total_bad
    g["cum_pct_bad"] = g["pct_bad"].cumsum()
    g["cum_pct_pop"] = g["n"].cumsum() / len(df)
    g["lift"] = g["bad_rate"] / (total_bad / len(df))
    g["cum_lift"] = g["cum_pct_bad"] / g["cum_pct_pop"]
    return g.reset_index()


__all__ = [
    "ClassificationReport",
    "evaluate",
    "ks_statistic",
    "expected_calibration_error",
    "business_cost",
    "optimal_threshold_by_cost",
    "gains_table",
]
