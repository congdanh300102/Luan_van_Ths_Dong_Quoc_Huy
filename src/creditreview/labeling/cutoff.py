"""Xác định điểm cắt phê duyệt trên thang xác suất vỡ nợ.

Mô hình học máy trả về một xác suất liên tục; để đưa ra kết quả hỗ trợ
kiểm duyệt cần chuyển xác suất đó thành khuyến nghị rời rạc. Ngưỡng mặc định 0,5
của bài toán phân loại **không phù hợp** với bài toán tín dụng vì hai lý do:

* Dữ liệu mất cân bằng: tỷ lệ nợ xấu khoảng 15%, nên rất ít hồ sơ có xác suất
  vượt 0,5; dùng ngưỡng này gần như duyệt toàn bộ hồ sơ.
* Chi phí hai loại sai lầm rất khác nhau: bỏ sót một hồ sơ xấu (bỏ lỡ dự báo
  vỡ nợ) gây tổn thất bằng một phần đáng kể dư nợ, trong khi từ chối sai một hồ
  sơ tốt chỉ làm mất phần lãi biên. Tỷ lệ chi phí thường được ước lượng trong
  khoảng 10:1 đến 15:1.

Module cung cấp ba cách xác định ngưỡng, phản ánh ba cách đặt bài toán khác nhau
trong thực tế nghiệp vụ:

``cutoff_by_cost``
    Tối thiểu hóa chi phí kỳ vọng. Dùng khi tổ chức tín dụng ước lượng được chi
    phí của hai loại sai lầm.
``cutoff_by_approval_rate``
    Chọn ngưỡng để đạt đúng một tỷ lệ duyệt mục tiêu. Dùng khi ngưỡng bị ràng
    buộc bởi kế hoạch tăng trưởng dư nợ.
``cutoff_by_bad_rate``
    Chọn ngưỡng sao cho tỷ lệ nợ xấu trong nhóm được duyệt không vượt trần cho
    phép. Dùng khi ràng buộc đến từ khẩu vị rủi ro.

Cuối cùng, :func:`three_zone_cutoffs` xác định **hai** ngưỡng chia thang điểm
thành ba vùng: tự động duyệt, chuyển thẩm định thủ công và từ chối. Đây chính là
cơ chế giúp mô hình đóng vai trò *hỗ trợ* thay vì thay thế cán bộ tín dụng:
phần hồ sơ nằm trong vùng không chắc chắn được chuyển cho người có thẩm quyền
xem xét thay vì bị quyết định máy móc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class CutoffResult:
    """Kết quả xác định một điểm cắt."""

    threshold: float
    approval_rate: float
    bad_rate_approved: float
    bad_rate_rejected: float
    expected_cost: float
    n_approved: int
    n_rejected: int
    criterion: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "Tiêu chí": self.criterion,
            "Điểm cắt PD": round(self.threshold, 4),
            "Tỷ lệ duyệt (%)": round(100.0 * self.approval_rate, 2),
            "Nợ xấu nhóm duyệt (%)": round(100.0 * self.bad_rate_approved, 2),
            "Nợ xấu nhóm từ chối (%)": round(100.0 * self.bad_rate_rejected, 2),
            "Chi phí kỳ vọng": round(self.expected_cost, 4),
        }


def _evaluate_cutoff(
    y: np.ndarray, p: np.ndarray, threshold: float, cost_fn: float, cost_fp: float
) -> CutoffResult:
    approve = p <= threshold
    n_app = int(approve.sum())
    n_rej = int((~approve).sum())
    # Bỏ sót hồ sơ xấu: hồ sơ thực sự xấu nhưng được duyệt
    fn = int(((y == 1) & approve).sum())
    # Từ chối sai hồ sơ tốt: hồ sơ thực sự tốt nhưng bị từ chối
    fp = int(((y == 0) & ~approve).sum())
    return CutoffResult(
        threshold=float(threshold),
        approval_rate=n_app / max(len(y), 1),
        bad_rate_approved=float(y[approve].mean()) if n_app else 0.0,
        bad_rate_rejected=float(y[~approve].mean()) if n_rej else 0.0,
        expected_cost=(cost_fn * fn + cost_fp * fp) / max(len(y), 1),
        n_approved=n_app,
        n_rejected=n_rej,
    )


def cutoff_by_cost(
    y: np.ndarray, p: np.ndarray, cost_fn: float = 12.0, cost_fp: float = 1.0,
    n_grid: int = 400,
) -> CutoffResult:
    """Điểm cắt tối thiểu hóa chi phí kỳ vọng của hai loại sai lầm."""
    grid = np.quantile(p, np.linspace(0.01, 0.99, n_grid))
    best = min(
        (_evaluate_cutoff(y, p, t, cost_fn, cost_fp) for t in np.unique(grid)),
        key=lambda r: r.expected_cost,
    )
    best.criterion = f"Tối thiểu chi phí (c_FN:c_FP = {cost_fn:.0f}:{cost_fp:.0f})"
    log.info(
        "Điểm cắt theo chi phí: PD ≤ %.4f | duyệt %.1f%% | nợ xấu nhóm duyệt %.2f%%",
        best.threshold, 100 * best.approval_rate, 100 * best.bad_rate_approved,
    )
    return best


def cutoff_by_approval_rate(
    y: np.ndarray, p: np.ndarray, target_rate: float = 0.70,
    cost_fn: float = 12.0, cost_fp: float = 1.0,
) -> CutoffResult:
    """Điểm cắt đạt đúng tỷ lệ duyệt mục tiêu."""
    t = float(np.quantile(p, np.clip(target_rate, 0.01, 0.99)))
    r = _evaluate_cutoff(y, p, t, cost_fn, cost_fp)
    r.criterion = f"Tỷ lệ duyệt mục tiêu {100 * target_rate:.0f}%"
    return r


def cutoff_by_bad_rate(
    y: np.ndarray, p: np.ndarray, max_bad_rate: float = 0.05,
    cost_fn: float = 12.0, cost_fp: float = 1.0, n_grid: int = 400,
) -> CutoffResult:
    """Điểm cắt cao nhất mà tỷ lệ nợ xấu nhóm duyệt vẫn dưới trần cho phép.

    Chọn ngưỡng cao nhất thỏa mãn ràng buộc để tối đa hóa quy mô cấp tín dụng
    trong giới hạn khẩu vị rủi ro.
    """
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n_grid)))
    feasible = [
        r for r in (_evaluate_cutoff(y, p, t, cost_fn, cost_fp) for t in grid)
        if r.bad_rate_approved <= max_bad_rate
    ]
    if not feasible:
        log.warning(
            "Không có điểm cắt nào đạt trần nợ xấu %.2f%% - trả về ngưỡng thấp nhất",
            100 * max_bad_rate,
        )
        r = _evaluate_cutoff(y, p, float(grid[0]), cost_fn, cost_fp)
    else:
        r = max(feasible, key=lambda x: x.threshold)
    r.criterion = f"Trần nợ xấu nhóm duyệt {100 * max_bad_rate:.0f}%"
    return r


def three_zone_cutoffs(
    y: np.ndarray,
    p: np.ndarray,
    target_review_rate: float = 0.20,
    cost_fn: float = 12.0,
    cost_fp: float = 1.0,
) -> Dict[str, float]:
    """Xác định hai ngưỡng chia thang PD thành ba vùng quyết định.

    Ngưỡng dưới (``pd_approve_max``) và ngưỡng trên (``pd_reject_min``) được đặt
    đối xứng quanh điểm cắt tối ưu chi phí, sao cho tỷ lệ hồ sơ rơi vào vùng
    chuyển thẩm định xấp xỉ ``target_review_rate``. Tỷ lệ này là tham số chính
    sách: tăng lên thì mức độ an toàn cao hơn nhưng khối lượng thẩm định thủ công
    lớn hơn; giảm xuống thì tự động hóa cao hơn nhưng rủi ro quyết định sai tăng.
    """
    base = cutoff_by_cost(y, p, cost_fn, cost_fp)
    q_base = float((p <= base.threshold).mean())
    half = target_review_rate / 2.0

    q_lo = np.clip(q_base - half, 0.02, 0.97)
    q_hi = np.clip(q_base + half, 0.03, 0.98)

    pd_approve_max = float(np.quantile(p, q_lo))
    pd_reject_min = float(np.quantile(p, q_hi))

    zones = {
        "pd_approve_max": pd_approve_max,
        "pd_reject_min": pd_reject_min,
        "cost_optimal": base.threshold,
    }
    log.info(
        "Ba vùng quyết định: duyệt PD ≤ %.4f | thẩm định (%.4f; %.4f] | từ chối PD > %.4f",
        pd_approve_max, pd_approve_max, pd_reject_min, pd_reject_min,
    )
    return zones


def cutoff_sensitivity_table(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: Sequence[float] | None = None,
    cost_fn: float = 12.0,
    cost_fp: float = 1.0,
) -> pd.DataFrame:
    """Bảng độ nhạy: hệ quả nghiệp vụ khi thay đổi điểm cắt.

    Bảng này là công cụ đối thoại giữa bộ phận mô hình và bộ phận kinh doanh:
    thay vì tranh luận về một con số ngưỡng trừu tượng, mỗi lựa chọn ngưỡng được
    quy đổi thành tỷ lệ duyệt và tỷ lệ nợ xấu tương ứng.
    """
    if thresholds is None:
        thresholds = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.32, 0.40]
    rows = []
    for t in thresholds:
        r = _evaluate_cutoff(y, p, float(t), cost_fn, cost_fp)
        rows.append(
            {
                "Điểm cắt PD": round(float(t), 4),
                "Số hồ sơ duyệt": r.n_approved,
                "Tỷ lệ duyệt (%)": round(100 * r.approval_rate, 2),
                "Nợ xấu nhóm duyệt (%)": round(100 * r.bad_rate_approved, 2),
                "Nợ xấu bị chặn (%)": round(100 * r.bad_rate_rejected, 2),
                "Chi phí kỳ vọng": round(r.expected_cost, 4),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CutoffResult",
    "cutoff_by_cost",
    "cutoff_by_approval_rate",
    "cutoff_by_bad_rate",
    "three_zone_cutoffs",
    "cutoff_sensitivity_table",
]
