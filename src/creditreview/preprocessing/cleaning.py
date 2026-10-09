"""Xử lý giá trị khuyết thiếu và giá trị ngoại lệ.

XỬ LÝ GIÁ TRỊ KHUYẾT THIẾU
--------------------------
Trong dữ liệu tín dụng, giá trị khuyết không phải lúc nào cũng là lỗi thu thập.
Cần phân biệt hai loại:

* **Khuyết mang thông tin** - khách hàng chưa từng có quan hệ tín dụng nên toàn
  bộ trường CIC trống (*thin-file*). Bản thân việc "không có thông tin" là một
  tín hiệu rủi ro. Với nhóm này, cách xử lý đúng là điền một giá trị quy ước
  theo nghiệp vụ **và** bổ sung cờ chỉ báo khuyết để mô hình học được tín hiệu.

* **Khuyết ngẫu nhiên** - khách hàng không khai báo thu nhập, thiếu thâm niên
  công việc. Với nhóm này, điền trung vị của tập huấn luyện là phù hợp vì trung
  vị ít nhạy cảm với đuôi phân phối lệch phải điển hình của biến tiền tệ.

Việc điền khuyết bằng thống kê (trung vị) được thực hiện *bên trong* pipeline
scikit-learn (xem :mod:`creditreview.preprocessing.encoding`) để tham số chỉ
được ước lượng từ tập huấn luyện của từng lần kiểm định chéo. Module này đảm
nhiệm phần điền khuyết theo **quy ước nghiệp vụ**, vốn không phụ thuộc dữ liệu
nên có thể áp dụng ngay ở bước làm sạch.

XỬ LÝ GIÁ TRỊ NGOẠI LỆ
----------------------
Ngoại lệ trong dữ liệu tín dụng thường là giá trị thật (một khách hàng thu nhập
rất cao, một giao dịch giá trị rất lớn) chứ không phải lỗi nhập liệu. Vì vậy
không loại bỏ bản ghi -- làm vậy sẽ mất chính nhóm khách hàng đáng quan tâm --
mà **cắt ngọn** (winsorize) tại phân vị 1% và 99% học từ tập huấn luyện. Cách
này giữ nguyên số lượng quan sát, đồng thời hạn chế ảnh hưởng quá mức của một
vài giá trị cực trị lên các mô hình tuyến tính và mạng nơ-ron.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ===========================================================================
# 1. Giá trị khuyết thiếu
# ===========================================================================
@dataclass
class MissingValueReport:
    """Kết quả phân tích giá trị khuyết của một khung dữ liệu."""

    table: pd.DataFrame
    n_rows: int

    @property
    def n_affected_columns(self) -> int:
        return int((self.table["missing_pct"] > 0).sum())

    def high_missing(self, threshold_pct: float = 30.0) -> List[str]:
        """Danh sách biến có tỷ lệ khuyết vượt ngưỡng, cần xem xét loại bỏ."""
        t = self.table
        return t.loc[t["missing_pct"] > threshold_pct, "variable"].tolist()


def missing_value_report(df: pd.DataFrame, group_map: Dict[str, str] | None = None) -> MissingValueReport:
    """Thống kê tỷ lệ khuyết theo từng biến, kèm nhóm nguồn dữ liệu."""
    rows = []
    for c in df.columns:
        if c.startswith("_"):
            continue
        n_missing = int(df[c].isna().sum())
        rows.append(
            {
                "variable": c,
                "group": (group_map or {}).get(c, ""),
                "n_missing": n_missing,
                "missing_pct": round(100.0 * n_missing / max(len(df), 1), 3),
                "strategy": _strategy_for(c, (group_map or {}).get(c, "")),
            }
        )
    table = pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)
    rep = MissingValueReport(table=table, n_rows=len(df))
    log.info(
        "Giá trị khuyết: %d/%d biến có dữ liệu trống (cao nhất %.2f%%)",
        rep.n_affected_columns, len(table),
        float(table["missing_pct"].max()) if len(table) else 0.0,
    )
    return rep


def _strategy_for(column: str, group: str) -> str:
    """Chiến lược điền khuyết dự kiến cho một biến."""
    if group == "CIC":
        return "Quy ước nghiệp vụ + cờ chỉ báo khuyết"
    if column in _ZERO_FILL:
        return "Điền 0 (không phát sinh)"
    return "Trung vị tập huấn luyện + cờ chỉ báo khuyết"


# Các trường mà giá trị khuyết có nghĩa là "không phát sinh", điền 0
_ZERO_FILL = {
    "ltv", "n_accounts", "total_avg_balance", "collateral_value",
}

# Quy ước nghiệp vụ cho khách hàng không có dữ liệu tại CIC (thin-file).
# Giá trị được chọn theo hướng *thận trọng*: không giả định khách hàng tốt cũng
# không giả định khách hàng xấu, mà đặt ở mức trung tính kèm cờ chỉ báo để mô
# hình tự học mức rủi ro thực tế của nhóm này từ dữ liệu.
CIC_DEFAULTS: Dict[str, float] = {
    "cic_score": 600.0,          # mức trung tính trong thang 300-850
    "current_group": 1.0,        # chưa phát sinh nợ -> nhóm 1
    "worst_group_12m": 1.0,
    "dpd_max_12m": 0.0,
    "num_institutions": 0.0,
    "num_active_contracts": 0.0,
    "num_closed_contracts": 0.0,
    "total_outstanding": 0.0,
    "credit_utilization": 0.0,
    "monthly_debt_payment": 0.0,
    "num_inquiries_3m": 0.0,
    "has_written_off_debt": 0.0,
    "oldest_account_months": 0.0,
    "num_bad_debt_institutions": 0.0,
    "is_thin_file": 1.0,
}


def impute_business_defaults(df: pd.DataFrame, add_indicator: bool = True) -> pd.DataFrame:
    """Điền giá trị khuyết theo quy ước nghiệp vụ (không phụ thuộc dữ liệu).

    Hàm chỉ xử lý các trường có quy ước rõ ràng; phần còn lại được điền bằng
    thống kê bên trong pipeline huấn luyện để tránh rò rỉ thông tin.
    """
    out = df.copy()

    if add_indicator and "cic_score" in out.columns and "cic_missing" not in out.columns:
        out["cic_missing"] = out["cic_score"].isna().astype(int)

    n_filled = 0
    for col, value in CIC_DEFAULTS.items():
        if col in out.columns:
            n = int(out[col].isna().sum())
            if n:
                out[col] = out[col].fillna(value)
                n_filled += n

    for col in _ZERO_FILL:
        if col in out.columns:
            n = int(out[col].isna().sum())
            if n:
                out[col] = out[col].fillna(0.0)
                n_filled += n

    log.info("Điền khuyết theo quy ước nghiệp vụ: %s ô dữ liệu", f"{n_filled:,}")
    return out


# ===========================================================================
# 2. Giá trị ngoại lệ
# ===========================================================================
def outlier_report(
    df: pd.DataFrame, columns: Sequence[str], k_iqr: float = 3.0
) -> pd.DataFrame:
    """Thống kê ngoại lệ theo quy tắc khoảng tứ phân vị (IQR).

    Dùng hệ số ``k = 3`` (thay vì 1,5 thông thường) vì các biến tiền tệ trong
    dữ liệu tín dụng có phân phối lệch phải mạnh; ngưỡng 1,5·IQR sẽ đánh dấu
    một tỷ lệ rất lớn quan sát là ngoại lệ mà không mang lại thông tin hữu ích.
    """
    rows = []
    for c in columns:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = df[c].dropna().to_numpy(dtype="float64")
        if v.size < 20:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k_iqr * iqr, q3 + k_iqr * iqr
        n_low = int((v < lo).sum())
        n_high = int((v > hi).sum())
        if n_low + n_high == 0:
            continue
        rows.append(
            {
                "variable": c,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_fence": float(lo),
                "upper_fence": float(hi),
                "n_below": n_low,
                "n_above": n_high,
                "outlier_pct": round(100.0 * (n_low + n_high) / v.size, 3),
                "p01": float(np.percentile(v, 1)),
                "p99": float(np.percentile(v, 99)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("outlier_pct", ascending=False).reset_index(drop=True)
        log.info(
            "Ngoại lệ (quy tắc %.1f×IQR): %d biến có giá trị ngoài ngưỡng, cao nhất %.2f%%",
            k_iqr, len(out), float(out["outlier_pct"].max()),
        )
    return out


class OutlierCapper:
    """Cắt ngọn giá trị ngoại lệ tại phân vị dưới/trên học từ tập huấn luyện.

    Tuân thủ giao diện ``fit``/``transform`` của scikit-learn để có thể đặt vào
    trong pipeline, bảo đảm ngưỡng cắt chỉ được ước lượng từ dữ liệu huấn luyện.
    """

    def __init__(self, columns: Iterable[str], lower_q: float = 0.01, upper_q: float = 0.99) -> None:
        self.columns = list(columns)
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.bounds_: Dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierCapper":  # noqa: N803
        self.bounds_ = {}
        for c in self.columns:
            if c not in X.columns or not pd.api.types.is_numeric_dtype(X[c]):
                continue
            v = X[c].dropna().to_numpy(dtype="float64")
            if v.size < 20:
                continue
            lo = float(np.quantile(v, self.lower_q))
            hi = float(np.quantile(v, self.upper_q))
            if hi > lo:
                self.bounds_[c] = (lo, hi)
        log.info("Đã học ngưỡng cắt ngọn cho %d biến", len(self.bounds_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            if c in out.columns:
                out[c] = out[c].clip(lower=lo, upper=hi)
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:  # noqa: N803
        return self.fit(X, y).transform(X)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"variable": c, "cap_lower": lo, "cap_upper": hi}
             for c, (lo, hi) in self.bounds_.items()]
        )


__all__ = [
    "MissingValueReport",
    "missing_value_report",
    "impute_business_defaults",
    "CIC_DEFAULTS",
    "outlier_report",
    "OutlierCapper",
]
