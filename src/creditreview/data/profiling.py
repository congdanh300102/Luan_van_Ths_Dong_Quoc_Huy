"""Phân tích thống kê mô tả và kiểm tra tính hợp lệ của tập dữ liệu tín dụng.

Module phục vụ hai nội dung của bước thu thập và tổng hợp dữ liệu:

* **Thống kê mô tả** - lập bảng đặc tả từng biến (kiểu dữ liệu, tỷ lệ khuyết,
  số giá trị phân biệt, các phân vị) làm căn cứ cho các quyết định tiền xử lý ở
  bước sau. Trong bài toán tín dụng, bảng mô tả còn có vai trò nghiệp vụ: nó cho
  biết biến nào đủ độ phủ để đưa vào mô hình và biến nào cần xử lý riêng.

* **Phát hiện bản ghi không hợp lệ** - kiểm tra các ràng buộc nghiệp vụ bắt buộc
  (tuổi, thu nhập, kỳ hạn, lãi suất, quan hệ thời gian giữa hồ sơ và bản tin
  CIC...). Khác với ngoại lệ thống kê -- vốn có thể là giá trị thật và mang
  thông tin rủi ro -- bản ghi vi phạm ràng buộc nghiệp vụ là dữ liệu *sai*, phải
  được loại bỏ hoặc sửa trước khi đưa vào mô hình.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ===========================================================================
# 1. Thống kê mô tả
# ===========================================================================
def describe_dataset(
    df: pd.DataFrame, exclude_prefix: str = "_", max_percentiles: bool = True
) -> pd.DataFrame:
    """Lập bảng mô tả cho từng biến trong khung dữ liệu.

    Các cột bắt đầu bằng ``exclude_prefix`` là biến nội bộ của bộ sinh dữ liệu
    (biến tiềm ẩn dùng để kiểm chứng), không thuộc dữ liệu nghiệp vụ nên được
    loại khỏi bảng mô tả.
    """
    cols = [c for c in df.columns if not c.startswith(exclude_prefix)]
    rows: List[Dict[str, object]] = []

    for c in cols:
        s = df[c]
        n_missing = int(s.isna().sum())
        row: Dict[str, object] = {
            "variable": c,
            "dtype": str(s.dtype),
            "n": int(len(s)),
            "n_missing": n_missing,
            "missing_pct": round(100.0 * n_missing / max(len(s), 1), 3),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = s.dropna().to_numpy(dtype="float64")
            if v.size:
                row.update(
                    {
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
                        "min": float(np.min(v)),
                        "p25": float(np.percentile(v, 25)),
                        "median": float(np.percentile(v, 50)),
                        "p75": float(np.percentile(v, 75)),
                        "max": float(np.max(v)),
                    }
                )
                if max_percentiles:
                    row["p01"] = float(np.percentile(v, 1))
                    row["p99"] = float(np.percentile(v, 99))
        else:
            vc = s.value_counts(dropna=True)
            if len(vc):
                row["top_value"] = str(vc.index[0])
                row["top_freq_pct"] = round(100.0 * float(vc.iloc[0]) / max(len(s), 1), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def dataset_summary_table(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    transactions_count: int,
    applications: pd.DataFrame,
    cic_reports: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Bảng tổng hợp quy mô các bảng dữ liệu đã thu thập."""
    rows = [
        ("Khách hàng", len(customers), "Thông tin nhân khẩu học, nghề nghiệp, thu nhập"),
        ("Tài khoản", len(accounts), "Tài khoản thanh toán/tiết kiệm và số dư bình quân"),
        ("Giao dịch", transactions_count, "Giao dịch tài khoản dùng để tính đặc trưng hành vi"),
        ("Hồ sơ vay", len(applications), "Hồ sơ đề nghị cấp tín dụng"),
        ("Bản tin CIC", len(cic_reports), "Thông tin lịch sử tín dụng từ CIC"),
        ("Nhãn rủi ro", len(labels), "Trạng thái vỡ nợ sau cửa sổ quan sát"),
    ]
    return pd.DataFrame(rows, columns=["Bảng dữ liệu", "Số bản ghi", "Nội dung"])


# ===========================================================================
# 2. Phát hiện bản ghi không hợp lệ theo ràng buộc nghiệp vụ
# ===========================================================================
# Mỗi ràng buộc: (mã, mô tả, hàm trả về mặt nạ các bản ghi VI PHẠM)
_APPLICATION_CONSTRAINTS: List[Tuple[str, str, str]] = [
    ("APP_AMOUNT_POSITIVE", "Số tiền đề nghị vay phải lớn hơn 0", "requested_amount <= 0"),
    # Kỳ hạn tối đa 360 tháng (30 năm) theo sản phẩm cho vay mua nhà dài hạn
    ("APP_TENOR_RANGE", "Kỳ hạn vay phải nằm trong khoảng 3-360 tháng",
     "tenor_months < 3 or tenor_months > 360"),
    ("APP_RATE_RANGE", "Lãi suất phải nằm trong khoảng 0-40%/năm",
     "interest_rate <= 0 or interest_rate > 0.40"),
    ("APP_INSTALLMENT_POSITIVE", "Khoản trả góp hàng tháng phải lớn hơn 0",
     "monthly_installment <= 0"),
    # LTV vượt 1,2 là dấu hiệu lỗi nhập liệu; mức 0,85 là ngưỡng *chính sách*
    # và được xử lý bởi luật nghiệp vụ chứ không phải bởi kiểm tra dữ liệu.
    ("APP_LTV_RANGE", "Tỷ lệ vay trên giá trị TSBĐ không được vượt 1,2",
     "ltv < 0 or ltv > 1.2"),
]

_CUSTOMER_CONSTRAINTS: List[Tuple[str, str, str]] = [
    ("CUS_AGE_RANGE", "Tuổi khách hàng phải nằm trong khoảng 18-75", "age < 18 or age > 75"),
    ("CUS_INCOME_POSITIVE", "Thu nhập khai báo phải lớn hơn 0", "monthly_income <= 0"),
    ("CUS_TENURE_NONNEG", "Thâm niên công việc không được âm", "job_tenure_months < 0"),
    ("CUS_DEPENDENTS_RANGE", "Số người phụ thuộc phải nằm trong khoảng 0-10",
     "num_dependents < 0 or num_dependents > 10"),
]

_CIC_CONSTRAINTS: List[Tuple[str, str, str]] = [
    ("CIC_SCORE_RANGE", "Điểm CIC phải nằm trong khoảng 300-850",
     "cic_score < 300 or cic_score > 850"),
    ("CIC_GROUP_RANGE", "Nhóm nợ phải nằm trong khoảng 1-5",
     "current_group < 1 or current_group > 5"),
    ("CIC_GROUP_ORDER", "Nhóm nợ xấu nhất 12 tháng không thể tốt hơn nhóm nợ hiện tại",
     "worst_group_12m < current_group"),
    ("CIC_DPD_NONNEG", "Số ngày quá hạn không được âm", "dpd_max_12m < 0"),
    ("CIC_UTILIZATION_RANGE", "Tỷ lệ sử dụng hạn mức phải nằm trong khoảng 0-2",
     "credit_utilization < 0 or credit_utilization > 2"),
]


def _apply_constraints(
    df: pd.DataFrame, constraints: Sequence[Tuple[str, str, str]], table: str
) -> pd.DataFrame:
    rows = []
    for code, desc, expr in constraints:
        try:
            mask = df.eval(expr)
        except Exception as exc:  # cột không tồn tại trong khung dữ liệu
            log.debug("Bỏ qua ràng buộc %s: %s", code, exc)
            continue
        mask = mask.fillna(False).to_numpy(dtype=bool)
        rows.append(
            {
                "table": table,
                "code": code,
                "description": desc,
                "n_violations": int(mask.sum()),
                "violation_pct": round(100.0 * float(mask.sum()) / max(len(df), 1), 4),
            }
        )
    return pd.DataFrame(rows)


def detect_invalid_records(
    customers: pd.DataFrame,
    applications: pd.DataFrame,
    cic_reports: pd.DataFrame,
) -> pd.DataFrame:
    """Kiểm tra toàn bộ ràng buộc nghiệp vụ trên ba bảng dữ liệu chính.

    Trả về bảng liệt kê số bản ghi vi phạm từng ràng buộc. Bảng này được đưa vào
    phần trình bày kết quả kiểm tra chất lượng dữ liệu của đề án.
    """
    parts = [
        _apply_constraints(customers, _CUSTOMER_CONSTRAINTS, "customers"),
        _apply_constraints(applications, _APPLICATION_CONSTRAINTS, "applications"),
        _apply_constraints(cic_reports, _CIC_CONSTRAINTS, "cic_reports"),
    ]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    # kiểm tra quan hệ thời gian giữa hồ sơ và bản tin CIC
    if {"application_id", "query_time"}.issubset(cic_reports.columns) and \
       {"application_id", "apply_time"}.issubset(applications.columns):
        merged = applications[["application_id", "apply_time"]].merge(
            cic_reports[["application_id", "query_time"]], on="application_id", how="inner"
        )
        late = merged["query_time"] > merged["apply_time"]
        out = pd.concat(
            [
                out,
                pd.DataFrame([{
                    "table": "cic_reports",
                    "code": "CIC_QUERY_BEFORE_APPLY",
                    "description": "Bản tin CIC phải được truy vấn không muộn hơn thời điểm nộp hồ sơ",
                    "n_violations": int(late.sum()),
                    "violation_pct": round(100.0 * float(late.sum()) / max(len(merged), 1), 4),
                }]),
            ],
            ignore_index=True,
        )

    n_bad = int(out["n_violations"].sum())
    log.info(
        "Kiểm tra tính hợp lệ: %d/%d ràng buộc phát hiện vi phạm, tổng %s bản ghi",
        int((out["n_violations"] > 0).sum()), len(out), f"{n_bad:,}",
    )
    return out


# ===========================================================================
# 3. Kiểm chứng cơ chế phát hiện bằng phép tiêm lỗi
# ===========================================================================
def inject_quality_issues(
    customers: pd.DataFrame,
    applications: pd.DataFrame,
    cic_reports: pd.DataFrame,
    rate: float = 0.01,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cố ý tạo lỗi dữ liệu trên một bản sao để kiểm chứng cơ chế phát hiện.

    Nếu dữ liệu đầu vào vốn đã sạch, bảng kiểm tra ràng buộc sẽ toàn số 0 và
    không chứng minh được rằng cơ chế kiểm tra thực sự hoạt động. Hàm này tạo
    một bản sao có lỗi đã biết trước, chạy lại cơ chế kiểm tra và đối chiếu số
    lỗi phát hiện được với số lỗi đã tiêm vào.

    Trả về ``(customers, applications, cic_reports, injected)`` trong đó
    ``injected`` là bảng ghi nhận số lỗi đã tiêm theo từng mã ràng buộc.
    """
    rng = np.random.default_rng(seed)
    cus, apps, cic = customers.copy(), applications.copy(), cic_reports.copy()
    records: List[Dict[str, object]] = []

    def _pick(df: pd.DataFrame) -> np.ndarray:
        n = max(int(len(df) * rate), 1)
        return rng.choice(len(df), size=n, replace=False)

    idx = _pick(cus)
    cus.loc[cus.index[idx], "age"] = 15
    records.append({"code": "CUS_AGE_RANGE", "n_injected": len(idx)})

    idx = _pick(cus)
    cus.loc[cus.index[idx], "monthly_income"] = -1.0
    records.append({"code": "CUS_INCOME_POSITIVE", "n_injected": len(idx)})

    idx = _pick(apps)
    apps.loc[apps.index[idx], "requested_amount"] = 0.0
    records.append({"code": "APP_AMOUNT_POSITIVE", "n_injected": len(idx)})

    idx = _pick(apps)
    apps.loc[apps.index[idx], "interest_rate"] = 0.95
    records.append({"code": "APP_RATE_RANGE", "n_injected": len(idx)})

    idx = _pick(cic)
    cic.loc[cic.index[idx], "cic_score"] = 1200.0
    records.append({"code": "CIC_SCORE_RANGE", "n_injected": len(idx)})

    idx = _pick(cic)
    cic.loc[cic.index[idx], "current_group"] = 5
    cic.loc[cic.index[idx], "worst_group_12m"] = 1
    records.append({"code": "CIC_GROUP_ORDER", "n_injected": len(idx)})

    return cus, apps, cic, pd.DataFrame(records)


def verify_detection(
    injected: pd.DataFrame, detected: pd.DataFrame
) -> pd.DataFrame:
    """Đối chiếu số lỗi đã tiêm với số lỗi cơ chế kiểm tra phát hiện được."""
    merged = injected.merge(
        detected[["code", "description", "n_violations"]], on="code", how="left"
    )
    merged["n_violations"] = merged["n_violations"].fillna(0).astype(int)
    merged["detected_pct"] = (
        100.0 * merged["n_violations"] / merged["n_injected"].clip(lower=1)
    ).round(2)
    merged["result"] = np.where(
        merged["n_violations"] >= merged["n_injected"], "Phát hiện đủ", "Bỏ sót"
    )
    return merged.rename(
        columns={
            "code": "Mã ràng buộc",
            "description": "Nội dung ràng buộc",
            "n_injected": "Số lỗi tiêm vào",
            "n_violations": "Số lỗi phát hiện",
            "detected_pct": "Tỷ lệ phát hiện (%)",
            "result": "Kết quả",
        }
    )


__all__ = [
    "describe_dataset",
    "dataset_summary_table",
    "detect_invalid_records",
    "inject_quality_issues",
    "verify_detection",
]
