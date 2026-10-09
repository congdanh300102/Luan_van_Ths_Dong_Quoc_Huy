"""Đánh giá mức độ quan trọng và lựa chọn tập đặc trưng tối ưu.

Tập đặc trưng ban đầu của mô hình gồm hơn 150 biến, trong đó có nhiều biến
được sinh theo cùng một mẫu trên các cửa sổ thời gian khác nhau (7, 30, 90, 180
ngày) nên tương quan rất mạnh với nhau. Đưa toàn bộ vào mô hình gây ba vấn đề:
hệ số của mô hình tuyến tính mất ổn định do đa cộng tuyến, chi phí tính toán
tăng, và khả năng diễn giải kết quả giảm.

Quy trình lựa chọn gồm ba bộ lọc nối tiếp, sắp xếp theo thứ tự từ rẻ đến đắt về
chi phí tính toán:

1. **Lọc theo năng lực dự báo** (Information Value). Loại các biến có IV dưới
   ngưỡng -- theo thực hành của ngành, IV < 0,02 được coi là không có khả năng
   phân biệt. Đồng thời cảnh báo các biến có IV quá cao (> 0,5), vì đó thường là
   dấu hiệu rò rỉ nhãn chứ không phải biến thực sự tốt.

2. **Lọc theo tương quan**. Trong mỗi cặp biến có hệ số tương quan tuyệt đối
   vượt ngưỡng, giữ lại biến có IV cao hơn.

3. **Lọc theo độ ổn định** (Population Stability Index). Loại các biến có phân
   phối thay đổi mạnh giữa giai đoạn huấn luyện và giai đoạn kiểm thử, vì mô
   hình phụ thuộc vào biến không ổn định sẽ suy giảm nhanh khi vận hành.

Tập đặc trưng sau lựa chọn được so sánh trực tiếp với tập đầy đủ trên cùng một
mô hình để định lượng ảnh hưởng của việc rút gọn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Ngưỡng diễn giải Information Value theo thực hành ngành tín dụng
IV_USELESS = 0.02
IV_WEAK = 0.10
IV_MEDIUM = 0.30
IV_STRONG = 0.50


def iv_strength(iv: float) -> str:
    """Diễn giải định tính giá trị IV."""
    if iv < IV_USELESS:
        return "Không có khả năng phân biệt"
    if iv < IV_WEAK:
        return "Yếu"
    if iv < IV_MEDIUM:
        return "Trung bình"
    if iv < IV_STRONG:
        return "Mạnh"
    return "Rất mạnh (cần kiểm tra rò rỉ nhãn)"


# ===========================================================================
# Bộ lọc 1: năng lực dự báo
# ===========================================================================
def iv_filter(
    iv_table: pd.DataFrame, min_iv: float = IV_USELESS, max_iv: float = 1.0
) -> List[str]:
    """Giữ các đặc trưng có IV nằm trong khoảng chấp nhận được."""
    t = iv_table.dropna(subset=["iv"])
    keep = t[(t["iv"] >= min_iv) & (t["iv"] <= max_iv)]["feature"].tolist()
    dropped = set(t["feature"]) - set(keep)
    log.info(
        "Lọc theo IV (∈ [%.3f; %.2f]): giữ %d, loại %d đặc trưng",
        min_iv, max_iv, len(keep), len(dropped),
    )
    return keep


# ===========================================================================
# Bộ lọc 2: tương quan
# ===========================================================================
def correlation_filter(
    df: pd.DataFrame,
    features: Sequence[str],
    importance: Dict[str, float],
    threshold: float = 0.95,
    max_features: int | None = None,
) -> tuple[List[str], pd.DataFrame]:
    """Loại bỏ đặc trưng dư thừa trong các cặp có tương quan rất cao.

    Với mỗi cặp ``(a, b)`` mà ``|corr(a, b)| > threshold``, biến có giá trị
    ``importance`` thấp hơn bị loại. Chỉ xét biến định lượng; biến định tính
    được giữ nguyên vì hệ số tương quan Pearson không áp dụng được.

    Trả về danh sách đặc trưng giữ lại và bảng ghi nhận các cặp đã xử lý.
    """
    numeric = [
        c for c in features
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    other = [c for c in features if c not in numeric]
    if len(numeric) < 2:
        return list(features), pd.DataFrame()

    # sắp theo tầm quan trọng giảm dần để biến mạnh hơn luôn được xét trước
    numeric = sorted(numeric, key=lambda c: -float(importance.get(c, 0.0)))
    if max_features is not None:
        numeric = numeric[: max(max_features * 3, max_features)]

    corr = df[numeric].corr(numeric_only=True).abs().to_numpy()
    n = len(numeric)
    dropped: Dict[str, str] = {}
    records: List[Dict[str, object]] = []

    for i in range(n):
        if numeric[i] in dropped:
            continue
        for j in range(i + 1, n):
            if numeric[j] in dropped:
                continue
            r = corr[i, j]
            if not np.isfinite(r) or r <= threshold:
                continue
            dropped[numeric[j]] = numeric[i]
            records.append(
                {
                    "feature_dropped": numeric[j],
                    "feature_kept": numeric[i],
                    "abs_correlation": round(float(r), 4),
                    "iv_dropped": round(float(importance.get(numeric[j], 0.0)), 4),
                    "iv_kept": round(float(importance.get(numeric[i], 0.0)), 4),
                }
            )

    keep = [c for c in numeric if c not in dropped] + other
    log.info(
        "Lọc theo tương quan (|r| > %.2f): loại %d đặc trưng dư thừa",
        threshold, len(dropped),
    )
    return keep, pd.DataFrame(records)


# ===========================================================================
# Bộ lọc 3: độ ổn định theo thời gian
# ===========================================================================
def stability_filter(
    features: Sequence[str], psi_table: pd.DataFrame, max_psi: float = 0.25
) -> List[str]:
    """Loại các đặc trưng có PSI vượt ngưỡng báo động giữa hai giai đoạn."""
    if psi_table is None or psi_table.empty or "psi" not in psi_table.columns:
        return list(features)
    unstable = set(
        psi_table.loc[psi_table["psi"] > max_psi, "feature"].tolist()
    )
    keep = [f for f in features if f not in unstable]
    log.info(
        "Lọc theo độ ổn định (PSI ≤ %.2f): loại %d đặc trưng không ổn định",
        max_psi, len(features) - len(keep),
    )
    return keep


# ===========================================================================
# Quy trình lựa chọn hoàn chỉnh
# ===========================================================================
@dataclass
class SelectionResult:
    """Kết quả lựa chọn tập đặc trưng."""

    selected: List[str]
    steps: pd.DataFrame
    correlation_pairs: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_selected(self) -> int:
        return len(self.selected)


def select_features(
    df: pd.DataFrame,
    features: Sequence[str],
    iv_table: pd.DataFrame,
    psi_table: pd.DataFrame | None = None,
    min_iv: float = IV_USELESS,
    max_iv: float = 1.0,
    corr_threshold: float = 0.95,
    max_psi: float = 0.25,
) -> SelectionResult:
    """Chạy tuần tự ba bộ lọc và trả về tập đặc trưng được giữ lại.

    Tham số ``df`` là tập **huấn luyện** -- mọi thống kê phục vụ lựa chọn đều
    phải được tính trên tập này để tránh rò rỉ thông tin từ tập kiểm thử.
    """
    steps: List[Dict[str, object]] = [
        {"Bước": "Tập đặc trưng ban đầu", "Số đặc trưng": len(features), "Đã loại": 0}
    ]

    importance = dict(zip(iv_table["feature"], iv_table["iv"])) if len(iv_table) else {}

    # bước 1 - IV
    keep_iv = set(iv_filter(iv_table, min_iv=min_iv, max_iv=max_iv))
    cur = [f for f in features if f in keep_iv or f not in importance]
    steps.append(
        {"Bước": f"Sau lọc theo IV (≥ {min_iv:.2f})",
         "Số đặc trưng": len(cur), "Đã loại": len(features) - len(cur)}
    )

    # bước 2 - tương quan
    before = len(cur)
    cur, pairs = correlation_filter(df, cur, importance, threshold=corr_threshold)
    steps.append(
        {"Bước": f"Sau lọc theo tương quan (|r| ≤ {corr_threshold:.2f})",
         "Số đặc trưng": len(cur), "Đã loại": before - len(cur)}
    )

    # bước 3 - độ ổn định
    before = len(cur)
    if psi_table is not None:
        cur = stability_filter(cur, psi_table, max_psi=max_psi)
    steps.append(
        {"Bước": f"Sau lọc theo độ ổn định (PSI ≤ {max_psi:.2f})",
         "Số đặc trưng": len(cur), "Đã loại": before - len(cur)}
    )

    log.info(
        "Lựa chọn đặc trưng: %d -> %d (giữ lại %.1f%%)",
        len(features), len(cur), 100.0 * len(cur) / max(len(features), 1),
    )
    return SelectionResult(
        selected=cur, steps=pd.DataFrame(steps), correlation_pairs=pairs
    )


__all__ = [
    "select_features",
    "SelectionResult",
    "iv_filter",
    "correlation_filter",
    "stability_filter",
    "iv_strength",
    "IV_USELESS",
    "IV_WEAK",
    "IV_MEDIUM",
    "IV_STRONG",
]
