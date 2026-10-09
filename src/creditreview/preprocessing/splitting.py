"""Phân chia tập dữ liệu huấn luyện - kiểm định - kiểm thử.

CHIẾN LƯỢC CHIA THEO THỜI GIAN (OUT-OF-TIME)
--------------------------------------------
Đề án chia dữ liệu theo trục thời gian nộp hồ sơ thay vì chia ngẫu nhiên. Lý do
xuất phát từ điều kiện vận hành thực tế: mô hình luôn được huấn luyện trên hồ sơ
quá khứ và áp dụng cho hồ sơ tương lai. Chia ngẫu nhiên sẽ trộn lẫn hai giai
đoạn, khiến mô hình gián tiếp "nhìn thấy" phân phối của giai đoạn sau và cho kết
quả đánh giá lạc quan hơn thực tế. Chia theo thời gian phản ánh cả ảnh hưởng của
hiện tượng trôi dạt dữ liệu -- yếu tố quyết định tuổi thọ của mô hình tín dụng.

Ngoài ra module cung cấp phương án chia theo **nhóm khách hàng**: một khách hàng
có thể nộp nhiều hồ sơ, nếu các hồ sơ này rơi vào cả tập huấn luyện lẫn tập kiểm
thử thì thông tin về khách hàng bị rò rỉ giữa hai tập. Phương án này được dùng
để kiểm chứng chéo mức độ ảnh hưởng của rò rỉ theo khách hàng.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def out_of_time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    time_col: str = "apply_time",
) -> Dict[str, pd.DataFrame]:
    """Chia dữ liệu thành ba phần liên tiếp theo thời gian.

        |------- train -------|--- valid ---|--- test ---|
    """
    d = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    n = len(d)
    i_tr = int(n * train_ratio)
    i_va = int(n * (train_ratio + valid_ratio))
    parts = {"train": d.iloc[:i_tr], "valid": d.iloc[i_tr:i_va], "test": d.iloc[i_va:]}
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def stratified_customer_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    customer_col: str = "customer_id",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Chia dữ liệu theo nhóm khách hàng: mọi hồ sơ của một khách hàng nằm cùng một tập.

    Dùng để đối chiếu với cách chia theo thời gian, nhằm tách bạch hai nguồn gây
    chênh lệch kết quả: trôi dạt theo thời gian và rò rỉ theo khách hàng.
    """
    rng = np.random.default_rng(seed)
    customers = df[customer_col].drop_duplicates().to_numpy()
    rng.shuffle(customers)
    n = len(customers)
    i_tr = int(n * train_ratio)
    i_va = int(n * (train_ratio + valid_ratio))
    groups = {
        "train": set(customers[:i_tr]),
        "valid": set(customers[i_tr:i_va]),
        "test": set(customers[i_va:]),
    }
    return {
        k: df[df[customer_col].isin(v)].reset_index(drop=True) for k, v in groups.items()
    }


def split_report(
    parts: Dict[str, pd.DataFrame],
    time_col: str = "apply_time",
    target_col: str = "default_flag",
) -> pd.DataFrame:
    """Bảng mô tả kết quả phân chia: quy mô, khoảng thời gian và tỷ lệ nợ xấu."""
    rows = []
    total = sum(len(v) for v in parts.values())
    for name, part in parts.items():
        row: Dict[str, object] = {
            "Tập dữ liệu": name,
            "Số hồ sơ": len(part),
            "Tỷ trọng (%)": round(100.0 * len(part) / max(total, 1), 2),
        }
        if time_col in part.columns and len(part):
            row["Từ ngày"] = str(pd.Timestamp(part[time_col].min()).date())
            row["Đến ngày"] = str(pd.Timestamp(part[time_col].max()).date())
        if target_col in part.columns and len(part):
            row["Tỷ lệ nợ xấu (%)"] = round(100.0 * float(part[target_col].mean()), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    log.info("Kết quả phân chia dữ liệu:\n%s", out.to_string(index=False))
    return out


__all__ = ["out_of_time_split", "stratified_customer_split", "split_report"]
