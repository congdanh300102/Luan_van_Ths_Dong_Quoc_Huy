"""Định nghĩa trạng thái rủi ro, tiêu chuẩn nợ xấu và phương pháp gán nhãn.

TIÊU CHUẨN NỢ XẤU
-----------------
Đề án sử dụng định nghĩa nợ xấu theo quy định phân loại nợ hiện hành của Ngân
hàng Nhà nước: khoản vay bị coi là **nợ xấu** khi được phân vào nhóm 3, 4 hoặc 5
-- tương ứng với số ngày quá hạn từ 90 ngày trở lên. Đây cũng là ngưỡng mà chuẩn
mực Basel sử dụng cho định nghĩa *default*, nhờ đó kết quả của đề án có thể đối
chiếu với các nghiên cứu quốc tế.

CỬA SỔ QUAN SÁT
---------------
Một hồ sơ chỉ được gán nhãn sau khi đã trải qua **cửa sổ quan sát hiệu quả trả
nợ** (*performance window*) đủ dài. Đề án chọn cửa sổ 12 tháng, dựa trên hai căn
cứ:

* Cửa sổ quá ngắn (6 tháng) bỏ sót phần lớn trường hợp vỡ nợ muộn, làm tỷ lệ nợ
  xấu quan sát được thấp hơn thực tế và mô hình bị lệch về phía lạc quan.
* Cửa sổ quá dài (24 tháng) buộc phải loại bỏ nhiều hồ sơ gần đây khỏi tập huấn
  luyện, làm dữ liệu cũ đi và mô hình kém phản ánh điều kiện hiện tại.

Hàm :func:`vintage_analysis` cung cấp căn cứ định lượng cho lựa chọn này bằng
cách đo tỷ lệ nợ xấu tích luỹ theo từng tháng kể từ thời điểm cấp tín dụng.

BA TRẠNG THÁI THAY VÌ HAI
-------------------------
Ngoài hai trạng thái Tốt/Xấu, quy trình gán nhãn còn nhận diện trạng thái
**chưa xác định** cho các hồ sơ chưa đi hết cửa sổ quan sát tại thời điểm trích
xuất dữ liệu. Các hồ sơ này bị loại khỏi tập huấn luyện: giữ lại và coi như hồ
sơ tốt sẽ đưa nhãn sai vào dữ liệu, làm mô hình đánh giá thấp rủi ro của những
hồ sơ mới nhất.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Ngưỡng số ngày quá hạn xác định nợ xấu
BAD_DEBT_DPD = 90

# Nhóm nợ được xếp vào nợ xấu theo quy định phân loại nợ
BAD_DEBT_GROUPS = (3, 4, 5)


class RiskStatus(str, Enum):
    """Trạng thái rủi ro của một hồ sơ sau cửa sổ quan sát."""

    GOOD = "GOOD"                    # không phát sinh nợ xấu trong cửa sổ
    BAD = "BAD"                      # phát sinh nợ xấu (DPD >= 90 / nhóm 3-5)
    INDETERMINATE = "INDETERMINATE"  # chưa đủ cửa sổ quan sát -> loại khỏi mô hình


@dataclass(frozen=True)
class LabelDefinition:
    """Đặc tả đầy đủ của quy tắc gán nhãn, dùng để ghi vào tài liệu thực nghiệm."""

    performance_window_months: int = 12
    bad_debt_dpd: int = BAD_DEBT_DPD
    bad_debt_groups: tuple = BAD_DEBT_GROUPS
    target_column: str = "default_flag"

    def describe(self) -> Dict[str, object]:
        return {
            "Cửa sổ quan sát (tháng)": self.performance_window_months,
            "Ngưỡng nợ xấu (ngày quá hạn)": self.bad_debt_dpd,
            "Nhóm nợ xấu": ", ".join(str(g) for g in self.bad_debt_groups),
            "Biến mục tiêu": self.target_column,
        }


# ===========================================================================
# Gán nhãn
# ===========================================================================
def assign_labels(
    labels: pd.DataFrame,
    definition: LabelDefinition | None = None,
    data_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Gán trạng thái rủi ro cho từng hồ sơ theo quy tắc đã định nghĩa.

    Tham số
    -------
    labels
        Bảng nhãn thô, cần có ``application_id``, ``observation_time``,
        ``performance_end``, ``months_to_default`` và ``default_flag``.
    data_cutoff
        Thời điểm trích xuất dữ liệu. Hồ sơ có ``performance_end`` vượt quá mốc
        này chưa đi hết cửa sổ quan sát nên được gán ``INDETERMINATE``. Mặc định
        lấy giá trị ``performance_end`` lớn nhất trong dữ liệu.

    Trả về bảng gốc bổ sung hai cột ``risk_status`` và ``is_modelable``.
    """
    d = definition or LabelDefinition()
    out = labels.copy()

    if data_cutoff is None:
        data_cutoff = pd.Timestamp(out["performance_end"].max())

    perf_end = pd.to_datetime(out["performance_end"])
    incomplete = perf_end > pd.Timestamp(data_cutoff)

    # Vỡ nợ được ghi nhận khi xảy ra *trong* cửa sổ quan sát. Trường hợp vỡ nợ
    # muộn hơn cửa sổ không thuộc phạm vi dự báo của mô hình.
    m2d = out["months_to_default"].to_numpy(dtype="float64")
    is_bad = (out["default_flag"].to_numpy() == 1) & (
        np.nan_to_num(m2d, nan=np.inf) <= d.performance_window_months
    )

    status = np.where(incomplete.to_numpy(), RiskStatus.INDETERMINATE.value,
                      np.where(is_bad, RiskStatus.BAD.value, RiskStatus.GOOD.value))
    out["risk_status"] = status
    out["is_modelable"] = out["risk_status"] != RiskStatus.INDETERMINATE.value
    # Đồng bộ biến mục tiêu với trạng thái vừa gán
    out[d.target_column] = (out["risk_status"] == RiskStatus.BAD.value).astype(int)

    n_ind = int((~out["is_modelable"]).sum())
    log.info(
        "Gán nhãn %s hồ sơ: %s xấu (%.2f%%), %s tốt, %s chưa đủ cửa sổ quan sát",
        f"{len(out):,}",
        f"{int((out['risk_status'] == RiskStatus.BAD.value).sum()):,}",
        100.0 * float(out.loc[out['is_modelable'], d.target_column].mean() or 0.0),
        f"{int((out['risk_status'] == RiskStatus.GOOD.value).sum()):,}",
        f"{n_ind:,}",
    )
    return out


def label_distribution(labeled: pd.DataFrame) -> pd.DataFrame:
    """Bảng phân bố ba trạng thái rủi ro."""
    vc = labeled["risk_status"].value_counts()
    total = int(vc.sum())
    name_vi = {
        RiskStatus.GOOD.value: "Tốt (không phát sinh nợ xấu)",
        RiskStatus.BAD.value: "Xấu (nợ nhóm 3-5, quá hạn ≥ 90 ngày)",
        RiskStatus.INDETERMINATE.value: "Chưa xác định (chưa đủ cửa sổ quan sát)",
    }
    rows = [
        {
            "Trạng thái": name_vi.get(k, k),
            "Số hồ sơ": int(v),
            "Tỷ trọng (%)": round(100.0 * int(v) / max(total, 1), 2),
        }
        for k, v in vc.items()
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# Phân tích vintage: căn cứ lựa chọn độ dài cửa sổ quan sát
# ===========================================================================
def vintage_analysis(labels: pd.DataFrame, max_months: int = 18) -> pd.DataFrame:
    """Tỷ lệ nợ xấu tích luỹ theo số tháng kể từ thời điểm cấp tín dụng.

    Đường cong tích luỹ cho biết cửa sổ quan sát cần dài bao nhiêu để ghi nhận
    được phần lớn trường hợp vỡ nợ. Điểm mà đường cong bắt đầu đi ngang chính là
    độ dài cửa sổ hợp lý.
    """
    n = len(labels)
    m2d = labels["months_to_default"].to_numpy(dtype="float64")
    rows = []
    for m in range(1, max_months + 1):
        n_bad = int(np.nansum(m2d <= m))
        rows.append(
            {
                "Tháng kể từ khi cấp tín dụng": m,
                "Số hồ sơ phát sinh nợ xấu": n_bad,
                "Tỷ lệ nợ xấu tích luỹ (%)": round(100.0 * n_bad / max(n, 1), 3),
            }
        )
    out = pd.DataFrame(rows)
    total_bad = int(np.nansum(~np.isnan(m2d)))
    if total_bad:
        out["Tỷ lệ ghi nhận được (%)"] = (
            100.0 * out["Số hồ sơ phát sinh nợ xấu"] / total_bad
        ).round(2)
    return out


__all__ = [
    "RiskStatus",
    "LabelDefinition",
    "assign_labels",
    "label_distribution",
    "vintage_analysis",
    "BAD_DEBT_DPD",
    "BAD_DEBT_GROUPS",
]
