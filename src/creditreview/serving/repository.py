"""Kho đặc trưng phục vụ tra cứu khi kiểm duyệt một hồ sơ.

Khi tiếp nhận một hồ sơ, vector đặc trưng đầu vào của mô hình có thể
đến từ hai nguồn:

* **Tra cứu** - hồ sơ đã được xử lý trong kho dữ liệu, đặc trưng đã tính sẵn.
  Đây là tình huống của các kịch bản thử nghiệm và của quy trình tái chấm điểm
  danh mục hiện hữu.
* **Tính trực tiếp** - hồ sơ mới, đặc trưng được tính ngay từ dữ liệu gốc bằng
  chính các hàm dùng khi huấn luyện (:mod:`creditreview.features.extraction`).

Lớp :class:`FeatureRepository` đảm nhiệm nguồn thứ nhất. Đặc trưng được nạp một
lần vào bộ nhớ khi dịch vụ khởi động và tra cứu theo chỉ mục băm, nhờ vậy thời
gian tra cứu không đáng kể so với thời gian chấm điểm mô hình.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FeatureRepository:
    """Tra cứu vector đặc trưng đã tính theo mã hồ sơ hoặc mã khách hàng."""

    def __init__(self, features_path: str | Path, columns: Optional[List[str]] = None) -> None:
        self.path = Path(features_path)
        self._by_application: Dict[str, Dict[str, Any]] = {}
        self._by_customer: Dict[str, str] = {}
        self.columns: List[str] = list(columns or [])
        self.n_lookups = 0
        self.n_hits = 0
        if self.path.exists():
            self._load()
        else:
            log.warning("Không tìm thấy kho đặc trưng tại %s", self.path)

    def _load(self) -> None:
        df = pd.read_parquet(self.path)
        keep = [c for c in df.columns if not c.startswith("_")]
        df = df[keep]
        records = df.to_dict(orient="records")
        for rec in records:
            app_id = str(rec.get("application_id", ""))
            cus_id = str(rec.get("customer_id", ""))
            clean = {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in rec.items()
            }
            self._by_application[app_id] = clean
            # giữ hồ sơ gần nhất của mỗi khách hàng
            self._by_customer[cus_id] = app_id
        if not self.columns:
            self.columns = [c for c in keep if c not in ("application_id", "customer_id")]
        log.info(
            "Đã nạp kho đặc trưng: %s hồ sơ, %d cột",
            f"{len(self._by_application):,}", len(self.columns),
        )

    # -- tra cứu -----------------------------------------------------------
    def get_by_application(self, application_id: str) -> Optional[Dict[str, Any]]:
        self.n_lookups += 1
        rec = self._by_application.get(str(application_id))
        if rec is not None:
            self.n_hits += 1
        return rec

    def get_by_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        app_id = self._by_customer.get(str(customer_id))
        return self.get_by_application(app_id) if app_id else None

    def sample_ids(self, n: int = 10) -> List[str]:
        return list(self._by_application.keys())[:n]

    @property
    def size(self) -> int:
        return len(self._by_application)

    def stats(self) -> Dict[str, Any]:
        return {
            "n_records": self.size,
            "n_lookups": self.n_lookups,
            "hit_rate": round(self.n_hits / max(self.n_lookups, 1), 4),
        }


__all__ = ["FeatureRepository"]
