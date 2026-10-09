"""Chính sách tín dụng: phân hạng rủi ro, hạn mức và định giá theo rủi ro.

Mô hình học máy chỉ trả về một con số - xác suất vỡ nợ (PD). Để con số đó trở
thành một quyết định tín dụng có thể thực thi, cần thêm ba lớp chính sách:

1. **Phân hạng rủi ro** (*risk grading*) - chuyển PD liên tục thành thang hạng
   rời rạc (AA, A, BBB, ...). Thang hạng là ngôn ngữ chung giữa bộ phận mô hình,
   bộ phận kinh doanh và bộ phận quản trị rủi ro; đồng thời là cơ sở tính dự
   phòng rủi ro theo quy định.

2. **Xác định hạn mức** (*limit setting*) - số tiền được duyệt không nhất thiết
   bằng số tiền đề nghị. Hạn mức bị chặn đồng thời bởi: khả năng trả nợ (DTI mục
   tiêu), hạng rủi ro, và giá trị tài sản bảo đảm. Nguyên tắc là lấy **giá trị
   nhỏ nhất** trong các ràng buộc.

3. **Định giá theo rủi ro** (*risk-based pricing*) - lãi suất được điều chỉnh theo
   hạng rủi ro sao cho phần bù rủi ro đủ bù đắp tổn thất kỳ vọng:

       ``EL = PD × LGD × EAD``

   trong đó ``EL`` là tổn thất kỳ vọng, ``LGD`` là tỷ lệ tổn thất khi vỡ nợ,
   ``EAD`` là dư nợ tại thời điểm vỡ nợ. Đây chính là cách tiếp cận của Basel II
   đối với rủi ro tín dụng.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import Config

log = logging.getLogger(__name__)

# Tỷ lệ tổn thất khi vỡ nợ (LGD) theo nhóm sản phẩm.
# Khoản vay có tài sản bảo đảm có LGD thấp hơn nhiều do khả năng thu hồi từ
# việc xử lý tài sản.
LGD_BY_PRODUCT: Dict[str, float] = {
    "HOME_EQUITY": 0.22,
    "AUTO_LOAN": 0.38,
    "CONSUMER_SEC": 0.42,
    "CONSUMER_UNSEC": 0.72,
    "CREDIT_CARD": 0.78,
    "BNPL": 0.85,
}
DEFAULT_LGD = 0.70

# Hệ số hạn mức tối đa theo hạng rủi ro (nhân với thu nhập hàng tháng)
LIMIT_MULTIPLIER_BY_BAND: Dict[str, float] = {
    "AA": 1.00, "A": 0.95, "BBB": 0.85, "BB": 0.70, "B": 0.50, "CCC": 0.30, "D": 0.0,
}

# Phần bù lãi suất theo hạng rủi ro (điểm phần trăm/năm)
RATE_PREMIUM_BY_BAND: Dict[str, float] = {
    "AA": -0.010, "A": -0.005, "BBB": 0.000, "BB": 0.012, "B": 0.028,
    "CCC": 0.048, "D": 0.0,
}

# DTI mục tiêu khi xác định hạn mức (thấp hơn ngưỡng loại trừ cứng)
TARGET_DTI = 0.45


@dataclass
class RiskBand:
    name: str
    pd_max: float


@dataclass
class PolicyOutcome:
    """Kết quả áp dụng chính sách tín dụng."""

    risk_band: str
    approved_amount: float
    approved_tenor: int
    priced_rate: float
    expected_loss: float
    expected_loss_rate: float
    limit_binding_constraint: str
    amount_reduced: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_band": self.risk_band,
            "approved_amount": self.approved_amount,
            "approved_tenor": self.approved_tenor,
            "priced_rate": round(self.priced_rate, 4),
            "expected_loss": round(self.expected_loss, 0),
            "expected_loss_rate": round(self.expected_loss_rate, 6),
            "limit_binding_constraint": self.limit_binding_constraint,
            "amount_reduced": self.amount_reduced,
        }


class CreditPolicy:
    """Áp dụng chính sách tín dụng lên kết quả chấm điểm."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.bands: List[RiskBand] = [
            RiskBand(b["name"], float(b["pd_max"]))
            for b in cfg.get("decision.risk_bands", [])
        ]
        if not self.bands:  # dự phòng nếu cấu hình thiếu
            self.bands = [
                RiskBand("AA", 0.02), RiskBand("A", 0.05), RiskBand("BBB", 0.08),
                RiskBand("BB", 0.15), RiskBand("B", 0.25), RiskBand("CCC", 0.40),
                RiskBand("D", 1.01),
            ]
        self.pd_approve_max = float(cfg.get("decision.pd_approve_max", 0.08))
        self.pd_reject_min = float(cfg.get("decision.pd_reject_min", 0.32))

    # -- phân hạng ---------------------------------------------------------
    def risk_band(self, pd_value: float) -> str:
        """Chuyển PD thành hạng rủi ro."""
        for b in self.bands:
            if pd_value < b.pd_max:
                return b.name
        return self.bands[-1].name

    # -- hạn mức -----------------------------------------------------------
    def compute_limit(
        self, features: Dict[str, Any], band: str, requested: float
    ) -> tuple[float, str]:
        """Xác định số tiền được duyệt và ràng buộc quyết định.

        Ba ràng buộc được xét đồng thời, kết quả là giá trị nhỏ nhất:
          (a) khả năng trả nợ theo DTI mục tiêu;
          (b) hệ số hạn mức theo hạng rủi ro;
          (c) tỷ lệ cho vay tối đa trên giá trị tài sản bảo đảm.
        """
        income = _f(features.get("monthly_income"), 0.0)
        existing = _f(features.get("monthly_debt_payment"), 0.0)
        tenor = max(int(_f(features.get("tenor_months"), 12)), 1)
        rate = _f(features.get("interest_rate"), 0.15)

        candidates: List[tuple[float, str]] = [(requested, "Số tiền đề nghị")]

        # (a) khả năng trả nợ: khoản trả góp tối đa cho phép theo DTI mục tiêu
        max_installment = max(TARGET_DTI * income - existing, 0.0)
        if max_installment > 0:
            r = rate / 12.0
            # đảo công thức niên kim để suy ra dư nợ gốc tối đa
            factor = (1.0 - (1.0 + r) ** (-tenor)) / r if r > 1e-12 else float(tenor)
            candidates.append((max_installment * factor, "Khả năng trả nợ (DTI mục tiêu)"))
        else:
            candidates.append((0.0, "Khả năng trả nợ (DTI mục tiêu)"))

        # (b) hệ số theo hạng rủi ro
        mult = LIMIT_MULTIPLIER_BY_BAND.get(band, 0.5)
        candidates.append((requested * mult, f"Hệ số hạn mức theo hạng {band}"))

        # (c) tài sản bảo đảm
        if _f(features.get("has_collateral"), 0) >= 1:
            coll = _f(features.get("collateral_value"), 0.0)
            max_ltv = float(self.cfg.get("decision.rules.max_ltv", 0.85))
            if coll > 0:
                candidates.append((coll * max_ltv, "Giá trị tài sản bảo đảm"))

        amount, constraint = min(candidates, key=lambda t: t[0])
        # làm tròn xuống đến triệu VND cho phù hợp thực tế giải ngân
        amount = float(int(max(amount, 0.0) / 1e6) * 1e6)
        return amount, constraint

    # -- định giá ----------------------------------------------------------
    def price(self, features: Dict[str, Any], band: str, pd_value: float) -> tuple[float, float, float]:
        """Xác định lãi suất và tổn thất kỳ vọng.

        Trả về ``(lãi suất, tổn thất kỳ vọng tuyệt đối, tỷ lệ tổn thất kỳ vọng)``.
        """
        base_rate = _f(features.get("interest_rate"), 0.15)
        premium = RATE_PREMIUM_BY_BAND.get(band, 0.03)
        product = str(features.get("product_code", ""))
        lgd = LGD_BY_PRODUCT.get(product, DEFAULT_LGD)

        # EAD xấp xỉ bằng số tiền vay (giả định vỡ nợ sớm trong kỳ hạn)
        ead = _f(features.get("requested_amount"), 0.0)
        el = pd_value * lgd * ead
        el_rate = pd_value * lgd

        # Phần bù rủi ro tối thiểu phải bù đắp tổn thất kỳ vọng hằng năm.
        # Chia cho kỳ hạn (năm) để quy về mức lãi suất năm.
        tenor_years = max(_f(features.get("tenor_months"), 12) / 12.0, 0.25)
        risk_spread = el_rate / tenor_years
        rate = base_rate + premium + max(risk_spread - 0.02, 0.0)
        rate = float(min(max(rate, 0.05), 0.40))
        return rate, el, el_rate

    # -- áp dụng đầy đủ ----------------------------------------------------
    def apply(self, features: Dict[str, Any], pd_value: float) -> PolicyOutcome:
        band = self.risk_band(pd_value)
        requested = _f(features.get("requested_amount"), 0.0)
        amount, constraint = self.compute_limit(features, band, requested)
        rate, el, el_rate = self.price(features, band, pd_value)
        tenor = int(_f(features.get("tenor_months"), 12))
        return PolicyOutcome(
            risk_band=band,
            approved_amount=amount,
            approved_tenor=tenor,
            priced_rate=rate,
            expected_loss=el,
            expected_loss_rate=el_rate,
            limit_binding_constraint=constraint,
            amount_reduced=amount < requested - 1.0,
        )

    def band_table(self) -> List[Dict[str, Any]]:
        """Bảng thang hạng rủi ro - dùng để trình bày trong luận văn."""
        rows = []
        prev = 0.0
        for b in self.bands:
            rows.append(
                {
                    "Hạng": b.name,
                    "PD từ": f"{prev:.2%}",
                    "PD đến": f"{min(b.pd_max, 1.0):.2%}",
                    "Hệ số hạn mức": LIMIT_MULTIPLIER_BY_BAND.get(b.name, 0.5),
                    "Phần bù lãi suất": f"{RATE_PREMIUM_BY_BAND.get(b.name, 0.03):+.2%}",
                }
            )
            prev = b.pd_max
        return rows


def _f(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


__all__ = ["CreditPolicy", "PolicyOutcome", "RiskBand", "LGD_BY_PRODUCT"]
