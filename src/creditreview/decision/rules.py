"""Rule Engine - bộ luật nghiệp vụ tín dụng.

VÌ SAO CẦN RULE ENGINE KHI ĐÃ CÓ MÔ HÌNH HỌC MÁY
------------------------------------------------
Một quy trình phê duyệt tín dụng không thể chỉ dựa trên xác suất vỡ nợ do mô hình
học máy ước lượng. Bốn lý do:

1. **Ràng buộc pháp lý và quy chế.** Một số điều kiện mang tính bắt buộc, không
   thể bù trừ bằng điểm rủi ro tốt ở mặt khác. Ví dụ: khách hàng đang có nợ nhóm
   3 trở lên tại một tổ chức tín dụng khác thì không được cấp tín dụng, bất kể
   mô hình dự báo PD thấp. Đây là luật *loại trừ cứng* (hard knock-out).

2. **Giới hạn ngoại suy của mô hình.** Mô hình chỉ đáng tin cậy trong vùng dữ
   liệu nó đã học. Với hồ sơ nằm ngoài vùng đó (thu nhập bất thường, số tiền vay
   vượt khung sản phẩm), luật đóng vai trò *lưới an toàn*.

3. **Khả năng giải trình và điều chỉnh nhanh.** Khi chính sách tín dụng thay đổi,
   việc sửa một tham số luật có hiệu lực ngay, trong khi huấn luyện lại mô hình
   mất nhiều tuần.

4. **Tách bạch trách nhiệm.** Luật thể hiện *chính sách* của tổ chức (khẩu vị rủi
   ro), mô hình thể hiện *ước lượng thống kê*. Hai thành phần này cần được phê
   duyệt, kiểm định và thay đổi độc lập với nhau.

PHÂN LOẠI LUẬT
--------------
* ``HARD_REJECT``  - vi phạm dẫn tới từ chối ngay, không xét mô hình.
* ``REVIEW``       - vi phạm buộc chuyển thẩm định thủ công.
* ``WARNING``      - ghi nhận cảnh báo nhưng không thay đổi quyết định; dùng để
                     theo dõi và điều chỉnh chính sách về sau.

Thứ tự ưu tiên: ``HARD_REJECT`` > ``REVIEW`` > mô hình > ``WARNING``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..config import Config

log = logging.getLogger(__name__)


class RuleAction(str, Enum):
    HARD_REJECT = "HARD_REJECT"
    REVIEW = "REVIEW"
    WARNING = "WARNING"


class RuleCategory(str, Enum):
    ELIGIBILITY = "Điều kiện cơ bản"
    CREDIT_HISTORY = "Lịch sử tín dụng"
    AFFORDABILITY = "Khả năng trả nợ"
    COLLATERAL = "Tài sản bảo đảm"
    DATA_QUALITY = "Chất lượng dữ liệu"
    FRAUD = "Dấu hiệu bất thường"


@dataclass(frozen=True)
class Rule:
    """Một luật nghiệp vụ.

    ``predicate`` trả về ``True`` khi luật **bị vi phạm**.
    """

    code: str
    name: str
    action: RuleAction
    category: RuleCategory
    predicate: Callable[[Dict[str, Any]], bool]
    message: str
    regulation: str = ""      # căn cứ pháp lý / quy chế nội bộ (nếu có)


@dataclass
class RuleHit:
    code: str
    name: str
    action: str
    category: str
    message: str
    regulation: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code, "name": self.name, "action": self.action,
            "category": self.category, "message": self.message,
            "regulation": self.regulation,
        }


@dataclass
class RuleResult:
    """Kết quả chạy toàn bộ tập luật trên một hồ sơ."""

    hits: List[RuleHit] = field(default_factory=list)
    evaluated: int = 0
    latency_ms: float = 0.0

    @property
    def hard_rejects(self) -> List[RuleHit]:
        return [h for h in self.hits if h.action == RuleAction.HARD_REJECT.value]

    @property
    def reviews(self) -> List[RuleHit]:
        return [h for h in self.hits if h.action == RuleAction.REVIEW.value]

    @property
    def warnings(self) -> List[RuleHit]:
        return [h for h in self.hits if h.action == RuleAction.WARNING.value]

    @property
    def must_reject(self) -> bool:
        return bool(self.hard_rejects)

    @property
    def must_review(self) -> bool:
        return bool(self.reviews)


# ---------------------------------------------------------------------------
# Tiện ích truy xuất giá trị an toàn
# ---------------------------------------------------------------------------
def _num(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Lấy giá trị số; giá trị thiếu/không hợp lệ trả về ``default``."""
    v = d.get(key)
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f  # loại NaN


def _missing(d: Dict[str, Any], key: str) -> bool:
    v = d.get(key)
    if v is None:
        return True
    try:
        f = float(v)
        return f != f
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Xây dựng tập luật từ cấu hình
# ---------------------------------------------------------------------------
def build_rules(cfg: Config) -> List[Rule]:
    """Khởi tạo tập luật nghiệp vụ theo tham số cấu hình.

    Việc lấy tham số từ cấu hình (thay vì gán cứng trong mã) cho phép điều chỉnh
    khẩu vị rủi ro mà không sửa mã nguồn - yêu cầu vận hành thực tế của khối
    quản trị rủi ro.
    """
    r = cfg.get("decision.rules", {}) or {}
    min_age = int(r.get("min_age", 20))
    max_age = int(r.get("max_age", 62))
    min_income = float(r.get("min_monthly_income", 6_000_000))
    max_dti = float(r.get("max_dti", 0.60))
    max_worst_group = int(r.get("max_cic_worst_group", 2))
    max_dpd = float(r.get("max_dpd_12m", 30))
    max_inst = int(r.get("max_num_institutions", 6))
    max_ltv = float(r.get("max_ltv", 0.85))
    min_tenure = int(r.get("min_job_tenure_months", 6))
    max_inq = int(r.get("max_inquiries_3m", 5))

    rules: List[Rule] = [
        # ================= Điều kiện cơ bản =================
        Rule(
            "R001", f"Tuổi nằm ngoài khoảng cho phép [{min_age}; {max_age}]",
            RuleAction.HARD_REJECT, RuleCategory.ELIGIBILITY,
            lambda d: not (min_age <= _num(d, "age", 0) <= max_age),
            f"Khách hàng phải trong độ tuổi từ {min_age} đến {max_age} tại thời điểm đề nghị vay",
            "Quy chế cho vay nội bộ",
        ),
        Rule(
            "R002", f"Thu nhập dưới mức tối thiểu {min_income/1e6:.0f} triệu VND",
            RuleAction.HARD_REJECT, RuleCategory.ELIGIBILITY,
            lambda d: _num(d, "monthly_income", 0) < min_income,
            "Thu nhập hàng tháng không đạt mức tối thiểu của sản phẩm",
        ),
        Rule(
            "R003", f"Thâm niên công việc dưới {min_tenure} tháng",
            RuleAction.REVIEW, RuleCategory.ELIGIBILITY,
            lambda d: _num(d, "job_tenure_months", 0) < min_tenure,
            "Thâm niên công việc chưa đủ để đánh giá tính ổn định của thu nhập",
        ),

        # ================= Lịch sử tín dụng =================
        Rule(
            "R101", f"Nhóm nợ xấu nhất 12 tháng vượt nhóm {max_worst_group}",
            RuleAction.HARD_REJECT, RuleCategory.CREDIT_HISTORY,
            lambda d: _num(d, "worst_group_12m", 1) > max_worst_group,
            "Khách hàng phát sinh nợ xấu trong 12 tháng gần nhất",
            "Thông tư 11/2021/TT-NHNN về phân loại nợ",
        ),
        Rule(
            "R102", "Đang có nợ nhóm 3 trở lên",
            RuleAction.HARD_REJECT, RuleCategory.CREDIT_HISTORY,
            lambda d: _num(d, "current_group", 1) >= 3,
            "Khách hàng đang có khoản nợ được phân loại từ nhóm 3 (nợ dưới tiêu chuẩn)",
            "Thông tư 11/2021/TT-NHNN",
        ),
        Rule(
            "R103", "Có nợ đã xử lý rủi ro / xóa nợ",
            RuleAction.HARD_REJECT, RuleCategory.CREDIT_HISTORY,
            lambda d: _num(d, "has_written_off_debt", 0) >= 1,
            "Lịch sử có khoản nợ đã được xử lý rủi ro hoặc xóa nợ",
        ),
        Rule(
            "R104", f"Số ngày quá hạn tối đa 12 tháng vượt {max_dpd:.0f} ngày",
            RuleAction.REVIEW, RuleCategory.CREDIT_HISTORY,
            lambda d: _num(d, "dpd_max_12m", 0) > max_dpd,
            "Từng chậm trả vượt ngưỡng cho phép trong 12 tháng gần nhất",
        ),
        Rule(
            "R105", f"Quan hệ tín dụng với hơn {max_inst} tổ chức",
            RuleAction.REVIEW, RuleCategory.CREDIT_HISTORY,
            lambda d: _num(d, "num_institutions", 0) > max_inst,
            "Số lượng tổ chức tín dụng đang có quan hệ ở mức cao, tiềm ẩn rủi ro vay đảo nợ",
        ),
        Rule(
            "R106", f"Truy vấn tín dụng 3 tháng vượt {max_inq} lần",
            RuleAction.WARNING, RuleCategory.CREDIT_HISTORY,
            lambda d: _num(d, "num_inquiries_3m", 0) > max_inq,
            "Khách hàng tìm kiếm tín dụng dồn dập tại nhiều tổ chức",
        ),
        Rule(
            "R107", "Tỷ lệ sử dụng hạn mức trên 95%",
            RuleAction.WARNING, RuleCategory.CREDIT_HISTORY,
            lambda d: _num(d, "credit_utilization", 0) > 0.95,
            "Hạn mức tín dụng hiện có đã được sử dụng gần hết",
        ),

        # ================= Khả năng trả nợ =================
        Rule(
            "R201", f"Tỷ lệ nợ trên thu nhập (DTI) vượt {max_dti:.0%}",
            RuleAction.HARD_REJECT, RuleCategory.AFFORDABILITY,
            lambda d: _num(d, "dti", 0) > max_dti,
            "Tổng nghĩa vụ trả nợ vượt quá khả năng chi trả từ thu nhập",
        ),
        Rule(
            "R202", "Thu nhập còn lại sau trả nợ không đủ chi phí sinh hoạt",
            RuleAction.HARD_REJECT, RuleCategory.AFFORDABILITY,
            lambda d: _num(d, "disposable_income", 1) <= 0,
            "Sau khi trừ nghĩa vụ trả nợ và chi phí sinh hoạt tối thiểu, thu nhập còn lại âm",
        ),
        Rule(
            "R203", "Tỷ lệ trả góp trên thu nhập (PTI) vượt 45%",
            RuleAction.REVIEW, RuleCategory.AFFORDABILITY,
            lambda d: _num(d, "pti", 0) > 0.45,
            "Riêng khoản vay mới đã chiếm tỷ trọng lớn trong thu nhập",
        ),
        Rule(
            "R204", "Số tiền vay vượt 60 lần thu nhập hàng tháng",
            RuleAction.REVIEW, RuleCategory.AFFORDABILITY,
            lambda d: _num(d, "amount_to_income", 0) > 60,
            "Quy mô khoản vay rất lớn so với thu nhập, cần thẩm định bổ sung",
        ),

        # ================= Tài sản bảo đảm =================
        Rule(
            "R301", f"Tỷ lệ vay trên giá trị TSBĐ (LTV) vượt {max_ltv:.0%}",
            RuleAction.REVIEW, RuleCategory.COLLATERAL,
            lambda d: _num(d, "has_collateral", 0) >= 1 and _num(d, "ltv", 0) > max_ltv,
            "Giá trị tài sản bảo đảm không đủ tỷ lệ bao phủ yêu cầu",
        ),

        # ================= Chất lượng dữ liệu =================
        Rule(
            "R401", "Không lấy được dữ liệu CIC",
            RuleAction.REVIEW, RuleCategory.DATA_QUALITY,
            lambda d: _num(d, "cic_missing", 0) >= 1,
            "Hệ thống CIC không phản hồi - chuyển thẩm định thủ công theo quy trình dự phòng",
        ),
        Rule(
            "R402", "Khách hàng thiếu lịch sử tín dụng (thin-file)",
            RuleAction.WARNING, RuleCategory.DATA_QUALITY,
            lambda d: _num(d, "is_thin_file", 0) >= 1,
            "Không có/ít lịch sử tín dụng; đánh giá dựa chủ yếu vào dữ liệu hành vi giao dịch",
        ),
        Rule(
            "R403", "Thiếu dữ liệu hành vi giao dịch",
            RuleAction.REVIEW, RuleCategory.DATA_QUALITY,
            lambda d: _num(d, "txn_cnt_180d", 0) < 5,
            "Số giao dịch trong 180 ngày quá ít để tính đặc trưng hành vi đáng tin cậy",
        ),

        # ================= Dấu hiệu bất thường =================
        Rule(
            "R501", "Dòng tiền vào thực tế thấp hơn nhiều so với thu nhập khai báo",
            RuleAction.REVIEW, RuleCategory.FRAUD,
            # Chỉ áp dụng với khách hàng NHẬN LƯƠNG QUA TÀI KHOẢN tại ngân hàng.
            # Với khách hàng không nhận lương qua tài khoản, dòng tiền vào tài
            # khoản không phản ánh thu nhập nên việc đối chiếu là vô nghĩa và sẽ
            # tạo ra cảnh báo sai trên một tỷ lệ lớn hồ sơ.
            lambda d: (
                _num(d, "has_salary_account", 0) >= 1
                and _num(d, "txn_cnt_180d", 0) >= 20
                and _num(d, "income_verification_ratio", 1.0) < 0.35
            ),
            "Khách hàng nhận lương qua tài khoản nhưng dòng tiền vào 30 ngày dưới 35% "
            "thu nhập khai báo - cần xác minh thu nhập",
        ),
        Rule(
            "R502", "Tỷ trọng chi tiêu nhóm rủi ro cao bất thường",
            RuleAction.REVIEW, RuleCategory.FRAUD,
            lambda d: _num(d, "risky_ratio_90d", 0) > 0.15,
            "Trên 15% dòng tiền ra thuộc nhóm giao dịch rủi ro cao",
        ),
        Rule(
            "R503", "Không phát sinh giao dịch trong 60 ngày gần nhất",
            RuleAction.WARNING, RuleCategory.FRAUD,
            lambda d: _num(d, "days_since_last_txn", 0) > 60,
            "Tài khoản không hoạt động trong thời gian dài trước khi đề nghị vay",
        ),
    ]
    return rules


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------
class RuleEngine:
    """Bộ thực thi luật nghiệp vụ."""

    def __init__(self, cfg: Config, rules: Optional[List[Rule]] = None) -> None:
        self.cfg = cfg
        self.rules = rules if rules is not None else build_rules(cfg)
        # thống kê tần suất vi phạm từng luật, phục vụ rà soát chính sách
        self.hit_counts: Dict[str, int] = {r.code: 0 for r in self.rules}
        self.n_evaluated = 0

    def evaluate(self, features: Dict[str, Any], short_circuit: bool = False) -> RuleResult:
        """Chạy toàn bộ luật trên một hồ sơ.

        ``short_circuit=True`` dừng ngay khi gặp luật ``HARD_REJECT`` đầu tiên -
        giảm độ trễ trong vận hành. Mặc định là ``False`` để thu thập đầy đủ
        thông tin phục vụ phân tích và giải trình.
        """
        import time

        t0 = time.perf_counter()
        hits: List[RuleHit] = []
        evaluated = 0
        for rule in self.rules:
            evaluated += 1
            try:
                violated = rule.predicate(features)
            except Exception as exc:  # pragma: no cover
                log.warning("Lỗi khi đánh giá luật %s: %s", rule.code, exc)
                continue
            if violated:
                self.hit_counts[rule.code] = self.hit_counts.get(rule.code, 0) + 1
                hits.append(
                    RuleHit(rule.code, rule.name, rule.action.value,
                            rule.category.value, rule.message, rule.regulation)
                )
                if short_circuit and rule.action is RuleAction.HARD_REJECT:
                    break
        self.n_evaluated += 1
        return RuleResult(
            hits=hits, evaluated=evaluated,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def catalogue(self) -> List[Dict[str, str]]:
        """Danh mục luật - dùng để xuất bảng mô tả luật trong luận văn."""
        return [
            {
                "Mã luật": r.code,
                "Tên luật": r.name,
                "Nhóm": r.category.value,
                "Hành động": r.action.value,
                "Diễn giải": r.message,
                "Căn cứ": r.regulation or "-",
            }
            for r in self.rules
        ]

    def hit_statistics(self) -> List[Dict[str, Any]]:
        """Tần suất vi phạm từng luật trên toàn bộ hồ sơ đã xử lý."""
        by_code = {r.code: r for r in self.rules}
        out = []
        for code, n in sorted(self.hit_counts.items(), key=lambda kv: -kv[1]):
            r = by_code.get(code)
            if r is None:
                continue
            out.append(
                {
                    "Mã luật": code,
                    "Tên luật": r.name,
                    "Nhóm": r.category.value,
                    "Hành động": r.action.value,
                    "Số hồ sơ vi phạm": n,
                    "Tỷ lệ vi phạm": round(n / max(self.n_evaluated, 1), 4),
                }
            )
        return out

    def reset_statistics(self) -> None:
        self.hit_counts = {r.code: 0 for r in self.rules}
        self.n_evaluated = 0


__all__ = [
    "Rule", "RuleHit", "RuleResult", "RuleEngine", "RuleAction", "RuleCategory",
    "build_rules",
]
