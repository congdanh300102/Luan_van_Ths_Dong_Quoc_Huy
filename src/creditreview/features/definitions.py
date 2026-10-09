"""Đăng ký (registry) các đặc trưng của mô hình kiểm duyệt tín dụng.

Việc khai báo tập trung đặc trưng phục vụ ba mục đích:

1. **Bảo đảm tính nhất quán giữa huấn luyện và suy luận** - cùng một danh sách
   đặc trưng được dùng khi huấn luyện mô hình và khi chấm điểm một hồ sơ mới
   trong bộ máy hỗ trợ ra quyết định, loại bỏ nguy cơ lệch định nghĩa đặc trưng.
2. **Phân tách nhóm đặc trưng** theo nguồn dữ liệu, đúng theo cách bài toán được
   phân tích trong đề án: ``STATIC`` (thông tin cá nhân và tài chính của khách
   hàng), ``LOAN`` (tham số khoản vay đề nghị), ``CIC`` (lịch sử tín dụng và
   quan hệ với các tổ chức tín dụng khác), ``BEHAVIOUR`` (hành vi giao dịch tài
   khoản), ``DERIVED`` (biến tổ hợp và biến tương tác).
3. **Sinh tài liệu** - bảng mô tả đặc trưng trong đề án được xuất trực tiếp từ
   registry này, tránh sai lệch giữa mã nguồn và tài liệu.

Việc phân nhóm còn cho phép trả lời một câu hỏi nghiên cứu cụ thể: *nhóm dữ liệu
nào đóng góp nhiều nhất vào chất lượng dự báo?* Kết quả phân tích Information
Value và SHAP đều được tổng hợp theo năm nhóm nói trên.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class FeatureGroup(str, Enum):
    """Nhóm đặc trưng theo nguồn dữ liệu."""

    STATIC = "STATIC"        # thông tin cá nhân và tài chính khách hàng
    LOAN = "LOAN"            # tham số khoản vay đề nghị
    CIC = "CIC"              # lịch sử tín dụng và quan hệ CIC
    BEHAVIOUR = "BEHAVIOUR"  # hành vi giao dịch tài khoản
    DERIVED = "DERIVED"      # biến tổ hợp và biến tương tác


class Source(str, Enum):
    """Nguồn cung cấp dữ liệu để tính đặc trưng."""

    CUSTOMER_PROFILE = "HỒ SƠ KH"    # hồ sơ khách hàng tại ngân hàng
    APPLICATION = "ĐƠN VAY"          # thông tin khai trên đơn đề nghị vay
    CIC_REPORT = "BẢN TIN CIC"       # bản tin tín dụng truy vấn từ CIC
    TRANSACTION = "GIAO DỊCH"        # lịch sử giao dịch tài khoản
    COMPUTED = "TÍNH TOÁN"           # tính từ các đặc trưng khác


@dataclass(frozen=True)
class Feature:
    name: str
    group: FeatureGroup
    dtype: str                 # num | cat | bin
    source: Source
    description: str


# ---------------------------------------------------------------------------
# 1. Nhóm đặc trưng thông tin cá nhân và tài chính (từ hồ sơ khách hàng)
# ---------------------------------------------------------------------------
STATIC_FEATURES: List[Feature] = [
    Feature("age", FeatureGroup.STATIC, "num", Source.CUSTOMER_PROFILE, "Tuổi khách hàng"),
    Feature("gender", FeatureGroup.STATIC, "cat", Source.CUSTOMER_PROFILE, "Giới tính"),
    Feature("marital_status", FeatureGroup.STATIC, "cat", Source.CUSTOMER_PROFILE, "Tình trạng hôn nhân"),
    Feature("num_dependents", FeatureGroup.STATIC, "num", Source.CUSTOMER_PROFILE, "Số người phụ thuộc"),
    Feature("education_code", FeatureGroup.STATIC, "num", Source.CUSTOMER_PROFILE, "Trình độ học vấn (1-5)"),
    Feature("occupation", FeatureGroup.STATIC, "cat", Source.CUSTOMER_PROFILE, "Nghề nghiệp"),
    Feature("employer_type", FeatureGroup.STATIC, "cat", Source.CUSTOMER_PROFILE, "Loại hình tổ chức làm việc"),
    Feature("job_tenure_months", FeatureGroup.STATIC, "num", Source.CUSTOMER_PROFILE, "Thâm niên công việc (tháng)"),
    Feature("monthly_income", FeatureGroup.STATIC, "num", Source.CUSTOMER_PROFILE, "Thu nhập khai báo hàng tháng"),
    Feature("province", FeatureGroup.STATIC, "cat", Source.CUSTOMER_PROFILE, "Tỉnh/thành cư trú"),
    Feature("residence_type", FeatureGroup.STATIC, "cat", Source.CUSTOMER_PROFILE, "Hình thức nhà ở"),
    Feature("customer_since_months", FeatureGroup.STATIC, "num", Source.CUSTOMER_PROFILE, "Thời gian quan hệ với ngân hàng"),
    Feature("has_salary_account", FeatureGroup.STATIC, "bin", Source.CUSTOMER_PROFILE, "Có tài khoản nhận lương"),
    Feature("n_accounts", FeatureGroup.STATIC, "num", Source.CUSTOMER_PROFILE, "Số tài khoản đang mở"),
    Feature("total_avg_balance", FeatureGroup.STATIC, "num", Source.CUSTOMER_PROFILE, "Tổng số dư bình quân 3 tháng"),
]

# ---------------------------------------------------------------------------
# 2. Đặc trưng từ tham số khoản vay đề nghị
# ---------------------------------------------------------------------------
LOAN_FEATURES: List[Feature] = [
    Feature("product_code", FeatureGroup.LOAN, "cat", Source.APPLICATION, "Mã sản phẩm tín dụng"),
    Feature("requested_amount", FeatureGroup.LOAN, "num", Source.APPLICATION, "Số tiền đề nghị vay"),
    Feature("tenor_months", FeatureGroup.LOAN, "num", Source.APPLICATION, "Kỳ hạn vay (tháng)"),
    Feature("interest_rate", FeatureGroup.LOAN, "num", Source.APPLICATION, "Lãi suất áp dụng (năm)"),
    Feature("loan_purpose", FeatureGroup.LOAN, "cat", Source.APPLICATION, "Mục đích vay"),
    Feature("channel", FeatureGroup.LOAN, "cat", Source.APPLICATION, "Kênh tiếp nhận hồ sơ"),
    Feature("has_collateral", FeatureGroup.LOAN, "bin", Source.APPLICATION, "Có tài sản bảo đảm"),
    Feature("monthly_installment", FeatureGroup.LOAN, "num", Source.APPLICATION, "Khoản trả góp hàng tháng"),
    Feature("ltv", FeatureGroup.LOAN, "num", Source.APPLICATION, "Tỷ lệ vay trên giá trị TSBĐ"),
    Feature("apply_hour", FeatureGroup.LOAN, "num", Source.APPLICATION, "Giờ nộp hồ sơ"),
    Feature("apply_dayofweek", FeatureGroup.LOAN, "num", Source.APPLICATION, "Ngày trong tuần nộp hồ sơ"),
]

# ---------------------------------------------------------------------------
# 3. Nhóm đặc trưng lịch sử tín dụng và quan hệ CIC
# ---------------------------------------------------------------------------
CIC_FEATURES: List[Feature] = [
    Feature("cic_score", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Điểm tín dụng CIC (300-850)"),
    Feature("current_group", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Nhóm nợ hiện tại (1-5)"),
    Feature("worst_group_12m", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Nhóm nợ xấu nhất 12 tháng"),
    Feature("dpd_max_12m", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Số ngày quá hạn tối đa 12 tháng"),
    Feature("num_institutions", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Số TCTD đang có quan hệ"),
    Feature("num_active_contracts", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Số hợp đồng còn hiệu lực"),
    Feature("num_closed_contracts", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Số hợp đồng đã tất toán"),
    Feature("total_outstanding", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Tổng dư nợ tại các TCTD"),
    Feature("credit_utilization", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Tỷ lệ sử dụng hạn mức"),
    Feature("monthly_debt_payment", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Nghĩa vụ trả nợ hàng tháng hiện hữu"),
    Feature("num_inquiries_3m", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Số lần truy vấn tín dụng 3 tháng"),
    Feature("has_written_off_debt", FeatureGroup.CIC, "bin", Source.CIC_REPORT, "Có nợ đã xử lý rủi ro"),
    Feature("oldest_account_months", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Độ dài lịch sử tín dụng (tháng)"),
    Feature("num_bad_debt_institutions", FeatureGroup.CIC, "num", Source.CIC_REPORT, "Số TCTD từng phát sinh nợ xấu"),
    Feature("is_thin_file", FeatureGroup.CIC, "bin", Source.CIC_REPORT, "Khách hàng thiếu lịch sử tín dụng"),
    Feature("cic_missing", FeatureGroup.CIC, "bin", Source.CIC_REPORT, "Không lấy được dữ liệu CIC"),
]

# ---------------------------------------------------------------------------
# 4. Nhóm đặc trưng hành vi tính từ lịch sử giao dịch tài khoản
# ---------------------------------------------------------------------------
# Các độ đo được luỹ kế theo cửa sổ thời gian; W là số ngày của cửa sổ.
BEHAVIOUR_TEMPLATES: List[tuple[str, str]] = [
    ("txn_cnt_{w}d", "Số giao dịch trong {w} ngày"),
    ("inflow_{w}d", "Tổng dòng tiền vào {w} ngày"),
    ("outflow_{w}d", "Tổng dòng tiền ra {w} ngày"),
    ("net_flow_{w}d", "Dòng tiền thuần {w} ngày"),
    ("avg_txn_amt_{w}d", "Giá trị giao dịch trung bình {w} ngày"),
    ("txn_amt_std_{w}d", "Độ lệch chuẩn giá trị giao dịch {w} ngày"),
    ("salary_cnt_{w}d", "Số lần nhận lương {w} ngày"),
    ("salary_amt_{w}d", "Tổng lương nhận {w} ngày"),
    ("repay_amt_{w}d", "Tổng tiền trả nợ {w} ngày"),
    ("risky_amt_{w}d", "Chi tiêu nhóm rủi ro {w} ngày"),
    ("cash_wd_cnt_{w}d", "Số lần rút tiền mặt {w} ngày"),
    ("cash_wd_amt_{w}d", "Giá trị rút tiền mặt {w} ngày"),
    ("ecom_amt_{w}d", "Chi tiêu thương mại điện tử {w} ngày"),
    ("utility_amt_{w}d", "Chi tiêu hóa đơn tiện ích {w} ngày"),
    ("night_txn_cnt_{w}d", "Số giao dịch ban đêm (0h-5h) {w} ngày"),
    ("large_txn_cnt_{w}d", "Số giao dịch giá trị lớn (>50 triệu) {w} ngày"),
    ("mobile_txn_cnt_{w}d", "Số giao dịch qua mobile banking {w} ngày"),
    ("outflow_ratio_{w}d", "Tỷ lệ dòng tiền ra trên dòng tiền vào {w} ngày"),
    ("risky_ratio_{w}d", "Tỷ trọng chi tiêu rủi ro {w} ngày"),
    ("cash_ratio_{w}d", "Tỷ trọng rút tiền mặt {w} ngày"),
    ("night_ratio_{w}d", "Tỷ trọng giao dịch ban đêm {w} ngày"),
    ("repay_burden_{w}d", "Gánh nặng trả nợ trên dòng tiền vào {w} ngày"),
    ("txn_amt_cv_{w}d", "Hệ số biến thiên giá trị giao dịch {w} ngày"),
]

# Đặc trưng hành vi không phụ thuộc cửa sổ
BEHAVIOUR_SINGLE: List[Feature] = [
    Feature("days_since_last_txn", FeatureGroup.BEHAVIOUR, "num", Source.TRANSACTION, "Số ngày từ giao dịch gần nhất"),
    Feature("txn_cnt_trend_30_90", FeatureGroup.BEHAVIOUR, "num", Source.TRANSACTION,
            "Xu thế số giao dịch: 30 ngày so với mức trung bình 90 ngày"),
    Feature("inflow_trend_30_90", FeatureGroup.BEHAVIOUR, "num", Source.TRANSACTION,
            "Xu thế dòng tiền vào: 30 ngày so với mức trung bình 90 ngày"),
    Feature("inflow_trend_90_180", FeatureGroup.BEHAVIOUR, "num", Source.TRANSACTION,
            "Xu thế dòng tiền vào: 90 ngày so với mức trung bình 180 ngày"),
    Feature("income_verification_ratio", FeatureGroup.BEHAVIOUR, "num", Source.TRANSACTION,
            "Tỷ lệ dòng tiền vào thực tế so với thu nhập khai báo"),
    Feature("salary_regularity", FeatureGroup.BEHAVIOUR, "num", Source.TRANSACTION,
            "Mức độ đều đặn của việc nhận lương trong 180 ngày"),
    Feature("income_stability", FeatureGroup.BEHAVIOUR, "num", Source.TRANSACTION,
            "Độ ổn định thu nhập (1 - hệ số biến thiên dòng tiền vào theo tháng)"),
]

# ---------------------------------------------------------------------------
# 5. Đặc trưng tổ hợp (derived) - các chỉ tiêu nghiệp vụ then chốt
# ---------------------------------------------------------------------------
DERIVED_FEATURES: List[Feature] = [
    Feature("dti", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Debt-to-Income: (trả góp mới + nghĩa vụ hiện hữu) / thu nhập"),
    Feature("pti", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Payment-to-Income: trả góp khoản vay mới / thu nhập"),
    Feature("amount_to_income", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Số tiền vay / thu nhập hàng tháng"),
    Feature("outstanding_to_income", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Tổng dư nợ hiện hữu / thu nhập hàng tháng"),
    Feature("disposable_income", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Thu nhập còn lại sau trả nợ và chi phí sinh hoạt tối thiểu"),
    Feature("disposable_ratio", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Tỷ lệ thu nhập còn lại trên thu nhập"),
    Feature("income_per_dependent", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Thu nhập bình quân đầu người trong hộ"),
    Feature("credit_history_ratio", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Độ dài lịch sử tín dụng so với tuổi trưởng thành"),
    Feature("debt_per_institution", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Dư nợ bình quân trên mỗi tổ chức tín dụng"),
    Feature("dpd_x_institutions", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Tương tác giữa số ngày quá hạn và số TCTD"),
    Feature("high_dti_short_tenure", FeatureGroup.DERIVED, "bin", Source.COMPUTED,
            "Cờ rủi ro: DTI > 0,5 và thâm niên công việc < 12 tháng"),
    Feature("balance_to_installment", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Số dư bình quân / khoản trả góp hàng tháng"),
    Feature("inflow_to_installment", FeatureGroup.DERIVED, "num", Source.COMPUTED,
            "Dòng tiền vào 30 ngày / khoản trả góp hàng tháng"),
]


def behaviour_features(windows: List[int]) -> List[Feature]:
    """Sinh danh sách đặc trưng hành vi cho các cửa sổ thời gian cho trước."""
    out: List[Feature] = []
    for w in windows:
        for tmpl, desc in BEHAVIOUR_TEMPLATES:
            out.append(
                Feature(
                    tmpl.format(w=w),
                    FeatureGroup.BEHAVIOUR,
                    "num",
                    Source.TRANSACTION,
                    desc.format(w=w),
                )
            )
    out.extend(BEHAVIOUR_SINGLE)
    return out


def all_features(windows: List[int]) -> List[Feature]:
    """Toàn bộ đặc trưng đầu vào của mô hình."""
    return (
        STATIC_FEATURES
        + LOAN_FEATURES
        + CIC_FEATURES
        + behaviour_features(windows)
        + DERIVED_FEATURES
    )


def feature_names(windows: List[int]) -> List[str]:
    return [f.name for f in all_features(windows)]


def categorical_features(windows: List[int]) -> List[str]:
    return [f.name for f in all_features(windows) if f.dtype == "cat"]


def numeric_features(windows: List[int]) -> List[str]:
    return [f.name for f in all_features(windows) if f.dtype in ("num", "bin")]


# Các cột định danh/thời gian KHÔNG được dùng làm đặc trưng đầu vào mô hình
ID_COLUMNS = ["application_id", "customer_id", "apply_time"]
TARGET = "default_flag"


__all__ = [
    "Feature",
    "FeatureGroup",
    "Source",
    "STATIC_FEATURES",
    "LOAN_FEATURES",
    "CIC_FEATURES",
    "BEHAVIOUR_TEMPLATES",
    "BEHAVIOUR_SINGLE",
    "DERIVED_FEATURES",
    "behaviour_features",
    "all_features",
    "feature_names",
    "categorical_features",
    "numeric_features",
    "ID_COLUMNS",
    "TARGET",
]
