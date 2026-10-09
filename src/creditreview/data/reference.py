"""Dữ liệu tham chiếu (danh mục) dùng cho quá trình sinh dữ liệu mô phỏng.

Các danh mục được thiết kế theo đặc thù thị trường tín dụng bán lẻ Việt Nam
(nhóm nợ theo Thông tư 11/2021/TT-NHNN, sản phẩm vay tiêu dùng, kênh phân phối
số hóa) nhằm bảo đảm dữ liệu mô phỏng có cấu trúc gần với dữ liệu nghiệp vụ.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Địa bàn
# ---------------------------------------------------------------------------
PROVINCES: list[tuple[str, float, float]] = [
    # (tên tỉnh/thành, trọng số dân số vay, hệ số thu nhập vùng)
    ("Hà Nội", 0.170, 1.28),
    ("TP. Hồ Chí Minh", 0.215, 1.34),
    ("Hải Phòng", 0.042, 1.06),
    ("Đà Nẵng", 0.038, 1.08),
    ("Cần Thơ", 0.028, 0.98),
    ("Bình Dương", 0.062, 1.12),
    ("Đồng Nai", 0.048, 1.08),
    ("Bắc Ninh", 0.030, 1.02),
    ("Quảng Ninh", 0.026, 1.04),
    ("Thanh Hóa", 0.035, 0.86),
    ("Nghệ An", 0.032, 0.84),
    ("Khánh Hòa", 0.024, 0.96),
    ("Thái Nguyên", 0.022, 0.94),
    ("Nam Định", 0.020, 0.84),
    ("Lâm Đồng", 0.018, 0.90),
    ("An Giang", 0.022, 0.82),
    ("Kiên Giang", 0.020, 0.88),
    ("Tiền Giang", 0.018, 0.84),
    ("Long An", 0.020, 0.92),
    ("Vĩnh Phúc", 0.018, 0.98),
    ("Hải Dương", 0.020, 0.94),
    ("Bình Thuận", 0.014, 0.88),
    ("Gia Lai", 0.012, 0.80),
    ("Đắk Lắk", 0.014, 0.82),
    ("Hưng Yên", 0.016, 0.96),
    ("Quảng Nam", 0.015, 0.86),
    ("Thừa Thiên Huế", 0.014, 0.88),
    ("Bà Rịa - Vũng Tàu", 0.020, 1.10),
    ("Sóc Trăng", 0.010, 0.78),
    ("Bến Tre", 0.011, 0.80),
]

# ---------------------------------------------------------------------------
# Trình độ học vấn: (mã, tên, trọng số, hệ số thu nhập, hiệu ứng lên z)
# ---------------------------------------------------------------------------
EDUCATION: list[tuple[int, str, float, float, float]] = [
    (1, "Trung học phổ thông", 0.240, 0.72, -0.22),
    (2, "Trung cấp", 0.155, 0.86, -0.08),
    (3, "Cao đẳng", 0.175, 0.96, 0.02),
    (4, "Đại học", 0.355, 1.22, 0.20),
    (5, "Sau đại học", 0.075, 1.62, 0.34),
]

# ---------------------------------------------------------------------------
# Nghề nghiệp: (tên, trọng số, hệ số thu nhập, độ biến động thu nhập, hiệu ứng z)
# ---------------------------------------------------------------------------
OCCUPATIONS: list[tuple[str, float, float, float, float]] = [
    ("Công nhân sản xuất", 0.170, 0.74, 0.16, -0.16),
    ("Nhân viên văn phòng", 0.205, 1.02, 0.10, 0.10),
    ("Kỹ sư / Kỹ thuật viên", 0.115, 1.28, 0.12, 0.18),
    ("Công chức / Viên chức", 0.105, 0.98, 0.06, 0.26),
    ("Giáo viên / Giảng viên", 0.058, 0.94, 0.07, 0.22),
    ("Nhân viên y tế", 0.046, 1.10, 0.09, 0.20),
    ("Kinh doanh tự do", 0.148, 1.16, 0.42, -0.24),
    ("Quản lý / Điều hành", 0.062, 1.94, 0.22, 0.30),
    ("Lao động phổ thông", 0.062, 0.62, 0.34, -0.36),
    ("Kinh doanh vận tải", 0.029, 0.92, 0.38, -0.20),
]

# ---------------------------------------------------------------------------
# Loại hình tổ chức làm việc: (tên, trọng số, ổn định thu nhập, hiệu ứng z)
# ---------------------------------------------------------------------------
EMPLOYER_TYPES: list[tuple[str, float, float, float]] = [
    ("Khu vực nhà nước", 0.185, 0.95, 0.24),
    ("Doanh nghiệp FDI", 0.205, 0.88, 0.16),
    ("Doanh nghiệp tư nhân", 0.398, 0.74, 0.00),
    ("Tự doanh / Hộ kinh doanh", 0.172, 0.48, -0.26),
    ("Khác", 0.040, 0.55, -0.12),
]

MARITAL_STATUS: list[tuple[str, float, float]] = [
    ("Độc thân", 0.360, -0.06),
    ("Đã kết hôn", 0.545, 0.14),
    ("Ly hôn / Ly thân", 0.072, -0.18),
    ("Khác", 0.023, -0.04),
]

RESIDENCE_TYPES: list[tuple[str, float, float]] = [
    ("Sở hữu riêng", 0.345, 0.30),
    ("Sở hữu cùng gia đình", 0.298, 0.10),
    ("Thuê / Trả góp", 0.297, -0.22),
    ("Khác", 0.060, -0.10),
]

# ---------------------------------------------------------------------------
# Sản phẩm tín dụng
# (mã, tên, trọng số, số tiền median (triệu VND), hệ số nhân thu nhập,
#  kỳ hạn hợp lệ, lãi suất năm, yêu cầu TSBĐ, hệ số rủi ro cơ sở)
# ---------------------------------------------------------------------------
PRODUCTS: list[dict] = [
    {
        "code": "CONSUMER_UNSEC",
        "name": "Vay tiêu dùng không có tài sản bảo đảm",
        "weight": 0.330,
        "median_amount": 85_000_000,
        "income_multiple": 6.5,
        "tenors": [12, 18, 24, 36, 48],
        "rate": 0.182,
        "secured": False,
        "risk_offset": 0.30,
    },
    {
        "code": "CREDIT_CARD",
        "name": "Thẻ tín dụng",
        "weight": 0.205,
        "median_amount": 55_000_000,
        "income_multiple": 3.5,
        "tenors": [12],
        "rate": 0.240,
        "secured": False,
        "risk_offset": 0.36,
    },
    {
        "code": "AUTO_LOAN",
        "name": "Vay mua xe",
        "weight": 0.145,
        "median_amount": 520_000_000,
        "income_multiple": 22.0,
        "tenors": [36, 48, 60, 72],
        "rate": 0.112,
        "secured": True,
        "risk_offset": -0.24,
    },
    {
        "code": "HOME_EQUITY",
        "name": "Vay thế chấp bất động sản",
        "weight": 0.120,
        "median_amount": 1_450_000_000,
        "income_multiple": 48.0,
        "tenors": [60, 120, 180, 240],
        "rate": 0.096,
        "secured": True,
        "risk_offset": -0.42,
    },
    {
        "code": "CONSUMER_SEC",
        "name": "Vay tiêu dùng có tài sản bảo đảm",
        "weight": 0.105,
        "median_amount": 240_000_000,
        "income_multiple": 12.0,
        "tenors": [24, 36, 48, 60],
        "rate": 0.138,
        "secured": True,
        "risk_offset": -0.14,
    },
    {
        "code": "BNPL",
        "name": "Mua trước trả sau",
        "weight": 0.095,
        "median_amount": 14_000_000,
        "income_multiple": 1.1,
        "tenors": [3, 6, 9, 12],
        "rate": 0.288,
        "secured": False,
        "risk_offset": 0.48,
    },
]

LOAN_PURPOSES: list[tuple[str, float]] = [
    ("Tiêu dùng sinh hoạt", 0.235),
    ("Mua sắm thiết bị / đồ gia dụng", 0.128),
    ("Sửa chữa nhà ở", 0.112),
    ("Mua xe", 0.135),
    ("Mua bất động sản", 0.118),
    ("Học tập", 0.052),
    ("Chữa bệnh", 0.044),
    ("Bổ sung vốn kinh doanh", 0.124),
    ("Khác", 0.052),
]

# Kênh tiếp nhận hồ sơ: (tên, trọng số, độ trễ nhập liệu trung bình (ms))
CHANNELS: list[tuple[str, float, float]] = [
    ("MOBILE_APP", 0.442, 18.0),
    ("WEB_PORTAL", 0.196, 24.0),
    ("BRANCH", 0.208, 45.0),
    ("PARTNER_API", 0.114, 12.0),
    ("CALL_CENTER", 0.040, 52.0),
]

# ---------------------------------------------------------------------------
# Giao dịch
# (mã ngành/mục đích, trọng số, hướng tiền, hệ số giá trị so với thu nhập)
# ---------------------------------------------------------------------------
# Lưu ý: nhóm SALARY ở đây chỉ đại diện cho các khoản thu nhập bất thường
# (thưởng, thu nhập ngoài). Khoản lương định kỳ hàng tháng được sinh riêng trong
# ``generator.generate_transactions`` để bảo đảm đúng tính chu kỳ - yếu tố quyết
# định giá trị của các đặc trưng về tính đều đặn và xác minh thu nhập.
TXN_CATEGORIES: list[tuple[str, float, str, float]] = [
    ("SALARY", 0.012, "CR", 0.420),
    ("TRANSFER_IN", 0.108, "CR", 0.180),
    ("CASH_DEPOSIT", 0.036, "CR", 0.240),
    ("GROCERY", 0.152, "DR", 0.028),
    ("FNB", 0.138, "DR", 0.014),
    ("ECOMMERCE", 0.104, "DR", 0.040),
    ("UTILITY", 0.072, "DR", 0.022),
    ("TELCO", 0.048, "DR", 0.006),
    ("FUEL", 0.058, "DR", 0.012),
    ("HEALTHCARE", 0.024, "DR", 0.046),
    ("EDUCATION", 0.021, "DR", 0.090),
    ("TRAVEL", 0.026, "DR", 0.075),
    ("ENTERTAINMENT", 0.032, "DR", 0.018),
    ("LOAN_REPAYMENT", 0.058, "DR", 0.150),
    ("CASH_WITHDRAWAL", 0.055, "DR", 0.085),
    ("GAMBLING_LIKE", 0.006, "DR", 0.065),
    ("CRYPTO_LIKE", 0.005, "DR", 0.110),
    ("INSURANCE", 0.005, "DR", 0.040),
]

TXN_CHANNELS: list[tuple[str, float]] = [
    ("MOBILE_BANKING", 0.402),
    ("QR_CODE", 0.168),
    ("POS", 0.152),
    ("ATM", 0.092),
    ("INTERNET_BANKING", 0.098),
    ("NAPAS_247", 0.062),
    ("BRANCH_COUNTER", 0.026),
]

# ---------------------------------------------------------------------------
# Hệ số rủi ro đặc thù theo địa bàn và nghề nghiệp (log-odds)
# Phản ánh khác biệt về điều kiện kinh tế vùng và mức độ ổn định thu nhập của
# từng nhóm nghề - các yếu tố này ảnh hưởng tới rủi ro vỡ nợ ngoài kênh thu nhập.
# ---------------------------------------------------------------------------
PROVINCE_RISK: dict[str, float] = {
    "Hà Nội": -0.10, "TP. Hồ Chí Minh": -0.06, "Hải Phòng": 0.02, "Đà Nẵng": -0.08,
    "Cần Thơ": 0.10, "Bình Dương": 0.16, "Đồng Nai": 0.12, "Bắc Ninh": 0.06,
    "Quảng Ninh": -0.02, "Thanh Hóa": 0.22, "Nghệ An": 0.26, "Khánh Hòa": 0.04,
    "Thái Nguyên": 0.08, "Nam Định": 0.18, "Lâm Đồng": 0.14, "An Giang": 0.30,
    "Kiên Giang": 0.24, "Tiền Giang": 0.20, "Long An": 0.12, "Vĩnh Phúc": 0.00,
    "Hải Dương": 0.06, "Bình Thuận": 0.18, "Gia Lai": 0.34, "Đắk Lắk": 0.32,
    "Hưng Yên": 0.04, "Quảng Nam": 0.16, "Thừa Thiên Huế": 0.10,
    "Bà Rịa - Vũng Tàu": -0.04, "Sóc Trăng": 0.36, "Bến Tre": 0.28,
}

OCCUPATION_RISK: dict[str, float] = {
    "Công nhân sản xuất": 0.16,
    "Nhân viên văn phòng": -0.06,
    "Kỹ sư / Kỹ thuật viên": -0.12,
    "Công chức / Viên chức": -0.28,
    "Giáo viên / Giảng viên": -0.24,
    "Nhân viên y tế": -0.18,
    "Kinh doanh tự do": 0.30,
    "Quản lý / Điều hành": -0.14,
    "Lao động phổ thông": 0.38,
    "Kinh doanh vận tải": 0.34,
}

# ---------------------------------------------------------------------------
# Hệ số nhân theo nhóm sản phẩm cho từng khối rủi ro.
# Mỗi dòng: (mã sản phẩm) -> (nhân khối lịch sử CIC, nhân khối hành vi,
#                             nhân khối cấu trúc khoản vay)
# Ý nghĩa nghiệp vụ: với sản phẩm thế chấp giá trị lớn, cấu trúc khoản vay
# (LTV, kỳ hạn) và khả năng trả nợ là yếu tố quyết định; với sản phẩm tín chấp
# ngắn hạn (thẻ tín dụng, BNPL), hành vi giao dịch và lịch sử tín dụng mới là
# yếu tố chi phối. Đây là dạng quan hệ "hệ số thay đổi theo phân khúc" mà một
# mô hình tuyến tính đơn nhất không thể biểu diễn.
# ---------------------------------------------------------------------------
PRODUCT_BLOCK_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "CONSUMER_UNSEC": (1.15, 1.10, 0.85),
    "CREDIT_CARD": (1.30, 1.45, 0.60),
    "AUTO_LOAN": (0.85, 0.75, 1.35),
    "HOME_EQUITY": (0.70, 0.60, 1.60),
    "CONSUMER_SEC": (0.90, 0.85, 1.25),
    "BNPL": (1.25, 1.60, 0.50),
}

# Nhóm nợ theo Thông tư 11/2021/TT-NHNN
DEBT_GROUPS: dict[int, str] = {
    1: "Nhóm 1 - Nợ đủ tiêu chuẩn",
    2: "Nhóm 2 - Nợ cần chú ý",
    3: "Nhóm 3 - Nợ dưới tiêu chuẩn",
    4: "Nhóm 4 - Nợ nghi ngờ",
    5: "Nhóm 5 - Nợ có khả năng mất vốn",
}


def normalize(weights) -> np.ndarray:
    """Chuẩn hóa vector trọng số về tổng bằng 1."""
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


__all__ = [
    "PROVINCES",
    "EDUCATION",
    "OCCUPATIONS",
    "EMPLOYER_TYPES",
    "MARITAL_STATUS",
    "RESIDENCE_TYPES",
    "PRODUCTS",
    "LOAN_PURPOSES",
    "CHANNELS",
    "TXN_CATEGORIES",
    "TXN_CHANNELS",
    "DEBT_GROUPS",
    "PROVINCE_RISK",
    "OCCUPATION_RISK",
    "PRODUCT_BLOCK_WEIGHTS",
    "normalize",
]
