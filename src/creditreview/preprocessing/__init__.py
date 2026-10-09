"""Tiền xử lý và làm sạch dữ liệu trước khi đưa vào mô hình học máy.

Gói con gồm ba nhóm chức năng tương ứng ba bước của quy trình tiền xử lý:

    cleaning   Xử lý giá trị khuyết thiếu và giá trị ngoại lệ
    encoding   Chuẩn hóa biến định lượng và mã hóa biến định tính
    splitting  Phân chia tập huấn luyện - kiểm định - kiểm thử

Nguyên tắc xuyên suốt: **mọi tham số biến đổi chỉ được ước lượng trên tập huấn
luyện** rồi áp dụng cho các tập còn lại. Nếu ước lượng trên toàn bộ dữ liệu,
thông tin từ tập kiểm thử sẽ rò rỉ vào quá trình huấn luyện và kết quả đánh giá
sẽ lạc quan hơn thực tế.
"""

from .cleaning import (  # noqa: F401
    MissingValueReport,
    OutlierCapper,
    impute_business_defaults,
    missing_value_report,
    outlier_report,
)
from .encoding import make_preprocessor  # noqa: F401
from .splitting import out_of_time_split, split_report, stratified_customer_split  # noqa: F401

__all__ = [
    "missing_value_report",
    "MissingValueReport",
    "impute_business_defaults",
    "outlier_report",
    "OutlierCapper",
    "make_preprocessor",
    "out_of_time_split",
    "stratified_customer_split",
    "split_report",
]
