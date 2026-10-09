"""CreditReview - Ứng dụng mô hình học máy trong hỗ trợ kiểm duyệt tín dụng.

Quy trình tiếp nhận thông tin hồ sơ tín dụng, xử lý và làm sạch dữ liệu, trích
xuất đặc trưng, sử dụng mô hình học máy để ước lượng xác suất vỡ nợ, kết hợp với
bộ luật nghiệp vụ để đưa ra kết quả **hỗ trợ** cán bộ tín dụng trong quá trình
kiểm duyệt. Quyết định cấp tín dụng cuối cùng vẫn thuộc thẩm quyền của người
hoặc bộ phận có thẩm quyền theo quy định của tổ chức tín dụng.

Luồng xử lý:
    Hồ sơ vay -> Tiền xử lý -> Trích xuất đặc trưng -> Mô hình học máy
              -> Luật nghiệp vụ -> Kết quả hỗ trợ kiểm duyệt

Các gói con (tổ chức theo đúng mạch nội dung của đề án):
    data           Thu thập, tổng hợp và thống kê mô tả dữ liệu tín dụng
    preprocessing  Tiền xử lý: giá trị khuyết, ngoại lệ, chuẩn hóa/mã hóa, chia tập
    features       Trích xuất và lựa chọn đặc trưng
    labeling       Xây dựng nhãn rủi ro và xác định điểm cắt phê duyệt
    models         Huấn luyện, tối ưu, hiệu chuẩn, đánh giá và giải thích mô hình
    decision       Bộ luật nghiệp vụ, chính sách tín dụng và bộ máy hỗ trợ quyết định
    serving        Dịch vụ REST tiếp nhận hồ sơ và trả kết quả hỗ trợ kiểm duyệt
    monitoring     Giám sát trôi dạt dữ liệu và độ ổn định mô hình theo thời gian
    dashboard      Giao diện trực quan hóa kết quả kiểm duyệt
"""

__version__ = "2.0.0"
__author__ = "Đồng Quốc Huy"

from .config import load_config  # noqa: F401

__all__ = ["load_config", "__version__"]
