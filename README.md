# CreditReview — Ứng dụng mô hình học máy trong hỗ trợ kiểm duyệt tín dụng

> **Đề án tốt nghiệp Thạc sĩ Kỹ thuật** — Chuyên ngành Hệ thống thông tin (8.48.01.04)
> **Học viên:** Đồng Quốc Huy — B25CHHT028
> **Hướng dẫn khoa học:** TS. Đỗ Tiến Dũng
> **Học viện Công nghệ Bưu chính Viễn thông**, 2026

Mô hình nguyên mẫu hỗ trợ kiểm duyệt tín dụng: tiếp nhận hồ sơ vay, xử lý và làm
sạch dữ liệu, trích xuất đặc trưng, dùng mô hình học máy ước lượng xác suất vỡ nợ,
kết hợp với bộ luật nghiệp vụ để đưa ra **khuyến nghị kèm lý do** cho cán bộ tín dụng.

```
Hồ sơ vay → Tiền xử lý → Trích xuất đặc trưng → Mô hình học máy
          → Luật nghiệp vụ → Khuyến nghị + lý do
```

Quyết định cấp tín dụng cuối cùng vẫn thuộc thẩm quyền của người hoặc bộ phận có
thẩm quyền theo quy định của tổ chức tín dụng.

---

## 1. Kết quả chính

| Hạng mục | Kết quả |
|---|---|
| Số mô hình so sánh | **12** (gồm hồi quy logistic, cây quyết định, rừng ngẫu nhiên, SVM, gradient boosting) |
| Mô hình tốt nhất | **XGBoost** — AUC 0,8625 · Gini 0,7250 · KS 0,5775 |
| Trần Bayes (giới hạn lý thuyết) | 0,9109 |
| Chênh lệch với hồi quy logistic | chỉ 0,0046 AUC |
| Sai số hiệu chuẩn xác suất (ECE) | 0,0053 |
| Đóng góp của đặc trưng hành vi giao dịch (SHAP) | **40,3 %** — cao hơn nhóm CIC truyền thống (27,6 %) |
| Tỷ lệ hồ sơ xử lý tự động | 67,4 % |
| Nợ xấu nhóm được duyệt | **3,21 %** so với 6,14 % của quy trình chỉ dùng luật (giảm 47,7 %) |
| Thời gian xử lý một hồ sơ | p50 ≈ 48 ms · p95 ≈ 78 ms |

**Hai phát hiện đáng chú ý:**

1. Nhóm đặc trưng hành vi giao dịch — nguồn dữ liệu mà ngân hàng tự sở hữu và luôn
   có sẵn kể cả với khách hàng thiếu lịch sử tín dụng — đóng góp phần lớn nhất vào
   quyết định của mô hình.
2. Khoảng cách giữa mô hình phức tạp nhất và hồi quy logistic chỉ 0,0046 AUC, đặt
   ra câu hỏi thực tế về đánh đổi giữa mức cải thiện nhỏ với chi phí mất tính minh bạch.

---

## 2. Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu Python ≥ 3.10.

---

## 3. Tái lập toàn bộ kết quả

Chạy toàn bộ bằng một lệnh:

```bash
python scripts/run_all.py            # đầy đủ
python scripts/run_all.py --quick    # kiểm tra nhanh
```

Hoặc chạy tuần tự từng bước:

```bash
# Bước 1 — Chuẩn bị dữ liệu: sinh, kiểm tra chất lượng, gán nhãn, chia tập
python scripts/01_prepare_data.py

# Bước 2 — Phân tích Information Value và lựa chọn đặc trưng
python scripts/02_build_features.py

# Bước 3 — Tinh chỉnh siêu tham số (tùy chọn)
python scripts/03_tune_hyperparams.py --trials 14

# Bước 4 — Huấn luyện và so sánh 12 mô hình
python scripts/04_train_models.py

# Bước 5 — Mô phỏng quy trình kiểm duyệt, xác định ngưỡng, so sánh quy trình truyền thống
python scripts/05_simulate_evaluate.py

# Bước 6 — Xuất bảng LaTeX cho đề án
python scripts/06_export_latex_tables.py

# Bước 7 — Xuất từ điển dữ liệu các bảng đầu vào (Excel, tùy chọn)
python scripts/07_export_data_dictionary.py
```

Hạt giống ngẫu nhiên cố định trong `config/config.yaml` (`project.seed`), bảo đảm
kết quả lặp lại được.

---

## 4. Dịch vụ và giao diện

```bash
# Dịch vụ REST hỗ trợ kiểm duyệt
uvicorn creditreview.serving.api:app --port 8000
#   Tài liệu API: http://localhost:8000/docs

# Giao diện trực quan hóa kết quả
PYTHONPATH=src streamlit run src/creditreview/dashboard/app.py
```

Năm điểm cuối chính của dịch vụ:

| Điểm cuối | Chức năng |
|---|---|
| `POST /api/v1/screen` | Kiểm duyệt một hồ sơ, trả khuyến nghị kèm lý do |
| `POST /api/v1/screen/batch` | Kiểm duyệt theo lô (tái chấm điểm danh mục) |
| `GET /api/v1/features/{id}` | Truy xuất vector đặc trưng của một hồ sơ |
| `GET /admin/rules` | Danh mục luật nghiệp vụ đang áp dụng |
| `GET /health`, `GET /metrics` | Trạng thái và chỉ tiêu xử lý tích lũy |

---

## 5. Cấu trúc dự án

Mỗi gói con tương ứng một mục của Chương 2 trong đề án:

```
config/config.yaml            Cấu hình tập trung toàn bộ quy trình
src/creditreview/
  data/                       2.2 Thu thập, tổng hợp và kiểm tra chất lượng dữ liệu
  preprocessing/              2.3 Xử lý khuyết thiếu, ngoại lệ, chuẩn hóa, chia tập
  features/                   2.4 Trích xuất và lựa chọn đặc trưng
  labeling/                   2.5 Gán nhãn rủi ro và xác định điểm cắt phê duyệt
  models/                     2.6, 2.7 Nhà máy mô hình, huấn luyện, giải thích
  decision/                   Luật nghiệp vụ · chính sách · bộ máy hỗ trợ quyết định
  serving/                    Dịch vụ REST và kho đặc trưng
  monitoring/                 Giám sát PSI và chất lượng theo thời gian
  dashboard/                  Giao diện Streamlit
scripts/                      6 bước tái lập kết quả
results/                      Mô hình · bảng · đồ thị
thesis/                       Mã nguồn LaTeX của đề án
```

---

## 6. Biên dịch đề án

```bash
cd thesis
xelatex main.tex && xelatex main.tex     # chạy 2 lần để cập nhật mục lục
```

Yêu cầu **XeLaTeX** (do dùng tiếng Việt và phông Times New Roman).
Kết quả: `thesis/main.pdf`.

Số liệu trong đề án được sinh tự động từ kết quả thực nghiệm qua
`scripts/06_export_latex_tables.py` — chạy lại thực nghiệm là toàn bộ bảng biểu và
các con số trích dẫn tự cập nhật, không phải sửa tay.

---

## 7. Ghi chú quan trọng về dữ liệu

Đề án **không sử dụng dữ liệu thật** của bất kỳ tổ chức tín dụng nào. Toàn bộ dữ
liệu được sinh mô phỏng bằng một mô hình biến tiềm ẩn được thiết kế để tái tạo cấu
trúc thống kê của dữ liệu tín dụng bán lẻ (xem `src/creditreview/data/generator.py`).

Các chỉ tiêu chất lượng mô hình vì vậy **không nên được hiểu là dự báo về hiệu quả
trên dữ liệu thực**. Giá trị của thực nghiệm nằm ở việc so sánh *tương đối* giữa các
mô hình trong cùng điều kiện và ở việc kiểm chứng tính đúng đắn của toàn bộ quy trình.

Chi tiết thiết kế và giới hạn được trình bày tại mục 3.1.3 của đề án.
