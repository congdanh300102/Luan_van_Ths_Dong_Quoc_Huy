# ===========================================================================
# Ảnh dùng chung cho hai dịch vụ Python của hệ thống hỗ trợ kiểm duyệt tín dụng:
#   - review-api  (FastAPI: chấm điểm rủi ro và đưa ra khuyến nghị)
#   - dashboard   (Streamlit: trực quan hóa kết quả)
#
# Lệnh khởi chạy cụ thể được chỉ định ở tầng docker-compose để cùng một ảnh
# phục vụ được cả hai dịch vụ.
# ===========================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/creditreview/src \
    PIP_NO_CACHE_DIR=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/creditreview

# Cài phụ thuộc trước để tận dụng cache tầng Docker
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000 8501

# Người dùng không đặc quyền
RUN useradd --create-home --shell /bin/bash crapp && \
    chown -R crapp:crapp /opt/creditreview
USER crapp

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "creditreview.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
