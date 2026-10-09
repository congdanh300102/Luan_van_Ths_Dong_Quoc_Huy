"""Dịch vụ REST hỗ trợ kiểm duyệt tín dụng.

VAI TRÒ TRONG HỆ THỐNG
----------------------
Dịch vụ là điểm tích hợp giữa mô hình học máy và quy trình nghiệp vụ: dịch vụ
nghiệp vụ của tổ chức tín dụng gửi thông tin hồ sơ tới đây và nhận về kết quả
đánh giá rủi ro kèm khuyến nghị và lý do. Luồng xử lý một hồ sơ gồm bốn bước:

    tiếp nhận hồ sơ -> lấy đặc trưng -> chấm điểm mô hình -> áp luật nghiệp vụ

Kết quả trả về là **thông tin hỗ trợ** cho cán bộ tín dụng, không phải quyết
định cấp tín dụng chính thức.

BA NGUYÊN TẮC THIẾT KẾ
----------------------
1. **Nạp mô hình một lần khi khởi động.** Nạp mô hình trong mỗi yêu cầu sẽ làm
   thời gian phản hồi tăng hàng trăm mili giây.
2. **Luôn trả về một kết quả.** Kể cả khi mô hình lỗi hoặc thiếu dữ liệu CIC,
   dịch vụ trả về ``REVIEW`` kèm lý do thay vì trả lỗi HTTP. Về nghiệp vụ, một
   hồ sơ "cần thẩm định thêm" là kết quả hợp lệ; một lỗi HTTP thì không xử lý
   được trong quy trình.
3. **Kèm thông tin thời gian xử lý trong mỗi phản hồi.** Cho phép giám sát liên
   tục trong vận hành, không chỉ trong lúc kiểm thử.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import load_config
from ..decision.engine import DecisionEngine
from .repository import FeatureRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("credit-review-api")

cfg = load_config()

# Trạng thái dịch vụ (khởi tạo một lần khi startup)
_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo tài nguyên khi dịch vụ bắt đầu và giải phóng khi dừng."""
    log.info("Khởi tạo dịch vụ hỗ trợ kiểm duyệt tín dụng...")

    repo = FeatureRepository(cfg.path("data_raw") / "features.parquet")

    model_path = cfg.path("models") / "production_model.joblib"
    engine = DecisionEngine(cfg)
    if model_path.exists():
        bundle = joblib.load(model_path)
        engine.load_model(bundle)
        log.info(
            "Đã nạp mô hình: %s (AUC = %.4f)",
            bundle.get("display_name"), bundle.get("test_auc", 0.0),
        )
    else:
        log.error(
            "KHÔNG tìm thấy mô hình tại %s - dịch vụ sẽ trả về REVIEW cho mọi hồ sơ",
            model_path,
        )

    _state.update({
        "repo": repo,
        "engine": engine,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    yield
    log.info("Dừng dịch vụ...")


app = FastAPI(
    title="Credit Review Support API",
    description="Dịch vụ đánh giá rủi ro và hỗ trợ kiểm duyệt hồ sơ tín dụng",
    version="2.0.0",
    lifespan=lifespan,
)


# ===========================================================================
# Mô hình dữ liệu vào/ra
# ===========================================================================
class ApplicationRequest(BaseModel):
    """Yêu cầu kiểm duyệt một hồ sơ đề nghị vay vốn.

    Có hai cách cung cấp dữ liệu đầu vào:

    * Chỉ truyền ``application_id`` - dịch vụ tra cứu đặc trưng đã tính sẵn
      trong kho dữ liệu. Dùng cho tái chấm điểm và cho các kịch bản thử nghiệm.
    * Truyền thêm ``features`` - dịch vụ dùng trực tiếp các giá trị này, ghi đè
      lên giá trị tra cứu được. Dùng khi hệ thống nghiệp vụ đã tự tính đặc trưng
      hoặc khi cần mô phỏng thay đổi một vài chỉ tiêu ("nếu thu nhập là X thì
      kết quả thay đổi thế nào").
    """

    application_id: str = Field(..., description="Mã hồ sơ")
    customer_id: Optional[str] = Field(None, description="Mã khách hàng")
    features: Optional[Dict[str, Any]] = Field(
        None, description="Giá trị đặc trưng truyền trực tiếp (ghi đè giá trị tra cứu)"
    )

    model_config = {"json_schema_extra": {"example": {
        "application_id": "APP00051234",
        "customer_id": "CUS0001234",
    }}}


class ScreeningResponse(BaseModel):
    """Kết quả hỗ trợ kiểm duyệt một hồ sơ."""

    decision_id: str
    application_id: str
    customer_id: str
    recommendation: str = Field(..., description="APPROVE | REVIEW | REJECT")
    reason_category: str
    reason_text: str
    pd_score: float = Field(..., description="Xác suất vỡ nợ dự báo")
    risk_band: str
    suggested_amount: float
    suggested_tenor: int
    suggested_rate: float
    rule_hits: List[Dict[str, str]]
    reason_codes: List[Dict[str, Any]]
    processing_ms: Dict[str, float]
    degraded_mode: bool
    model_version: str
    decided_at: str


# ===========================================================================
# Điểm cuối nghiệp vụ
# ===========================================================================
@app.post("/api/v1/screen", response_model=ScreeningResponse,
          summary="Kiểm duyệt một hồ sơ đề nghị vay vốn")
def screen(req: ApplicationRequest) -> ScreeningResponse:
    """Đánh giá rủi ro một hồ sơ và trả về khuyến nghị kèm lý do."""
    if not _state:
        raise HTTPException(503, "Dịch vụ chưa khởi tạo xong")

    repo: FeatureRepository = _state["repo"]
    engine: DecisionEngine = _state["engine"]

    # --- (1) lấy đặc trưng --------------------------------------------------
    t0 = time.perf_counter()
    features = repo.get_by_application(req.application_id)
    if features is None and req.customer_id:
        features = repo.get_by_customer(req.customer_id)
    if features is None:
        features = {}
    else:
        features = dict(features)
    if req.features:
        features.update(req.features)
    lat_feature = (time.perf_counter() - t0) * 1000.0

    if not features:
        raise HTTPException(
            404,
            f"Không tìm thấy dữ liệu cho hồ sơ {req.application_id}; "
            "hãy truyền trực tiếp trường 'features'",
        )

    cic_missing = bool(features.get("cic_missing", 0))

    # --- (2) chấm điểm và áp luật ------------------------------------------
    result = engine.decide(
        features,
        application_id=req.application_id,
        customer_id=req.customer_id or str(features.get("customer_id", "")),
        cic_degraded=cic_missing,
        latency_feature_ms=lat_feature,
    )

    return ScreeningResponse(
        decision_id=result.decision_id,
        application_id=result.application_id,
        customer_id=result.customer_id,
        recommendation=result.decision,
        reason_category=result.reason_category,
        reason_text=result.reason_text,
        pd_score=result.pd_score,
        risk_band=result.risk_band,
        suggested_amount=result.approved_amount,
        suggested_tenor=result.approved_tenor,
        suggested_rate=result.priced_rate,
        rule_hits=result.rule_hits,
        reason_codes=result.reason_codes,
        processing_ms=result.latency.to_dict(),
        degraded_mode=result.degraded_mode,
        model_version=result.model_version,
        decided_at=result.decided_at,
    )


@app.post("/api/v1/screen/batch", summary="Kiểm duyệt nhiều hồ sơ trong một yêu cầu")
def screen_batch(reqs: List[ApplicationRequest]) -> Dict[str, Any]:
    """Xử lý theo lô - dùng cho tái chấm điểm danh mục hiện hữu."""
    t0 = time.perf_counter()
    results = [screen(r).model_dump() for r in reqs]
    elapsed = (time.perf_counter() - t0) * 1000.0
    mix: Dict[str, int] = {}
    for r in results:
        mix[r["recommendation"]] = mix.get(r["recommendation"], 0) + 1
    return {
        "n": len(results),
        "total_ms": round(elapsed, 2),
        "avg_ms_per_application": round(elapsed / max(len(results), 1), 3),
        "recommendation_mix": mix,
        "results": results,
    }


@app.get("/api/v1/features/{application_id}",
         summary="Xem vector đặc trưng đầu vào của một hồ sơ")
def get_features(application_id: str) -> Dict[str, Any]:
    if not _state:
        raise HTTPException(503, "Dịch vụ chưa khởi tạo xong")
    t0 = time.perf_counter()
    feats = _state["repo"].get_by_application(application_id)
    if not feats:
        raise HTTPException(404, f"Không có dữ liệu cho hồ sơ {application_id}")
    return {
        "application_id": application_id,
        "n_features": len(feats),
        "lookup_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "features": feats,
    }


# ===========================================================================
# Điểm cuối vận hành và giám sát
# ===========================================================================
@app.get("/health", summary="Kiểm tra trạng thái dịch vụ")
def health() -> Dict[str, Any]:
    if not _state:
        return {"status": "starting"}
    return {
        "status": "healthy",
        "started_at": _state.get("started_at"),
        "feature_repository": _state["repo"].stats(),
        "decision_engine": _state["engine"].health(),
    }


@app.get("/metrics", summary="Chỉ tiêu xử lý tích lũy")
def metrics() -> Dict[str, Any]:
    if not _state:
        raise HTTPException(503, "Dịch vụ chưa khởi tạo xong")
    return {
        "screening": _state["engine"].stats.summary(),
        "feature_repository": _state["repo"].stats(),
        "rule_hits": _state["engine"].rule_engine.hit_statistics()[:15],
    }


@app.post("/admin/reset-metrics", summary="Xóa chỉ tiêu tích lũy")
def reset_metrics() -> Dict[str, str]:
    if _state:
        _state["engine"].stats.reset()
        _state["engine"].rule_engine.reset_statistics()
    return {"status": "ok"}


@app.get("/admin/rules", summary="Danh mục luật nghiệp vụ đang áp dụng")
def list_rules() -> Dict[str, Any]:
    if not _state:
        raise HTTPException(503, "Dịch vụ chưa khởi tạo xong")
    engine: DecisionEngine = _state["engine"]
    return {
        "n_rules": len(engine.rule_engine.rules),
        "thresholds": {
            "pd_approve_max": engine.pd_approve_max,
            "pd_reject_min": engine.pd_reject_min,
        },
        "risk_bands": engine.policy.band_table(),
        "rules": engine.rule_engine.catalogue(),
    }
