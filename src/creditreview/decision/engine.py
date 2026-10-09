"""Bộ máy hỗ trợ ra quyết định - kết hợp luật nghiệp vụ và mô hình học máy.

VAI TRÒ HỖ TRỢ, KHÔNG THAY THẾ
------------------------------
Kết quả của module này là **khuyến nghị** kèm lý do, phục vụ cán bộ tín dụng
trong quá trình kiểm duyệt; thẩm quyền quyết định cấp tín dụng vẫn thuộc về
người hoặc bộ phận có thẩm quyền. Thiết kế ba vùng quyết định (duyệt / chuyển
thẩm định / từ chối) phản ánh trực tiếp định hướng này: mô hình chỉ đưa ra
khuyến nghị dứt khoát ở hai vùng biên, còn vùng giữa -- nơi mô hình không đủ
chắc chắn -- được chuyển cho con người xem xét.

LOGIC RA QUYẾT ĐỊNH
-------------------
Khuyến nghị cuối cùng là kết quả của một chuỗi ưu tiên rõ ràng, trong đó luật
nghiệp vụ có quyền phủ quyết cao hơn mô hình thống kê:

    1. Có luật HARD_REJECT bị vi phạm            -> REJECT
    2. Dữ liệu CIC không khả dụng (suy giảm)     -> REVIEW
    3. PD > pd_reject_min                        -> REJECT
    4. Có luật REVIEW bị vi phạm                 -> REVIEW
    5. PD > pd_approve_max                       -> REVIEW
    6. Hạn mức duyệt được bằng 0                 -> REJECT
    7. Còn lại                                   -> APPROVE

Thiết kế theo ba vùng quyết định (thay vì hai) là điểm then chốt về nghiệp vụ:
vùng REVIEW ở giữa giúp quy trình **xử lý tự động phần hồ sơ có thể xử lý tự
động** mà không buộc phải đưa ra khuyến nghị máy móc với những hồ sơ nằm trong
vùng không chắc chắn. Tỷ lệ hồ sơ được xử lý tự động (APPROVE + REJECT) chính là
thước đo mức độ giảm tải công việc thủ công mà mô hình mang lại.

GIẢI TRÌNH QUYẾT ĐỊNH
---------------------
Mọi khuyến nghị đều kèm ``reason_codes`` -- danh sách yếu tố có ảnh hưởng lớn
nhất, gồm các luật nghiệp vụ bị vi phạm và các chỉ tiêu định lượng then chốt.
Đây là yêu cầu bắt buộc đối với mô hình dùng trong hoạt động cấp tín dụng: cán
bộ tín dụng phải hiểu được vì sao mô hình đưa ra khuyến nghị đó thì mới có căn
cứ để chấp nhận hoặc bác bỏ.

ĐO THỜI GIAN XỬ LÝ
------------------
Mỗi quyết định được đo thời gian theo từng chặng: chuẩn bị đặc trưng, truy vấn
CIC, chấm điểm mô hình, chạy luật. Phân rã này cho biết thành phần nào chiếm
phần lớn thời gian xử lý một hồ sơ, làm căn cứ đánh giá khả năng tích hợp hệ
thống vào quy trình kiểm duyệt trực tuyến.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from .policy import CreditPolicy
from .rules import RuleEngine, RuleResult

log = logging.getLogger(__name__)


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class ReasonCategory(str, Enum):
    RULE_KNOCKOUT = "RULE_KNOCKOUT"
    HIGH_RISK_SCORE = "HIGH_RISK_SCORE"
    RULE_REVIEW = "RULE_REVIEW"
    SCORE_GREY_ZONE = "SCORE_GREY_ZONE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    ZERO_LIMIT = "ZERO_LIMIT"
    AUTO_APPROVED = "AUTO_APPROVED"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class LatencyBreakdown:
    """Phân rã thời gian xử lý theo từng chặng (mili giây)."""

    feature_ms: float = 0.0
    cic_ms: float = 0.0
    model_ms: float = 0.0
    rule_ms: float = 0.0
    policy_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass
class DecisionResult:
    """Kết quả phê duyệt một hồ sơ."""

    decision_id: str
    application_id: str
    customer_id: str
    decision: str
    reason_category: str
    reason_text: str
    pd_score: float
    risk_band: str
    approved_amount: float
    approved_tenor: int
    priced_rate: float
    expected_loss: float
    rule_hits: List[Dict[str, str]] = field(default_factory=list)
    reason_codes: List[Dict[str, Any]] = field(default_factory=list)
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    cic_from_cache: bool = False
    degraded_mode: bool = False
    model_version: str = ""
    decided_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["latency"] = self.latency.to_dict()
        return d

    @property
    def is_automated(self) -> bool:
        """Hồ sơ được xử lý tự động (không cần can thiệp của con người)."""
        return self.decision in (Decision.APPROVE.value, Decision.REJECT.value)


@dataclass
class EngineStats:
    total: int = 0
    approve: int = 0
    reject: int = 0
    review: int = 0
    degraded: int = 0
    cache_hits: int = 0
    errors: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    stage_latencies: Dict[str, List[float]] = field(
        default_factory=lambda: {k: [] for k in
                                ["feature_ms", "cic_ms", "model_ms", "rule_ms", "policy_ms"]}
    )
    reason_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def automation_rate(self) -> float:
        return (self.approve + self.reject) / self.total if self.total else 0.0

    def record(self, r: DecisionResult) -> None:
        self.total += 1
        if r.decision == Decision.APPROVE.value:
            self.approve += 1
        elif r.decision == Decision.REJECT.value:
            self.reject += 1
        else:
            self.review += 1
        if r.degraded_mode:
            self.degraded += 1
        if r.cic_from_cache:
            self.cache_hits += 1
        self.latencies_ms.append(r.latency.total_ms)
        for k, v in r.latency.to_dict().items():
            if k in self.stage_latencies:
                self.stage_latencies[k].append(v)
        self.reason_counts[r.reason_category] = self.reason_counts.get(r.reason_category, 0) + 1

    def summary(self) -> Dict[str, Any]:
        lat = np.asarray(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        out = {
            "total": self.total,
            "approve": self.approve,
            "reject": self.reject,
            "review": self.review,
            "approve_rate": round(self.approve / max(self.total, 1), 4),
            "reject_rate": round(self.reject / max(self.total, 1), 4),
            "review_rate": round(self.review / max(self.total, 1), 4),
            "automation_rate": round(self.automation_rate, 4),
            "degraded_rate": round(self.degraded / max(self.total, 1), 4),
            "cache_hit_rate": round(self.cache_hits / max(self.total, 1), 4),
            "errors": self.errors,
            "latency_mean_ms": round(float(lat.mean()), 3),
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 3),
            "latency_p90_ms": round(float(np.percentile(lat, 90)), 3),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 3),
            "latency_p99_ms": round(float(np.percentile(lat, 99)), 3),
            "latency_max_ms": round(float(lat.max()), 3),
        }
        for stage, vals in self.stage_latencies.items():
            if vals:
                a = np.asarray(vals)
                out[f"{stage}_mean"] = round(float(a.mean()), 3)
                out[f"{stage}_p95"] = round(float(np.percentile(a, 95)), 3)
        return out

    def reset(self) -> None:
        self.total = self.approve = self.reject = self.review = 0
        self.degraded = self.cache_hits = self.errors = 0
        self.latencies_ms.clear()
        for v in self.stage_latencies.values():
            v.clear()
        self.reason_counts.clear()


# ===========================================================================
# Decision Engine
# ===========================================================================
class DecisionEngine:
    """Bộ máy ra quyết định tín dụng.

    Tham số
    -------
    model_bundle
        Dict chứa ``model``, ``calibrator``, ``threshold``, ``feature_cols``,
        ``categorical_cols``, ``cat_categories`` - kết quả của bước huấn luyện.
    explainer
        Đối tượng sinh *reason codes* (tùy chọn). Nếu có, mỗi quyết định sẽ kèm
        danh sách yếu tố ảnh hưởng lớn nhất - phục vụ yêu cầu giải trình.
    """

    def __init__(
        self,
        cfg: Config,
        model_bundle: Optional[Dict[str, Any]] = None,
        explainer: Any = None,
    ) -> None:
        self.cfg = cfg
        self.rule_engine = RuleEngine(cfg)
        self.policy = CreditPolicy(cfg)
        self.stats = EngineStats()
        self.explainer = explainer

        self.pd_approve_max = float(cfg.get("decision.pd_approve_max", 0.08))
        self.pd_reject_min = float(cfg.get("decision.pd_reject_min", 0.32))

        self.model = None
        self.calibrator = None
        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.cat_categories: Dict[str, Any] | None = None
        self.model_version = "none"
        if model_bundle:
            self.load_model(model_bundle)

    # -- nạp mô hình -------------------------------------------------------
    def load_model(self, bundle: Dict[str, Any]) -> None:
        self.model = bundle["model"]
        self.calibrator = bundle.get("calibrator")
        self.feature_cols = list(bundle.get("feature_cols", []))
        self.categorical_cols = list(bundle.get("categorical_cols", []))
        self.cat_categories = bundle.get("cat_categories")
        self.model_version = (
            f"{bundle.get('spec_name', 'model')}"
            f"@auc={bundle.get('test_auc', 0):.4f}" if bundle.get("test_auc")
            else str(bundle.get("spec_name", "model"))
        )
        log.info("Decision Engine đã nạp mô hình: %s", self.model_version)

    # -- suy luận ----------------------------------------------------------
    def _score(self, features: Dict[str, Any]) -> float:
        """Tính PD cho một hồ sơ."""
        if self.model is None:
            raise RuntimeError("Chưa nạp mô hình vào Decision Engine")
        row = pd.DataFrame([{c: features.get(c) for c in self.feature_cols}])

        # Khi dựng khung dữ liệu một dòng từ dict, một cột số có giá trị thiếu
        # (None) sẽ được suy ra kiểu ``object`` thay vì kiểu số. Một số thư viện
        # -- XGBoost chẳng hạn -- từ chối kiểu này. Vì vậy cần ép kiểu tường minh
        # cho toàn bộ cột không phải đặc trưng phân loại.
        numeric_cols = [c for c in self.feature_cols if c not in set(self.categorical_cols)]
        for c in numeric_cols:
            row[c] = pd.to_numeric(row[c], errors="coerce")

        # bảo đảm định dạng đặc trưng phân loại giống lúc huấn luyện
        if self.cat_categories:
            for c in self.categorical_cols:
                if c in self.cat_categories:
                    row[c] = pd.Categorical(row[c], categories=self.cat_categories[c])
        elif self.categorical_cols:
            for c in self.categorical_cols:
                row[c] = row[c].astype("object").where(row[c].notna(), "__NA__").astype(str)
        p = float(np.asarray(self.model.predict_proba(row))[:, 1][0])
        if self.calibrator is not None:
            p = float(self.calibrator.predict([p])[0])
        return float(min(max(p, 1e-6), 1 - 1e-6))

    # -- ra quyết định -----------------------------------------------------
    def decide(
        self,
        features: Dict[str, Any],
        application_id: str = "",
        customer_id: str = "",
        cic_from_cache: bool = False,
        cic_degraded: bool = False,
        latency_feature_ms: float = 0.0,
        latency_cic_ms: float = 0.0,
    ) -> DecisionResult:
        """Ra quyết định cho một hồ sơ đã có đầy đủ đặc trưng."""
        t_start = time.perf_counter()
        lat = LatencyBreakdown(feature_ms=latency_feature_ms, cic_ms=latency_cic_ms)

        app_id = application_id or str(features.get("application_id", ""))
        cus_id = customer_id or str(features.get("customer_id", ""))

        # --- (1) chạy luật nghiệp vụ --------------------------------------
        t0 = time.perf_counter()
        rule_result: RuleResult = self.rule_engine.evaluate(features)
        lat.rule_ms = (time.perf_counter() - t0) * 1000.0

        # --- (2) chấm điểm mô hình ----------------------------------------
        pd_score = float("nan")
        model_error: Optional[str] = None
        t0 = time.perf_counter()
        try:
            pd_score = self._score(features)
        except Exception as exc:
            model_error = str(exc)
            log.error("Lỗi chấm điểm hồ sơ %s: %s", app_id, exc)
            self.stats.errors += 1
        lat.model_ms = (time.perf_counter() - t0) * 1000.0

        # --- (3) áp dụng chính sách ---------------------------------------
        t0 = time.perf_counter()
        pd_for_policy = pd_score if pd_score == pd_score else 1.0
        outcome = self.policy.apply(features, pd_for_policy)
        lat.policy_ms = (time.perf_counter() - t0) * 1000.0

        # --- (4) tổng hợp quyết định theo thứ tự ưu tiên ------------------
        decision, reason_cat, reason_text = self._resolve(
            rule_result, pd_score, outcome, cic_degraded, model_error
        )

        # --- (5) sinh reason codes ----------------------------------------
        reason_codes = self._build_reason_codes(rule_result, features, pd_score, outcome)

        lat.total_ms = (time.perf_counter() - t_start) * 1000.0 + lat.feature_ms + lat.cic_ms

        result = DecisionResult(
            decision_id=str(uuid.uuid4()),
            application_id=app_id,
            customer_id=cus_id,
            decision=decision.value,
            reason_category=reason_cat.value,
            reason_text=reason_text,
            pd_score=round(pd_score, 6) if pd_score == pd_score else -1.0,
            risk_band=outcome.risk_band,
            approved_amount=outcome.approved_amount if decision is Decision.APPROVE else 0.0,
            approved_tenor=outcome.approved_tenor if decision is Decision.APPROVE else 0,
            priced_rate=outcome.priced_rate if decision is Decision.APPROVE else 0.0,
            expected_loss=outcome.expected_loss if decision is Decision.APPROVE else 0.0,
            rule_hits=[h.to_dict() for h in rule_result.hits],
            reason_codes=reason_codes,
            latency=lat,
            cic_from_cache=cic_from_cache,
            degraded_mode=cic_degraded or model_error is not None,
            model_version=self.model_version,
            decided_at=datetime.now(timezone.utc).isoformat(),
        )
        self.stats.record(result)
        return result

    def _resolve(
        self,
        rule_result: RuleResult,
        pd_score: float,
        outcome: Any,
        cic_degraded: bool,
        model_error: Optional[str],
    ) -> tuple[Decision, ReasonCategory, str]:
        """Áp dụng thứ tự ưu tiên để xác định quyết định cuối cùng."""
        # 1. luật loại trừ cứng có quyền phủ quyết cao nhất
        if rule_result.must_reject:
            h = rule_result.hard_rejects[0]
            return (
                Decision.REJECT, ReasonCategory.RULE_KNOCKOUT,
                f"Từ chối do vi phạm luật {h.code}: {h.message}",
            )

        # 2. lỗi mô hình -> không tự động hóa
        if model_error is not None:
            return (
                Decision.REVIEW, ReasonCategory.SYSTEM_ERROR,
                "Chuyển thẩm định: không chấm điểm được do lỗi khi chạy mô hình",
            )

        # 3. dữ liệu CIC không khả dụng -> không phê duyệt tự động
        if cic_degraded:
            return (
                Decision.REVIEW, ReasonCategory.DATA_UNAVAILABLE,
                "Chuyển thẩm định: không truy vấn được dữ liệu CIC trong thời gian quy định",
            )

        # 4. rủi ro quá cao theo mô hình
        if pd_score > self.pd_reject_min:
            return (
                Decision.REJECT, ReasonCategory.HIGH_RISK_SCORE,
                f"Từ chối do xác suất vỡ nợ dự báo {pd_score:.2%} "
                f"vượt ngưỡng {self.pd_reject_min:.2%}",
            )

        # 5. luật yêu cầu thẩm định
        if rule_result.must_review:
            h = rule_result.reviews[0]
            return (
                Decision.REVIEW, ReasonCategory.RULE_REVIEW,
                f"Chuyển thẩm định do luật {h.code}: {h.message}",
            )

        # 6. vùng xám của điểm rủi ro
        if pd_score > self.pd_approve_max:
            return (
                Decision.REVIEW, ReasonCategory.SCORE_GREY_ZONE,
                f"Chuyển thẩm định: xác suất vỡ nợ {pd_score:.2%} nằm trong vùng "
                f"[{self.pd_approve_max:.2%}; {self.pd_reject_min:.2%}]",
            )

        # 7. hạn mức tính được bằng 0
        if outcome.approved_amount <= 0:
            return (
                Decision.REJECT, ReasonCategory.ZERO_LIMIT,
                f"Từ chối: hạn mức tính được bằng 0 do ràng buộc "
                f"'{outcome.limit_binding_constraint}'",
            )

        # 8. đủ điều kiện phê duyệt tự động
        note = ""
        if outcome.amount_reduced:
            note = (f" (số tiền duyệt giảm so với đề nghị do ràng buộc "
                    f"'{outcome.limit_binding_constraint}')")
        return (
            Decision.APPROVE, ReasonCategory.AUTO_APPROVED,
            f"Phê duyệt tự động: hạng rủi ro {outcome.risk_band}, "
            f"PD = {pd_score:.2%}{note}",
        )

    def _build_reason_codes(
        self, rule_result: RuleResult, features: Dict[str, Any],
        pd_score: float, outcome: Any,
    ) -> List[Dict[str, Any]]:
        """Sinh danh sách lý do phục vụ giải trình quyết định."""
        codes: List[Dict[str, Any]] = []
        # ưu tiên các luật bị vi phạm (lý do tường minh, dễ giải thích nhất)
        for h in rule_result.hits[:5]:
            codes.append({"source": "RULE", "code": h.code, "detail": h.message,
                          "action": h.action})
        # bổ sung các yếu tố định lượng chính
        for key, label in [
            ("dti", "Tỷ lệ nợ trên thu nhập"),
            ("cic_score", "Điểm tín dụng CIC"),
            ("dpd_max_12m", "Số ngày quá hạn tối đa 12 tháng"),
            ("credit_utilization", "Tỷ lệ sử dụng hạn mức"),
        ]:
            v = features.get(key)
            if v is not None and v == v:
                codes.append({"source": "FEATURE", "code": key, "detail": label,
                              "value": round(float(v), 4)})
        if self.explainer is not None:
            try:
                codes.extend(self.explainer(features))
            except Exception as exc:  # pragma: no cover
                log.debug("Explainer lỗi: %s", exc)
        return codes[:12]

    # -- báo cáo -----------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "model_loaded": self.model is not None,
            "n_rules": len(self.rule_engine.rules),
            "pd_approve_max": self.pd_approve_max,
            "pd_reject_min": self.pd_reject_min,
            **self.stats.summary(),
        }


__all__ = [
    "DecisionEngine", "DecisionResult", "Decision", "ReasonCategory",
    "LatencyBreakdown", "EngineStats",
]
