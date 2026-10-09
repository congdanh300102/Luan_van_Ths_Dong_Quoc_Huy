"""Bước 5 - Mô phỏng quy trình kiểm duyệt và đánh giá chung.

Script thực hiện phần thực nghiệm ở mức quy trình của đề án:

    (1) Mô phỏng xác định ngưỡng phê duyệt theo ba tiêu chí nghiệp vụ
    (2) Mô phỏng luồng xử lý hồ sơ qua bộ máy hỗ trợ ra quyết định
    (3) Thử nghiệm tích hợp dịch vụ REST tiếp nhận và xử lý hồ sơ
    (4) So sánh với quy trình thẩm định truyền thống chỉ dựa trên luật nghiệp vụ

Kết quả (``results/tables``):
    cutoff_criteria.csv         So sánh ba tiêu chí xác định điểm cắt
    cutoff_sensitivity.csv      Độ nhạy của kết quả theo điểm cắt
    decision_mix.csv            Cơ cấu khuyến nghị của mô hình
    rule_hit_statistics.csv     Tần suất vi phạm từng luật nghiệp vụ
    rule_catalogue.csv          Danh mục luật nghiệp vụ đang áp dụng
    risk_bands.csv              Bảng phân hạng rủi ro
    processing_time.csv         Thời gian xử lý một hồ sơ theo từng chặng
    api_integration.csv         Kết quả thử nghiệm tích hợp dịch vụ REST
    traditional_comparison.csv  So sánh với quy trình thẩm định truyền thống

Cách dùng:
    py scripts/05_simulate_evaluate.py
    py scripts/05_simulate_evaluate.py --n-applications 2000 --no-api
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd

from creditreview.config import load_config
from creditreview.decision.engine import Decision, DecisionEngine
from creditreview.decision.rules import RuleEngine
from creditreview.labeling import cutoff as CO
from creditreview.models import plots as P
from creditreview.models import train as T

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")
for noisy in ["matplotlib", "PIL"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ===========================================================================
# 1. Mô phỏng xác định ngưỡng phê duyệt
# ===========================================================================
def analyse_cutoffs(cfg, y: np.ndarray, p: np.ndarray, tables: Path,
                    figures: Path) -> dict:
    log.info("=" * 72)
    log.info("1. Mô phỏng xác định ngưỡng phê duyệt")

    cost_fn = float(cfg["training.cost_fn"])
    cost_fp = float(cfg["training.cost_fp"])
    target_bad = float(cfg.get("labeling.max_bad_rate_approved", 0.05))

    criteria = [
        CO.cutoff_by_cost(y, p, cost_fn, cost_fp),
        CO.cutoff_by_approval_rate(y, p, target_rate=0.70, cost_fn=cost_fn, cost_fp=cost_fp),
        CO.cutoff_by_bad_rate(y, p, max_bad_rate=target_bad, cost_fn=cost_fn, cost_fp=cost_fp),
    ]
    table = pd.DataFrame([c.to_dict() for c in criteria])
    table.to_csv(tables / "cutoff_criteria.csv", index=False, encoding="utf-8-sig")
    log.info("So sánh ba tiêu chí xác định điểm cắt:\n%s", table.to_string(index=False))

    sens = CO.cutoff_sensitivity_table(y, p, cost_fn=cost_fn, cost_fp=cost_fp)
    sens.to_csv(tables / "cutoff_sensitivity.csv", index=False, encoding="utf-8-sig")

    zones = CO.three_zone_cutoffs(
        y, p,
        target_review_rate=float(cfg.get("labeling.target_review_rate", 0.20)),
        cost_fn=cost_fn, cost_fp=cost_fp,
    )
    with open(tables / "cutoff_zones.json", "w", encoding="utf-8") as fh:
        json.dump({k: round(float(v), 6) for k, v in zones.items()}, fh,
                  indent=2, ensure_ascii=False)

    try:
        P.plot_cutoff_sensitivity(sens, figures, zones)
    except Exception as exc:
        log.debug("Không vẽ được đồ thị độ nhạy điểm cắt: %s", exc)
    return zones


# ===========================================================================
# 2. Mô phỏng luồng xử lý hồ sơ
# ===========================================================================
def simulate_screening(cfg, engine: DecisionEngine, part: pd.DataFrame,
                       tables: Path, figures: Path) -> pd.DataFrame:
    log.info("=" * 72)
    log.info("2. Mô phỏng luồng xử lý %s hồ sơ", f"{len(part):,}")

    records = part.to_dict(orient="records")
    rows = []
    t0 = time.perf_counter()
    for rec in records:
        clean = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                 for k, v in rec.items()}
        res = engine.decide(
            clean,
            application_id=str(clean.get("application_id", "")),
            customer_id=str(clean.get("customer_id", "")),
            cic_degraded=bool(clean.get("cic_missing", 0)),
        )
        rows.append({
            "application_id": res.application_id,
            "decision": res.decision,
            "reason_category": res.reason_category,
            "pd_score": res.pd_score,
            "risk_band": res.risk_band,
            "n_rule_hits": len(res.rule_hits),
            "y_true": int(rec.get("default_flag", 0)),
            "total_ms": res.latency.total_ms,
            "model_ms": res.latency.model_ms,
            "rule_ms": res.latency.rule_ms,
            "policy_ms": res.latency.policy_ms,
        })
    elapsed = time.perf_counter() - t0
    out = pd.DataFrame(rows)
    log.info("Xử lý xong trong %.1f giây (%.2f ms/hồ sơ)",
             elapsed, 1000 * elapsed / max(len(records), 1))

    # --- cơ cấu khuyến nghị ------------------------------------------------
    mix_rows = []
    label_vi = {
        Decision.APPROVE.value: "Khuyến nghị duyệt",
        Decision.REVIEW.value: "Chuyển thẩm định thủ công",
        Decision.REJECT.value: "Khuyến nghị từ chối",
    }
    for d in [Decision.APPROVE.value, Decision.REVIEW.value, Decision.REJECT.value]:
        sub = out[out["decision"] == d]
        mix_rows.append({
            "Kết quả": label_vi[d],
            "Số hồ sơ": len(sub),
            "Tỷ trọng (%)": round(100.0 * len(sub) / max(len(out), 1), 2),
            "Tỷ lệ nợ xấu thực tế (%)": round(100.0 * float(sub["y_true"].mean()), 2)
            if len(sub) else 0.0,
        })
    mix = pd.DataFrame(mix_rows)
    mix.to_csv(tables / "decision_mix.csv", index=False, encoding="utf-8-sig")
    log.info("Cơ cấu khuyến nghị:\n%s", mix.to_string(index=False))

    automation = float((out["decision"] != Decision.REVIEW.value).mean())
    log.info("Tỷ lệ hồ sơ được xử lý tự động: %.2f%%", 100 * automation)

    try:
        P.plot_decision_mix(mix, figures)
    except Exception as exc:
        log.debug("Không vẽ được đồ thị cơ cấu quyết định: %s", exc)

    # --- thời gian xử lý theo chặng ---------------------------------------
    stages = [("Chấm điểm mô hình", "model_ms"), ("Áp dụng luật nghiệp vụ", "rule_ms"),
              ("Áp dụng chính sách", "policy_ms"), ("Tổng cộng", "total_ms")]
    timing = pd.DataFrame([
        {
            "Chặng xử lý": name,
            "Trung bình (ms)": round(float(out[col].mean()), 3),
            "p50 (ms)": round(float(np.percentile(out[col], 50)), 3),
            "p95 (ms)": round(float(np.percentile(out[col], 95)), 3),
            "p99 (ms)": round(float(np.percentile(out[col], 99)), 3),
        }
        for name, col in stages
    ])
    timing.to_csv(tables / "processing_time.csv", index=False, encoding="utf-8-sig")
    log.info("Thời gian xử lý một hồ sơ:\n%s", timing.to_string(index=False))

    # --- tần suất vi phạm luật --------------------------------------------
    hits = pd.DataFrame(engine.rule_engine.hit_statistics())
    if not hits.empty:
        hits.to_csv(tables / "rule_hit_statistics.csv", index=False, encoding="utf-8-sig")
        try:
            P.plot_rule_hits(hits, figures)
        except Exception as exc:
            log.debug("Không vẽ được đồ thị tần suất luật: %s", exc)

    pd.DataFrame(engine.rule_engine.catalogue()).to_csv(
        tables / "rule_catalogue.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(engine.policy.band_table()).to_csv(
        tables / "risk_bands.csv", index=False, encoding="utf-8-sig"
    )
    return out


# ===========================================================================
# 3. Thử nghiệm tích hợp dịch vụ REST
# ===========================================================================
def test_api_integration(cfg, part: pd.DataFrame, tables: Path, n: int = 50) -> None:
    log.info("=" * 72)
    log.info("3. Thử nghiệm tích hợp dịch vụ REST")
    try:
        from fastapi.testclient import TestClient

        from creditreview.serving.api import app
    except Exception as exc:
        log.warning("Bỏ qua thử nghiệm API (%s)", exc)
        return

    app_ids = part["application_id"].head(n).astype(str).tolist()
    rows = []
    with TestClient(app) as client:
        r = client.get("/health")
        rows.append({
            "Điểm cuối": "GET /health",
            "Mô tả": "Kiểm tra trạng thái dịch vụ",
            "Mã HTTP": r.status_code,
            "Kết quả": "Đạt" if r.status_code == 200 else "Không đạt",
        })

        t0 = time.perf_counter()
        ok = 0
        for aid in app_ids:
            resp = client.post("/api/v1/screen", json={"application_id": aid})
            if resp.status_code == 200:
                ok += 1
        single_ms = 1000 * (time.perf_counter() - t0) / max(len(app_ids), 1)
        rows.append({
            "Điểm cuối": "POST /api/v1/screen",
            "Mô tả": f"Kiểm duyệt từng hồ sơ ({len(app_ids)} lần gọi)",
            "Mã HTTP": 200,
            "Kết quả": f"{ok}/{len(app_ids)} thành công, {single_ms:.1f} ms/hồ sơ",
        })

        t0 = time.perf_counter()
        rb = client.post(
            "/api/v1/screen/batch",
            json=[{"application_id": a} for a in app_ids],
        )
        batch_ms = 1000 * (time.perf_counter() - t0)
        if rb.status_code == 200:
            body = rb.json()
            rows.append({
                "Điểm cuối": "POST /api/v1/screen/batch",
                "Mô tả": f"Kiểm duyệt theo lô {len(app_ids)} hồ sơ",
                "Mã HTTP": 200,
                "Kết quả": f"{body['n']} hồ sơ, {batch_ms:.0f} ms "
                           f"({body['avg_ms_per_application']:.2f} ms/hồ sơ)",
            })

        rf = client.get(f"/api/v1/features/{app_ids[0]}")
        rows.append({
            "Điểm cuối": "GET /api/v1/features/{id}",
            "Mô tả": "Truy xuất vector đặc trưng đầu vào",
            "Mã HTTP": rf.status_code,
            "Kết quả": f"{rf.json().get('n_features', 0)} đặc trưng"
            if rf.status_code == 200 else "Không đạt",
        })

        rr = client.get("/admin/rules")
        rows.append({
            "Điểm cuối": "GET /admin/rules",
            "Mô tả": "Danh mục luật nghiệp vụ đang áp dụng",
            "Mã HTTP": rr.status_code,
            "Kết quả": f"{rr.json().get('n_rules', 0)} luật"
            if rr.status_code == 200 else "Không đạt",
        })

    table = pd.DataFrame(rows)
    table.to_csv(tables / "api_integration.csv", index=False, encoding="utf-8-sig")
    log.info("Kết quả tích hợp API:\n%s", table.to_string(index=False))


# ===========================================================================
# 4. So sánh với quy trình thẩm định truyền thống
# ===========================================================================
def compare_with_traditional(cfg, part: pd.DataFrame, screened: pd.DataFrame,
                             tables: Path, figures: Path) -> None:
    """Đối chiếu mô hình đề xuất với quy trình chỉ dùng luật nghiệp vụ.

    Quy trình thẩm định truyền thống được mô phỏng bằng bộ luật nghiệp vụ đang
    áp dụng: hồ sơ vi phạm luật loại trừ cứng bị từ chối, hồ sơ vi phạm luật
    cảnh báo được chuyển thẩm định, phần còn lại được duyệt. Đây là mô phỏng
    tối giản của cách làm dựa trên bộ tiêu chí cố định, dùng làm mốc so sánh.
    """
    log.info("=" * 72)
    log.info("4. So sánh với quy trình thẩm định truyền thống")

    rule_engine = RuleEngine(cfg)
    records = part.to_dict(orient="records")
    decisions = []
    t0 = time.perf_counter()
    for rec in records:
        clean = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                 for k, v in rec.items()}
        rr = rule_engine.evaluate(clean)
        if rr.must_reject:
            decisions.append(Decision.REJECT.value)
        elif rr.must_review:
            decisions.append(Decision.REVIEW.value)
        else:
            decisions.append(Decision.APPROVE.value)
    rule_ms = 1000 * (time.perf_counter() - t0) / max(len(records), 1)

    base = pd.DataFrame({
        "decision": decisions,
        "y_true": part["default_flag"].to_numpy(),
    })

    def _summarise(frame: pd.DataFrame, name: str, ms: float) -> dict:
        approved = frame[frame["decision"] == Decision.APPROVE.value]
        review = frame[frame["decision"] == Decision.REVIEW.value]
        rejected = frame[frame["decision"] == Decision.REJECT.value]
        n = max(len(frame), 1)
        return {
            "Phương pháp": name,
            "Tỷ lệ duyệt (%)": round(100.0 * len(approved) / n, 2),
            "Tỷ lệ thẩm định thủ công (%)": round(100.0 * len(review) / n, 2),
            "Tỷ lệ từ chối (%)": round(100.0 * len(rejected) / n, 2),
            "Nợ xấu nhóm duyệt (%)": round(100.0 * float(approved["y_true"].mean()), 2)
            if len(approved) else 0.0,
            "Nợ xấu chặn được (%)": round(100.0 * float(rejected["y_true"].mean()), 2)
            if len(rejected) else 0.0,
            "Tỷ lệ tự động hóa (%)": round(
                100.0 * (len(approved) + len(rejected)) / n, 2),
            "Thời gian xử lý (ms/hồ sơ)": round(ms, 3),
        }

    model_ms = float(screened["total_ms"].mean())
    table = pd.DataFrame([
        _summarise(base, "Quy trình truyền thống (chỉ luật nghiệp vụ)", rule_ms),
        _summarise(screened[["decision", "y_true"]],
                   "Mô hình đề xuất (luật + mô hình học máy)", model_ms),
    ])
    table.to_csv(tables / "traditional_comparison.csv", index=False, encoding="utf-8-sig")
    log.info("So sánh hai phương pháp:\n%s", table.to_string(index=False))
    try:
        P.plot_traditional_comparison(table, figures)
    except Exception as exc:
        log.debug("Không vẽ được đồ thị so sánh: %s", exc)

    trad_bad = float(table.iloc[0]["Nợ xấu nhóm duyệt (%)"])
    ml_bad = float(table.iloc[1]["Nợ xấu nhóm duyệt (%)"])
    if trad_bad > 0:
        log.info(
            "Tỷ lệ nợ xấu trong nhóm hồ sơ được duyệt giảm %.2f điểm phần trăm "
            "(tương đương %.1f%%)",
            trad_bad - ml_bad, 100.0 * (trad_bad - ml_bad) / trad_bad,
        )


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Mô phỏng và đánh giá mô hình")
    ap.add_argument("--n-applications", type=int, default=None,
                    help="Số hồ sơ dùng trong mô phỏng")
    ap.add_argument("--no-api", action="store_true", help="Bỏ qua thử nghiệm tích hợp API")
    args = ap.parse_args()

    cfg = load_config()
    tables = cfg.path("tables")
    figures = cfg.path("figures")
    t_start = time.perf_counter()

    model_path = cfg.path("models") / "production_model.joblib"
    if not model_path.exists():
        log.error("Chưa có mô hình tại %s - hãy chạy scripts/04_train_models.py trước",
                  model_path)
        return 1

    df = T.load_training_data(cfg)
    splits = T.make_splits(df, cfg)
    test = splits["test"]

    n = args.n_applications or int(cfg.get("serving.simulation_applications", 5000))
    part = test.head(min(n, len(test))).copy()

    # ngưỡng phê duyệt tính trên tập kiểm định để không dùng tập kiểm thử hai lần
    preds_path = cfg.path("data_processed") / "test_predictions.parquet"
    bundle = joblib.load(model_path)
    engine = DecisionEngine(cfg, model_bundle=bundle)

    valid = splits["valid"]
    p_valid = np.array([
        engine._score({k: (None if isinstance(v, float) and np.isnan(v) else v)
                       for k, v in rec.items()})
        for rec in valid.to_dict(orient="records")
    ])
    analyse_cutoffs(cfg, valid["default_flag"].to_numpy(), p_valid, tables, figures)

    screened = simulate_screening(cfg, engine, part, tables, figures)

    if not args.no_api:
        test_api_integration(cfg, part, tables)

    compare_with_traditional(cfg, part, screened, tables, figures)

    log.info("=" * 72)
    log.info("HOÀN THÀNH sau %.1f phút", (time.perf_counter() - t_start) / 60)
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
