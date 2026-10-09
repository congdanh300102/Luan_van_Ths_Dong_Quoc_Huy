"""Bước 4 - Huấn luyện, hiệu chuẩn và so sánh các mô hình học máy.

Script sinh toàn bộ bảng biểu và đồ thị phục vụ phần thực nghiệm mô hình:

Bảng (``results/tables``):
    model_comparison.csv            So sánh các mô hình theo đầy đủ chỉ tiêu
    scorecard.csv                   Thẻ điểm WOE
    gains_table.csv                 Phân tích theo thập phân vị
    shap_group_share.csv            Tỷ trọng đóng góp theo nhóm đặc trưng
    shap_importance.csv             Tầm quan trọng đặc trưng theo SHAP
    permutation_importance.csv      Tầm quan trọng theo phép hoán vị
    drift_psi.csv                   PSI giữa tập huấn luyện và tập kiểm thử
    performance_over_time.csv       AUC/KS theo từng tháng
    reason_codes_example.csv        Ví dụ lý do từ chối cho một hồ sơ cụ thể
    experiment_meta.json            Siêu dữ liệu thực nghiệm

Đồ thị (``results/figures``): xem ``creditreview/models/plots.py``.

Cách dùng:
    py scripts/04_train_models.py
    py scripts/04_train_models.py --models logistic_regression lightgbm
    py scripts/04_train_models.py --no-shap --no-permutation
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
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from creditreview.config import load_config
from creditreview.features import definitions as fd
from creditreview.models import explain as X
from creditreview.models import metrics as M
from creditreview.models import plots as P
from creditreview.models import train as T
from creditreview.monitoring import drift as D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")
for noisy in ["matplotlib", "shap", "numba", "PIL"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser(description="Huấn luyện mô hình rủi ro tín dụng")
    ap.add_argument("--models", nargs="*", default=None, help="Danh sách mô hình cần huấn luyện")
    ap.add_argument("--no-stacking", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-permutation", action="store_true")
    ap.add_argument("--no-scorecard", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    figures = cfg.path("figures")
    tables = cfg.path("tables")
    t_start = time.perf_counter()

    # =====================================================================
    # 1. HUẤN LUYỆN VÀ SO SÁNH MÔ HÌNH
    # =====================================================================
    res = T.run_experiment(cfg, model_names=args.models, include_stacking=not args.no_stacking)
    trained, splits, comparison = res["trained"], res["splits"], res["comparison"]
    feature_cols, cat_cols = res["feature_cols"], res["categorical_cols"]
    best_name, best = res["best_name"], trained[res["best_name"]]
    meta = res["meta"]

    log.info("\n%s", comparison.drop(columns=["model_key"]).to_string(index=False))

    # =====================================================================
    # 2. ĐỒ THỊ ĐÁNH GIÁ MÔ HÌNH
    # =====================================================================
    log.info("=" * 72)
    log.info("Sinh đồ thị đánh giá mô hình...")
    y_test = splits["test"]["default_flag"].to_numpy()
    preds = {tm.display_name: tm.predict_from_raw(splits["test"]) for tm in trained.values()}
    P.plot_roc_curves(y_test, preds, figures)
    P.plot_pr_curves(y_test, preds, figures)
    P.plot_calibration(y_test, preds, figures)
    P.plot_metric_comparison(comparison, figures)
    P.plot_auc_vs_latency(comparison, figures)

    p_best = preds[best.display_name]
    P.plot_ks_curve(y_test, p_best, figures, best.display_name)
    tr = best.test_report
    P.plot_confusion_matrix(tr.tn, tr.fp, tr.fn, tr.tp, figures, best.display_name)
    P.plot_score_distribution(
        y_test, p_best, figures,
        {
            "Ngưỡng khuyến nghị duyệt": float(cfg["decision.pd_approve_max"]),
            "Ngưỡng khuyến nghị từ chối": float(cfg["decision.pd_reject_min"]),
            "Ngưỡng tối ưu chi phí": best.threshold,
        },
    )
    gains = M.gains_table(y_test, p_best, n_bins=10)
    P.plot_gains(gains, figures)

    # =====================================================================
    # 3. THẺ ĐIỂM WOE
    # =====================================================================
    if not args.no_scorecard:
        card = T.export_scorecard(trained, cfg)
        if card is not None:
            log.info("Đã xuất thẻ điểm: %d dòng", len(card))

    # =====================================================================
    # 4. GIẢI THÍCH MÔ HÌNH: SHAP
    # =====================================================================
    windows = list(cfg.get("features.txn_windows_days", [7, 30, 90, 180]))
    group_map = {f.name: f.group.value for f in fd.all_features(windows)}

    if not args.no_shap:
        log.info("=" * 72)
        log.info("Phân tích SHAP cho mô hình %s...", best.display_name)
        X_test = best.prepare(splits["test"])
        shap_res = X.shap_analysis(best.model, X_test, max_samples=2500, seed=cfg.seed)
        if shap_res:
            shap_res["importance"].to_csv(
                tables / "shap_importance.csv", index=False, encoding="utf-8-sig"
            )
            X.plot_shap_summary(shap_res, figures, top_n=20)
            grp = X.shap_by_group(shap_res["importance"], group_map)
            grp.to_csv(tables / "shap_group_share.csv", index=False, encoding="utf-8-sig")
            X.plot_shap_group_share(grp, figures)
            log.info("Đóng góp theo nhóm đặc trưng:\n%s", grp.to_string(index=False))
            P.plot_feature_importance(
                shap_res["importance"].rename(columns={"shap_mean_abs": "importance"}),
                figures, top_n=25, value_col="importance",
                title=f"Tầm quan trọng đặc trưng theo SHAP - {best.display_name}",
                name="shap_importance_bar",
            )

            # ví dụ lý do từ chối cho một hồ sơ rủi ro cao
            risky_idx = int(np.argmax(shap_res["values"].sum(axis=1)))
            reasons = X.local_explanation(shap_res, risky_idx, top_n=8)
            if not reasons.empty:
                reasons.to_csv(tables / "reason_codes_example.csv", index=False,
                               encoding="utf-8-sig")
                log.info("Ví dụ lý do quyết định:\n%s", reasons.to_string(index=False))

    # =====================================================================
    # 5. TẦM QUAN TRỌNG THEO PHÉP HOÁN VỊ
    # =====================================================================
    if not args.no_permutation:
        log.info("=" * 72)
        X_test = best.prepare(splits["test"])
        # giới hạn ở 40 đặc trưng mạnh nhất theo SHAP để kiểm soát thời gian tính
        try:
            top_feats = pd.read_csv(tables / "shap_importance.csv")["feature"].head(40).tolist()
        except Exception:
            top_feats = feature_cols[:40]
        perm = X.permutation_analysis(
            best.predict_proba, X_test, y_test, features=top_feats,
            n_repeats=5, max_samples=6000, seed=cfg.seed,
        )
        perm.to_csv(tables / "permutation_importance.csv", index=False, encoding="utf-8-sig")
        P.plot_feature_importance(
            perm, figures, top_n=25, value_col="importance",
            title=f"Tầm quan trọng đặc trưng theo phép hoán vị - {best.display_name}"
                  f"\n(mức suy giảm AUC, AUC gốc = {perm.attrs.get('base_auc', 0):.4f})",
            name="permutation_importance_bar",
        )

    # =====================================================================
    # 6. ĐỘ ỔN ĐỊNH CỦA MÔ HÌNH THEO THỜI GIAN
    # =====================================================================
    log.info("=" * 72)
    log.info("Phân tích trôi dạt dữ liệu và chất lượng mô hình theo thời gian...")
    numeric = [c for c in feature_cols if c not in cat_cols]
    psi = D.drift_report(
        splits["train"], splits["test"], numeric, cat_cols,
        n_bins=int(cfg.get("monitoring.psi_bins", 10)),
        warn=float(cfg.get("monitoring.psi_warn", 0.10)),
        alert=float(cfg.get("monitoring.psi_alert", 0.25)),
    )
    psi.to_csv(tables / "drift_psi.csv", index=False, encoding="utf-8-sig")
    P.plot_drift(psi, figures, top_n=20)
    log.info("PSI: %d đặc trưng cảnh báo, %d báo động",
             int((psi["status"] == "CẢNH BÁO").sum()), int((psi["status"] == "BÁO ĐỘNG").sum()))

    perf = D.performance_over_time(
        y_test, p_best, splits["test"]["apply_time"], freq="M", min_samples=150
    )
    if not perf.empty:
        perf.to_csv(tables / "performance_over_time.csv", index=False, encoding="utf-8-sig")
        D.plot_performance_over_time(
            perf, figures, auc_alert_drop=float(cfg.get("monitoring.auc_drop_alert", 0.05))
        )
        log.info("Chất lượng theo tháng:\n%s", perf.to_string(index=False))

    # =====================================================================
    # 7. ĐỒ THỊ MÔ TẢ DỮ LIỆU
    # =====================================================================
    log.info("=" * 72)
    raw = cfg.path("data_raw")
    try:
        P.plot_data_overview(
            pd.read_parquet(raw / "customers.parquet"),
            pd.read_parquet(raw / "applications.parquet"),
            pd.read_parquet(raw / "labels.parquet"),
            figures,
        )
    except Exception as exc:
        log.warning("Không sinh được đồ thị mô tả dữ liệu: %s", exc)

    # =====================================================================
    # Tổng kết
    # =====================================================================
    elapsed = time.perf_counter() - t_start
    meta["total_training_seconds"] = round(elapsed, 1)
    meta["n_models_trained"] = len(trained)
    with open(tables / "experiment_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    log.info("=" * 72)
    log.info("HOÀN THÀNH sau %.1f phút", elapsed / 60)
    log.info("Số mô hình đã huấn luyện: %d", len(trained))
    log.info("Mô hình tốt nhất: %s", best.display_name)
    log.info("  AUC  = %.4f | Gini = %.4f | KS = %.4f", tr.roc_auc, tr.gini, tr.ks)
    log.info("  Trần Bayes (PD thực) = %.4f", meta["bayes_auc_test"])
    log.info("  Thời gian chấm điểm p95 = %.3f ms", best.latency_ms["p95"])
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
