"""Bước 1 - Thu thập, tổng hợp, làm sạch dữ liệu và xây dựng nhãn rủi ro.

Script thực hiện toàn bộ khối công việc về dữ liệu của đề án:

    (1) Sinh/thu thập dữ liệu tín dụng và ghi ra kho dữ liệu cục bộ (Parquet)
    (2) Phân tích thống kê mô tả tập dữ liệu
    (3) Phát hiện bản ghi không hợp lệ theo ràng buộc nghiệp vụ
    (4) Phân tích giá trị khuyết thiếu và giá trị ngoại lệ
    (5) Xây dựng nhãn rủi ro theo tiêu chuẩn nợ xấu và cửa sổ quan sát
    (6) Phân chia tập huấn luyện - kiểm định - kiểm thử theo thời gian

Kết quả (``results/tables``):
    dataset_summary.csv          Quy mô các bảng dữ liệu
    data_profile.csv             Bảng mô tả từng biến
    data_validity.csv            Kết quả kiểm tra ràng buộc nghiệp vụ
    missing_values.csv           Tỷ lệ khuyết theo biến và chiến lược xử lý
    outliers.csv                 Thống kê ngoại lệ theo quy tắc IQR
    label_distribution.csv       Phân bố ba trạng thái rủi ro
    vintage_analysis.csv         Tỷ lệ nợ xấu tích luỹ theo tháng
    label_definition.csv         Đặc tả quy tắc gán nhãn
    data_split.csv               Kết quả phân chia tập dữ liệu

Cách dùng:
    py scripts/01_prepare_data.py
    py scripts/01_prepare_data.py --n-customers 5000 --n-applications 8000
    py scripts/01_prepare_data.py --skip-generate     # dùng lại dữ liệu đã sinh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from creditreview.config import load_config
from creditreview.data import CreditDataGenerator
from creditreview.data import profiling as PF
from creditreview.features import definitions as fd
from creditreview.labeling import target as LB
from creditreview.preprocessing import cleaning as CL
from creditreview.preprocessing import splitting as SP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare")


def main() -> int:
    ap = argparse.ArgumentParser(description="Chuẩn bị dữ liệu tín dụng")
    ap.add_argument("--n-customers", type=int, default=None)
    ap.add_argument("--n-applications", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip-generate", action="store_true",
                    help="Bỏ qua bước sinh dữ liệu, dùng lại dữ liệu đã có")
    args = ap.parse_args()

    cfg = load_config()
    if args.n_customers:
        cfg.data["datagen"]["n_customers"] = args.n_customers
    if args.n_applications:
        cfg.data["datagen"]["n_applications"] = args.n_applications

    raw = cfg.path("data_raw")
    tables = cfg.path("tables")
    t_start = time.perf_counter()

    # =====================================================================
    # 1. THU THẬP VÀ TỔNG HỢP DỮ LIỆU
    # =====================================================================
    if not args.skip_generate:
        log.info("=" * 72)
        log.info("1. Thu thập và tổng hợp dữ liệu")
        t0 = time.perf_counter()
        gen = CreditDataGenerator(cfg, seed=args.seed)
        data = gen.run()
        elapsed = time.perf_counter() - t0

        log.info("Ghi dữ liệu ra %s ...", raw)
        for name, df in data.tables().items():
            path = raw / f"{name}.parquet"
            df.to_parquet(path, index=False, compression="snappy")
            log.info("  %-14s %10s dòng  %7.1f MB",
                     name, f"{len(df):,}", path.stat().st_size / 1e6)

        data.meta["generation_seconds"] = round(elapsed, 2)
        with open(raw / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(data.meta, fh, indent=2, ensure_ascii=False)

    customers = pd.read_parquet(raw / "customers.parquet")
    accounts = pd.read_parquet(raw / "accounts.parquet")
    applications = pd.read_parquet(raw / "applications.parquet")
    cic_reports = pd.read_parquet(raw / "cic_reports.parquet")
    labels = pd.read_parquet(raw / "labels.parquet")
    features = pd.read_parquet(raw / "features.parquet")
    with open(raw / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    summary = PF.dataset_summary_table(
        customers, accounts, int(meta.get("n_transactions", 0)),
        applications, cic_reports, labels,
    )
    summary.to_csv(tables / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    log.info("Quy mô dữ liệu:\n%s", summary.to_string(index=False))

    # =====================================================================
    # 2. PHÂN TÍCH THỐNG KÊ MÔ TẢ
    # =====================================================================
    log.info("=" * 72)
    log.info("2. Phân tích thống kê mô tả")
    profile = PF.describe_dataset(features)
    windows = list(cfg.get("features.txn_windows_days", [7, 30, 90, 180]))
    group_map = {f.name: f.group.value for f in fd.all_features(windows)}
    desc_map = {f.name: f.description for f in fd.all_features(windows)}
    profile.insert(1, "group", profile["variable"].map(group_map).fillna(""))
    profile.insert(2, "description", profile["variable"].map(desc_map).fillna(""))
    profile.to_csv(tables / "data_profile.csv", index=False, encoding="utf-8-sig")
    log.info("Đã lập bảng mô tả cho %d biến", len(profile))

    # =====================================================================
    # 3. PHÁT HIỆN DỮ LIỆU KHÔNG HỢP LỆ
    # =====================================================================
    log.info("=" * 72)
    log.info("3. Kiểm tra tính hợp lệ theo ràng buộc nghiệp vụ")
    validity = PF.detect_invalid_records(customers, applications, cic_reports)
    validity.to_csv(tables / "data_validity.csv", index=False, encoding="utf-8-sig")
    n_bad = int(validity["n_violations"].sum())
    log.info("Số bản ghi vi phạm ràng buộc nghiệp vụ: %s", f"{n_bad:,}")

    # Kiểm chứng cơ chế phát hiện bằng phép tiêm lỗi: nếu dữ liệu vốn đã sạch,
    # bảng kiểm tra toàn số 0 không chứng minh được cơ chế hoạt động đúng.
    cus_i, app_i, cic_i, injected = PF.inject_quality_issues(
        customers, applications, cic_reports, rate=0.01, seed=cfg.seed
    )
    detected = PF.detect_invalid_records(cus_i, app_i, cic_i)
    check = PF.verify_detection(injected, detected)
    check.to_csv(tables / "data_validity_check.csv", index=False, encoding="utf-8-sig")
    log.info("Kiểm chứng cơ chế phát hiện lỗi dữ liệu:\n%s", check.to_string(index=False))

    # =====================================================================
    # 4. GIÁ TRỊ KHUYẾT THIẾU VÀ NGOẠI LỆ
    # =====================================================================
    log.info("=" * 72)
    log.info("4. Phân tích giá trị khuyết thiếu và ngoại lệ")
    mv = CL.missing_value_report(features, group_map)
    mv.table.to_csv(tables / "missing_values.csv", index=False, encoding="utf-8-sig")
    high = mv.high_missing(float(cfg.get("preprocessing.max_missing_pct", 30.0)))
    if high:
        log.warning("Biến có tỷ lệ khuyết cao cần xem xét: %s", ", ".join(high[:10]))

    numeric_cols = [c for c in fd.numeric_features(windows) if c in features.columns]
    out = CL.outlier_report(
        features, numeric_cols, k_iqr=float(cfg.get("preprocessing.outlier_iqr_k", 3.0))
    )
    out.to_csv(tables / "outliers.csv", index=False, encoding="utf-8-sig")

    # =====================================================================
    # 5. XÂY DỰNG NHÃN RỦI RO
    # =====================================================================
    log.info("=" * 72)
    log.info("5. Xây dựng nhãn rủi ro")
    definition = LB.LabelDefinition(
        performance_window_months=int(cfg.get("labeling.performance_window_months", 12)),
        bad_debt_dpd=int(cfg.get("labeling.bad_debt_dpd", 90)),
        bad_debt_groups=tuple(cfg.get("labeling.bad_debt_groups", [3, 4, 5])),
    )
    pd.DataFrame(
        [{"Tiêu chí": k, "Giá trị": v} for k, v in definition.describe().items()]
    ).to_csv(tables / "label_definition.csv", index=False, encoding="utf-8-sig")

    labeled = LB.assign_labels(labels, definition)
    dist = LB.label_distribution(labeled)
    dist.to_csv(tables / "label_distribution.csv", index=False, encoding="utf-8-sig")
    log.info("Phân bố nhãn:\n%s", dist.to_string(index=False))

    vintage = LB.vintage_analysis(labels, max_months=18)
    vintage.to_csv(tables / "vintage_analysis.csv", index=False, encoding="utf-8-sig")
    at_window = vintage[
        vintage["Tháng kể từ khi cấp tín dụng"] == definition.performance_window_months
    ]
    if not at_window.empty and "Tỷ lệ ghi nhận được (%)" in at_window.columns:
        log.info(
            "Cửa sổ %d tháng ghi nhận được %.1f%% tổng số trường hợp vỡ nợ",
            definition.performance_window_months,
            float(at_window["Tỷ lệ ghi nhận được (%)"].iloc[0]),
        )

    # ghi lại nhãn đã chuẩn hóa để các bước sau dùng chung
    labeled.to_parquet(raw / "labels.parquet", index=False, compression="snappy")

    # =====================================================================
    # 6. PHÂN CHIA TẬP DỮ LIỆU
    # =====================================================================
    log.info("=" * 72)
    log.info("6. Phân chia tập dữ liệu theo thời gian")
    merged = features.merge(
        labeled[["application_id", "default_flag", "is_modelable"]],
        on="application_id", how="inner",
    )
    modelable = merged[merged["is_modelable"]].copy()
    if len(modelable) < len(merged):
        log.info(
            "Loại %s hồ sơ chưa đủ cửa sổ quan sát khỏi tập mô hình hóa",
            f"{len(merged) - len(modelable):,}",
        )

    parts = SP.out_of_time_split(
        modelable,
        train_ratio=float(cfg["split.train_ratio"]),
        valid_ratio=float(cfg["split.valid_ratio"]),
    )
    report = SP.split_report(parts)
    report.to_csv(tables / "data_split.csv", index=False, encoding="utf-8-sig")

    # =====================================================================
    log.info("=" * 72)
    log.info("HOÀN THÀNH bước chuẩn bị dữ liệu sau %.1f giây",
             time.perf_counter() - t_start)
    log.info("  Số hồ sơ dùng được cho mô hình : %s", f"{len(modelable):,}")
    log.info("  Tỷ lệ nợ xấu                   : %.2f%%",
             100 * float(modelable["default_flag"].mean()))
    log.info("  Số đặc trưng                   : %d", features.shape[1] - 3)
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
