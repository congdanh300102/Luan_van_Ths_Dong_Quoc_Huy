"""Bước 2 - Phân tích năng lực dự báo và lựa chọn tập đặc trưng tối ưu.

Script thực hiện phần trích xuất và lựa chọn đặc trưng của đề án:

    (1) Thống kê tập đặc trưng theo năm nhóm nguồn dữ liệu
    (2) Đánh giá năng lực dự báo của từng đặc trưng bằng Information Value
    (3) Đo độ ổn định của từng đặc trưng theo thời gian bằng PSI
    (4) Chạy ba bộ lọc lựa chọn: năng lực dự báo, tương quan, độ ổn định
    (5) Ghi tập đặc trưng đã chọn để bước huấn luyện sử dụng

Kết quả (``results/tables``):
    feature_catalogue.csv           Danh mục đặc trưng theo nhóm và nguồn dữ liệu
    information_value.csv           IV của từng đặc trưng
    information_value_by_group.csv  IV tổng hợp theo nhóm nguồn dữ liệu
    feature_psi.csv                 PSI giữa tập huấn luyện và tập kiểm thử
    feature_selection_steps.csv     Số đặc trưng còn lại sau từng bộ lọc
    feature_correlation_pairs.csv   Các cặp đặc trưng bị loại do tương quan cao
    selected_features.json          Danh sách đặc trưng được giữ lại

Cách dùng:
    py scripts/02_build_features.py
    py scripts/02_build_features.py --no-selection   # chỉ phân tích, không lọc
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
from creditreview.features import definitions as fd
from creditreview.features import selection as FS
from creditreview.models import plots as P
from creditreview.models import train as T
from creditreview.monitoring import drift as D

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("features")
for noisy in ["matplotlib", "PIL"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phân tích và lựa chọn đặc trưng")
    ap.add_argument("--no-selection", action="store_true",
                    help="Chỉ phân tích IV/PSI, không ghi tập đặc trưng đã chọn")
    args = ap.parse_args()

    cfg = load_config()
    tables = cfg.path("tables")
    figures = cfg.path("figures")
    t_start = time.perf_counter()

    df = T.load_training_data(cfg)
    splits = T.make_splits(df, cfg)

    windows = list(cfg.get("features.txn_windows_days", [7, 30, 90, 180]))
    numeric = [c for c in fd.numeric_features(windows) if c in df.columns]
    categorical = [c for c in fd.categorical_features(windows) if c in df.columns]
    all_feats = numeric + categorical

    # =====================================================================
    # 1. DANH MỤC ĐẶC TRƯNG
    # =====================================================================
    log.info("=" * 72)
    log.info("1. Danh mục đặc trưng")
    catalogue = pd.DataFrame(
        [
            {
                "Tên đặc trưng": f.name,
                "Nhóm": f.group.value,
                "Nguồn dữ liệu": f.source.value,
                "Kiểu": f.dtype,
                "Mô tả": f.description,
            }
            for f in fd.all_features(windows)
            if f.name in df.columns
        ]
    )
    catalogue.to_csv(tables / "feature_catalogue.csv", index=False, encoding="utf-8-sig")
    by_group = (
        catalogue.groupby("Nhóm", observed=True)
        .size().reset_index(name="Số đặc trưng")
        .sort_values("Số đặc trưng", ascending=False)
    )
    log.info("Tập đặc trưng: %d biến\n%s", len(catalogue), by_group.to_string(index=False))

    # =====================================================================
    # 2. INFORMATION VALUE
    # =====================================================================
    log.info("=" * 72)
    log.info("2. Đánh giá năng lực dự báo bằng Information Value")
    iv = T.analyse_information_value(splits, numeric, categorical, cfg)
    iv["strength"] = iv["iv"].map(FS.iv_strength)
    iv.to_csv(tables / "information_value.csv", index=False, encoding="utf-8-sig")
    P.plot_iv_ranking(iv, figures, top_n=25)
    iv_group = pd.read_csv(tables / "information_value_by_group.csv")
    P.plot_iv_by_group(iv_group, figures)

    top = iv.nlargest(10, "iv")[["feature", "group", "iv", "strength"]]
    log.info("Mười đặc trưng có IV cao nhất:\n%s", top.to_string(index=False))

    # =====================================================================
    # 3. ĐỘ ỔN ĐỊNH THEO THỜI GIAN (PSI)
    # =====================================================================
    log.info("=" * 72)
    log.info("3. Đo độ ổn định của đặc trưng theo thời gian")
    psi = D.drift_report(
        splits["train"], splits["test"], numeric, categorical,
        n_bins=int(cfg.get("monitoring.psi_bins", 10)),
        warn=float(cfg.get("monitoring.psi_warn", 0.10)),
        alert=float(cfg.get("monitoring.psi_alert", 0.25)),
    )
    psi.to_csv(tables / "feature_psi.csv", index=False, encoding="utf-8-sig")
    log.info(
        "PSI: %d đặc trưng cảnh báo, %d đặc trưng báo động",
        int((psi["status"] == "CẢNH BÁO").sum()),
        int((psi["status"] == "BÁO ĐỘNG").sum()),
    )

    # =====================================================================
    # 4. LỰA CHỌN TẬP ĐẶC TRƯNG TỐI ƯU
    # =====================================================================
    log.info("=" * 72)
    log.info("4. Lựa chọn tập đặc trưng tối ưu")
    sel_cfg = cfg.get("features.selection", {}) or {}
    result = FS.select_features(
        splits["train"],
        all_feats,
        iv_table=iv,
        psi_table=psi,
        min_iv=float(sel_cfg.get("min_iv", FS.IV_USELESS)),
        max_iv=float(sel_cfg.get("max_iv", 1.0)),
        corr_threshold=float(sel_cfg.get("max_correlation", 0.95)),
        max_psi=float(sel_cfg.get("max_psi", 0.25)),
    )
    result.steps.to_csv(
        tables / "feature_selection_steps.csv", index=False, encoding="utf-8-sig"
    )
    if not result.correlation_pairs.empty:
        result.correlation_pairs.to_csv(
            tables / "feature_correlation_pairs.csv", index=False, encoding="utf-8-sig"
        )
    log.info("Quy trình lựa chọn:\n%s", result.steps.to_string(index=False))

    # cơ cấu tập đặc trưng sau lựa chọn theo nhóm
    kept = catalogue[catalogue["Tên đặc trưng"].isin(result.selected)]
    mix = (
        kept.groupby("Nhóm", observed=True).size().reset_index(name="Sau lựa chọn")
        .merge(by_group.rename(columns={"Số đặc trưng": "Ban đầu"}), on="Nhóm", how="right")
        .fillna(0)
    )
    mix["Sau lựa chọn"] = mix["Sau lựa chọn"].astype(int)
    mix["Tỷ lệ giữ lại (%)"] = (100 * mix["Sau lựa chọn"] / mix["Ban đầu"]).round(1)
    mix.to_csv(tables / "feature_selection_by_group.csv", index=False, encoding="utf-8-sig")
    log.info("Cơ cấu đặc trưng sau lựa chọn:\n%s", mix.to_string(index=False))

    if not args.no_selection and bool(sel_cfg.get("enabled", True)):
        payload = {
            "n_initial": len(all_feats),
            "n_selected": result.n_selected,
            "criteria": {
                "min_iv": float(sel_cfg.get("min_iv", FS.IV_USELESS)),
                "max_iv": float(sel_cfg.get("max_iv", 1.0)),
                "max_correlation": float(sel_cfg.get("max_correlation", 0.95)),
                "max_psi": float(sel_cfg.get("max_psi", 0.25)),
            },
            "selected": result.selected,
        }
        with open(tables / "selected_features.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        log.info("Đã ghi %d đặc trưng vào selected_features.json", result.n_selected)
    else:
        path = tables / "selected_features.json"
        if path.exists():
            path.unlink()
        log.info("Không áp dụng lựa chọn đặc trưng - mô hình sẽ dùng toàn bộ tập đặc trưng")

    log.info("=" * 72)
    log.info("HOÀN THÀNH sau %.1f giây", time.perf_counter() - t_start)
    log.info("  Tập đặc trưng: %d -> %d", len(all_feats), result.n_selected)
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
