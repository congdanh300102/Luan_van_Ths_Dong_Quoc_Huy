"""Bước 3 (tùy chọn) - Tinh chỉnh siêu tham số cho các mô hình chính.

Sử dụng tìm kiếm ngẫu nhiên kết hợp kiểm định chéo theo chuỗi thời gian
(``TimeSeriesSplit``) trên tập huấn luyện. Kết quả được ghi ra
``results/tables/tuning_<mô hình>.csv`` và ``results/tables/tuning_best.json``
để bước huấn luyện chính sử dụng lại.

Cách dùng:
    py scripts/03_tune_hyperparams.py
    py scripts/03_tune_hyperparams.py --models lightgbm --trials 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

from creditreview.config import load_config
from creditreview.features import definitions as fd
from creditreview.models import train as T
from creditreview.models import tuning as TU

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tune")


def main() -> int:
    ap = argparse.ArgumentParser(description="Tinh chỉnh siêu tham số")
    ap.add_argument("--models", nargs="*", default=["lightgbm", "xgboost", "random_forest"])
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    tables = cfg.path("tables")

    df = T.load_training_data(cfg)
    splits = T.make_splits(df, cfg)
    train = splits["train"]  # dùng toàn bộ tập huấn luyện cho CV

    windows = list(cfg.get("features.txn_windows_days", [7, 30, 90, 180]))
    num = [c for c in fd.numeric_features(windows) if c in df.columns]
    cat = [c for c in fd.categorical_features(windows) if c in df.columns]

    # Dùng đúng tập đặc trưng mà bước huấn luyện sẽ dùng: nếu tinh chỉnh trên
    # tập đầy đủ rồi áp dụng tham số cho tập đã rút gọn, siêu tham số tìm được
    # không còn tối ưu cho mô hình thực tế.
    selected = T.load_selected_features(tables / "selected_features.json")
    if selected:
        keep = set(selected)
        num = [c for c in num if c in keep]
        cat = [c for c in cat if c in keep]
        log.info("Dùng tập đặc trưng đã lựa chọn: %d đặc trưng", len(num) + len(cat))

    feature_cols = num + cat

    y = train["default_flag"].to_numpy()
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    log.info("Tập tinh chỉnh: %s hồ sơ, %d đặc trưng, scale_pos_weight = %.3f",
             f"{len(train):,}", len(feature_cols), spw)

    best_all = {}
    for name in args.models:
        if name not in TU.SPACES:
            log.warning("Bỏ qua mô hình chưa có không gian tham số: %s", name)
            continue
        log.info("=" * 72)
        # các mô hình boosting dùng đặc trưng phân loại gốc
        if name in ("lightgbm", "xgboost"):
            X, _ = T.prepare_native_categorical(train[feature_cols], cat)
        else:
            X = train[feature_cols]

        res = TU.tune_model(
            name, X, y, cat, scale_pos_weight=spw,
            n_trials=args.trials, n_splits=args.splits, seed=cfg.seed,
            n_jobs=int(cfg.get("training.n_jobs", -1)),
        )
        res.table().to_csv(tables / f"tuning_{name}.csv", index=False, encoding="utf-8-sig")
        best_all[name] = {
            "best_params": res.best_params,
            "cv_auc": round(res.best_auc, 5),
            "n_trials": len(res.trials),
            "seconds": round(res.total_seconds, 1),
        }
        log.info("Đã lưu %d lần thử vào tuning_%s.csv (%.1f phút)",
                 len(res.trials), name, res.total_seconds / 60)

    with open(tables / "tuning_best.json", "w", encoding="utf-8") as fh:
        json.dump(best_all, fh, indent=2, ensure_ascii=False)

    log.info("=" * 72)
    log.info("HOÀN THÀNH tinh chỉnh siêu tham số")
    for name, info in best_all.items():
        log.info("  %-16s AUC(CV) = %.4f  %s", name, info["cv_auc"], info["best_params"])
    log.info("Kết quả đã ghi vào %s", tables / "tuning_best.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
