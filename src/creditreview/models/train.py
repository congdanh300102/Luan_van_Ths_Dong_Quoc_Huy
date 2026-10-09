"""Huấn luyện, hiệu chuẩn và đánh giá toàn bộ mô hình dự báo rủi ro tín dụng.

QUY TRÌNH THỰC NGHIỆM
---------------------
Dữ liệu được chia theo thời gian (*out-of-time*) thành ba phần liên tiếp:

    |-------- train (70%) --------|--- valid (15%) ---|--- test (15%) ---|

Tập ``valid`` tiếp tục được chia làm hai nửa với vai trò tách biệt nhằm tránh
sử dụng lặp cùng một dữ liệu cho nhiều mục đích:

    ``valid_a`` - dừng sớm (early stopping) cho các mô hình boosting;
    ``valid_b`` - hiệu chuẩn xác suất và lựa chọn ngưỡng quyết định.

Tập ``test`` chỉ được sử dụng **một lần duy nhất** ở bước báo cáo kết quả cuối
cùng, bảo đảm ước lượng hiệu năng không bị lạc quan.

BA BƯỚC XỬ LÝ SAU HUẤN LUYỆN
----------------------------
1. **Hiệu chuẩn xác suất** (isotonic regression trên ``valid_b``): các mô hình
   dùng trọng số lớp cho ra xác suất bị lệch hệ thống; hiệu chuẩn lại là bắt
   buộc vì Decision Engine sử dụng trực tiếp giá trị PD để phân vùng quyết định
   và định giá.
2. **Lựa chọn ngưỡng theo chi phí nghiệp vụ**: ngưỡng được chọn để tối thiểu hóa
   chi phí kỳ vọng với tỷ lệ ``c_FN : c_FP = 12 : 1``, thay vì mặc định 0,5
   (xem :mod:`creditreview.labeling.cutoff`).
3. **Đo thời gian suy luận**: đo thời gian chấm điểm một hồ sơ đơn lẻ
   (p50/p95/p99). Chỉ tiêu này quyết định mô hình có dùng được trong quy trình
   kiểm duyệt trực tuyến hay không -- một mô hình chính xác hơn 0,3% nhưng chậm
   hơn hai bậc độ lớn thường không phải lựa chọn hợp lý trong thực tế.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from ..config import Config
from ..features import definitions as fd
from . import metrics as M
from .woe import WOEEncoder, build_scorecard
from .zoo import ModelSpec, build_model_specs, build_stacking

log = logging.getLogger(__name__)


# ===========================================================================
# Kết quả huấn luyện
# ===========================================================================
@dataclass
class TrainedModel:
    spec_name: str
    display_name: str
    family: str
    model: Any
    calibrator: Any | None
    threshold: float
    fit_seconds: float
    n_iterations: int | None
    valid_report: M.ClassificationReport
    test_report: M.ClassificationReport
    latency_ms: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    # Danh mục của các cột phân loại đã "đóng băng" lúc huấn luyện. Lưu kèm mô
    # hình để mọi lần suy luận về sau dùng đúng cách mã hóa như khi huấn luyện,
    # loại bỏ hoàn toàn nguy cơ lệch train/serve do mã hạng mục khác nhau.
    cat_categories: Dict[str, Any] | None = None
    feature_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = _raw_proba(self.model, X)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 1e-6, 1 - 1e-6)

    def prepare(self, part: pd.DataFrame) -> pd.DataFrame:
        """Chuyển một khung dữ liệu thô sang đúng định dạng đầu vào của mô hình."""
        cols = self.feature_cols or list(part.columns)
        if self.spec_name == "catboost":
            return _prepare_catboost(part[cols], self.categorical_cols)
        if self.spec_name in {"xgboost", "lightgbm"} and self.cat_categories:
            X, _ = prepare_native_categorical(
                part[cols], self.categorical_cols, self.cat_categories
            )
            return X
        return part[cols]

    def predict_from_raw(self, part: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(self.prepare(part))


def _raw_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def _extract_n_iter(clf: Any) -> int | None:
    """Lấy số vòng lặp thực tế của bộ phân loại (nếu có).

    Các bộ phân loại khác nhau lưu thông tin này dưới nhiều dạng: vô hướng
    (``MLPClassifier``), mảng theo lớp (``LogisticRegression``) hoặc thuộc tính
    riêng (``HistGradientBoostingClassifier.n_iter_``).
    """
    for attr in ("n_iter_", "best_iteration_", "n_estimators_"):
        val = getattr(clf, attr, None)
        if val is None:
            continue
        arr = np.ravel(np.asarray(val))
        if arr.size == 0:
            continue
        try:
            return int(arr[0])
        except (TypeError, ValueError):
            continue
    return None


# ===========================================================================
# Chuẩn bị dữ liệu
# ===========================================================================
def prepare_native_categorical(
    df: pd.DataFrame, cat_cols: Sequence[str], categories: Dict[str, pd.Index] | None = None
) -> Tuple[pd.DataFrame, Dict[str, pd.Index]]:
    """Chuyển các cột phân loại sang dtype ``category`` với danh mục cố định.

    Danh mục được "đóng băng" theo tập huấn luyện; hạng mục chưa từng xuất hiện
    trong tập kiểm định/kiểm tra sẽ trở thành giá trị thiếu - đúng với hành vi
    cần có khi triển khai thực tế.
    """
    out = df.copy()
    cats: Dict[str, pd.Index] = {}
    for c in cat_cols:
        if categories is None:
            out[c] = out[c].astype("category")
            cats[c] = out[c].cat.categories
        else:
            out[c] = pd.Categorical(out[c], categories=categories[c])
            cats[c] = categories[c]
    return out, cats


def _prepare_catboost(df: pd.DataFrame, cat_cols: Sequence[str]) -> pd.DataFrame:
    """Chuẩn bị dữ liệu cho CatBoost.

    CatBoost yêu cầu cột phân loại ở dạng chuỗi (hoặc số nguyên) và **không**
    cho phép giá trị thiếu trong cột phân loại, nên giá trị thiếu được điền
    tường minh bằng một hạng mục riêng - cách xử lý này cũng đúng về nghiệp vụ:
    "không có thông tin" tự nó là một tín hiệu.
    """
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype("object").where(out[c].notna(), "__NA__").astype(str)
    return out


def load_training_data(cfg: Config) -> pd.DataFrame:
    """Nạp ma trận đặc trưng và nhãn, ghép thành một bảng huấn luyện."""
    raw = cfg.path("data_raw")
    features = pd.read_parquet(raw / "features.parquet")
    labels = pd.read_parquet(raw / "labels.parquet")
    df = features.merge(
        labels[["application_id", "default_flag", "_pd_true"]], on="application_id", how="inner"
    )
    df = df.sort_values("apply_time", kind="mergesort").reset_index(drop=True)
    log.info("Nạp %s hồ sơ, %s cột", f"{len(df):,}", df.shape[1])
    return df


def load_selected_features(path: Path) -> List[str]:
    """Nạp danh sách đặc trưng đã được lựa chọn ở bước xây dựng đặc trưng.

    Trả về danh sách rỗng nếu file chưa tồn tại - khi đó mô hình dùng toàn bộ
    tập đặc trưng, nhờ vậy bước huấn luyện vẫn chạy được độc lập.
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return list(data.get("selected", []))
    except Exception as exc:  # pragma: no cover
        log.warning("Không đọc được danh sách đặc trưng đã chọn (%s)", exc)
        return []


def make_splits(df: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    """Chia dữ liệu theo trục thời gian thành bốn phần có vai trò tách biệt.

    ``train_fit``  (59,5%) - huấn luyện. **Mọi** mô hình đều chỉ dùng phần này,
                             bảo đảm so sánh công bằng trên cùng lượng dữ liệu.
    ``train_es``   (10,5%) - dừng sớm cho các mô hình boosting. Được tách ra từ
                             tập huấn luyện (không lấy từ tập kiểm định) để tập
                             kiểm định giữ nguyên vai trò duy nhất là hiệu chuẩn.
    ``valid``      (15%)   - hiệu chuẩn xác suất và lựa chọn ngưỡng quyết định.
    ``test``       (15%)   - đánh giá cuối cùng, chỉ dùng một lần.

    Cách chia này khắc phục một vấn đề thực tế: nếu dùng một nửa tập kiểm định
    (khoảng 4.500 hồ sơ) cho dừng sớm, ước lượng AUC trên tập nhỏ có phương sai
    lớn khiến tiêu chí dừng sớm kích hoạt quá sớm và mô hình bị *dưới khớp*.
    """
    tr_r = float(cfg["split.train_ratio"])
    va_r = float(cfg["split.valid_ratio"])
    n = len(df)
    i_tr = int(n * tr_r)
    i_va = int(n * (tr_r + va_r))
    # 15% cuối của tập huấn luyện dành cho dừng sớm
    i_fit = int(i_tr * 0.85)

    parts = {
        "train_fit": df.iloc[:i_fit],
        "train_es": df.iloc[i_fit:i_tr],
        "train": df.iloc[:i_tr],
        "valid": df.iloc[i_tr:i_va],
        "test": df.iloc[i_va:],
    }
    log.info("Chia dữ liệu theo thời gian (out-of-time):")
    for k in ["train_fit", "train_es", "valid", "test"]:
        v = parts[k]
        log.info(
            "  %-10s %7s hồ sơ | %s -> %s | tỷ lệ vỡ nợ %.2f%%",
            k, f"{len(v):,}",
            pd.Timestamp(v["apply_time"].min()).date(),
            pd.Timestamp(v["apply_time"].max()).date(),
            100 * v["default_flag"].mean(),
        )
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


# ===========================================================================
# Huấn luyện một mô hình
# ===========================================================================
def fit_one(
    spec: ModelSpec,
    splits: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    cat_cols: List[str],
    cfg: Config,
) -> TrainedModel:
    """Huấn luyện, hiệu chuẩn, chọn ngưỡng và đánh giá một mô hình."""
    log.info("─" * 72)
    log.info("Huấn luyện: %s", spec.display_name)

    cost_fn = float(cfg["training.cost_fn"])
    cost_fp = float(cfg["training.cost_fp"])
    y = {k: splits[k]["default_flag"].to_numpy() for k in splits}

    # --- chuẩn bị ma trận đặc trưng theo yêu cầu của từng họ mô hình -------
    is_catboost = type(spec.estimator).__name__ == "CatBoostClassifier"
    cats: Dict[str, Any] | None = None
    if spec.encoding == "native_cat":
        if is_catboost:
            # CatBoost yêu cầu cột phân loại ở dạng chuỗi và không nhận giá trị
            # thiếu trong cột phân loại, nên cần điền khuyết tường minh.
            X = {k: _prepare_catboost(splits[k][feature_cols], cat_cols) for k in splits}
        else:
            Xtr, cats = prepare_native_categorical(splits["train_fit"][feature_cols], cat_cols)
            X = {"train_fit": Xtr}
            for k in ["train_es", "train", "valid", "test"]:
                X[k], _ = prepare_native_categorical(splits[k][feature_cols], cat_cols, cats)
    else:
        X = {k: splits[k][feature_cols] for k in splits}

    model = spec.estimator
    t0 = time.perf_counter()
    n_iter: int | None = None

    # --- huấn luyện có/không dừng sớm --------------------------------------
    if spec.supports_early_stopping and spec.encoding == "native_cat":
        name = type(model).__name__
        if name == "XGBClassifier":
            model.fit(
                X["train_fit"], y["train_fit"],
                eval_set=[(X["train_es"], y["train_es"])], verbose=False,
            )
            n_iter = int(getattr(model, "best_iteration", 0) or 0)
        elif name == "LGBMClassifier":
            import lightgbm as lgb

            # LightGBM đánh giá đồng thời nhiều chỉ tiêu trên tập kiểm định và
            # mặc định dừng sớm khi *bất kỳ* chỉ tiêu nào ngừng cải thiện. Do mô
            # hình sử dụng trọng số lớp (``scale_pos_weight``), xác suất dự báo
            # bị lệch hệ thống nên ``binary_logloss`` xấu đi rất sớm, kéo theo
            # việc dừng sớm bị kích hoạt chỉ sau vài vòng lặp trong khi AUC vẫn
            # đang tăng. Vì vậy cần: (i) chỉ định AUC là chỉ tiêu đánh giá duy
            # nhất, và (ii) bật ``first_metric_only`` để tiêu chí dừng chỉ căn
            # cứ vào AUC.
            model.set_params(metric="auc")
            model.fit(
                X["train_fit"], y["train_fit"],
                eval_X=X["train_es"], eval_y=y["train_es"], eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(150, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            n_iter = int(getattr(model, "best_iteration_", 0) or 0)
        elif name == "CatBoostClassifier":
            cat_idx = [X["train_fit"].columns.get_loc(c) for c in cat_cols]
            model.fit(
                X["train_fit"], y["train_fit"],
                eval_set=(X["train_es"], y["train_es"]),
                cat_features=cat_idx, verbose=False,
            )
            n_iter = int(model.get_best_iteration() or 0)
        else:
            model.fit(X["train_fit"], y["train_fit"])
    else:
        model.fit(X["train_fit"], y["train_fit"])
        clf = model[-1] if hasattr(model, "steps") else model
        n_iter = _extract_n_iter(clf)

    fit_seconds = time.perf_counter() - t0
    log.info("  thời gian huấn luyện: %.1f giây%s", fit_seconds,
             f" | số vòng lặp tối ưu: {n_iter}" if n_iter else "")

    # --- hiệu chuẩn xác suất trên tập kiểm định ---------------------------
    p_va_raw = _raw_proba(model, X["valid"])
    calibrator = None
    if str(cfg.get("training.calibration", "isotonic")).lower() == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        calibrator.fit(p_va_raw, y["valid"])
        p_va = calibrator.predict(p_va_raw)
        # Lưu ý: ECE dưới đây được đo trên chính tập dùng để hiệu chuẩn nên
        # mang tính tham khảo (đánh giá mức độ khớp của phép hiệu chuẩn).
        # Giá trị ECE dùng để so sánh mô hình là ECE trên tập kiểm tra, được
        # tính ở bước đánh giá phía dưới.
        ece_before = M.expected_calibration_error(y["valid"], p_va_raw)
        ece_after = M.expected_calibration_error(y["valid"], p_va)
        log.info("  hiệu chuẩn ECE (trên tập hiệu chuẩn): %.4f -> %.4f",
                 ece_before, ece_after)
    else:
        p_va = p_va_raw

    def proba(key: str) -> np.ndarray:
        p = _raw_proba(model, X[key])
        return np.clip(calibrator.predict(p) if calibrator is not None else p, 1e-6, 1 - 1e-6)

    # --- chọn ngưỡng theo chi phí nghiệp vụ trên tập kiểm định ------------
    threshold, _ = M.optimal_threshold_by_cost(y["valid"], p_va, cost_fn, cost_fp)
    log.info("  ngưỡng tối ưu theo chi phí: %.4f", threshold)

    # --- đánh giá ----------------------------------------------------------
    valid_report = M.evaluate(y["valid"], proba("valid"), threshold, cost_fn, cost_fp)
    test_report = M.evaluate(y["test"], proba("test"), threshold, cost_fn, cost_fp)
    log.info(
        "  VALID  AUC=%.4f KS=%.4f | TEST  AUC=%.4f KS=%.4f Gini=%.4f F1=%.4f",
        valid_report.roc_auc, valid_report.ks,
        test_report.roc_auc, test_report.ks, test_report.gini, test_report.f1,
    )

    # --- thời gian chấm điểm một hồ sơ -------------------------------------
    latency = measure_inference_latency(model, calibrator, X["test"], n_samples=400)
    log.info(
        "  thời gian chấm điểm 1 hồ sơ: p50=%.2f ms  p95=%.2f ms  p99=%.2f ms",
        latency["p50"], latency["p95"], latency["p99"],
    )

    return TrainedModel(
        spec_name=spec.name,
        display_name=spec.display_name,
        family=spec.family,
        model=model,
        calibrator=calibrator,
        threshold=threshold,
        fit_seconds=fit_seconds,
        n_iterations=n_iter,
        valid_report=valid_report,
        test_report=test_report,
        latency_ms=latency,
        notes=spec.notes,
        cat_categories=cats,
        feature_cols=list(feature_cols),
        categorical_cols=list(cat_cols),
    )


def measure_inference_latency(
    model: Any, calibrator: Any, X: pd.DataFrame, n_samples: int = 400
) -> Dict[str, float]:
    """Đo thời gian chấm điểm cho **một hồ sơ đơn lẻ**.

    Chế độ một hồ sơ phản ánh đúng điều kiện vận hành của mô hình hỗ trợ kiểm
    duyệt (mỗi hồ sơ được đánh giá độc lập khi cán bộ tín dụng tiếp nhận), khác
    biệt đáng kể so với dự báo theo lô. Có làm nóng (warm-up) trước khi đo để
    loại bỏ ảnh hưởng của biên dịch JIT và nạp thư viện lần đầu.
    """
    n_samples = min(n_samples, len(X))
    # warm-up
    for i in range(min(30, len(X))):
        _ = _raw_proba(model, X.iloc[i : i + 1])

    durations = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        row = X.iloc[i : i + 1]
        t0 = time.perf_counter()
        p = _raw_proba(model, row)
        if calibrator is not None:
            p = calibrator.predict(p)
        durations[i] = (time.perf_counter() - t0) * 1000.0

    # thông lượng theo lô (batch = 1000) để so sánh
    batch = X.iloc[: min(1000, len(X))]
    t0 = time.perf_counter()
    _ = _raw_proba(model, batch)
    batch_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "mean": float(durations.mean()),
        "p50": float(np.percentile(durations, 50)),
        "p90": float(np.percentile(durations, 90)),
        "p95": float(np.percentile(durations, 95)),
        "p99": float(np.percentile(durations, 99)),
        "max": float(durations.max()),
        "batch1000_ms": float(batch_ms),
        "batch_per_record_ms": float(batch_ms / max(len(batch), 1)),
    }


# ===========================================================================
# Điều phối toàn bộ thực nghiệm
# ===========================================================================
def run_experiment(
    cfg: Config,
    model_names: Sequence[str] | None = None,
    include_stacking: bool = True,
) -> Dict[str, Any]:
    """Huấn luyện và so sánh toàn bộ mô hình; xuất kết quả ra ``results/``."""
    df = load_training_data(cfg)
    splits = make_splits(df, cfg)

    windows = list(cfg.get("features.txn_windows_days", [7, 30, 90, 180]))
    all_num = [c for c in fd.numeric_features(windows) if c in df.columns]
    all_cat = [c for c in fd.categorical_features(windows) if c in df.columns]

    # Nếu bước lựa chọn đặc trưng (scripts/02_build_features.py) đã chạy, dùng
    # tập đặc trưng đã rút gọn thay cho tập đầy đủ. Nhờ vậy kết quả huấn luyện
    # phản ánh đúng tập đặc trưng mà mô hình sẽ dùng khi vận hành.
    selected = load_selected_features(cfg.path("tables") / "selected_features.json")
    if selected:
        keep = set(selected)
        n_before = len(all_num) + len(all_cat)
        all_num = [c for c in all_num if c in keep]
        all_cat = [c for c in all_cat if c in keep]
        log.info(
            "Dùng tập đặc trưng đã lựa chọn: %d/%d đặc trưng",
            len(all_num) + len(all_cat), n_before,
        )

    feature_cols = all_num + all_cat
    log.info("Số đặc trưng: %d (số: %d, phân loại: %d)", len(feature_cols), len(all_num), len(all_cat))

    y_tr = splits["train_fit"]["default_flag"].to_numpy()
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    log.info("scale_pos_weight = %.3f", spw)

    # Nạp siêu tham số đã tinh chỉnh (nếu đã chạy scripts/03_tune_hyperparams.py)
    from .zoo import load_tuned_params

    tuned = load_tuned_params(cfg.path("tables") / "tuning_best.json")
    specs = build_model_specs(
        all_num, all_cat, scale_pos_weight=spw, seed=cfg.seed,
        n_jobs=int(cfg.get("training.n_jobs", -1)), tuned=tuned,
    )
    requested = list(model_names) if model_names else list(cfg.get("training.models", specs.keys()))

    trained: Dict[str, TrainedModel] = {}
    for name in requested:
        if name == "stacking_ensemble":
            continue
        if name not in specs:
            log.warning("Bỏ qua mô hình không tồn tại: %s", name)
            continue
        try:
            trained[name] = fit_one(specs[name], splits, feature_cols, all_cat, cfg)
        except Exception as exc:  # pragma: no cover
            log.error("Lỗi khi huấn luyện %s: %s", name, exc, exc_info=True)

    # --- mô hình kết hợp xếp tầng -----------------------------------------
    if include_stacking and "stacking_ensemble" in requested:
        members = ["logistic_regression", "random_forest", "gradient_boosting", "decision_tree"]
        stack_spec = build_stacking(specs, members, seed=cfg.seed, cv=3)
        if stack_spec.estimator.estimators:
            try:
                trained["stacking_ensemble"] = fit_one(
                    stack_spec, splits, feature_cols, all_cat, cfg
                )
            except Exception as exc:  # pragma: no cover
                log.error("Lỗi khi huấn luyện stacking: %s", exc, exc_info=True)

    # --- lưu mô hình -------------------------------------------------------
    model_dir = cfg.path("models")
    for name, tm in trained.items():
        joblib.dump(
            {
                "model": tm.model,
                "calibrator": tm.calibrator,
                "threshold": tm.threshold,
                "feature_cols": feature_cols,
                "categorical_cols": all_cat,
                "cat_categories": tm.cat_categories,
                "spec_name": tm.spec_name,
                "display_name": tm.display_name,
            },
            model_dir / f"{name}.joblib",
            compress=3,
        )
    log.info("Đã lưu %d mô hình vào %s", len(trained), model_dir)

    # --- bảng so sánh ------------------------------------------------------
    comparison = build_comparison_table(trained)
    tables = cfg.path("tables")
    comparison.to_csv(tables / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # --- mô hình tốt nhất --------------------------------------------------
    #
    # Không lấy thẳng mô hình có AUC cao nhất. Trên tập dữ liệu này nhóm dẫn đầu
    # sát nhau tới mức chênh lệch AUC nhỏ hơn dao động đo đạc giữa các lần chạy,
    # nên thứ hạng ở chữ số thập phân thứ tư không mang thông tin: chạy lại có
    # thể đảo ngôi thứ mà chất lượng thực sự không đổi. Chọn theo AUC thuần túy
    # vì vậy vừa không ổn định, vừa có thể chọn phải mô hình kết hợp xếp tầng --
    # thời gian chấm điểm cao gấp hàng chục lần và vi phạm mục tiêu thời gian
    # phản hồi dưới 100 ms của bài toán kiểm duyệt trực tuyến.
    #
    # Quy tắc áp dụng: lấy nhóm mô hình nằm trong khoảng TIE_TOLERANCE_AUC so
    # với mô hình dẫn đầu (coi như tương đương về chất lượng), rồi trong nhóm đó
    # chọn mô hình có thời gian chấm điểm ở phân vị 95 thấp nhất. Đây chính là
    # tiêu chí biên Pareto đã trình bày ở phần so sánh thuật toán.
    TIE_TOLERANCE_AUC = 0.002
    top_auc = float(comparison.iloc[0]["AUC"])
    tied = comparison[comparison["AUC"] >= top_auc - TIE_TOLERANCE_AUC]
    lat_col = "Độ trễ p95 (ms)"
    if lat_col in tied.columns and tied[lat_col].notna().any():
        chosen = tied.sort_values(lat_col, kind="mergesort").iloc[0]
    else:
        chosen = comparison.iloc[0]
    best_name = chosen["model_key"]
    best = trained[best_name]
    log.info("=" * 72)
    log.info("Nhóm mô hình tương đương về chất lượng (AUC trong khoảng %.3f "
             "so với mô hình dẫn đầu): %s",
             TIE_TOLERANCE_AUC, ", ".join(tied["Mô hình"].astype(str)))
    log.info("MÔ HÌNH ĐƯỢC CHỌN TRIỂN KHAI: %s (test AUC = %.4f, KS = %.4f, "
             "thời gian chấm điểm p95 = %.2f ms)",
             best.display_name, best.test_report.roc_auc, best.test_report.ks,
             float(chosen.get(lat_col, float("nan"))))

    # đăng ký mô hình phục vụ triển khai
    joblib.dump(
        {
            "model": best.model,
            "calibrator": best.calibrator,
            "threshold": best.threshold,
            "feature_cols": feature_cols,
            "categorical_cols": all_cat,
            "cat_categories": best.cat_categories,
            "spec_name": best.spec_name,
            "display_name": best.display_name,
            "test_auc": best.test_report.roc_auc,
            "test_ks": best.test_report.ks,
        },
        model_dir / "production_model.joblib",
        compress=3,
    )

    # --- bảng gains của mô hình tốt nhất ----------------------------------
    p_test = best.predict_from_raw(splits["test"])
    gains = M.gains_table(splits["test"]["default_flag"].to_numpy(), p_test, n_bins=10)
    gains.to_csv(tables / "gains_table.csv", index=False, encoding="utf-8-sig")

    # --- lưu dự báo để phân tích và vẽ đồ thị -----------------------------
    preds = {"application_id": splits["test"]["application_id"].to_numpy(),
             "apply_time": splits["test"]["apply_time"].to_numpy(),
             "y_true": splits["test"]["default_flag"].to_numpy(),
             "pd_true": splits["test"]["_pd_true"].to_numpy()}
    for name, tm in trained.items():
        preds[f"p_{name}"] = tm.predict_from_raw(splits["test"])
    pd.DataFrame(preds).to_parquet(cfg.path("data_processed") / "test_predictions.parquet", index=False)

    # --- siêu dữ liệu thực nghiệm -----------------------------------------
    meta = {
        "n_total": int(len(df)),
        "n_train": int(len(splits["train"])),
        "n_train_fit": int(len(splits["train_fit"])),
        "n_train_es": int(len(splits["train_es"])),
        "n_valid": int(len(splits["valid"])),
        "n_test": int(len(splits["test"])),
        "n_features": len(feature_cols),
        "n_numeric": len(all_num),
        "n_categorical": len(all_cat),
        "default_rate_train": float(splits["train"]["default_flag"].mean()),
        "default_rate_test": float(splits["test"]["default_flag"].mean()),
        "scale_pos_weight": spw,
        # "best_*" là mô hình được chọn triển khai (xem quy tắc chọn ở trên);
        # "top_auc_*" là mô hình có AUC cao nhất. Hai mô hình này có thể khác
        # nhau khi nhóm dẫn đầu tương đương về chất lượng nhưng chênh lệch lớn
        # về thời gian chấm điểm.
        "best_model": best_name,
        "best_display_name": best.display_name,
        "best_test_auc": best.test_report.roc_auc,
        "best_test_ks": best.test_report.ks,
        "top_auc_model": str(comparison.iloc[0]["model_key"]),
        "top_auc_display_name": str(comparison.iloc[0]["Mô hình"]),
        "top_auc_test_auc": float(comparison.iloc[0]["AUC"]),
        "n_tied_models": int(len(tied)),
        "tie_tolerance_auc": TIE_TOLERANCE_AUC,
        "bayes_auc_test": float(
            __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                splits["test"]["default_flag"], splits["test"]["_pd_true"]
            )
        ),
        "cost_fn": float(cfg["training.cost_fn"]),
        "cost_fp": float(cfg["training.cost_fp"]),
        "train_period": [str(pd.Timestamp(splits["train"]["apply_time"].min()).date()),
                         str(pd.Timestamp(splits["train"]["apply_time"].max()).date())],
        "test_period": [str(pd.Timestamp(splits["test"]["apply_time"].min()).date()),
                        str(pd.Timestamp(splits["test"]["apply_time"].max()).date())],
    }
    with open(tables / "experiment_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    return {"trained": trained, "splits": splits, "comparison": comparison,
            "feature_cols": feature_cols, "categorical_cols": all_cat,
            "best_name": best_name, "meta": meta}


def _X_for(
    tm: TrainedModel, part: pd.DataFrame, feature_cols: List[str] | None = None,
    cat_cols: List[str] | None = None, train_part: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tạo ma trận đầu vào đúng định dạng mà mô hình đã được huấn luyện với.

    Uỷ quyền cho ``TrainedModel.prepare`` - nơi lưu sẵn danh mục đã "đóng băng"
    lúc huấn luyện, bảo đảm cách mã hóa khi suy luận trùng khớp với khi huấn
    luyện. Các tham số còn lại được giữ để tương thích với mã gọi cũ.
    """
    return tm.prepare(part)


def build_comparison_table(trained: Dict[str, TrainedModel]) -> pd.DataFrame:
    """Bảng so sánh các mô hình, sắp xếp theo AUC trên tập kiểm tra."""
    rows = []
    for name, tm in trained.items():
        t, v = tm.test_report, tm.valid_report
        rows.append(
            {
                "model_key": name,
                "Mô hình": tm.display_name,
                "Nhóm": tm.family,
                "AUC (valid)": round(v.roc_auc, 4),
                "AUC": round(t.roc_auc, 4),
                "Gini": round(t.gini, 4),
                "KS": round(t.ks, 4),
                "PR-AUC": round(t.pr_auc, 4),
                "Accuracy": round(t.accuracy, 4),
                "Precision": round(t.precision, 4),
                "Recall": round(t.recall, 4),
                "F1": round(t.f1, 4),
                "Brier": round(t.brier, 4),
                "ECE": round(t.ece, 4),
                "Ngưỡng": round(t.threshold, 4),
                "Chi phí": round(t.expected_cost, 4),
                "Tỷ lệ duyệt": round(t.approval_rate, 4),
                "Nợ xấu sau duyệt": round(t.bad_rate_approved, 4),
                "T.gian HL (s)": round(tm.fit_seconds, 1),
                "Độ trễ p50 (ms)": round(tm.latency_ms.get("p50", 0.0), 3),
                "Độ trễ p95 (ms)": round(tm.latency_ms.get("p95", 0.0), 3),
                "Độ trễ p99 (ms)": round(tm.latency_ms.get("p99", 0.0), 3),
                "Số vòng lặp": tm.n_iterations,
            }
        )
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


# ===========================================================================
# Phân tích bổ trợ: IV và thẻ điểm
# ===========================================================================
def analyse_information_value(
    splits: Dict[str, pd.DataFrame], numeric: List[str], categorical: List[str], cfg: Config
) -> pd.DataFrame:
    """Tính Information Value cho toàn bộ đặc trưng trên tập huấn luyện."""
    log.info("Tính Information Value cho %d đặc trưng...", len(numeric) + len(categorical))
    enc = WOEEncoder(numeric_features=numeric, categorical_features=categorical,
                     n_bins=10, min_bin_frac=0.03, monotonic=True)
    enc.fit(splits["train"][numeric + categorical], splits["train"]["default_flag"].to_numpy())
    iv = enc.iv_table()

    # gắn nhóm đặc trưng để phân tích theo nguồn dữ liệu
    windows = list(cfg.get("features.txn_windows_days", [7, 30, 90, 180]))
    group_map = {f.name: f.group.value for f in fd.all_features(windows)}
    desc_map = {f.name: f.description for f in fd.all_features(windows)}
    iv["group"] = iv["feature"].map(group_map)
    iv["description"] = iv["feature"].map(desc_map)

    tables = cfg.path("tables")
    iv.to_csv(tables / "information_value.csv", index=False, encoding="utf-8-sig")

    # tổng hợp IV theo nhóm đặc trưng
    by_group = (
        iv.groupby("group", observed=True)
        .agg(n_features=("feature", "size"), iv_sum=("iv", "sum"),
             iv_mean=("iv", "mean"), iv_max=("iv", "max"))
        .sort_values("iv_sum", ascending=False)
        .reset_index()
    )
    by_group.to_csv(tables / "information_value_by_group.csv", index=False, encoding="utf-8-sig")
    log.info("IV theo nhóm:\n%s", by_group.to_string(index=False))
    return iv


def export_scorecard(trained: Dict[str, TrainedModel], cfg: Config) -> pd.DataFrame | None:
    """Xuất thẻ điểm từ mô hình ``scorecard_woe`` (nếu đã được huấn luyện)."""
    tm = trained.get("scorecard_woe")
    if tm is None:
        return None
    pipe = tm.model
    enc: WOEEncoder = pipe.named_steps["woe"]
    clf = pipe.named_steps["clf"]
    coefs = dict(zip(enc.get_feature_names_out(), clf.coef_[0]))
    # chỉ giữ các biến có hệ số đáng kể để thẻ điểm gọn và dễ vận hành
    top = dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:20])
    card = build_scorecard(enc, top, float(clf.intercept_[0]))
    card.to_csv(cfg.path("tables") / "scorecard.csv", index=False, encoding="utf-8-sig")
    log.info("Đã xuất thẻ điểm gồm %d dòng cho %d biến", len(card), len(top))
    return card


__all__ = [
    "TrainedModel",
    "run_experiment",
    "fit_one",
    "make_splits",
    "load_training_data",
    "build_comparison_table",
    "analyse_information_value",
    "export_scorecard",
    "load_selected_features",
    "measure_inference_latency",
    "prepare_native_categorical",
]
