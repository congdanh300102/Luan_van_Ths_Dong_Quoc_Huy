"""Tinh chỉnh siêu tham số bằng kiểm định chéo theo chuỗi thời gian.

VÌ SAO KHÔNG DÙNG K-FOLD THÔNG THƯỜNG
------------------------------------
Với dữ liệu tín dụng, k-fold ngẫu nhiên tạo ra hai vấn đề nghiêm trọng:

1. **Rò rỉ thông tin thời gian.** Một fold huấn luyện có thể chứa hồ sơ *sau* hồ
   sơ trong fold kiểm định. Mô hình khi đó gián tiếp "biết trước tương lai" -
   điều không xảy ra trong vận hành thực tế, khiến ước lượng hiệu năng lạc quan.

2. **Rò rỉ theo khách hàng.** Một khách hàng có thể nộp nhiều hồ sơ. Nếu các hồ
   sơ này bị chia vào các fold khác nhau, mô hình đã thấy khách hàng đó khi huấn
   luyện và gặp lại khi kiểm định.

Vì vậy module sử dụng ``TimeSeriesSplit`` - chia dữ liệu thành các đoạn liên tiếp
theo thời gian, mỗi lần huấn luyện trên quá khứ và kiểm định trên tương lai gần:

    fold 1:  [--- train ---][valid]
    fold 2:  [------ train ------][valid]
    fold 3:  [--------- train ---------][valid]

Cách chia này mô phỏng đúng điều kiện triển khai và cho ước lượng hiệu năng
không lạc quan. Đổi lại, các fold đầu có ít dữ liệu huấn luyện hơn, nên điểm số
trung bình thường thấp hơn so với k-fold - đây là đặc điểm cần lưu ý khi so sánh
với các nghiên cứu dùng k-fold ngẫu nhiên.

TÌM KIẾM NGẪU NHIÊN THAY VÌ TÌM KIẾM TOÀN BỘ
-------------------------------------------
Không gian siêu tham số của các mô hình boosting có 5-7 chiều. Tìm kiếm lưới đầy
đủ cần hàng nghìn lần huấn luyện. Tìm kiếm ngẫu nhiên (*random search*) đạt chất
lượng tương đương với số lần thử ít hơn nhiều bậc, bởi trong thực tế chỉ một vài
siêu tham số thực sự quan trọng và tìm kiếm ngẫu nhiên khám phá từng chiều với
nhiều giá trị khác nhau hơn (Bergstra & Bengio, 2012).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

log = logging.getLogger(__name__)


@dataclass
class TuningTrial:
    trial: int
    params: Dict[str, Any]
    cv_auc_mean: float
    cv_auc_std: float
    fold_aucs: List[float] = field(default_factory=list)
    fit_seconds: float = 0.0
    mean_iterations: float = 0.0


@dataclass
class TuningResult:
    model_name: str
    best_params: Dict[str, Any]
    best_auc: float
    trials: List[TuningTrial]
    total_seconds: float

    def table(self) -> pd.DataFrame:
        """Bảng kết quả tinh chỉnh, xếp theo AUC giảm dần."""
        rows = []
        for t in sorted(self.trials, key=lambda x: -x.cv_auc_mean):
            row = {"Lần thử": t.trial, "AUC (CV)": round(t.cv_auc_mean, 4),
                   "Độ lệch chuẩn": round(t.cv_auc_std, 4),
                   "Số vòng lặp TB": round(t.mean_iterations, 0) or None,
                   "Thời gian (s)": round(t.fit_seconds, 1)}
            row.update({k: v for k, v in t.params.items()})
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Không gian siêu tham số
# ---------------------------------------------------------------------------
def lightgbm_space(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "num_leaves": int(rng.choice([15, 23, 31, 47, 63, 95])),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        "min_child_samples": int(rng.choice([20, 40, 80, 150, 250])),
        "subsample": float(rng.uniform(0.65, 1.0)),
        "colsample_bytree": float(rng.uniform(0.4, 0.9)),
        "reg_lambda": float(rng.choice([0.0, 1.0, 5.0, 20.0, 60.0])),
        "reg_alpha": float(rng.choice([0.0, 0.5, 2.0, 10.0])),
        "max_depth": int(rng.choice([-1, 5, 7, 10])),
    }


def xgboost_space(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "max_depth": int(rng.choice([3, 4, 5, 6, 8])),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        "min_child_weight": float(rng.choice([1, 5, 12, 25, 50])),
        "subsample": float(rng.uniform(0.65, 1.0)),
        "colsample_bytree": float(rng.uniform(0.4, 0.9)),
        "reg_lambda": float(rng.choice([1.0, 5.0, 20.0, 60.0])),
        "gamma": float(rng.choice([0.0, 0.1, 0.5, 2.0])),
    }


def random_forest_space(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "n_estimators": int(rng.choice([200, 300, 500])),
        "max_depth": int(rng.choice([8, 12, 16, 22, 30])),
        "min_samples_leaf": int(rng.choice([5, 10, 25, 50, 100])),
        "max_features": float(rng.choice([0.1, 0.2, 0.35, 0.5])),
    }


SPACES: Dict[str, Callable[[np.random.Generator], Dict[str, Any]]] = {
    "lightgbm": lightgbm_space,
    "xgboost": xgboost_space,
    "random_forest": random_forest_space,
}


# ---------------------------------------------------------------------------
# Tinh chỉnh
# ---------------------------------------------------------------------------
def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: Sequence[str],
    scale_pos_weight: float,
    n_trials: int = 15,
    n_splits: int = 3,
    seed: int = 42,
    n_jobs: int = -1,
) -> TuningResult:
    """Tìm siêu tham số tốt nhất cho một mô hình bằng tìm kiếm ngẫu nhiên + CV.

    ``X`` phải được sắp xếp theo thời gian tăng dần, vì ``TimeSeriesSplit`` chia
    theo thứ tự chỉ số.
    """
    if model_name not in SPACES:
        raise ValueError(f"Chưa định nghĩa không gian tham số cho: {model_name}")

    rng = np.random.default_rng(seed)
    space = SPACES[model_name]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = list(splitter.split(X))

    log.info(
        "Tinh chỉnh %s: %d lần thử x %d fold (TimeSeriesSplit), %s mẫu",
        model_name, n_trials, n_splits, f"{len(X):,}",
    )
    trials: List[TuningTrial] = []
    t_start = time.perf_counter()

    for i in range(n_trials):
        params = space(rng)
        t0 = time.perf_counter()
        aucs: List[float] = []
        iters: List[int] = []

        for tr_idx, va_idx in folds:
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            try:
                auc, n_it = _fit_fold(
                    model_name, params, Xtr, ytr, Xva, yva,
                    cat_cols, scale_pos_weight, seed, n_jobs,
                )
                aucs.append(auc)
                if n_it:
                    iters.append(n_it)
            except Exception as exc:  # pragma: no cover
                log.warning("  lần thử %d, fold lỗi: %s", i + 1, exc)

        if not aucs:
            continue
        trial = TuningTrial(
            trial=i + 1, params=params,
            cv_auc_mean=float(np.mean(aucs)), cv_auc_std=float(np.std(aucs)),
            fold_aucs=aucs, fit_seconds=time.perf_counter() - t0,
            mean_iterations=float(np.mean(iters)) if iters else 0.0,
        )
        trials.append(trial)
        log.info(
            "  lần thử %2d/%d: AUC = %.4f ± %.4f  (%.0fs)  %s",
            i + 1, n_trials, trial.cv_auc_mean, trial.cv_auc_std, trial.fit_seconds,
            _fmt(params),
        )

    if not trials:
        raise RuntimeError(f"Không có lần thử nào thành công cho {model_name}")

    best = max(trials, key=lambda t: t.cv_auc_mean)
    log.info("=> Tham số tốt nhất cho %s (AUC CV = %.4f): %s",
             model_name, best.cv_auc_mean, _fmt(best.params))
    return TuningResult(
        model_name=model_name, best_params=best.params, best_auc=best.cv_auc_mean,
        trials=trials, total_seconds=time.perf_counter() - t_start,
    )


def _fit_fold(
    model_name: str, params: Dict[str, Any],
    Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray,
    cat_cols: Sequence[str], spw: float, seed: int, n_jobs: int,
) -> tuple[float, int]:
    """Huấn luyện một fold và trả về ``(AUC, số vòng lặp)``."""
    if model_name == "lightgbm":
        import lightgbm as lgb

        # Chia tiếp một phần nhỏ của fold huấn luyện để dừng sớm, giữ tập kiểm
        # định của fold nguyên vẹn cho việc đánh giá.
        k = int(len(Xtr) * 0.88)
        m = lgb.LGBMClassifier(
            n_estimators=3000, scale_pos_weight=spw, subsample_freq=1,
            random_state=seed, n_jobs=n_jobs, verbose=-1,
            # Xem ghi chú ở zoo.py: không có hai cờ này, AUC kiểm định chéo lệch
            # khoảng 3e-4 giữa các lần chạy. Mức lệch đó đủ để đảo thứ hạng giữa
            # hai bộ siêu tham số sát nhau và chọn ra cấu hình khác.
            deterministic=True, force_row_wise=True,
            **params,
        )
        m.fit(
            Xtr.iloc[:k], ytr[:k],
            eval_X=Xtr.iloc[k:], eval_y=ytr[k:], eval_metric="auc",
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1])), int(m.best_iteration_ or 0)

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        k = int(len(Xtr) * 0.88)
        m = XGBClassifier(
            n_estimators=3000, scale_pos_weight=spw, tree_method="hist",
            enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
            early_stopping_rounds=150, random_state=seed, n_jobs=n_jobs, **params,
        )
        m.fit(Xtr.iloc[:k], ytr[:k], eval_set=[(Xtr.iloc[k:], ytr[k:])], verbose=False)
        return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1])), int(m.best_iteration or 0)

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.compose import ColumnTransformer

        num = [c for c in Xtr.columns if c not in cat_cols]
        prep = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), num),
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
                    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]), list(cat_cols)),
            ]
        )
        m = Pipeline([
            ("prep", prep),
            ("clf", RandomForestClassifier(
                class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed, **params
            )),
        ])
        m.fit(Xtr, ytr)
        return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1])), 0

    raise ValueError(model_name)


def _fmt(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(parts)


__all__ = ["tune_model", "TuningResult", "TuningTrial", "SPACES"]
