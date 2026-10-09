"""Tập hợp mô hình học máy dùng cho bài toán dự báo rủi ro tín dụng.

Module cung cấp một *nhà máy mô hình* (model factory) trả về các pipeline tương
thích scikit-learn, bảo đảm mọi mô hình đều được huấn luyện và đánh giá trên
cùng một tập đặc trưng, cùng một cách chia dữ liệu và cùng một bộ chỉ tiêu.

MƯỜI HAI MÔ HÌNH ĐƯỢC SO SÁNH
-----------------------------
Năm thuật toán mà đề án xác định là trọng tâm nghiên cứu -- hồi quy logistic,
cây quyết định, rừng ngẫu nhiên, máy vectơ hỗ trợ và gradient boosting -- đều có
mặt. Bên cạnh đó là các biến thể hiện đại của cùng những họ thuật toán này, đưa
vào để đánh giá xem mức cải thiện thực tế có tương xứng với chi phí triển khai
hay không.

Nhóm tuyến tính / giải thích được:
  1. ``logistic_regression``  - Hồi quy logistic có điều chuẩn L2 trên đặc trưng
                                đã chuẩn hóa và mã hóa one-hot.
  2. ``scorecard_woe``        - Hồi quy logistic trên đặc trưng WOE (thẻ điểm
                                theo chuẩn thực tiễn ngân hàng).
Nhóm cây đơn:
  3. ``decision_tree``        - Cây quyết định CART có tỉa nhánh.
Nhóm học tập hợp song song (bagging):
  4. ``random_forest``        - Rừng ngẫu nhiên.
  5. ``extra_trees``          - Cây cực ngẫu nhiên.
Nhóm máy vectơ hỗ trợ:
  6. ``svm``                  - SVM hạt nhân RBF (xấp xỉ Nystroem + Platt scaling).
Nhóm học tập hợp tuần tự (boosting):
  7. ``gradient_boosting``    - Histogram Gradient Boosting (scikit-learn).
  8. ``xgboost``              - XGBoost.
  9. ``lightgbm``             - LightGBM.
 10. ``catboost``             - CatBoost.
Nhóm mạng nơ-ron:
 11. ``mlp``                  - Mạng nơ-ron truyền thẳng nhiều lớp.
Nhóm kết hợp:
 12. ``stacking_ensemble``    - Kết hợp xếp tầng các mô hình cơ sở khác biệt nhất.

XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU
--------------------------
Tỷ lệ hồ sơ vỡ nợ khoảng 15%, tương ứng mức mất cân bằng vừa phải. Thay vì lấy
mẫu lại (dễ làm sai lệch phân phối xác suất và phá vỡ tính hiệu chuẩn), các mô
hình sử dụng **trọng số lớp** (``class_weight`` / ``scale_pos_weight``). Cách
tiếp cận này giữ nguyên phân phối dữ liệu gốc, sau đó xác suất được *hiệu chuẩn
lại* ở bước riêng (xem ``models/train.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ..preprocessing.encoding import make_preprocessor
from .woe import WOEEncoder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Siêu dữ liệu mô hình
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """Đặc tả một mô hình trong thực nghiệm so sánh."""

    name: str
    display_name: str
    family: str
    estimator: Any
    # Cần bộ đặc trưng dạng nào: "numeric_ohe" | "ordinal" | "native_cat" | "woe"
    encoding: str
    # Có hỗ trợ dừng sớm dựa trên tập kiểm định hay không
    supports_early_stopping: bool = False
    notes: str = ""
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)


FAMILY_LINEAR = "Tuyến tính"
FAMILY_TREE = "Cây quyết định"
FAMILY_BAGGING = "Học tập hợp song song"
FAMILY_SVM = "Máy vectơ hỗ trợ"
FAMILY_BOOSTING = "Học tập hợp tuần tự"
FAMILY_NN = "Mạng nơ-ron"
FAMILY_ENSEMBLE = "Kết hợp xếp tầng"


# ---------------------------------------------------------------------------
# Nhà máy mô hình
# ---------------------------------------------------------------------------
def load_tuned_params(path: Any) -> Dict[str, Dict[str, Any]]:
    """Nạp siêu tham số đã tinh chỉnh từ ``results/tables/tuning_best.json``.

    Nếu file không tồn tại, trả về dict rỗng và các mô hình dùng tham số mặc
    định - nhờ đó bước huấn luyện luôn chạy được kể cả khi chưa tinh chỉnh.
    """
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        out = {k: v.get("best_params", {}) for k, v in data.items() if v.get("best_params")}
        if out:
            log.info("Đã nạp siêu tham số tinh chỉnh cho: %s", ", ".join(out))
        return out
    except Exception as exc:  # pragma: no cover
        log.warning("Không đọc được file siêu tham số (%s) - dùng tham số mặc định", exc)
        return {}


def build_model_specs(
    numeric: Sequence[str],
    categorical: Sequence[str],
    scale_pos_weight: float,
    seed: int = 42,
    n_jobs: int = -1,
    tuned: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, ModelSpec]:
    """Khởi tạo toàn bộ đặc tả mô hình.

    ``scale_pos_weight`` là tỷ lệ ``n_negative / n_positive`` trên tập huấn
    luyện, dùng để bù mất cân bằng lớp cho các mô hình boosting.

    ``tuned`` là dict siêu tham số đã tinh chỉnh theo tên mô hình; nếu có, các
    giá trị này ghi đè tham số mặc định.
    """
    numeric, categorical = list(numeric), list(categorical)
    tuned = tuned or {}
    specs: Dict[str, ModelSpec] = {}

    def P(model: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Trộn tham số mặc định với tham số đã tinh chỉnh (nếu có)."""
        merged = dict(defaults)
        merged.update(tuned.get(model, {}))
        return merged

    # ---------------------- 1. Hồi quy logistic ------------------------------
    specs["logistic_regression"] = ModelSpec(
        name="logistic_regression",
        display_name="Hồi quy logistic",
        family=FAMILY_LINEAR,
        encoding="numeric_ohe",
        estimator=Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, "numeric_ohe")),
                ("clf", LogisticRegression(
                    C=0.25, solver="lbfgs", max_iter=5000,
                    class_weight="balanced", random_state=seed,
                )),
            ]
        ),
        notes="Điều chuẩn L2, trọng số lớp cân bằng. Mô hình cơ sở tham chiếu.",
        param_grid={"clf__C": [0.05, 0.1, 0.25, 0.5, 1.0]},
    )

    # ---------------------- 2. Thẻ điểm WOE ---------------------------------
    specs["scorecard_woe"] = ModelSpec(
        name="scorecard_woe",
        display_name="Thẻ điểm WOE + Hồi quy logistic",
        family=FAMILY_LINEAR,
        encoding="woe",
        estimator=Pipeline(
            [
                ("woe", WOEEncoder(numeric_features=numeric, categorical_features=categorical,
                                   n_bins=10, min_bin_frac=0.03, monotonic=True)),
                ("clf", LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=5000,
                    class_weight="balanced", random_state=seed,
                )),
            ]
        ),
        notes="Rời rạc hóa + WOE có ràng buộc đơn điệu; đáp ứng yêu cầu giải thích được.",
        param_grid={"clf__C": [0.1, 0.5, 1.0, 2.0]},
    )

    # ---------------------- 3. Cây quyết định -------------------------------
    specs["decision_tree"] = ModelSpec(
        name="decision_tree",
        display_name="Cây quyết định (CART)",
        family=FAMILY_TREE,
        encoding="ordinal",
        estimator=Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, "ordinal")),
                ("clf", DecisionTreeClassifier(
                    criterion="gini", max_depth=8, min_samples_leaf=120,
                    min_samples_split=250, ccp_alpha=1e-4,
                    class_weight="balanced", random_state=seed,
                )),
            ]
        ),
        notes="Có tỉa nhánh theo độ phức tạp chi phí (ccp_alpha) để hạn chế quá khớp.",
        param_grid={"clf__max_depth": [4, 6, 8, 10], "clf__min_samples_leaf": [50, 120, 250]},
    )

    # ---------------------- 4. Rừng ngẫu nhiên ------------------------------
    specs["random_forest"] = ModelSpec(
        name="random_forest",
        display_name="Rừng ngẫu nhiên",
        family=FAMILY_BAGGING,
        encoding="ordinal",
        estimator=Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, "ordinal")),
                ("clf", RandomForestClassifier(
                    class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed,
                    **P("random_forest", {
                        "n_estimators": 500, "max_depth": 14,
                        "min_samples_leaf": 25, "max_features": "sqrt",
                    }),
                )),
            ]
        ),
        notes="500 cây, lấy mẫu đặc trưng theo căn bậc hai để giảm tương quan giữa các cây.",
        param_grid={"clf__max_depth": [10, 14, 20], "clf__min_samples_leaf": [10, 25, 50]},
    )

    # ---------------------- 5. Extra Trees ----------------------------------
    specs["extra_trees"] = ModelSpec(
        name="extra_trees",
        display_name="Cây cực ngẫu nhiên (Extra Trees)",
        family=FAMILY_BAGGING,
        encoding="ordinal",
        estimator=Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, "ordinal")),
                ("clf", ExtraTreesClassifier(
                    n_estimators=500, max_depth=18, min_samples_leaf=20,
                    max_features="sqrt", class_weight="balanced_subsample",
                    n_jobs=n_jobs, random_state=seed,
                )),
            ]
        ),
        notes="Chọn điểm chia ngẫu nhiên, phương sai thấp hơn nhưng độ chệch cao hơn.",
    )

    # ---------------------- 6. Máy vectơ hỗ trợ (SVM) -----------------------
    # SVM hạt nhân có chi phí huấn luyện từ O(n²) đến O(n³) theo số quan sát.
    # Với tập huấn luyện hàng chục nghìn hồ sơ, việc giải bài toán đối ngẫu đầy
    # đủ là không khả thi về thời gian. Đề án sử dụng phép xấp xỉ Nystroem:
    # ánh xạ tường minh dữ liệu sang không gian đặc trưng xấp xỉ của hạt nhân
    # RBF bằng một tập nhỏ điểm mốc, sau đó huấn luyện SVM tuyến tính trên
    # không gian đó. Cách này cho nghiệm xấp xỉ nghiệm của SVM-RBF với chi phí
    # tuyến tính theo số quan sát.
    #
    # SVM cho ra khoảng cách tới siêu phẳng chứ không phải xác suất. Để Decision
    # Engine dùng được kết quả, khoảng cách này được quy đổi sang xác suất bằng
    # phương pháp Platt (hồi quy logistic trên giá trị hàm quyết định), thực
    # hiện qua kiểm định chéo ba tầng để tránh quá khớp.
    specs["svm"] = ModelSpec(
        name="svm",
        display_name="Máy vectơ hỗ trợ (SVM-RBF)",
        family=FAMILY_SVM,
        encoding="numeric_ohe",
        estimator=Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, "numeric_ohe")),
                ("kernel", Nystroem(
                    kernel="rbf", gamma=None, n_components=600, random_state=seed,
                )),
                ("clf", CalibratedClassifierCV(
                    LinearSVC(
                        C=0.05, loss="squared_hinge", dual="auto",
                        class_weight="balanced", max_iter=5000, random_state=seed,
                    ),
                    method="sigmoid", cv=3, n_jobs=n_jobs,
                )),
            ]
        ),
        notes="Xấp xỉ hạt nhân RBF bằng Nystroem (600 điểm mốc); "
              "xác suất quy đổi theo phương pháp Platt.",
        param_grid={"clf__estimator__C": [0.01, 0.05, 0.1, 0.5]},
    )

    # ---------------------- 7. Hist Gradient Boosting -----------------------
    specs["gradient_boosting"] = ModelSpec(
        name="gradient_boosting",
        display_name="Gradient Boosting (histogram)",
        family=FAMILY_BOOSTING,
        encoding="ordinal",
        supports_early_stopping=True,
        estimator=Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, "ordinal")),
                ("clf", HistGradientBoostingClassifier(
                    max_iter=1000, learning_rate=0.05, max_leaf_nodes=31,
                    min_samples_leaf=60, l2_regularization=2.0,
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=60, class_weight="balanced", random_state=seed,
                )),
            ]
        ),
        notes="Dừng sớm nội bộ trên 15% dữ liệu huấn luyện.",
    )

    # ---------------------- 8. XGBoost --------------------------------------
    try:
        from xgboost import XGBClassifier

        specs["xgboost"] = ModelSpec(
            name="xgboost",
            display_name="XGBoost",
            family=FAMILY_BOOSTING,
            encoding="native_cat",
            supports_early_stopping=True,
            estimator=XGBClassifier(
                n_estimators=3000,
                scale_pos_weight=scale_pos_weight,
                tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                eval_metric="auc", early_stopping_rounds=150,
                random_state=seed, n_jobs=n_jobs,
                **P("xgboost", {
                    "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 12,
                    "subsample": 0.8, "colsample_bytree": 0.6,
                    "reg_lambda": 5.0, "gamma": 0.1,
                }),
            ),
            notes="Hỗ trợ đặc trưng phân loại gốc; dừng sớm theo AUC trên tập kiểm định.",
            param_grid={"max_depth": [4, 5, 6, 8], "min_child_weight": [5, 12, 25],
                        "learning_rate": [0.02, 0.03, 0.05]},
        )
    except ImportError:  # pragma: no cover
        log.warning("Không tìm thấy xgboost - bỏ qua mô hình này")

    # ---------------------- 9. LightGBM -------------------------------------
    try:
        from lightgbm import LGBMClassifier

        specs["lightgbm"] = ModelSpec(
            name="lightgbm",
            display_name="LightGBM",
            family=FAMILY_BOOSTING,
            encoding="native_cat",
            supports_early_stopping=True,
            estimator=LGBMClassifier(
                n_estimators=3000, subsample_freq=1,
                scale_pos_weight=scale_pos_weight,
                max_cat_threshold=32, cat_smooth=20.0,
                random_state=seed, n_jobs=n_jobs, verbose=-1,
                # Bắt buộc để kết quả lặp lại được: mặc định LightGBM dựng
                # histogram song song theo cột, nên thứ tự cộng dồn số thực phụ
                # thuộc số luồng thực tế được cấp phát. Chạy lại trên máy có mức
                # tải khác sẽ cho AUC lệch ở chữ số thập phân thứ tư.
                deterministic=True, force_row_wise=True,
                **P("lightgbm", {
                    "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 80,
                    "subsample": 0.8, "colsample_bytree": 0.6,
                    "reg_lambda": 5.0, "reg_alpha": 1.0,
                }),
            ),
            notes="Sinh cây theo chiều sâu lá (leaf-wise), hỗ trợ đặc trưng phân loại gốc.",
            param_grid={"num_leaves": [15, 31, 63], "min_child_samples": [40, 80, 150],
                        "learning_rate": [0.02, 0.03, 0.05]},
        )
    except ImportError:  # pragma: no cover
        log.warning("Không tìm thấy lightgbm - bỏ qua mô hình này")

    # ---------------------- 10. CatBoost -----------------------------------
    try:
        from catboost import CatBoostClassifier

        specs["catboost"] = ModelSpec(
            name="catboost",
            display_name="CatBoost",
            family=FAMILY_BOOSTING,
            encoding="native_cat",
            supports_early_stopping=True,
            estimator=CatBoostClassifier(
                iterations=3000, learning_rate=0.04, depth=6,
                l2_leaf_reg=6.0, random_strength=1.0, bagging_temperature=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="AUC", od_type="Iter", od_wait=120,
                random_seed=seed, verbose=False, allow_writing_files=False,
                thread_count=n_jobs if n_jobs and n_jobs > 0 else -1,
            ),
            notes="Sử dụng target statistics có thứ tự cho đặc trưng phân loại, hạn chế rò rỉ mục tiêu.",
        )
    except ImportError:  # pragma: no cover
        log.warning("Không tìm thấy catboost - bỏ qua mô hình này")

    # ---------------------- 11. Mạng nơ-ron --------------------------------
    specs["mlp"] = ModelSpec(
        name="mlp",
        display_name="Mạng nơ-ron truyền thẳng (MLP)",
        family=FAMILY_NN,
        encoding="numeric_ohe",
        estimator=Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, "numeric_ohe")),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                    alpha=1e-3, batch_size=512, learning_rate_init=1e-3,
                    max_iter=300, early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=25, random_state=seed,
                )),
            ]
        ),
        notes="Hai lớp ẩn 128-64, dừng sớm. Không dùng trọng số lớp (bù bằng hiệu chuẩn).",
    )

    return specs


def build_stacking(
    base_specs: Dict[str, ModelSpec],
    members: Sequence[str],
    seed: int = 42,
    n_jobs: int = -1,
    cv: int = 3,
) -> ModelSpec:
    """Tạo mô hình kết hợp xếp tầng từ các mô hình cơ sở đã có.

    Mô hình tổng hợp (meta-learner) là hồi quy logistic trên xác suất dự báo của
    các mô hình cơ sở, được tạo qua dự báo chéo (out-of-fold) để tránh rò rỉ.
    Các thành viên được chọn theo nguyên tắc *đa dạng hóa*: một mô hình tuyến
    tính, một mô hình bagging và các mô hình boosting khác nhau về cơ chế.
    """
    estimators = []
    for m in members:
        if m not in base_specs:
            continue
        if base_specs[m].encoding == "native_cat":
            # các mô hình dùng đặc trưng phân loại gốc không tương thích trực
            # tiếp với StackingClassifier (yêu cầu ma trận số), nên bỏ qua
            continue
        estimators.append((m, base_specs[m].estimator))

    return ModelSpec(
        name="stacking_ensemble",
        display_name="Kết hợp xếp tầng (Stacking)",
        family=FAMILY_ENSEMBLE,
        encoding="ordinal",
        estimator=StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=1.0, max_iter=3000, class_weight="balanced", random_state=seed
            ),
            cv=cv,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=1,   # các mô hình cơ sở đã sử dụng đa luồng nội bộ
        ),
        notes=f"Kết hợp {len(estimators)} mô hình cơ sở qua dự báo chéo {cv} tầng.",
    )


__all__ = [
    "ModelSpec",
    "build_model_specs",
    "build_stacking",
    "make_preprocessor",
    "FAMILY_LINEAR",
    "FAMILY_TREE",
    "FAMILY_BAGGING",
    "FAMILY_SVM",
    "FAMILY_BOOSTING",
    "FAMILY_NN",
    "FAMILY_ENSEMBLE",
]
