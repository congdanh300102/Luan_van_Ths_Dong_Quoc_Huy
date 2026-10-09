"""Giải thích mô hình: SHAP và tầm quan trọng theo phép hoán vị.

YÊU CẦU GIẢI THÍCH ĐƯỢC TRONG TÍN DỤNG
--------------------------------------
Mô hình tín dụng chịu ràng buộc pháp lý và nghiệp vụ đặc thù: tổ chức tín dụng
phải có khả năng giải trình *lý do* từ chối một hồ sơ cụ thể. Do đó bên cạnh
năng lực dự báo, khả năng giải thích là tiêu chí lựa chọn mô hình bắt buộc.

Module cung cấp ba mức giải thích:

1. **Toàn cục** (*global*) - mức độ ảnh hưởng bình quân của từng đặc trưng trên
   toàn bộ tập dữ liệu, tính bằng trung bình giá trị SHAP tuyệt đối. Dùng cho
   báo cáo kiểm định mô hình và phát hiện đặc trưng bất thường.
2. **Theo nhóm đặc trưng** - tổng hợp ảnh hưởng theo nguồn dữ liệu (tĩnh, CIC,
   hành vi giao dịch), trả lời câu hỏi liệu dữ liệu luồng có thực sự đóng góp
   vào quyết định hay không.
3. **Cục bộ** (*local*) - phân rã điểm rủi ro của một hồ sơ cụ thể thành đóng góp
   của từng đặc trưng, là cơ sở sinh *lý do từ chối* (reason codes) trả về cho
   cán bộ tín dụng và khách hàng.

Phép hoán vị (permutation importance) được dùng bổ trợ vì nó đo trực tiếp mức
suy giảm AUC khi phá vỡ quan hệ giữa một đặc trưng và nhãn - một thước đo không
phụ thuộc cấu trúc mô hình, hữu ích để kiểm tra chéo kết quả SHAP.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


# ===========================================================================
# SHAP
# ===========================================================================
def shap_analysis(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 3000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Tính giá trị SHAP cho mô hình dạng cây.

    Với các mô hình cây (XGBoost, LightGBM, CatBoost, Random Forest) thuật toán
    TreeSHAP cho phép tính chính xác giá trị Shapley trong thời gian đa thức,
    nên có thể áp dụng trực tiếp trên hàng nghìn mẫu.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover
        log.warning("Không tìm thấy thư viện shap - bỏ qua phân tích SHAP")
        return {}

    rng = np.random.default_rng(seed)
    n = min(max_samples, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xs = X.iloc[np.sort(idx)]

    estimator = _final_estimator(model)
    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(Xs)
        # một số phiên bản trả về danh sách theo lớp
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:            # (n, features, classes)
            values = values[:, :, -1]
    except Exception as exc:
        log.warning("TreeExplainer thất bại (%s) - bỏ qua SHAP", exc)
        return {}

    mean_abs = np.abs(values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": Xs.columns, "shap_mean_abs": mean_abs})
        .sort_values("shap_mean_abs", ascending=False)
        .reset_index(drop=True)
    )
    return {"values": values, "X": Xs, "importance": importance,
            "base_value": getattr(explainer, "expected_value", None)}


def shap_by_group(
    importance: pd.DataFrame, group_map: Dict[str, str]
) -> pd.DataFrame:
    """Tổng hợp ảnh hưởng SHAP theo nhóm nguồn dữ liệu."""
    d = importance.copy()
    d["group"] = d["feature"].map(group_map).fillna("KHÁC")
    g = (
        d.groupby("group", observed=True)
        .agg(n_features=("feature", "size"), shap_sum=("shap_mean_abs", "sum"),
             shap_mean=("shap_mean_abs", "mean"), shap_max=("shap_mean_abs", "max"))
        .reset_index()
    )
    g["share_pct"] = 100 * g["shap_sum"] / g["shap_sum"].sum()
    return g.sort_values("shap_sum", ascending=False).reset_index(drop=True)


def plot_shap_summary(shap_result: Dict[str, Any], out_dir, top_n: int = 20,
                      name: str = "shap_summary") -> None:
    """Đồ thị beeswarm SHAP - phân bố ảnh hưởng của từng đặc trưng."""
    if not shap_result:
        return
    try:
        import shap
    except ImportError:  # pragma: no cover
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plots import apply_style, save

    apply_style()
    fig = plt.figure(figsize=(7.8, max(4.5, 0.32 * top_n)))
    shap.summary_plot(
        shap_result["values"], shap_result["X"], max_display=top_n,
        show=False, plot_size=None, color_bar_label="Giá trị đặc trưng",
    )
    plt.title("Phân bố giá trị SHAP theo đặc trưng", fontsize=11)
    plt.xlabel("Giá trị SHAP (ảnh hưởng tới log-odds vỡ nợ)")
    save(fig, out_dir, name)


def plot_shap_group_share(group_df: pd.DataFrame, out_dir, name: str = "shap_group_share") -> None:
    """Tỷ trọng đóng góp của từng nhóm đặc trưng vào quyết định của mô hình."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plots import PALETTE, apply_style, save

    apply_style()
    GROUP_LABELS = {
        "STATIC": "Tĩnh (hồ sơ KH)",
        "LOAN": "Tham số khoản vay",
        "CIC": "Lịch sử tín dụng CIC",
        "BEHAVIOUR": "Hành vi giao dịch (luồng)",
        "DERIVED": "Tổ hợp (derived)",
        "KHÁC": "Khác",
    }
    d = group_df.copy()
    d["label"] = d["group"].map(GROUP_LABELS).fillna(d["group"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    ax = axes[0]
    dd = d.sort_values("shap_sum")
    ax.barh(dd["label"], dd["shap_sum"], color=PALETTE[0], height=0.6,
            edgecolor="white", linewidth=0.6)
    for i, (v, p) in enumerate(zip(dd["shap_sum"], dd["share_pct"])):
        ax.text(v, i, f"  {v:.3f} ({p:.1f}%)", va="center", fontsize=8.5)
    ax.set_xlabel("Tổng |SHAP| bình quân của nhóm")
    ax.set_title("Đóng góp tuyệt đối theo nhóm đặc trưng")
    ax.set_xlim(0, float(dd["shap_sum"].max()) * 1.42)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ax.pie(d["share_pct"], labels=d["label"], autopct="%1.1f%%", startangle=100,
           colors=PALETTE[: len(d)], textprops=dict(fontsize=8.5),
           wedgeprops=dict(edgecolor="white", linewidth=1.2))
    ax.set_title("Tỷ trọng đóng góp vào quyết định")
    save(fig, out_dir, name)


def local_explanation(
    shap_result: Dict[str, Any], row_index: int, top_n: int = 8
) -> pd.DataFrame:
    """Phân rã điểm rủi ro của một hồ sơ thành các đóng góp - sinh *reason codes*.

    Kết quả là danh sách đặc trưng có ảnh hưởng lớn nhất tới quyết định đối với
    hồ sơ đó, kèm chiều ảnh hưởng (làm tăng hay giảm rủi ro). Đây chính là dữ
    liệu mà Decision Engine trả về trong trường ``reason_codes``.
    """
    if not shap_result:
        return pd.DataFrame()
    vals = shap_result["values"][row_index]
    X = shap_result["X"]
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "value": X.iloc[row_index].to_numpy(),
            "shap": vals,
        }
    )
    df["abs_shap"] = df["shap"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    df["direction"] = np.where(df["shap"] > 0, "Tăng rủi ro", "Giảm rủi ro")
    return df.drop(columns=["abs_shap"])


# ===========================================================================
# Permutation importance
# ===========================================================================
def permutation_analysis(
    predict_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str] | None = None,
    n_repeats: int = 5,
    max_samples: int = 8000,
    seed: int = 42,
) -> pd.DataFrame:
    """Đo mức suy giảm AUC khi hoán vị ngẫu nhiên từng đặc trưng.

    Thực hiện thủ công (không dùng ``sklearn.inspection``) để có thể áp dụng cho
    cả các mô hình có hiệu chuẩn xác suất bên ngoài pipeline.
    """
    rng = np.random.default_rng(seed)
    n = min(max_samples, len(X))
    idx = np.sort(rng.choice(len(X), size=n, replace=False))
    Xs, ys = X.iloc[idx].reset_index(drop=True), np.asarray(y)[idx]

    base_auc = float(roc_auc_score(ys, predict_fn(Xs)))
    cols = list(features) if features else list(Xs.columns)
    log.info("Permutation importance: %d đặc trưng x %d lần lặp (AUC gốc = %.4f)",
             len(cols), n_repeats, base_auc)

    rows: List[dict] = []
    for j, col in enumerate(cols):
        drops = np.empty(n_repeats, dtype=float)
        original = Xs[col]
        for r in range(n_repeats):
            Xp = Xs.copy()
            # Giữ nguyên kiểu dữ liệu của cột khi hoán vị. Nếu gán trực tiếp một
            # mảng NumPy đã hoán vị, cột kiểu ``category`` sẽ bị chuyển thành
            # ``object``/``str`` và các mô hình dùng đặc trưng phân loại gốc
            # (XGBoost, LightGBM) sẽ báo lỗi kiểu dữ liệu không hợp lệ.
            order = rng.permutation(len(Xs))
            Xp[col] = pd.Series(
                original.to_numpy()[order], index=Xs.index, dtype=original.dtype
            )
            drops[r] = base_auc - float(roc_auc_score(ys, predict_fn(Xp)))
        rows.append({"feature": col, "importance": float(drops.mean()),
                     "std": float(drops.std())})
        if (j + 1) % 25 == 0:
            log.info("  ... %d/%d", j + 1, len(cols))

    out = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    out.attrs["base_auc"] = base_auc
    return out


def _final_estimator(model: Any) -> Any:
    """Lấy bộ phân loại cuối cùng nếu đầu vào là một pipeline."""
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


__all__ = [
    "shap_analysis",
    "shap_by_group",
    "plot_shap_summary",
    "plot_shap_group_share",
    "local_explanation",
    "permutation_analysis",
]
