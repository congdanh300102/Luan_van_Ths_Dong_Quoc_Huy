"""Biến đổi WOE và chỉ số IV - nền tảng của phương pháp thẻ điểm (scorecard).

CƠ SỞ PHƯƠNG PHÁP
-----------------
Trong thực tiễn xây dựng thẻ điểm tín dụng, các biến đầu vào thường không được
đưa trực tiếp vào mô hình hồi quy logistic mà được *rời rạc hóa* thành các
khoảng (binning) rồi thay thế bằng giá trị **Weight of Evidence**:

    WOE_i = ln( (Good_i / Good_total) / (Bad_i / Bad_total) )

và mức độ dự báo của biến được lượng hóa bằng **Information Value**:

    IV = Σ_i ( Good_i/Good_total − Bad_i/Bad_total ) · WOE_i

Cách làm này mang lại bốn lợi ích quan trọng cho bài toán tín dụng:

1. **Tuyến tính hóa quan hệ** giữa biến đầu vào và log-odds, do đó hồi quy
   logistic - vốn giả định quan hệ tuyến tính trên thang log-odds - có thể nắm
   bắt cả các quan hệ phi tuyến và phi đơn điệu (ví dụ quan hệ chữ U của tỷ lệ
   sử dụng hạn mức).
2. **Xử lý tự nhiên giá trị thiếu**: giá trị thiếu được coi là một nhóm riêng có
   WOE riêng, thay vì phải điền khuyết - đặc biệt quan trọng với khách hàng
   thin-file có phần lớn chỉ tiêu CIC bị thiếu.
3. **Giảm ảnh hưởng của giá trị ngoại lai** do dữ liệu đã được rời rạc hóa.
4. **Tính giải thích được**: mỗi khoảng giá trị tương ứng một số điểm cụ thể,
   đáp ứng yêu cầu minh bạch của cơ quan quản lý đối với mô hình tín dụng.

Thang đo IV thường dùng trong thực tiễn:
``IV < 0,02`` không có giá trị dự báo; ``0,02-0,1`` yếu; ``0,1-0,3`` trung bình;
``0,3-0,5`` mạnh; ``> 0,5`` rất mạnh (cần kiểm tra khả năng rò rỉ dữ liệu).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger(__name__)

# Hằng số điều chỉnh Laplace, tránh log(0) khi một nhóm không có hồ sơ xấu/tốt
LAPLACE = 0.5

# Nhãn của nhóm giá trị thiếu
MISSING_BIN = "__MISSING__"


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Bộ biến đổi WOE cho cả biến liên tục và biến phân loại.

    Tham số
    -------
    numeric_features, categorical_features
        Danh sách tên cột theo từng loại.
    n_bins
        Số khoảng tối đa khi rời rạc hóa biến liên tục (theo phân vị).
    min_bin_frac
        Tỷ lệ mẫu tối thiểu của một khoảng; các khoảng nhỏ hơn sẽ được gộp.
    monotonic
        Nếu ``True``, gộp các khoảng liền kề để chuỗi WOE trở nên đơn điệu.
        Ràng buộc đơn điệu giúp thẻ điểm phù hợp trực giác nghiệp vụ và ổn định
        hơn khi dữ liệu thay đổi.
    """

    def __init__(
        self,
        numeric_features: Sequence[str] = (),
        categorical_features: Sequence[str] = (),
        n_bins: int = 10,
        min_bin_frac: float = 0.03,
        monotonic: bool = True,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.n_bins = n_bins
        self.min_bin_frac = min_bin_frac
        self.monotonic = monotonic

    # -- huấn luyện --------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WOEEncoder":
        y = np.asarray(y).astype(int)
        n_good = float((y == 0).sum()) or 1.0
        n_bad = float((y == 1).sum()) or 1.0
        self.n_good_, self.n_bad_ = n_good, n_bad

        self.edges_: Dict[str, np.ndarray] = {}
        self.woe_maps_: Dict[str, Dict] = {}
        self.iv_: Dict[str, float] = {}
        self.bin_tables_: Dict[str, pd.DataFrame] = {}

        for col in self.numeric_features:
            v = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype="float64")
            edges = self._quantile_edges(v)
            self.edges_[col] = edges
            codes = self._numeric_codes(v, edges)
            self._fit_column(col, codes, y)

        for col in self.categorical_features:
            codes = X[col].astype("object").where(X[col].notna(), MISSING_BIN).astype(str).to_numpy()
            # gộp các hạng mục có tần suất thấp thành nhóm "OTHER"
            vc = pd.Series(codes).value_counts(normalize=True)
            rare = set(vc[vc < self.min_bin_frac].index)
            self.rare_levels_ = getattr(self, "rare_levels_", {})
            self.rare_levels_[col] = rare
            codes = np.where(np.isin(codes, list(rare)), "__OTHER__", codes)
            self._fit_column(col, codes, y)

        self.feature_names_out_ = [f"woe_{c}" for c in self.numeric_features + self.categorical_features]
        return self

    def _quantile_edges(self, v: np.ndarray) -> np.ndarray:
        """Xác định biên khoảng theo phân vị, loại bỏ biên trùng lặp."""
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return np.array([-np.inf, np.inf])
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(finite, qs))
        return np.concatenate([[-np.inf], edges, [np.inf]])

    @staticmethod
    def _numeric_codes(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Gán mã khoảng cho từng giá trị; giá trị thiếu vào nhóm riêng."""
        codes = np.digitize(v, edges[1:-1], right=True).astype("object")
        codes = np.array([f"B{int(c):02d}" for c in codes], dtype="object")
        codes[~np.isfinite(v)] = MISSING_BIN
        return codes

    def _fit_column(self, col: str, codes: np.ndarray, y: np.ndarray) -> None:
        """Tính WOE/IV cho một cột đã được rời rạc hóa."""
        df = pd.DataFrame({"bin": codes, "y": y})
        g = df.groupby("bin", observed=True)["y"].agg(["size", "sum"])
        g.columns = ["n", "n_bad"]
        g["n_good"] = g["n"] - g["n_bad"]
        g["dist_good"] = (g["n_good"] + LAPLACE) / (self.n_good_ + LAPLACE * len(g))
        g["dist_bad"] = (g["n_bad"] + LAPLACE) / (self.n_bad_ + LAPLACE * len(g))
        g["woe"] = np.log(g["dist_good"] / g["dist_bad"])
        g["iv"] = (g["dist_good"] - g["dist_bad"]) * g["woe"]
        g["bad_rate"] = g["n_bad"] / g["n"]

        if self.monotonic and col in self.numeric_features:
            g = self._enforce_monotonic(g)

        self.woe_maps_[col] = g["woe"].to_dict()
        self.iv_[col] = float(g["iv"].sum())
        self.bin_tables_[col] = g.reset_index()

    def _enforce_monotonic(self, g: pd.DataFrame) -> pd.DataFrame:
        """Làm trơn chuỗi WOE theo hướng đơn điệu bằng hồi quy đơn điệu (PAVA).

        Sử dụng thuật toán Pool Adjacent Violators trên các khoảng có thứ tự,
        giữ nguyên nhóm giá trị thiếu (không thuộc trật tự tự nhiên).
        """
        ordered = g.loc[[i for i in g.index if i != MISSING_BIN]].copy()
        if len(ordered) < 3:
            return g
        ordered = ordered.sort_index()
        w = ordered["n"].to_numpy(dtype=float)
        v = ordered["woe"].to_numpy(dtype=float)
        # xác định hướng đơn điệu theo tương quan với thứ tự khoảng
        direction = 1.0 if np.corrcoef(np.arange(len(v)), v)[0, 1] >= 0 else -1.0
        smoothed = _pava(v * direction, w) * direction
        ordered["woe"] = smoothed
        g = g.copy()
        g.loc[ordered.index, "woe"] = ordered["woe"]
        return g

    # -- áp dụng -----------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.numeric_features:
            v = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype="float64")
            codes = self._numeric_codes(v, self.edges_[col])
            out[f"woe_{col}"] = self._map(col, codes)
        for col in self.categorical_features:
            codes = X[col].astype("object").where(X[col].notna(), MISSING_BIN).astype(str).to_numpy()
            rare = self.rare_levels_.get(col, set())
            codes = np.where(np.isin(codes, list(rare)), "__OTHER__", codes)
            out[f"woe_{col}"] = self._map(col, codes)
        return pd.DataFrame(out, index=X.index)

    def _map(self, col: str, codes: np.ndarray) -> np.ndarray:
        """Tra WOE; hạng mục chưa từng thấy được gán WOE = 0 (trung tính)."""
        m = self.woe_maps_[col]
        return np.array([m.get(c, 0.0) for c in codes], dtype="float64")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    # -- báo cáo -----------------------------------------------------------
    def iv_table(self) -> pd.DataFrame:
        """Bảng xếp hạng biến theo Information Value."""
        rows = [
            {
                "feature": c,
                "iv": v,
                "n_bins": len(self.bin_tables_[c]),
                "strength": _iv_strength(v),
            }
            for c, v in self.iv_.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def select_by_iv(self, min_iv: float = 0.02, max_iv: float = 1.5) -> List[str]:
        """Chọn biến có IV nằm trong khoảng hợp lệ.

        Chặn trên ``max_iv`` nhằm phát hiện các biến có năng lực dự báo bất
        thường - dấu hiệu điển hình của rò rỉ dữ liệu (data leakage).
        """
        return [c for c, v in self.iv_.items() if min_iv <= v <= max_iv]


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Hồi quy đơn điệu không giảm theo thuật toán Pool Adjacent Violators."""
    y = y.astype(float).copy()
    w = w.astype(float).copy()
    n = len(y)
    # mỗi khối lưu (tổng trọng số, tổng có trọng số, số phần tử)
    blocks: List[list] = [[w[i], w[i] * y[i], 1] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        m1 = blocks[i][1] / blocks[i][0]
        m2 = blocks[i + 1][1] / blocks[i + 1][0]
        if m1 <= m2 + 1e-12:
            i += 1
            continue
        # vi phạm tính đơn điệu -> gộp hai khối và lùi lại kiểm tra
        blocks[i] = [
            blocks[i][0] + blocks[i + 1][0],
            blocks[i][1] + blocks[i + 1][1],
            blocks[i][2] + blocks[i + 1][2],
        ]
        del blocks[i + 1]
        if i > 0:
            i -= 1
    out = np.empty(n, dtype=float)
    pos = 0
    for wt, ws, cnt in blocks:
        out[pos : pos + cnt] = ws / wt
        pos += cnt
    return out


def _iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "Không có giá trị"
    if iv < 0.10:
        return "Yếu"
    if iv < 0.30:
        return "Trung bình"
    if iv < 0.50:
        return "Mạnh"
    return "Rất mạnh"


# ---------------------------------------------------------------------------
# Chuyển hệ số hồi quy logistic thành thẻ điểm
# ---------------------------------------------------------------------------
def build_scorecard(
    encoder: WOEEncoder,
    coefficients: Dict[str, float],
    intercept: float,
    base_score: float = 600.0,
    base_odds: float = 20.0,
    pdo: float = 40.0,
) -> pd.DataFrame:
    """Chuyển mô hình hồi quy logistic trên biến WOE thành thẻ điểm.

    Theo phương pháp chuẩn của ngành, điểm được xây dựng sao cho mỗi lần tăng
    ``pdo`` điểm thì tỷ số odds tốt/xấu tăng gấp đôi:

        ``factor = pdo / ln(2)``, ``offset = base_score − factor · ln(base_odds)``

    Điểm của một khoảng giá trị: ``−(β_j · WOE_ij + α/m) · factor``, trong đó
    ``m`` là số biến trong mô hình.
    """
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    m = max(len(coefficients), 1)

    rows = []
    for feat, beta in coefficients.items():
        col = feat.replace("woe_", "")
        table = encoder.bin_tables_.get(col)
        if table is None:
            continue
        for _, r in table.iterrows():
            points = -(beta * r["woe"] + intercept / m) * factor + offset / m
            rows.append(
                {
                    "feature": col,
                    "bin": r["bin"],
                    "n": int(r["n"]),
                    "bad_rate": round(float(r["bad_rate"]), 4),
                    "woe": round(float(r["woe"]), 4),
                    "beta": round(float(beta), 4),
                    "points": round(float(points), 1),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["WOEEncoder", "build_scorecard", "MISSING_BIN"]
