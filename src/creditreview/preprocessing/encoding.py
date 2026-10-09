"""Chuẩn hóa biến định lượng và mã hóa biến định tính.

Mỗi họ mô hình có yêu cầu khác nhau về dạng đầu vào, nên đề án sử dụng ba
phương án biến đổi song song thay vì một phương án chung:

``numeric_ohe``
    Điền khuyết bằng trung vị, chuẩn hóa z-score, mã hóa one-hot cho biến định
    tính. Dùng cho hồi quy logistic và mạng nơ-ron -- các mô hình dựa trên
    khoảng cách/gradient nên nhạy cảm với chênh lệch thang đo giữa các biến
    (thu nhập tính bằng đồng, tỷ lệ DTI nằm trong [0; 1]).

``ordinal``
    Điền khuyết, mã hóa thứ tự cho biến định tính. Dùng cho cây quyết định và
    các mô hình học tập hợp của scikit-learn: cây chia theo ngưỡng nên bất biến
    với phép biến đổi đơn điệu, việc chuẩn hóa không những không cần thiết mà
    còn làm mất khả năng diễn giải ngưỡng chia.

``native_cat``
    Giữ nguyên biến định tính ở dạng hạng mục để XGBoost, LightGBM và CatBoost
    tự xử lý bằng cơ chế riêng của từng thư viện (xem
    :mod:`creditreview.models.train`).

``woe``
    Rời rạc hóa và thay mỗi khoảng bằng giá trị Weight of Evidence. Dùng cho mô
    hình thẻ điểm (xem :mod:`creditreview.models.woe`).

Toàn bộ phép biến đổi được gói trong ``ColumnTransformer`` của scikit-learn nên
tham số (trung vị, trung bình, độ lệch chuẩn, danh mục hạng mục) chỉ được ước
lượng trên dữ liệu huấn luyện của từng lần khớp mô hình.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

log = logging.getLogger(__name__)

ENCODINGS = ("numeric_ohe", "ordinal", "native_cat", "woe")


def make_preprocessor(
    numeric: Sequence[str], categorical: Sequence[str], encoding: str
) -> ColumnTransformer:
    """Tạo khối tiền xử lý phù hợp với từng họ mô hình.

    Tham số
    -------
    numeric, categorical
        Danh sách tên biến định lượng và định tính.
    encoding
        ``"numeric_ohe"`` hoặc ``"ordinal"`` (hai phương án cần khối tiền xử lý
        tường minh). Hai phương án còn lại (``native_cat``, ``woe``) được xử lý
        ở tầng mô hình nên không đi qua hàm này.
    """
    numeric = list(numeric)
    categorical = list(categorical)

    if encoding == "numeric_ohe":
        num_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median", add_indicator=True)),
             ("scale", StandardScaler())]
        )
        cat_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="constant", fill_value="__NA__")),
             ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                   min_frequency=0.01, sparse_output=False))]
        )
    elif encoding == "ordinal":
        num_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        )
        cat_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="constant", fill_value="__NA__")),
             ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
        )
    else:
        raise ValueError(
            f"encoding không hỗ trợ: {encoding!r} (chấp nhận: 'numeric_ohe', 'ordinal')"
        )

    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


__all__ = ["make_preprocessor", "ENCODINGS"]
