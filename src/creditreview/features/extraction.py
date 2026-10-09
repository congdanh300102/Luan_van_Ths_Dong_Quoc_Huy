"""Trích xuất đặc trưng: xây dựng ma trận đầu vào cho mô hình học máy.

Module thực hiện bước trích xuất đặc trưng của quy trình: ghép nối năm nguồn dữ
liệu (hồ sơ khách hàng, tài khoản, đơn vay, bản tin CIC, lịch sử giao dịch)
thành một ma trận ở cấp độ *hồ sơ vay*, trong đó mỗi dòng là một hồ sơ và mỗi
cột là một đặc trưng đã khai báo trong :mod:`creditreview.features.definitions`.

TÍNH ĐÚNG ĐẮN THEO THỜI ĐIỂM (POINT-IN-TIME CORRECTNESS)
-------------------------------------------------------
Yêu cầu cốt lõi của bài toán chấm điểm tín dụng là mọi đặc trưng của một hồ sơ
chỉ được tính từ dữ liệu **đã tồn tại trước hoặc tại thời điểm nộp hồ sơ**. Nếu
vi phạm, mô hình sẽ "nhìn thấy tương lai" (*time travel* / *label leakage*), cho
kết quả đánh giá cao giả tạo và sụp đổ khi triển khai thực tế.

Để bảo đảm điều này với chi phí tính toán chấp nhận được, module sử dụng kỹ
thuật **luỹ kế theo nhóm + ghép gần nhất theo thời gian** (cumulative sum +
``merge_asof``):

1. Với mỗi độ đo ``m`` (số giao dịch, dòng tiền vào/ra, chi tiêu nhóm rủi ro...),
   tính chuỗi luỹ kế ``C_m(i, t)`` theo từng khách hàng ``i`` trên trục thời
   gian giao dịch.
2. Tổng của độ đo trong cửa sổ ``(t - W, t]`` được lấy bằng hiệu hai giá trị
   luỹ kế: ``S_m(i, t, W) = C_m(i, t) - C_m(i, t - W)``.
3. Hai giá trị luỹ kế này được truy xuất bằng ``merge_asof`` hướng *backward* -
   tương đương phép "tìm bản ghi gần nhất không vượt quá mốc thời gian", có độ
   phức tạp tuyến tính sau khi dữ liệu đã được sắp xếp.

Nhờ đó toàn bộ đặc trưng cửa sổ cho N hồ sơ được tính bằng ``2 × |W|`` phép
ghép vector hóa thay vì ``N × |W|`` truy vấn riêng lẻ -- khác biệt quyết định
khi tập dữ liệu có hàng triệu giao dịch.

Cùng một hàm được dùng cho hai tình huống: tính hàng loạt khi xây dựng tập huấn
luyện, và tính cho một hồ sơ đơn lẻ khi tiếp nhận hồ sơ mới. Nhờ dùng
chung mã nguồn, định nghĩa đặc trưng lúc huấn luyện và lúc suy luận chắc chắn
trùng khớp.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from . import definitions as fd

log = logging.getLogger(__name__)

# Đơn vị tiền tệ nội bộ: triệu VND (giảm độ lớn số học, tăng ổn định số)
MONEY_UNIT = 1_000_000.0

# Nhóm giao dịch được coi là chi tiêu rủi ro cao
RISKY_CATEGORIES = ("GAMBLING_LIKE", "CRYPTO_LIKE")

# Ngưỡng giao dịch giá trị lớn (VND)
LARGE_TXN_THRESHOLD = 50_000_000.0


# ===========================================================================
# 1. Luỹ kế các độ đo giao dịch
# ===========================================================================
def _transaction_measures(txn: pd.DataFrame) -> pd.DataFrame:
    """Chuyển mỗi giao dịch thành vector các độ đo cần luỹ kế."""
    amount = txn["amount"].to_numpy(dtype="float64") / MONEY_UNIT
    is_cr = (txn["direction"].to_numpy() == "CR")
    cat = txn["category"].to_numpy()
    ch = txn["channel"].to_numpy()
    hour = txn["txn_time"].dt.hour.to_numpy()

    is_salary = cat == "SALARY"
    is_cash_wd = cat == "CASH_WITHDRAWAL"

    out = pd.DataFrame(
        {
            "customer_id": txn["customer_id"].to_numpy(),
            "txn_time": txn["txn_time"].to_numpy(),
            "m_cnt": np.ones(len(txn), dtype="float64"),
            "m_amt": amount,
            "m_amt_sq": amount * amount,
            "m_inflow": np.where(is_cr, amount, 0.0),
            "m_outflow": np.where(~is_cr, amount, 0.0),
            "m_salary_cnt": is_salary.astype("float64"),
            "m_salary_amt": np.where(is_salary, amount, 0.0),
            "m_repay_amt": np.where(cat == "LOAN_REPAYMENT", amount, 0.0),
            "m_risky_amt": np.where(np.isin(cat, RISKY_CATEGORIES), amount, 0.0),
            "m_cash_wd_cnt": is_cash_wd.astype("float64"),
            "m_cash_wd_amt": np.where(is_cash_wd, amount, 0.0),
            "m_ecom_amt": np.where(cat == "ECOMMERCE", amount, 0.0),
            "m_utility_amt": np.where(np.isin(cat, ("UTILITY", "TELCO")), amount, 0.0),
            "m_night_cnt": ((hour >= 0) & (hour < 5)).astype("float64"),
            "m_large_cnt": (amount * MONEY_UNIT > LARGE_TXN_THRESHOLD).astype("float64"),
            "m_mobile_cnt": (ch == "MOBILE_BANKING").astype("float64"),
        }
    )
    return out


MEASURE_COLS: List[str] = [
    "m_cnt", "m_amt", "m_amt_sq", "m_inflow", "m_outflow",
    "m_salary_cnt", "m_salary_amt", "m_repay_amt", "m_risky_amt",
    "m_cash_wd_cnt", "m_cash_wd_amt", "m_ecom_amt", "m_utility_amt",
    "m_night_cnt", "m_large_cnt", "m_mobile_cnt",
]


def _cumulative_frame(txn: pd.DataFrame) -> pd.DataFrame:
    """Tính chuỗi luỹ kế theo khách hàng, sắp xếp theo thời gian giao dịch."""
    m = _transaction_measures(txn)
    # Sắp xếp theo (customer_id, txn_time) để luỹ kế trong nhóm
    m = m.sort_values(["customer_id", "txn_time"], kind="mergesort", ignore_index=True)
    cum = m.groupby("customer_id", observed=True, sort=False)[MEASURE_COLS].cumsum()
    cum["customer_id"] = m["customer_id"].to_numpy()
    cum["txn_time"] = m["txn_time"].to_numpy()
    # merge_asof yêu cầu khóa thời gian được sắp xếp toàn cục
    cum = cum.sort_values("txn_time", kind="mergesort", ignore_index=True)
    return cum


def _asof_lookup(
    keys: pd.DataFrame, cum: pd.DataFrame, allow_exact: bool
) -> pd.DataFrame:
    """Lấy giá trị luỹ kế tại mốc thời gian ``keys['t']`` cho từng khách hàng."""
    left = keys.sort_values("t", kind="mergesort")
    merged = pd.merge_asof(
        left,
        cum,
        left_on="t",
        right_on="txn_time",
        by="customer_id",
        direction="backward",
        allow_exact_matches=allow_exact,
    )
    merged = merged.set_index("application_id")
    return merged


def windowed_transaction_features(
    applications: pd.DataFrame, transactions: pd.DataFrame, windows: List[int]
) -> pd.DataFrame:
    """Tính toàn bộ đặc trưng hành vi theo cửa sổ, đúng theo thời điểm."""
    log.info("Tính đặc trưng hành vi theo cửa sổ %s ngày...", windows)
    cum = _cumulative_frame(transactions)

    base = applications[["application_id", "customer_id", "apply_time"]].copy()
    base = base.rename(columns={"apply_time": "t"})

    # C(i, t): giá trị luỹ kế tại thời điểm nộp hồ sơ (bao gồm mốc)
    hi = _asof_lookup(base, cum, allow_exact=True)
    last_txn_time = hi["txn_time"]

    feats: Dict[str, np.ndarray] = {}
    order = applications["application_id"].to_numpy()
    hi = hi.reindex(order)
    hi_vals = {c: np.nan_to_num(hi[c].to_numpy(dtype="float64"), nan=0.0) for c in MEASURE_COLS}

    for w in windows:
        lo_keys = base.copy()
        lo_keys["t"] = lo_keys["t"] - pd.Timedelta(days=w)
        lo = _asof_lookup(lo_keys, cum, allow_exact=False).reindex(order)
        lo_vals = {c: np.nan_to_num(lo[c].to_numpy(dtype="float64"), nan=0.0) for c in MEASURE_COLS}
        s = {c: np.maximum(hi_vals[c] - lo_vals[c], 0.0) for c in MEASURE_COLS}

        cnt = s["m_cnt"]
        cnt_safe = np.maximum(cnt, 1.0)
        inflow = s["m_inflow"]
        outflow = s["m_outflow"]

        feats[f"txn_cnt_{w}d"] = cnt
        feats[f"inflow_{w}d"] = inflow
        feats[f"outflow_{w}d"] = outflow
        feats[f"net_flow_{w}d"] = inflow - outflow
        feats[f"avg_txn_amt_{w}d"] = s["m_amt"] / cnt_safe
        # Phương sai theo công thức E[x^2] - (E[x])^2, chặn dưới tại 0.
        # Với cửa sổ có 0 hoặc 1 giao dịch, độ lệch chuẩn được đặt bằng 0: về mặt
        # thống kê không có biến thiên để đo, còn về mặt số học công thức trên
        # cho ra giá trị rất nhỏ khác 0 do sai số dấu phẩy động - đủ để gây lệch
        # khi so sánh kết quả giữa hai lần tính nếu không xử lý tường minh.
        mean_amt = s["m_amt"] / cnt_safe
        var = np.maximum(s["m_amt_sq"] / cnt_safe - mean_amt**2, 0.0)
        std = np.where(cnt > 1, np.sqrt(var), 0.0)
        feats[f"txn_amt_std_{w}d"] = std
        feats[f"txn_amt_cv_{w}d"] = np.where(mean_amt > 1e-9, std / np.maximum(mean_amt, 1e-9), 0.0)
        feats[f"salary_cnt_{w}d"] = s["m_salary_cnt"]
        feats[f"salary_amt_{w}d"] = s["m_salary_amt"]
        feats[f"repay_amt_{w}d"] = s["m_repay_amt"]
        feats[f"risky_amt_{w}d"] = s["m_risky_amt"]
        feats[f"cash_wd_cnt_{w}d"] = s["m_cash_wd_cnt"]
        feats[f"cash_wd_amt_{w}d"] = s["m_cash_wd_amt"]
        feats[f"ecom_amt_{w}d"] = s["m_ecom_amt"]
        feats[f"utility_amt_{w}d"] = s["m_utility_amt"]
        feats[f"night_txn_cnt_{w}d"] = s["m_night_cnt"]
        feats[f"large_txn_cnt_{w}d"] = s["m_large_cnt"]
        feats[f"mobile_txn_cnt_{w}d"] = s["m_mobile_cnt"]
        feats[f"outflow_ratio_{w}d"] = outflow / np.maximum(inflow, 1e-6)
        feats[f"risky_ratio_{w}d"] = s["m_risky_amt"] / np.maximum(outflow, 1e-6)
        feats[f"cash_ratio_{w}d"] = s["m_cash_wd_amt"] / np.maximum(outflow, 1e-6)
        feats[f"night_ratio_{w}d"] = s["m_night_cnt"] / cnt_safe
        feats[f"repay_burden_{w}d"] = s["m_repay_amt"] / np.maximum(inflow, 1e-6)

    df = pd.DataFrame(feats, index=order)

    # --- đặc trưng hành vi không phụ thuộc cửa sổ --------------------------
    apply_ts = applications["apply_time"].to_numpy()
    last_ts = last_txn_time.reindex(order).to_numpy()
    delta_days = (apply_ts - last_ts) / np.timedelta64(1, "D")
    df["days_since_last_txn"] = np.nan_to_num(delta_days.astype("float64"), nan=999.0)

    def _trend(short: str, long: str, ratio: float) -> np.ndarray:
        s_, l_ = df[short].to_numpy(), df[long].to_numpy()
        return np.log1p(s_) - np.log1p(l_ / ratio)

    if 30 in windows and 90 in windows:
        df["txn_cnt_trend_30_90"] = _trend("txn_cnt_30d", "txn_cnt_90d", 3.0)
        df["inflow_trend_30_90"] = _trend("inflow_30d", "inflow_90d", 3.0)
    if 90 in windows and 180 in windows:
        df["inflow_trend_90_180"] = _trend("inflow_90d", "inflow_180d", 2.0)

    # tính đều đặn của việc nhận lương: kỳ vọng 6 lần trong 180 ngày
    if 180 in windows:
        df["salary_regularity"] = np.clip(df["salary_cnt_180d"].to_numpy() / 6.0, 0.0, 1.5)
        # độ ổn định thu nhập: so sánh dòng tiền vào các cửa sổ lồng nhau
        i30 = df["inflow_30d"].to_numpy()
        i90 = df["inflow_90d"].to_numpy() / 3.0
        i180 = df["inflow_180d"].to_numpy() / 6.0
        monthly = np.vstack([i30, i90, i180])
        mu = monthly.mean(axis=0)
        sd = monthly.std(axis=0)
        df["income_stability"] = np.clip(
            1.0 - np.where(mu > 1e-9, sd / np.maximum(mu, 1e-9), 0.6), 0.0, 1.0
        )

    df.index.name = "application_id"
    return df


# ===========================================================================
# 2. Ma trận đặc trưng đầy đủ
# ===========================================================================
def build_features(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    applications: pd.DataFrame,
    transactions: pd.DataFrame,
    cic_reports: pd.DataFrame,
    windows: List[int],
) -> pd.DataFrame:
    """Ghép nối toàn bộ nguồn dữ liệu thành ma trận đặc trưng cấp hồ sơ."""
    log.info("Xây dựng ma trận đặc trưng cho %s hồ sơ...", f"{len(applications):,}")

    order = applications["application_id"].to_numpy()
    df = pd.DataFrame(
        {
            "application_id": order,
            "customer_id": applications["customer_id"].to_numpy(),
            "apply_time": applications["apply_time"].to_numpy(),
        }
    ).set_index("application_id")

    # --- (a) đặc trưng tĩnh từ khách hàng ---------------------------------
    cust_cols = [
        "age", "gender", "marital_status", "num_dependents", "education_code",
        "occupation", "employer_type", "job_tenure_months", "monthly_income",
        "province", "residence_type", "customer_since_months", "has_salary_account",
    ]
    cust = customers.set_index("customer_id")[cust_cols]
    df = df.join(cust, on="customer_id")

    # --- (b) thông tin tài khoản -------------------------------------------
    acc_agg = accounts.groupby("customer_id", observed=True).agg(
        n_accounts=("account_id", "size"),
        total_avg_balance=("avg_balance_3m", "sum"),
    )
    df = df.join(acc_agg, on="customer_id")
    df["n_accounts"] = df["n_accounts"].fillna(0)
    df["total_avg_balance"] = df["total_avg_balance"].fillna(0.0)

    # --- (c) tham số khoản vay ---------------------------------------------
    loan_cols = [
        "product_code", "requested_amount", "tenor_months", "interest_rate",
        "loan_purpose", "channel", "has_collateral", "monthly_installment", "ltv",
    ]
    loan = applications.set_index("application_id")[loan_cols]
    df = df.join(loan)
    at = applications.set_index("application_id")["apply_time"]
    df["apply_hour"] = at.dt.hour.reindex(df.index).to_numpy()
    df["apply_dayofweek"] = at.dt.dayofweek.reindex(df.index).to_numpy()
    # LTV bằng 0 cho hồ sơ không có tài sản bảo đảm (thay vì NaN)
    df["ltv"] = df["ltv"].fillna(0.0)

    # --- (d) bản tin CIC ----------------------------------------------------
    cic_cols = [f.name for f in fd.CIC_FEATURES if f.name != "cic_missing"]
    cic = cic_reports.set_index("application_id")[cic_cols]
    df = df.join(cic)
    df["cic_missing"] = df["cic_score"].isna().astype(int)

    # --- (e) đặc trưng hành vi theo cửa sổ ---------------------------------
    beh = windowed_transaction_features(applications, transactions, windows)
    df = df.join(beh)

    # --- (f) đặc trưng tổ hợp ----------------------------------------------
    df = add_derived_features(df)

    log.info("  -> ma trận đặc trưng: %s dòng x %s cột", f"{len(df):,}", df.shape[1])
    return df.reset_index()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính các chỉ tiêu nghiệp vụ tổ hợp (dùng chung offline/online)."""
    income = df["monthly_income"].to_numpy(dtype="float64")
    income_safe = np.maximum(income, 1.0)
    installment = df["monthly_installment"].to_numpy(dtype="float64")
    existing = np.nan_to_num(df["monthly_debt_payment"].to_numpy(dtype="float64"), nan=0.0)
    outstanding = np.nan_to_num(df["total_outstanding"].to_numpy(dtype="float64"), nan=0.0)
    deps = df["num_dependents"].to_numpy(dtype="float64")
    age = df["age"].to_numpy(dtype="float64")
    oldest = np.nan_to_num(df["oldest_account_months"].to_numpy(dtype="float64"), nan=0.0)
    n_inst = np.nan_to_num(df["num_institutions"].to_numpy(dtype="float64"), nan=0.0)
    dpd = np.nan_to_num(df["dpd_max_12m"].to_numpy(dtype="float64"), nan=0.0)
    tenure = df["job_tenure_months"].to_numpy(dtype="float64")
    balance = df["total_avg_balance"].to_numpy(dtype="float64")

    df["dti"] = np.clip((installment + existing) / income_safe, 0.0, 5.0)
    df["pti"] = np.clip(installment / income_safe, 0.0, 5.0)
    df["amount_to_income"] = df["requested_amount"].to_numpy(dtype="float64") / income_safe
    df["outstanding_to_income"] = outstanding / income_safe

    # Chi phí sinh hoạt tối thiểu của hộ gia đình.
    #
    # Không dùng một mức cố định: chi phí sinh hoạt vừa có phần "sàn" (mức tối
    # thiểu để duy trì cuộc sống, không phụ thuộc thu nhập) vừa có phần tỷ lệ
    # thuận với thu nhập (mức sống tăng theo thu nhập). Nếu chỉ dùng mức sàn cố
    # định, chỉ tiêu thu nhập còn lại sẽ bị bóp méo ở cả hai đầu: quá khắt khe
    # với nhóm thu nhập thấp và quá dễ dãi với nhóm thu nhập cao.
    living_floor = 2_500_000.0 + 1_500_000.0 * deps
    living_cost = np.maximum(living_floor, 0.25 * income)
    df["disposable_income"] = income - installment - existing - living_cost
    df["disposable_ratio"] = df["disposable_income"].to_numpy() / income_safe
    df["income_per_dependent"] = income / (1.0 + deps)

    df["credit_history_ratio"] = oldest / np.maximum((age - 18.0) * 12.0, 1.0)
    df["debt_per_institution"] = outstanding / np.maximum(n_inst, 1.0)
    df["dpd_x_institutions"] = dpd * n_inst / 10.0
    df["high_dti_short_tenure"] = ((df["dti"].to_numpy() > 0.50) & (tenure < 12)).astype(int)
    df["balance_to_installment"] = balance / np.maximum(installment, 1.0)
    inflow_30 = df["inflow_30d"].to_numpy(dtype="float64") * MONEY_UNIT if "inflow_30d" in df else np.zeros(len(df))
    df["inflow_to_installment"] = inflow_30 / np.maximum(installment, 1.0)
    df["income_verification_ratio"] = inflow_30 / income_safe
    return df


def split_out_of_time(
    df: pd.DataFrame, train_ratio: float, valid_ratio: float, time_col: str = "apply_time"
) -> Dict[str, pd.DataFrame]:
    """Chia dữ liệu theo thời gian (out-of-time validation).

    Chiến lược này mô phỏng đúng điều kiện triển khai thực tế: mô hình được
    huấn luyện trên dữ liệu quá khứ và đánh giá trên dữ liệu tương lai, do đó
    phản ánh cả ảnh hưởng của trôi dạt dữ liệu theo thời gian.
    """
    d = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    n = len(d)
    i_tr = int(n * train_ratio)
    i_va = int(n * (train_ratio + valid_ratio))
    parts = {"train": d.iloc[:i_tr], "valid": d.iloc[i_tr:i_va], "test": d.iloc[i_va:]}
    for k, v in parts.items():
        log.info(
            "  %-5s: %s hồ sơ | %s -> %s",
            k,
            f"{len(v):,}",
            v[time_col].min().date(),
            v[time_col].max().date(),
        )
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


__all__ = [
    "build_features",
    "windowed_transaction_features",
    "add_derived_features",
    "split_out_of_time",
    "MONEY_UNIT",
    "MEASURE_COLS",
]
