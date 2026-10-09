"""Sinh dữ liệu mô phỏng cho bài toán kiểm duyệt tín dụng.

THIẾT KẾ QUÁ TRÌNH SINH DỮ LIỆU
-------------------------------
Dữ liệu được sinh theo một *mô hình biến tiềm ẩn* (latent variable model) nhằm
tái tạo cấu trúc phụ thuộc của dữ liệu tín dụng thực tế thay vì gán nhãn theo
một công thức hiển thị trên chính các đặc trưng đầu vào (điều này sẽ tạo ra
quan hệ tất định và làm sai lệch kết quả đánh giá mô hình).

Cụ thể, với mỗi khách hàng ``i`` tồn tại một biến tiềm ẩn không quan sát được
``z_i`` biểu diễn "chất lượng tín dụng nội tại". Biến này đồng thời chi phối:

  (1) các biến nhân khẩu - kinh tế quan sát được (thu nhập, học vấn, thâm niên);
  (2) các biến lịch sử tín dụng tại CIC (điểm CIC, nhóm nợ, số ngày quá hạn) -
      được sinh như *phép đo có nhiễu* của ``z_i``;
  (3) hành vi giao dịch (tính ổn định dòng tiền, tỷ trọng chi tiêu rủi ro);
  (4) biến mục tiêu ``default_flag`` thông qua một hàm log-odds phi tuyến.

Do CIC và hành vi giao dịch chỉ là phép đo có nhiễu của ``z_i``, mô hình học máy
chỉ có thể phục hồi một phần thông tin về ``z_i``. Điều này tạo ra một *giới hạn
trên* tự nhiên cho năng lực phân biệt (AUC), phù hợp với mức 0,75-0,88 được
báo cáo trong các nghiên cứu thực nghiệm về chấm điểm tín dụng
(Lessmann và cộng sự, 2015; Shi và cộng sự, 2022).

Hàm log-odds được thiết kế có chứa:
  * hiệu ứng ngưỡng (kink) tại DTI = 0,35 và DTI = 0,50;
  * quan hệ phi tuyến dạng chữ U theo tuổi;
  * tương tác giữa DTI và thâm niên công việc, giữa số ngày quá hạn và số tổ
    chức tín dụng đang có quan hệ;
  * trôi dạt khái niệm (concept drift) theo thời gian.

Nhờ đó các mô hình phi tuyến (cây quyết định, rừng ngẫu nhiên, boosting) có
không gian để thể hiện ưu thế so với hồi quy logistic tuyến tính - đúng như
kỳ vọng lý thuyết và là cơ sở cho phần so sánh mô hình ở Chương 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from ..config import Config
from . import reference as ref

log = logging.getLogger(__name__)

# Hệ số của hàm log-odds sinh nhãn vỡ nợ.
# Các hệ số được lựa chọn sao cho: (a) dấu phù hợp với lý thuyết tín dụng,
# (b) độ lớn tạo ra tỷ lệ vỡ nợ và năng lực phân biệt nằm trong khoảng thực tế.
BETA: Dict[str, float] = {
    # Các hệ số dưới đây áp lên số hạng ĐÃ chuẩn hóa phương sai đơn vị, nên
    # ``β`` chính là độ lệch chuẩn đóng góp của yếu tố vào log-odds và tỷ trọng
    # phương sai của yếu tố bằng ``β²/Σβ²``. Bộ hệ số được lựa chọn sao cho:
    #   - nhóm lịch sử tín dụng CIC là nhóm mạnh nhất (phù hợp thực tiễn);
    #   - nhóm hành vi giao dịch có vai trò đáng kể nhưng không áp đảo;
    #   - không yếu tố đơn lẻ nào vượt quá ~12% tổng phương sai tín hiệu, tương
    #     ứng IV của đặc trưng mạnh nhất nằm trong khoảng 0,3-0,6.
    # --- Khối 1: khả năng trả nợ (capacity) ---
    "c_dti_sat": 0.440,        # hiệu ứng bão hòa quanh ngưỡng DTI theo sản phẩm
    "c_income": -0.190,
    "c_disposable": 0.240,     # thu nhập còn lại sau chi phí tối thiểu quá thấp
    "c_deps": 0.120,
    # --- Khối 2: lịch sử tín dụng (history) - chỉ với thick-file ---
    "h_score": -0.400,
    "h_worst": 0.285,
    "h_dpd": 0.300,
    "h_util": 0.245,
    "h_wo": 0.180,
    "h_inq": 0.195,
    "h_oldest": -0.165,
    "h_thin_penalty": 0.150,
    # --- Khối 3: hành vi giao dịch (behaviour) - trọng số cao với thin-file ---
    "v_stability": -0.170,
    "v_risky": 0.220,
    "v_trend": 0.150,
    "v_balance": -0.190,
    "v_burden": 0.180,
    "v_night": 0.130,
    "v_cash": 0.200,
    "v_verify": 0.210,
    # --- Khối 4: cấu trúc khoản vay (structure) ---
    "s_ltv": 0.170,
    "s_amount": 0.190,
    "s_tenor": 0.120,
    "s_product": 0.200,
    # --- Khối 5: nhân khẩu học (demographic) ---
    "d_age": 0.150,            # hệ số của ((age-39)/12)^2 -> quan hệ chữ U
    "d_tenure": 0.170,
    "d_edu": 0.110,
    "d_province": 0.120,
    "d_occupation": 0.140,
    # --- Khối 6: tương tác ---
    "ix_dti_tenure": 0.150,    # DTI>0,5 VÀ thâm niên < 12 tháng
    "ix_dpd_inst": 0.170,
    "ix_unsec_highdti": 0.130, # vay không TSBĐ VÀ DTI > 0,45
    "ix_util_inst": 0.130,     # dùng hết hạn mức tại nhiều TCTD
    "ix_thin_bigloan": 0.150,  # thin-file xin vay lớn so với thu nhập
    "ix_inq_dti": 0.120,       # truy vấn dồn dập kèm DTI cao
    "ix_risky_lowinc": 0.130,  # chi tiêu rủi ro ở nhóm thu nhập thấp
    # --- hiệu chỉnh cường độ tín hiệu ---
    # Tổng đóng góp của các yếu tố quan sát được (phần "tín hiệu") được chuẩn
    # hóa về phương sai đơn vị rồi nhân với ``signal_scale``. Nhờ đó các hệ số
    # ở trên chỉ quy định *trọng số tương đối* giữa các yếu tố, trong khi
    # ``signal_scale`` điều khiển *tỷ lệ tín hiệu trên nhiễu* của toàn bộ bài
    # toán. Giá trị được hiệu chỉnh thực nghiệm để năng lực phân biệt của mô
    # hình tốt nhất đạt AUC ≈ 0,85 - nằm trong khoảng 0,75-0,88 mà các nghiên
    # cứu thực nghiệm về chấm điểm tín dụng báo cáo (Lessmann et al., 2015).
    "signal_scale": 2.00,
    # --- thành phần đặc thù không quan sát được (đặt trần Bayes cho AUC) ---
    "noise_sd": 1.05,
}


@dataclass
class GeneratedData:
    """Tập dữ liệu mô phỏng hoàn chỉnh."""

    customers: pd.DataFrame
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    applications: pd.DataFrame
    cic_reports: pd.DataFrame
    labels: pd.DataFrame
    features: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    def tables(self) -> Dict[str, pd.DataFrame]:
        t = {
            "customers": self.customers,
            "accounts": self.accounts,
            "transactions": self.transactions,
            "applications": self.applications,
            "cic_reports": self.cic_reports,
            "labels": self.labels,
        }
        if self.features is not None:
            t["features"] = self.features
        return t


class CreditDataGenerator:
    """Bộ sinh dữ liệu tín dụng mô phỏng, hoàn toàn vector hóa bằng NumPy."""

    def __init__(self, cfg: Config, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed if seed is not None else cfg.seed)
        g = cfg.get("datagen", {})
        self.n_customers = int(g["n_customers"])
        self.n_applications = int(g["n_applications"])
        self.avg_txn_per_month = float(g["avg_txn_per_month"])
        self.txn_lookback_days = int(g["txn_lookback_days"])
        self.t0 = pd.Timestamp(g["observation_start"])
        self.t1 = pd.Timestamp(g["observation_end"])
        self.window_days = int((self.t1 - self.t0).days)
        self.target_dr = float(g["target_default_rate"])
        self.thin_file_rate = float(g["thin_file_rate"])
        self.missing_income_rate = float(g["missing_income_rate"])
        self.drift = float(g["drift_logodds_per_year"])

    # =====================================================================
    # 1. KHÁCH HÀNG
    # =====================================================================
    def generate_customers(self) -> pd.DataFrame:
        n = self.n_customers
        rng = self.rng
        log.info("Sinh %s khách hàng...", f"{n:,}")

        # --- biến tiềm ẩn chất lượng tín dụng -------------------------------
        z = rng.normal(0.0, 1.0, n)

        # --- tuổi: phân phối lệch phải, tập trung 25-45 ---------------------
        age = np.clip(rng.gamma(shape=7.2, scale=3.4, size=n) + 19, 20, 68).astype(int)

        # --- học vấn: xác suất phụ thuộc z và tuổi --------------------------
        edu_codes = np.array([e[0] for e in ref.EDUCATION])
        edu_base = ref.normalize([e[2] for e in ref.EDUCATION])
        edu_z = np.array([e[4] for e in ref.EDUCATION])
        # logit dịch chuyển theo z: người có z cao thiên về học vấn cao hơn
        edu_logits = np.log(edu_base)[None, :] + 1.15 * np.outer(z, edu_z)
        education = edu_codes[_categorical(edu_logits, rng)]
        edu_income_mult = np.array([e[3] for e in ref.EDUCATION])[education - 1]

        # --- nghề nghiệp ----------------------------------------------------
        occ_names = np.array([o[0] for o in ref.OCCUPATIONS])
        occ_logits = (
            np.log(ref.normalize([o[1] for o in ref.OCCUPATIONS]))[None, :]
            + 0.75 * np.outer(z, np.array([o[4] for o in ref.OCCUPATIONS]))
            + 0.55 * np.outer((education - 3) / 2.0, np.array([o[2] for o in ref.OCCUPATIONS]) - 1.0)
        )
        occ_idx = _categorical(occ_logits, rng)
        occupation = occ_names[occ_idx]
        occ_income_mult = np.array([o[2] for o in ref.OCCUPATIONS])[occ_idx]
        occ_income_vol = np.array([o[3] for o in ref.OCCUPATIONS])[occ_idx]

        # --- loại hình tổ chức làm việc ------------------------------------
        emp_names = np.array([e[0] for e in ref.EMPLOYER_TYPES])
        emp_logits = np.log(ref.normalize([e[1] for e in ref.EMPLOYER_TYPES]))[None, :] + 0.80 * np.outer(
            z, np.array([e[3] for e in ref.EMPLOYER_TYPES])
        )
        emp_idx = _categorical(emp_logits, rng)
        employer_type = emp_names[emp_idx]
        income_stability_base = np.array([e[2] for e in ref.EMPLOYER_TYPES])[emp_idx]

        # --- địa bàn --------------------------------------------------------
        prov_names = np.array([p[0] for p in ref.PROVINCES])
        prov_p = ref.normalize([p[1] for p in ref.PROVINCES])
        prov_idx = rng.choice(len(prov_names), size=n, p=prov_p)
        province = prov_names[prov_idx]
        prov_income_mult = np.array([p[2] for p in ref.PROVINCES])[prov_idx]

        # --- tình trạng hôn nhân (phụ thuộc tuổi) ---------------------------
        mar_names = np.array([m[0] for m in ref.MARITAL_STATUS])
        age_c = (age - 30) / 10.0
        mar_logits = np.log(ref.normalize([m[1] for m in ref.MARITAL_STATUS]))[None, :] + np.outer(
            age_c, np.array([-0.85, 0.95, 0.30, 0.0])
        )
        mar_idx = _categorical(mar_logits, rng)
        marital_status = mar_names[mar_idx]

        # --- hình thức nhà ở -----------------------------------------------
        res_names = np.array([r[0] for r in ref.RESIDENCE_TYPES])
        res_logits = (
            np.log(ref.normalize([r[1] for r in ref.RESIDENCE_TYPES]))[None, :]
            + 0.62 * np.outer(z, np.array([r[2] for r in ref.RESIDENCE_TYPES]))
            + np.outer(age_c, np.array([0.42, -0.10, -0.35, 0.0]))
        )
        residence_type = res_names[_categorical(res_logits, rng)]

        # --- thâm niên công việc -------------------------------------------
        max_tenure = np.maximum((age - 20) * 12, 6)
        job_tenure = np.minimum(
            rng.gamma(shape=1.9, scale=21.0, size=n) * (0.75 + 0.5 * _sig(z)),
            max_tenure,
        )
        job_tenure_months = np.clip(job_tenure, 0, 420).astype(int)

        # --- thu nhập hàng tháng (log-normal có cấu trúc) -------------------
        log_income = (
            np.log(9_600_000)
            + np.log(edu_income_mult)
            + np.log(occ_income_mult)
            + np.log(prov_income_mult)
            + 0.145 * np.log1p(job_tenure_months / 12.0)
            + 0.085 * np.log1p(np.maximum(age - 22, 0) / 10.0)
            + 0.150 * z
            + rng.normal(0, 0.34, n)
        )
        monthly_income = np.round(np.exp(log_income) / 100_000) * 100_000
        monthly_income = np.clip(monthly_income, 4_500_000, 900_000_000)

        # --- số người phụ thuộc --------------------------------------------
        dep_lam = 0.28 + 0.85 * (marital_status == "Đã kết hôn") + 0.030 * np.maximum(age - 26, 0)
        num_dependents = np.minimum(rng.poisson(np.clip(dep_lam, 0.05, 4.0)), 6)

        # --- quan hệ với ngân hàng -----------------------------------------
        customer_since_months = np.clip(
            (rng.gamma(3.0, 14.0, n) * (0.7 + 0.6 * _sig(z))).astype(int), 0, 300
        )
        has_salary_account = (
            rng.random(n) < (0.28 + 0.42 * income_stability_base + 0.10 * _sig(z))
        ).astype(int)

        # --- độ ổn định thu nhập (dùng cho sinh giao dịch) -----------------
        income_volatility = np.clip(
            occ_income_vol * (1.35 - 0.45 * income_stability_base) * np.exp(rng.normal(0, 0.20, n)),
            0.03,
            0.95,
        )

        # --- khách hàng thin-file (ít/không có lịch sử tại CIC) ------------
        thin_p = np.clip(
            self.thin_file_rate * (1.0 + 0.55 * (customer_since_months < 12) - 0.30 * _sig(z)),
            0.005,
            0.60,
        )
        is_thin_file = (rng.random(n) < thin_p).astype(int)

        cust_ids = np.array([f"CUS{i:07d}" for i in range(1, n + 1)])
        created_at = self.t0 - pd.to_timedelta(customer_since_months * 30, unit="D")

        df = pd.DataFrame(
            {
                "customer_id": cust_ids,
                "cif_number": np.array([f"CIF{800_000_000 + i}" for i in range(1, n + 1)]),
                "age": age,
                "gender": np.where(rng.random(n) < 0.517, "M", "F"),
                "marital_status": marital_status,
                "num_dependents": num_dependents,
                "education_code": education,
                "occupation": occupation,
                "employer_type": employer_type,
                "job_tenure_months": job_tenure_months,
                "monthly_income": monthly_income,
                "province": province,
                "residence_type": residence_type,
                "customer_since_months": customer_since_months,
                "has_salary_account": has_salary_account,
                "created_at": created_at,
                "updated_at": created_at,
                # --- các cột nội bộ phục vụ sinh dữ liệu (không dùng làm đặc trưng) ---
                "_z": z,
                "_income_volatility": income_volatility,
                "_income_stability_base": income_stability_base,
                "_is_thin_file": is_thin_file,
            }
        )
        return df

    # =====================================================================
    # 2. TÀI KHOẢN
    # =====================================================================
    def generate_accounts(self, customers: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        n = len(customers)
        log.info("Sinh tài khoản...")

        n_acc = 1 + rng.binomial(2, 0.32 + 0.18 * _sig(customers["_z"].to_numpy()))
        cust_rep = np.repeat(customers["customer_id"].to_numpy(), n_acc)
        z_rep = np.repeat(customers["_z"].to_numpy(), n_acc)
        inc_rep = np.repeat(customers["monthly_income"].to_numpy(), n_acc)
        total = int(n_acc.sum())

        # số dư bình quân tương quan với thu nhập và z
        avg_balance = np.exp(
            np.log(np.maximum(inc_rep, 1.0)) * 0.92
            + 0.42 * z_rep
            + rng.normal(-0.55, 0.85, total)
        )
        avg_balance = np.round(np.clip(avg_balance, 0, 6e9), -3)

        acct_type = rng.choice(
            ["CURRENT", "SAVINGS", "PAYROLL", "CARD"], size=total, p=[0.44, 0.28, 0.18, 0.10]
        )
        open_offset = rng.integers(30, 2600, total)
        df = pd.DataFrame(
            {
                "account_id": np.array([f"ACC{i:08d}" for i in range(1, total + 1)]),
                "customer_id": cust_rep,
                "account_type": acct_type,
                "currency": "VND",
                "open_date": self.t0 - pd.to_timedelta(open_offset, unit="D"),
                "avg_balance_3m": avg_balance,
                "current_balance": np.round(
                    np.maximum(avg_balance * rng.lognormal(0, 0.38, total), 0), -3
                ),
                "status": rng.choice(["ACTIVE", "DORMANT"], size=total, p=[0.955, 0.045]),
            }
        )
        return df

    # =====================================================================
    # 3. HỒ SƠ VAY
    # =====================================================================
    def generate_applications(self, customers: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        n_app = self.n_applications
        n_cus = len(customers)
        log.info("Sinh %s hồ sơ vay...", f"{n_app:,}")

        # Một số khách hàng nộp nhiều hồ sơ (tối đa 3), khách hàng có z thấp
        # có xu hướng nộp nhiều hồ sơ hơn (hành vi "shopping" tín dụng).
        z = customers["_z"].to_numpy()
        w = np.exp(-0.30 * z)
        w = w / w.sum()
        extra = n_app - n_cus
        if extra > 0:
            picks = rng.choice(n_cus, size=extra, p=w, replace=True)
            cust_idx = np.concatenate([np.arange(n_cus), picks])
        else:
            cust_idx = rng.choice(n_cus, size=n_app, replace=False)
        rng.shuffle(cust_idx)
        cust = customers.iloc[cust_idx].reset_index(drop=True)

        # --- thời điểm nộp hồ sơ: có tính mùa vụ theo ngày trong tuần và giờ ---
        day_offset = rng.integers(0, self.window_days, n_app)
        base_date = self.t0 + pd.to_timedelta(day_offset, unit="D")
        dow = base_date.dayofweek.to_numpy()
        # loại bỏ mẫu vào cuối tuần với xác suất cao (nghiệp vụ ít phát sinh)
        keep = ~(((dow == 5) & (rng.random(n_app) < 0.55)) | ((dow == 6) & (rng.random(n_app) < 0.72)))
        shift = np.where(keep, 0, rng.integers(1, 3, n_app))
        base_date = base_date + pd.to_timedelta(shift, unit="D")
        base_date = base_date.where(base_date <= self.t1, self.t1)

        # giờ nộp: hỗn hợp 3 đỉnh (sáng, chiều, tối)
        comp = rng.choice(3, size=n_app, p=[0.34, 0.36, 0.30])
        hour_mu = np.array([9.8, 14.6, 20.9])[comp]
        hour_sd = np.array([1.35, 1.55, 1.25])[comp]
        hour = np.clip(rng.normal(hour_mu, hour_sd), 0, 23.999)
        apply_time = base_date + pd.to_timedelta(hour * 3600, unit="s")

        # --- sản phẩm ------------------------------------------------------
        prod_p = ref.normalize([p["weight"] for p in ref.PRODUCTS])
        income = cust["monthly_income"].to_numpy()
        # khách thu nhập cao thiên về sản phẩm giá trị lớn
        inc_rank = (np.argsort(np.argsort(income)) / max(n_app - 1, 1) - 0.5) * 2.0
        prod_bias = np.array([0.0, -0.15, 0.55, 0.95, 0.35, -0.75])
        prod_logits = np.log(prod_p)[None, :] + np.outer(inc_rank, prod_bias)
        p_idx = _categorical(prod_logits, rng)

        prod_code = np.array([p["code"] for p in ref.PRODUCTS])[p_idx]
        prod_median = np.array([p["median_amount"] for p in ref.PRODUCTS])[p_idx]
        prod_inc_mult = np.array([p["income_multiple"] for p in ref.PRODUCTS])[p_idx]
        prod_rate = np.array([p["rate"] for p in ref.PRODUCTS])[p_idx]
        prod_secured = np.array([p["secured"] for p in ref.PRODUCTS])[p_idx]
        prod_risk = np.array([p["risk_offset"] for p in ref.PRODUCTS])[p_idx]

        # --- số tiền đề nghị vay -------------------------------------------
        anchor = np.sqrt(prod_median * np.maximum(income * prod_inc_mult, 1.0))
        requested_amount = anchor * rng.lognormal(0.0, 0.46, n_app)
        requested_amount = np.round(np.clip(requested_amount, 5_000_000, 12_000_000_000), -6)

        # --- kỳ hạn --------------------------------------------------------
        tenor = np.zeros(n_app, dtype=int)
        for k, p in enumerate(ref.PRODUCTS):
            m = p_idx == k
            if m.any():
                tenors = np.array(p["tenors"])
                # kỳ hạn dài hơn khi số tiền lớn so với thu nhập
                ratio = requested_amount[m] / np.maximum(income[m], 1.0)
                r = np.clip((np.argsort(np.argsort(ratio)) / max(m.sum() - 1, 1)), 0, 1)
                lg = np.log(ref.normalize(np.ones(len(tenors))))[None, :] + np.outer(
                    (r - 0.5) * 2.4, np.linspace(-0.6, 0.6, len(tenors))
                )
                tenor[m] = tenors[_categorical(lg, rng)]

        # --- lãi suất (rủi ro cơ sở + biến động thị trường theo thời gian) --
        years_elapsed = (apply_time - self.t0).days.to_numpy() / 365.25
        rate_market = 0.004 * np.sin(2 * np.pi * years_elapsed) + 0.0035 * years_elapsed
        interest_rate = np.round(
            np.clip(prod_rate + rate_market + rng.normal(0, 0.010, n_app), 0.055, 0.345), 4
        )

        # --- tài sản bảo đảm ------------------------------------------------
        has_collateral = prod_secured | (rng.random(n_app) < 0.06)
        # LTV mục tiêu theo sản phẩm
        ltv_target = np.where(prod_code == "HOME_EQUITY", 0.62, np.where(prod_code == "AUTO_LOAN", 0.74, 0.58))
        ltv = np.clip(ltv_target * rng.lognormal(0, 0.20, n_app), 0.10, 1.05)
        collateral_value = np.where(has_collateral, np.round(requested_amount / ltv, -6), 0.0)
        ltv_final = np.where(has_collateral, requested_amount / np.maximum(collateral_value, 1.0), np.nan)

        # --- kênh tiếp nhận -------------------------------------------------
        ch_names = np.array([c[0] for c in ref.CHANNELS])
        ch_idx = rng.choice(len(ch_names), size=n_app, p=ref.normalize([c[1] for c in ref.CHANNELS]))
        channel = ch_names[ch_idx]

        purpose_names = np.array([p[0] for p in ref.LOAN_PURPOSES])
        purpose = purpose_names[
            rng.choice(len(purpose_names), size=n_app, p=ref.normalize([p[1] for p in ref.LOAN_PURPOSES]))
        ]

        # --- khoản trả nợ hàng tháng của khoản vay mới (công thức niên kim) --
        installment = _annuity_payment(requested_amount, interest_rate / 12.0, tenor)

        # --- tự cân đối khả năng trả nợ của người vay -----------------------
        # Người đi vay không đề nghị một số tiền bất kỳ: họ tự ước lượng mức trả
        # góp có thể chi trả được. Trong thực tế, khoản trả góp thường chiếm
        # khoảng 15-30% thu nhập hàng tháng. Nếu bỏ qua cơ chế tự chọn lọc này,
        # dữ liệu sinh ra sẽ có tỷ lệ rất lớn hồ sơ vượt quá khả năng trả nợ và
        # bị loại bởi luật ngay từ đầu - không phản ánh đúng cơ cấu hồ sơ mà một
        # tổ chức tín dụng thực sự tiếp nhận.
        #
        # Mức trả góp mục tiêu được lấy từ phân phối log-normal quanh 22% thu
        # nhập. Phần đuôi phải của phân phối vẫn tạo ra những hồ sơ vay quá khả
        # năng - đây chính là nhóm hồ sơ mà Rule Engine cần phát hiện và loại.
        target_pti = np.clip(rng.lognormal(np.log(0.22), 0.42, n_app), 0.04, 1.20)
        affordable_installment = income * target_pti
        scale = np.minimum(1.0, affordable_installment / np.maximum(installment, 1.0))
        requested_amount = np.round(
            np.clip(requested_amount * scale, 5_000_000, 12_000_000_000), -6
        )
        installment = _annuity_payment(requested_amount, interest_rate / 12.0, tenor)

        # giá trị tài sản bảo đảm phải được tính lại theo số tiền vay đã điều chỉnh
        collateral_value = np.where(has_collateral, np.round(requested_amount / ltv, -6), 0.0)
        ltv_final = np.where(
            has_collateral, requested_amount / np.maximum(collateral_value, 1.0), np.nan
        )

        df = pd.DataFrame(
            {
                "application_id": np.array([f"APP{i:08d}" for i in range(1, n_app + 1)]),
                "customer_id": cust["customer_id"].to_numpy(),
                "apply_time": apply_time,
                "product_code": prod_code,
                "requested_amount": requested_amount,
                "tenor_months": tenor,
                "interest_rate": interest_rate,
                "loan_purpose": purpose,
                "channel": channel,
                "has_collateral": has_collateral.astype(int),
                "collateral_value": collateral_value,
                "ltv": np.round(ltv_final, 4),
                "monthly_installment": np.round(installment, -3),
                "_prod_risk_offset": prod_risk,
                "_prod_secured": prod_secured.astype(int),
            }
        )
        df = df.sort_values("apply_time", kind="mergesort").reset_index(drop=True)
        # đánh lại mã hồ sơ theo trật tự thời gian để mô phỏng đúng luồng sự kiện
        df["application_id"] = [f"APP{i:08d}" for i in range(1, len(df) + 1)]
        return df

    # =====================================================================
    # 4. GIAO DỊCH
    # =====================================================================
    def generate_transactions(
        self, customers: pd.DataFrame, applications: pd.DataFrame, accounts: pd.DataFrame
    ) -> pd.DataFrame:
        """Sinh giao dịch trong cửa sổ lịch sử trước hồ sơ cuối cùng của mỗi KH.

        Cửa sổ được giới hạn ở ``txn_lookback_days`` ngày trước hồ sơ *sớm nhất*
        và kéo dài tới hồ sơ *muộn nhất* của khách hàng, bảo đảm mọi đặc trưng
        cửa sổ (7/30/90/180 ngày) đều có dữ liệu tại thời điểm chấm điểm.
        """
        rng = self.rng
        log.info("Sinh giao dịch...")

        span = applications.groupby("customer_id", observed=True)["apply_time"].agg(["min", "max"])
        cust = customers.set_index("customer_id")
        span = span.join(
            cust[["monthly_income", "_z", "_income_volatility", "_income_stability_base"]], how="inner"
        )
        start = span["min"] - pd.Timedelta(days=self.txn_lookback_days)
        end = span["max"]
        duration_days = np.maximum((end - start).dt.total_seconds().to_numpy() / 86400.0, 30.0)

        # cường độ giao dịch: Gamma (quá phân tán) * hệ số theo thu nhập/z
        n_c = len(span)
        intensity = (
            rng.gamma(3.6, 1.0 / 3.6, n_c)
            * self.avg_txn_per_month
            * (0.72 + 0.55 * _sig(span["_z"].to_numpy()))
            * np.clip((span["monthly_income"].to_numpy() / 12_000_000) ** 0.22, 0.55, 2.4)
        )
        n_txn = rng.poisson(intensity * duration_days / 30.0)
        n_txn = np.clip(n_txn, 3, 2400)
        total = int(n_txn.sum())
        log.info("  -> %s giao dịch", f"{total:,}")

        cust_rep = np.repeat(span.index.to_numpy(), n_txn)
        income_rep = np.repeat(span["monthly_income"].to_numpy(), n_txn)
        z_rep = np.repeat(span["_z"].to_numpy(), n_txn)
        vol_rep = np.repeat(span["_income_volatility"].to_numpy(), n_txn)
        start_rep = np.repeat(start.to_numpy().astype("datetime64[ns]").astype("int64"), n_txn)
        dur_rep = np.repeat(duration_days, n_txn)

        # thời điểm giao dịch: phân bố đều trên cửa sổ, dày hơn về cuối kỳ
        u = rng.beta(1.18, 1.0, total)
        txn_time = (start_rep + (u * dur_rep * 86400.0 * 1e9)).astype("int64").astype("datetime64[ns]")

        # --- nhóm giao dịch: khách z thấp chi nhiều hơn vào nhóm rủi ro -----
        # Quan trọng: hành vi giao dịch được chi phối bởi một biến *riêng*
        # ``z_behaviour`` chỉ tương quan vừa phải với năng lực tín dụng tiềm ẩn
        # (hệ số tương quan ≈ 0,55). Nếu dùng trực tiếp ``z``, việc tổng hợp
        # hàng trăm giao dịch sẽ cho một phép đo gần như chính xác của ``z``,
        # khiến các đặc trưng hành vi có năng lực dự báo cao bất thường - không
        # phản ánh đúng thực tế, nơi dữ liệu giao dịch chỉ là tín hiệu gián tiếp
        # và nhiều nhiễu về khả năng trả nợ.
        z_beh = 0.55 * span["_z"].to_numpy() + rng.normal(0, 0.835, n_c)
        z_beh_rep = np.repeat(z_beh, n_txn)

        cat_names = np.array([c[0] for c in ref.TXN_CATEGORIES])
        cat_p = ref.normalize([c[1] for c in ref.TXN_CATEGORIES])
        risky = np.isin(cat_names, ["GAMBLING_LIKE", "CRYPTO_LIKE", "CASH_WITHDRAWAL"])
        bias = np.where(risky, -0.85, 0.0) + np.where(cat_names == "LOAN_REPAYMENT", -0.35, 0.0)
        cat_logits = np.log(cat_p)[None, :] + np.outer(z_beh_rep, bias)
        c_idx = _categorical(cat_logits, rng, chunk=400_000)
        category = cat_names[c_idx]
        direction = np.array([c[2] for c in ref.TXN_CATEGORIES])[c_idx]
        amt_ratio = np.array([c[3] for c in ref.TXN_CATEGORIES])[c_idx]

        # --- giá trị giao dịch ----------------------------------------------
        is_salary = category == "SALARY"
        amount = income_rep * amt_ratio * rng.lognormal(0.0, 0.62, total)
        # giao dịch lương: biến động theo độ ổn định thu nhập của khách hàng
        amount = np.where(
            is_salary,
            income_rep * np.exp(rng.normal(0, vol_rep * 0.65)),
            amount,
        )
        amount = np.round(np.clip(amount, 5_000, 4e9), -3)

        ch_names = np.array([c[0] for c in ref.TXN_CHANNELS])
        txn_channel = ch_names[
            rng.choice(len(ch_names), size=total, p=ref.normalize([c[1] for c in ref.TXN_CHANNELS]))
        ]

        # gán tài khoản: lấy tài khoản đầu tiên của khách hàng (đơn giản hóa)
        first_acct = accounts.drop_duplicates("customer_id").set_index("customer_id")["account_id"]
        account_id = first_acct.reindex(cust_rep).to_numpy()

        df = pd.DataFrame(
            {
                "customer_id": cust_rep,
                "account_id": account_id,
                "txn_time": txn_time,
                "amount": amount,
                "direction": direction,
                "category": category,
                "channel": txn_channel,
            }
        )

        # --- bổ sung giao dịch nhận lương định kỳ hàng tháng ----------------
        salary_df = self._generate_salary_transactions(span, customers, first_acct)
        if len(salary_df):
            df = pd.concat([df, salary_df], ignore_index=True)
            log.info("  -> bổ sung %s giao dịch nhận lương định kỳ", f"{len(salary_df):,}")

        df = df.sort_values("txn_time", kind="mergesort").reset_index(drop=True)
        df.insert(0, "txn_id", np.arange(1, len(df) + 1, dtype=np.int64))
        return df

    def _generate_salary_transactions(
        self, span: pd.DataFrame, customers: pd.DataFrame, first_acct: pd.Series
    ) -> pd.DataFrame:
        """Sinh giao dịch nhận lương định kỳ hàng tháng.

        Khoản lương được sinh riêng thay vì lấy từ phân phối hỗn hợp các nhóm
        giao dịch, vì hai lý do:

        1. **Tính chu kỳ.** Lương về đều đặn mỗi tháng vào khoảng ngày cố định.
           Nếu lấy mẫu ngẫu nhiên từ phân phối hỗn hợp, tần suất nhận lương sẽ
           không đúng chu kỳ và các đặc trưng về *tính đều đặn của thu nhập* mất
           hoàn toàn ý nghĩa.

        2. **Tính nhất quán với thu nhập khai báo.** Với khách hàng nhận lương
           qua tài khoản, tổng dòng tiền vào phải xấp xỉ thu nhập khai báo. Đây
           chính là cơ sở của đặc trưng ``income_verification_ratio`` và của luật
           phát hiện chênh lệch giữa thu nhập khai báo và dòng tiền thực tế.
           Nếu quan hệ này không được tái tạo đúng, luật sẽ kích hoạt sai trên
           phần lớn hồ sơ.
        """
        rng = self.rng
        cust = customers.set_index("customer_id")
        idx = span.index
        has_salary = cust["has_salary_account"].reindex(idx).fillna(0).to_numpy().astype(bool)
        if not has_salary.any():
            return pd.DataFrame()

        sel = idx[has_salary]
        income = cust["monthly_income"].reindex(sel).to_numpy(dtype="float64")
        vol = cust["_income_volatility"].reindex(sel).to_numpy(dtype="float64")
        start = span["min"].reindex(sel) - pd.Timedelta(days=self.txn_lookback_days)
        end = span["max"].reindex(sel)

        # số kỳ lương của từng khách hàng trong cửa sổ quan sát
        months = np.maximum(
            ((end - start).dt.total_seconds().to_numpy() / (86400.0 * 30.44)).astype(int), 1
        )
        n_per = months
        total = int(n_per.sum())

        cust_rep = np.repeat(sel.to_numpy(), n_per)
        income_rep = np.repeat(income, n_per)
        vol_rep = np.repeat(vol, n_per)
        start_rep = np.repeat(start.to_numpy().astype("datetime64[ns]").astype("int64"), n_per)
        # ngày trả lương cố định theo từng khách hàng (giữa hoặc cuối tháng)
        payday = np.repeat(rng.choice([5, 10, 15, 25], size=len(sel)), n_per)
        k = np.concatenate([np.arange(m) for m in n_per])

        # thời điểm nhận lương: mỗi 30,44 ngày kể từ mốc đầu, cộng ngày trả lương
        offset_ns = ((k * 30.44 + payday) * 86400.0 * 1e9).astype("int64")
        # dao động nhỏ quanh ngày trả lương (cuối tuần, ngày lễ)
        jitter = (rng.normal(0, 0.6, total) * 86400.0 * 1e9).astype("int64")
        ts = (start_rep + offset_ns + jitter).astype("int64").astype("datetime64[ns]")

        # giá trị lương: dao động quanh thu nhập khai báo theo độ ổn định của
        # từng khách hàng (khách có thu nhập biến động nhận lương không đều)
        amount = income_rep * np.exp(rng.normal(-0.02, vol_rep * 0.35))
        amount = np.round(np.clip(amount, 1_000_000, 1.5e9), -3)

        out = pd.DataFrame(
            {
                "customer_id": cust_rep,
                "account_id": first_acct.reindex(cust_rep).to_numpy(),
                "txn_time": ts,
                "amount": amount,
                "direction": "CR",
                "category": "SALARY",
                "channel": "NAPAS_247",
            }
        )
        # loại các kỳ lương rơi ngoài cửa sổ quan sát
        end_rep = np.repeat(end.to_numpy().astype("datetime64[ns]"), n_per)
        return out.loc[(out["txn_time"].to_numpy() <= end_rep)].reset_index(drop=True)

    # =====================================================================
    # 5. BẢN TIN CIC
    # =====================================================================
    def generate_cic_reports(
        self, customers: pd.DataFrame, applications: pd.DataFrame
    ) -> pd.DataFrame:
        """Sinh bản tin CIC tại thời điểm nộp hồ sơ.

        Các chỉ tiêu CIC được sinh như *phép đo có nhiễu* của biến tiềm ẩn ``z``.
        Khách hàng thin-file có phần lớn chỉ tiêu bị thiếu (NaN).
        """
        rng = self.rng
        n = len(applications)
        log.info("Sinh bản tin CIC cho %s hồ sơ...", f"{n:,}")

        cust = customers.set_index("customer_id").reindex(applications["customer_id"].to_numpy())
        z = cust["_z"].to_numpy()
        income = cust["monthly_income"].to_numpy()
        thin = cust["_is_thin_file"].to_numpy().astype(bool)
        age = cust["age"].to_numpy()

        # q = phép đo có nhiễu của z. Mức nhiễu này quyết định phần biến thiên
        # của năng lực tín dụng tiềm ẩn mà mô hình KHÔNG thể phục hồi, và do
        # đó đặt giới hạn trên cho năng lực phân biệt của mọi mô hình.
        q = z + rng.normal(0, 0.78, n)

        # --- điểm CIC (thang 300-850) ---------------------------------------
        cic_score = np.clip(np.round(585 + 86.0 * q + rng.normal(0, 26.0, n)), 300, 850)

        # --- số tổ chức tín dụng đang có quan hệ ----------------------------
        inst_lam = np.clip(1.55 - 0.42 * q + 0.28 * np.log1p(income / 12_000_000), 0.10, 9.0)
        num_institutions = np.minimum(rng.poisson(inst_lam), 12)

        # --- số hợp đồng ----------------------------------------------------
        num_active_contracts = np.minimum(
            num_institutions + rng.poisson(np.clip(0.9 - 0.20 * q, 0.05, 4.0)), 20
        )
        num_closed_contracts = rng.poisson(np.clip(1.4 + 0.30 * q + 0.02 * age, 0.1, 12.0))

        # --- nhóm nợ (Thông tư 11/2021/TT-NHNN) ------------------------------
        # ngưỡng trên thang -q: q thấp -> nhóm nợ xấu hơn
        u = -q + rng.normal(0, 0.30, n)
        worst_group_12m = np.digitize(u, [0.92, 1.62, 2.18, 2.72]) + 1
        # nhóm nợ hiện tại không xấu hơn nhóm xấu nhất 12 tháng
        improve = rng.random(n) < 0.42
        current_group = np.where(improve, np.maximum(worst_group_12m - 1, 1), worst_group_12m)

        # --- số ngày quá hạn tối đa 12 tháng --------------------------------
        dpd_base = np.array([0.0, 22.0, 58.0, 132.0, 260.0])[worst_group_12m - 1]
        dpd_max_12m = np.where(
            worst_group_12m == 1,
            np.where(rng.random(n) < 0.16, rng.integers(1, 10, n), 0),
            np.round(dpd_base * rng.lognormal(0, 0.30, n)),
        )
        dpd_max_12m = np.clip(dpd_max_12m, 0, 720).astype(int)

        # --- dư nợ và hạn mức ----------------------------------------------
        outstanding = np.where(
            num_active_contracts > 0,
            income * np.clip(3.2 - 0.55 * q, 0.2, 14.0) * rng.lognormal(0, 0.72, n),
            0.0,
        )
        total_outstanding = np.round(np.clip(outstanding, 0, 3e10), -3)
        total_credit_limit = np.round(
            np.maximum(total_outstanding * np.clip(rng.lognormal(0.42, 0.34, n), 1.02, 6.0), 1_000_000), -3
        )
        credit_utilization = np.clip(total_outstanding / np.maximum(total_credit_limit, 1.0), 0, 1.0)

        # --- nghĩa vụ trả nợ hàng tháng hiện hữu ---------------------------
        monthly_debt_payment = np.round(
            total_outstanding * np.clip(rng.normal(0.036, 0.010, n), 0.012, 0.085), -3
        )

        # --- số lần truy vấn tín dụng 3 tháng -------------------------------
        inq_lam = np.clip(0.85 - 0.34 * q, 0.05, 7.0)
        num_inquiries_3m = np.minimum(rng.poisson(inq_lam), 15)

        # --- nợ đã xử lý rủi ro / xóa nợ ------------------------------------
        p_wo = _sig(-2.95 - 1.05 * q + 0.16 * (worst_group_12m - 1))
        has_written_off_debt = (rng.random(n) < p_wo).astype(int)

        # --- tuổi quan hệ tín dụng dài nhất ---------------------------------
        oldest_account_months = np.clip(
            (rng.gamma(2.6, 16.0, n) * (0.70 + 0.55 * _sig(q))).astype(int), 0, 360
        )

        # --- số tổ chức đã từng có nợ xấu ------------------------------------
        num_bad_debt_institutions = np.where(
            worst_group_12m >= 3, 1 + rng.binomial(2, 0.30, n), 0
        )

        query_time = applications["apply_time"].to_numpy()
        ttl_h = float(self.cfg.get("cic.cache_ttl_hours", 72))

        df = pd.DataFrame(
            {
                "report_id": np.array([f"CIC{i:08d}" for i in range(1, n + 1)]),
                "application_id": applications["application_id"].to_numpy(),
                "customer_id": applications["customer_id"].to_numpy(),
                "query_time": query_time,
                "valid_until": query_time + np.timedelta64(int(ttl_h * 3600), "s"),
                "cic_score": cic_score,
                "current_group": current_group,
                "worst_group_12m": worst_group_12m,
                "dpd_max_12m": dpd_max_12m,
                "num_institutions": num_institutions,
                "num_active_contracts": num_active_contracts,
                "num_closed_contracts": num_closed_contracts,
                "total_outstanding": total_outstanding,
                "total_credit_limit": total_credit_limit,
                "credit_utilization": np.round(credit_utilization, 4),
                "monthly_debt_payment": monthly_debt_payment,
                "num_inquiries_3m": num_inquiries_3m,
                "has_written_off_debt": has_written_off_debt,
                "oldest_account_months": oldest_account_months,
                "num_bad_debt_institutions": num_bad_debt_institutions,
                "is_thin_file": thin.astype(int),
            }
        )

        # --- áp dụng thiếu dữ liệu cho khách hàng thin-file -----------------
        thin_cols = [
            "cic_score",
            "current_group",
            "worst_group_12m",
            "dpd_max_12m",
            "total_outstanding",
            "total_credit_limit",
            "credit_utilization",
            "monthly_debt_payment",
            "oldest_account_months",
            "num_bad_debt_institutions",
        ]
        for c in thin_cols:
            df[c] = df[c].astype("float64")
            df.loc[thin, c] = np.nan
        df.loc[thin, "num_institutions"] = 0
        df.loc[thin, "num_active_contracts"] = 0
        df.loc[thin, "num_closed_contracts"] = 0
        df.loc[thin, "has_written_off_debt"] = 0

        # --- thiếu dữ liệu ngẫu nhiên (MCAR) ở một số trường ----------------
        for c, rate in [("oldest_account_months", 0.022), ("credit_utilization", 0.014)]:
            m = rng.random(n) < rate
            df.loc[m, c] = np.nan

        # --- cột nội bộ phục vụ sinh nhãn -----------------------------------
        df["_q"] = q
        return df

    # =====================================================================
    # 6. NHÃN VỠ NỢ
    # =====================================================================
    def generate_labels(
        self,
        features: pd.DataFrame,
        customers: pd.DataFrame,
        applications: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sinh nhãn ``default_flag`` từ hàm log-odds phi tuyến.

        Nhãn được sinh từ *ma trận đặc trưng đã tính đúng theo thời điểm* cộng
        với biến tiềm ẩn ``z`` và nhiễu. Nhờ đó bảo đảm nhãn không chứa thông
        tin nằm ngoài phạm vi quan sát của mô hình tại thời điểm chấm điểm, và
        mọi yếu tố chi phối nhãn đều là yếu tố mà mô hình *có thể* quan sát
        (hoặc quan sát gián tiếp qua phép đo có nhiễu).
        """
        log.info("Sinh nhãn vỡ nợ...")
        rng = self.rng
        n = len(applications)
        b = BETA

        f = features.set_index("application_id").reindex(applications["application_id"].to_numpy())
        cust = customers.set_index("customer_id").reindex(applications["customer_id"].to_numpy())
        z = cust["_z"].to_numpy()

        def col(name: str, fill: float = 0.0) -> np.ndarray:
            return np.nan_to_num(f[name].to_numpy(dtype="float64"), nan=fill)

        income = col("monthly_income", 12_000_000.0)
        age = col("age", 35.0)
        tenure = np.minimum(col("job_tenure_months", 24.0), 120.0)
        deps = col("num_dependents", 0.0)

        # --- khả năng trả nợ ------------------------------------------------
        dti = np.clip(col("dti", 0.0), 0.0, 3.0)

        # --- các chỉ tiêu CIC (thin-file -> giá trị trung tính + cờ) --------
        thin = col("is_thin_file", 0.0)
        cic_score = col("cic_score", 585.0)
        worst = col("worst_group_12m", 1.0)
        dpd = np.minimum(col("dpd_max_12m", 0.0), 150.0)
        n_inst = col("num_institutions", 0.0)
        inq = col("num_inquiries_3m", 0.0)
        wo = col("has_written_off_debt", 0.0)
        util = col("credit_utilization", 0.30)
        oldest = col("oldest_account_months", 24.0)

        # --- đặc điểm khoản vay ---------------------------------------------
        ltv = col("ltv", 0.0)
        amt_to_inc = col("amount_to_income", 0.0)
        tenor = col("tenor_months", 12.0)
        unsec = 1.0 - applications["_prod_secured"].to_numpy()

        # --- hành vi (tính từ luồng giao dịch, đúng theo thời điểm) ---------
        stab = col("income_stability", 0.45)
        risky = col("risky_ratio_180d", 0.0)
        burden = np.clip(col("repay_burden_180d", 0.0), 0, 3)
        trend = col("inflow_trend_90_180", 0.0)
        avg_bal = col("total_avg_balance", 0.0)

        # --- trôi dạt theo thời gian ---------------------------------------
        years = (applications["apply_time"] - self.t0).dt.total_seconds().to_numpy() / (365.25 * 86400)

        # ===================================================================
        # PHẦN TÍN HIỆU: hàm rủi ro phi tuyến, phân khúc theo trạng thái dữ liệu
        # ===================================================================
        # Hàm rủi ro được tổ chức thành 6 khối. Điểm khác biệt cơ bản so với
        # một tổ hợp tuyến tính là:
        #   * các hiệu ứng đều *bão hòa* (dạng sigmoid/tanh) thay vì tăng vô hạn;
        #   * ngưỡng rủi ro *khác nhau theo phân khúc sản phẩm* (có/không TSBĐ);
        #   * trọng số của nhóm đặc trưng CIC và nhóm đặc trưng hành vi *đổi chỗ*
        #     giữa khách hàng thick-file và thin-file - đây là hiện tượng đã được
        #     ghi nhận trong thực tiễn: với khách hàng thiếu lịch sử tín dụng,
        #     dữ liệu hành vi giao dịch trở thành nguồn thông tin chủ đạo
        #     (Onay & Öztürk, 2018);
        #   * nhiều tương tác bậc hai và bậc ba giữa các nhóm đặc trưng.
        # Cấu trúc này khiến mô hình tuyến tính chỉ xấp xỉ được một phần quan hệ,
        # tạo dư địa để các mô hình phi tuyến thể hiện ưu thế.

        thick = 1.0 - thin  # khách hàng có đủ lịch sử tín dụng

        # Hệ số nhân theo nhóm sản phẩm cho từng khối rủi ro: cùng một giá trị
        # đặc trưng có mức ảnh hưởng khác nhau tùy sản phẩm tín dụng.
        prod = applications["product_code"].to_numpy()
        w_hist = np.array([ref.PRODUCT_BLOCK_WEIGHTS[p][0] for p in prod])
        w_beh = np.array([ref.PRODUCT_BLOCK_WEIGHTS[p][1] for p in prod])
        w_str = np.array([ref.PRODUCT_BLOCK_WEIGHTS[p][2] for p in prod])

        # Mỗi số hạng được *chuẩn hóa về phương sai đơn vị* trước khi nhân hệ số.
        # Nhờ vậy hệ số trong ``BETA`` mang ý nghĩa trực tiếp là **độ lệch chuẩn
        # đóng góp** của yếu tố đó vào log-odds, và tỷ trọng phương sai của một
        # yếu tố bằng ``β² / Σβ²``. Đây là điều kiện cần để kiểm soát việc không
        # một đặc trưng nào có năng lực dự báo cao bất thường (IV quá lớn) - dấu
        # hiệu mà trong dữ liệu thực tế thường chỉ xuất hiện khi có rò rỉ dữ liệu.
        u = _unit  # hàm chuẩn hóa, viết tắt cho gọn

        # --- Khối 1: khả năng trả nợ (ngưỡng phụ thuộc sản phẩm) ------------
        # Sản phẩm có TSBĐ được chấp nhận mức DTI cao hơn trước khi rủi ro tăng
        dti_threshold = np.where(unsec > 0.5, 0.38, 0.55)
        capacity = (
            b["c_dti_sat"] * u(_sig(7.0 * (dti - dti_threshold)))
            + b["c_income"] * u(np.tanh(np.log(income / 12_000_000) / 1.2))
            + b["c_disposable"] * u(_sig(-4.0 * (col("disposable_ratio", 0.2) - 0.12)))
            + b["c_deps"] * u(np.tanh(deps / 2.0))
        )

        # --- Khối 2: lịch sử tín dụng (chỉ có ý nghĩa với thick-file) -------
        history = thick * w_hist * (
            b["h_score"] * u(np.tanh((cic_score - 580.0) / 72.0))
            + b["h_worst"] * u(np.minimum(worst - 1.0, 3.0))
            + b["h_dpd"] * u(_sig((dpd - 28.0) / 18.0))
            # tỷ lệ sử dụng hạn mức: quan hệ chữ U - dùng quá ít cũng bất lợi
            + b["h_util"] * u(np.tanh(2.4 * util - 1.0) + 0.34 * _sig(-18.0 * (util - 0.06)))
            + b["h_wo"] * u(wo)
            + b["h_inq"] * u(np.tanh(inq / 2.5))
            + b["h_oldest"] * u(np.tanh(oldest / 60.0))
        ) + b["h_thin_penalty"] * u(thin)

        # --- Khối 3: hành vi giao dịch (trọng số tăng mạnh với thin-file) ---
        # Hệ số nhân: 1,0 với thick-file, 2,3 với thin-file, đồng thời điều
        # chỉnh theo nhóm sản phẩm. Chính cơ chế "đổi trọng số theo phân khúc"
        # này tạo ra quan hệ mà mô hình tuyến tính đơn nhất không biểu diễn được.
        behaviour_weight = (1.0 + 1.3 * thin) * w_beh
        behaviour = behaviour_weight * (
            b["v_stability"] * u(np.tanh(2.0 * (stab - 0.5)))
            + b["v_risky"] * u(_sig((risky - 0.012) / 0.022))
            + b["v_trend"] * u(np.tanh(-trend))
            + b["v_balance"] * u(np.tanh(np.log1p(avg_bal / 10_000_000) / 1.5))
            + b["v_burden"] * u(_sig(5.0 * (burden - 0.32)))
            + b["v_night"] * u(np.tanh(col("night_ratio_90d", 0.0) / 0.14))
            + b["v_cash"] * u(np.tanh(col("cash_ratio_90d", 0.0) / 0.45))
            + b["v_verify"] * u(_sig(-3.0 * (col("income_verification_ratio", 1.0) - 0.55)))
        )

        # --- Khối 4: cấu trúc khoản vay -------------------------------------
        structure = w_str * (
            b["s_ltv"] * u(_sig(5.0 * (ltv - 0.78)) * (ltv > 0))
            + b["s_amount"] * u(np.tanh(np.log1p(amt_to_inc) / 2.2))
            + b["s_tenor"] * u(_sig((tenor - 48.0) / 16.0))
        ) + b["s_product"] * u(applications["_prod_risk_offset"].to_numpy())

        # --- Khối 5: nhân khẩu học và địa bàn (quan hệ phi đơn điệu) --------
        prov = applications["customer_id"].map(
            customers.set_index("customer_id")["province"]
        ).to_numpy()
        occ = applications["customer_id"].map(
            customers.set_index("customer_id")["occupation"]
        ).to_numpy()
        demographic = (
            b["d_age"] * u(((age - 39.0) / 12.0) ** 2)
            + b["d_tenure"] * u(_sig(-(tenure - 18.0) / 10.0))
            + b["d_edu"] * u(np.tanh((3.0 - col("education_code", 3.0)) / 1.5))
            + b["d_province"] * u(np.array([ref.PROVINCE_RISK.get(p, 0.0) for p in prov]))
            + b["d_occupation"] * u(np.array([ref.OCCUPATION_RISK.get(o, 0.0) for o in occ]))
        )

        # --- Khối 6: tương tác giữa các khối (bậc 2 và bậc 3) --------------
        interaction = (
            # gánh nặng nợ cao + thâm niên ngắn: rủi ro cộng hưởng
            b["ix_dti_tenure"] * u(((dti > 0.50) & (tenure < 12)).astype(float))
            # quá hạn nhiều nơi: dấu hiệu khó khăn dòng tiền hệ thống
            + b["ix_dpd_inst"] * u(np.tanh(dpd * n_inst / 180.0))
            # vay tín chấp với DTI cao
            + b["ix_unsec_highdti"] * u((unsec * (dti > 0.45)).astype(float))
            # dùng hết hạn mức tại nhiều tổ chức
            + b["ix_util_inst"] * u(((util > 0.85) & (n_inst >= 4)).astype(float) * thick)
            # thin-file xin vay số tiền lớn so với thu nhập (bậc 3)
            + b["ix_thin_bigloan"] * u(thin * _sig((amt_to_inc - 7.0) / 2.0))
            # tìm kiếm tín dụng dồn dập kèm gánh nặng nợ
            + b["ix_inq_dti"] * u(((inq >= 4) & (dti > 0.40)).astype(float) * thick)
            # chi tiêu rủi ro ở khách hàng thu nhập thấp
            + b["ix_risky_lowinc"] * u((risky > 0.015).astype(float) * _sig(-(income - 14e6) / 6e6))
        )

        signal = capacity + history + behaviour + structure + demographic + interaction

        # ghi lại đóng góp phương sai của từng khối để kiểm tra thiết kế
        self.block_variance_ = {
            "capacity": float(np.std(capacity)),
            "history": float(np.std(history)),
            "behaviour": float(np.std(behaviour)),
            "structure": float(np.std(structure)),
            "demographic": float(np.std(demographic)),
            "interaction": float(np.std(interaction)),
        }

        # Chuẩn hóa phần tín hiệu về phương sai đơn vị rồi áp cường độ mong
        # muốn. Bước này tách biệt hai vấn đề: (i) trọng số tương đối giữa các
        # yếu tố rủi ro - do các hệ số BETA quy định; (ii) tỷ lệ tín hiệu/nhiễu
        # tổng thể - do ``signal_scale`` quy định.
        signal = (signal - signal.mean()) / max(float(signal.std()), 1e-9)

        # -------------------------------------------------------------------
        # Thành phần đặc thù KHÔNG quan sát được
        # -------------------------------------------------------------------
        # Thành phần này được sinh **độc lập** với mọi đặc trưng quan sát được.
        # Lý do thiết kế: nếu dùng trực tiếp biến tiềm ẩn ``z`` (biến đã chi phối
        # việc sinh các chỉ tiêu CIC và hành vi giao dịch) thì các đặc trưng là
        # "phép đo tốt nhất" của ``z`` - trước hết là điểm CIC - sẽ thừa hưởng
        # thêm năng lực dự báo từ số hạng tiềm ẩn, dẫn tới Information Value bị
        # đẩy lên mức bất thường và làm sai lệch kết luận về tầm quan trọng
        # tương đối giữa các nhóm đặc trưng.
        #
        # Vai trò của ``z`` trong thiết kế là tạo ra **cấu trúc tương quan thực
        # tế giữa các đặc trưng** (thu nhập - học vấn - điểm CIC - hành vi giao
        # dịch cùng biến thiên), còn phần biến thiên không thể giải thích của
        # kết quả trả nợ do thành phần độc lập dưới đây đảm nhiệm. Độ lệch chuẩn
        # của nó (``noise_sd``) chính là đại lượng đặt **giới hạn trên lý thuyết**
        # cho năng lực phân biệt của mọi mô hình (trần Bayes).
        idiosyncratic = rng.normal(0.0, b["noise_sd"], n)

        eta = (
            b["signal_scale"] * signal
            # trôi dạt khái niệm theo thời gian
            + self.drift * years
            + idiosyncratic
        )

        # --- hiệu chỉnh hệ số chặn để đạt tỷ lệ vỡ nợ mục tiêu -------------
        intercept = _solve_intercept(eta, self.target_dr)
        pd_true = _sig(eta + intercept)
        default_flag = (rng.random(n) < pd_true).astype(int)

        # --- thời điểm quan sát kết quả (performance window) ----------------
        pw = int(self.cfg.get("datagen.performance_window_months", 12))
        # thời điểm vỡ nợ phân bố trong cửa sổ quan sát, tập trung tháng 4-9
        months_to_default = np.where(
            default_flag == 1, np.clip(rng.gamma(4.2, 1.55, n), 1, pw).round(), np.nan
        )

        out = pd.DataFrame(
            {
                "application_id": applications["application_id"].to_numpy(),
                "customer_id": applications["customer_id"].to_numpy(),
                "observation_time": applications["apply_time"].to_numpy(),
                "performance_end": applications["apply_time"].to_numpy()
                + np.timedelta64(pw * 30, "D"),
                "months_to_default": months_to_default,
                "default_flag": default_flag,
                "_pd_true": np.round(pd_true, 6),
                "_eta": np.round(eta + intercept, 6),
            }
        )
        log.info(
            "  -> tỷ lệ vỡ nợ thực tế: %.3f%% (mục tiêu %.3f%%)",
            100 * default_flag.mean(),
            100 * self.target_dr,
        )
        return out

    # =====================================================================
    # Điều phối
    # =====================================================================
    def run(self) -> GeneratedData:
        """Sinh toàn bộ tập dữ liệu.

        Trình tự bắt buộc: dữ liệu thô -> ma trận đặc trưng (point-in-time)
        -> nhãn. Ma trận đặc trưng được trả về kèm theo để bước huấn luyện
        không phải tính lại.
        """
        from ..features.extraction import build_features  # nhập cục bộ, tránh vòng lặp

        customers = self.generate_customers()
        accounts = self.generate_accounts(customers)
        applications = self.generate_applications(customers)
        transactions = self.generate_transactions(customers, applications, accounts)
        cic = self.generate_cic_reports(customers, applications)

        windows = list(self.cfg.get("features.txn_windows_days", [7, 30, 90, 180]))
        features = build_features(customers, accounts, applications, transactions, cic, windows)
        labels = self.generate_labels(features, customers, applications)

        meta = {
            "n_customers": len(customers),
            "n_accounts": len(accounts),
            "n_transactions": len(transactions),
            "n_applications": len(applications),
            "n_cic_reports": len(cic),
            "default_rate": float(labels["default_flag"].mean()),
            "observation_start": str(self.t0.date()),
            "observation_end": str(self.t1.date()),
            "seed": int(self.cfg.seed),
            "thin_file_rate": float(cic["is_thin_file"].mean()),
            "n_features": int(features.shape[1] - 3),
        }
        return GeneratedData(
            customers=customers,
            accounts=accounts,
            transactions=transactions,
            applications=applications,
            cic_reports=cic,
            labels=labels,
            features=features,
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------
def _sig(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def _unit(x: np.ndarray) -> np.ndarray:
    """Chuẩn hóa một số hạng về trung bình 0, phương sai đơn vị.

    Cho phép hệ số nhân của số hạng mang ý nghĩa "độ lệch chuẩn đóng góp vào
    log-odds", nhờ đó kiểm soát được tỷ trọng phương sai của từng yếu tố rủi ro
    một cách tường minh.
    """
    x = np.asarray(x, dtype="float64")
    sd = float(x.std())
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def _categorical(logits: np.ndarray, rng: np.random.Generator, chunk: int = 0) -> np.ndarray:
    """Lấy mẫu phân phối phân loại theo hàng của ma trận logits (Gumbel-max).

    Dùng thủ thuật Gumbel-max để lấy mẫu vector hóa cho từng hàng có phân phối
    riêng, thay vì vòng lặp Python. ``chunk`` cho phép xử lý theo lô để giới
    hạn bộ nhớ với ma trận rất lớn.
    """
    n = logits.shape[0]
    if chunk and n > chunk:
        parts = []
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            gum = rng.gumbel(size=(e - s, logits.shape[1]))
            parts.append(np.argmax(logits[s:e] + gum, axis=1))
        return np.concatenate(parts)
    gum = rng.gumbel(size=logits.shape)
    return np.argmax(logits + gum, axis=1)


def _annuity_payment(principal: np.ndarray, monthly_rate: np.ndarray, n_months: np.ndarray) -> np.ndarray:
    """Khoản trả góp hàng tháng theo phương pháp niên kim (annuity)."""
    r = np.asarray(monthly_rate, dtype=float)
    n = np.maximum(np.asarray(n_months, dtype=float), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(r > 1e-12, r / (1.0 - np.power(1.0 + r, -n)), 1.0 / n)
    return principal * factor


def _solve_intercept(eta: np.ndarray, target_rate: float, tol: float = 1e-5) -> float:
    """Tìm hệ số chặn sao cho ``mean(sigmoid(eta + c)) == target_rate``.

    Hàm mục tiêu đơn điệu tăng theo ``c`` nên dùng chia đôi (bisection).
    """
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(_sig(eta + mid)))
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


__all__ = ["CreditDataGenerator", "GeneratedData", "BETA"]
