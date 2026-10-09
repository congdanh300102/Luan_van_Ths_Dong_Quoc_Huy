"""Giao diện trực quan hóa kết quả mô hình hỗ trợ kiểm duyệt tín dụng.

Giao diện gồm sáu tab, bám theo mạch nội dung của đề án:

1. **Tổng quan**       - đề tài, quy trình xử lý, kết quả chính, cấu trúc mã nguồn
2. **Dữ liệu**         - quy mô, chất lượng, phân bố nhãn rủi ro
3. **Đặc trưng**       - Information Value và kết quả lựa chọn đặc trưng
4. **Mô hình**         - so sánh chất lượng dự báo, chi phí tính toán, giải thích
5. **Kiểm duyệt hồ sơ**- ngưỡng phê duyệt, cơ cấu khuyến nghị, chấm điểm thử
6. **Giám sát**        - trôi dạt dữ liệu và chất lượng mô hình theo thời gian

Chạy:  streamlit run src/creditreview/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from creditreview.config import load_config  # noqa: E402

st.set_page_config(
    page_title="Mô hình hỗ trợ kiểm duyệt tín dụng",
    page_icon="📋",
    layout="wide",
)

cfg = load_config()
TABLES = cfg.path("tables")
FIGURES = cfg.path("figures")

# ---------------------------------------------------------------------------
# Nhãn tiếng Việt cho dữ liệu đọc từ CSV
#
# Các file kết quả dùng tên cột ngắn không dấu cho tiện xử lý; giao diện hiển
# thị phải dùng tiếng Việt như bảng biểu trong đề án.
# ---------------------------------------------------------------------------
COLUMN_LABELS = {
    "feature": "Đặc trưng", "group": "Nhóm", "type": "Kiểu",
    "psi": "PSI", "status": "Trạng thái",
    "iv": "IV", "strength": "Mức độ",
    "n_features": "Số đặc trưng", "iv_sum": "Tổng IV",
    "iv_mean": "IV trung bình", "iv_max": "IV lớn nhất",
    "period": "Tháng", "n": "Số hồ sơ",
    "actual_default_rate": "Nợ xấu thực tế",
    "predicted_pd_mean": "PD dự báo trung bình",
    "calibration_gap": "Chênh lệch hiệu chuẩn",
    "auc": "AUC", "ks": "KS", "gini": "Gini",
    "shap_sum": "Tổng |SHAP|", "shap_mean": "|SHAP| trung bình",
    "shap_max": "|SHAP| lớn nhất", "share_pct": "Tỷ trọng đóng góp (%)",
    "importance": "Mức quan trọng", "description": "Mô tả",
    "bin": "Nhóm", "n_bad": "Số hồ sơ xấu", "bad_rate": "Tỷ lệ nợ xấu",
    "pd_mean": "PD trung bình", "pct_bad": "Tỷ trọng nợ xấu",
    "cum_pct_bad": "Nợ xấu tích luỹ", "cum_pct_pop": "Hồ sơ tích luỹ",
    "lift": "Lift", "cum_lift": "Lift tích luỹ",
    "pd_min": "PD nhỏ nhất", "pd_max": "PD lớn nhất",
    "variable": "Đặc trưng", "missing_pct": "Tỷ lệ khuyết (%)",
    "strategy": "Cách xử lý", "outlier_pct": "Tỷ lệ ngoại lệ (%)",
    "p01": "Phân vị 1%", "p99": "Phân vị 99%",
}

GROUP_LABELS = {
    "STATIC": "Thông tin cá nhân",
    "LOAN": "Tham số khoản vay",
    "CIC": "Lịch sử tín dụng (CIC)",
    "BEHAVIOUR": "Hành vi giao dịch",
    "DERIVED": "Biến tổ hợp",
}


# ---------------------------------------------------------------------------
# Tiện ích nạp và hiển thị dữ liệu
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict | None:
    path = TABLES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def vi(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Đổi tên cột sang tiếng Việt và dịch nhãn nhóm đặc trưng."""
    if df is None or df.empty:
        return df
    out = df.rename(columns=COLUMN_LABELS)
    for col in ("Nhóm", "Nhóm đặc trưng"):
        if col in out.columns:
            out[col] = out[col].map(lambda g: GROUP_LABELS.get(str(g), str(g)))
    return out


def show_table(df: pd.DataFrame | None, empty_msg: str = "Chưa có dữ liệu.") -> None:
    df = vi(df)
    if df is None or df.empty:
        st.info(empty_msg)
    else:
        st.dataframe(df, width="stretch", hide_index=True)


def show_figure(name: str, caption: str = "") -> None:
    path = FIGURES / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chưa có đồ thị `{name}` — hãy chạy lại quy trình thực nghiệm.")


# ---------------------------------------------------------------------------
# Đầu trang
# ---------------------------------------------------------------------------
st.title("Ứng dụng mô hình học máy trong hỗ trợ kiểm duyệt tín dụng")
st.caption(
    "Đề án tốt nghiệp Thạc sĩ Kỹ thuật · Đồng Quốc Huy — B25CHHT028 · "
    "Học viện Công nghệ Bưu chính Viễn thông"
)

meta = load_json("experiment_meta.json") or {}
if meta:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Số hồ sơ", f"{meta.get('n_total', 0):,}".replace(",", "."))
    c2.metric("Số đặc trưng", meta.get("n_features", 0))
    c3.metric("Mô hình tốt nhất", meta.get("best_display_name", "—"))
    c4.metric("AUC (kiểm thử)", f"{meta.get('best_test_auc', 0):.4f}")
    c5.metric(
        "Trần Bayes", f"{meta.get('bayes_auc_test', 0):.4f}",
        delta=f"{meta.get('best_test_auc', 0) - meta.get('bayes_auc_test', 0):+.4f}",
        delta_color="off",
    )
else:
    st.warning(
        "Chưa có kết quả thực nghiệm. Hãy chạy `python scripts/run_all.py` trước."
    )

tabs = st.tabs([
    "Tổng quan", "Dữ liệu", "Đặc trưng", "Mô hình", "Kiểm duyệt hồ sơ", "Giám sát"
])


# ===========================================================================
# Tab 1 — Tổng quan dự án
# ===========================================================================
with tabs[0]:
    st.subheader("Quy trình xử lý một hồ sơ")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Segoe UI",
                  fontsize=11, margin="0.18,0.10"];
            edge [color="#555555"];

            hoso   [label="Hồ sơ vay", fillcolor="#e8eef7"];
            tienxl [label="Tiền xử lý\\nlàm sạch dữ liệu", fillcolor="#fdf0e0"];
            dactr  [label="Trích xuất\\nđặc trưng", fillcolor="#fdf0e0"];
            mohinh [label="Mô hình học máy\\n(ước lượng PD)", fillcolor="#e8f0e0"];
            luat   [label="Luật nghiệp vụ", fillcolor="#f7e4e3"];
            kq     [label="Khuyến nghị\\n+ lý do", fillcolor="#f7e4e3"];
            nguoi  [label="Cán bộ tín dụng\\nquyết định", fillcolor="#eeeeee"];

            hoso -> tienxl -> dactr -> mohinh -> kq;
            luat -> kq;
            kq -> nguoi;
        }
        """,
        width="stretch",
    )
    st.caption(
        "Mô hình cung cấp thông tin hỗ trợ; thẩm quyền quyết định cấp tín dụng "
        "thuộc về người hoặc bộ phận có thẩm quyền của tổ chức tín dụng."
    )

    st.subheader("Kết quả chính")
    tc = load_csv("traditional_comparison.csv")
    dm = load_csv("decision_mix.csv")
    if tc is not None and not tc.empty and dm is not None and not dm.empty:
        trad_bad = float(tc.iloc[0]["Nợ xấu nhóm duyệt (%)"])
        prop_bad = float(tc.iloc[1]["Nợ xấu nhóm duyệt (%)"])
        review = float(
            dm.loc[dm["Kết quả"].str.contains("thẩm định", case=False),
                   "Tỷ trọng (%)"].iloc[0]
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Nợ xấu nhóm duyệt", f"{prop_bad:.2f}%",
                  delta=f"{prop_bad - trad_bad:+.2f} đpt so với quy trình chỉ dùng luật")
        k2.metric("Tỷ lệ xử lý tự động", f"{100 - review:.2f}%")
        k3.metric("Số mô hình so sánh", meta.get("n_models_trained", "—"))
        pt = load_csv("processing_time.csv")
        if pt is not None and not pt.empty:
            total = pt[pt["Chặng xử lý"].str.contains("Tổng")]
            if not total.empty:
                k4.metric("Thời gian xử lý p95",
                          f"{float(total['p95 (ms)'].iloc[0]):.0f} ms")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("So sánh với quy trình thẩm định truyền thống")
        show_table(tc)
    with col2:
        st.subheader("Cấu trúc mã nguồn")
        st.code(
            "src/creditreview/\n"
            "  data/            Thu thập, kiểm tra chất lượng dữ liệu\n"
            "  preprocessing/   Khuyết thiếu, ngoại lệ, mã hóa, chia tập\n"
            "  features/        Trích xuất và lựa chọn đặc trưng\n"
            "  labeling/        Gán nhãn rủi ro, điểm cắt phê duyệt\n"
            "  models/          Huấn luyện, đánh giá, giải thích\n"
            "  decision/        Luật nghiệp vụ + bộ máy quyết định\n"
            "  serving/         Dịch vụ REST\n"
            "  monitoring/      Giám sát trôi dạt",
            language="text",
        )


# ===========================================================================
# Tab 2 — Dữ liệu
# ===========================================================================
with tabs[1]:
    st.subheader("Quy mô tập dữ liệu")
    show_table(load_csv("dataset_summary.csv"))

    st.subheader("Kiểm tra chất lượng dữ liệu")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Vi phạm ràng buộc nghiệp vụ**")
        show_table(load_csv("data_validity.csv"))
    with col2:
        st.markdown("**Kiểm chứng cơ chế phát hiện (phép tiêm lỗi)**")
        show_table(load_csv("data_validity_check.csv"))

    st.subheader("Giá trị khuyết thiếu và ngoại lệ")
    col1, col2 = st.columns(2)
    with col1:
        mv = load_csv("missing_values.csv")
        if mv is not None:
            mv = mv[mv["missing_pct"] > 0]
        show_table(mv, "Không có biến nào thiếu dữ liệu.")
    with col2:
        show_table(load_csv("outliers.csv"))

    st.subheader("Nhãn rủi ro")
    col1, col2 = st.columns(2)
    with col1:
        show_table(load_csv("label_distribution.csv"))
        show_table(load_csv("data_split.csv"))
    with col2:
        vintage = load_csv("vintage_analysis.csv")
        if vintage is not None and not vintage.empty:
            st.markdown("**Tỷ lệ nợ xấu tích luỹ theo tháng**")
            st.line_chart(
                vintage.set_index("Tháng kể từ khi cấp tín dụng")[
                    "Tỷ lệ nợ xấu tích luỹ (%)"
                ]
            )
            st.caption(
                "Đường cong đi ngang từ tháng thứ 12 là căn cứ chọn cửa sổ quan sát."
            )

    show_figure("data_overview", "Phân bố các biến chính trong tập dữ liệu")


# ===========================================================================
# Tab 3 — Đặc trưng
# ===========================================================================
with tabs[2]:
    catalogue = load_csv("feature_catalogue.csv")

    st.subheader("Cơ cấu tập đặc trưng")
    if catalogue is not None and not catalogue.empty:
        by_group = (
            catalogue.groupby("Nhóm", observed=True)
            .size().reset_index(name="Số đặc trưng")
            .sort_values("Số đặc trưng", ascending=False)
        )
        by_group["Nhóm"] = by_group["Nhóm"].map(
            lambda g: GROUP_LABELS.get(str(g), str(g))
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(by_group, width="stretch", hide_index=True)
        with col2:
            st.bar_chart(by_group.set_index("Nhóm")["Số đặc trưng"])

    st.subheader("Năng lực dự báo theo Information Value")
    col1, col2 = st.columns(2)
    with col1:
        iv = load_csv("information_value.csv")
        if iv is not None and not iv.empty:
            top_n = st.slider("Số đặc trưng hiển thị", 5, 40, 15)
            cols = [c for c in ["feature", "group", "iv", "strength"] if c in iv.columns]
            show_table(iv.nlargest(top_n, "iv")[cols])
    with col2:
        show_table(load_csv("information_value_by_group.csv"))

    show_figure("iv_ranking", "Xếp hạng đặc trưng theo Information Value")

    st.subheader("Kết quả lựa chọn tập đặc trưng")
    col1, col2 = st.columns(2)
    with col1:
        show_table(load_csv("feature_selection_steps.csv"))
    with col2:
        show_table(load_csv("feature_selection_by_group.csv"))

    st.subheader("Danh mục đặc trưng đầy đủ")
    if catalogue is not None and not catalogue.empty:
        groups = ["Tất cả"] + sorted(catalogue["Nhóm"].unique().tolist())
        chosen = st.selectbox(
            "Lọc theo nhóm", groups,
            format_func=lambda g: GROUP_LABELS.get(str(g), str(g)),
        )
        view = catalogue if chosen == "Tất cả" else catalogue[catalogue["Nhóm"] == chosen]
        show_table(view)


# ===========================================================================
# Tab 4 — Mô hình
# ===========================================================================
with tabs[3]:
    cmp_df = load_csv("model_comparison.csv")

    st.subheader("So sánh chất lượng dự báo")
    if cmp_df is not None and not cmp_df.empty:
        quality_cols = [
            c for c in ["Mô hình", "Nhóm", "AUC", "Gini", "KS", "PR-AUC",
                        "F1", "Brier", "ECE"] if c in cmp_df.columns
        ]
        st.dataframe(cmp_df[quality_cols], width="stretch", hide_index=True)

        chart = cmp_df.set_index("Mô hình")[["AUC"]].copy()
        if meta.get("bayes_auc_test"):
            chart["Trần Bayes"] = meta["bayes_auc_test"]
        st.bar_chart(chart)
        st.caption(
            "Khoảng cách giữa cột AUC và đường trần Bayes là phần thông tin mà "
            "không mô hình nào khai thác được từ dữ liệu quan sát."
        )

    col1, col2 = st.columns(2)
    with col1:
        show_figure("roc_curves", "Đường cong ROC")
    with col2:
        show_figure("calibration", "Đường cong hiệu chuẩn xác suất")

    st.subheader("Tác động nghiệp vụ và chi phí tính toán")
    if cmp_df is not None and not cmp_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            business = [c for c in ["Mô hình", "Ngưỡng", "Tỷ lệ duyệt",
                                    "Nợ xấu sau duyệt", "Chi phí"]
                        if c in cmp_df.columns]
            st.dataframe(cmp_df[business], width="stretch", hide_index=True)
        with col2:
            cost = [c for c in ["Mô hình", "T.gian HL (s)", "Độ trễ p50 (ms)",
                                "Độ trễ p95 (ms)"] if c in cmp_df.columns]
            st.dataframe(cmp_df[cost], width="stretch", hide_index=True)

    show_figure("auc_vs_latency", "Đánh đổi giữa chất lượng và thời gian chấm điểm")

    st.subheader("Giải thích mô hình")
    col1, col2 = st.columns([1, 2])
    with col1:
        show_table(load_csv("shap_group_share.csv"))
    with col2:
        show_figure("shap_group_share", "Đóng góp theo nhóm nguồn dữ liệu")
    show_figure("shap_summary", "Biểu đồ tóm tắt SHAP")

    st.subheader("Tinh chỉnh siêu tham số")
    tuning = load_json("tuning_best.json") or {}
    if tuning:
        rows = [
            {
                "Mô hình": name,
                "AUC (CV)": info.get("cv_auc"),
                "Số lần thử": info.get("n_trials"),
                "Thời gian (phút)": round(info.get("seconds", 0) / 60, 1),
                "Tham số tốt nhất": json.dumps(info.get("best_params", {}),
                                               ensure_ascii=False),
            }
            for name, info in tuning.items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ===========================================================================
# Tab 5 — Kiểm duyệt hồ sơ
# ===========================================================================
with tabs[4]:
    st.subheader("Ngưỡng phê duyệt")
    zones = load_json("cutoff_zones.json") or {}
    if zones:
        z1, z2, z3 = st.columns(3)
        z1.metric("Khuyến nghị duyệt", f"PD ≤ {zones.get('pd_approve_max', 0):.4f}")
        z2.metric("Chuyển thẩm định",
                  f"{zones.get('pd_approve_max', 0):.4f} – {zones.get('pd_reject_min', 0):.4f}")
        z3.metric("Khuyến nghị từ chối", f"PD > {zones.get('pd_reject_min', 0):.4f}")

    col1, col2 = st.columns(2)
    with col1:
        show_table(load_csv("cutoff_criteria.csv"))
    with col2:
        show_table(load_csv("cutoff_sensitivity.csv"))
    show_figure("cutoff_sensitivity", "Đánh đổi giữa tỷ lệ duyệt và tỷ lệ nợ xấu")

    st.subheader("Cơ cấu khuyến nghị của mô hình")
    col1, col2 = st.columns(2)
    with col1:
        show_table(load_csv("decision_mix.csv"))
    with col2:
        show_table(load_csv("processing_time.csv"))
    show_figure("decision_mix", "Cơ cấu khuyến nghị và tỷ lệ nợ xấu thực tế")

    st.subheader("Chấm điểm thử một hồ sơ")
    preds_path = cfg.path("data_processed") / "test_predictions.parquet"
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        app_ids = preds["application_id"].astype(str).head(200).tolist()
        chosen = st.selectbox("Chọn mã hồ sơ trong tập kiểm thử", app_ids)
        row = preds[preds["application_id"].astype(str) == chosen]
        if not row.empty:
            best_key = meta.get("best_model", "")
            p_col = f"p_{best_key}" if f"p_{best_key}" in row.columns else None
            c1, c2, c3 = st.columns(3)
            if p_col:
                pd_value = float(row[p_col].iloc[0])
                approve_max = float(zones.get("pd_approve_max", 0.08))
                reject_min = float(zones.get("pd_reject_min", 0.32))
                if pd_value <= approve_max:
                    rec, icon = "Khuyến nghị duyệt", "✅"
                elif pd_value <= reject_min:
                    rec, icon = "Chuyển thẩm định thủ công", "⚠️"
                else:
                    rec, icon = "Khuyến nghị từ chối", "⛔"
                c1.metric("PD dự báo", f"{pd_value:.4f}")
                c2.metric("Khuyến nghị", f"{icon} {rec}")
            c3.metric("Kết quả thực tế",
                      "Nợ xấu" if int(row["y_true"].iloc[0]) == 1 else "Tốt")
        st.caption(
            "Đối chiếu khuyến nghị với kết quả thực tế cho thấy mô hình đúng hay "
            "sai trên từng hồ sơ cụ thể."
        )
    else:
        st.info("Chưa có dự báo trên tập kiểm thử — hãy chạy bước huấn luyện mô hình.")

    st.subheader("Luật nghiệp vụ và phân hạng rủi ro")
    col1, col2 = st.columns(2)
    with col1:
        show_table(load_csv("rule_hit_statistics.csv"))
    with col2:
        show_table(load_csv("risk_bands.csv"))
    show_figure("rule_hits", "Tần suất vi phạm luật nghiệp vụ")

    st.subheader("Tích hợp dịch vụ REST")
    show_table(load_csv("api_integration.csv"))


# ===========================================================================
# Tab 6 — Giám sát
# ===========================================================================
with tabs[5]:
    st.subheader("Trôi dạt phân phối dữ liệu (PSI)")
    psi = load_csv("drift_psi.csv")
    if psi is not None and not psi.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("PSI lớn nhất", f"{psi['psi'].max():.4f}")
        c2.metric("Số biến cảnh báo", int((psi["status"] == "CẢNH BÁO").sum()))
        c3.metric("Số biến báo động", int((psi["status"] == "BÁO ĐỘNG").sum()))
        st.caption("Ngưỡng cảnh báo PSI = 0,10; ngưỡng báo động PSI = 0,25.")
        show_table(psi.head(20))
    show_figure("drift_psi", "Mức trôi dạt phân phối đặc trưng")

    st.subheader("Chất lượng mô hình theo thời gian")
    perf = load_csv("performance_over_time.csv")
    if perf is not None and not perf.empty:
        show_table(perf)
        chart_cols = [c for c in ["auc", "ks"] if c in perf.columns]
        if chart_cols and "period" in perf.columns:
            st.line_chart(perf.set_index("period")[chart_cols])
    show_figure("performance_over_time", "Diễn biến chất lượng dự báo theo tháng")

st.divider()
st.caption(
    "Toàn bộ số liệu hiển thị được sinh từ `python scripts/run_all.py`. "
    "Mô hình cung cấp thông tin hỗ trợ kiểm duyệt; quyết định cấp tín dụng "
    "cuối cùng thuộc thẩm quyền của tổ chức tín dụng."
)
