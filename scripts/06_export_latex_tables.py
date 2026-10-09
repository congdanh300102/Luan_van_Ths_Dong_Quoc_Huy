"""Bước 6 - Xuất bảng LaTeX từ kết quả thực nghiệm.

Sinh các file ``.tex`` trong ``thesis/tables/`` để chương 3 của luận văn chèn
trực tiếp bằng ``\\input{}``. Cách làm này bảo đảm **số liệu trong luận văn luôn
khớp với kết quả thực nghiệm**: khi chạy lại thực nghiệm, chỉ cần chạy lại script
này là toàn bộ bảng trong luận văn được cập nhật, không phải sửa tay.

Đồng thời sinh file ``thesis/tables/macros_results.tex`` chứa các macro số liệu
then chốt (AUC, KS, độ trễ...) để trích dẫn trong phần diễn giải.

Cách dùng:  py scripts/06_export_latex_tables.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from creditreview.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("export")


# ---------------------------------------------------------------------------
# Tiện ích định dạng
# ---------------------------------------------------------------------------
# Độ dài tối đa của một "từ" (chuỗi không có khoảng trắng) trước khi cần chèn
# điểm ngắt dòng. Tên định danh như ``num_bad_debt_institutions`` dài 25 ký tự,
# vượt chiều rộng mọi cột hợp lý và sẽ gây tràn lề nếu không có chỗ ngắt.
MAX_TOKEN_LEN = 11


def _add_breakpoints(s: str) -> str:
    """Chèn điểm ngắt dòng vào các chuỗi định danh dài.

    Tên biến và tên topic (``customer_id``, ``cdc.public.transactions``) là một
    khối liền không có khoảng trắng nên TeX không thể xuống dòng, gây tràn lề
    trong ô bảng có chiều rộng cố định. Hàm chèn ``\\allowbreak`` ngay sau các
    dấu phân tách tự nhiên (gạch dưới, chấm, gạch nối) để TeX có chỗ ngắt mà
    không cần thêm dấu gạch nối nhân tạo.
    """
    out = []
    for word in s.split(" "):
        if len(word) > MAX_TOKEN_LEN and re.search(r"[\\_.\-]", word):
            # Ngắt sau dấu gạch dưới: luôn hợp lệ với tên định danh
            word = re.sub(r"(\\_)", r"\1\\allowbreak ", word)
            # Ngắt sau dấu chấm/gạch nối, NHƯNG bỏ qua khi nằm giữa hai chữ số.
            # Nếu không loại trừ, các giá trị như ``learning_rate=0.08'' sẽ bị
            # ngắt ngay sau dấu thập phân thành ``0.'' và ``08'' - vừa sai về
            # trình bày vừa gây hiểu nhầm khi đọc số.
            word = re.sub(r"(?<!\d)([.\-])(?!\d)", r"\1\\allowbreak ", word)
        out.append(word)
    return " ".join(out)


def esc(s) -> str:
    """Thoát ký tự đặc biệt của LaTeX và chèn điểm ngắt cho chuỗi dài."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return _add_breakpoints(s)


def _longest_token_cm(values, font_pt: float = 11.86) -> float:
    """Ước lượng chiều rộng của ``từ`` dài nhất trong một cột (đơn vị cm).

    Dùng để bảo đảm cột không bị thu hẹp quá mức khi tự động co bảng: một cột
    hẹp hơn từ dài nhất của nó sẽ luôn tràn lề dù tổng chiều rộng bảng hợp lệ.
    Hệ số 0,52em/ký tự là ước lượng thiên về an toàn cho phông Times New Roman.
    """
    em_cm = font_pt / 28.4527
    longest = 0
    for v in values:
        for word in str(v).split():
            # tính theo đoạn dài nhất giữa các điểm ngắt đã chèn
            for piece in re.split(r"[_.\-]", word):
                longest = max(longest, len(piece))
    return longest * 0.52 * em_cm


def num(v, dec: int = 4) -> str:
    """Định dạng số theo chuẩn Việt Nam (dấu phẩy thập phân)."""
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer() and abs(v) < 1e15):
        return f"{int(v):,}".replace(",", ".")
    return f"{v:.{dec}f}".replace(".", ",")


def pct(v, dec: int = 2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    return f"{100*v:.{dec}f}".replace(".", ",") + r"\%"


# Nhãn tiếng Việt của các nhóm đặc trưng. Mã nguồn dùng tên nhóm viết hoa không
# dấu cho gọn, nhưng bảng biểu trong đề án phải hiển thị bằng tiếng Việt.
GROUP_LABELS = {
    "STATIC": "Thông tin cá nhân",
    "LOAN": "Tham số khoản vay",
    "CIC": "Lịch sử tín dụng (CIC)",
    "BEHAVIOUR": "Hành vi giao dịch",
    "DERIVED": "Biến tổ hợp",
}


def group_label(value) -> str:
    """Chuyển mã nhóm đặc trưng sang nhãn tiếng Việt (giữ nguyên nếu không có)."""
    return GROUP_LABELS.get(str(value).strip(), str(value))


# Chiều rộng vùng chữ của khổ trang: A4 (21cm) trừ lề trái 3,5cm và lề phải 2cm
TEXTWIDTH_CM = 15.5
# Chừa một biên an toàn nhỏ để tránh tràn do làm tròn
SAFE_WIDTH_CM = 15.3


def _count_columns(colspec: str) -> int:
    """Đếm số cột khai báo trong chuỗi định dạng bảng LaTeX."""
    # loại bỏ phần {..} của L{2cm}, C{...}, R{...}, p{...}
    stripped = re.sub(r"\{[^{}]*\}", "", colspec)
    return sum(1 for ch in stripped if ch in "lcrpLCR")


def _fit_colspec(colspec: str, tabcolsep_pt: float = 3.0,
                 min_widths: list[float] | None = None) -> tuple[str, float]:
    """Co chiều rộng cột để bảng vừa khổ chữ.

    Chiều rộng thực tế của một bảng không chỉ bằng tổng chiều rộng các cột: giữa
    hai cột liền nhau LaTeX chèn thêm ``2 × \\tabcolsep``. Với bảng nhiều cột,
    phần khoảng đệm này có thể lên tới vài centimét và là nguyên nhân phổ biến
    nhất gây tràn lề mà khai báo cột nhìn qua vẫn có vẻ hợp lệ.

    Hàm tính tổng chiều rộng thực tế (cột + khoảng đệm); nếu vượt khổ chữ thì thu
    nhỏ toàn bộ chiều rộng cột theo cùng một tỷ lệ, giữ nguyên tỷ lệ tương đối
    giữa các cột.

    Trả về ``(colspec đã điều chỉnh, tabcolsep dùng cho bảng)``.
    """
    widths = [float(w) for w in re.findall(r"\{([\d.]+)cm\}", colspec)]
    if not widths:
        return colspec, tabcolsep_pt

    n_cols = _count_columns(colspec)
    sep_cm = 2 * (tabcolsep_pt / 28.4527) * max(n_cols - 1, 0)  # 1cm = 28,4527pt
    total = sum(widths) + sep_cm

    if total <= SAFE_WIDTH_CM and not min_widths:
        return colspec, tabcolsep_pt

    budget = SAFE_WIDTH_CM - sep_cm
    mins = list(min_widths) if min_widths else [0.0] * len(widths)
    mins = (mins + [0.0] * len(widths))[: len(widths)]

    # Thu nhỏ theo tỷ lệ nhưng không đưa cột nào xuống dưới chiều rộng tối thiểu
    # cần thiết để chứa từ dài nhất của cột đó. Phần không thể thu nhỏ được bù
    # bằng cách thu hẹp thêm các cột còn dư địa.
    new = list(widths)
    for _ in range(12):
        total_now = sum(new)
        if total_now <= budget + 1e-6:
            break
        flexible = [i for i, (w, m) in enumerate(zip(new, mins)) if w > m + 1e-6]
        if not flexible:
            break
        fixed = sum(new[i] for i in range(len(new)) if i not in flexible)
        avail = max(budget - fixed, 0.0)
        cur = sum(new[i] for i in flexible)
        if cur <= 1e-9:
            break
        k = avail / cur
        for i in flexible:
            new[i] = max(mins[i], new[i] * k)

    if abs(sum(new) - sum(widths)) < 1e-6:
        return colspec, tabcolsep_pt

    it = iter(new)
    new_spec = re.sub(r"\{([\d.]+)cm\}", lambda m: "{%.3fcm}" % next(it), colspec)
    final = sum(new) + sep_cm
    log.info("    co bảng: %.2fcm -> %.2fcm%s", total, final,
             "  (vẫn vượt khổ - cần rút gọn nội dung)" if final > TEXTWIDTH_CM else "")
    return new_spec, tabcolsep_pt


def latex_table(
    df: pd.DataFrame, colspec: str, caption: str, label: str,
    formatters: dict | None = None, bold_first_row: bool = False,
    small: bool = True, note: str = "",
) -> str:
    """Sinh mã LaTeX cho một bảng.

    Kiểm tra số cột khai báo có khớp với số cột dữ liệu hay không; nếu lệch sẽ
    báo lỗi ngay thay vì sinh ra mã LaTeX hỏng (lỗi ``Extra alignment tab'')
    chỉ phát hiện được khi biên dịch.
    """
    formatters = formatters or {}
    n_spec, n_data = _count_columns(colspec), len(df.columns)
    if n_spec != n_data:
        raise ValueError(
            f"Bảng '{label}': khai báo {n_spec} cột nhưng dữ liệu có {n_data} cột "
            f"({list(df.columns)})"
        )
    # Tự động co cột cho vừa khổ chữ, tính cả khoảng đệm giữa các cột và bảo
    # đảm mỗi cột vẫn đủ rộng cho từ dài nhất của chính nó
    tabcolsep = 3.0 if n_spec >= 6 else 5.0
    mins = []
    for c in df.columns:
        vals = [str(c)] + [
            (formatters[c](v) if c in formatters else esc(v)) for v in df[c]
        ]
        # Cộng biên an toàn: ước lượng theo số ký tự chưa tính hết bề rộng thực
        # của chữ đậm có dấu tiếng Việt ở dòng tiêu đề.
        mins.append(_longest_token_cm(vals) + 0.22)
    colspec, tabcolsep = _fit_colspec(colspec, tabcolsep, mins)

    # Dùng [!htbp] thay vì [H]: [H] ép bảng đứng đúng vị trí khai báo, nên khi
    # bảng không đủ chỗ ở cuối trang, LaTeX phải sang trang mới và để lại khoảng
    # trắng lớn. Với số lượng bảng và hình của đề án, cách đặt [H] gây lãng phí
    # khoảng tám trang. Dấu ``!'' yêu cầu LaTeX bỏ qua các ràng buộc tỷ lệ float
    # mặc định, còn ``h'' được ưu tiên trước nên bảng vẫn nằm sát phần nội dung
    # tham chiếu tới nó.
    lines = ["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}"]
    if small:
        lines.append("\\small")
    # Thu hẹp khoảng đệm cột với bảng nhiều cột để tránh tràn lề
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep:.1f}pt}}")
    lines.append(f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(f"\\textbf{{{esc(c)}}}" for c in df.columns) + r" \\")
    lines.append("\\midrule")
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for c in df.columns:
            f = formatters.get(c)
            cells.append(f(row[c]) if f else esc(row[c]))
        line = " & ".join(cells)
        if bold_first_row and i == 0:
            line = " & ".join(f"\\textbf{{{c}}}" for c in cells)
        lines.append(line + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note:
        lines.append(f"\n\\vspace{{2pt}}\n{{\\footnotesize\\itshape {note}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def latex_longtable(
    df: pd.DataFrame, colspec: str, caption: str, label: str,
    formatters: dict | None = None, note: str = "",
) -> str:
    """Sinh mã LaTeX cho bảng dài, tự động ngắt sang nhiều trang.

    Dùng cho các bảng phụ lục có hàng trăm dòng, nơi môi trường ``table``
    thông thường sẽ tràn khỏi một trang.

    Giống :func:`latex_table`, chiều rộng cột được co lại cho vừa khổ chữ: tổng
    chiều rộng thực tế còn phải cộng thêm khoảng đệm ``2 × \\tabcolsep`` giữa mỗi
    cặp cột, nên một khai báo nhìn qua vẫn hợp lệ vẫn có thể tràn lề.
    """
    formatters = formatters or {}
    n_spec, n_data = _count_columns(colspec), len(df.columns)
    if n_spec != n_data:
        raise ValueError(
            f"Bảng '{label}': khai báo {n_spec} cột nhưng dữ liệu có {n_data} cột"
        )

    tabcolsep = 4.0
    mins = []
    for c in df.columns:
        vals = [str(c)] + [
            (formatters[c](v) if c in formatters else esc(v)) for v in df[c]
        ]
        mins.append(_longest_token_cm(vals) + 0.10)
    colspec, tabcolsep = _fit_colspec(colspec, tabcolsep, mins)

    head = " & ".join(f"\\textbf{{{esc(c)}}}" for c in df.columns) + r" \\"
    lines = [
        "\\begingroup",
        "\\small",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep:.1f}pt}}",
        f"\\begin{{longtable}}{{@{{}}{colspec}@{{}}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        "\\toprule", head, "\\midrule", "\\endfirsthead",
        "\\multicolumn{%d}{@{}l}{\\small\\textit{(tiếp theo trang trước)}}\\\\" % n_spec,
        "\\toprule", head, "\\midrule", "\\endhead",
        "\\bottomrule",
        "\\multicolumn{%d}{r@{}}{\\small\\textit{(xem tiếp trang sau)}}\\\\" % n_spec,
        "\\endfoot",
        "\\bottomrule", "\\endlastfoot",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            f = formatters.get(c)
            cells.append(f(row[c]) if f else esc(row[c]))
        lines.append(" & ".join(cells) + r" \\")
    lines.append("\\end{longtable}")
    if note:
        lines.append(f"\n{{\\footnotesize\\itshape {note}}}")
    lines.append("\\endgroup")
    return "\n".join(lines) + "\n"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info("  %s", path.name)


# ---------------------------------------------------------------------------
def main() -> int:
    cfg = load_config()
    T = cfg.path("tables")
    OUT = cfg.root / "thesis" / "tables"
    OUT.mkdir(parents=True, exist_ok=True)
    log.info("Xuất bảng LaTeX vào %s", OUT)

    macros: dict[str, str] = {}

    def csv(name):
        p = T / name
        return pd.read_csv(p) if p.exists() else None

    def js(name):
        p = T / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    meta = js("experiment_meta.json") or {}
    zones = js("cutoff_zones.json") or {}
    selected = js("selected_features.json") or {}
    raw_meta_path = cfg.path("data_raw") / "meta.json"
    raw_meta = (
        json.loads(raw_meta_path.read_text(encoding="utf-8"))
        if raw_meta_path.exists() else {}
    )

    # =====================================================================
    # 1. DỮ LIỆU
    # =====================================================================
    ds = csv("dataset_summary.csv")
    if ds is not None:
        d = ds.copy()
        write(OUT / "tab_dataset.tex", latex_table(
            d, "L{3.0cm}R{2.4cm}L{7.6cm}",
            "Quy mô các bảng dữ liệu sử dụng trong đề án", "tab:dataset",
            formatters={"Số bản ghi": lambda v: num(int(v))},
        ))
        macros["NCustomers"] = num(int(ds.iloc[0]["Số bản ghi"]))
        macros["NAccounts"] = num(int(ds.iloc[1]["Số bản ghi"]))
        macros["NTransactions"] = num(int(ds.iloc[2]["Số bản ghi"]))

    # --- kiểm chứng cơ chế phát hiện lỗi dữ liệu -------------------------
    vc = csv("data_validity_check.csv")
    if vc is not None and not vc.empty:
        d = vc[["Mã ràng buộc", "Nội dung ràng buộc", "Số lỗi tiêm vào",
                "Số lỗi phát hiện", "Kết quả"]].copy()
        write(OUT / "tab_validity.tex", latex_table(
            d, "L{3.3cm}L{5.2cm}R{1.7cm}R{1.7cm}L{1.9cm}",
            "Kiểm chứng cơ chế phát hiện dữ liệu không hợp lệ bằng phép tiêm lỗi",
            "tab:validity",
            formatters={"Số lỗi tiêm vào": lambda v: num(int(v)),
                        "Số lỗi phát hiện": lambda v: num(int(v))},
            note="Phép tiêm lỗi được thực hiện trên một bản sao dữ liệu với tỷ lệ "
                 "1\\% mỗi loại lỗi; dữ liệu gốc không bị thay đổi.",
        ))
        macros["ValidityDetected"] = num(
            float(vc["Tỷ lệ phát hiện (%)"].min()), 1) + r"\%"

    va = csv("data_validity.csv")
    if va is not None:
        macros["NConstraints"] = num(int(len(va)))
        macros["NViolations"] = num(int(va["n_violations"].sum()))

    # --- giá trị khuyết ---------------------------------------------------
    mv = csv("missing_values.csv")
    if mv is not None and not mv.empty:
        d = mv[mv["missing_pct"] > 0].copy()
        if not d.empty:
            d = d.rename(columns={
                "variable": "Đặc trưng", "group": "Nhóm",
                "missing_pct": "Tỷ lệ khuyết (%)", "strategy": "Cách xử lý",
            })[["Đặc trưng", "Nhóm", "Tỷ lệ khuyết (%)", "Cách xử lý"]]
            d["Nhóm"] = d["Nhóm"].map(group_label)
            write(OUT / "tab_missing.tex", latex_table(
                d.head(12), "L{4.4cm}L{2.0cm}R{2.4cm}L{5.4cm}",
                "Các đặc trưng có giá trị khuyết và chiến lược xử lý",
                "tab:missing",
                formatters={"Tỷ lệ khuyết (%)": lambda v: num(float(v), 2)},
            ))
            macros["NMissingVars"] = num(int(len(d)))
            macros["MaxMissingPct"] = num(float(d["Tỷ lệ khuyết (%)"].max()), 2) + r"\%"

    # --- ngoại lệ ---------------------------------------------------------
    ou = csv("outliers.csv")
    if ou is not None and not ou.empty:
        d = ou.head(10)[["variable", "outlier_pct", "p01", "p99"]].copy()
        d.columns = ["Đặc trưng", "Tỷ lệ ngoại lệ (%)",
                     "Phân vị 1%", "Phân vị 99%"]
        write(OUT / "tab_outliers.tex", latex_table(
            d, "L{5.2cm}R{3.0cm}R{2.6cm}R{2.6cm}",
            "Các đặc trưng có tỷ lệ giá trị ngoại lệ cao nhất (quy tắc $3\\times$IQR)",
            "tab:outliers",
            formatters={
                "Tỷ lệ ngoại lệ (%)": lambda v: num(float(v), 2),
                "Phân vị 1%": lambda v: num(float(v), 2),
                "Phân vị 99%": lambda v: num(float(v), 2),
            },
            note="Giá trị ngoại lệ được xử lý bằng phép cắt ngọn tại phân vị 1\\% "
                 "và 99\\% học từ tập huấn luyện, không loại bỏ quan sát.",
        ))
        macros["NOutlierVars"] = num(int(len(ou)))

    # =====================================================================
    # 2. NHÃN RỦI RO
    # =====================================================================
    ld = csv("label_distribution.csv")
    if ld is not None:
        write(OUT / "tab_label.tex", latex_table(
            ld, "L{8.0cm}R{2.6cm}R{2.6cm}",
            "Phân bố trạng thái rủi ro sau khi gán nhãn", "tab:label",
            formatters={"Số hồ sơ": lambda v: num(int(v)),
                        "Tỷ trọng (%)": lambda v: num(float(v), 2)},
        ))

    vt = csv("vintage_analysis.csv")
    if vt is not None and not vt.empty:
        d = vt[vt["Tháng kể từ khi cấp tín dụng"].isin([3, 6, 9, 12, 15, 18])].copy()
        write(OUT / "tab_vintage.tex", latex_table(
            d, "R{3.4cm}" + "R{3.5cm}" * (len(d.columns) - 1),
            "Tỷ lệ nợ xấu tích luỹ theo số tháng kể từ khi cấp tín dụng",
            "tab:vintage",
            formatters={
                "Tháng kể từ khi cấp tín dụng": lambda v: num(int(v)),
                "Số hồ sơ phát sinh nợ xấu": lambda v: num(int(v)),
                "Tỷ lệ nợ xấu tích luỹ (%)": lambda v: num(float(v), 2),
                "Tỷ lệ ghi nhận được (%)": lambda v: num(float(v), 1),
            },
            note="Cửa sổ quan sát 12 tháng được chọn tại điểm đường cong tích luỹ "
                 "bắt đầu đi ngang.",
        ))
        at12 = vt[vt["Tháng kể từ khi cấp tín dụng"] == 12]
        if not at12.empty and "Tỷ lệ ghi nhận được (%)" in vt.columns:
            macros["VintageCoverage"] = num(
                float(at12["Tỷ lệ ghi nhận được (%)"].iloc[0]), 1) + r"\%"

    # --- chia tập ---------------------------------------------------------
    if meta:
        rows = [
            {"Tập": "Huấn luyện (train_fit)", "Số hồ sơ": meta.get("n_train_fit"),
             "Tỷ lệ": meta.get("n_train_fit", 0) / max(meta.get("n_total", 1), 1),
             "Vai trò": "Ước lượng tham số mô hình"},
            {"Tập": "Dừng sớm (train_es)", "Số hồ sơ": meta.get("n_train_es"),
             "Tỷ lệ": meta.get("n_train_es", 0) / max(meta.get("n_total", 1), 1),
             "Vai trò": "Xác định số vòng lặp tối ưu"},
            {"Tập": "Kiểm định (valid)", "Số hồ sơ": meta.get("n_valid"),
             "Tỷ lệ": meta.get("n_valid", 0) / max(meta.get("n_total", 1), 1),
             "Vai trò": "Hiệu chuẩn xác suất, chọn ngưỡng"},
            {"Tập": "Kiểm thử (test)", "Số hồ sơ": meta.get("n_test"),
             "Tỷ lệ": meta.get("n_test", 0) / max(meta.get("n_total", 1), 1),
             "Vai trò": "Đánh giá cuối cùng, dùng một lần"},
        ]
        write(OUT / "tab_split.tex", latex_table(
            pd.DataFrame(rows), "L{3.6cm}R{2.0cm}R{1.6cm}L{5.6cm}",
            "Phân chia dữ liệu theo thời gian", "tab:split",
            formatters={"Số hồ sơ": lambda v: num(int(v)),
                        "Tỷ lệ": lambda v: pct(v, 1)},
            note=f"Giai đoạn huấn luyện: {meta.get('train_period', ['',''])[0]} đến "
                 f"{meta.get('train_period', ['',''])[1]}; "
                 f"giai đoạn kiểm thử: {meta.get('test_period', ['',''])[0]} đến "
                 f"{meta.get('test_period', ['',''])[1]}.",
        ))

    # =====================================================================
    # 3. ĐẶC TRƯNG
    # =====================================================================
    fc = csv("feature_catalogue.csv")
    if fc is not None and not fc.empty:
        d = (fc.groupby(["Nhóm", "Nguồn dữ liệu"], observed=True)
               .size().reset_index(name="Số đặc trưng")
               .sort_values("Số đặc trưng", ascending=False))
        d["Nhóm"] = d["Nhóm"].map(group_label)
        write(OUT / "tab_feature_groups.tex", latex_table(
            d, "L{4.2cm}L{4.2cm}R{3.0cm}",
            "Cơ cấu tập đặc trưng theo nhóm và nguồn dữ liệu",
            "tab:feature-groups",
            formatters={"Số đặc trưng": lambda v: num(int(v))},
        ))

    iv = csv("information_value.csv")
    if iv is not None and not iv.empty:
        d = iv.nlargest(15, "iv")[["feature", "group", "iv", "strength"]].copy()
        d.columns = ["Đặc trưng", "Nhóm", "IV", "Mức độ"]
        d["Nhóm"] = d["Nhóm"].map(group_label)
        write(OUT / "tab_iv_top.tex", latex_table(
            d, "L{5.0cm}L{2.4cm}R{1.8cm}L{5.0cm}",
            "Mười lăm đặc trưng có năng lực dự báo cao nhất theo Information Value",
            "tab:iv-top",
            formatters={"IV": lambda v: num(float(v), 3)},
        ))
        macros["MaxIV"] = num(float(iv["iv"].max()), 3)

    ivg = csv("information_value_by_group.csv")
    if ivg is not None and not ivg.empty:
        d = ivg.copy()
        d.columns = ["Nhóm", "Số đặc trưng", "Tổng IV", "IV trung bình", "IV lớn nhất"]
        d["Nhóm"] = d["Nhóm"].map(group_label)
        write(OUT / "tab_iv_group.tex", latex_table(
            d, "L{3.6cm}R{2.6cm}R{2.6cm}R{3.0cm}R{2.8cm}",
            "Information Value tổng hợp theo nhóm nguồn dữ liệu", "tab:iv-group",
            formatters={
                "Số đặc trưng": lambda v: num(int(v)),
                "Tổng IV": lambda v: num(float(v), 3),
                "IV trung bình": lambda v: num(float(v), 3),
                "IV lớn nhất": lambda v: num(float(v), 3),
            },
        ))

    fs = csv("feature_selection_steps.csv")
    if fs is not None and not fs.empty:
        write(OUT / "tab_selection.tex", latex_table(
            fs, "L{7.4cm}R{3.2cm}R{3.2cm}",
            "Kết quả lựa chọn tập đặc trưng qua ba bộ lọc", "tab:selection",
            formatters={"Số đặc trưng": lambda v: num(int(v)),
                        "Đã loại": lambda v: num(int(v))},
        ))
        macros["NFeaturesInitial"] = num(int(fs.iloc[0]["Số đặc trưng"]))
        macros["NFeaturesSelected"] = num(int(fs.iloc[-1]["Số đặc trưng"]))

    fsg = csv("feature_selection_by_group.csv")
    if fsg is not None and not fsg.empty:
        d = fsg[["Nhóm", "Ban đầu", "Sau lựa chọn", "Tỷ lệ giữ lại (%)"]].copy()
        d.columns = ["Nhóm", "Ban đầu", "Sau lựa chọn", "Tỷ lệ giữ lại (%)"]
        d["Nhóm"] = d["Nhóm"].map(group_label)
        write(OUT / "tab_selection_group.tex", latex_table(
            d, "L{4.0cm}R{2.8cm}R{3.0cm}R{3.2cm}",
            "Cơ cấu tập đặc trưng trước và sau lựa chọn", "tab:selection-group",
            formatters={
                "Ban đầu": lambda v: num(int(v)),
                "Sau lựa chọn": lambda v: num(int(v)),
                "Tỷ lệ giữ lại (%)": lambda v: num(float(v), 1),
            },
        ))

    # =====================================================================
    # 4. MÔ HÌNH
    # =====================================================================
    tuning = js("tuning_best.json") or {}
    if tuning:
        rows = []
        for name, info in tuning.items():
            rows.append({
                "Mô hình": name,
                "Số lần thử": info.get("n_trials"),
                "AUC (CV)": info.get("cv_auc"),
                "Thời gian (phút)": round(info.get("seconds", 0) / 60, 1),
            })
        write(OUT / "tab_tuning.tex", latex_table(
            pd.DataFrame(rows), "L{4.6cm}R{2.6cm}R{2.6cm}R{3.4cm}",
            "Kết quả tinh chỉnh siêu tham số bằng tìm kiếm ngẫu nhiên",
            "tab:tuning",
            formatters={
                "Số lần thử": lambda v: num(int(v)),
                "AUC (CV)": lambda v: num(float(v), 4),
                "Thời gian (phút)": lambda v: num(float(v), 1),
            },
            note="Kiểm định chéo theo chuỗi thời gian (TimeSeriesSplit) ba lần khớp "
                 "trên tập huấn luyện.",
        ))

    cmp_df = csv("model_comparison.csv")
    if cmp_df is not None:
        # (a) chất lượng dự báo
        d = cmp_df[["Mô hình", "AUC", "Gini", "KS", "PR-AUC", "F1", "Brier", "ECE"]].copy()
        write(OUT / "tab_model_quality.tex", latex_table(
            d, "L{4.6cm}" + "R{1.35cm}" * 7,
            "So sánh chất lượng dự báo của các mô hình trên tập kiểm thử",
            "tab:model-quality",
            formatters={c: (lambda v: num(float(v), 4))
                        for c in ["AUC", "Gini", "KS", "PR-AUC", "F1", "Brier", "ECE"]},
            bold_first_row=True,
            note=f"Sắp xếp theo ROC-AUC giảm dần. Trần Bayes (tính từ xác suất vỡ nợ "
                 f"thực của quá trình sinh dữ liệu) là "
                 f"{num(meta.get('bayes_auc_test', 0), 4)}.",
        ))

        # (b) tác động nghiệp vụ
        d = cmp_df[["Mô hình", "Ngưỡng", "Tỷ lệ duyệt", "Nợ xấu sau duyệt",
                    "Chi phí", "Accuracy", "Precision", "Recall"]].copy()
        # Tám cột trên khổ A4 khiến các tiêu đề dài như "Precision" không đủ chỗ.
        # Rút gọn cột tên mô hình để nhường chiều rộng cho các cột số.
        write(OUT / "tab_model_business.tex", latex_table(
            d, "L{3.3cm}R{1.5cm}R{1.6cm}R{1.9cm}R{1.5cm}R{1.75cm}R{1.75cm}R{1.6cm}",
            "Tác động nghiệp vụ của các mô hình tại ngưỡng tối ưu chi phí",
            "tab:model-business",
            formatters={
                "Ngưỡng": lambda v: num(float(v), 4),
                "Tỷ lệ duyệt": lambda v: pct(float(v), 1),
                "Nợ xấu sau duyệt": lambda v: pct(float(v), 2),
                "Chi phí": lambda v: num(float(v), 4),
                "Accuracy": lambda v: num(float(v), 4),
                "Precision": lambda v: num(float(v), 4),
                "Recall": lambda v: num(float(v), 4),
            },
            bold_first_row=True,
            note="Ngưỡng được chọn theo hàm chi phí với tỷ lệ "
                 "$c_{FN}:c_{FP} = 12:1$ trên tập kiểm định.",
        ))

        # (c) mức độ quá khớp: chênh lệch AUC giữa tập kiểm định và kiểm thử
        d = cmp_df[["Mô hình", "AUC (valid)", "AUC"]].copy()
        d["Chênh lệch"] = (d["AUC (valid)"] - d["AUC"]).round(4)
        d.columns = ["Mô hình", "AUC (kiểm định)", "AUC (kiểm thử)", "Chênh lệch"]
        write(OUT / "tab_overfit.tex", latex_table(
            d, "L{5.4cm}R{2.9cm}R{2.9cm}R{2.6cm}",
            "Mức độ quá khớp đo bằng chênh lệch AUC giữa hai tập dữ liệu",
            "tab:overfit",
            formatters={c: (lambda v: num(float(v), 4)) for c in d.columns[1:]},
            note="Chênh lệch dương lớn cho thấy mô hình phụ thuộc quá mức vào đặc "
                 "thù của giai đoạn huấn luyện.",
        ))
        gaps = (cmp_df["AUC (valid)"] - cmp_df["AUC"]).abs()
        macros["MaxOverfitGap"] = num(float(gaps.max()), 4)
        macros["MeanOverfitGap"] = num(float(gaps.mean()), 4)

        # (c) chi phí tính toán
        d = cmp_df[["Mô hình", "Nhóm", "T.gian HL (s)", "Độ trễ p50 (ms)",
                    "Độ trễ p95 (ms)", "Số vòng lặp"]].copy()
        d.columns = ["Mô hình", "Nhóm", "Thời gian HL (s)",
                     "Chấm điểm p50 (ms)", "Chấm điểm p95 (ms)", "Số vòng lặp"]
        write(OUT / "tab_model_cost.tex", latex_table(
            d, "L{4.0cm}L{3.0cm}R{2.0cm}R{2.1cm}R{2.1cm}R{1.8cm}",
            "Chi phí huấn luyện và thời gian chấm điểm một hồ sơ",
            "tab:model-cost",
            formatters={
                "Thời gian HL (s)": lambda v: num(float(v), 1),
                "Chấm điểm p50 (ms)": lambda v: num(float(v), 2),
                "Chấm điểm p95 (ms)": lambda v: num(float(v), 2),
                "Số vòng lặp": lambda v: "--" if pd.isna(v) else num(int(v)),
            },
        ))

        # Nhóm macro "Best*" mô tả mô hình ĐƯỢC CHỌN TRIỂN KHAI, không phải mô
        # hình có AUC cao nhất. Hai mô hình này tách nhau khi nhóm dẫn đầu tương
        # đương về chất lượng nhưng chênh lệch lớn về thời gian chấm điểm; khi
        # đó bước huấn luyện chọn mô hình nhanh nhất trong nhóm (xem train.py).
        # Mô hình dẫn đầu về AUC được xuất riêng qua nhóm macro "TopAUC*".
        dep_key = meta.get("best_model")
        dep_rows = cmp_df[cmp_df["model_key"] == dep_key] if dep_key else cmp_df.iloc[0:0]
        best = dep_rows.iloc[0] if not dep_rows.empty else cmp_df.iloc[0]

        top = cmp_df.iloc[0]
        macros["TopAUCModel"] = str(top["Mô hình"])
        macros["TopAUC"] = num(float(top["AUC"]), 4)
        macros["TopAUCLatPFive"] = num(float(top["Độ trễ p95 (ms)"]), 2)
        if "n_tied_models" in meta:
            macros["NTiedModels"] = num(int(meta["n_tied_models"]))
        if "tie_tolerance_auc" in meta:
            macros["TieTolerance"] = num(float(meta["tie_tolerance_auc"]), 3)
        if float(top["Độ trễ p95 (ms)"]) > 0 and float(best["Độ trễ p95 (ms)"]) > 0:
            macros["LatRatioTopDeploy"] = num(
                float(top["Độ trễ p95 (ms)"]) / float(best["Độ trễ p95 (ms)"]), 1)
        macros["GapTopDeploy"] = num(float(top["AUC"]) - float(best["AUC"]), 4)

        macros["BestModel"] = str(best["Mô hình"])
        macros["BestAUC"] = num(float(best["AUC"]), 4)
        macros["BestGini"] = num(float(best["Gini"]), 4)
        macros["BestKS"] = num(float(best["KS"]), 4)
        macros["BestFOne"] = num(float(best["F1"]), 4)
        macros["BestBrier"] = num(float(best["Brier"]), 4)
        macros["BestECE"] = num(float(best["ECE"]), 4)
        macros["BestThreshold"] = num(float(best["Ngưỡng"]), 4)
        macros["BestApprovalRate"] = pct(float(best["Tỷ lệ duyệt"]), 2)
        macros["BestBadRateApproved"] = pct(float(best["Nợ xấu sau duyệt"]), 2)
        macros["BestLatPFive"] = num(float(best["Độ trễ p95 (ms)"]), 2)
        macros["NModels"] = num(int(len(cmp_df)))

        def _auc(key):
            r = cmp_df[cmp_df["model_key"] == key]
            return float(r["AUC"].iloc[0]) if not r.empty else None

        for key, mac in [("logistic_regression", "LRAUC"), ("svm", "SVMAUC"),
                         ("decision_tree", "DTAUC"), ("random_forest", "RFAUC"),
                         ("gradient_boosting", "GBAUC"), ("scorecard_woe", "WOEAUC"),
                         ("xgboost", "XGBAUC"), ("catboost", "CatAUC"),
                         ("lightgbm", "LGBAUC"), ("mlp", "MLPAUC"),
                         ("extra_trees", "ETAUC"),
                         ("stacking_ensemble", "StackAUC")]:
            v = _auc(key)
            if v is not None:
                macros[mac] = num(v, 4)

        def _lat(key):
            r = cmp_df[cmp_df["model_key"] == key]
            return float(r["Độ trễ p95 (ms)"].iloc[0]) if not r.empty else None

        for key, mac in [("xgboost", "XGBLat"), ("catboost", "CatLat"),
                         ("stacking_ensemble", "StackLat")]:
            v = _lat(key)
            if v is not None:
                macros[mac] = num(v, 2)
        lr = _auc("logistic_regression")
        if lr is not None:
            macros["GapBestLR"] = num(float(best["AUC"]) - lr, 4)
        sv = _auc("svm")
        if sv is not None:
            macros["GapBestSVM"] = num(float(best["AUC"]) - sv, 4)

        # Khoảng cách giữa mô hình dẫn đầu và mô hình đứng ngay sau. Con số này
        # cần thiết cho phần bàn luận: ở bài toán này nhóm dẫn đầu sát nhau tới
        # mức chênh lệch nhỏ hơn dao động đo đạc giữa các lần chạy, nên không thể
        # kết luận một thuật toán vượt trội chỉ dựa trên thứ hạng AUC.
        ranked = cmp_df.sort_values("AUC", ascending=False).reset_index(drop=True)
        if len(ranked) >= 2:
            macros["SecondModel"] = str(ranked.loc[1, "Mô hình"])
            macros["SecondAUC"] = num(float(ranked.loc[1, "AUC"]), 4)
            macros["GapTopSecond"] = num(
                float(ranked.loc[0, "AUC"]) - float(ranked.loc[1, "AUC"]), 4)
        svr = cmp_df[cmp_df["model_key"] == "svm"]
        if not svr.empty and "T.gian HL (s)" in svr.columns:
            macros["SVMTrainSec"] = num(float(svr["T.gian HL (s)"].iloc[0]), 1)

    # --- gains -------------------------------------------------------------
    gains = csv("gains_table.csv")
    if gains is not None and not gains.empty:
        # Ba con số được trích dẫn trong phần diễn giải khả năng phân tách rủi
        # ro. Xuất thành macro để chúng tự cập nhật khi mô hình triển khai đổi.
        macros["TopDecileBad"] = pct(float(gains.iloc[0]["pct_bad"]), 1)
        if len(gains) >= 3:
            macros["TopThreeDecileBad"] = pct(float(gains.iloc[2]["cum_pct_bad"]), 1)
        if "BestKS" in macros:
            ks = float(macros["BestKS"].replace(",", "."))
            macros["KSPoints"] = num(100.0 * ks, 1)
        # Bảng gains gốc có 12 cột mang tên biến tiếng Anh. Với khổ trang A4 chỉ
        # nên giữ các cột thực sự được diễn giải trong phần phân tích, và tiêu
        # đề phải là tiếng Việt như mọi bảng khác của đề án.
        keep = {
            "bin": "Nhóm",
            "n": "Số hồ sơ",
            "n_bad": "Số hồ sơ xấu",
            "pd_mean": "PD trung bình",
            "bad_rate": "Tỷ lệ nợ xấu (%)",
            "pct_bad": "Tỷ trọng nợ xấu (%)",
            "cum_pct_bad": "Nợ xấu tích luỹ (%)",
            "lift": "Lift",
            "cum_lift": "Lift tích luỹ",
        }
        cols = [c for c in keep if c in gains.columns]
        d = gains[cols].copy()
        for c in ["bad_rate", "pct_bad", "cum_pct_bad"]:
            if c in d.columns:
                d[c] = d[c] * 100.0
        d.columns = [keep[c] for c in cols]
        write(OUT / "tab_gains.tex", latex_table(
            d, "R{1.3cm}R{1.6cm}R{1.7cm}R{2.0cm}R{2.0cm}R{2.1cm}R{2.1cm}"
               "R{1.3cm}R{1.8cm}"[: None] if len(d.columns) == 9
               else "R{1.6cm}" + "R{1.8cm}" * (len(d.columns) - 1),
            "Phân tích theo thập phân vị điểm rủi ro", "tab:gains",
            formatters={
                "Nhóm": lambda v: num(int(v)),
                "Số hồ sơ": lambda v: num(int(v)),
                "Số hồ sơ xấu": lambda v: num(int(v)),
                "PD trung bình": lambda v: num(float(v), 4),
                "Tỷ lệ nợ xấu (%)": lambda v: num(float(v), 2),
                "Tỷ trọng nợ xấu (%)": lambda v: num(float(v), 2),
                "Nợ xấu tích luỹ (%)": lambda v: num(float(v), 2),
                "Lift": lambda v: num(float(v), 2),
                "Lift tích luỹ": lambda v: num(float(v), 2),
            },
            note="Nhóm 1 gồm 10\\% hồ sơ có điểm rủi ro cao nhất. Lift là tỷ số "
                 "giữa mật độ nợ xấu của nhóm và mật độ nợ xấu trung bình.",
        ))

    # --- SHAP theo nhóm ----------------------------------------------------
    sg = csv("shap_group_share.csv")
    if sg is not None and not sg.empty:
        d = sg.rename(columns={
            "group": "Nhóm đặc trưng",
            "n_features": "Số đặc trưng",
            "shap_sum": "Tổng |SHAP|",
            "shap_mean": "|SHAP| trung bình",
            "shap_max": "|SHAP| lớn nhất",
            "share_pct": "Tỷ trọng đóng góp (%)",
        }).copy()
        if "Nhóm đặc trưng" in d.columns:
            d["Nhóm đặc trưng"] = d["Nhóm đặc trưng"].map(group_label)
        write(OUT / "tab_shap_group.tex", latex_table(
            d, "L{4.2cm}R{2.2cm}R{2.2cm}R{2.6cm}R{2.4cm}R{2.8cm}"
               if len(d.columns) == 6
               else "L{4.2cm}" + "R{2.4cm}" * (len(d.columns) - 1),
            "Tỷ trọng đóng góp vào quyết định của mô hình theo nhóm đặc trưng",
            "tab:shap-group",
            formatters={
                "Số đặc trưng": lambda v: num(int(v)),
                "Tổng |SHAP|": lambda v: num(float(v), 4),
                "|SHAP| trung bình": lambda v: num(float(v), 4),
                "|SHAP| lớn nhất": lambda v: num(float(v), 4),
                "Tỷ trọng đóng góp (%)": lambda v: num(float(v), 2),
            },
            note="Tính bằng tổng giá trị SHAP tuyệt đối trung bình của các đặc "
                 "trưng trong từng nhóm, chuẩn hóa về tổng bằng 100\\%.",
        ))
        share_col = next((c for c in sg.columns if "share" in c.lower()
                          or "%" in c or "trọng" in c.lower()), None)
        group_col = sg.columns[0]
        if share_col is not None:
            for g, mac in [("BEHAVIOUR", "ShapBehaviourPct"), ("CIC", "ShapCICPct"),
                           ("DERIVED", "ShapDerivedPct"), ("STATIC", "ShapStaticPct"),
                           ("LOAN", "ShapLoanPct")]:
                r = sg[sg[group_col].astype(str).str.upper() == g]
                if not r.empty:
                    v = float(r[share_col].iloc[0])
                    macros[mac] = num(v * 100 if v <= 1 else v, 1)

    # --- độ ổn định --------------------------------------------------------
    psi = csv("drift_psi.csv")
    if psi is not None and not psi.empty:
        d = psi.head(10)[["feature", "psi", "status"]].copy()
        d.columns = ["Đặc trưng", "PSI", "Trạng thái"]
        write(OUT / "tab_drift.tex", latex_table(
            d, "L{6.4cm}R{2.6cm}L{4.4cm}",
            "Mười đặc trưng có mức trôi dạt phân phối lớn nhất", "tab:drift",
            formatters={"PSI": lambda v: num(float(v), 4)},
            note="Ngưỡng cảnh báo PSI $= 0{,}10$; ngưỡng báo động PSI $= 0{,}25$.",
        ))
        macros["MaxPSI"] = num(float(psi["psi"].max()), 4)
        macros["NPSIWarn"] = num(int((psi["status"] == "CẢNH BÁO").sum()))
        macros["NPSIAlert"] = num(int((psi["status"] == "BÁO ĐỘNG").sum()))

    pt = csv("performance_over_time.csv")
    if pt is not None and not pt.empty:
        d = pt.rename(columns={
            "period": "Tháng",
            "n": "Số hồ sơ",
            "actual_default_rate": "Nợ xấu thực tế (%)",
            "predicted_pd_mean": "PD dự báo trung bình (%)",
            "calibration_gap": "Chênh lệch (đpt)",
            "auc": "AUC",
            "ks": "KS",
            "gini": "Gini",
        }).copy()
        for c in ["Nợ xấu thực tế (%)", "PD dự báo trung bình (%)",
                  "Chênh lệch (đpt)"]:
            if c in d.columns:
                d[c] = d[c] * 100.0
        write(OUT / "tab_perf_time.tex", latex_table(
            d, "L{1.9cm}R{1.7cm}R{2.3cm}R{2.6cm}R{2.1cm}R{1.5cm}R{1.5cm}R{1.5cm}"
               if len(d.columns) == 8
               else "L{2.6cm}" + "R{2.2cm}" * (len(d.columns) - 1),
            "Chất lượng dự báo của mô hình theo từng tháng của giai đoạn kiểm thử",
            "tab:perf-time",
            formatters={
                "Số hồ sơ": lambda v: num(int(v)),
                "Nợ xấu thực tế (%)": lambda v: num(float(v), 2),
                "PD dự báo trung bình (%)": lambda v: num(float(v), 2),
                "Chênh lệch (đpt)": lambda v: num(float(v), 2),
                "AUC": lambda v: num(float(v), 4),
                "KS": lambda v: num(float(v), 4),
                "Gini": lambda v: num(float(v), 4),
            },
            note="Chênh lệch tính bằng điểm phần trăm giữa mức PD dự báo trung "
                 "bình và tỷ lệ nợ xấu thực tế của tháng.",
        ))

    # =====================================================================
    # 5. HỆ THỐNG HỖ TRỢ KIỂM DUYỆT
    # =====================================================================
    cc = csv("cutoff_criteria.csv")
    if cc is not None and not cc.empty:
        d = cc.copy()
        write(OUT / "tab_cutoff.tex", latex_table(
            d, "L{4.6cm}R{2.0cm}R{2.0cm}R{2.4cm}R{2.4cm}R{2.0cm}",
            "So sánh ba tiêu chí xác định điểm cắt phê duyệt", "tab:cutoff",
            formatters={c: (lambda v: num(v, 2)) for c in d.columns[1:]},
        ))
        try:
            macros["CutoffCostBadRate"] = num(
                float(cc.iloc[0]["Nợ xấu nhóm duyệt (%)"]), 2) + r"\%"
            macros["CutoffRateBadRate"] = num(
                float(cc.iloc[1]["Nợ xấu nhóm duyệt (%)"]), 2) + r"\%"
            macros["CutoffCapBadRate"] = num(
                float(cc.iloc[2]["Nợ xấu nhóm duyệt (%)"]), 2) + r"\%"
        except Exception as exc:
            log.warning("Không trích được macro điểm cắt: %s", exc)

    cs = csv("cutoff_sensitivity.csv")
    if cs is not None and not cs.empty:
        d = cs.copy()
        write(OUT / "tab_cutoff_sens.tex", latex_table(
            d, "R{2.0cm}R{2.2cm}R{2.2cm}R{2.6cm}R{2.4cm}R{2.2cm}",
            "Độ nhạy của kết quả kiểm duyệt theo điểm cắt", "tab:cutoff-sens",
            formatters={c: (lambda v: num(v, 2)) for c in d.columns[1:]},
        ))

    if zones:
        macros["PDApproveMax"] = num(float(zones.get("pd_approve_max", 0)), 4)
        macros["PDRejectMin"] = num(float(zones.get("pd_reject_min", 0)), 4)
        macros["CostOptimalCut"] = num(float(zones.get("cost_optimal", 0)), 4)

    dm = csv("decision_mix.csv")
    if dm is not None and not dm.empty:
        d = dm.copy()
        write(OUT / "tab_decision_mix.tex", latex_table(
            d, "L{6.0cm}R{2.6cm}R{2.6cm}R{3.4cm}",
            "Cơ cấu khuyến nghị của mô hình và tỷ lệ nợ xấu thực tế theo từng nhóm",
            "tab:decision-mix",
            formatters={
                "Số hồ sơ": lambda v: num(int(v)),
                "Tỷ trọng (%)": lambda v: num(float(v), 2),
                "Tỷ lệ nợ xấu thực tế (%)": lambda v: num(float(v), 2),
            },
        ))
        for key, mac, bad_mac in [
            ("duyệt", "ApproveRate", "BadApprove"),
            ("thẩm định", "ReviewRate", "BadReview"),
            ("từ chối", "RejectRate", "BadReject"),
        ]:
            r = d[d["Kết quả"].str.contains(key, case=False)]
            if not r.empty:
                macros[mac] = num(float(r["Tỷ trọng (%)"].iloc[0]), 2) + r"\%"
                macros[bad_mac] = num(
                    float(r["Tỷ lệ nợ xấu thực tế (%)"].iloc[0]), 2) + r"\%"
        rev = d[d["Kết quả"].str.contains("thẩm định", case=False)]
        if not rev.empty:
            macros["AutomationRate"] = num(
                100.0 - float(rev["Tỷ trọng (%)"].iloc[0]), 2) + r"\%"

    ptime = csv("processing_time.csv")
    if ptime is not None and not ptime.empty:
        write(OUT / "tab_processing_time.tex", latex_table(
            ptime, "L{4.6cm}R{2.6cm}R{2.2cm}R{2.2cm}R{2.2cm}",
            "Thời gian xử lý một hồ sơ theo từng chặng", "tab:processing-time",
            formatters={c: (lambda v: num(float(v), 3)) for c in ptime.columns[1:]},
            note="Đo trên một tiến trình đơn, không tính thời gian truyền qua mạng "
                 "và thời gian truy vấn dữ liệu từ hệ thống nguồn.",
        ))
        tot = ptime[ptime["Chặng xử lý"].str.contains("Tổng")]
        if not tot.empty:
            macros["SysMean"] = num(float(tot["Trung bình (ms)"].iloc[0]), 2)
            macros["SysPFifty"] = num(float(tot["p50 (ms)"].iloc[0]), 2)
            macros["SysPNinetyFive"] = num(float(tot["p95 (ms)"].iloc[0]), 2)
            macros["SysPNinetyNine"] = num(float(tot["p99 (ms)"].iloc[0]), 2)

    rh = csv("rule_hit_statistics.csv")
    if rh is not None and not rh.empty:
        cols = [c for c in ["Mã luật", "Tên luật", "Hành động", "Tỷ lệ vi phạm"]
                if c in rh.columns]
        d = rh[cols].head(12).copy()
        write(OUT / "tab_rule_hits.tex", latex_table(
            d, "L{2.8cm}L{6.0cm}L{2.6cm}R{2.6cm}",
            "Tần suất vi phạm các luật nghiệp vụ", "tab:rule-hits",
            formatters={"Tỷ lệ vi phạm": lambda v: pct(float(v), 2)},
        ))
        macros["NRules"] = num(int(len(rh)))

    api = csv("api_integration.csv")
    if api is not None and not api.empty:
        write(OUT / "tab_api.tex", latex_table(
            api, "L{4.6cm}L{4.4cm}R{1.8cm}L{4.0cm}",
            "Kết quả thử nghiệm tích hợp dịch vụ REST", "tab:api",
            formatters={"Mã HTTP": lambda v: num(int(v))},
        ))

    tc = csv("traditional_comparison.csv")
    if tc is not None and not tc.empty:
        d = tc.copy()
        write(OUT / "tab_traditional.tex", latex_table(
            d.T.reset_index().rename(
                columns={"index": "Chỉ tiêu", 0: "Truyền thống", 1: "Đề xuất"}
            ).iloc[1:],
            "L{6.4cm}R{4.2cm}R{4.2cm}",
            "So sánh mô hình đề xuất với quy trình thẩm định truyền thống",
            "tab:traditional",
            note="Quy trình truyền thống được mô phỏng bằng bộ luật nghiệp vụ đang "
                 "áp dụng, không sử dụng mô hình học máy.",
        ))
        try:
            trad = float(tc.iloc[0]["Nợ xấu nhóm duyệt (%)"])
            prop = float(tc.iloc[1]["Nợ xấu nhóm duyệt (%)"])
            macros["TradBadRate"] = num(trad, 2) + r"\%"
            macros["PropBadRate"] = num(prop, 2) + r"\%"
            macros["BadRateReduction"] = num(trad - prop, 2)
            if trad > 0:
                macros["BadRateReductionPct"] = num(
                    100.0 * (trad - prop) / trad, 1) + r"\%"
            macros["TradReviewRate"] = num(
                float(tc.iloc[0]["Tỷ lệ thẩm định thủ công (%)"]), 2) + r"\%"
            macros["PropReviewRate"] = num(
                float(tc.iloc[1]["Tỷ lệ thẩm định thủ công (%)"]), 2) + r"\%"
            trad_app = float(tc.iloc[0]["Tỷ lệ duyệt (%)"])
            prop_app = float(tc.iloc[1]["Tỷ lệ duyệt (%)"])
            macros["TradApproveRate"] = num(trad_app, 2) + r"\%"
            macros["ApproveRateDrop"] = num(trad_app - prop_app, 2)
            macros["ReviewRateRise"] = num(
                float(tc.iloc[1]["Tỷ lệ thẩm định thủ công (%)"])
                - float(tc.iloc[0]["Tỷ lệ thẩm định thủ công (%)"]), 2)
        except Exception as exc:
            log.warning("Không trích được macro so sánh truyền thống: %s", exc)

    # --- bảng dài cho phụ lục ---------------------------------------------
    rc = csv("rule_catalogue.csv")
    if rc is not None and not rc.empty:
        d = rc[["Mã luật", "Tên luật", "Nhóm", "Hành động", "Diễn giải"]].copy()
        write(OUT / "tab_rule_catalogue.tex", latex_longtable(
            d, "L{1.5cm}L{3.8cm}L{2.5cm}L{2.4cm}L{4.5cm}",
            "Danh mục đầy đủ luật nghiệp vụ được cài đặt trong mô hình",
            "tab:rule-catalogue",
        ))

    fcat = csv("feature_catalogue.csv")
    if fcat is not None and not fcat.empty:
        d = fcat[["Tên đặc trưng", "Nhóm", "Nguồn dữ liệu", "Kiểu", "Mô tả"]].copy()
        d["Nhóm"] = d["Nhóm"].map(group_label)

        # Các đặc trưng hành vi được sinh theo cùng một mẫu trên bốn cửa sổ thời
        # gian, nên liệt kê đủ 4 biến thể của mỗi độ đo chỉ làm bảng dài thêm mà
        # không thêm thông tin. Gộp chúng thành một dòng mẫu với hậu tố {W}.
        windows = list(cfg.get("features.txn_windows_days", [7, 30, 90, 180]))
        suffixes = tuple(f"_{w}d" for w in windows)
        pattern = re.compile(r"_(%s)d$" % "|".join(str(w) for w in windows))

        rows, seen = [], set()
        for _, row in d.iterrows():
            name = str(row["Tên đặc trưng"])
            if name.endswith(suffixes):
                template = pattern.sub("_{W}d", name)
                if template in seen:
                    continue
                seen.add(template)
                row = row.copy()
                row["Tên đặc trưng"] = template
                desc = re.sub(r"\s*(trong\s+)?\d+\s*ngày\s*$", "", str(row["Mô tả"]))
                desc = re.sub(r"\s+trong$", "", desc.strip())
                row["Mô tả"] = desc.strip() + " trong cửa sổ W ngày"
            rows.append(row)

        d = pd.DataFrame(rows)
        n_collapsed = len(fcat) - len(d)
        write(OUT / "tab_feature_catalogue.tex", latex_longtable(
            d, "L{3.9cm}L{2.2cm}L{2.4cm}L{1.1cm}L{5.1cm}",
            "Danh mục đặc trưng đầu vào của mô hình",
            "tab:feature-catalogue",
            note="Kiểu dữ liệu: num -- định lượng; cat -- định tính; bin -- nhị "
                 "phân. Mỗi đặc trưng có hậu tố \\code{\\_\\{W\\}d} được tính "
                 f"trên {len(windows)} cửa sổ thời gian "
                 f"$W \\in \\{{{', '.join(str(w) for w in windows)}\\}}$ ngày; "
                 f"{len(seen)} dòng dạng này tương ứng {n_collapsed + len(seen)} "
                 f"đặc trưng, đưa tổng số đặc trưng lên {len(fcat)}.",
        ))

    rb = csv("risk_bands.csv")
    if rb is not None and not rb.empty:
        write(OUT / "tab_risk_bands.tex", latex_table(
            rb, "L{2.4cm}" + "R{2.6cm}" * (len(rb.columns) - 1),
            "Bảng phân hạng rủi ro và điều kiện cấp tín dụng tương ứng",
            "tab:risk-bands",
        ))

    # =====================================================================
    # 6. MACRO SỐ LIỆU
    # =====================================================================
    if meta:
        macros["NTotal"] = num(int(meta.get("n_total", 0)))
        macros["NTrain"] = num(int(meta.get("n_train", 0)))
        macros["NTest"] = num(int(meta.get("n_test", 0)))
        macros["NFeatures"] = num(int(meta.get("n_features", 0)))
        macros["NNumeric"] = num(int(meta.get("n_numeric", 0)))
        macros["NCategorical"] = num(int(meta.get("n_categorical", 0)))
        macros["DRTrain"] = pct(meta.get("default_rate_train", 0), 2)
        macros["DRTest"] = pct(meta.get("default_rate_test", 0), 2)
        macros["BayesAUC"] = num(meta.get("bayes_auc_test", 0), 4)
        # Khoảng cách tới trần Bayes được trích dẫn ở ba chỗ trong phần diễn
        # giải; xuất thành macro để không phải sửa tay khi chạy lại thực nghiệm.
        # Tính theo mô hình được chọn triển khai để khớp với câu diễn giải đi
        # kèm (vốn trích \BestAUC), không lấy theo mô hình có AUC cao nhất.
        if "BestAUC" in macros:
            best_auc = float(macros["BestAUC"].replace(".", "").replace(",", "."))
            macros["GapBayes"] = num(
                float(meta.get("bayes_auc_test", 0)) - best_auc, 4)
        macros["TrainMinutes"] = num(
            float(meta.get("total_training_seconds", 0)) / 60, 1)
    if raw_meta:
        macros["ThinFileRate"] = pct(raw_meta.get("thin_file_rate", 0), 2)
        macros["GenSeconds"] = num(float(raw_meta.get("generation_seconds", 0)), 1)

    lines = [
        "% Macro số liệu thực nghiệm - SINH TỰ ĐỘNG, KHÔNG SỬA TAY",
        "% Sinh bởi scripts/06_export_latex_tables.py",
        "",
    ]
    for k in sorted(macros):
        lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    write(OUT / "macros_results.tex", "\n".join(lines) + "\n")

    log.info("Đã xuất %d macro số liệu", len(macros))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
