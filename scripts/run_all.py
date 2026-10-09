"""Chạy toàn bộ quy trình thực nghiệm theo đúng thứ tự.

Thực hiện tuần tự sáu bước và dừng ngay khi có bước lỗi, nhờ đó tránh trường hợp
một bước dùng kết quả cũ của bước trước đó mà không nhận ra.

Cách dùng:
    py scripts/run_all.py                # chạy đầy đủ
    py scripts/run_all.py --quick        # chạy nhanh để kiểm tra
    py scripts/run_all.py --skip-tuning  # bỏ qua bước tinh chỉnh siêu tham số
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(label: str, args: list[str]) -> float:
    """Chạy một bước và trả về thời gian thực hiện (giây)."""
    print("\n" + "=" * 78)
    print(f"  {label}")
    print("=" * 78, flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, *args], cwd=ROOT)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"\nLỖI: bước '{label}' kết thúc với mã {proc.returncode}. Dừng quy trình.")
        raise SystemExit(proc.returncode)
    print(f"\n-> Hoàn thành '{label}' sau {dt/60:.1f} phút")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser(description="Chạy toàn bộ quy trình thực nghiệm")
    ap.add_argument("--quick", action="store_true", help="Chạy quy mô nhỏ để kiểm tra")
    ap.add_argument("--skip-tuning", action="store_true",
                    help="Bỏ qua bước tinh chỉnh siêu tham số")
    args = ap.parse_args()

    steps: list[tuple[str, list[str]]] = []

    # --- Bước 1: chuẩn bị dữ liệu ------------------------------------------
    prepare = [str(SCRIPTS / "01_prepare_data.py")]
    if args.quick:
        prepare += ["--n-customers", "4000", "--n-applications", "6000"]
    steps.append(("BƯỚC 1 - Chuẩn bị dữ liệu và xây dựng nhãn rủi ro", prepare))

    # --- Bước 2: đặc trưng --------------------------------------------------
    steps.append((
        "BƯỚC 2 - Phân tích và lựa chọn đặc trưng",
        [str(SCRIPTS / "02_build_features.py")],
    ))

    # --- Bước 3: tinh chỉnh siêu tham số ------------------------------------
    if not args.skip_tuning and not args.quick:
        steps.append((
            "BƯỚC 3 - Tinh chỉnh siêu tham số",
            [str(SCRIPTS / "03_tune_hyperparams.py"), "--trials", "14"],
        ))

    # --- Bước 4: huấn luyện mô hình -----------------------------------------
    train = [str(SCRIPTS / "04_train_models.py")]
    if args.quick:
        train += ["--models", "logistic_regression", "decision_tree", "svm",
                  "lightgbm", "--no-stacking", "--no-permutation"]
    steps.append(("BƯỚC 4 - Huấn luyện và so sánh mô hình", train))

    # --- Bước 5: mô phỏng quy trình kiểm duyệt ------------------------------
    evaluate = [str(SCRIPTS / "05_simulate_evaluate.py")]
    if args.quick:
        evaluate += ["--n-applications", "1000"]
    steps.append(("BƯỚC 5 - Mô phỏng quy trình kiểm duyệt và đánh giá chung", evaluate))

    # --- Bước 6: xuất bảng --------------------------------------------------
    steps.append((
        "BƯỚC 6 - Xuất bảng LaTeX cho đề án",
        [str(SCRIPTS / "06_export_latex_tables.py")],
    ))

    print("QUY TRÌNH THỰC NGHIỆM - ỨNG DỤNG MÔ HÌNH HỌC MÁY "
          "TRONG HỖ TRỢ KIỂM DUYỆT TÍN DỤNG")
    print(f"Chế độ: {'nhanh (kiểm tra)' if args.quick else 'đầy đủ'}")
    print(f"Số bước: {len(steps)}")

    total = 0.0
    for i, (label, cmd) in enumerate(steps, start=1):
        total += run(f"{label}  [{i}/{len(steps)}]", cmd)

    print("\n" + "=" * 78)
    print(f"  HOÀN THÀNH TOÀN BỘ QUY TRÌNH sau {total/60:.1f} phút")
    print("=" * 78)
    print("\nKết quả:")
    print(f"  Bảng   : {ROOT / 'results' / 'tables'}")
    print(f"  Đồ thị : {ROOT / 'results' / 'figures'}")
    print(f"  Mô hình: {ROOT / 'results' / 'models'}")
    print("\nBiên dịch đề án:")
    print("  cd thesis && xelatex main.tex && xelatex main.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
