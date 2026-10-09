"""Thu thập, tổng hợp và phân tích thống kê mô tả dữ liệu tín dụng."""

from .generator import BETA, CreditDataGenerator, GeneratedData  # noqa: F401
from .profiling import (  # noqa: F401
    dataset_summary_table,
    describe_dataset,
    detect_invalid_records,
    inject_quality_issues,
    verify_detection,
)

__all__ = [
    "CreditDataGenerator",
    "GeneratedData",
    "BETA",
    "describe_dataset",
    "detect_invalid_records",
    "dataset_summary_table",
    "inject_quality_issues",
    "verify_detection",
]
