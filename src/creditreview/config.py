"""Nạp và truy cập cấu hình của mô hình hỗ trợ kiểm duyệt tín dụng.

Cấu hình được đặt tập trung trong ``config/config.yaml`` để mọi thành phần
(sinh dữ liệu, tiền xử lý, xây dựng đặc trưng, huấn luyện mô hình, bộ máy hỗ trợ
ra quyết định) dùng chung một nguồn tham số duy nhất. Cách tổ chức này tránh sai
lệch giữa môi trường huấn luyện và môi trường suy luận: định nghĩa cửa sổ đặc
trưng, ngưỡng quyết định và tham số luật nghiệp vụ chỉ tồn tại ở một nơi.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Gốc dự án = thư mục cha của src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Bọc dict cấu hình, cho phép truy cập theo đường dẫn ``a.b.c``."""

    def __init__(self, data: Dict[str, Any], root: Path = PROJECT_ROOT) -> None:
        self._data = data
        self.root = root

    # -- truy cập ----------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise KeyError(f"Không tìm thấy khóa cấu hình: {dotted}")
        return value

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    # -- đường dẫn ---------------------------------------------------------
    def path(self, key: str) -> Path:
        """Trả về đường dẫn tuyệt đối cho một khóa trong mục ``paths``."""
        rel = self.get(f"paths.{key}")
        if rel is None:
            raise KeyError(f"paths.{key} chưa được khai báo trong config")
        p = self.root / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 42))

    # -- ghi đè bằng biến môi trường --------------------------------------
    def apply_env_overrides(self) -> "Config":
        """Cho phép ghi đè cấu hình qua biến môi trường ``CR_<A>__<B>``.

        Ví dụ ``CR_DECISION__PD_APPROVE_MAX=0.06`` sẽ ghi đè
        ``decision.pd_approve_max``. Cần thiết khi chạy nhiều kịch bản thực
        nghiệm mà không phải sửa file cấu hình.
        """
        for env_key, raw in os.environ.items():
            if not env_key.startswith("CR_"):
                continue
            dotted = env_key[3:].lower().replace("__", ".")
            self._set(dotted, _coerce(raw))
        return self

    def _set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


class _Missing:
    pass


_MISSING = _Missing()


def _coerce(raw: str) -> Any:
    """Chuyển chuỗi biến môi trường về kiểu Python phù hợp."""
    low = raw.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Nạp cấu hình (được cache để tránh đọc lại file nhiều lần)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, root=cfg_path.resolve().parents[1]).apply_env_overrides()


__all__ = ["Config", "load_config", "PROJECT_ROOT"]
