"""Logging, atomic IO, hashing, seeding and environment capture helpers.

Every mutating write in this repository goes through :func:`atomic_write_bytes`
or :func:`atomic_write_text` (temp file in the *same* directory, then
``os.replace``), so a crash can never leave a corrupt cache or output file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO,
                  name: str = "esmc_target") -> logging.Logger:
    """Configure timestamped console + optional file logging. Idempotent."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if _CONFIGURED and log_file is None:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - defensive
            pass
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    _CONFIGURED = True
    return logger


def get_logger(name: str = "esmc_target") -> logging.Logger:
    return logging.getLogger(name)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    """SHA256 of raw file *bytes* (never of a dataframe)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object (sorted keys, compact)."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                  default=str))


def atomic_write_bytes(path: os.PathLike | str, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically (temp file in same dir + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:  # pragma: no cover - defensive
                pass
        raise
    return path


def atomic_write_text(path: os.PathLike | str, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: os.PathLike | str, obj: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=False,
                                              default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    import numpy as np
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@contextmanager
def atomic_path(path: os.PathLike | str, binary: bool = True) -> Iterator[Path]:
    """Context manager yielding a temp path that is ``os.replace``d on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                    dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(str(tmp), str(path))
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover
                pass
        raise


def seed_everything(seed: int = 0) -> Dict[str, int]:
    """Seed torch/numpy/python. Environment reproducibility only.

    The default pipeline contains no stochastic operation; seeding exists so the
    manifest records an identical starting environment.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass
    return {"python_random": seed, "numpy": seed, "torch": seed}


@dataclass
class SoftwareVersions:
    python: str
    pytorch: Optional[str]
    transformers: Optional[str]
    cuda: Optional[str]
    cudnn: Optional[str]
    numpy: Optional[str]
    pandas: Optional[str]
    platform: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python": self.python,
            "pytorch": self.pytorch,
            "transformers": self.transformers,
            "cuda": self.cuda,
            "cudnn": self.cudnn,
            "numpy": self.numpy,
            "pandas": self.pandas,
            "platform": self.platform,
        }


def software_versions() -> SoftwareVersions:
    import numpy as np
    import pandas as pd
    try:
        import torch
        torch_version: Optional[str] = torch.__version__
        cuda_version: Optional[str] = getattr(torch.version, "cuda", None)
        cudnn_version: Optional[str] = None
        if torch.backends.cudnn.is_available():
            cudnn_version = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover
        torch_version = None
        cuda_version = None
        cudnn_version = None
    try:
        import transformers
        transformers_version: Optional[str] = transformers.__version__
    except Exception:
        transformers_version = None
    return SoftwareVersions(
        python=platform.python_version(),
        pytorch=torch_version,
        transformers=transformers_version,
        cuda=cuda_version,
        cudnn=cudnn_version,
        numpy=np.__version__,
        pandas=pd.__version__,
        platform=platform.platform(),
    )


def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_total_bytes": None,
        "gpu_count": 0,
        "torch_cuda_version": None,
    }
    try:
        import torch
    except Exception:  # pragma: no cover
        return info
    info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        return info
    info["cuda_available"] = True
    info["gpu_count"] = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    info["gpu_name"] = props.name
    info["gpu_memory_total_bytes"] = int(props.total_memory)
    return info


def peak_gpu_memory_bytes() -> Optional[int]:
    try:
        import torch
    except Exception:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def human_bytes(num: Optional[int]) -> Optional[str]:
    if num is None:
        return None
    step = 1024.0
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PiB"


class Stopwatch:
    """Wall-clock accounting per protein and for the whole run."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.per_item: Dict[str, float] = {}

    @contextmanager
    def item(self, key: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.per_item[key] = time.perf_counter() - t0

    def record(self, key: str, seconds: float) -> None:
        self.per_item[key] = seconds

    def total_seconds(self) -> float:
        return time.perf_counter() - self.start


def unique_stable(values: Iterable[Any]) -> list:
    """Deduplicate while preserving first-seen order, then (optionally) sort."""
    seen = set()
    out = []
    for value in values:
        if value in seen or value is None:
            continue
        seen.add(value)
        out.append(value)
    return out
