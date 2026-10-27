import json
from pathlib import Path
from typing import Any

import numpy as np

TENSOR_DTYPES = {"float32": np.dtype("<f4"), "int32": np.dtype("<i4")}


class CaseWriter:
    def __init__(self, out_dir: Path, step: str) -> None:
        self.out_dir = out_dir
        self.step = step
        self.tensors: dict[str, dict[str, Any]] = {}
        self.results: dict[str, Any] = {}
        self.meta: dict[str, Any] = {}
        out_dir.mkdir(parents=True, exist_ok=True)

    def tensor(
        self,
        boundary: str,
        source: str,
        values: np.ndarray,
        dtype: str = "float32",
    ) -> None:
        array = np.ascontiguousarray(values, dtype=TENSOR_DTYPES[dtype])
        file_name = f"{boundary}.{source}.bin"
        array.tofile(self.out_dir / file_name)
        self.tensors[f"{boundary}@{source}"] = {
            "file": file_name,
            "dtype": dtype,
            "shape": list(array.shape),
        }

    def result(self, boundary: str, source: str, value: Any) -> None:
        self.results[f"{boundary}@{source}"] = value

    def finish(self) -> None:
        manifest = {
            "step": self.step,
            "meta": self.meta,
            "tensors": self.tensors,
            "results": self.results,
        }
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
