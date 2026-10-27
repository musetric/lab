# ruff: noqa: T201

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper
from onnx.external_data_helper import uses_external_data

MATRIX_DIMS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="rebuilt core")
    parser.add_argument("--reference", required=True, type=Path, help="published core")
    parser.add_argument("--out", required=True, type=Path, help="rewritten core")
    return parser.parse_args()


def external(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def read(directory: Path, tensor: onnx.TensorProto) -> bytes:
    info = external(tensor)
    with (directory / info["location"]).open("rb") as handle:
        handle.seek(int(info.get("offset", "0")))
        return handle.read(int(info["length"]))


def point(tensor: onnx.TensorProto, found: tuple[str, int, int]) -> None:
    location, offset, length = found
    del tensor.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def fold_transposes(
    model: onnx.ModelProto, directory: Path, offsets: dict[str, tuple[str, int, int]]
) -> set[str]:
    graph = model.graph
    tensors = {tensor.name: tensor for tensor in graph.initializer}
    folded: set[str] = set()
    for node in list(graph.node):
        source = tensors.get(node.input[0]) if node.op_type == "Transpose" else None
        if (
            source is None
            or not uses_external_data(source)
            or len(source.dims) != MATRIX_DIMS
        ):
            continue
        dtype = helper.tensor_dtype_to_np_dtype(source.data_type)
        matrix = np.frombuffer(read(directory, source), dtype=dtype)
        transposed = np.ascontiguousarray(matrix.reshape(list(source.dims)).T)
        found = offsets.get(hashlib.sha256(transposed.tobytes()).hexdigest())
        if found is None:
            continue
        folded_tensor = onnx.TensorProto()
        folded_tensor.name = node.output[0]
        folded_tensor.data_type = source.data_type
        folded_tensor.dims.extend(transposed.shape)
        folded_tensor.data_location = onnx.TensorProto.EXTERNAL
        point(folded_tensor, found)
        graph.initializer.append(folded_tensor)
        graph.node.remove(node)
        if not any(source.name in other.input for other in graph.node):
            graph.initializer.remove(source)
        folded.add(folded_tensor.name)
    return folded


def main() -> None:
    args = parse_args()
    reference = onnx.load(args.reference, load_external_data=False)
    locations = set()
    offsets: dict[str, tuple[str, int, int]] = {}
    for tensor in reference.graph.initializer:
        if not uses_external_data(tensor):
            continue
        data = read(args.reference.parent, tensor)
        info = external(tensor)
        locations.add(info["location"])
        offsets[hashlib.sha256(data).hexdigest()] = (
            info["location"],
            int(info.get("offset", "0")),
            len(data),
        )
    if len(locations) != 1:
        message = f"the reference uses {len(locations)} weights files"
        raise SystemExit(message)

    model = onnx.load(args.model, load_external_data=False)
    folded = fold_transposes(model, args.model.parent, offsets)
    missing = []
    moved = 0
    for tensor in model.graph.initializer:
        if not uses_external_data(tensor) or tensor.name in folded:
            continue
        digest = hashlib.sha256(read(args.model.parent, tensor)).hexdigest()
        found = offsets.get(digest)
        if found is None:
            missing.append(tensor.name)
            continue
        point(tensor, found)
        moved += 1
    if missing:
        message = f"{len(missing)} tensors are not in the reference: {missing[:5]}"
        raise SystemExit(message)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    print(f"folded {len(folded)} transposes")
    print(f"{moved} tensors now read from {locations.pop()}")
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
