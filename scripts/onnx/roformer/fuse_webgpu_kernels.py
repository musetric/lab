# ruff: noqa: T201
"""Rewrite node chains of the Duality core into kernels onnxruntime-web runs at once.

The exported core spells several operations as chains of nodes, and on WebGPU every
node of a chain is a dispatch over the whole tensor:

- GELU as Div, Erf, Add, Mul and Mul over the feed-forward's hidden layer;
- the interleaved rotary embedding as Mul, two Slices, Neg, Concat, Mul and Add over
  the queries and the keys;
- the projection to heads as a MatMul and a Transpose that moves the head axis ahead
  of the sequence;
- the query scale of the time attentions as a Slice and a Mul per block of query rows.

This rewrites them into one Gelu, one RotaryEmbedding, one MatMul batched over the
heads (its weight reshaped from the stored one, which onnxruntime folds at load) and
one Mul ahead of a Split. The weights and their file stay; only the .onnx changes.

Usage:
    uv run python scripts/onnx/roformer/fuse_webgpu_kernels.py \
        --model published/duality_core_t1100.onnx --out out/duality_core_t1100.onnx
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

CONSTANT_TOLERANCE = 1e-3
HEAD_DIM = 64
PAIR = 2
HEAD_SPLIT_RANK = 5
QUERY_RANK = 4
QUERY_AXIS = 2
SLICE_STEPS = 4
# A Split touches its input and every output: at most 8 outputs keep it within 9
# storage buffers, under the 10 Dawn allows on Metal (see split_concat_webgpu.py).
SPLIT_GROUP = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="published core")
    parser.add_argument("--out", required=True, type=Path, help="rewritten core")
    return parser.parse_args()


@dataclass
class Edit:
    anchor: onnx.NodeProto
    nodes: list[onnx.NodeProto]
    removed: list[onnx.NodeProto]


@dataclass
class Graph:
    model: onnx.ModelProto
    directory: Path
    shapes: dict[str, list[int]] = field(default_factory=dict)
    producer: dict[str, onnx.NodeProto] = field(default_factory=dict)
    consumers: dict[str, list[onnx.NodeProto]] = field(default_factory=dict)
    initializers: dict[str, onnx.TensorProto] = field(default_factory=dict)
    count: int = 0

    def __post_init__(self) -> None:
        # Every rewrite keeps the names of the tensors it reads and writes, so the
        # shapes of the source graph stay valid for all passes.
        inferred = shape_inference.infer_shapes(self.model).graph
        self.shapes = {
            value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
            for value in [*inferred.input, *inferred.value_info, *inferred.output]
        }
        self.refresh()

    def refresh(self) -> None:
        graph = self.model.graph
        self.initializers = {init.name: init for init in graph.initializer}
        self.producer = {name: node for node in graph.node for name in node.output}
        self.consumers = {}
        for node in graph.node:
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)

    def name(self, hint: str) -> str:
        self.count += 1
        return f"fuse/{self.count}/{hint}"

    def constant(self, values: np.ndarray, hint: str) -> str:
        name = self.name(hint)
        self.model.graph.initializer.append(numpy_helper.from_array(values, name))
        return name

    def indices(self, values: list[int], hint: str) -> str:
        return self.constant(np.array(values, dtype=np.int64), hint)

    def value(self, name: str) -> np.ndarray | None:
        """An initializer's values; external ones are read without attaching them."""
        init = self.initializers.get(name)
        if init is None:
            return None
        if init.data_location != onnx.TensorProto.EXTERNAL:
            return numpy_helper.to_array(init)
        info = {entry.key: entry.value for entry in init.external_data}
        with (self.directory / info["location"]).open("rb") as handle:
            handle.seek(int(info.get("offset", "0")))
            raw = handle.read(int(info["length"]))
        dtype = helper.tensor_dtype_to_np_dtype(init.data_type)
        return np.frombuffer(raw, dtype=dtype).reshape(list(init.dims))

    def only(self, name: str, op: str) -> onnx.NodeProto | None:
        """The `op` that writes `name`, when nothing else reads `name`."""
        node = self.producer.get(name)
        if node is None or node.op_type != op or len(self.consumers.get(name, [])) != 1:
            return None
        return node

    def sole(self, name: str, op: str) -> onnx.NodeProto | None:
        """The one node that reads `name`, when it is an `op`."""
        readers = self.consumers.get(name, [])
        if len(readers) != 1 or readers[0].op_type != op:
            return None
        return readers[0]

    def apply(self, edits: list[Edit]) -> int:
        """Put each edit's nodes where its anchor is and drop the nodes it removes.

        Edits are collected on one unchanged graph and applied in one rebuild: the
        rebuild copies every node, so references taken before it go stale.
        """
        graph = self.model.graph
        at = {id(edit.anchor): edit.nodes for edit in edits}
        gone = {id(node) for edit in edits for node in edit.removed}
        kept = []
        for node in graph.node:
            kept.extend(at.get(id(node), []))
            if id(node) not in gone:
                kept.append(node)
        del graph.node[:]
        graph.node.extend(kept)
        self.refresh()
        return len(edits)


def scalar(graph: Graph, name: str) -> float | None:
    values = graph.value(name)
    if values is None or values.size != 1:
        return None
    return float(values.reshape(()))


def near(value: float | None, expected: float) -> bool:
    if value is None:
        return False
    return abs(value - expected) <= CONSTANT_TOLERANCE * abs(expected)


def other(node: onnx.NodeProto, name: str) -> str:
    return node.input[1] if node.input[0] == name else node.input[0]


def constant_input(graph: Graph, node: onnx.NodeProto, expected: float) -> str | None:
    return next((x for x in node.input if near(scalar(graph, x), expected)), None)


def fuse_gelu(graph: Graph) -> int:
    """x * (0.5 * (erf(x / sqrt(2)) + 1)) -> Gelu(x)."""
    edits = []
    for outer in [node for node in graph.model.graph.node if node.op_type == "Mul"]:
        for x in outer.input:
            scaled = graph.only(other(outer, x), "Mul")
            half = scaled and constant_input(graph, scaled, 0.5)
            shifted = half and graph.only(other(scaled, half), "Add")
            one = shifted and constant_input(graph, shifted, 1.0)
            erf = one and graph.only(other(shifted, one), "Erf")
            divided = erf and graph.only(erf.input[0], "Div")
            if not divided or divided.input[0] != x:
                continue
            if not near(scalar(graph, divided.input[1]), float(np.sqrt(2.0))):
                continue
            gelu = helper.make_node(
                "Gelu", [x], [outer.output[0]], name=graph.name("gelu")
            )
            edits.append(Edit(outer, [gelu], [outer, scaled, shifted, erf, divided]))
            break
    return graph.apply(edits)


def rotated_pairs(graph: Graph, flat: str) -> tuple[str, list[onnx.NodeProto]] | None:
    """Match Reshape(Concat(-x1, x0)) over the pairs (x0, x1) of a Reshape of x."""
    flatten = graph.only(flat, "Reshape")
    stack = flatten and graph.only(flatten.input[0], "Concat")
    if not stack or len(stack.input) != PAIR:
        return None
    first, second = (graph.only(name, "Unsqueeze") for name in stack.input)
    negate = first and graph.only(first.input[0], "Neg")
    odd = negate and graph.only(negate.input[0], "Squeeze")
    even = second and graph.only(second.input[0], "Squeeze")
    odd_slice = odd and graph.only(odd.input[0], "Slice")
    even_slice = even and graph.only(even.input[0], "Slice")
    if not odd_slice or not even_slice or odd_slice.input[0] != even_slice.input[0]:
        return None
    starts = (scalar(graph, even_slice.input[1]), scalar(graph, odd_slice.input[1]))
    pairs = graph.producer.get(odd_slice.input[0])
    if starts != (0.0, 1.0) or not pairs or pairs.op_type != "Reshape":
        return None
    if len(graph.consumers[pairs.output[0]]) != PAIR:
        return None
    nodes = [flatten, stack, first, second, negate, odd, even, odd_slice, even_slice]
    return pairs.input[0], [*nodes, pairs]


def rotary_chain(graph: Graph, add: onnx.NodeProto) -> dict | None:
    """Match x * cos + rotate_pairs(x) * sin, rotate_pairs(x0, x1) = (-x1, x0)."""
    for straight_name in add.input:
        straight = graph.only(straight_name, "Mul")
        rotated = graph.only(other(add, straight_name), "Mul")
        if not straight or not rotated:
            continue
        cos = next((x for x in straight.input if x in graph.initializers), None)
        sin = next((x for x in rotated.input if x in graph.initializers), None)
        if not cos or not sin:
            continue
        x = other(straight, cos)
        match = rotated_pairs(graph, other(rotated, sin))
        if not match or match[0] != x:
            continue
        return {"x": x, "cos": cos, "sin": sin, "nodes": [straight, rotated, *match[1]]}
    return None


def fuse_rotary(graph: Graph) -> int:
    edits = []
    halves: dict[str, str] = {}
    positions: dict[tuple[int, int], str] = {}
    for add in [node for node in graph.model.graph.node if node.op_type == "Add"]:
        chain = rotary_chain(graph, add)
        if chain is None:
            continue
        batch, _, sequence, head = graph.shapes[chain["x"]]
        if head != HEAD_DIM:
            continue
        nodes = []
        for table in (chain["cos"], chain["sin"]):
            values = graph.value(table)
            if not np.array_equal(values[:, 0::2], values[:, 1::2]):
                message = f"{table}: the table does not repeat each angle for a pair"
                raise ValueError(message)
            if table not in halves:
                # Every other column: each pair's angle once, as the kernel takes it.
                halves[table] = graph.name(f"half_{table}")
                bounds = [graph.indices(v, "index") for v in ([0], [head], [1], [2])]
                nodes.append(
                    helper.make_node(
                        "Slice", [table, *bounds], [halves[table]], name=halves[table]
                    )
                )
        if (batch, sequence) not in positions:
            ids = np.tile(np.arange(sequence, dtype=np.int64), (batch, 1))
            positions[batch, sequence] = graph.constant(ids, "positions")
        rotary = helper.make_node(
            "RotaryEmbedding",
            [
                chain["x"],
                halves[chain["cos"]],
                halves[chain["sin"]],
                positions[batch, sequence],
            ],
            [add.output[0]],
            name=graph.name("rotary"),
            interleaved=1,
        )
        edits.append(Edit(add, [*nodes, rotary], [add, *chain["nodes"]]))
    return graph.apply(edits)


def fuse_heads(graph: Graph) -> int:
    """MatMul(x, W^T) to [1, a, s, h, d], then heads first -> MatMul batched over h."""
    edits = []
    for matmul in [node for node in graph.model.graph.node if node.op_type == "MatMul"]:
        weight_t = graph.producer.get(matmul.input[1])
        if not weight_t or weight_t.op_type != "Transpose":
            continue
        if weight_t.input[0] not in graph.initializers:
            continue
        split = graph.sole(matmul.output[0], "Reshape")
        permute = split and graph.sole(split.output[0], "Transpose")
        merge = permute and graph.sole(permute.output[0], "Reshape")
        if not merge or len(graph.shapes[split.output[0]]) != HEAD_SPLIT_RANK:
            continue
        if list(helper.get_attribute_value(permute.attribute[0])) != [0, 1, 3, 2, 4]:
            continue
        _, outer, sequence, heads, head = graph.shapes[split.output[0]]
        width = graph.shapes[matmul.input[0]][-1]
        if graph.shapes[merge.output[0]] != [outer, heads, sequence, head]:
            continue
        rows, stacked, weight = (graph.name(hint) for hint in ("rows", "by_head", "w"))
        nodes = [
            helper.make_node(
                "Reshape",
                [matmul.input[0], graph.indices([outer, 1, sequence, width], "shape")],
                [rows],
                name=rows,
            ),
            helper.make_node(
                "Reshape",
                [weight_t.input[0], graph.indices([heads, head, width], "shape")],
                [stacked],
                name=stacked,
            ),
            helper.make_node(
                "Transpose", [stacked], [weight], name=weight, perm=[0, 2, 1]
            ),
            helper.make_node(
                "MatMul", [rows, weight], [merge.output[0]], name=graph.name("heads")
            ),
        ]
        removed = [matmul, split, permute, merge]
        if len(graph.consumers[weight_t.output[0]]) == 1:
            removed.append(weight_t)
        edits.append(Edit(merge, nodes, removed))
    return graph.apply(edits)


def split_tree(
    graph: Graph, source: str, blocks: list[tuple[int, str]]
) -> list[onnx.NodeProto]:
    """Split `source` on the query axis into `blocks` of (rows, name), 8 wide."""
    if len(blocks) <= SPLIT_GROUP:
        sizes = graph.indices([rows for rows, _ in blocks], "sizes")
        outputs = [name for _, name in blocks]
        split = helper.make_node(
            "Split",
            [source, sizes],
            outputs,
            name=graph.name("blocks"),
            axis=QUERY_AXIS,
        )
        return [split]
    group = max(SPLIT_GROUP, -(-len(blocks) // SPLIT_GROUP))
    groups = [blocks[i : i + group] for i in range(0, len(blocks), group)]
    parts = [
        (
            sum(rows for rows, _ in part),
            part[0][1] if len(part) == 1 else graph.name("group"),
        )
        for part in groups
    ]
    nodes = split_tree(graph, source, parts)
    for part, (_, name) in zip(groups, parts, strict=True):
        if len(part) > 1:
            nodes.extend(split_tree(graph, name, part))
    return nodes


def query_blocks(
    graph: Graph, name: str
) -> list[tuple[int, int, onnx.NodeProto, onnx.NodeProto]] | None:
    """The blocks a time attention cuts from its queries: (start, end, Slice, Mul)."""
    blocks = []
    for node in graph.consumers.get(name, []):
        if node.op_type != "Slice":
            return None
        starts, ends, axes = (graph.value(i) for i in node.input[1:SLICE_STEPS])
        steps = (
            graph.value(node.input[SLICE_STEPS])
            if len(node.input) > SLICE_STEPS
            else None
        )
        if axes is None or axes.tolist() != [QUERY_AXIS]:
            return None
        if steps is not None and steps.tolist() != [1]:
            return None
        mul = graph.sole(node.output[0], "Mul")
        if not mul or scalar(graph, other(mul, node.output[0])) is None:
            return None
        end = int(min(ends[0], graph.shapes[name][QUERY_AXIS]))
        blocks.append((int(starts[0]), end, node, mul))
    blocks.sort(key=lambda block: block[0])
    return blocks


def fuse_query_scale(graph: Graph) -> int:
    """A Slice and a Mul(scale) per block of query rows -> one Mul, one Split."""
    edits = []
    order = {id(node): index for index, node in enumerate(graph.model.graph.node)}
    for name in list(graph.consumers):
        if (
            len(graph.consumers[name]) < PAIR
            or len(graph.shapes.get(name, [])) != QUERY_RANK
        ):
            continue
        blocks = query_blocks(graph, name)
        if not blocks:
            continue
        bounds = [(start, end) for start, end, _, _ in blocks]
        if bounds[0][0] != 0 or bounds[-1][1] != graph.shapes[name][QUERY_AXIS]:
            continue
        if any(bounds[i][1] != bounds[i + 1][0] for i in range(len(bounds) - 1)):
            continue
        factors = {other(mul, node.output[0]) for _, _, node, mul in blocks}
        if len({scalar(graph, factor) for factor in factors}) != 1:
            continue
        scaled = graph.name("query_scaled")
        outputs = [(end - start, mul.output[0]) for start, end, _, mul in blocks]
        nodes = [
            helper.make_node("Mul", [name, next(iter(factors))], [scaled], name=scaled),
            *split_tree(graph, scaled, outputs),
        ]
        anchor = min(
            (node for _, _, node, _ in blocks), key=lambda node: order[id(node)]
        )
        removed = [item for _, _, node, mul in blocks for item in (node, mul)]
        edits.append(Edit(anchor, nodes, removed))
    return graph.apply(edits)


def prune(graph: Graph) -> None:
    model_graph = graph.model.graph
    outputs = {value.name for value in model_graph.output}
    while True:
        used = {name for node in model_graph.node for name in node.input} | outputs
        kept = [node for node in model_graph.node if set(node.output) & used]
        if len(kept) == len(model_graph.node):
            break
        del model_graph.node[:]
        model_graph.node.extend(kept)
    used = {name for node in model_graph.node for name in node.input}
    initializers = [init for init in model_graph.initializer if init.name in used]
    del model_graph.initializer[:]
    model_graph.initializer.extend(initializers)
    del model_graph.value_info[:]


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model, load_external_data=False)
    graph = Graph(model, args.model.parent)
    before = len(model.graph.node)
    counts = {
        "gelu": fuse_gelu(graph),
        "rotary": fuse_rotary(graph),
        "heads": fuse_heads(graph),
        "query scale": fuse_query_scale(graph),
    }
    prune(graph)
    shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    fused = ", ".join(f"{key} {value}" for key, value in counts.items())
    print(f"{before} -> {len(model.graph.node)} nodes; {fused}")


if __name__ == "__main__":
    main()
