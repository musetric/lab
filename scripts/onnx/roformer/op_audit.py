# ruff: noqa: T201 -- CLI inspection tool: stdout (the op histogram) is its interface.

import argparse
from collections import Counter

import onnx

parser = argparse.ArgumentParser(description="Print an ONNX op histogram.")
parser.add_argument("model")
args = parser.parse_args()

model = onnx.load(args.model, load_external_data=False)
counts = Counter(n.op_type for n in model.graph.node)

print("opset:", [(i.domain or "ai.onnx", i.version) for i in model.opset_import])
print(f"total nodes: {sum(counts.values())}  unique op types: {len(counts)}")
for op, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {op:24s} {n}")
