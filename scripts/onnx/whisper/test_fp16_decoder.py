import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from fp16_decoder import check_and_save_model, toposort_graph
from onnx import TensorProto, helper, numpy_helper


class Fp16DecoderTest(unittest.TestCase):
    def test_toposort_orders_nodes_and_captured_branch_inputs(self) -> None:
        def branch(name: str) -> onnx.GraphProto:
            result = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1])
            return helper.make_graph(
                [helper.make_node("Identity", ["second"], ["result"])],
                name,
                [],
                [result],
            )

        condition = numpy_helper.from_array(np.array(True), "condition")
        graph = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["condition"],
                    ["output"],
                    name="branch",
                    then_branch=branch("then"),
                    else_branch=branch("else"),
                ),
                helper.make_node("Identity", ["first"], ["second"], name="second"),
                helper.make_node("Identity", ["input"], ["first"], name="first"),
            ],
            "test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
            [condition],
        )

        toposort_graph(graph)

        self.assertEqual(
            [node.name for node in graph.node], ["first", "second", "branch"]
        )
        onnx.checker.check_model(helper.make_model(graph))

    def test_toposort_rejects_missing_producer(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Identity", ["missing"], ["output"])], "test", [], []
        )
        with self.assertRaisesRegex(RuntimeError, "missing producer"):
            toposort_graph(graph)

    def test_saves_and_checks_external_weights(self) -> None:
        tensor = numpy_helper.from_array(np.arange(4096, dtype=np.float32), "weight")
        graph = helper.make_graph(
            [helper.make_node("Identity", ["weight"], ["output"])],
            "test",
            [],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4096])],
            [tensor],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decoder.onnx"
            with patch.object(onnx.checker, "MAXIMUM_PROTOBUF", 1):
                check_and_save_model(model, output)
            self.assertTrue(output.with_name("decoder.onnx_data").is_file())
            loaded = onnx.load(str(output))
            np.testing.assert_array_equal(
                numpy_helper.to_array(loaded.graph.initializer[0]),
                np.arange(4096, dtype=np.float32),
            )

    def test_invalid_graph_leaves_previous_output_intact(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Identity", ["missing"], ["output"])],
            "test",
            [],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decoder.onnx"
            data = output.with_name("decoder.onnx_data")
            output.write_bytes(b"previous graph")
            data.write_bytes(b"previous weights")
            with self.assertRaises(onnx.checker.ValidationError):
                check_and_save_model(helper.make_model(graph), output)
            self.assertEqual(output.read_bytes(), b"previous graph")
            self.assertEqual(data.read_bytes(), b"previous weights")


if __name__ == "__main__":
    unittest.main()
