##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import onnx
import argparse
from pathlib import Path


def change_input(node, tensor_producer, rm_list):
    # input 0
    cast_node = tensor_producer[node.input[0]]
    node.input[0] = cast_node.input[0]
    rm_list.append(cast_node)
    # input 1
    cast_node = tensor_producer[node.input[1]]
    node.input[1] = cast_node.input[0]
    rm_list.append(cast_node)
    # input 2
    cast_node = tensor_producer[node.input[2]]
    node.input[2] = cast_node.input[0]
    rm_list.append(cast_node)


def main():
    parser = argparse.ArgumentParser(
        description="Modify ONNX model by changing MultiHeadAttention nodes to AMDMultiHeadAttention nodes."
    )
    parser.add_argument("--src", type=str, help="Path to the source ONNX model.")
    parser.add_argument("--dst", type=str, help="Path to save the modified ONNX model.")

    args = parser.parse_args()

    print(f"Loading model from {args.src}")
    g = onnx.load(args.src)

    mha_cnt = 0

    # tensor name -> producer node map
    tensor_producer = {}
    # tensor name -> consumer nodes map
    tensor_consumer = {}

    rm_list = []

    # collect info and change node type/domain
    for node in g.graph.node:
        for node_out in node.output:
            if node_out not in tensor_producer:
                tensor_producer[node_out] = node
        for node_in in node.input:
            if node_in not in tensor_consumer:
                tensor_consumer[node_in] = [node]
            else:
                if node_in not in tensor_consumer[node_in]:
                    tensor_consumer[node_in].append(node)
        if node.op_type == "MultiHeadAttention":
            print(
                f"processing {node.name}, with input len {len(node.input)}, output len {len(node.output)}"
            )
            node.op_type = "AMDMultiHeadAttention"
            node.domain = "com.amd"
            mha_cnt += 1

    # second iteration to change input/output with collected info
    for node in g.graph.node:
        if node.op_type == "AMDMultiHeadAttention":
            # change input
            change_input(node, tensor_producer, rm_list)
            # change output
            assert len(tensor_consumer[node.output[0]]) == 1
            consumer_cast_node = tensor_consumer[node.output[0]][0]
            node.output[0] = consumer_cast_node.output[0]
            rm_list.append(consumer_cast_node)

    print(f"{len(rm_list)} Cast nodes to be removed")
    # remove used cast nodes
    for rm_node in rm_list:
        g.graph.node.remove(rm_node)

    print(f"{mha_cnt} MultiHeadAttention nodes modified.")

    dat_file_name = Path(args.dst).stem + ".dat"

    print(f"writing output to {args.dst}")

    onnx.save_model(
        g,
        args.dst,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=dat_file_name,
        size_threshold=1024,
        convert_attribute=False,
    )


if __name__ == "__main__":
    main()
