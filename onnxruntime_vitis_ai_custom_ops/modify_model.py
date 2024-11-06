##
##     The Xilinx Vitis AI Vaip in this distribution are provided under the
## following free and permissive binary-only license, but are not provided in
## source code form.  While the following free and permissive license is similar
## to the BSD open source license, it is NOT the BSD open source license nor
## other OSI-approved open source license.
##
##      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
##      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
## reserved.
##
##      Redistribution and use in binary form only, without modification, is
## permitted provided that the following conditions are met:
##
##      1. Redistributions must reproduce the above copyright notice, this list
## of conditions and the following disclaimer in the documentation and/or other
## materials provided with the distribution.
##
##      2. The name of Xilinx, Inc. may not be used to endorse or promote
## products redistributed with this software without specific prior written
## permission.
##
##      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
## IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
## EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
##      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
## EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
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
