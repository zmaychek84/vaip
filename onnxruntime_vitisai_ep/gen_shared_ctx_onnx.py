##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import onnx
import sys
import tarfile
import os
import hashlib
import argparse
import json


# get node attribute "main_context" == 1  for node with op_type "EPContext"
def get_main_context_node(graph):
    for node in graph.node:
        if node.op_type == "EPContext":
            for attr in node.attribute:
                if attr.name == "main_context" and attr.i == 1:
                    return node
    return None


# get embed_mode attribute from node
def get_embed_mode(node):
    for attr in node.attribute:
        if attr.name == "embed_mode":
            return attr.i


# get ep_cache_context attribute from node
def get_ep_cache_context(node):
    for attr in node.attribute:
        if attr.name == "ep_cache_context":
            # attr.s is a byte string, convert to string
            return attr.s.decode("utf-8")


# modefiy ep_cache_context attribute from node
def modify_ep_cache_context_attr(node, value):
    for attr in node.attribute:
        if attr.name == "ep_cache_context":
            attr.s = value.encode("utf-8")


# calculate md5 for each file in tarfile
def calculate_md5_in_tar(tar_file):
    md5_dict = {}
    for member in tar_file.getmembers():
        f = tar_file.extractfile(member)
        if f is not None:
            md5 = hashlib.md5()
            while True:
                data = f.read(10240)
                if not data:
                    break
                md5.update(data)
            md5_dict[member.name] = md5.hexdigest()
    return md5_dict


def convert_to_shared_ctx(model_path, shared_ctx_bin):
    # model_path convert to full absolute path
    print(f"=============== begin combin model : {model_path} ====================")
    model_dir = os.path.dirname(model_path)
    model = onnx.load(model_path)
    # onnx.checker.check_model(model)
    graph = model.graph

    main_context_node = get_main_context_node(graph)
    ## embed_mode , only support non-embed mode now
    embed_mode = get_embed_mode(main_context_node)
    ## ep_cache_context
    ep_cache_context = get_ep_cache_context(main_context_node)

    ## print main_context_node name
    print(f"main_context_node: {main_context_node.name}")
    print(f"embed_mode: {embed_mode}")
    print(f"ep_cache_context: {ep_cache_context}")

    # non-embed mode, open file model_dir/ep_cache_context and convert to a tarfile object
    if embed_mode == 0:
        tar_file = tarfile.open(os.path.join(model_dir, ep_cache_context), "r")
    ## embed_mode TODO

    cache_files_md5 = calculate_md5_in_tar(tar_file)
    # print(f"cache_files_md5: {cache_files_md5}")

    # main_context_node add attribute "ep_cache_md5" with cacahe_files_md5
    ep_cache_md5_attr = onnx.helper.make_attribute(
        "ep_cache_md5s", json.dumps(cache_files_md5)
    )
    main_context_node.attribute.extend([ep_cache_md5_attr])

    # fill in shared_ctx.bin
    for member in tar_file.getmembers():
        f = tar_file.extractfile(member)
        # print(f"Adding {member.name} -> {cache_files_md5[member.name]} to shared_ctx.bin")
        member.name = cache_files_md5[member.name]
        ## member.name in shared_ctx_bin.getnames() or not
        if member.name in shared_ctx_bin.getnames():
            continue

        if f is not None:
            shared_ctx_bin.addfile(member, fileobj=f)
        else:
            # error
            print(f"Error: {member.name} not found in tar file")
            exit(1)

    # get tarfile file name from shared_ctx_bin
    shared_ctx_bin_file_name = os.path.basename(shared_ctx_bin.name)
    print(f"shared_ctx_bin_file_name: {shared_ctx_bin_file_name}")
    modify_ep_cache_context_attr(main_context_node, shared_ctx_bin_file_name)
    tar_file.close()
    # checker model
    # onnx.checker.check_model(model)
    onnx.save(model, model_path.replace("_ctx.onnx", "_ctx_shared.onnx"))
    print(
        f"Generator shared EP context cache model {model_path.replace('_ctx.onnx', '_ctx_shared.onnx')} done"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate shared context cache onnx model"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Directory containing VitisAI EP context cache ONNX models",
    )
    # parser.add_argument("--inputs", nargs="+", required=True, help="List of VitisAI EP context cache ONNX models")
    # parser.add_argument("--output", required=True, help="Output shared context cache ONNX model")
    args = parser.parse_args()

    ctx_models_dir = args.input_dir
    ctx_models_dir = os.path.abspath(ctx_models_dir)
    share_ctx_bin_file_name = "shared_ctx.bin"

    # get all ctx onnx models in ctx_models_dir
    ctx_onnx_models = [
        os.path.join(ctx_models_dir, f)
        for f in os.listdir(ctx_models_dir)
        if f.endswith("_ctx.onnx")
    ]
    print(f"ctx_onnx_models: {ctx_onnx_models}")

    shared_ctx_bin = tarfile.open(
        os.path.join(ctx_models_dir, share_ctx_bin_file_name), "w"
    )

    for ctx_model in ctx_onnx_models:
        convert_to_shared_ctx(ctx_model, shared_ctx_bin)

    shared_ctx_bin.close()
    print(
        f"============= Generator shared EP context cache model done =================="
    )


if __name__ == "__main__":
    main()
