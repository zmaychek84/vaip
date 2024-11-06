# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# --------------------------------------------------------------------------

import collections
import os
import hashlib
import pathlib
import copy
import json
import traceback

import onnx
import tvm
from tqdm import tqdm
from tvm import relay
from tvm.contrib import graph_executor

from tvm.relay.op.contrib.versal_aie import partition_for_aie
import tvm.relay.op.contrib.versal_aie
from tvm.contrib.target import versal_aie
from tvm.contrib.target.versal_aie.config import get_target_config
from tvm.contrib.debugger.build_logger import (
    AieBuildLogger,
    pretty_print_op,
    pretty_print_op_shape,
)
import logging


@tvm.register_func("tvm_onnx_import_and_compile")
def onnx_compile(
    model_path,
    cache_key,
    target,
    target_host,
    opt_level,
    freeze_params,
    input_shapes,
    build_version="aie.build",
    aie_target="aieml-gemm-asr-qdq -device=aiemaize -mattr=+bdrp,+opt,+double-buffer",
    aiectrl_cfg="aie_control_config.json",
    xclbin="asr_qdq_4x2.xclbin",
):
    if build_version not in ["cpu", "aie.build"]:
        raise ValueError(f"unsupported build_version: {build_version}")
    print(f"[ort.py] build version: {build_version}")

    _target = aie_target.split(" ")[0]

    def ensure_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            if not os.path.isdir(dir_path):
                raise NotADirectoryError(
                    f"{dir_path} already exists and is not a directory!"
                )

    def append_string(buffer, *args):
        for string in args:
            buffer.extend(string.encode())

    def append_file_contents(buffer, *args):
        for fname in args:
            with open(fname, "rb") as fdata:
                contents = fdata.read()
                buffer.extend(contents)

    def dump_build_config(fpath, **kwargs):
        with open(fpath, "w", encoding="utf-8") as cfg_file:
            json.dump(kwargs, cfg_file)

    print(f"reading from {model_path}, vaip cache key {cache_key}")
    # with open(model_path, "rb") as fdata:
    #     bytes_to_hash = bytearray(fdata.read())
    bytes_to_hash = bytearray(cache_key, encoding="utf-8")  # using cache_key

    # print(f"bytes read {len(bytes_to_hash)}")
    if build_version == "cpu":
        append_string(bytes_to_hash, target)
    else:
        des_path = tvm.get_global_func("tvm.aie.get_config_path")(_target)
        with open(des_path, "rb") as f:
            cfg_context = f.read()
            bytes_to_hash.extend(cfg_context)

    build_md5 = hashlib.md5(bytes_to_hash).hexdigest()
    print(f"gemm build hash: {build_md5}")
    # cache dir
    cache_dir = pathlib.Path.home() / ".gemm_cache" / build_md5
    ensure_dir(cache_dir)
    cpu_ctx = tvm.device(target, 0)
    lib_path = cache_dir / "libtvmbuild.dll"
    cfg_path = cache_dir / "config.json"
    # check cache first
    graph_mod = None
    if os.path.exists(lib_path):
        print("loading cached library")
        with tqdm(total=2, desc="Loding cached model") as pbar:
            lib = tvm.runtime.load_module(lib_path)
            pbar.update(1)
            if not lib:
                return None
            if build_version == "cpu":
                graph_mod = graph_executor.GraphModule(lib["default"](cpu_ctx))
            else:
                # above code for cpu should also work for aie but it doesn't
                graph_mod = versal_aie.AieGraphModule(
                    lib["aie_create"](tvm.cpu(0), tvm.aie())
                )
            pbar.update(1)
            return graph_mod.module

    try:
        with tqdm(total=4, desc="Compiling model") as pbar:
            model = onnx.load(model_path)
            opset_version = (
                model.opset_import[0].version if len(model.opset_import) > 0 else None
            )
            # Collect only feed input names from all input names
            all_input_names = [node.name for node in model.graph.input]
            all_initializer = [node.name for node in model.graph.initializer]
            # breakpoint()
            all_node_input_names = set()
            for node in model.graph.node:
                all_node_input_names.update(node.input)
            net_feed_input_names = list(set(all_input_names) - set(all_initializer))
            # net_feed_input_names = ["Input3"] # hardcoded
            dangling_inputs = set(all_input_names) - all_node_input_names
            print(f"net_feed_input_names before filter {net_feed_input_names}")
            if dangling_inputs:
                print(f"dangling_inputs detected: {dangling_inputs}")
                net_feed_input_names = list(set(net_feed_input_names) - dangling_inputs)
            print(f"net_feed_input_names after filter {net_feed_input_names}")

            # Match names and input shapes
            all_input_mapping = list(zip(all_input_names, input_shapes))
            # Using an ordereddict maintains input ordering.
            shape_dict = collections.OrderedDict(all_input_mapping)
            # Get only feed input pairs
            feed_shape_dict = {}
            for name in net_feed_input_names:
                feed_shape_dict[name] = shape_dict[name]
            pbar.update(1)
            irmod, params = relay.frontend.from_onnx(
                model, feed_shape_dict, opset=opset_version, freeze_params=freeze_params
            )
            irmod = relay.transform.DynamicToStatic()(irmod)
            pbar.update(1)
            tvm_target = tvm.target.Target(target, host=target_host)
            print("[ort.py] build_version:", build_version)
            if build_version == "cpu":
                with tvm.transform.PassContext(opt_level=opt_level):
                    lib = relay.build(irmod, target=tvm_target, params=params)
                    pbar.update(1)
                if lib:
                    lib.export_library(lib_path)
                    dump_build_config(
                        cfg_path, target=target, build_version=build_version
                    )
                    graph_mod = graph_executor.GraphModule(lib["default"](cpu_ctx))
                    pbar.update(1)
            else:
                pass_cfg = {
                    "relay.ext.versal_aie.options.target": f"{aie_target}",
                    "relay.ext.versal_aie.options.aiectrl": aiectrl_cfg,
                    "relay.ext.versal_aie.options.xclbin": xclbin,
                    "relay.ext.versal_aie.options.mode": "xrt",
                    "relay.ext.versal_aie.options.debug": "true",
                }
                logger = logging.getLogger("versal_aie_build")
                with versal_aie.Environment.from_target(_target):
                    with tvm.transform.PassContext(
                        opt_level=3, config=pass_cfg, instruments=[AieBuildLogger()]
                    ):
                        open("origin_relay_ir.txt", "w").write(str(irmod))
                        print(f"[Info] Saving original relay into origin_relay_ir.txt")
                        irmod = partition_for_aie(
                            irmod,
                            should_partition=True,
                            should_preprocess=True,
                            should_pad=True,
                        )
                        op_profile = pretty_print_op(irmod)
                        op_shape_profile = pretty_print_op_shape(irmod)
                        lib = relay.build(irmod, target="llvm", params=params)
                        open("op_shape_profile.txt", "w").write(op_shape_profile)
                        logger.info(f"OP Profile:\n{op_profile}")
                        pbar.update(1)
                        graph_json = json.loads(lib.get_graph_json())
                        lib_ir_mod = json.loads(tvm.ir.save_json(lib.ir_mod))
                        if lib:
                            lib.export_library(lib_path)
                            dump_build_config(
                                cfg_path,
                                target=target,
                                build_version=build_version,
                                aie_target=aie_target,
                                aiectrl_cfg=aiectrl_cfg,
                                xclbin=xclbin,
                            )
                            dump_build_config(
                                cache_dir / "build_data.json",
                                graph_json=graph_json,
                                ir_mod=lib_ir_mod,
                            )
                            graph_mod = versal_aie.AieGraphModule(
                                lib["aie_create"](tvm.cpu(0), tvm.aie())
                            )
                            pbar.update(1)

    except Exception as inst:
        print(traceback.format_exc())
        print(
            f"compilation failed with exception:\n{repr(inst)}\nfallback to OnnxRuntime CPU."
        )
    if graph_mod:
        return graph_mod.module
    return None
