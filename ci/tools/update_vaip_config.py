##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##
import logging
import os


def vaip_perf_workaround(
    json_dict,
    env_name,
    convert_softmax_to_hard_softmax,
):
    try:
        for pass_item in json_dict.get("passes", {}):
            if pass_item.get("name", "") != "fuse_DPU":
                continue
            sub_passes = pass_item.get("passDpuParam", {}).get("subPass", [])
            for sub_pass in sub_passes:
                if sub_pass.get("name", "") == env_name.lower():
                    if convert_softmax_to_hard_softmax == "true":
                        if sub_pass.get("disabled", None) is True:
                            sub_pass["disabled"] = False
                    elif convert_softmax_to_hard_softmax == "false":
                        if sub_pass.get("disabled", None) is False:
                            sub_pass["disabled"] = True
                    break

    except Exception as e:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)
        logging.warning(f"!!! warning :write new vaip_config.json failed! {e}.)")


def workaround_4x2(curr_env, json_dict):
    try:
        gen_bd = curr_env.get("GEN_BD", "")
        set_conv_aie_mode = curr_env.get("SET_CONV_AIE_MODE", "")
        specify_pdis = curr_env.get("SPECIFY_PDIS", "")
        enable_ac_tek_opti = curr_env.get("ENABLE_AC_TEK_OPTI", "")
        verify_slnode = curr_env.get("VERIFY_SLNODE", "")
        prefetch_lp_load_reorder = curr_env.get("PREFETCH_LP_LOAD_REORDER", "")

        for each in json_dict["passes"]:
            if "passDpuParam" in each:
                if "xcompilerAttrs" not in each["passDpuParam"]:
                    continue
                # each["passDpuParam"]["xcompilerAttrs"] = {}
                if gen_bd != "":
                    each["passDpuParam"]["xcompilerAttrs"]["gen_bd"] = {
                        "boolValue": bool(gen_bd == "true")
                    }
                if set_conv_aie_mode != "":
                    each["passDpuParam"]["xcompilerAttrs"]["set_conv_aie_mode"] = {
                        "uintValue": int(set_conv_aie_mode)
                    }
                if specify_pdis != "":
                    each["passDpuParam"]["xcompilerAttrs"]["specify_pdis"] = {
                        "intValues": [int(specify_pdis)]
                    }
                if enable_ac_tek_opti != "":
                    each["passDpuParam"]["xcompilerAttrs"]["enable_ac_tek_opti"] = {
                        "uintValue": int(enable_ac_tek_opti)
                    }
                if verify_slnode != "":
                    each["passDpuParam"]["xcompilerAttrs"]["verify_slnode"] = {
                        "boolValue": bool(verify_slnode == "true")
                    }
                if prefetch_lp_load_reorder != "":
                    each["passDpuParam"]["xcompilerAttrs"][
                        "prefetch_lp_load_reorder"
                    ] = {"boolValue": bool(prefetch_lp_load_reorder == "true")}

    except Exception as e:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)
        logging.warning(f"!!! warning :write new vaip_config.json failed! {e}.)")


def update_json_dict(json_dict, env_name, kw, val):
    try:
        for each in json_dict["passes"]:
            if "passDpuParam" in each:
                if "xcompilerAttrs" not in each["passDpuParam"]:
                    continue
                each["passDpuParam"]["xcompilerAttrs"][env_name.lower()] = {kw: val}

        for each in json_dict["targets"]:
            if each.get("target_opts", {}).get("xcompilerAttrs", {}):
                each["target_opts"]["xcompilerAttrs"][env_name.lower()] = {kw: val}
    except Exception as e:
        tb = traceback.format_exc()
        if not tb is None:
            print(tb)
        logging.warning(f"!!! warning :write new vaip_config.json failed! {e}.)")


def update_from_env(
    curr_env, xcompiler_attrs, injected_tiling_param, inject_order_json, json_dict
):
    # try:
    #     for i in range(0, len(json_dict["targets"])):
    #         if "xclbin" in json_dict["targets"][i]:
    #             json_dict["targets"][i].pop("xclbin")
    # except Exception as e:
    #     logging.warning(f"!!! warning :write new vaip_config.json failed! {e}.)")

    if curr_env.get("DPU_SUBGRAPH_NUM", ""):
        update_json_dict(
            json_dict,
            "DPU_SUBGRAPH_NUM",
            "uintValue",
            int(curr_env.get("DPU_SUBGRAPH_NUM", "")),
        )

    if curr_env.get("FORCE_MODE", ""):
        update_json_dict(
            json_dict,
            "FORCE_MODE",
            "uintValue",
            int(curr_env.get("FORCE_MODE")),
        )

    if curr_env.get("OPT_LEVEL", ""):
        update_json_dict(
            json_dict,
            "OPT_LEVEL",
            "uintValue",
            int(curr_env.get("OPT_LEVEL")),
        )

    if curr_env.get("TRACK_CODE_GEN", ""):
        update_json_dict(
            json_dict,
            "TRACK_CODE_GEN",
            "boolValue",
            bool(curr_env.get("TRACK_CODE_GEN", "false") == "true"),
        )

    if curr_env.get("PREFETCH_LP", ""):
        update_json_dict(
            json_dict,
            "PREFETCH_LP",
            "intValue",
            int(curr_env.get("PREFETCH_LP")),
        )

    if curr_env.get("VERIFY_SLNODE", ""):
        update_json_dict(
            json_dict,
            "VERIFY_SLNODE",
            "boolValue",
            int(curr_env.get("VERIFY_SLNODE")),
        )

    if curr_env.get("DUMP_SUBGRAPH_OPS", ""):
        update_json_dict(
            json_dict,
            "DUMP_SUBGRAPH_OPS",
            "boolValue",
            bool(curr_env.get("DUMP_SUBGRAPH_OPS", "false") == "true"),
        )

    if curr_env.get("USE_GEMM_KERNEL", ""):
        update_json_dict(
            json_dict,
            "USE_GEMM_KERNEL",
            "boolValue",
            bool(curr_env.get("USE_GEMM_KERNEL", "false") == "true"),
        )

    if curr_env.get("AIE_PROFILE_PERFORMANCE", ""):
        local_workspace = curr_env.get("WORKSPACE", "")
        perf_control_config = os.path.join(
            local_workspace, "ci", "tools", "config", "perf_control_config.json"
        ).replace("/", "\\")

        update_json_dict(
            json_dict,
            "perf_profile_cfg",
            "stringValue",
            perf_control_config,
        )

    if curr_env.get("ENABLE_FAST_PM", ""):
        update_json_dict(
            json_dict,
            "ENABLE_FAST_PM",
            "boolValue",
            bool(curr_env.get("ENABLE_FAST_PM", "false") == "true"),
        )

        pdi_elf = curr_env.get("PDI_ELF_PATH", "")
        if pdi_elf != "":
            update_json_dict(
                json_dict,
                "PDI_ELF_PATH",
                "stringValue",
                pdi_elf,
            )

    if curr_env.get("PROFILE", ""):
        update_json_dict(
            json_dict, "PROFILE", "uintValue", int(curr_env.get("PROFILE", "0"))
        )

    if curr_env.get("EXTEND_DDR_LAYOUT", ""):
        update_json_dict(
            json_dict,
            "EXTEND_DDR_LAYOUT",
            "intValue",
            int(curr_env.get("EXTEND_DDR_LAYOUT")),
        )

    if curr_env.get("ENABLE_FM_MT2AIE2_ITERS", ""):
        update_json_dict(
            json_dict,
            "ENABLE_FM_MT2AIE2_ITERS",
            "boolValue",
            bool(curr_env.get("ENABLE_FM_MT2AIE2_ITERS", "false") == "true"),
        )

    if curr_env.get("ADVANCED_OPT", ""):
        update_json_dict(
            json_dict,
            "ADVANCED_OPT",
            "boolValue",
            bool(curr_env.get("ADVANCED_OPT", "false") == "true"),
        )

    if curr_env.get("ENABLE_EVENTILING", ""):
        update_json_dict(
            json_dict,
            "ENABLE_EVENTILING",
            "boolValue",
            bool(curr_env.get("ENABLE_EVENTILING", "false") == "true"),
        )

    if curr_env.get("ENABLE_OVERHEAD_CALCULATION", ""):
        update_json_dict(
            json_dict,
            "ENABLE_OVERHEAD_CALCULATION",
            "boolValue",
            bool(curr_env.get("ENABLE_OVERHEAD_CALCULATION", "false") == "true"),
        )

    if curr_env.get("DUMP_MCCODE", ""):
        update_json_dict(
            json_dict,
            "DUMP_MCCODE",
            "boolValue",
            bool(curr_env.get("DUMP_MCCODE", "false") == "true"),
        )

    if curr_env.get("ENABLE_MODE2", ""):
        update_json_dict(
            json_dict,
            "ENABLE_MODE2",
            "boolValue",
            bool(curr_env.get("ENABLE_MODE2", "false") == "true"),
        )

    if curr_env.get("ENABLE_WEIGHTS_PREFETCH", ""):
        update_json_dict(
            json_dict,
            "ENABLE_WEIGHTS_PREFETCH",
            "boolValue",
            bool(curr_env.get("ENABLE_WEIGHTS_PREFETCH", "false") == "true"),
        )

    if curr_env.get("ENABLE_COST_MODEL_TILING", ""):
        update_json_dict(
            json_dict,
            "ENABLE_COST_MODEL_TILING",
            "boolValue",
            bool(curr_env.get("ENABLE_COST_MODEL_TILING", "false") == "true"),
        )

    if curr_env.get("ENABLE_MERGESYNC", ""):
        update_json_dict(
            json_dict,
            "ENABLE_MERGESYNC",
            "boolValue",
            bool(curr_env.get("ENABLE_MERGESYNC", "false") == "true"),
        )

    if curr_env.get("ENABLE_CONTROL_OPTIMIZATION", ""):
        update_json_dict(
            json_dict,
            "ENABLE_CONTROL_OPTIMIZATION",
            "boolValue",
            bool(curr_env.get("ENABLE_CONTROL_OPTIMIZATION", "false") == "true"),
        )

    if curr_env.get("XLNX_ENABLE_OP_NAME_PROTECTION", ""):
        update_json_dict(
            json_dict,
            "enable_op_tensor_name_protection",
            "boolValue",
            bool(curr_env.get("XLNX_ENABLE_OP_NAME_PROTECTION", "0") == "1"),
        )

    if curr_env.get("ENABLE_SHIM_DMA_BD_CONFIG", ""):
        update_json_dict(
            json_dict,
            "ENABLE_SHIM_DMA_BD_CONFIG",
            "boolValue",
            bool(curr_env.get("ENABLE_SHIM_DMA_BD_CONFIG", "false") == "true"),
        )

    if curr_env.get("VAIP_COMPILE_RESERVE_CONST_DATA", ""):
        update_json_dict(
            json_dict,
            "reserve_const_data",
            "boolValue",
            bool(curr_env.get("VAIP_COMPILE_RESERVE_CONST_DATA", "false") == "true"),
        )

    if curr_env.get("PREASSIGN", ""):
        update_json_dict(
            json_dict,
            "PREASSIGN",
            "boolValue",
            bool(curr_env.get("PREASSIGN", "false") == "true"),
        )

    if curr_env.get("ENABLE_CONV_MODE0_MT_FUSION", ""):
        update_json_dict(
            json_dict,
            "ENABLE_CONV_MODE0_MT_FUSION",
            "boolValue",
            bool(curr_env.get("ENABLE_CONV_MODE0_MT_FUSION", "false") == "true"),
        )

    if curr_env.get("ENABLE_MT_FUSION", ""):
        update_json_dict(
            json_dict,
            "ENABLE_MT_FUSION",
            "boolValue",
            bool(curr_env.get("ENABLE_MT_FUSION", "false") == "true"),
        )

    if curr_env.get("AIE_TILE_FUSION_TYPE", ""):
        update_json_dict(
            json_dict,
            "AIE_TILE_FUSION_TYPE",
            "stringValues",
            curr_env.get("AIE_TILE_FUSION_TYPE", "").split(","),
        )

    if curr_env.get("ENABLE_AIE_TILE_FUSION", ""):
        update_json_dict(
            json_dict,
            "ENABLE_AIE_TILE_FUSION",
            "stringValues",
            curr_env.get("ENABLE_AIE_TILE_FUSION", "").split(","),
        )

    if curr_env.get("CONVERT_SOFTMAX_TO_HARD_SOFTMAX", ""):
        vaip_perf_workaround(
            json_dict,
            "CONVERT_SOFTMAX_TO_HARD_SOFTMAX",
            curr_env.get("CONVERT_SOFTMAX_TO_HARD_SOFTMAX", ""),
        )

    if curr_env.get("XLNX_TARGET_NAME", "") == "DPUCV3DX8G_ISA0_C8SP1":
        workaround_4x2(curr_env, json_dict)

    if curr_env.get("SPECIFIED_COL_NUM", ""):
        update_json_dict(
            json_dict,
            "SPECIFIED_COL_NUM",
            "uintValue",
            int(curr_env.get("SPECIFIED_COL_NUM")),
        )

    local_csv = ""
    if injected_tiling_param:
        local_csv = os.path.split(str(injected_tiling_param.as_posix()))[-1]
    inject_flag = curr_env.get("ENABLE_PARAM_INJECTION", "false") == "true"
    if inject_flag and local_csv != "":
        update_json_dict(
            json_dict,
            "INJECTED_TILING_PARAM",
            "stringValue",
            local_csv,
        )

    local_json = ""
    if inject_order_json:
        local_json = os.path.split(str(inject_order_json.as_posix()))[-1]
    inject_flag = curr_env.get("ENABLE_INJECT_ORDER", "false") == "true"
    if inject_flag and local_json != "":
        update_json_dict(
            json_dict,
            "INJECT_ORDER",
            "stringValue",
            local_json,
        )

    if xcompiler_attrs:
        for each in json_dict["passes"]:
            if "passDpuParam" not in each:
                continue
            if "xcompilerAttrs" not in each["passDpuParam"]:
                continue
            print(xcompiler_attrs, flush=True)
            for opt, values in xcompiler_attrs.items():
                for kw, val in values.items():
                    each["passDpuParam"]["xcompilerAttrs"][opt] = {kw: val}

    if curr_env.get("CI_VAIP_FLOW", "") == "xcompiler":
        for each in json_dict["mepTable"]:
            if (
                each["modelName"].find("PSS") != -1
                or each["modelName"].find("PST") != -1
            ):
                each["target"] = "RyzenAI_vision_config_3_mha"

    if curr_env.get("SHARE_HW_CONTEXT", "true") == "false":
        for each in json_dict["targets"]:
            if "share_hw_context" in each:
                each["share_hw_context"] = False

    if curr_env.get("XLNX_VART_FIRMWARE", ""):
        target_xclbin = curr_env["XLNX_VART_FIRMWARE"]
        target_names = {
            each["target"] for each in json_dict["mepTable"] if each.get("target")
        }
        for target in json_dict["targets"]:
            if target.get("name") in target_names:
                target["xclbin"] = target_xclbin
