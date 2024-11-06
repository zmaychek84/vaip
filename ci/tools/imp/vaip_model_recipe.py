#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free
#   and permissive binary-only license, but are not provided in source code form.  While the following free
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2022 Xilinx, Inc. All rights reserved.
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC.
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#


import logging
from threading import Lock, Thread

mutex = Lock()


def compare_or_set(recipe, ref, key):
    if key in recipe:
        if not key in ref:
            ref[key] = recipe[key]
        if ref[key] != recipe[key]:
            ref[
                "result_compare_ref"
            ] += f"compare {key} failed: {ref[key]} expected, actual value is {recipe[key]}\n"
    else:
        ref["result_compare_ref"] += f"compare {key} failed: has no key.\n"


def _compare_with_ref(recipe_json, ref_json):
    id = recipe_json["id"]
    if not id in ref_json:
        ref_json[id] = {"id": id}
    ref = ref_json[id]
    ref["result_compare_ref"] = ""
    ref["result"] = recipe_json["result"]
    ref["detail"] = recipe_json
    if ref["result"] != "OK":
        return
    compare_or_set(recipe_json, ref, "ir")
    compare_or_set(recipe_json, ref, "producer_name")
    compare_or_set(recipe_json, ref, "md5sum")
    compare_or_set(recipe_json, ref, "producer_version")
    if "context" in recipe_json:
        compare_meta_def(recipe_json["context"], ref)
    else:
        ref["result"] = f"FAILED@compare_meta_def no context"


def compare_meta_def(context, ref):
    if not "metaDef" in context:
        ref["result_compare_ref"] = f"no metadef"
        meta_defs = []
    else:
        meta_defs = context["metaDef"]
    if not "metaDef" in ref:
        ref["metaDef"] = [{} for i in range(len(meta_defs))]
    actual_len = len(meta_defs)
    expected_len = len(ref["metaDef"])
    if actual_len != expected_len:
        ref[
            "result_compare_ref"
        ] = f"wrong number of subgraphs: {expected_len} is expected, but actual value is {actual_len}"
        ref["result"] = "FAILED@compare_meta_def_len"
        return
    # for i in range(len(meta_defs)):
    #     actual_meta_def = meta_defs[i]
    #     expected_meta_def = ref['metaDef'][i]
    #     if not 'inputs' in expected_meta_def:
    #         expected_meta_def['inputs'] = actual_meta_def['inputs']
    #     if not 'outputs' in expected_meta_def:
    #         expected_meta_def['outputs'] = actual_meta_def['outputs']
    #     if sorted(actual_meta_def['inputs']) != sorted(
    #             expected_meta_def['inputs']):
    #         ref['result_compare_ref'] = f"subgraphs[{i}] does not have same inputs"
    #         ref['result'] = f"FAILED@compare_meta_def_input{i}"
    #         return
    #     if sorted(actual_meta_def['outputs']) != sorted(
    #             expected_meta_def['outputs']):
    #         ref['result_compare_ref'] = f"subgraphs[{i}] does not have same outputs"
    #         ref['result'] = f"FAILED@compare_meta_def_outputs{i}"
    #         return


def compare_with_ref(recipe_json, ref_json):
    mutex.acquire()
    try:
        _compare_with_ref(recipe_json, ref_json)
    except:
        import traceback

        id = recipe_json["id"]
        ref_json[id]["result"] = str(traceback.format_exc())
    finally:
        mutex.release()
