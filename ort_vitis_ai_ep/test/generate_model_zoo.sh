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

#!/bin/bash
set -e

cat model_zoo.list | while read id onnx xmodel ; do
	md5=(`md5sum $MODEL_ZOO_ROOT/$onnx || echo -n "not_found"`);
    if [ x$md5 == x"not_found" ]; then
        echo "onnx file is not found:  $MODEL_ZOO_ROOT/$onnx ignored." 1>&2
        continue;
    fi
    base_dir=$TARGET_MODEL_ROOT/$md5;
    mkdir -p $base_dir;
	echo "$base_dir/compiled.ref.xmodel: $MODEL_ZOO_ROOT/$xmodel";
    echo -ne "\t"; echo "$XCOMPILER -i $MODEL_ZOO_ROOT/$xmodel -o $base_dir/compiled.ref.xmodel -t $TARGET_NAME"

	echo -n "$base_dir/xir.xmodel ";
	echo -n "$base_dir/onnx.onnx ";
    echo -n "$base_dir/compiled.xmodel ";
    echo -n "$base_dir/build.log ";
    echo -n "$base_dir/onnx.txt ";
    echo -n "$base_dir/compile_status.txt: ";
	echo -n "$MODEL_ZOO_ROOT/$onnx $TEST_ONNX_RUNNER "
    echo -n `ldd $TEST_ONNX_RUNNER | grep 'vaip\|onnx' | awk '{print $3}'`
    echo
    declare args=(env);
    args+=(ENABLE_SAVE_GRAPH_TXT=1)
    args+=(XLNX_CACHE_DIR=\$\(TARGET_MODEL_ROOT\))
    args+=(XLNX_ENABLE_CACHE=1)
    args+=(XLNX_ENABLE_DUMP_XIR_MODEL=1)
    args+=(XLNX_ENABLE_DUMP_ONNX_MODEL=1)
    args+=(XLNX_ENABLE_DUMP_COMPILED_MODEL=1)
    args+=(XLNX_TARGET_NAME=\$\(TARGET_NAME\))
    args+=(DUMMY_RUNNER_BATCH_SIZE=1)
    args+=(DEBUG_VITIS_AI_EP_DUMMY_RUNNER=1)
    args+=(LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/Ubuntu.20.04.x86_64.Debug/lib)
    echo -ne "\t";
    if [ x"$GDB" == x"0" ]; then
        echo "set -o pipefail;rm $base_dir/xir.xmodel $base_dir/onnx.onnx $base_dir/compiled.xmodel $base_dir/build.log $base_dir/onnx.txt $base_dir/compile_status.txt || true;\\"
        echo "${args[@]} $TEST_ONNX_RUNNER $MODEL_ZOO_ROOT/$onnx 2>&1 | tee $base_dir/build.log && echo compile OK > $base_dir/compile_status.txt || echo compile FAIL >$base_dir/compile_status.txt"
    else
        echo "${args[@]} --args $TEST_ONNX_RUNNER $MODEL_ZOO_ROOT/$onnx"
    fi

	echo ".PHONY: $id run_$id compile_$id clean_$id show_$id";

    echo "$id: compile_$id run_$id"

    echo "compile: compile_$id";
	echo "compile_$id: $base_dir/compiled.ref.xmodel";
	echo "compile_$id: $base_dir/compiled.xmodel";
    echo -ne "\t"; echo @echo "$id" '`cat' "$base_dir/compile_status.txt" '`'

	echo "run_$id: $base_dir/run_status.txt";
    echo -ne "\t"; echo @echo "$id" '`cat' "$base_dir/run_status.txt" '`'

    echo "show_$id:"
    echo -ne "\t"; echo @echo "$id" '`cat' "$base_dir/compile_status.txt" ' 2>/dev/null || echo no compile `' '`cat' "$base_dir/run_status.txt" ' 2>/dev/null|| echo no run`'

    echo "show: show_$id";

    echo "$base_dir/run_status.txt: $base_dir/compiled.ref.xmodel $base_dir/compiled.xmodel run_model.sh"
    echo -e "\tset -o pipefail;bash run_model.sh $base_dir $md5 2>&1 | tee $base_dir/report.txt && echo run OK > $base_dir/run_status.txt || echo run FAIL > $base_dir/run_status.txt"
    echo "all: compile_$id run_$id"

    echo "run: run_$id";

	echo "clean_$id: ";
	echo -en "\t"; echo "rm $base_dir/compile_status.txt $base_dir/run_status.txt $base_dir/compiled.xmodel"
done
