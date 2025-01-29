#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#

dev:
	python setup.py develop --user
test:
	python setup.py test

passes += const_fold_batchnorm_to_scale
passes += merge_fix

png:
	for i in $(passes); do \
		python -mvoe.vis_pass  -m voe.passes.$$i -o ./vaip/passes/$$i.dot; \
	done

.PHONY: build
build:
	p=vaip; cmake --build ~/build/build.Ubuntu.20.04.x86_64.Debug/$$p   --verbose -j $$(nproc) && cmake --install ~/build/build.Ubuntu.20.04.x86_64.Debug/$$p

test_5: build
	~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/vaip/voe_py_pass \
	-i /tmp/$(USER)/vaip/.cache/d111d3ee33c942c0b6f1134bc6a0f56b/vaip.*.remove_reshape_fix.action_0.onnx \
	-m voe.passes.const_fold_batchnorm_to_scale2 \
	-t /tmp/a.onnx.txt \
	-o /tmp/a.onnx \
	-c /tmp/$(USER)/vaip/.cache/d111d3ee33c942c0b6f1134bc6a0f56b/ \

test_merge_fix: build
	~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/vaip/voe_py_pass \
	-i /tmp/$(USER)/vaip/.cache/b67dda654698f6fa1d77db9d2d9594f3/vaip.*.add_fix_after_const.action_0.onnx \
	-m voe.passes.merge_fix \
	-t /tmp/a.onnx.txt \
	-o /tmp/a.onnx \
	-c /tmp/$(USER)/vaip/.cache/b67dda654698f6fa1d77db9d2d9594f3/ \


test_merge_mul: build
	~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/vaip/voe_py_pass \
	-i /tmp/$(USER)/vaip/.cache/1fc7144c0c251d033f788d35e9a7745e/vaip.*.merge_hard_sigmoid.action_0.onnx \
	-m voe.passes.merge_mul \
	-t /tmp/a.onnx.txt \
	-o /tmp/a.onnx \
	-c /tmp/$(USER)/vaip/.cache/1fc7144c0c251d033f788d35e9a7745e/ \

test_around_add_transpose_reshape: build
	~/build/build.Ubuntu.20.04.x86_64.Debug/vaip/vaip/voe_py_pass \
	-i /tmp/$(USER)/vaip/.cache/c16d40525783acc93afa0ba77976863a/vaip.*.gc_after_layout_transform.action_0.onnx \
	-m voe.passes.add_transpose_reshape_transpose_reshape \
	-t /tmp/a.onnx.txt \
	-o /tmp/a.onnx \
	-c /tmp/$(USER)/vaip/.cache/c16d40525783acc93afa0ba77976863a/ \
