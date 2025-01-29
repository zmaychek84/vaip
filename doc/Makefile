#
#  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#  Licensed under the MIT License.
#

# sudo env https_proxy=$https_proxy npm install -g @mermaid-js/mermaid-cli
%.svg: %.mmd
	mmdc -p puppeteer-config.json -i $<  -o $@
