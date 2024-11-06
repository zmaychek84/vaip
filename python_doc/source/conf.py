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
import pathlib
import sys
import os

current_file_directory = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
target_directory = current_file_directory / ".." / ".." / "vaip" / "python"
sys.path.append(str(target_directory))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "vaip"
copyright = "2023, AMD developers"
author = "AMD developers"
release = "phx-release-81523"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.graphviz", "breathe"]

templates_path = ["_templates"]
exclude_patterns = []

breathe_default_project = "VAIP"

breathe_projects = {"VAIP": current_file_directory / ".." / "build" / "doxygen" / "xml"}
