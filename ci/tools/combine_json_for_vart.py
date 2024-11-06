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
import json
import os


def load_json(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"[CI][Combine_json_for_vart]Error: File not found - {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"[CI][Combine_json_for_vart]Error: Failed to decode JSON - {file_path}")
        raise


def save_json(data, file_path):
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except IOError as e:
        print(
            f"[CI][Combine_json_for_vart]Error: Failed to write to file - {file_path}"
        )
        raise


def convert_path(original_path, new_prefix):
    if not os.path.isabs(original_path):
        raise ValueError(f"[CI][Combine_json_for_vart]Invalid path: {original_path}")

    last_folder = os.path.basename(os.path.dirname(original_path))
    file_name = os.path.basename(original_path)
    new_path = os.path.join(new_prefix, last_folder, file_name)
    return new_path


def merge_json(json_a, json_b):
    json_c = []
    for key in json_a.keys():
        input_paths = [
            convert_path(path, "C:/win24_models_input") for path in json_a[key]
        ]
        output_paths = [
            convert_path(path, "C:/win24_models_golden") for path in json_b[key]
        ]

        merged_entry = []

        for input_path in input_paths:
            merged_entry.append({"type": "input", "file_name": input_path})

        for output_path in output_paths:
            merged_entry.append({"type": "output", "file_name": output_path})

        json_c.append(merged_entry)
    return json_c


def combine_json_for_vart(input_json_path, golden_json_path, combined_json_path):
    input_json = load_json(input_json_path)
    golden_json = load_json(golden_json_path)

    combined_json = merge_json(input_json, golden_json)

    save_json(combined_json, combined_json_path)

    print(f"[CI][Combine_json_for_vart]Json C has been saved to {combined_json_path}")


# Example function call
if __name__ == "__main__":
    input_json_path = "path/to/input_json.json"
    golden_json_path = "path/to/golden_json.json"
    combined_json_path = "path/to/combined_json.json"

    combine_json_for_vart(input_json_path, golden_json_path, combined_json_path)
