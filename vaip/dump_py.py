##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import sys


def read_py_file(file):
    ret = []
    with open(file, "r") as file:
        for line in file.readlines():
            for c in line:
                if c == "#":
                    break
                ret.append(ord(c))
    return ret


def main():
    py_file = sys.argv[1]
    py_content = read_py_file(py_file)

    output_str = ""
    for b in py_content:
        output_str += str(b)
        output_str += ","
    output_str = output_str[:-1]

    print(output_str)


if __name__ == "__main__":
    main()
