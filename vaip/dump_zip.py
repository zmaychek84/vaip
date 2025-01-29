##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
##  Licensed under the MIT License.
##
import sys


def read_zip(file):
    ret = []

    with open(file, "rb") as file:
        for byte in file.read():
            ret.append(byte)
    return ret


def main():
    zip_file = sys.argv[1]
    zip_content = read_zip(zip_file)

    output_str = ""
    for b in zip_content:
        output_str += str(b)
        output_str += ","
    output_str = output_str[:-1]

    print(output_str)


if __name__ == "__main__":
    main()
