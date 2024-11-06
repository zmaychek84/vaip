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
import sys
import os
import argparse
import subprocess

WOMBAT_MAP = {
    "xsjsharelib165": "10.228.214.223",
    "xsjstrix13": "10.23.187.149",
    "xsjstrix14": "10.23.187.204",
    "xsjstrix15": "10.23.187.43",
    "xsjstrix17": "10.23.187.197",
    "xsjstrix18": "10.23.187.86",
    "xsjstrix19": "10.23.187.213",
    "xsjstrix20": "10.23.187.32",
    "xsjstrix21": "10.23.187.51",
}


def reboot_xcd(board, addr, outlet):
    import pexpect

    retry_count = 3
    while retry_count > 0:
        try:
            child = pexpect.spawn("telnet %s 23" % addr)
            child.expect("User Name :")
            child.sendline("deephi")
            child.expect("Password  :")
            child.sendline("d33ph1")
            child.expect("apc>")
            child.sendline("olReboot %s" % outlet)
            child.expect("apc>")
            child.sendline("exit")
        except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
            retry_count -= 1
            pexpect_returncode = 1
        else:
            retry_count = 0
            pexpect_returncode = 0
    if pexpect_returncode == 0:
        print(f"Initiated Restart {board}", flush=True)
    else:
        print(f"Failed to Restart {board}!!!", flush=True)
    exit(pexpect_returncode)


if __name__ == "__main__":
    if os.environ.get("NODE", "").startswith("xcd"):
        board = os.environ.get("NODE", "")
        print(f"reboot board {board}", flush=True)
        reboot_xcd(board, sys.argv[1], sys.argv[2])
    else:
        board_id = sys.argv[1]
        unlock_user = sys.argv[2]
        unlock_pass = sys.argv[3]
        dpm_level = sys.argv[4]
        wombat_ip = WOMBAT_MAP.get(board_id, "")
        if not wombat_ip:
            sys.exit()
        if dpm_level == "7":
            script_file = os.path.join(
                os.environ.get("WORKSPACE", ""), "ci", "tools", "stxB0_unlock_dpm7.py"
            )
        elif dpm_level == "1":
            script_file = os.path.join(
                os.environ.get("WORKSPACE", ""), "ci", "tools", "stxB0_unlock_dpm1.py"
            )
        else:
            print("Unsupported dpm level", dpm_level, flush=True)
            sys.exit()
        args = [
            "python",
            script_file,
            "-ip",
            wombat_ip,
            "-unlock_user",
            unlock_user,
            "-unlock_pass",
            unlock_pass,
        ]
        try:
            subprocess.run(args, shell=True)
        except Exception as e:
            print(e)
