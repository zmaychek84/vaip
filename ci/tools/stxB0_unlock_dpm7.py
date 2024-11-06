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
########################################################################################################################

# Usage:
#
# 1. Set the HDT IP
# 2. Set the required DPM level
# 3. run the script


import sys
import ipulib
import argparse


MCDM_DRIVER = True
L2PG_DISABLE = True
MAX_PPT_ENABLE = True
MAX_CCX_ENABLE = True


def ForceDPM(dev, dpm, dpm_level):
    if dpm == "LCLK":
        # LCLK DPM Setting
        arg = 0x00020000 | dpm_level
        print(hex(arg))
        dev.mp1fw.send_message("TEST", "TESTSMC_MSG_ForceDpm", arg)
    elif dpm == "SOCCLK":
        # LCLK DPM Setting
        arg = 0x00040000 | dpm_level
        print(hex(arg))
        dev.mp1fw.send_message("TEST", "TESTSMC_MSG_ForceDpm", arg)
    else:
        print("Wrong argument for DPM. Exiting...")
        sys.exit(-1)


def ForceDFPState(dev, ipu, dfp_state):
    dev.mp1fw.send_message("TEST", "TESTSMC_MSG_ForceDfPstate", dfp_state, 1)


def setIpuHclkAndMpIpuClk(ipu, ipuhclk, mpipuclk):
    ipu.pmfw.msg.power_up_ipu()
    if MCDM_DRIVER == True:
        ipu.pmfw.set_clks(ipuhclk, mpipuclk)
    else:
        ipu.pmfw.msg.set_clks(ipuhclk, mpipuclk)
    ipu.pmfw.get_ipuhclk()
    ipu.pmfw.get_mpipuclk()

    # Disable DS in IPUHCLK and MPIPUCLK
    ipu.pysy.ppr.clk.clk7.socket0.iod.clk7.clk7_clk0_allow_ds_n0.clk0_ds_override = 1
    ipu.pysy.ppr.clk.clk7.socket0.iod.clk7.clk7_clk0_allow_ds_n0.clk0_allow_ds = 0
    ipu.pmfw.get_ipuhclk()

    ipu.pysy.ppr.clk.clk7.socket0.iod.clk7.clk7_clk1_allow_ds_n0.clk1_ds_override = 1
    ipu.pysy.ppr.clk.clk7.socket0.iod.clk7.clk7_clk1_allow_ds_n0.clk1_allow_ds = 0
    ipu.pmfw.get_mpipuclk()


def Disable_L2_PG(ipu):
    # This disables L2 Memory Power Gating
    ipu.pysy.ppr.iommul2.socket0.iod.iommul2.l2_l2_a_mempwr_gate_1.l2_areg_sd_en = 0
    ipu.pysy.ppr.iommul2.socket0.iod.iommul2.l2_l2_a_mempwr_gate_1.l2_areg_ls_en = 0
    ipu.pysy.ppr.iommul2.socket0.iod.iommul2.l2_l2_a_mempwr_gate_1.l2_areg_ds_en = 0
    ipu.pysy.ppr.iommul2.socket0.iod.iommul2.l2_l2_b_mempwr_gate_1.l2_breg_sd_en = 0
    ipu.pysy.ppr.iommul2.socket0.iod.iommul2.l2_l2_b_mempwr_gate_1.l2_breg_ls_en = 0
    ipu.pysy.ppr.iommul2.socket0.iod.iommul2.l2_l2_b_mempwr_gate_1.l2_breg_ds_en = 0
    ipu.pysy.ppr.iommul2.socket0.iod.iommul2.l2_pwrgate_cntrl_reg_3.ip_pg_en = 0


def ForceCCX_Clock(dev):
    # Papi command to set classic and dense core frequencies
    # Set Classic cores to 4 GHz
    dev.mp1fw.send_message("TEST", "TESTSMC_MSG_ForceAllCclkFrequency", 4000)

    # Set Dense cores to 3 GHz
    dev.mp1fw.send_message(
        "TEST", "TESTSMC_MSG_ForceAllCclkFrequency", 3000 + (1 << 28)
    )


def ForcePowerLimit(dev):
    stapm = 54000
    sppt = 90000
    fppt = 90000
    dev.mp1fw.send_message("TEST", "TESTSMC_MSG_SETSUSTAINEDPOWERLIMIT", stapm)
    dev.mp1fw.send_message("TEST", "TESTSMC_MSG_SETSLOWPPTLIMIT", sppt)
    dev.mp1fw.send_message("TEST", "TESTSMC_MSG_SETFASTPPTLIMIT", fppt)


def update_ipu_config(hdt_ip, user_id, password, dpm_level=7):
    hdt_ip = hdt_ip
    DPM_LEVEL = dpm_level

    if DPM_LEVEL == 1:
        ipuhclk = 1056  # sets 1056 MHz
        mpipuclk = 792  # sets 792 MHz
        dfp_state = 1  # this sets the fclk in MHz (0->400, 1->800, 2->1000 3->1200, 4->1400, 5->1600, 6 ->1800, 7-> 1960)
        dpmclk = "LCLK"  # name of the clk you need to change
        dpm_state = 1  # actual dpm setting you need for above signal (1->492 MHz)
    elif DPM_LEVEL == 7:
        ipuhclk = 1810  # 1800 #use ipuhclk
        mpipuclk = 1267  # 1200 #use mpipuclk
        dfp_state = 7  # this sets the fclk (0->400, 1->800, 2->1000 3->1200, 4->1400, 5->1600, 6 ->1800, 7-> 1960)
        dpmclk = "LCLK"  # name of the clk you need to change
        dpm_state = 7  # actual dpm setting you need for above signal (7->1039 MHz)
    else:
        ipuhclk = 800  # 1800 #use ipuhclk
        mpipuclk = 600  # 1200 #use mpipuclk
        dfp_state = 0  # this sets the fclk (0->400, 1->800, 2->1000 3->1200, 4->1400, 5->1600, 6 ->1800, 7-> 1960)
        dpmclk = "LCLK"  # name of the clk you need to change
        dpm_state = 0  # actual dpm setting you need for above signal (7->1039)

    ipu = ipulib.IPU(project="strix1", wombat=hdt_ip)
    from papi2 import runtime_config, initialize

    runtime_config.register_access_method = "hkysy"
    runtime_config.kysy_folder = r"C:\AMD\Kysy4"
    runtime_config.wombat_ip = hdt_ip
    runtime_config.wombat_user = "kysy"
    runtime_config.wombat_pass = "kysy"

    print("unlock hdt %s" % hdt_ip)
    ipu.wombat.unlock(user_id, password)

    dev, papi = initialize()

    desired_features = 1 << 15
    enabled_features = dev.mp1fw.send_message(
        "BIOS", "BIOSSMC_MSG_GETENABLEDSMUFEATURES"
    )
    if (enabled_features & desired_features) == desired_features:
        print("Features are enabled")
    else:
        print("Features are not enabled, trying to enable")
        dev.mp1fw.send_message(
            "BIOS", "BIOSSMC_MSG_ENABLESMUFEATURES", desired_features
        )
        enabled_features = dev.mp1fw.send_message(
            "BIOS", "BIOSSMC_MSG_GETENABLEDSMUFEATURES"
        )
        if (enabled_features & desired_features) == desired_features:
            print("Features are now enabled")
        else:
            print("Could not enable features")
            exit()

    # Set config
    ForceDFPState(dev, ipu, dfp_state)  # 0,1,2,3,4,5
    ForceDPM(dev, dpmclk, dpm_state)  # 0,1,2,3,4,5,6,7
    setIpuHclkAndMpIpuClk(ipu, ipuhclk, mpipuclk)
    if L2PG_DISABLE == True:
        Disable_L2_PG(ipu)
    if MAX_PPT_ENABLE == True:
        ForcePowerLimit(dev)
    if MAX_CCX_ENABLE == True:
        ForceCCX_Clock(dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-ip", help="")
    parser.add_argument("-unlock_user", help="")
    parser.add_argument("-unlock_pass", help="")
    args = parser.parse_args()
    hdt_id = args.ip
    user_id = args.unlock_user
    password = args.unlock_pass

    update_ipu_config(hdt_id, user_id, password)
