
def createRegenerateParams(){
    return [choice(
        name: 'regenerateParams',
        choices: ['','user_4x4', 'user_4x2', 'user_2x4x1', 'developer_mode', 'voe_build'], 
        description: 'Regenerate all the parameters',
	)]
}

def paramCleanWorkspace(){
   return [booleanParam(
        name: 'CleanWorkspace',
        defaultValue: false, 
        description: '',
    )]
}

def paramDumpMCCode(){
    return [booleanParam(
        name: 'DUMP_MC_CODE',
        defaultValue: false, 
        description: 'Chose this option to dump ac_code.txt and mc_code.txt and will store in build_log*.zip',
	)]
}

def paramReleaseFile(){
    return [string(
        name: "RELEASE_FILE",
        defaultValue: 'release_file/latest_dev.txt', 
        trim: true,
        description: '''release_file/latest_dev.txt
release_file/latest_stx.txt'''
    )]
}

def paramIgnoreBuild(){
    return [booleanParam(
        name: 'IGNORE_BUILD',
        defaultValue: false, 
        description: '''If the VAI_RT_BRANCH defined prID  have already built, and the code have no change,
then we can chose IGNORE_BUILD and using the current prID's package.''',
	)]
}

def paramNode(){
    return [string(
        name: 'NODE', 
        defaultValue: 'xcowin_server', 
        trim: true,
        description: "Compile server"
    )]
}

def paramBoard(board){
    return [string(
        name: 'BOARD', 
        defaultValue: board, 
        trim: true,
        description: '''stx_benchmark: xsjstrix17/xsjstrix19/xsjstrix20/xsjstrix21
xsjstrix13
xsjstrix17
xsjstrix19
xsjstrix20
xsjstrix21'''
    )]
}

def paramVairtBranch(){
    return [string(
        name: 'VAI_RT_BRANCH', 
        defaultValue: '', 
        trim: true,
        description: '''Support branch and pr build
pr for build: pr1234
branch for build: dev'''
    )]
}

def paramGlobalPackage(){
    return [string(
        name: 'GLOBAL_PACKAGE', 
        defaultValue: '', 
        trim: true,
        description: '''NOTE: priority higher than VAI_RT_BRANCH, using other site prebuilt package.
latest package example:
https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/voe_test_package/latest/voe-win_amd64-with_xcompiler_on-latest_dev.zip
pr package example:
http://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/voe_test_package/pr_verify/pr1560/voe-win_amd64-with_xcompiler_on-2b80295-latest_4x2_dev.zip'''
    )]
}

def paramModelzoo(modelzoo){
    return [string(
        name: "MODEL_ZOO",
        defaultValue: modelzoo, 
        trim: true,
        description: 'Model zoo file name'
    )]
}

def paramUserModelPath(){
    return [string(
        name: "USER_ONNX_PATH",
        defaultValue: '', 
        trim: true,
        description: '''NOTE: priority higher than MODEL_ZOO
Support user's onnx or xmodel directory'''
    )]
}


def paramCaseName(){
    return [string(
        name: "CASE_NAME", 
        defaultValue: '', 
        trim: true,
        description: '''specify models to test, ex:
A3
A3 C2
If not specify models, default run https://gitenterprise.xilinx.com/VitisAI/vaip/blob/dev/ci/ipu_benchmark_real_data.json except FORBID_CASE'''
    )]
}

def paramForbidCaseName(forbid){
    return [string(
        name: "FORBID_CASE",
        defaultValue: forbid, 
        trim: true,
        description: ''
    )]
}

def paramProfile(profile){
    return [string(
        name: "PROFILE",
        defaultValue: profile, 
        trim: true,
        description: '''3: Layer level profile
2: PDIlevel profile
1: 4x2 profile
0: No profile
7: internal debug'''
    )]
}

def paramUserEnv(user_env){
    return [string(
        name: "USER_ENV",
        defaultValue: user_env, 
        trim: true,
        description: '''xcompiler options:
PROFILE, OPT_LEVEL,ENABLE_OVERHEAD_CALCULATION, DPU_SUBGRAPH_NUM, XLNX_TARGET_NAME, ENABLE_COST_MODEL_TILING, ENABLE_WEIGHTS_PREFETCH, ENABLE_MERGESYNC, ENABLE_FAST_PM, USE_GEMM_KERNEL, EXTEND_DDR_LAYOUT, ENABLE_FM_MT2AIE2_ITERS, ENABLE_CONTROL_OPTIMIZATION, ENABLE_SHIM_DMA_BD_CONFIG, PREASSIGN, CONVERT_SOFTMAX_TO_HARD_SOFTMAX, ENABLE_PARAM_INJECTION

vaip options:
XLNX_MINIMUM_NUM_OF_CONV XLNX_ENABLE_OP_NAME_PROTECTION'''
    )]
}

def paramCiEnv(ci_env){
    return [string(
        name: "CI_ENV",
        defaultValue: ci_env,
        trim: true,
        description: '''profiling: VAITRACE_PROFILING, VAI_AIE_OVERCLOCK_MHZ
ci option: TIMEOUT(model timeout time, unit s)
accuracy: ACCURACY_TEST=true

use TIMEOUT to control test time for each case'''
    )]
}

def paramVartFirmware(firmware){
    return [string(
        name: "XLNX_VART_FIRMWARE",
        defaultValue: firmware, 
        trim: true,
        description: '''xcd path example:  /group/dphi_arch/xcd_jenkins/xclbins/stx-ipu-board-test-1/2225/AMD_AIE2P_4x4_Overlay_CFG0.xclbin
xsj path example: /proj/xsjhdstaff6/AMD_AIE2P_4x4_Overlay_CFG0.xclbin
xclbin daily:
https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest_strix/AMD_AIE2P_4x4_Overlay_CFG0.xclbin'''
    )]
}

def paramPdiElf(pdi_elf){
    return [string(
        name: "PDI_ELF",
        defaultValue: pdi_elf, 
        trim: true,
        description: '''xcd control package directory example:   /proj/xsjhdstaff1/AMD_AIE2P_4x4_Overlay_CFG0_pdi_ctl_pkts
xsj control package directory example:    /proj/xsjhdstaff6/AMD_AIE2P_4x4_Overlay_CFG0_pdi_ctl_pkts

xcoatrifactory pdi elf zip link example:
https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest_strix/AMD_AIE2P_4x4_Overlay_CFG0_pdi_ctl_pkts.zip
Note: this param will ignore if ENABLE_FAST_PM=false'''
    )]
}

def paramPipelineRepo(){
    return [string(
        name: "PIPELINE_REPO",
        defaultValue: 'https://gitenterprise.xilinx.com/VitisAI/vaip.git', 
        trim: true,
        description: ''
    )]
}

def paramPipelineBranch(){
    return [string(
        name: "PIPELINE_BRANCH",
        defaultValue: 'dev', 
        trim: true,
        description: ''
    )]
}

def paramCustomOpDll(){
    return [string(
        name: 'CUSTOM_OP_DLL', 
        defaultValue: '/group/modelzoo/vai_q_onnx/release_package/20240319/vai_custom_op.dll', 
        trim: true,
        description: "custom op dll"
    )]
}

def paramModelType(){
    return [choice(
        name: "MODEL_TYPE",
        choices: ['onnx', 'xmodel'], 
        description: 'onnx: test onnx model\nxmodel: test compiled xmodel'
    )]
}

def paramTestMode(){
    return [choice(
        name: "TEST_MODE",
        choices: ['performance', 'accuracy', 'functionality'], 
        description: '''performance: test model performance and profiling
accuracy: test model accuracy
functionality: test model functionality'''
    )]
}

def createOutputChecking(){
    return [[
        $class: 'CascadeChoiceParameter',
        name: "OUTPUT_CHECKING",
        referencedParameters: "TEST_MODE,MODEL_TYPE",
        script: [
            $class: 'GroovyScript',
            script: [
                $class: 'SecureGroovyScript',
                script: """
                    if (MODEL_TYPE == 'xmodel'){
                        if (TEST_MODE != 'performance'){
                            return ['']
                        }
                        return ['cpu_runner,ipu']
                    }
                    if (TEST_MODE == 'performance'){
                        return ['cpu_runner,onnx_ep', 'false']
                    }
                    return ['cpu_ep,onnx_ep', 'cpu_ep,cpu_runner', 'cpu_runner,onnx_ep', 'cpu_ep,cpu_ep', 'cpu_ep,golden_file']
                """,
                sandbox: true,
                classpath: null
            ],
            fallbackScript: [
                $class: 'SecureGroovyScript',
                script: """return ["No target found"]""",
                sandbox: true,
                classpath: null
            ]
        ],
        choiceType: 'PT_SINGLE_SELECT',
        filterable: false,
        description: ""
    ]]
}


def generateParams(){
    def mode = params.regenerateParams
    def generatedParameters = []
    def user_env = 'OPT_LEVEL=65536 ENABLE_OVERHEAD_CALCULATION=true ENABLE_FAST_PM=true'
    def ci_env = 'TIMEOUT=300 VAITRACE_PROFILING=true VAI_AIE_OVERCLOCK_MHZ=1810'
    def modelzoo = 'ipu_benchmark_real_data'
    def profile = '3'
    def board = 'stx_benchmark'
    def forbid = ''
    def firmware = ''
    def pdi_elf = ''
    if (mode == 'user_4x4') {
        forbid = 'F1 F2'
        firmware = 'https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest_strix/AMD_AIE2P_4x4_Overlay_CFG0.xclbin'
        pdi_elf = 'https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest_strix/AMD_AIE2P_4x4_Overlay_CFG0_pdi_ctl_pkts.zip'
    } else if (mode == 'user_4x2') {
        profile = '1'
        modelzoo = '4x2_debug'
        user_env = 'ENABLE_AIE_TILE_FUSION=conv2d_elew_fusion EXTEND_DDR_LAYOUT=2 ENABLE_CONTROL_OPTIMIZATION=true ENABLE_MERGESYNC=true'
        board = 'xsjstrix38'
    } else if (mode == 'user_2x4x1') {
        modelzoo = 'ipu_benchmark_PSO_PSA'
        firmware = 'https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest_strix/AMD_AIE2P_2x4x1_Overlay.xclbin'
        pdi_elf = 'https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/xclbin/latest_strix/AMD_AIE2P_2x4x1_Overlay_pdi_ctl_pkts.zip'
        user_env = 'XLNX_USE_SHARED_CONTEXT=0 ENABLE_FAST_PM=true XLNX_ENABLE_BATCH=0 XLNX_ENABLE_OLD_QDQ=0 PROFILE=3 OPT_LEVEL=3 ENABLE_WEIGHTS_PREFETCH=true ENABLE_MERGESYNC=true ENABLE_COST_MODEL_TILING=true'
        ci_env = 'TEST_TIME=60 VAITRACE_PROFILING=true VAI_AIE_OVERCLOCK_MHZ=1800'
    }
   
    generatedParameters += createRegenerateParams()
    generatedParameters += paramCleanWorkspace()
    generatedParameters += paramDumpMCCode()
    generatedParameters += paramNode()
    generatedParameters += paramReleaseFile()
    generatedParameters += paramIgnoreBuild()
    generatedParameters += paramVairtBranch()
    generatedParameters += paramGlobalPackage()
    generatedParameters += paramBoard(board)
    generatedParameters += paramModelzoo(modelzoo)
    generatedParameters += paramUserModelPath()
    generatedParameters += paramCaseName()
    generatedParameters += paramForbidCaseName(forbid)
    generatedParameters += paramProfile(profile)
    generatedParameters += paramUserEnv(user_env)
    generatedParameters += paramCiEnv(ci_env)
    generatedParameters += paramVartFirmware(firmware)
    generatedParameters += paramPdiElf(pdi_elf)
    generatedParameters += paramModelType()
    generatedParameters += paramTestMode()
    generatedParameters += createOutputChecking()
    if (mode == 'user_4x2' || mode == 'developer_mode') {
        generatedParameters += paramCustomOpDll()
    }
    if (mode == 'developer_mode') {
        generatedParameters += paramPipelineRepo()
        generatedParameters += paramPipelineBranch()
    }

    properties(
        properties: [
            parameters(parameterDefinitions: generatedParameters),
        ]
    )
}

return this
