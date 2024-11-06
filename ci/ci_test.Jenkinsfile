import groovy.transform.Field

@Field
def globalCredentialsID = 'GHE'

@Field
def globalPackage = ''

@Field
//def suiteRunName = 'artifactory.xilinx.com/vitis-ai-docker-dev/aisw/dev:v5.0-46-gf767eb4'
def suiteRunName = "xcoartifactory.xilinx.com/vitis-ai-docker-dev-local/voe-dev-build:latest"

@Field
def gPackagePath = '/group/dphi_software/software/IPU_packages'

@Field
def gArtidir = "voe_test_package"

@Field
def gArtiSubdir = ""

@Field
def gLatestPackage = 'voe-win_amd64-with_xcompiler_on-latest_dev.zip'

@Field
def gDriverPath = "c:\\IPU_driver\\latest"

@Field
def gIPUWorkspace = "c:\\IPU_workspace"

@Field
def gOriginalDriver = ""

@Field
def gRessultPath = '/group/dphi_software/software/IPU_test_results'

@Field
def gXoahJson = ""

@Field
def gXoahJsonPerf = ""

@Field
def gOutputJson = ""

@Field
def gOutputHtml = ""

@Field
def gCompareHtml = ""

@Field
def gPerfHtml = ""

@Field
def gBuildLogZip = ""

@Field
def gCacheLogZip = ""

@Field
def gPerfJson = ""

@Field
def gAccJson = ""

@Field
def gSubmoduleCommit = ""

@Field
def gReleaseFile = ""

@Field
def gTestTxt = ""

@Field
def gDpuName = "cpu"

@Field
def gOptLevel = "0"

@Field
def gSubgraphNum = ""

@Field
def gModelzoo = ""

@Field
def gUserModelzoo = ""

@Field
def gTargetType = "STRIX"

@Field
def gTestDate = ""

@Field
def gTestCase = ""

@Field
def gSetupScript = ""

@Field
def gEmailHeaderExtend = ""

@Field
def gLinuxNode = "xcdl190260"

@Field
def gJumpHost = "xcdl190074.xilinx.com"

@Field
def gIgnoreBuild = true

@Field
def gSendMail = true

@Field
def gRebootBoard = false

@Field
def gBuildXclbin = false

@Field
def gVoeArchive = false

@Field
def gCINewFlow = true

@Field
def gNode = ""

@Field
def gTimeLimit = 24

@Field
def gTimeUnit = 'HOURS'

@Field
def gTestTimeLimit = 3600

@Field
def gTestTimeUnit = 'SECONDS'

@Field
def gCCList = ""

@Field
def gDpmLevel = 7

@Field
def gIncrementalFlag = false

def findPackage(postfix){
    if (postfix == "zip") {
        println "Search voe package *.${postfix}"
        globalPackage = findFiles(glob:"*.${postfix}")[0].name
        println "Found ${globalPackage}"
    } else if (postfix == "tgz") {
        println "Search voe package *.${postfix}"
        globalPackage = findFiles(glob:"*.${postfix}")[0].name
        println "Found ${globalPackage}"
        gLatestPackage = "voe-linux_amd64-with_xcompiler_on-latest_dev.${postfix}"
    }
    stashPackage()
}

def stashPackage() {
    stash name: "tmpzip", includes: "${globalPackage}"
}

def uploadCache(){
    def node_name = gLinuxNode
    if (env.COMPUTERNAME =~ /^(XSJ|XCO)/ || env.HOSTNAME =~ /^(xsj|xco)/ || gNode =~ /^(xsj|xco)/){
        node_name = "xsjncuph07"
    }
    node(node_name) {
        unstash name: "cacheZip"
        if (! fileExists(gCacheLogZip)){
            return
        }
        def tmp_cache_dir = "tmp_cache_dir"
        if (fileExists(tmp_cache_dir)){
            dir(tmp_cache_dir){ deleteDir() }
        }
        unzip zipFile: gCacheLogZip, dir: tmp_cache_dir

        dir(tmp_cache_dir){
            dir("cache"){
                def dlist = findFiles excludes: '', glob: ''
                dlist.sort()

                for (d in dlist){
                    println d.name
                    zip dir: d.name, zipFile: "${d.name}.zip"
                    if (! fileExists("${d.name}.zip")){
                        continue
                    }
                    sh "/home/${env.USER}/bin/jf rt u ${d.name}.zip PHX_test_case_package/compiled_xmodel/latest/${d.name}.zip"
                }
            }
            deleteDir()
        }
    }
}

def uploadPackage() {
    def node_name = gLinuxNode
    if (env.COMPUTERNAME =~ /^(XSJ|XCO)/ || env.HOSTNAME =~ /^(xsj|xco)/ || gNode =~ /^(xsj|xco)/){
        node_name = "xsjncuph07"
    }
    node(node_name) {
        dir(gPackagePath){
            if (gArtiSubdir =~ '_pr' || gArtiSubdir =~ 'pr_' || gArtiSubdir =~ env.JOB_BASE_NAME) {
                println "Clean old pr packages..."
                sh "rm -rf * || true"
            }
            unstash name: "tmpzip"
            if (params.UPLOAD_PACKAGE == null || params.UPLOAD_PACKAGE){
                try {
                    timeout(time: 10, unit: "MINUTES") {
                        println "Upload package to https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/${gArtidir}/${gArtiSubdir}/${globalPackage}"
                        sh "/home/${env.USER}/bin/jf rt u ${globalPackage} PHX_test_case_package/${gArtidir}/${gArtiSubdir}/${globalPackage}"
                        if (!( gArtiSubdir =~ '_pr' || gArtiSubdir =~ 'pr_') && params.VAI_RT_BRANCH != null && params.VAI_RT_BRANCH == '' && ("${env.JOB_BASE_NAME}" == "daily_voe_win_build" || "${env.JOB_BASE_NAME}" == "daily_voe_linux_build") && params.UPLOAD_PACKAGE == true) {
                            sh "cp -rf ${globalPackage} ${gLatestPackage}"
                            sh "/home/${env.USER}/bin/jf rt u ${gLatestPackage} PHX_test_case_package/${gArtidir}/latest/${gLatestPackage}"
                            println "Upload package to https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/${gArtidir}/latest/${gLatestPackage}"
                        }
                    }
                } catch(err) {
                    println err
                }
            }
        }
    }
}

def getLatestPackage() {
    globalPackage = "https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/${gArtidir}/latest/${gLatestPackage}"
    println "Found ${globalPackage}"
    return

    node(gLinuxNode) {
        dir(gPackagePath){
            globalPackage = gLatestPackage
            stash name: "tmpzip", includes: "${globalPackage}"
        }
    }
    unstash name: "tmpzip"
}

def getPrPackage() {
    node(gLinuxNode) {
        dir(gPackagePath){
            findPackage("zip")
        }
    }
    // unstash name: "tmpzip"
    globalPackage = "https://xcoartifactory.xilinx.com/artifactory/PHX_test_case_package/${gArtidir}/${gArtiSubdir}/${globalPackage}"
}

def unstashPackage() {
    if(params.GLOBAL_PACKAGE != null && params.GLOBAL_PACKAGE != '') {
        globalPackage = params.GLOBAL_PACKAGE
    } else if (params.VAI_RT_BRANCH != null && params.VAI_RT_BRANCH != '') {
        if (gIgnoreBuild) {
            getPrPackage()
        }
        unstash name: "tmpzip"
    } else if (globalPackage != ''){
        unstash name: "tmpzip"
    } else {
        getLatestPackage()
    }
}

def update_board_driver() {
    if(params.ORIGINAL_DRIVER == null || params.ORIGINAL_DRIVER == '') {
        if (fileExists("${gDriverPath}/README")) {
            try {
                dir(gDriverPath){
                    gOriginalDriver = readFile file: "README"
                }
            } catch(err) {
                println err
            }
        }
        return
    }
    stage("Install driver"){
        withCredentials([usernamePassword(credentialsId: globalCredentialsID, passwordVariable: 'password', usernameVariable: 'username')]) {
            def wgetCmd = "wget -q --no-check-certificate --http-user=${username} --http-passwd=${password}"
            def artifactoryLink = params.ORIGINAL_DRIVER.trim()
            def siliconZipName = "ipu_stack_rel_silicon"
            gOriginalDriver = params.ORIGINAL_DRIVER
            if (params.ORIGINAL_DRIVER =~ 'http') {
                if (fileExists(gDriverPath)) {
                    bat "rd /S /Q ${gDriverPath}"
                    bat "md ${gDriverPath}"
                }	    
                dir(gDriverPath){
                    bat "${wgetCmd} ${artifactoryLink}${siliconZipName}.zip"
                    unzip zipFile: "${siliconZipName}.zip", dir: siliconZipName
                    bat "cd ${siliconZipName} && echo y|amd_install_kipudrv.bat"
                }
            } else {
	        gDriverPath = artifactoryLink
                dir(gDriverPath) {
                    unzip zipFile: "${siliconZipName}.zip", dir: siliconZipName
                    bat "cd ${siliconZipName} && echo y|amd_install_kipudrv.bat"
                }
            }
        }
    }
}

def checkout_code(){
  try{
      stage('Checkout') {
          retry(2){
            sleep 2
            checkout scm
          }
      }
  } catch(err) {
      println err
  }
}


def xcd_cold_reboot(board){
    node('xcdl190255'){
        try{
            retry(2){
                sleep 2
                checkout scm
            }
        } catch(err) {
            println err
        }
        powerConfig = readJSON file:"${env.WORKSPACE}/ci/tools/ipu_board.json"
        powerConfig.each { name, info ->
            if (name == board) {
                sh "python3 ${env.WORKSPACE}/ci/tools/update_ipu_config.py ${info.power.strip_addr} ${info.power.outlet}"
                sleep 120
            }
        }
    }
}

def update_driver(board) {
  try{
    retry(2){
      timeout(time: 5) {
        if (board == gNode) {
            update_board_driver()
            return
        }
        stage('Update driver') {
          node(board){
              update_board_driver()
          }
        }
      }
    }
  } catch(err) {
      println err
      xcd_cold_reboot(board)
  }
}

def update_package() {
      stage('Update package') {
          unstashPackage()
      }
}

def verifyPackage() {
    stage('Verify') {
        if (params.USER_ONNX_PATH != null && params.USER_ONNX_PATH != '') {
            dir('ci'){
                unstash name: "gModelzooJson"
            }
        }
        if (isUnix() && params.TEST_IN_DOCKER) {
            testInDocker()
            return
        }
        if (! isUnix()) {
            try{
                def tracer_analyze_path = 'tracer_analyze'
                def fileExist = fileExists(tracer_analyze_path)
                if (fileExist) {
                    dir("tracer_analyze"){
                        deleteDir()
                    }
                }
            } catch(err) {
                println err
            }
        }
    }
}

def archivePackage() {
    if (params.IGNORE_ARCHIVE || gIgnoreBuild) {
        return
    }
    stage('Archive') {
        archiveArtifacts artifacts: "hb/${globalPackage}"
        archiveArtifacts artifacts: "hb/tmp/version_info.txt"
        // if archive here, setup_workspace*.py will fail
        // gVoeArchive = true
    }
    uploadPackage()
}

def run_test(model_zoo) {
    preProcess(get_ci_list())
    withEnv(["DRIVER_PATH=${gDriverPath}",
             "OUTPUT_JSON=${gOutputJson}",
             "OUTPUT_HTML=${gOutputHtml}",
             "COMPARE_HTML=${gCompareHtml}",
             "PERF_HTML=${gPerfHtml}",
             "PROFILING_EXCEL=${gProfilingCSV}",
             "BENCHMARK_RESULT_JSON=${gPerfJson}",
             "BENCHMARK_RESULT_TXT=${gTestTxt}",
             "SETUP_WORKSPACE_SCRIPT=${gSetupScript}",
             "XOAH_JSON=${gXoahJson}",
             "XOAH_JSON_PERF=${gXoahJsonPerf}",
             "BUILD_LOG_ZIP=${gBuildLogZip}",
             "CACHE_LOG_ZIP=${gCacheLogZip}",
	     "DPU_NAME=${gDpuName}",
	     "ORIGINAL_DRIVER=${gOriginalDriver}",
	     "MODEL_ZOO=${model_zoo}",
             "GLOBAL_PACKAGE=${globalPackage}",
	     "USER_MODEL_ZOO=${gUserModelzoo}"
        ]) {
            if(model_zoo =~ 'weekly'){
                gTimeLimit = 40
                gTimeUnit = 'HOURS'
            }
            try{
                if (isUnix()) {
                    sh "/bin/bash -c 'source /opt/xilinx/xrt/setup.sh && python ci/ci_main.py init_test -p ${globalPackage} && python ci/ci_main.py test_modelzoo -f ${get_ci_list()} && python ci/ci_main.py get_result -f ${get_ci_list()}'"
                } else {
                    if (! gCINewFlow) {
                        if (gIncrementalFlag) {
                            env.INCREMENTAL_TEST = true
                        }
                        timeout(time: gTimeLimit, unit: gTimeUnit) {
                            bat "python.exe ci/ci_main.py init_test -p ${globalPackage}"
                            bat "python.exe ci/ci_main.py test_modelzoo -f ${get_ci_list()}"
                            bat "python.exe ci/ci_main.py get_result -f ${get_ci_list()}"
                        }
                    } else {
                        if (gIncrementalFlag) {
                            env.INCREMENTAL_TEST = true
                        }
                        bat "python.exe ci/ci_main.py init_test -p ${globalPackage}"
                        if (env.CASE_NAME != "" && env.CASE_NAME != null) {
                            for (case_name in env.CASE_NAME.split(' ')) {
                                timeout(time: gTestTimeLimit, unit: gTestTimeUnit) {
                                    bat "python.exe ci/ci_main.py test_modelzoo -c ${case_name} -f ${get_ci_list()}"
                                }
                            }
                        } else {
                            def modelzoo_json = "ci/${model_zoo}.json"
                            if (fileExists(modelzoo_json)){
                                echo "reading ${modelzoo_json}"
                                def modelzoo = readJSON file: "${modelzoo_json}"
                                modelzoo.each { model ->
                                    def case_name = model.id
                                    timeout(time: gTestTimeLimit, unit: gTestTimeUnit) {
                                        bat "python.exe ci/ci_main.py test_modelzoo -c ${case_name} -f ${get_ci_list()}"
                                    }
                                }
                            } else {
                                echo "cannot read modelzoo json"
                                return
                            }
                        }
                        bat "python.exe ci/ci_main.py get_result -f ${get_ci_list()}"
                    }
                }
            } catch(err) {
                println err
                gIncrementalFlag = true
                if(env.CI_REBOOT_BOARD != "false"){
                    if (env.COMPUTERNAME =~ /^XCD/ || env.HOSTNAME =~ /^xcd/ || gNode =~ /^xcd/){
                        xcd_cold_reboot(gNode)
                    } else {
                        xsj_cold_reboot(gNode)
                    }
                    throw new Exception(err)
                }
            }
        }
        postProcess(get_ci_list())
}

def run_linux_test(env_cmd) {
    preProcess(get_ci_list())
    withEnv([ "OUTPUT_JSON=${gOutputJson}",
      "OUTPUT_HTML=${gOutputHtml}",
      "COMPARE_HTML=${gCompareHtml}",
      "PERF_HTML=${gPerfHtml}",
      "XOAH_JSON=${gXoahJson}",
      "PROFILING_EXCEL=${gProfilingCSV}",
      "BENCHMARK_RESULT_JSON=${gPerfJson}",
      "BENCHMARK_RESULT_TXT=${gTestTxt}",
      "XOAH_JSON_PERF=${gXoahJsonPerf}",
      "BUILD_LOG_ZIP=${gBuildLogZip}",
      "CACHE_LOG_ZIP=${gCacheLogZip}",
      "DPU_NAME=${gDpuName}",
      "MODEL_ZOO=${gModelzoo}",
      "XLNX_CACHE_DIR=${env.WORKSPACE}/build/.cache"
    ]) {
        sh "/bin/bash -c 'source /opt/xilinx/xrt/setup.sh && ${env_cmd} python ci/ci_main.py init_test -p ${globalPackage} && python ci/ci_main.py test_modelzoo -f ${get_ci_list()} -j 16 && python ci/ci_main.py get_result -f ${get_ci_list()}'"
    }
    postProcess(get_ci_list())
}

def buildInDocker(){
    def uid = sh(returnStdout: true, script: 'id -u').trim()
    def gid = sh(returnStdout: true, script: 'id -g').trim()
    sh "echo ${env.USER}:x:${uid}:${gid}::/home/${env.USER}:/bin/sh > ${env.WORKSPACE}/passwd"
    sh "mkdir -p .ssh; cp /home/${env.USER}/.ssh/* .ssh/"
    def extra_docker_option = ""
    def test_cmd = "/bin/bash -c 'source /opt/xilinx/xrt/setup.sh &&  python ci/ci_main.py build'"
    if (env.COMPUTERNAME =~ /^XCD/ || env.HOSTNAME =~ /^xcd/ || gNode =~ /^xcd/){
        test_cmd = "/bin/bash -c 'source /opt/xilinx/xrt/setup.sh && env https_proxy=http://xcdl190074:9181 python ci/ci_main.py build'"
        extra_docker_option = "-v /group/dphi_software:/group/dphi_software -v /opt/xilinx/dsa:/opt/xilinx/dsa -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins"
    } else {
        extra_docker_option = "-v /proj/xsjhdstaff6/${env.USER}:/proj/xsjhdstaff6/${env.USER}"
    }
    docker.image("${suiteRunName}").inside(""" --network=host -v ${env.WORKSPACE}:/home/${env.USER} -v ${env.WORKSPACE}/passwd:/etc/passwd -v /proj/rdi/staff/${env.USER}:/proj/rdi/staff/${env.USER} --entrypoint='' ${extra_docker_option} -v /dev/shm:/dev/shm -e USER=${env.USER} -e UID=${uid} -e GID=${gid} --security-opt seccomp=unconfined""") {
        if(params.CORSS_COMPILING) {
            def sdk_name = 'sdk-2023.1'
            def petalinux_path = "${env.WORKSPACE}/${sdk_name}"
            def sdk_path = "/group/dphi_software/software/petalinux_sdk/${sdk_name}.sh"
            if (! fileExists(petalinux_path)){
                sh " unset LD_LIBRARY_PATH; ${sdk_path} -y -d ${petalinux_path};  . ${petalinux_path}/environment-setup-*; rm -rf \${OECORE_TARGET_SYSROOT}/usr/share/cmake/XRT || true; python ci/ci_main.py build"
            } else {
                sh "unset LD_LIBRARY_PATH; . ${petalinux_path}/environment-setup-*; rm -rf \${OECORE_TARGET_SYSROOT}/install \${OECORE_TARGET_SYSROOT}/usr/share/cmake/XRT || true; python ci/ci_main.py build"
            }
        } else {
            sh test_cmd
        }
        if(fileExists("vai-rt/release_file_${env.BUILD_NUMBER}.txt")){
            archiveArtifacts artifacts: "vai-rt/release_file_${env.BUILD_NUMBER}.txt"
        }
    }
}

def testInDocker(){
    def uid = sh(returnStdout: true, script: 'id -u').trim()
    def gid = sh(returnStdout: true, script: 'id -g').trim()
    sh "echo ${env.USER}:x:${uid}:${gid}::/home/${env.USER}:/bin/sh > ${env.WORKSPACE}/passwd"
    sh "mkdir -p .ssh; cp /home/${env.USER}/.ssh/* .ssh/"
    sh "cp /home/${env.USER}/.gitconfig ."
    sh "cp /home/${env.USER}/.git-credentials ."
    env_cmd = ""
    def extra_docker_option = ""
    if (env.COMPUTERNAME =~ /^XCD/ || env.HOSTNAME =~ /^xcd/ || gNode =~ /^xcd/){
        env_cmd = "env https_proxy=http://xcdl190074:9181"
        extra_docker_option = "-v /group/dphi_software:/group/dphi_software -v /opt/xilinx/dsa:/opt/xilinx/dsa -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins"
    }
    docker.image("${suiteRunName}").inside(""" --network=host -v ${env.WORKSPACE}:/home/${env.USER} -v ${env.WORKSPACE}/passwd:/etc/passwd --entrypoint='' ${extra_docker_option} -v /proj/rdi/staff/${env.USER}:/proj/rdi/staff/${env.USER} -v /dev/shm:/dev/shm -e USER=${env.USER} -e UID=${uid} -e GID=${gid} """) {
        run_linux_test(env_cmd)
    }
}

def get_ci_list(){
    def ci_list = ['ipu_ci.env']
    if (params.CI_GROUP!= null && params.CI_GROUP != '') {
        ci_list = params.CI_GROUP.split(',')
    }
    return ci_list[0]
}

def get_modelzoo(){
    def modelzoo= ''
    if (params.USER_ONNX_PATH != null && params.USER_ONNX_PATH != '') {
        gModelzoo = "modelzoo_${env.BUILD_NUMBER}"
        modelzoo = generate_modelzoo()
    } else if (params.MODEL_ZOO != null && params.MODEL_ZOO != '') {
        modelzoo = params.MODEL_ZOO
    }
    return modelzoo
}

def generate_modelzoo(){
    try{
        def modelzoo_list = ''
        def onnx_path = params.USER_ONNX_PATH.trim()
        def postfix = "onnx"
        node(gLinuxNode){
            try{
              retry(2){
                  checkout scm
              }
            } catch(err) {
                println err
            }
            modelzoo_json = "${gModelzoo}.json"
            if (onnx_path.startsWith("http")) {
                sh "wget --quiet --no-check-certificate ${onnx_path}"
                def filename = onnx_path.tokenize('/')[-1]
                sh "rm -rf /proj/xsjhdstaff6/huizhan1/cache_zip"
                sh "unzip -o ${filename} -d /proj/xsjhdstaff6/huizhan1/cache_zip"
                sh "python3 ci/tools/make_modelzoo_json.py ${gJumpHost} /proj/xsjhdstaff6/huizhan1/cache_zip ${modelzoo_json}"
            } else {
                sh "python3 ci/tools/make_modelzoo_json.py ${gJumpHost} ${onnx_path} ${modelzoo_json}"
            }
            
            if (! fileExists(modelzoo_json)) {
                println "copy models from ${onnx_path}..."
                def dest = "/proj/rdi/staff/huizhan1/modelzoo/${env.JOB_BASE_NAME}"
                sh "mkdir -p ${dest}; chmod 777 ${dest} || true"
                sh "scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null xcdl190074.xilinx.com:${onnx_path} ${dest}"
                onnx_path = onnx_path.tokenize('/')[-1]
                sh "python3 ci/tools/make_modelzoo_json.py ${gJumpHost} ${dest}/${onnx_path} ${modelzoo_json}"
            }
            sleep 2
            stash name: "gModelzooJson", includes: modelzoo_json
            modelzoo_list = gModelzoo
            gUserModelzoo = gModelzoo
            archiveArtifacts artifacts: modelzoo_json
        }
        return modelzoo_list
    } catch(err) {
        println err
        return ''
    }
}

def testGroups(modelzoo){
    stage('Run test') {
        run_test(modelzoo)
    }
}

def preProcess(name) {
    if (params.XLNX_VART_FIRMWARE != null && params.XLNX_VART_FIRMWARE != ''){
        firmware_xclbin_name = params.XLNX_VART_FIRMWARE
        base_name = firmware_xclbin_name.split('/')[-1]
        if (firmware_xclbin_name.contains("/")){
            base_name = firmware_xclbin_name.split('/')[-1]
        } else {
            base_name = firmware_xclbin_name.split('\\\\')[-1]
        }
        println "----------------> " + base_name
        gDpuName = base_name.replace('.xclbin','')
        if (env.XLNX_TARGET_NAME == null || env.XLNX_TARGET_NAME == ''){
            env["XLNX_TARGET_NAME"] = gDpuName
        }
    }
    if (params.TARGET_TYPE != null && params.TARGET_TYPE != ''){
        gTargetType = params.TARGET_TYPE
    }

    def add_opt = false
    if (params.USER_ENV != null && params.USER_ENV != ""){
        env_list = params.USER_ENV.split(" ")
        env_list.each{
            each_env = it.split("=")
            if(each_env.size() > 1 && each_env[0] == "OPT_LEVEL"){
                gOptLevel = each_env[1]
            } else if(each_env.size() > 1 && each_env[0] == "TEST_DATE"){
                gTestDate = each_env[1]
            } else if(each_env.size() > 1 && each_env[0] == "TEST_CASE"){
                gTestCase = each_env[1]
            }
            if (params.TEST_MODE != "performance"){
                gEmailHeaderExtend += " ${each_env[0]}${each_env[1]}"
            }
        }
    }
    if (params.GLOBAL_PACKAGE != null && params.GLOBAL_PACKAGE =~ "onnx-rt.zip" ){
        gEmailHeaderExtend += " ONNX-RT"
    }
    if (! add_opt) {
        gEmailHeaderExtend += " OPT${gOptLevel}"
    }
    if (params.DPU_SUBGRAPH_NUM != null && params.DPU_SUBGRAPH_NUM != ""){
        gEmailHeaderExtend += " DPU_SUBGRAPH_NUM=${params.DPU_SUBGRAPH_NUM}"
        gSubgraphNum = params.DPU_SUBGRAPH_NUM
    } else {
        gEmailHeaderExtend += " "
        gSubgraphNum = ""
    }
    if (params.USER_ONNX_PATH != null && params.USER_ONNX_PATH != '') {
        gModelzoo = "modelzoo_${env.BUILD_NUMBER}"
    } else if (params.MODEL_ZOO != null && params.MODEL_ZOO != ''){
        gModelzoo = params.MODEL_ZOO
        if (params.MODEL_ZOO =~ ".json" ){
            base_name_json = params.MODEL_ZOO.split('/')[-1]
            gModelzoo = base_name_json.replace('.json','')
        }
    }
    def run_date = new java.text.SimpleDateFormat("yyyy-MM-dd").format(new Date())
    key_name = "${gTargetType}_${gModelzoo}_${gDpuName}_subgraph${gSubgraphNum}_opt${gOptLevel}"
    gOutputJson = "${name}_${key_name}_${env.BUILD_NUMBER}.json"
    gOutputHtml= "${name}_${key_name}_${env.BUILD_NUMBER}.html"
    gCompareHtml = "${key_name}_${name}_${env.BUILD_NUMBER}_compare.html"
    gPerfHtml = "${key_name}_${name}_${env.BUILD_NUMBER}_compare_performance.html"
    gXoahJson = "xoah_${key_name}_${env.BUILD_NUMBER}.json"
    gXoahJsonPerf = "xoah_${key_name}_${env.BUILD_NUMBER}_perf.json"
    gBuildLogZip = "build_log_${key_name}_${env.BUILD_NUMBER}.zip"
    gCacheLogZip = "cache_${key_name}_${env.BUILD_NUMBER}.zip"
    gPerfJson = "benchmark_result_${env.JOB_BASE_NAME}_${env.BUILD_NUMBER}_${run_date}.json"
    gTestTxt = "user_control_test_${run_date}.txt"
    gOpnameCsv = "${gTestCase}_silicon_time_${env.BUILD_NUMBER}.csv"
    gProfilingCSV = "vaitrace_profiling_${gTargetType}_${gDpuName}_${env.BUILD_NUMBER}_${run_date}"
    gSetupScript = "setup_workspace_${env.BUILD_NUMBER}.py"
}

def postProcess(name) {
    if (params.TEST_MODE != "performance"){
        try{
            archiveArtifacts artifacts: gOutputJson
        } catch(err) {
            println err
        }
        try{
            archiveArtifacts artifacts: gOutputHtml
        } catch(err) {
            println err
        }
    }
    try{
        if (fileExists(gCompareHtml) && params.TEST_MODE == "mismatch"){
            archiveArtifacts artifacts: gCompareHtml
            sendMail(gCompareHtml, gXoahJson, "PASS")
        } else if (fileExists(gPerfHtml) && params.TEST_MODE =~ "perf"){
            archiveArtifacts artifacts: gPerfHtml
            sendMail(gPerfHtml, gXoahJsonPerf, "PASS")
        } else if (fileExists(gOutputJson)) {
            sendMail(gOutputHtml, gOutputJson, "OK")
        }
    } catch(err) {
        println err
    }

    if (fileExists(gPerfJson)){
       try{
           archiveArtifacts artifacts: gPerfJson
       } catch(err) {
           println err
       }
    }

    def firmware = "${gDpuName}.xclbin"
    if (fileExists(firmware)){
        archiveArtifacts artifacts: firmware
    }
    firmware = "2x4x2_pss_pst_model_mha_qdq.xclbin"
    if (fileExists(firmware)){
        archiveArtifacts artifacts: firmware
        bat "del ${firmware}"
    }

    try{
        def profilingExcelList = findFiles(glob:"${gProfilingCSV}_*.xlsx")
        if (profilingExcelList.size() > 0 && fileExists(profilingExcelList[0].name)){
            archiveArtifacts artifacts: profilingExcelList[0].name
        } else {
            println "not found profiling data excel"
        }

        def profilingJsonList = findFiles(glob:"${gProfilingCSV}_*.xlsx.json")
        if (profilingJsonList.size() > 0 && fileExists(profilingJsonList[0].name)){
            archiveArtifacts artifacts: profilingJsonList[0].name
        } else {
            println "not found profiling data xlsx.json"
        }
    } catch(err) {
        println err
    }

    try{
        def fileList = findFiles(glob:"*_ONNX_Ops_Coverage.xlsx")
        if (fileList.size() > 0 && fileExists(fileList[0].name)){
            archiveArtifacts artifacts: fileList[0].name
        } else {
            println "not found Onnx Ops Xlsx"
        }
    } catch(err) {
        println err
    }

    try{
        local_package = globalPackage.split('/')[-1]
        if (!gVoeArchive && fileExists(local_package)){
            println "archive1: ${local_package}"
            archiveArtifacts artifacts: local_package
        } else if (!gVoeArchive && fileExists(gLatestPackage)){
            println "archive2: ${gLatestPackage}"
            archiveArtifacts artifacts: gLatestPackage
        }
    } catch(err) {
        println err
    }

    if (params.PDI_ELF != null
        && params.PDI_ELF != ''){
        ctl_pkts = params.PDI_ELF.split('/')[-1]
        if (!(ctl_pkts =~ /.zip/)){
            try{
                 ctl_pkts_zip = "${ctl_pkts}.zip"
                 if (fileExists(ctl_pkts_zip)){
                    bat "del ${ctl_pkts_zip}"
                 }
                zip dir: ctl_pkts, zipFile: ctl_pkts_zip
                ctl_pkts = ctl_pkts_zip
            } catch(err) {
                println err
            }
        }
        if (fileExists(ctl_pkts)){
            try{
                archiveArtifacts artifacts: ctl_pkts
            } catch(err) {
                println err
            }
        }
    }
    
    if (fileExists(gSetupScript)){
        try{
            archiveArtifacts artifacts: gSetupScript
        } catch(err) {
            println err
        }
    }

    if (!params.IGNORE_ARCHIVE){
        try{
            dir(gDriverPath){
                def driver_name = "ipu_mcdm_stack.zip"
                if (fileExists(driver_name)){
                    archiveArtifacts artifacts: driver_name
                } else if (fileExists("ipu_stack_rel_silicon.zip")){
                    driver_name = "ipu_stack_rel_silicon.zip"
                    archiveArtifacts artifacts: driver_name
                }
            }
        } catch(err) {
            println err
        }
    }

    try{
        if (fileExists(gBuildLogZip)){
            archiveArtifacts artifacts: gBuildLogZip
            def log_dir = gBuildLogZip.replace('.zip','')
            //archive all
            //archiveArtifacts artifacts: "${log_dir}/**/*.*"
            if (fileExists(gPerfJson) && env.CI_ARCHIVE_LOG == "true"){
                def model_results = readJSON file: gPerfJson
                model_results.each { model, info ->
                    if (!(info instanceof String) && (info.Summary.Functionality != "PASS" || info.EXIT_CODE != 0)){
                        archiveArtifacts artifacts: "${log_dir}/${model}/build.log"
                    }
                }
            }
            def build_log_stash_name = "buildLog"
            stash name: build_log_stash_name, includes: gBuildLogZip

            def build_log_det = "/group/dphi_software/software/workspace/yoda_suite_data/ipu/Log/${gBuildLogZip}"
            sendXoah(build_log_stash_name, gBuildLogZip, build_log_det)
            if (! isUnix()) {
                bat "del ${gBuildLogZip}"
            }
        }
    } catch(err) {
        println "generate build log ${err}"
    }

    try{
        if (fileExists(gCacheLogZip)){
            archiveArtifacts artifacts: gCacheLogZip
            if (env.REGRESSION_TYPE == "daily"){
                def stash_name = "cacheZip"
                stash name: stash_name, includes: gCacheLogZip
                uploadCache()
            }
            if (! isUnix()) {
                bat "del ${gCacheLogZip}"
            }
        }
    } catch(err) {
        println "generate cache ${err}"
    }

    if (fileExists(gXoahJson)){
        try{
            def stash_name = "xoahJson"
            archiveArtifacts artifacts: gXoahJson
            stash name: stash_name, includes: gXoahJson
            def xoah_det = "/group/dphi_software/software/workspace/yoda_suite_data/ipu/${gXoahJson}"
            sendXoah(stash_name, gXoahJson, xoah_det)
        } catch(err) {
            println err
        }
    }

    if (fileExists(gTestTxt)){
       try{
           archiveArtifacts artifacts: gTestTxt
       } catch(err) {
           println err
       }
    }

    if (fileExists(gOpnameCsv)){
       try{
           archiveArtifacts artifacts: gOpnameCsv
       } catch(err) {
           println err
       }
    }

    if (fileExists(gXoahJsonPerf)){
        try{
            def stash_name = "xoahJsonPerf"
            archiveArtifacts artifacts: gXoahJsonPerf
            stash name: stash_name, includes: gXoahJsonPerf
//             def xoah_perf_det = "/group/dphi_software/software/workspace/yoda_suite_data/ipu/xoah_${gModelzoo}_${gDpuName}_perf_opt${gOptLevel}.json"
//             sendXoah(stash_name, gXoahJsonPerf, xoah_perf_det)
        } catch(err) {
            println err
        }
    }
}


def getResultDir(){
    result_dir = "${gRessultPath}/${gTargetType}_${gDpuName}"
    if (gTestDate != ""){
        result_dir = "${gRessultPath}/${gTargetType}_${gDpuName}_${gTestDate}"
    }
    return result_dir
}

def sendMail(local_html, json_file, key){
    println "debug jsonfile ${json_file}"
    def recipients = getRecipients()
    if (gTestDate != "" || (gSendMail && recipients != '')){
        def stash_name = "htmlReport"
        stash name: stash_name, includes: local_html 
        stash name: "local_outputJson", includes: json_file

        email_header_key = "IPU Regression"
        if (params.TEST_MODE == "performance"){
            if (env.COMPUTERNAME =~ /^XCD/ || env.HOSTNAME =~ /^xcd/ || gNode =~ /^xcd/){
                gLinuxNode = "xcdl190260_yanjunz"
            }
            email_header_key = "High Performance"
        }

        println "mail server: ${gLinuxNode}"   
        node(gLinuxNode) {
          result_dir = getResultDir()
          sh "mkdir -p ${result_dir}; chmod 777 ${result_dir} || true"
          dir(result_dir){
            try{
                unstash name: "local_outputJson"
                def userid = getBuildCausesUser()
                def status = sh(returnStatus: true, script: "grep ${key} ${json_file}")
                if ( status != 0 && (userid == 'anonymous' || userid == null)) {
                    return
                }
                unstash name: stash_name
                if (fileExists(local_html) && gSendMail && recipients != '') {
                    println "Send mail : ${local_html}"   
                    def curr_full_date = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm").format(new Date())
                    email_header = "${email_header_key} ${gTargetType} ${gModelzoo} ${gDpuName} ${gEmailHeaderExtend} ${curr_full_date} ${env.JOB_BASE_NAME}#${env.BUILD_NUMBER} Finished"
                    sh "unset LD_LIBRARY_PATH;mutt -s \"${email_header}\" -e \"set content_type=text/html\" ${recipients} < ${local_html}"
                }
            } catch(err) {
                println err
            }
          }
        }
    }
}

def getBuildCausesUser(){
    def userid = currentBuild.buildCauses[0].userName
    if (currentBuild.buildCauses.size() == 3){
        userid = currentBuild.buildCauses[1].userName
    }
    return userid
}

def getRecipients(){
    def recipients = ""
    try{
        def userid = getBuildCausesUser()
        println "Job triggered by ${userid}"
        if (userid != null && userid != 'anonymous') {
            recipients = "${userid}@amd.com"
            println "recipients: ${recipients}"
        } 
    } catch(err) {
        println err
    }
    if (params.EMAIL_RECEPIENTS != null && params.EMAIL_RECEPIENTS != ''){
        recipients += " ${params.EMAIL_RECEPIENTS}"
    }
    return recipients
}

def getCurrentTime(){
    return new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm").format(new Date())
}

def sendStartNoticeMail(){
    try{
      def userid = getBuildCausesUser()
      def recipients = getRecipients()
      if (gSendMail && recipients != '' && userid != null){
        node(gLinuxNode) {
          def run_date = getCurrentTime()
          def content = """
JOB_START: ${run_date}
JOB_CONSOLE: ${BUILD_URL}/console
JOB_PARAMETERS:
        JOB_BASE_NAME=${env.JOB_BASE_NAME}
        BUILD_NUMBER=${env.BUILD_NUMBER}
        VAI_RT_BRANCH=${env.VAI_RT_BRANCH}
        NODE=${env.NODE}
        BOARD=${env.BOARD}
        TEST_MODE=${env.TEST_MODE}
        MODEL_ZOO=${env.MODEL_ZOO}
        CASE_NAME=${env.CASE_NAME}
        PROFILE=${env.PROFILE}
        USER_ENV=${env.USER_ENV}
        CI_ENV=${env.CI_ENV}
        GLOBAL_PACKAGE=${env.GLOBAL_PACKAGE}
        XLNX_VART_FIRMWARE=${env.XLNX_VART_FIRMWARE}
        PDI_ELF=${env.PDI_ELF}
        CUSTOM_OP_DLL=${env.CUSTOM_OP_DLL}
            """
            sh "unset LD_LIBRARY_PATH;echo \"${content}\"  | mutt -s \"IPU Benchmark Job ${env.JOB_BASE_NAME} build#${env.BUILD_NUMBER} Start \" ${recipients}"
        }
      }
    } catch(err) {
        println err
    }
}

def buildFailSendMail(){
    println "build ${env.JOB_BASE_NAME}#${env.BUILD_NUMBER} failed!"
    def recipients = getRecipients()
    if (gSendMail && recipients != ''){
        node(gLinuxNode) {
          try{
              sh "unset LD_LIBRARY_PATH;echo \"build failed: ${BUILD_URL}/console\" | mutt -s \"build failed ${env.JOB_BASE_NAME}#${env.BUILD_NUMBER} \" ${recipients} ${gCCList}"
          } catch(err) {
              println err
          }
        }
    }
}


def sendXoah(stash_name, file_name, destination){
  if (params.SendYodaData){
    node(gLinuxNode) {
        unstash name: stash_name
        try{
            sh "cp ${file_name} ${destination}"
        } catch(err) {
            println err
        }
    }
  }
}

@NonCPS
def readCommit(contant, cfg) {
    contant.eachLine {
        def list = it.trim().split(':')
        cfg += "${list[0]}_commit_ID=${list[1]}\n"
    }
    return cfg
}

def winBuild(){
    bat "python ci/ci_main.py config"
    try{
        def cfg0_4x4 = '''
phxTarget=
stxTarget=AMD_AIE2P_4x4_Overlay_CFG0
makeOptions=fast_load_pdi: 1;enable_fast_pm: 1
uploadXclbin=true
boardTesting=false
'''
        def default_4x4 = '''
phxTarget=
stxTarget=AMD_AIE2P_4x4_Overlay;AMD_AIE2P_Nx4_Overlay
makeOptions=fast_load_pdi: 0;enable_fast_pm: 0
uploadXclbin=true
boardTesting=false
'''
        def stagesForParallel = [:]
        if (gBuildXclbin){
            cfg = cfg0_4x4
            if (fileExists(gSubmoduleCommit)){
                cfg = readCommit(readFile(gSubmoduleCommit), cfg)
                println cfg
            }
            stagesForParallel['buildXclbin'] = {
                timeout(time: 180, unit: "MINUTES"){
                    triggerRemoteJob(job: "http://xcdl190269:8081/job/nightly_build_benchmark", parameters: cfg, auth: CredentialsAuth(credentials: globalCredentialsID), pollInterval: 60, shouldNotFailBuild: true)
                }
            }
        }
        stagesForParallel['buildVoe'] = {
            bat "python ci/ci_main.py build"
        }
        parallel stagesForParallel
    } catch (Exception err){
        println err
    }

}

def buildPackage() {
    stage('Build') {
        if(gIgnoreBuild) {
            return
        }
        try{
          withEnv([ "JENKINS_CI_BUILD=1",
                    "CURRENT_RELEASE_FILE=${gReleaseFile}",
                    "DPU_PHX_SUBMODULE_COMMIT=${gSubmoduleCommit}"
          ]) {
            if (isUnix()) {
                sh "python ci/ci_main.py config"
                buildInDocker()
                dir('hb') {
                    findPackage("tgz")
                }
            } else {
                if (env.COMPUTERNAME =~ /^XCD/){
                    withEnv(["https_proxy=http://xcdl190074:9181"]){
                        winBuild()
                    }
                } else {
                    winBuild()
                }
                dir('hb') {
                    findPackage("zip")
                }
            }
            if (globalPackage == ''){
                buildFailSendMail()
            }
          }
        } catch (err){
            buildFailSendMail()
            throw new Exception(err)
        }
    }
}

def update_env(env_key) {
    // if (env_key =~ "ACCURACY_TEST=true"){
    if (params.TEST_MODE =~ "accuracy"){
        gCINewFlow = false
    }
    if (env_key =~ "CI_SEND_MAIL=false"){
        gSendMail = false
    }
    if (env_key =~ "CI_REBOOT_BOARD=true"){
        gRebootBoard = true
    }
    if (env_key =~ "CI_ENABLE_CC=true"){
        gCCList = "-c h.zhang@amd.com -c yanjunz@amd.com -c qingz@amd.com -c runfengw@amd.com"
    }
    if (env_key =~ "CI_BUILD_XCLBIN"){
        gBuildXclbin = true
    }
    env_list = env_key.split(" ")
    env_list.each {
        each_env = it.split("=")
        println "${each_env[0]}=${each_env[1]}"
        env[each_env[0]] = each_env[1]
        if (each_env.size() > 1 && each_env[0] == "ACCURACY_TEST_TIMEOUT"){
            gTimeLimit = Integer.parseInt(each_env[1])
            gTimeUnit = 'SECONDS'
            println "timeout limit for accuracy set to: ${gTimeLimit} ${gTimeUnit}"
        } else if (each_env.size() > 1 && each_env[0] == "TIMEOUT"){
            gTestTimeLimit = Integer.parseInt(each_env[1])
            gTestTimeUnit = 'SECONDS'
            println "timeout limit for each case set to: ${gTestTimeLimit} ${gTestTimeUnit}"
        } else if (each_env.size() > 1 && each_env[0] == "DPM_LEVEL"){
            gDpmLevel = Integer.parseInt(each_env[1])
        }
    }
}

def updateGlobalViarable() {
    if (env.COMPUTERNAME =~ /^(XSJ|XCO)/ || env.HOSTNAME =~ /^(xsj|xco)/ || gNode =~ /^(xsj|xco)/){
        gJumpHost = "xsjncuph07.xilinx.com"
        gLinuxNode = "local"
        user_name = env.USER
        if (! user_name) {
            user_name = env.USERNAME
        }
        gPackagePath = "/proj/rdi/staff/huizhan1/IPU_packages"
        gRessultPath = "/proj/rdi/staff/huizhan1/IPU_test_results"
        if (isUnix()) {
            gLatestPackage = "voe-linux_amd64-with_xcompiler_on-latest_dev.tgz"
        }
        suiteRunName = "xcoartifactory.xilinx.com/vitis-ai-docker-dev-local/voe-dev-build:latest"
    }
    def postfix = 'zip'
    if(isUnix()){
        postfix = 'tgz'
    }
    // if (params.ACCURACY_TEST == "true"){
    if (params.TEST_MODE =~ "accuracy"){
        gCINewFlow = false
    }
    if (params.USER_ENV != null && params.USER_ENV != ''){
        update_env(params.USER_ENV)
    }
    if (params.CI_ENV != null && params.CI_ENV != ''){
        update_env(params.CI_ENV)
    }
    gArtiSubdir = new java.text.SimpleDateFormat("yyyyMMdd").format(new Date())
    if (params.RELEASE_FILE != null && params.RELEASE_FILE =~ 'latest.txt') {
        gArtidir = "voe_test_package_latest_txt"
        gLatestPackage = "voe-win_amd64-with_xcompiler_on-latest_txt.${postfix}"
        if (isUnix()) {
            gLatestPackage = "voe-linux_amd64-with_xcompiler_on-latest_txt.${postfix}"
        }
    } else if (params.RELEASE_FILE != null && params.RELEASE_FILE =~ 'latest_stx.txt') {
        gArtidir = "voe_test_package_latest_stx"
        gLatestPackage = "voe-win_amd64-with_xcompiler_on-latest_stx.${postfix}"
        if (isUnix()) {
            gLatestPackage = "voe-linux_amd64-with_xcompiler_on-latest_stx.${postfix}"
        }
    } else if (params.RELEASE_FILE != null && params.RELEASE_FILE =~ 'latest_qdq.txt') {
        gArtidir = "voe_test_package_qdq"
        gLatestPackage = "voe-win_amd64-with_xcompiler_on-latest_qdq.${postfix}"
        if (isUnix()) {
            gLatestPackage = "voe-linux_amd64-with_xcompiler_on-latest_qdq.${postfix}"
        }
    } else if (params.RELEASE_FILE != null && params.RELEASE_FILE =~ 'latest_qdq_shell.txt') {
        gArtidir = "voe_test_package_qdq_shell"
        gLatestPackage = "voe-win_amd64-with_xcompiler_on-latest_qdq_shell.${postfix}"
        if (isUnix()) {
            gLatestPackage = "voe-linux_amd64-with_xcompiler_on-latest_qdq_shell.${postfix}"
        }
    } else if (params.RELEASE_FILE != null && params.RELEASE_FILE =~ 'latest_4x2_dev.txt') {
        gArtidir = "voe_test_package_4x2"
        gLatestPackage = "voe-win_amd64-with_xcompiler_on-latest_4x2_dev.${postfix}"
        if (isUnix()) {
            gLatestPackage = "voe-linux_amd64-with_xcompiler_on-latest_4x2_dev.${postfix}"
        }
    }
    if(params.VAI_RT_BRANCH != null && params.VAI_RT_BRANCH != '') {
        if (params.VAI_RT_BRANCH =~ /^pr/) {
            gArtiSubdir = "pr_verify/${params.VAI_RT_BRANCH}"
        } else {
            gArtiSubdir = "${params.VAI_RT_BRANCH}/${env.JOB_BASE_NAME}_${env.VAI_RT_BRANCH}"
            if (params.DUMP_MC_CODE){
                gArtiSubdir = "${params.VAI_RT_BRANCH}/${env.JOB_BASE_NAME}_${env.VAI_RT_BRANCH}_dump_mc_code"
            }
        }
        gPackagePath = "${gPackagePath}/${gArtidir}/${gArtiSubdir}"
    } else if(params.XCOMPILER_BRANCH != null && params.XCOMPILER_BRANCH != '') {
        gArtiSubdir = "xcompiler_pr/${params.XCOMPILER_BRANCH}"
        gPackagePath = "${gPackagePath}/${gArtidir}/${gArtiSubdir}"
    } else if(params.VAIP_BRANCH != null && params.VAIP_BRANCH != '') {
        gArtiSubdir = "vaip_pr/${params.VAIP_BRANCH}"
        gPackagePath = "${gPackagePath}/${gArtidir}/${gArtiSubdir}"
    } else if(params.VART_BRANCH != null && params.VART_BRANCH != '') {
        gArtiSubdir = "vart_pr/${params.VART_BRANCH}"
        gPackagePath = "${gPackagePath}/${gArtidir}/${gArtiSubdir}"
    } else if(params.XIR_BRANCH != null && params.XIR_BRANCH != '') {
        gArtiSubdir = "xir_pr/${params.XIR_BRANCH}"
        gPackagePath = "${gPackagePath}/${gArtidir}/${gArtiSubdir}"
    } else if(params.TARGET_FACTORY_BRANCH != null && params.TARGET_FACTORY_BRANCH != '') {
        gArtiSubdir = "target_factory_pr/${params.TARGET_FACTORY_BRANCH}"
        gPackagePath = "${gPackagePath}/${gArtidir}/${gArtiSubdir}"
    }
    gSubmoduleCommit = "dpu_phx_submodule_commit_${env.BUILD_NUMBER}.txt"
    gReleaseFile = "release_file_${env.BUILD_NUMBER}.txt"
}


def setDescription(){
    if (env.COMPUTERNAME =~ /^XCD/ || env.HOSTNAME =~ /^xcd/ || gNode =~ /^xcd/){
        return
    }
    
    try{
        println currentBuild.buildCauses
        def userid = getBuildCausesUser()
        def desc = ""
        if (params.Description != null && params.Description != '') {
            desc = "${params.Description}"
        }
        else if(params.CASE_NAME!= null && params.CASE_NAME != '') {
            desc = "${params.CASE_NAME}"
        }
        else if(params.VAI_RT_BRANCH != null && params.VAI_RT_BRANCH != '') {
            desc = "${params.VAI_RT_BRANCH} ${desc}"
        }
        if ("${userid}" != "null" && "${userid}" != "anonymous") {
            buildDescription "${userid}: ${desc}"
        } else {
            buildDescription "${desc}"
        }
    } catch(err) {
      println err
    }

}

def xsj_cold_reboot(board){
  node(gLinuxNode) {
    println "reboot board ${board}"
    def nodeComputer = Jenkins.instance.getNode(board).toComputer()
    try{
        def remote = [:]
        remote.name = 'xsjengvm209067'
        remote.host = 'xsjengvm209067'
        remote.user = 'pact2'
        remote.password = 'p1Q$AgPLYQrbxw!wuzg1'
        remote.allowAnyHosts = true
        stage('Remote SSH') {
          sleep 2
          nodeComputer.disconnect()
          sshCommand remote: remote, command: "/proj/sdxbf/prod/cmbin/util/phreboot ${board} reboot force"
        }
    } catch(err) {
      println err
    } finally {
        echo 'Reconnect node'
        def status = nodeComputer.isOnline()
        if (status) {
            nodeComputer.disconnect()
            status = false
        }
        sleep 60
    }
    update_ipu_config(gDpmLevel)
  }
}

def getHostName(){
    return env.NODE_NAME
}

def delIPUWorkspace(){
    dir(gIPUWorkspace){
        if (fileExists(env.JOB_BASE_NAME)) {
            dir(env.JOB_BASE_NAME){
                deleteDir()
            }
        }
    }
}

def checkIfCleanAndReboot(){
    try{
        if (params.CleanWorkspace) { 
            cleanWs()
            delIPUWorkspace()
        }
    } catch(err) {
        gRebootBoard = true
        println err
    }
}

def update_ipu_config(dpm_level){
    if (env.COMPUTERNAME =~ /^XCD/ || env.HOSTNAME =~ /^xcd/ || gNode =~ /^xcd/){
        return
    }
    try{
        node("xsjstrhalo10"){
            checkout_code()
            withCredentials([usernamePassword(credentialsId: "177c7a23-df5f-4dd9-b969-2b839901516b", usernameVariable: "username", passwordVariable: "password")]) {
                bat "echo ${username}"
                bat "python ci/tools/update_ipu_config.py ${gNode} ${username} ${password} ${dpm_level}"
            }
        }
    } catch(err) {
        println err
    }
}

def combine_xclbin(){
  //download origin xclbin on board for archive
  bat "python.exe ci/ci_main.py download_xclbin"
  node(gLinuxNode) {
    dir("${gRessultPath}/${env.JOB_BASE_NAME}"){
        deleteDir()
        checkout scm
        sh "python3 ci/ci_main.py download_xclbin"
        sh "rm -rf tools; git clone https://gitenterprise.xilinx.com/haozhu12/tools.git; cp tools/combine_xclbin/combine_xclbin.sh tools/combine_xclbin/add_kernel.json ."
        sh "rm -rf DynamicDispatch || true; git clone https://gitenterprise.xilinx.com/VitisAI/DynamicDispatch.git; cp DynamicDispatch/xclbin/stx/mladf_2x4x2_matmul_softmax_mul_a16w16.xclbin ."
        sh "bash combine_xclbin.sh"
        env["XLNX_VART_FIRMWARE"] = "${gRessultPath}/${env.JOB_BASE_NAME}/2x4x2_pss_pst_model_mha_qdq.xclbin"
    }
  }
}

def build_stage(){
    if (params.CleanWorkspace) { 
        cleanWs()
    }
    checkout_code()

    if (params.regenerateParams != null && params.regenerateParams != ''){
        def paramsUtils = load 'ci/params_utils.groovy'
        paramsUtils.generateParams()
        def userid = getBuildCausesUser()
        buildDescription "${userid}: regenerate params ${params.regenerateParams}"
        return
    }
    updateGlobalViarable()

    setDescription()

    sendStartNoticeMail()

    buildPackage()

    archivePackage()
}

def test_stage(check) {
    if (params.regenerateParams != null && params.regenerateParams != ''){
        return
    }
    def origNode = gNode
    gNode = getHostName()

    if(check){
        checkIfCleanAndReboot()
        checkout_code() 
    } else if (params.CleanWorkspace) {
        delIPUWorkspace()
    }

    def modelzoo = get_modelzoo()
    if (modelzoo == ''){
        return
    }
    if (gLinuxNode == 'local') {
        if (gRebootBoard) {
            xsj_cold_reboot(gNode)
            cleanWs()
            checkout_code() 
        }
        if (env.CI_VAIP_FLOW == 'xcompiler+dd'){
            combine_xclbin()
        }
    }

    verifyPackage()
    
    update_package()

    userid = getBuildCausesUser()
    env.BUILD_USER = userid
    
    retry(3) {
        if(! gRebootBoard && gDpmLevel != 7) {
            update_ipu_config(gDpmLevel)
        }
        if (isUnix() && params.TEST_IN_DOCKER) {
            testInDocker()
            return
        }
        testGroups(modelzoo)
    }
    if (gDpmLevel != 7){
        update_ipu_config(7)
    }
    
    gNode = origNode
}

if (params.IGNORE_BUILD == true || (params.GLOBAL_PACKAGE != null && params.GLOBAL_PACKAGE != '')) {
    gIgnoreBuild = true
    if (params.BOARD != null && params.BOARD != '') {
        gNode = params.BOARD
    } else {
        gNode = params.NODE
    }
} else {
    gNode = params.NODE
    gIgnoreBuild = false
}

timestamps{
    node(gNode) {
        build_stage()
        if (params.BOARD == null || params.BOARD == '' || gNode == params.BOARD) {
            println "--- run test stage NODE: ${gNode} BOARD: ${params.BOARD}"
            test_stage(false)
        }
    }
    
    if (params.BOARD != null && gNode != params.BOARD) {
        println "=== run test stage NODE: ${gNode} BOARD: ${params.BOARD}"
        node(params.BOARD){
            test_stage(true)
        }
    }
}
