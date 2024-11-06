pipeline {
    parameters {
        booleanParam(name: 'CleanWorkspace', defaultValue: false, description: 'clean workspace')
        string(name: 'NODE', defaultValue: 'xcdl220228', description: '')
        string(name: 'MODEL_FILTER', defaultValue: 'all', description: '')
        string(name: 'TARGET_FILTER', defaultValue: 'all', description: '')
        string(name: 'VART_BRANCH', defaultValue: '', description: 'define a branch or a pr')
        string(name: 'VAIP_BRANCH', defaultValue: '', description: 'define a branch or a pr')
        string(name: 'TEST_ONNX_RUNNER_BRANCH', defaultValue: '', description: 'define a branch or a pr')
        string(name: 'PARALLEL_NUM', defaultValue: '10', description: '')
    }
    agent {
        label params.NODE
    }
    environment {
        https_proxy = 'http://localhost:9181'

    }
    stages {
        stage('initialize') {
            steps {
                script{
                    if (params.CleanWorkspace) { 
                        cleanWs()
                        checkout scm
                    }
                    def uid = sh(returnStdout: true, script: 'id -u').trim()
                    def gid = sh(returnStdout: true, script: 'id -g').trim()
                    sh "echo ${env.USER}:x:${uid}:${gid}::/home/${env.USER}:/bin/sh > ${env.WORKSPACE}/passwd"
                    sh "mkdir -p .ssh; cp /home/${env.USER}/.ssh/* .ssh/"
                }
            }
        }
        stage('linux build') {
            agent {
                docker {
                    image 'artifactory.xilinx.com/vitis-ai-docker-dev/aisw/dev:v5.0-46-gf767eb4'
                    args "--network=host --entrypoint= -v /group/modelzoo:/group/modelzoo -v ${env.WORKSPACE}/passwd:/etc/passwd -v ${env.WORKSPACE}:/home/${env.USER} -v /proj/rdi/staff/${env.USER}:/proj/rdi/staff/${env.USER} -v /group/dphi_software:/group/dphi_software -v /dev/shm:/dev/shm -e USER=${env.USER}"
                    reuseNode true
                }
            }
            steps {
                sh "python ci/ci_main.py build"
                if(fileExists("vai-rt/release_file_${env.BUILD_NUMBER}.txt")){
                    archiveArtifacts artifacts: "vai-rt/release_file_${env.BUILD_NUMBER}.txt"
                }
            }
        }
        stage("test"){
            matrix {
                agent {
                    docker {
                        image 'artifactory.xilinx.com/vitis-ai-docker-dev/aisw/dev:v5.0-46-gf767eb4'
                        args "--network=host --entrypoint= -v /group/modelzoo:/group/modelzoo -v ${env.WORKSPACE}/passwd:/etc/passwd -v ${env.WORKSPACE}:/home/${env.USER} -v /proj/xcdhdstaff1/${env.USER}:/proj/xcdhdstaff1/${env.USER} -v /group/dphi_software:/group/dphi_software -v /dev/shm:/dev/shm -e USER=${env.USER}"
                        reuseNode true
                    }
                }
                axes {
                    axis {
                        name 'MODEL'
                        values 'hugging_face.env', 'model_zoo_3.5_release.env'
                    }
                    axis {
                        name 'TARGET'
                        values 'IPU.env', 'VEK280.env'
                    }
                }
                when {allOf{
                        expression { return params.MODEL_FILTER == 'all' || params.MODEL_FILTER == MODEL }
                        expression { return params.TARGET_FILTER == 'all' || params.TARGET_FILTER == TARGET }
    
                }}
                stages {
                    stage("test"){
                        steps {
                            sh "rm -rf ${env.WORKSPACE}/${MODEL}_${TARGET}.cache build/${MODEL}_${TARGET}"
                            sh "env XLNX_CACHE_DIR=${env.WORKSPACE}/${MODEL}_${TARGET}.cache VAIP_REGRESSION=${MODEL}_${TARGET} python ci/ci_main.py test -j ${params.PARALLEL_NUM} -f default.env,${MODEL},${TARGET}"
                            archiveArtifacts artifacts: "${MODEL}_${TARGET}.html"
                        }
                    }
                }
            }
        }
    }

}

