#
#   The Xilinx Vitis AI Vaip in this distribution are provided under the following free 
#   and permissive binary-only license, but are not provided in source code form.  While the following free 
#   and permissive license is similar to the BSD open source license, it is NOT the BSD open source license 
#   nor other OSI-approved open source license.
#
#    Copyright (C) 2022 Xilinx, Inc. All rights reserved.
#    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
#    Redistribution and use in binary form only, without modification, is permitted provided that the following conditions are met:
#
#    1. Redistributions must reproduce the above copyright notice, this list of conditions and the following disclaimer in 
#    the documentation and/or other materials provided with the distribution.
#
#    2. The name of Xilinx, Inc. may not be used to endorse or promote products redistributed with this software without specific 
#    prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL XILINX, INC. 
#    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
#    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#


#!/bin/bash
set -e

script_path=$(dirname "$(realpath $0)")
project_name=$(basename ${script_path})

function    usage() {
    echo "./cmake.sh [options]"
    echo "    --help                    show help"
    echo "    --clean                   discard build dir before build"
    echo "    --build-only              build only, will not install"
    echo "    --build-python            build python. if --pack is declared, will build conda package"
    echo "    --type[=TYPE]             build type. VAR {release, debug(default)}"
    echo "    --pack[=FORMAT]           enable packing and set package format. VAR {deb, rpm}"
    echo "    --build-dir[=DIR]         set customized build directory. default directory is ${build_dir_default}"
    echo "    --install-prefix[=PREFIX] set customized install prefix. default prefix is ${install_prefix_default}"
    echo "    --cmake-options[=OPTIONS] append more cmake options"
    exit 0
}

# cmake args
declare -a args
args=(-DBUILD_TEST=ON)
args+=(-DBUILD_SHARED_LIBS=ON)
args+=(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
args+=(-DCMAKE_BUILD_TYPE=Debug)
build_type=Debug
# parse options
options=$(getopt -a -n 'parse-options' -o h \
		         -l help,ninja,build-python,clean,build-only,type:,project:,pack:,build-dir:,install-prefix:,cmake-options:,home,user \
		         -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
        -h | --help) show_help=true; usage; break;;
	    --clean) clean=true;;
	    --build-only) build_only=true;;
	    --type)
	        shift
	        case "$1" in
                release)
                    build_type=Release;
                    args+=(-DCMAKE_BUILD_TYPE=${build_type:="Release"});;
                debug)
                    build_type=Debug;
                    args+=(-DCMAKE_BUILD_TYPE=${build_type:="Debug"});;
		        *) echo "Invalid build type \"$1\"! try --help"; exit 1;;
	        esac
	        ;;
	    --pack)
	        shift
                build_package=true
                cpack_generator=
	        case "$1" in
		        deb)
                            cpack_generator=DEB;
                            args+=(-DCPACK_GENERATOR=${cpack_generator});;
		        rpm)
                            cpack_generator=RPM;
                            args+=(-DCPACK_GENERATOR=${cpack_generator});;
		        *) echo "Invalid pack format \"$1\"! try --help"; exit 1;;
	        esac
	        ;;
	    --build-dir) shift; build_dir=$1;;
	    --install-prefix) shift; install_prefix=$1;;
	    --cmake-options) shift; args+=($1);;
        --ninja) args+=(-G Ninja);;
        --project) shift;script_path="$(realpath ../$1)"; project_name=$1;;
        --build-python) args+=(-DBUILD_PYTHON=ON);;
	    --user) args+=(-DINSTALL_USER=ON);;
	    --home) args+=(-DINSTALL_HOME=ON);;
	    --) shift; break;;
    esac
    shift
done

if which ninja >/dev/null; then
    args+=(-G Ninja)
fi
# detect target & set install prefix
if [ -z ${OECORE_TARGET_SYSROOT:+x} ]; then
    echo "Native-platform building..."
    os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
    os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
    arch=`uname -p`
    target_info=${os}.${os_version}.${arch}.${build_type}
    install_prefix_default=$HOME/.local/${target_info}
    args+=(-DCMAKE_PREFIX_PATH=${install_prefix:="${install_prefix_default}"})
else
    echo "Cross-platform building..."
    echo "Found target sysroot ${OECORE_TARGET_SYSROOT}"
    target_info=${OECORE_TARGET_OS}.${OECORE_SDK_VERSION}.${OECORE_TARGET_ARCH}.${build_type}
    install_prefix=${OECORE_TARGET_SYSROOT}/install/${build_type}
    args+=(-DCMAKE_TOOLCHAIN_FILE=${OECORE_NATIVE_SYSROOT}/usr/share/cmake/OEToolchainConfig.cmake)
    args+=(-DCMAKE_PREFIX_PATH=${OECORE_TARGET_SYSROOT}/install/${build_type})
fi
args+=(-DCMAKE_INSTALL_PREFIX=${install_prefix:="${install_prefix_default}"})

# set build dir
build_dir_default=$HOME/build/build.${target_info}/${project_name}
[ -z ${build_dir:+x} ] && build_dir=${build_dir_default}

if [ x${clean:=false} == x"true" ] && [ -d ${build_dir} ];then
    echo "cleaning: rm -fr ${build_dir}"
    rm -fr "${build_dir}"
fi

mkdir -p ${build_dir}
cd -P ${build_dir}
echo "cd $PWD"
echo cmake "${args[@]}" "$script_path"
cmake "${args[@]}" "$script_path"
cmake --build . -j $(nproc)
${build_only:=false} || cmake --install .
${build_package:=false} && cpack -G ${cpack_generator}
if [ -f compile_commands.json ]; then
    cp -av compile_commands.json "$script_path"
fi

exit 0
