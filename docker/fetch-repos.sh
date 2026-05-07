#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

# Color functions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

gecho() { echo -e "${GREEN}$1${NC}"; }
recho() { echo -e "${RED}$1${NC}"; }
yecho() { echo -e "${YELLOW}$1${NC}"; }

# Dependency Git URLs, hashes/branches, and directory names
QONNX_URL="https://github.com/fastmachinelearning/qonnx.git"
FINN_URL="https://github.com/Xilinx/finn.git"
FINN_EXP_URL="https://github.com/Xilinx/finn-experimental.git"
BREVITAS_URL="https://github.com/Xilinx/brevitas.git"
CNPY_URL="https://github.com/maltanar/cnpy.git"
HLSLIB_URL="https://github.com/Xilinx/finn-hlslib.git"
OMX_URL="https://github.com/maltanar/oh-my-xilinx.git"
AVNET_BDF_URL="https://github.com/Avnet/bdf.git"
XIL_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"
RFSOC4x2_BDF_URL="https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git"
KV260_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"

QONNX_COMMIT="27c05ebfe0e7960bbf83761858da56238f17f0e3"
FINN_COMMIT="custom/experimental_rtl_softmax"
FINN_EXP_COMMIT="0724be21111a21f0d81a072fccc1c446e053f851"
BREVITAS_COMMIT="aad4d5a293db6f2ec622a92a5d3278e47072453e"
CNPY_COMMIT="8c82362372ce600bbd1cf11d64661ab69d38d7de"
HLSLIB_COMMIT="a2cd3e6ce653a03e59af6bcb9fbeaa71618d160e"
OMX_COMMIT="0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a"
AVNET_BDF_COMMIT="2d49cfc25766f07792c0b314489f21fe916b639b"
XIL_BDF_COMMIT="8cf4bb674a919ac34e3d99d8d71a9e60af93d14e"
RFSOC4x2_BDF_COMMIT="13fb6f6c02c7dfd7e4b336b18b959ad5115db696"
KV260_BDF_COMMIT="98e0d3efc901f0b974006bc4370c2a7ad8856c79"
EXP_BOARD_FILES_MD5="226ca927a16ea4ce579f1332675e9e9a"

QONNX_DIR="qonnx"
FINN_DIR="finn"
FINN_EXP_DIR="finn-experimental"
BREVITAS_DIR="brevitas"
CNPY_DIR="cnpy"
HLSLIB_DIR="finn-hlslib"
OMX_DIR="oh-my-xilinx"
AVNET_BDF_DIR="avnet-bdf"
XIL_BDF_DIR="xil-bdf"
RFSOC4x2_BDF_DIR="rfsoc4x2-bdf"
KV260_SOM_BDF_DIR="kv260-som-bdf"

# Validate environment variables for licensed Xilinx tools
if [ -z "$BSMITH_XILINX_PATH" ];then
  recho "Please set the BSMITH_XILINX_PATH environment variable to the path to your Xilinx tools installation directory (e.g. /opt/Xilinx)."
  recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
fi
if [ -z "$BSMITH_XILINX_VERSION" ];then
  recho "Please set the BSMITH_XILINX_VERSION to the version of the Xilinx tools to use (e.g. 2022.2)"
  recho "FINN functionality depending on Vivado, Vitis or HLS will not be available."
fi
if [ -z "$PLATFORM_REPO_PATHS" ];then
  recho "Please set PLATFORM_REPO_PATHS pointing to Vitis platform files (DSAs)."
  recho "This is required to be able to use Alveo PCIe cards."
fi

# Define functions
fetch_repo() {
    # URL for git repo to be cloned
    REPO_URL=$1
    # commit hash for repo
    REPO_COMMIT=$2
    # directory to clone to under deps
    REPO_DIR=$3
    # absolute path for the repo local copy
    CLONE_TO=$BSMITH_DIR/deps/$REPO_DIR

    echo "Fetching $REPO_DIR from $REPO_URL..."

    # clone repo if dir not found
    if [ ! -d "$CLONE_TO" ]; then
        echo "Cloning $REPO_DIR..."
        # Use retry logic for git clone in CI (but with full clone for dependency resolution)
        local attempt=1
        local max_attempts=3

        while [ $attempt -le $max_attempts ]; do
            if git clone $REPO_URL $CLONE_TO; then
                echo "Successfully cloned $REPO_DIR on attempt $attempt"
                break
            else
                echo "Clone attempt $attempt failed for $REPO_DIR"
                if [ $attempt -lt $max_attempts ]; then
                    echo "Retrying in 10 seconds..."
                    sleep 10
                    rm -rf "$CLONE_TO" 2>/dev/null || true
                fi
                attempt=$((attempt + 1))
            fi
        done

        if [ $attempt -gt $max_attempts ]; then
            echo "ERROR: Failed to clone $REPO_DIR after $max_attempts attempts"
            return 1
        fi
    fi

    # verify and try to pull repo if not at correct commit
    CURRENT_COMMIT=$(git -C $CLONE_TO rev-parse HEAD 2>/dev/null || echo "unknown")
    if [ "$CURRENT_COMMIT" != "$REPO_COMMIT" ]; then
        echo "Current commit $CURRENT_COMMIT != expected $REPO_COMMIT for $REPO_DIR"

        # Try to pull first to get latest refs
        echo "Pulling latest changes for $REPO_DIR..."
        git -C $CLONE_TO pull || echo "Pull failed, continuing with checkout..."

        # checkout the expected commit
        echo "Checking out commit $REPO_COMMIT for $REPO_DIR..."
        if ! git -C $CLONE_TO checkout $REPO_COMMIT; then
            echo "ERROR: Could not checkout commit $REPO_COMMIT for $REPO_DIR"
            return 1
        fi
    fi

    # verify one last time
    CURRENT_COMMIT=$(git -C $CLONE_TO rev-parse HEAD 2>/dev/null || echo "unknown")
    if [ "$CURRENT_COMMIT" = "$REPO_COMMIT" ]; then
        echo "Successfully checked out $REPO_DIR at commit $CURRENT_COMMIT"
        return 0
    else
        echo "ERROR: Final verification failed for $REPO_DIR. Expected: $REPO_COMMIT, Got: $CURRENT_COMMIT"
        return 1
    fi
}

fetch_board_files() {
    echo "Downloading and extracting board files..."
    mkdir -p "$BSMITH_DIR/deps/board_files"
    OLD_PWD=$(pwd)
    cd "$BSMITH_DIR/deps/board_files"
    wget -q https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip
    wget -q https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip
    unzip -q pynq-z1.zip
    unzip -q pynq-z2.zip
    cp -r $BSMITH_DIR/deps/$AVNET_BDF_DIR/* $BSMITH_DIR/deps/board_files/
    cp -r $BSMITH_DIR/deps/$XIL_BDF_DIR/boards/Xilinx/rfsoc2x2 $BSMITH_DIR/deps/board_files/;
    cp -r $BSMITH_DIR/deps/$RFSOC4x2_BDF_DIR/board_files/rfsoc4x2 $BSMITH_DIR/deps/board_files/;
    cp -r $BSMITH_DIR/deps/$KV260_SOM_BDF_DIR/boards/Xilinx/kv260_som $BSMITH_DIR/deps/board_files/;
    cd $OLD_PWD
}

fetch_repo $QONNX_URL $QONNX_COMMIT $QONNX_DIR
fetch_repo $FINN_URL $FINN_COMMIT $FINN_DIR
fetch_repo $FINN_EXP_URL $FINN_EXP_COMMIT $FINN_EXP_DIR
fetch_repo $BREVITAS_URL $BREVITAS_COMMIT $BREVITAS_DIR
fetch_repo $CNPY_URL $CNPY_COMMIT $CNPY_DIR
fetch_repo $HLSLIB_URL $HLSLIB_COMMIT $HLSLIB_DIR
fetch_repo $OMX_URL $OMX_COMMIT $OMX_DIR

# Can skip downloading of board files entirely if desired
if [ "$BSMITH_DOWNLOAD_BOARDS" == "1" ]; then
    # Fetch board-related repositories
    fetch_repo $AVNET_BDF_URL $AVNET_BDF_COMMIT $AVNET_BDF_DIR
    fetch_repo $XIL_BDF_URL $XIL_BDF_COMMIT $XIL_BDF_DIR
    fetch_repo $RFSOC4x2_BDF_URL $RFSOC4x2_BDF_COMMIT $RFSOC4x2_BDF_DIR
    fetch_repo $KV260_BDF_URL $KV260_BDF_COMMIT $KV260_SOM_BDF_DIR

    # download extra board files and extract if needed
    if [ ! -d "$BSMITH_DIR/deps/board_files" ]; then
        fetch_board_files
    else
        cd $BSMITH_DIR
        BOARD_FILES_MD5=$(find deps/board_files/ -type f -exec md5sum {} \; | sort -k 2 | md5sum | cut -d' ' -f 1)
        if [ "$BOARD_FILES_MD5" = "$EXP_BOARD_FILES_MD5" ]; then
            echo "Verified board files folder content md5: $BOARD_FILES_MD5"
        else
            echo "Board files folder md5: expected $BOARD_FILES_MD5 found $EXP_BOARD_FILES_MD5"
            echo "Board files folder content mismatch, removing and re-downloading"
            rm -rf deps/board_files/
            fetch_board_files
        fi
    fi
else
    echo "Skipping board file repositories and downloads (set BSMITH_DOWNLOAD_BOARDS=1 to enable)"
fi

gecho "Docker container is named $DOCKER_INST_NAME"
gecho "Docker tag is named $BSMITH_DOCKER_TAG"
gecho "Mounting $BSMITH_BUILD_DIR into $BSMITH_BUILD_DIR"
gecho "Mounting $BSMITH_XILINX_PATH into $BSMITH_XILINX_PATH"
gecho "Port-forwarding for Netron $NETRON_PORT:$NETRON_PORT"
gecho "Vivado IP cache dir is at $VIVADO_IP_CACHE"
