#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Main entrypoint for Brainsmith development environment
# Handles full setup including dependency fetching and installation

# Enhanced logging for debugging
log_debug() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: $1" >&2
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Status emission for container synchronization
BRAINSMITH_STATUS_PREFIX="BRAINSMITH_STATUS:"
emit_status() {
    local status="$1"
    local detail="${2:-}"
    echo "${BRAINSMITH_STATUS_PREFIX}${status}${detail:+:$detail}"
    log_info "Status: $status${detail:+ - $detail}"
}

log_info "Starting Brainsmith entrypoint"
emit_status "INITIALIZING"

cd $BSMITH_DIR

# First: Fetch dependencies if they don't exist (before environment setup)
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ] && ([ ! -d "$BSMITH_DIR/deps/qonnx" ] || [ ! -d "$BSMITH_DIR/deps/finn" ]); then
    emit_status "FETCHING_DEPENDENCIES"
    log_info "Fetching dependencies to $BSMITH_DIR/deps/ (required before environment setup)"
    
    if source docker/fetch-repos.sh; then
        log_info "Dependencies fetched successfully"
    else
        emit_status "ERROR" "Failed to fetch dependencies"
        log_error "Failed to fetch dependencies"
        exit 1
    fi
    
    log_info "Dependencies ready at $(date)"
else
    log_info "Dependencies already exist, ready at $(date)"
fi

# Second: Load environment setup (now that dependencies exist)
source /usr/local/bin/setup_env.sh

# Check FINN dependency after environment is loaded (so recho function is available)
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ] && [ ! -f "$BSMITH_DIR/deps/finn/setup.py" ]; then
    recho "FINN dependency not found or not fetched"
    recho "Dependencies should be automatically fetched during container initialization"
    exit 1
fi

# Function to build finnxsi if needed
build_finnxsi_if_needed() {
    if [ ! -z "${XILINX_VIVADO}" ] && [ -d "${BSMITH_DIR}/deps/finn/finn_xsi" ] && [ ! -f "${BSMITH_DIR}/deps/finn/xsi.so" ]; then
        emit_status "BUILDING_FINNXSI"
        log_info "Building finnxsi (Vivado available and finnxsi source exists)"
        OLDPWD=$(pwd)
        cd ${BSMITH_DIR}/deps/finn/finn_xsi || {
            emit_status "ERROR" "Failed to enter finnxsi directory"
            log_error "Failed to enter finnxsi directory"
            exit 1
        }
        if python -m finn.xsi.setup --quiet; then
            log_info "finnxsi built successfully"
        else
            emit_status "ERROR" "Failed to build finnxsi"
            log_error "Failed to build finnxsi"
            exit 1
        fi
        cd $OLDPWD
    elif [ -z "${XILINX_VIVADO}" ]; then
        log_info "Skipping finnxsi build - Vivado not available"
    elif [ ! -d "${BSMITH_DIR}/deps/finn/finn_xsi" ]; then
        log_info "Skipping finnxsi build - finnxsi source not available"
    else
        log_info "finnxsi already built - skipping"
    fi
}

# Smart package management with persistent state
CACHE_FILE="/tmp/.brainsmith_packages_installed"

# Function to check if packages are already installed and working
packages_already_installed() {
    if [ -f "$CACHE_FILE" ]; then
        # Quick check if all key packages can be imported
        python -c "
try:
    import qonnx, finnexperimental, brevitas, finn, brainsmith
    exit(0)
except ImportError as e:
    exit(1)
" 2>/dev/null
        return $?
    fi
    return 1
}

# Function to install packages with proper error handling and progress
install_packages_with_progress() {
    log_info "Starting package installation process"
    emit_status "INSTALLING_PACKAGES" "starting"
    
    [ "$BSMITH_CONTAINER_MODE" != "daemon" ] && gecho "Installing development packages (this may take a moment)..."
    
    # Ensure deps directory exists
    mkdir -p "$BSMITH_DIR/deps"
    
    # Ensure Python output is unbuffered for real-time package installation output
    export PYTHONUNBUFFERED=1
    
    local install_success=true
    local failed_packages=""
    
    # qonnx (using workaround for https://github.com/pypa/pip/issues/7953)
    if [ -d "${BSMITH_DIR}/deps/qonnx" ]; then
        emit_status "INSTALLING_PACKAGES" "qonnx"
        [ "$BSMITH_CONTAINER_MODE" != "daemon" ] && gecho "Installing qonnx..."
        mv ${BSMITH_DIR}/deps/qonnx/pyproject.toml ${BSMITH_DIR}/deps/qonnx/pyproject.tmp 2>/dev/null || true
        if ! pip install --user -e ${BSMITH_DIR}/deps/qonnx; then
            install_success=false
            failed_packages+="qonnx "
        fi
        mv ${BSMITH_DIR}/deps/qonnx/pyproject.tmp ${BSMITH_DIR}/deps/qonnx/pyproject.toml 2>/dev/null || true
    fi

    # finn-experimental
    if [ -d "${BSMITH_DIR}/deps/finn-experimental" ]; then
        emit_status "INSTALLING_PACKAGES" "finn-experimental"
        [ "$BSMITH_CONTAINER_MODE" != "daemon" ] && gecho "Installing finn-experimental..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn-experimental; then
            install_success=false
            failed_packages+="finn-experimental "
        fi
    fi

    # brevitas
    if [ -d "${BSMITH_DIR}/deps/brevitas" ]; then
        emit_status "INSTALLING_PACKAGES" "brevitas"
        [ "$BSMITH_CONTAINER_MODE" != "daemon" ] && gecho "Installing brevitas..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/brevitas; then
            install_success=false
            failed_packages+="brevitas "
        fi
    fi

    # finn
    if [ -d "${BSMITH_DIR}/deps/finn" ]; then
        emit_status "INSTALLING_PACKAGES" "finn"
        [ "$BSMITH_CONTAINER_MODE" != "daemon" ] && gecho "Installing finn..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn; then
            install_success=false
            failed_packages+="finn "
        fi
    fi

    # brainsmith
    if [ -f "${BSMITH_DIR}/setup.py" ]; then
        emit_status "INSTALLING_PACKAGES" "brainsmith"
        [ "$BSMITH_CONTAINER_MODE" != "daemon" ] && gecho "Installing brainsmith..."
        if ! pip install --user -e ${BSMITH_DIR}; then
            install_success=false
            failed_packages+="brainsmith "
        fi
    else
        emit_status "ERROR" "Unable to find Brainsmith source code"
        recho "Unable to find Brainsmith source code in ${BSMITH_DIR}"
        recho "Ensure you have passed -v <path-to-brainsmith-repo>:<path-to-brainsmith-repo> to the docker run command"
        exit 1
    fi
    
    if [ "$install_success" = true ]; then
        # Mark packages as successfully installed
        touch "$CACHE_FILE"
        [ "$BSMITH_CONTAINER_MODE" != "daemon" ] && gecho "Development packages installed and cached successfully!"
        return 0
    else
        emit_status "ERROR" "Package installation failed: $failed_packages"
        recho "Failed to install packages: $failed_packages"
        recho "Some functionality may not work properly."
        return 1
    fi
}


# For daemon mode, complete ALL setup before going into background
if [ "$BSMITH_CONTAINER_MODE" = "daemon" ]; then
    log_info "Daemon mode: ensuring all packages are installed before going into background"
    # Install packages if needed
    if ! packages_already_installed; then
        install_packages_with_progress
        build_finnxsi_if_needed
    else
        log_info "Development packages already installed - using cached setup"
    fi
    
    # Create readiness marker ONLY after everything is truly ready
    log_info "Creating dependency readiness marker"
    touch /tmp/.brainsmith_deps_ready
    
    # Emit final ready status for log monitoring
    emit_status "READY"
    log_info "All setup complete - container is now fully ready for exec commands"
    # Common approach: use tail -f /dev/null to keep container alive
    exec tail -f /dev/null
fi

# execute the provided command(s)
if [ $# -gt 0 ] && [ "$1" != "" ]; then
    # For direct commands, install packages only if needed
    if ! packages_already_installed; then
        install_packages_with_progress
        build_finnxsi_if_needed
    fi
    exec bash -c "$*"
else
    # For interactive mode, install packages
    if ! packages_already_installed; then
        install_packages_with_progress
    else
        gecho "Development packages already installed - using cached setup"
    fi
    exec bash
fi
