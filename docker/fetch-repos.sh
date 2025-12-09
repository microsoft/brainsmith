#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Git dependency management for Brainsmith
# Downloads and updates local dependencies to match pyproject.toml specifications

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
FORCE=false
INFO_ONLY=false
SPECIFIC_DEPS=()

# Parse command line arguments
show_help() {
    echo "Usage: $0 [OPTIONS] [DEPENDENCIES...]"
    echo ""
    echo "Manage Git dependencies for Brainsmith development"
    echo ""
    echo "Options:"
    echo "  --info, -i      Show status information only (no changes)"
    echo "  --force, -f     Force re-download even if repo exists and is clean"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Arguments:"
    echo "  DEPENDENCIES    Specific dependencies to fetch (default: all)"
    echo "                  Available: brevitas, qonnx, finn, onnxscript, finn-experimental, dataset-loading"
    echo ""
    echo "Examples:"
    echo "  $0                    # Fetch all dependencies"
    echo "  $0 --info             # Show status of all dependencies"
    echo "  $0 finn qonnx         # Fetch only finn and qonnx"
    echo "  $0 --info finn        # Show status of finn only"
    echo "  $0 --force brevitas   # Force re-download brevitas"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --info|-i)
            INFO_ONLY=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --*)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            SPECIFIC_DEPS+=("$1")
            shift
            ;;
    esac
done

if [ "$INFO_ONLY" = true ]; then
    echo -e "${BLUE}Analyzing Git dependencies for Brainsmith${NC}"
else
    echo -e "${BLUE}Managing Git dependencies for Brainsmith${NC}"
fi

# Define our Git dependencies - URLs and revisions
declare -A GIT_DEPS=(
    ["brevitas"]="https://github.com/Xilinx/brevitas.git@c10ef8764967e9cacc60347ce185be14e4ad97c4"
    ["qonnx"]="https://github.com/fastmachinelearning/qonnx.git@custom/brainsmith"
    ["finn"]="https://github.com/tafk7/finn.git@feature/mlo-merge"
    ["finn-experimental"]="https://github.com/Xilinx/finn-experimental.git@0724be21111a21f0d81a072fccc1c446e053f851"
    ["dataset-loading"]="https://github.com/fbcotter/dataset_loading.git@0.0.4"
)

# Create deps directory
mkdir -p deps
cd deps

# Function to check if repository has uncommitted changes
has_changes() {
    local name="$1"
    cd "$name"

    # Check if there are any uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        cd ..
        return 0  # Has changes
    fi

    # Check if there are untracked files (excluding common build artifacts)
    local untracked_count=$(git ls-files --others --exclude-standard | wc -l)
    cd ..

    if [ "$untracked_count" -gt 0 ]; then
        return 0  # Has untracked files
    fi

    return 1  # No changes
}

# Function to get current commit hash
get_current_commit() {
    local name="$1"
    cd "$name"
    local commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    cd ..
    echo "$commit"
}

# Function to resolve tag/branch/hash to commit hash
resolve_ref_to_commit() {
    local name="$1"
    local ref="$2"

    cd "$name"

    # Fetch latest from remote to ensure we have up-to-date refs
    git fetch origin --quiet 2>/dev/null || true

    local resolved_commit=""

    # For branch refs, always prefer origin's version to detect when local is behind
    # Try origin/$ref first (remote branch)
    resolved_commit=$(git rev-parse "origin/$ref" 2>/dev/null || echo "")

    # If that fails, try as-is (works for tags and commit hashes)
    if [ -z "$resolved_commit" ]; then
        resolved_commit=$(git rev-parse "$ref" 2>/dev/null || echo "")
    fi

    # If still no luck, return "unknown"
    if [ -z "$resolved_commit" ]; then
        resolved_commit="unknown"
    fi

    cd ..
    echo "$resolved_commit"
}

# Function to analyze repository status (info mode)
analyze_repo() {
    local name="$1"
    local url_rev="$2"
    local url="${url_rev%@*}"
    local rev="${url_rev#*@}"

    if [ ! -d "$name" ]; then
        echo -e "  ${RED}✗ Missing${NC} $name"
        echo -e "    Expected: $rev"
        echo -e "    Status: Not cloned"
        return 0
    fi

    local current_commit=$(get_current_commit "$name")
    local expected_commit=$(resolve_ref_to_commit "$name" "$rev")
    local status_text=""
    local changes_status=""

    # Check if at correct revision (compare actual commit hashes)
    if [ "$current_commit" = "$expected_commit" ]; then
        if has_changes "$name"; then
            echo -e "  ${YELLOW}⚠️ Modified${NC} $name"
            echo -e "    Current: ${current_commit:0:8} ${GREEN}✓${NC} (matches $rev)"
            changes_status=" (has local changes)"
        else
            echo -e "  ${GREEN}✓ Ready${NC} $name"
            echo -e "    Current: ${current_commit:0:8} ${GREEN}✓${NC} (matches $rev)"
            changes_status=" (clean)"
        fi
    else
        if has_changes "$name"; then
            echo -e "  ${RED}Outdated + Modified${NC} $name"
            echo -e "    Current: ${current_commit:0:8} (expected: $rev → ${expected_commit:0:8})"
            changes_status=" (has local changes)"
        else
            echo -e "  ${CYAN}Outdated${NC} $name"
            echo -e "    Current: ${current_commit:0:8} (expected: $rev → ${expected_commit:0:8})"
            changes_status=" (can update)"
        fi
    fi

    # Show additional details
    cd "$name"
    local branch_info=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")
    local origin_url=$(git remote get-url origin 2>/dev/null || echo "unknown")

    echo -e "    Branch: $branch_info"
    echo -e "    Remote: ${origin_url##*/}"  # Just show repo name

    # Check for origin mismatch
    if [ "$origin_url" != "$url" ]; then
        echo -e "    ${YELLOW}⚠️ Origin mismatch${NC}: expected ${url##*/}"
    fi
    cd ..
}

# Function to clone or update a repository
update_repo() {
    local name="$1"
    local url_rev="$2"
    local url="${url_rev%@*}"
    local rev="${url_rev#*@}"

    if [ -d "$name" ]; then
        # Check if origin URL matches expected
        cd "$name"
        local current_origin=$(git remote get-url origin 2>/dev/null || echo "")
        cd ..

        local origin_mismatch=false
        if [ "$current_origin" != "$url" ]; then
            origin_mismatch=true
        fi

        local current_commit=$(get_current_commit "$name")
        local expected_commit=$(resolve_ref_to_commit "$name" "$rev")

        # Check for origin mismatch first
        if [ "$origin_mismatch" = true ]; then
            if [ "$FORCE" = true ]; then
                echo -e "  ${YELLOW}Fixing origin${NC} $name"
                echo -e "    Updating origin: ${current_origin} → ${url}"
                cd "$name"
                git remote set-url origin "$url"
                git fetch origin --quiet
                cd ..
            else
                echo -e "  ${CYAN}Skipping${NC} $name - origin URL mismatch"
                echo -e "    Current origin: ${current_origin}"
                echo -e "    Expected origin: ${url}"
                echo -e "    Use --force to update origin URL"
                return 0
            fi
        fi

        # Check if already at correct revision and no changes
        if [ "$current_commit" = "$expected_commit" ]; then
            if has_changes "$name"; then
                if [ "$FORCE" = true ]; then
                    echo -e "  ${RED}Force removing${NC} $name (has local changes)"
                    rm -rf "$name"
                else
                    echo -e "  ${CYAN}Skipping${NC} $name - at correct revision but has local changes"
                    echo -e "    Current: ${current_commit:0:8}"
                    echo -e "    Use --force to overwrite local changes"
                    return 0
                fi
            else
                echo -e "  ${GREEN}✓ Skip${NC} $name - already at correct revision"
                echo -e "    Current: ${current_commit:0:8}"
                return 0
            fi
        else
            if has_changes "$name"; then
                if [ "$FORCE" = true ]; then
                    echo -e "  ${RED}Force removing${NC} $name (has local changes)"
                    rm -rf "$name"
                else
                    echo -e "  ${CYAN}Skipping${NC} $name - has local changes"
                    echo -e "    Current: ${current_commit:0:8} (expected: $rev → ${expected_commit:0:8})"
                    echo -e "    Use --force to overwrite local changes"
                    return 0
                fi
            else
                echo -e "  ${YELLOW}Updating${NC} $name"
                echo -e "    ${current_commit:0:8} → $rev (${expected_commit:0:8})"
                cd "$name"
                git fetch --all --quiet
                # Checkout the resolved commit hash to ensure we get the remote version
                git -c advice.detachedHead=false checkout "$expected_commit" --quiet
                echo -e "    ${GREEN}✓${NC} Updated to $rev"
                cd ..
                return 0
            fi
        fi
    fi

    # Clone new repository (either didn't exist or was force-removed)
    echo -e "  ${GREEN}Cloning${NC} $name"
    echo -e "    Target: $rev"
    git clone --quiet "$url" "$name"
    cd "$name"
    git -c advice.detachedHead=false checkout "$rev" --quiet
    local final_commit=$(git rev-parse HEAD 2>/dev/null | cut -c1-8 || echo "unknown")
    echo -e "    ${GREEN}✓${NC} Cloned and checked out $rev (${final_commit})"
    cd ..
}

# Determine which dependencies to process
if [ ${#SPECIFIC_DEPS[@]} -eq 0 ]; then
    # No specific deps specified, process all
    DEPS_TO_PROCESS=("${!GIT_DEPS[@]}")
else
    # Validate specified dependencies
    DEPS_TO_PROCESS=()
    for dep in "${SPECIFIC_DEPS[@]}"; do
        if [[ -v "GIT_DEPS[$dep]" ]]; then
            DEPS_TO_PROCESS+=("$dep")
        else
            echo -e "${RED}Error: Unknown dependency '$dep'${NC}"
            echo "Available dependencies: ${!GIT_DEPS[*]}"
            exit 1
        fi
    done
fi

# Show what we're going to do
if [ "$INFO_ONLY" = true ]; then
    if [ ${#DEPS_TO_PROCESS[@]} -eq ${#GIT_DEPS[@]} ]; then
        echo "Analyzing all dependencies..."
    else
        echo "Analyzing dependencies: ${DEPS_TO_PROCESS[*]}"
    fi
else
    if [ ${#DEPS_TO_PROCESS[@]} -eq ${#GIT_DEPS[@]} ]; then
        echo "Processing all dependencies..."
    else
        echo "Processing dependencies: ${DEPS_TO_PROCESS[*]}"
    fi

    if [ "$FORCE" = true ]; then
        echo -e "${YELLOW}Force mode enabled - will overwrite local changes${NC}"
    fi
fi

echo ""

# Process specified repositories
if [ "$INFO_ONLY" = true ]; then
    # Info mode - analyze status only
    for name in "${DEPS_TO_PROCESS[@]}"; do
        analyze_repo "$name" "${GIT_DEPS[$name]}"
    done
else
    # Normal mode - update repositories
    for name in "${DEPS_TO_PROCESS[@]}"; do
        update_repo "$name" "${GIT_DEPS[$name]}"
    done
fi

cd ..

echo ""

if [ "$INFO_ONLY" = true ]; then
    echo -e "${BLUE}Analysis complete!${NC}"
    echo ""
    echo "To update dependencies:"
    echo "  ./fetch-repos.sh             # Update all"
    echo "  ./fetch-repos.sh <names>     # Update specific ones"
    echo "  ./fetch-repos.sh --force     # Overwrite local changes"
else
    echo -e "${GREEN}✓ Git dependencies ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: poetry install"
    echo "  2. Optional: poetry run python -m brainsmith.core.plugins.dependencies setup_cppsim"
fi
