# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
V80 Deployment Build Steps

Custom steps for building the V80 shell and generating deployment artifacts.
These steps interface with the CMake-based V80 shell build system (v80_shell/).
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Any, Optional

from brainsmith.core.plugins import step

logger = logging.getLogger(__name__)


def _find_v80_shell_dir(cfg: Any) -> Path:
    """Locate the V80 shell source directory (v80_shell/).

    Resolution priority:
    1. cfg.v80_shell_dir (explicit config)
    2. BWAVE_DIR environment variable (legacy compatibility)
    3. <brainsmith_root>/v80_shell/ (default)
    """
    # 1. Check config override
    if hasattr(cfg, 'v80_shell_dir') and cfg.v80_shell_dir:
        path = Path(cfg.v80_shell_dir)
        if path.exists():
            return path
        raise RuntimeError(f"v80_shell_dir specified but not found: {path}")

    # 2. Check environment variable (legacy)
    if 'BWAVE_DIR' in os.environ:
        path = Path(os.environ['BWAVE_DIR'])
        if path.exists():
            return path
        logger.warning(f"BWAVE_DIR set but path not found: {path}")

    # 3. Default: relative to brainsmith installation
    brainsmith_root = Path(__file__).parent.parent.parent
    default_path = brainsmith_root / 'v80_shell'
    if default_path.exists():
        return default_path

    raise RuntimeError(
        "V80 shell directory (v80_shell/) not found. "
        "Set v80_shell_dir in config or BWAVE_DIR environment variable."
    )


def _check_tool_available(tool: str) -> bool:
    """Check if a tool is available in PATH."""
    try:
        result = subprocess.run(
            ['which', tool],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _check_vivado_available() -> bool:
    """Check if Vivado is available in PATH or XILINX_VIVADO is set."""
    # Check XILINX_VIVADO environment variable
    if 'XILINX_VIVADO' in os.environ:
        vivado_path = Path(os.environ['XILINX_VIVADO']) / 'bin' / 'vivado'
        if vivado_path.exists():
            return True

    return _check_tool_available('vivado')


def _get_torch_cmake_prefix() -> Optional[str]:
    """Get the CMake prefix path for PyTorch/LibTorch.

    Returns the path to PyTorch's CMake config files, or None if PyTorch
    is not installed or doesn't provide cmake_prefix_path.
    """
    try:
        from torch.utils import cmake_prefix_path
        cmake_path = cmake_prefix_path
        if cmake_path and Path(cmake_path).exists():
            logger.debug(f"Found PyTorch CMake prefix: {cmake_path}")
            return cmake_path
    except ImportError:
        logger.debug("PyTorch not installed or cmake_prefix_path not available")
    except Exception as e:
        logger.debug(f"Could not get PyTorch cmake_prefix_path: {e}")

    return None


def _run_cmake_build(
    build_dir: Path,
    target: str,
    cores: int = 4,
    log_file: Optional[Path] = None
) -> int:
    """Run a make target and stream output.

    Args:
        build_dir: CMake build directory
        target: Make target to build (e.g., 'hw_project')
        cores: Number of parallel jobs
        log_file: Optional file to write logs to

    Returns:
        Exit code (0 = success)
    """
    cmd = ['make', '-j', str(cores), target]
    logger.info(f"Running: {' '.join(cmd)} in {build_dir}")

    process = subprocess.Popen(
        cmd,
        cwd=build_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )

    log_handle = open(log_file, 'w') if log_file else None

    try:
        for line in process.stdout:
            logger.debug(line.rstrip())
            if log_handle:
                log_handle.write(line)

        process.wait()
        return process.returncode
    finally:
        if log_handle:
            log_handle.close()


def _setup_v80_build_environment(cfg: Any) -> tuple:
    """Common setup for V80 build steps.

    Returns:
        Tuple of (stitched_ip_dir, handover_file, v80_shell_dir, build_dir, deploy_dir, log_dir)
    """
    from finn.builder.build_dataflow_config import DataflowOutputType

    # Check prerequisites
    if DataflowOutputType.STITCHED_IP not in cfg.generate_outputs:
        return None

    stitched_ip_dir = (Path(cfg.output_dir) / 'stitched_ip').resolve()
    if not stitched_ip_dir.exists():
        raise RuntimeError(
            f"Stitched IP directory not found: {stitched_ip_dir}. "
            "Ensure create_stitched_ip and shell_metadata_handover ran successfully."
        )

    handover_file = stitched_ip_dir / 'shell_handover.json'
    if not handover_file.exists():
        raise RuntimeError(
            f"shell_handover.json not found in {stitched_ip_dir}. "
            "Ensure shell_metadata_handover step completed."
        )

    # Check required tools
    if not _check_tool_available('cmake'):
        raise RuntimeError(
            "CMake not found. Install with: apt-get install cmake"
        )

    if not _check_tool_available('make'):
        raise RuntimeError(
            "Make not found. Install with: apt-get install build-essential"
        )

    # Locate V80 shell sources
    v80_shell_dir = _find_v80_shell_dir(cfg)
    logger.info(f"Using V80 shell source: {v80_shell_dir}")

    # Create build directory
    build_dir = Path(cfg.output_dir) / 'v80_build'
    build_dir.mkdir(parents=True, exist_ok=True)

    # Create sw/export subdirectories (needed by hw_project even if BUILD_SW/BUILD_PY are OFF)
    sw_export_dir = build_dir / 'sw' / 'export'
    (sw_export_dir / 'include').mkdir(parents=True, exist_ok=True)
    (sw_export_dir / 'config').mkdir(parents=True, exist_ok=True)
    (sw_export_dir / 'reference').mkdir(parents=True, exist_ok=True)

    # Create deployment directory
    deploy_dir = Path(cfg.output_dir) / 'deployment'
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_dir = deploy_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    return stitched_ip_dir, handover_file, v80_shell_dir, build_dir, deploy_dir, log_dir


def _run_cmake_configure(
    cfg: Any,
    stitched_ip_dir: Path,
    v80_shell_dir: Path,
    build_dir: Path,
    log_dir: Path,
) -> None:
    """Run CMake configure if not already done."""
    clock_mhz = getattr(cfg, 'v80_clock_mhz', 250)
    compile_cores = getattr(cfg, 'v80_compile_cores', 4)

    # Check if CMake was already configured (Makefile exists)
    makefile = build_dir / 'Makefile'
    if makefile.exists():
        logger.info("CMake already configured, skipping configure step")
        return

    # Get optional debug configuration
    en_vio_debug = getattr(cfg, 'v80_enable_vio_debug', False)

    logger.info("Configuring V80 deployment build...")
    cmake_cmd = [
        'cmake',
        '-S', str(v80_shell_dir),
        '-B', str(build_dir),
        f'-DCORE_PATH={stitched_ip_dir}',
        f'-DBWAVE_DIR={v80_shell_dir}',
        f'-DACLK_F={clock_mhz}',
        f'-DCOMP_CORES={compile_cores}',
        '-DBUILD_HW=ON',
        '-DBUILD_PY=ON',  # Always enable so both hw and sw targets are available
        '-DBUILD_SW=OFF',  # C++ runtime not needed for Python workflow
        f'-DEN_VIO_DEBUG={"ON" if en_vio_debug else "OFF"}',
    ]

    # Add PyTorch CMake prefix path for Python bindings
    torch_cmake_prefix = _get_torch_cmake_prefix()
    if not torch_cmake_prefix:
        # Fallback to common location
        fallback_path = Path('/usr/local/lib/python3.10/dist-packages/torch/share/cmake')
        if fallback_path.exists():
            torch_cmake_prefix = str(fallback_path)
            logger.info(f"Using fallback PyTorch CMake prefix: {torch_cmake_prefix}")
    if torch_cmake_prefix:
        cmake_cmd.append(f'-DCMAKE_PREFIX_PATH={torch_cmake_prefix}')
        logger.info(f"Using PyTorch CMake prefix: {torch_cmake_prefix}")
    else:
        logger.warning(
            "PyTorch cmake_prefix_path not found. If CMake fails to find Torch, "
            "install PyTorch or set CMAKE_PREFIX_PATH manually."
        )

    logger.info(f"CMake command: {' '.join(cmake_cmd)}")

    # Build environment with CMAKE_PREFIX_PATH for Torch
    cmake_env = os.environ.copy()
    if torch_cmake_prefix:
        existing_prefix = cmake_env.get('CMAKE_PREFIX_PATH', '')
        if existing_prefix:
            cmake_env['CMAKE_PREFIX_PATH'] = f"{torch_cmake_prefix}:{existing_prefix}"
        else:
            cmake_env['CMAKE_PREFIX_PATH'] = torch_cmake_prefix
        logger.info(f"Setting CMAKE_PREFIX_PATH={cmake_env['CMAKE_PREFIX_PATH']}")

    # Write CMake configure log
    cmake_log = log_dir / 'cmake_configure.log'
    with open(cmake_log, 'w') as f:
        result = subprocess.run(
            cmake_cmd,
            cwd=cfg.output_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=cmake_env
        )

    if result.returncode != 0:
        logger.error(f"CMake configure failed. See {cmake_log}")
        raise RuntimeError(f"CMake configure failed. Check {cmake_log} for details.")

    logger.info("CMake configuration complete")


@step(
    name="v80_hw_build",
    category="shell",
    dependencies=["shell_metadata_handover"],
    description="Build V80 hardware (synthesis and implementation)"
)
def v80_hw_build(model: Any, cfg: Any) -> Any:
    """
    Build the V80 hardware from stitched IP.

    This step:
    1. Configures CMake with the stitched IP path (if not already done)
    2. Runs hw_project -> hw_synth -> hw_compile
    3. Collects hardware outputs into deployment folder

    Configuration options (from cfg):
        - v80_clock_mhz: int (default 250) - clock frequency
        - v80_compile_cores: int (default 4) - parallel compilation
        - v80_shell_dir: str (optional) - path to V80 shell sources
        - v80_enable_vio_debug: bool (default False) - enable VIO debug core
    """
    # Check Vivado is available for hardware build
    if not _check_vivado_available():
        raise RuntimeError(
            "Vivado not found. Ensure Vivado is in PATH or XILINX_VIVADO is set."
        )

    # Setup build environment
    env = _setup_v80_build_environment(cfg)
    if env is None:
        logger.warning("Skipping v80_hw_build: STITCHED_IP not in generate_outputs")
        return model

    stitched_ip_dir, handover_file, v80_shell_dir, build_dir, deploy_dir, log_dir = env
    compile_cores = getattr(cfg, 'v80_compile_cores', 4)

    # Run CMake configure if needed
    _run_cmake_configure(cfg, stitched_ip_dir, v80_shell_dir, build_dir, log_dir)

    # === Hardware Build ===
    # hw_project
    logger.info("Creating Vivado project (hw_project)...")
    ret = _run_cmake_build(
        build_dir, 'hw_project',
        cores=1,  # Project creation is not parallelizable
        log_file=log_dir / 'hw_project.log'
    )
    if ret != 0:
        raise RuntimeError(f"hw_project failed. Check {log_dir}/hw_project.log")

    # hw_synth
    logger.info("Running synthesis (hw_synth)...")
    ret = _run_cmake_build(
        build_dir, 'hw_synth',
        cores=compile_cores,
        log_file=log_dir / 'hw_synth.log'
    )
    if ret != 0:
        raise RuntimeError(f"hw_synth failed. Check {log_dir}/hw_synth.log")

    # hw_compile
    # Use single core for implementation to avoid PLM "Bad file descriptor" issue
    logger.info("Running implementation (hw_compile)...")
    ret = _run_cmake_build(
        build_dir, 'hw_compile',
        cores=1,
        log_file=log_dir / 'hw_compile.log'
    )
    if ret != 0:
        raise RuntimeError(f"hw_compile failed. Check {log_dir}/hw_compile.log")

    logger.info("Hardware build complete")

    # === Collect Hardware Artifacts ===
    hw_root = build_dir / 'hw'

    # Bitstreams
    bitstream_src = hw_root / 'bitstreams'
    bitstream_dst = deploy_dir / 'bitstreams'
    if bitstream_src.exists():
        shutil.copytree(bitstream_src, bitstream_dst, dirs_exist_ok=True)
        logger.info(f"Copied bitstreams to {bitstream_dst}")
    else:
        logger.warning(f"Bitstream directory not found: {bitstream_src}")

    # Reports
    report_src = hw_root / 'reports'
    report_dst = deploy_dir / 'reports'
    if report_src.exists():
        shutil.copytree(report_src, report_dst, dirs_exist_ok=True)
        logger.info(f"Copied reports to {report_dst}")

    # Copy shell_handover.json for reference
    shutil.copy2(handover_file, deploy_dir / 'shell_handover.json')

    logger.info(f"Hardware artifacts collected in {deploy_dir}")

    return model


@step(
    name="v80_sw_build",
    category="shell",
    dependencies=["shell_metadata_handover"],
    description="Build V80 software (Python bindings)"
)
def v80_sw_build(model: Any, cfg: Any) -> Any:
    """
    Build the V80 Python bindings from stitched IP.

    This step:
    1. Configures CMake with the stitched IP path (if not already done)
    2. Builds sw_python (Python bindings)
    3. Collects software outputs into deployment folder

    Configuration options (from cfg):
        - v80_clock_mhz: int (default 250) - clock frequency
        - v80_compile_cores: int (default 4) - parallel compilation
        - v80_shell_dir: str (optional) - path to V80 shell sources
    """
    # Setup build environment
    env = _setup_v80_build_environment(cfg)
    if env is None:
        logger.warning("Skipping v80_sw_build: STITCHED_IP not in generate_outputs")
        return model

    stitched_ip_dir, handover_file, v80_shell_dir, build_dir, deploy_dir, log_dir = env
    compile_cores = getattr(cfg, 'v80_compile_cores', 4)

    # Run CMake configure if needed
    _run_cmake_configure(cfg, stitched_ip_dir, v80_shell_dir, build_dir, log_dir)

    # === Software Build ===
    logger.info("Building Python bindings (sw_python)...")
    ret = _run_cmake_build(
        build_dir, 'sw_python',
        cores=compile_cores,
        log_file=log_dir / 'sw_python.log'
    )
    if ret != 0:
        raise RuntimeError(f"sw_python failed. Check {log_dir}/sw_python.log")

    logger.info("Python bindings build complete")

    # === Collect Software Artifacts ===
    sw_root = build_dir / 'sw'

    # Python module
    python_src = sw_root / 'python'
    python_dst = deploy_dir / 'python'
    if python_src.exists():
        shutil.copytree(python_src, python_dst, dirs_exist_ok=True)
        logger.info(f"Copied Python module to {python_dst}")
    else:
        logger.warning(f"Python module directory not found: {python_src}")

    # Config files
    config_src = sw_root / 'export' / 'config'
    config_dst = deploy_dir / 'config'
    if config_src.exists():
        shutil.copytree(config_src, config_dst, dirs_exist_ok=True)
        logger.info(f"Copied config to {config_dst}")

    # Reference files
    ref_src = sw_root / 'export' / 'reference'
    ref_dst = deploy_dir / 'reference'
    if ref_src.exists():
        shutil.copytree(ref_src, ref_dst, dirs_exist_ok=True)
        logger.info(f"Copied reference files to {ref_dst}")

    # Copy shell_handover.json for reference (if not already copied by hw_build)
    handover_dst = deploy_dir / 'shell_handover.json'
    if not handover_dst.exists():
        shutil.copy2(handover_file, handover_dst)

    logger.info(f"Software artifacts collected in {deploy_dir}")

    # Store deployment path in model metadata
    model.set_metadata_prop("v80_deployment_dir", str(deploy_dir))

    return model


