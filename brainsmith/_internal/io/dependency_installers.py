# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dependency installer implementations for non-Python dependencies.

Simple installer classes for different dependency types. Each installer handles
a specific installation method (git, zip, build) independently.
"""

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


# Custom exceptions for better error handling
class DependencyError(Exception):
    """Base exception for dependency operations."""

    pass


class UnknownDependencyError(DependencyError):
    """Requested dependency not found in registry."""

    pass


class InstallationError(DependencyError):
    """Dependency installation failed."""

    pass


class BuildError(DependencyError):
    """Dependency build failed."""

    pass


class RequirementError(DependencyError):
    """Required tool or dependency missing."""

    pass


class RemovalError(DependencyError):
    """Dependency removal failed."""

    pass


# Module-level helpers
def _remove_directory(name: str, dest: Path, quiet: bool) -> None:
    """Raises RemovalError if removal fails."""
    if not dest.exists():
        if not quiet:
            logger.info("%s is not installed", name)
        return

    try:
        if not quiet:
            logger.info("Removing %s from %s", name, dest)
        shutil.rmtree(dest)
    except OSError as e:
        error_msg = f"Failed to remove {name}: {e}"
        logger.error(error_msg)
        raise RemovalError(error_msg) from e


class GitDependencyInstaller:
    """Installer for git-based dependencies."""

    def install(self, name: str, dep: dict, dest: Path, force: bool, quiet: bool) -> None:
        """Clone git repository.

        Args:
            dep: Must contain 'url', optional 'ref' and 'sparse_dirs'

        Raises:
            InstallationError: If git clone fails
        """
        # Check if already exists
        if dest.exists() and not force:
            if not quiet:
                logger.info("%s already installed at %s", name, dest)
            return

        # Remove existing if force
        if dest.exists():
            shutil.rmtree(dest)

        cmd = ["git", "clone", "--quiet"]

        if "sparse_dirs" in dep:
            cmd.extend(["--filter=blob:none", "--sparse"])

        cmd.extend([dep["url"], str(dest)])

        if not quiet:
            logger.debug("Cloning %s from %s", name, dep['url'])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = f"Failed to clone {name}: {result.stderr}"
            logger.error(error_msg)
            raise InstallationError(error_msg)

        if "ref" in dep:
            subprocess.run(
                ["git", "-C", str(dest), "fetch", "--quiet", "origin", dep["ref"]],
                capture_output=True,
            )

            result = subprocess.run(
                ["git", "-C", str(dest), "checkout", "--quiet", dep["ref"]],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                error_msg = f"Failed to checkout {dep['ref']} for {name}: {result.stderr}"
                logger.error(error_msg)
                raise InstallationError(error_msg)

        if "sparse_dirs" in dep:
            subprocess.run(["git", "-C", str(dest), "sparse-checkout", "init"], capture_output=True)
            subprocess.run(
                ["git", "-C", str(dest), "sparse-checkout", "set"] + dep["sparse_dirs"],
                capture_output=True,
            )

    def remove(self, name: str, dest: Path, quiet: bool) -> None:
        """Raises RemovalError if removal fails."""
        _remove_directory(name, dest, quiet)


class ZipDependencyInstaller:
    """Installer for zip-based dependencies."""

    def __init__(self, temp_dir: Path | None = None):
        self.temp_dir = temp_dir

    def install(self, name: str, dep: dict, dest: Path, force: bool, quiet: bool) -> None:
        """Download and extract zip file.

        Args:
            dep: Must contain 'url'

        Raises:
            InstallationError: If download or extraction fails
        """
        # Check if already exists
        if dest.exists() and not force:
            if not quiet:
                logger.info("%s already installed at %s", name, dest)
            return

        # Remove existing if force
        if dest.exists():
            shutil.rmtree(dest)

        temp_dir = self.temp_dir or dest.parent
        zip_path = temp_dir / f"{name}.zip"

        try:
            if not quiet:
                logger.debug("Downloading %s from %s", name, dep['url'])

            urlretrieve(dep["url"], zip_path)

            dest.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)

        except Exception as e:
            error_msg = f"Failed to download/extract {name}: {e}"
            logger.error(error_msg)
            raise InstallationError(error_msg)

        finally:
            if zip_path.exists():
                zip_path.unlink()

    def remove(self, name: str, dest: Path, quiet: bool) -> None:
        """Raises RemovalError if removal fails."""
        _remove_directory(name, dest, quiet)


class BuildDependencyInstaller:
    """Installer for build-based dependencies."""

    def install(self, name: str, dep: dict, dest: Path, force: bool, quiet: bool) -> None:
        """Build a local dependency.

        Args:
            dep: Must contain 'build_cmd', optional 'source'

        Raises:
            BuildError: If build fails
        """
        # Special handling for finn-xsim
        if name == "finn-xsim":
            self._install_finn_xsim(force, quiet)
        else:
            self._install_generic_build(name, dep, force, quiet)

    def _install_finn_xsim(self, force: bool, quiet: bool) -> None:
        """Install FINN XSI module with Vivado.

        Raises:
            BuildError: If build fails or requirements missing
        """
        # Check if FINN package is available
        try:
            import finn

            # Check if already built
            finn_root = Path(finn.__file__).parent.parent.parent
            output_file = finn_root / "finn_xsi" / "xsi.so"

            if output_file.exists() and not force:
                if not quiet:
                    logger.info("finn-xsim already built at %s", output_file)
                return

        except ImportError:
            error_msg = "FINN package not found. Please run 'poetry install' first."
            logger.error(error_msg)
            raise BuildError(error_msg)

        # Get Vivado settings path from config
        from brainsmith.settings import get_config

        config = get_config()
        vivado_path = config.vivado_path

        if not vivado_path:
            error_msg = "Vivado path not configured. Set xilinx_path in config or BSMITH_XILINX_PATH env var."
            logger.error(error_msg)
            raise RequirementError(error_msg)

        settings_script = Path(vivado_path) / "settings64.sh"
        if not settings_script.exists():
            error_msg = f"settings64.sh not found at {settings_script}"
            logger.error(error_msg)
            raise RequirementError(error_msg)

        # Build with finn-xsim
        if not quiet:
            logger.debug("Building finn-xsim...")

        # Construct build command
        build_cmd = ["python3", "-m", "finn.xsi.setup"]
        if force:
            build_cmd.append("--force")

        # Build bash command that sources Vivado settings
        python_cmd = " ".join(build_cmd)
        bash_cmd = f"source {settings_script} && {python_cmd}"

        logger.debug("Running: %s", bash_cmd)

        # Execute build
        result = subprocess.run(["bash", "-c", bash_cmd], capture_output=True, text=True)

        # Log output at DEBUG level (visible with --logs debug)
        if result.stdout:
            for line in result.stdout.splitlines():
                if line.strip():
                    logger.debug(line)

        if result.stderr:
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.warning(line)

        # Check for errors
        if result.returncode != 0:
            error_msg = f"Failed to build finn-xsim (exit code: {result.returncode})"
            logger.error(error_msg)
            if result.stdout:
                print("\n--- Build Output ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("\n--- Build Errors ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise BuildError(error_msg)

        # Verify output file exists
        if not output_file.exists():
            error_msg = f"Build completed but output file not found: {output_file}"
            logger.error(error_msg)
            raise BuildError(error_msg)

    def _install_generic_build(self, name: str, dep: dict, force: bool, quiet: bool) -> None:
        """Install generic build dependency.

        Args:
            dep: Dependency metadata with 'source' and 'build_cmd'

        Raises:
            BuildError: If build fails
        """
        # Get deps_dir from config
        from brainsmith.settings import get_config

        deps_dir = get_config().deps_dir

        source_dir = Path(deps_dir) / dep["source"]
        if not source_dir.exists():
            error_msg = (
                f"Source directory not found: {source_dir}. Install source dependency first."
            )
            logger.error(error_msg)
            raise BuildError(error_msg)

        if not quiet:
            logger.debug("Building %s in %s", name, source_dir)

        # Run build command
        env = os.environ.copy()
        result = subprocess.run(
            dep["build_cmd"], cwd=source_dir, capture_output=True, text=True, env=env
        )

        # Log output
        if result.stdout:
            for line in result.stdout.splitlines():
                if line.strip():
                    logger.debug(line)

        if result.stderr:
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.warning(line)

        # Check for errors
        if result.returncode != 0:
            error_msg = f"Failed to build {name} (exit code: {result.returncode})"
            logger.error(error_msg)
            if result.stdout:
                print("\n--- Build Output ---", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
            if result.stderr:
                print("\n--- Build Errors ---", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
            raise BuildError(error_msg)

    def remove(self, name: str, dest: Path, quiet: bool) -> None:
        # Special handling for finn-xsim
        if name == "finn-xsim":
            try:
                import finn

                finn_root = Path(finn.__file__).parent.parent.parent
                output_file = finn_root / "finn_xsi" / "xsi.so"

                if not output_file.exists():
                    if not quiet:
                        logger.info("finn-xsim is not built")
                    return

                # Run clean command if available
                if not quiet:
                    logger.info("Cleaning finn-xsim...")

                clean_cmd = ["python3", "-m", "finn.xsi.setup", "--clean"]
                result = subprocess.run(clean_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    # Try manual removal as fallback
                    try:
                        output_file.unlink()
                    except Exception as e:
                        error_msg = f"Failed to remove finn-xsim: {e}"
                        logger.error(error_msg)
                        raise RemovalError(error_msg)

            except ImportError:
                if not quiet:
                    logger.info("finn-xsim is not installed (FINN not found)")

        else:
            # Generic build dependency removal not implemented
            if not quiet:
                logger.info("Build dependency %s removal not implemented", name)
