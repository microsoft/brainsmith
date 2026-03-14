"""Global pytest configuration and fixtures.

This conftest.py is the root configuration for the DSE integration test suite.
"""


import pytest

from brainsmith.registry import reset_registry
from brainsmith.registry._state import _discovered_sources
from brainsmith.settings import reset_config
from brainsmith.settings.validation import ensure_environment_sourced

# Validate environment is sourced before any tests run
# This ensures FINN_ROOT, VIVADO_PATH, etc. are available for tests
ensure_environment_sourced()

# Populate discovered sources for proper source detection
# This allows @backend/@kernel decorators to correctly detect "brainsmith" source
# without running full discovery (which test framework avoids for speed)
_discovered_sources.add("brainsmith")
_discovered_sources.add("finn")
_discovered_sources.add("project")

# Import test components - these register @step, @kernel, @backend decorators
# Available for tests that need globally-registered test components
import tests.fixtures.components.backends  # noqa: F401 - Registers test backends via @backend decorator
import tests.fixtures.components.kernels  # noqa: F401 - Registers test kernels via @kernel decorator
import tests.fixtures.components.steps  # noqa: F401 - Registers test steps via @step decorator
from tests.fixtures.dse.blueprints import *  # noqa: F403
from tests.fixtures.dse.design_spaces import *  # noqa: F403

# Import kernel test helpers for easy access in all kernel tests
# Use these for unit testing kernels (schema, inference, transformation)
# For parity testing (comparing implementations), use tests/parity/ParityTestBase
# Phase 4: Fixture imports for global availability
from tests.fixtures.models import *  # noqa: F403


def pytest_addoption(parser):
    """Add custom command-line options for pytest.

    Custom options:
    - --seed: Random seed for test data generation (default: 42)

    Usage:
        pytest --seed=12345          # Use specific seed
        pytest --seed=42             # Default seed (deterministic)
        pytest                       # Uses default seed (42)

    Reproducibility workflow:
        1. Test fails with seed=42
        2. Re-run with same seed: pytest --seed=42
        3. Same random data generated, failure reproduces
    """
    parser.addoption(
        "--seed",
        action="store",
        default=42,
        type=int,
        help="Random seed for deterministic test data generation (default: 42)",
    )


def pytest_configure(config):
    """Configure pytest with custom settings.

    Validates that brainsmith environment is sourced before running tests.
    Environment must be activated via:
        source .brainsmith/env.sh
    or:
        direnv allow
    """
    # Validate environment is sourced before running tests
    from brainsmith.settings.validation import ensure_environment_sourced

    ensure_environment_sourced()

    # Register v5.0 markers programmatically (also defined in pytest.ini)
    config.addinivalue_line(
        "markers",
        "certification: Comprehensive test sweep across all supported configurations (v5.0)",
    )
    config.addinivalue_line("markers", "validation_v5: Tactical corner case tests for CI/CD (v5.0)")


# ============================================================================
# v5.0 Session-Scoped Fixtures (Orthogonal Parameterization)
# ============================================================================


@pytest.fixture(scope="session")
def model_cache():
    """Session-scoped model cache for computational reuse.

    Provides lazy caching of:
    - Stage 1 Models: f(test_id)
    - Golden References: f(test_id)
    - Stage 2 Models: f(test_id)
    - Stage 3 Models: f(test_id, fpgapart)

    Cache is shared across all tests in session, enabling
    reuse when running multiple depths for same test case.

    Scope: session (entire pytest run)
    Lifetime: Cleared when pytest exits
    """
    from tests.fixtures.model_cache import ModelCache

    cache = ModelCache()

    # Optional: Print stats at end of session
    yield cache

    # Uncomment to see cache statistics
    # cache.print_stats()


@pytest.fixture
def platform_config():
    """Default platform configuration.

    Validation tests use this default (single platform).
    Certification tests override with @pytest.fixture(params=[...]).

    Default: ZYNQ_7020 (xc7z020clg400-1)
    Override: Define in kernel-specific conftest.py or test file
    """
    from tests.frameworks.test_config import PlatformConfig

    return PlatformConfig(fpgapart="xc7z020clg400-1")


@pytest.fixture
def validation_config():
    """Default validation configuration.

    Provides standard tolerances for most tests.
    Individual tests can override for specific needs.

    Default: VALIDATION_STANDARD (standard tolerances)
    Override: Define in test file for custom tolerances
    """
    from tests.frameworks.test_config import ValidationConfig

    return ValidationConfig()  # Uses defaults: rtol=1e-7, atol=1e-9


# ============================================================================
# Original Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_seed(request):
    """Provide deterministic random seed for test data generation.

    Returns the seed value from the --seed CLI option (default: 42).
    Session-scoped to ensure consistent seed across entire test run.

    Individual test classes can override via get_test_seed() method.

    Usage in tests:
        def test_something(test_seed):
            # test_seed is available (e.g., 42)
            inputs = make_execution_context(..., seed=test_seed)

    Usage in test frameworks:
        class TestMyKernel(KernelTest):
            def get_test_seed(self):
                # Override to customize seed
                return 99

    CLI usage:
        pytest --seed=12345    # Custom seed
        pytest                 # Default seed (42)

    Reproducibility:
        Same seed → Same random data → Same test results
    """
    return request.config.getoption("--seed")


@pytest.fixture(scope="session")
def test_workspace(tmp_path_factory):
    """Session-scoped workspace for FINN builds.

    All integration tests use this shared workspace for output directories.
    Avoids creating temporary directories per test.
    """
    return tmp_path_factory.mktemp("dse_workspace")


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Isolated environment with real project structure and test plugins.

    Creates:
    - tmp_path/test_project/.brainsmith/
    - tmp_path/test_project/plugins/
    - Real brainsmith.yaml
    - BSMITH_PROJECT_DIR env var

    No mocking. Real settings system. Real component discovery.

    Test components (kernels, steps, backends) are globally available via
    decorator registration in conftest.py imports, not via file discovery.
    """
    # Create project structure
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    brainsmith_dir = project_dir / ".brainsmith"
    brainsmith_dir.mkdir()

    plugins_dir = project_dir / "plugins"
    plugins_dir.mkdir()

    # Write real config file
    config_file = project_dir / "brainsmith.yaml"
    config_file.write_text(
        """
cache_components: true
component_sources: {}
"""
    )

    # Set environment (monkeypatch auto-cleans on teardown)
    monkeypatch.setenv("BSMITH_PROJECT_DIR", str(project_dir))

    # Clear registry and config state using public API
    reset_registry()
    reset_config()

    yield project_dir

    # Cleanup using public API
    reset_registry()
    reset_config()


@pytest.fixture
def empty_env(tmp_path, monkeypatch):
    """Minimal environment with no component sources.

    Creates:
    - tmp_path/test_project/.brainsmith/
    - Real brainsmith.yaml (no component sources)
    - BSMITH_PROJECT_DIR env var

    No mocking. For tests that only need core brainsmith components.

    Use this fixture when you don't need test plugins, just core components.
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    brainsmith_dir = project_dir / ".brainsmith"
    brainsmith_dir.mkdir()

    # Empty config - no component sources
    config_file = project_dir / "brainsmith.yaml"
    config_file.write_text(
        """
cache_components: false
component_sources: {}
"""
    )

    # Set environment
    monkeypatch.setenv("BSMITH_PROJECT_DIR", str(project_dir))

    # Clear state using public API
    reset_registry()
    reset_config()

    yield project_dir

    # Cleanup using public API
    reset_registry()
    reset_config()


@pytest.fixture(scope="session")
def setup_parity_imports():
    """Setup imports for parity test modules.

    NOTE: Old parity frameworks (ParityTestBase) have been replaced.
    Use new frameworks from tests/frameworks/ instead.

    Adds tests/parity/ to sys.path for remaining parity utilities:
    - assertions.py (parity-specific assertions)
    - test_fixtures.py (make_execution_context)

    New framework usage:
        from tests.frameworks.kernel_test import KernelTest
        from tests.frameworks.kernel_parity_test import KernelParityTest

    This eliminates brittle sys.path manipulation in individual test files.
    """
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent.parent
    tests_parity_dir = repo_root / "tests" / "parity"

    if str(tests_parity_dir) not in sys.path:
        sys.path.insert(0, str(tests_parity_dir))

    yield

    # Cleanup: remove from sys.path
    if str(tests_parity_dir) in sys.path:
        sys.path.remove(str(tests_parity_dir))


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests and handle parameterization sweeps.

    1. Auto-marks tests based on location:
       - tests/unit/ -> @pytest.mark.unit
       - tests/integration/ -> @pytest.mark.integration
       - integration/fast/ -> @pytest.mark.fast
       - integration/finn/ -> @pytest.mark.finn_build
       - integration/rtl/ -> @pytest.mark.rtlsim, @pytest.mark.slow
       - integration/hardware/ -> @pytest.mark.bitfile, @pytest.mark.hardware

    2. Parameterizes tests for classes with sweep methods:
       - Classes with get_dtype_sweep() get N copies per dtype config
       - Classes with get_shape_sweep() get N copies per shape config
    """
    # First pass: Auto-mark based on directory
    for item in items:
        # Mark unit tests
        if "/tests/unit/" in str(item.fspath) or str(item.fspath).endswith("/tests/unit"):
            item.add_marker(pytest.mark.unit)
        # Mark all integration tests
        if "/tests/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Mark specific integration subdirectories
        if "integration/fast" in str(item.fspath):
            item.add_marker(pytest.mark.fast)
        elif "integration/finn" in str(item.fspath):
            item.add_marker(pytest.mark.finn_build)
        elif "integration/rtl" in str(item.fspath):
            item.add_marker(pytest.mark.rtlsim)
            item.add_marker(pytest.mark.slow)
        elif "integration/hardware" in str(item.fspath):
            item.add_marker(pytest.mark.bitfile)
            item.add_marker(pytest.mark.hardware)

    # Second pass: Handle sweep parameterization
    # Build map of classes that need parameterization
    sweep_classes = {}  # cls -> (sweep_type, configs)

    for item in items:
        if item.cls is None:
            continue

        # Check if already processed this class
        if item.cls in sweep_classes:
            continue

        # Check for dtype sweep
        if hasattr(item.cls, "get_dtype_sweep"):
            instance = item.cls()
            sweep_configs = instance.get_dtype_sweep()
            if sweep_configs:
                sweep_classes[item.cls] = ("dtype", sweep_configs)
                continue

        # Check for shape sweep
        if hasattr(item.cls, "get_shape_sweep"):
            instance = item.cls()
            sweep_configs = instance.get_shape_sweep()
            if sweep_configs:
                sweep_classes[item.cls] = ("shape", sweep_configs)

    # If no sweep classes found, done
    if not sweep_classes:
        return

    # Third pass: Duplicate test items for sweep classes
    new_items = []

    for item in items:
        if item.cls not in sweep_classes:
            # Keep non-sweep tests as-is
            new_items.append(item)
            continue

        # This test belongs to a sweep class - create N copies
        sweep_type, configs = sweep_classes[item.cls]

        for idx, sweep_config in enumerate(configs):
            # Generate ID for this config
            if sweep_type == "dtype":
                id_parts = []
                for key, dtype in sweep_config.items():
                    dtype_name = str(dtype).split("[")[-1].rstrip("]").strip("'\"")
                    id_parts.append(f"{key}={dtype_name}")
                config_id = "_".join(id_parts)
            else:  # shape
                id_parts = []
                for key, value in sweep_config.items():
                    if isinstance(value, tuple):
                        shape_str = "x".join(str(d) for d in value)
                        id_parts.append(f"{key}={shape_str}")
                    else:
                        id_parts.append(f"{key}={value}")
                config_id = "_".join(id_parts)

            # Create new test item using pytest.Function
            # This is the proper way to create parameterized test items
            new_item = pytest.Function.from_parent(
                parent=item.parent,
                name=f"{item.name}[{config_id}]",
                callobj=item.obj,
            )

            # Store sweep config on the item for pytest_runtest_setup
            new_item._sweep_config = sweep_config
            new_item._sweep_type = sweep_type

            new_items.append(new_item)

    # Update items list
    items[:] = new_items


def pytest_runtest_setup(item):
    """Apply sweep configuration before each test runs.

    This hook runs before each test method executes and applies
    dtype/shape configuration to the test instance if present.
    """
    # Check if this test has sweep configuration (set by pytest_collection_modifyitems)
    if hasattr(item, "_sweep_config") and hasattr(item, "_sweep_type"):
        config = item._sweep_config
        sweep_type = item._sweep_type

        # Get the test instance (only available for class-based tests)
        if hasattr(item, "instance") and item.instance is not None:
            # Apply appropriate configuration
            if sweep_type == "dtype":
                if hasattr(item.instance, "configure_for_dtype_config"):
                    item.instance.configure_for_dtype_config(config)
            elif sweep_type == "shape":
                if hasattr(item.instance, "configure_for_shape_config"):
                    item.instance.configure_for_shape_config(config)
