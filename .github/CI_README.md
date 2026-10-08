# Brainsmith CI/CD System

## Architecture

```
.github/
├── actions/          # 7 modular composite actions
│   ├── build-docker/        # Docker build with verification
│   ├── check-disk/          # Disk space validation
│   ├── collect-artifacts/   # Safe artifact collection
│   ├── docker-cleanup/      # Container & build cleanup
│   ├── run-test-with-artifacts/  # Complete test lifecycle
│   ├── docker-exec/         # Command execution with container lifecycle
│   └── workflow-setup/      # Standard initialization
└── workflows/        # 3 focused workflows
    ├── pr-validation.yml.disabled (DISABLED, see note at top of file)     # Pytest + BERT Quicktest (fail-fast)
    ├── biweekly-tests.yml    # BERT Large Model Test
    └── docs.yml              # Documentation deployment
```

## Workflows

### PR Validation (`pr-validation.yml.disabled (DISABLED, see note at top of file)`)
Fast validation for pull requests and develop branch pushes.

**Triggers**: Push to `develop`, Pull Requests
**Runtime**: ~5 hours total (3 min pytest + 4 hours BERT + 1 hour setup/cleanup)
**Jobs**:
1. `pytest-validation` (Pytest Test Suite) - Fast fail-fast validation
2. `bert-quicktest` (BERT Quicktest) - Only runs if pytest passes

**Job 1: Pytest Test Suite** (~3 minutes)
1. Checkout repository
2. Setup workflow (disk check, cleanup, build)
3. Run full pytest suite with coverage (unit + fast integration + FINN integration)
4. Upload coverage report and test results
5. Collect failure artifacts if tests fail

**Job 2: BERT Quicktest** (runs only if Job 1 passes)
1. Checkout repository
2. Setup workflow (disk check, cleanup, build)
3. Run E2E test with artifact collection using `./examples/bert/quicktest.sh`

### Biweekly Tests (`biweekly-tests.yml`)
Comprehensive testing for large model validation.

**Triggers**: Biweekly schedule (Monday/Thursday 00:00 UTC)  
**Runtime**: ~24 hours  
**Job**: `bert-large-comprehensive-test` (BERT Large Model Comprehensive Test)

**Steps**:
1. Checkout repository
2. Setup workflow (disk check, cleanup, build)
3. Run BERT Large test with artifact collection

## Build Caching

BuildKit layer caching enabled (`BSMITH_DOCKER_NO_CACHE: "0"`). Typical build times:
- First build: ~12 min
- Code changes: ~1-2 min
- Dependency/Dockerfile changes: ~3-10 min

Cache (~5GB) persists on runner. Cleanup removes commit-tagged images but preserves cache layers.

## Action Architecture

### Layer 2: Composite Actions (Orchestration)

#### `workflow-setup`
Standard initialization for all workflows.
```yaml
- uses: ./.github/actions/workflow-setup
  with:
    disk-threshold-gb: 20  # or 40 for biweekly
```
**Process**: Check disk → Clean Docker → Build image

#### `run-test-with-artifacts`
Complete test lifecycle with conditional artifact collection.
```yaml
- uses: ./.github/actions/run-test-with-artifacts
  with:
    command: "cd examples/bert && make single_layer"
    timeout-minutes: 240
    artifact-name: "test-results"
    collect-on: "failure"  # or "always"
    retention-days: 7
```
**Process**: Execute test → Collect artifacts → Upload → Cleanup

### Layer 3: Core Actions (Specific Tasks)

#### Infrastructure Actions
- `check-disk` - Validates available disk space with configurable thresholds
- `docker-cleanup` - Cleans containers AND persistent build directories
- `collect-artifacts` - Collects system info, container logs, and test artifacts

#### Docker Actions  
- `build-docker` - Builds image with verification and timing fixes
- `docker-exec` - Executes commands with container lifecycle management

## Adding New Workflows

The modular architecture makes adding new workflows trivial:

```yaml
name: New Test Type

on:
  workflow_dispatch:

jobs:
  my-test:
    runs-on: pre-release
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0

      - name: Setup workflow
        uses: ./.github/actions/workflow-setup
        with:
          disk-threshold-gb: 30

      - name: Run my test
        uses: ./.github/actions/run-test-with-artifacts
        with:
          command: "my test command"
          timeout-minutes: 60
          artifact-name: "my-test-results"
          collect-on: "failure"
          retention-days: 7
```
