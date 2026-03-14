# Blueprint Schema Reference

Blueprints are YAML files that define the design space for FPGA accelerator generation, including hardware kernels, build steps, and exploration parameters.

## Schema Structure

```yaml
# Required: Blueprint metadata
name: "string"                          # Blueprint name
description: "string"                   # Optional: Blueprint description

# Optional: Inherit from parent blueprint
extends: "relative/path/to/parent.yaml" # Path relative to child blueprint

# Required: Core configuration
clock_ns: 5.0                          # Target clock period in nanoseconds (required)
output: "estimates"                    # Output type: estimates | rtl | bitfile
                                      # Default: "estimates"
board: "Pynq-Z1"                      # Target FPGA board (required for rtl/bitfile)

# Optional: Direct FINN parameter overrides
finn_config:                          # Maps internally to finn_overrides
  minimize_bit_width: false
  rtlsim_batch_size: 100
  shell_flow_type: "vivado_zynq"
  # Any other FINN DataflowBuildConfig parameter...

# Required: Design space definition
design_space:
  # Kernel definitions (for hardware mapping)
  kernels:
    - KernelName                       # Use all available backends
    - KernelName: BackendName          # Specific backend only
    - KernelName: [Backend1, Backend2] # Multiple specific backends

  # Build pipeline steps
  steps:
    - "step_name"                     # Single step
    - ["optionA", "optionB"]          # Branch: mutually exclusive options
    - ["step", ~]                     # Optional step (~ or null = skip)

    # Step operations (for inheritance or organization)
    - after: "target_step"
      insert: "new_step"              # Insert single step
    - before: "target_step"
      insert:
        - "step1"                     # Insert multiple steps
        - "step2"
        - ["step3a", "step3b"]        # Insert a branching point
    - replace: "old_step"
      with: [["new1", "new2"]]        # Replace with a branching point
    - remove: "unwanted_step"         # Remove a step
    - at_start:
        insert: "first_step"
    - at_end:
        insert: ["last1", "last2"]
```

## Field Definitions

### Core Configuration

#### clock_ns (required)
The target clock period in nanoseconds. This is the only required configuration field.
```yaml
clock_ns: 5.0    # 5ns = 200MHz clock frequency
```

#### output
Determines how far to proceed in the build pipeline:
- `"estimates"` (default) - Generate resource estimates only
- `"rtl"` - Generate RTL code and IP blocks
- `"bitfile"` - Full synthesis to FPGA bitstream

#### board
Target FPGA board. Required when `output` is `"rtl"` or `"bitfile"`.
Common boards include:
- `"Pynq-Z1"`, `"Pynq-Z2"` - Xilinx PYNQ boards
- `"Ultra96"` - Avnet Ultra96
- `"ZCU104"`, `"ZCU102"` - Xilinx ZCU boards
- `"VCK190"` - Xilinx Versal board
- `"V80"` - AMD Versal V80

**CLI Overrides:**
CLI flags `--start-step` and `--stop-step` override blueprint values:
```bash
# Override blueprint to test single step
smith dfc model.onnx blueprint.yaml --start-step streamline --stop-step streamline

# Run from beginning up to a checkpoint
smith dfc model.onnx blueprint.yaml --stop-step specialize_layers
```

**Notes:**
- Steps are identified by name and must match step names in the `steps` list
- For branch points (list of steps), specify any step name within the branch
- Slicing preserves branch structure within the specified range
- Use with `save_intermediate_models: true` to enable checkpointing

### FINN Configuration Overrides

The `finn_config` section allows direct access to FINN's DataflowBuildConfig parameters:
```yaml
finn_config:
  minimize_bit_width: false      # Skip bit-width optimization
  rtlsim_batch_size: 100        # Batch size for RTL simulation
  shell_flow_type: "vivado_zynq" # Shell type for Zynq devices
  generate_outputs: ["estimate_only", "rtlsim_performance"]
```

## Design Space Definition

### Kernels

Kernels define the available hardware implementations for the dataflow graph. The `kernels:` section controls which backends are available during the `specialize_layers` step.

**Two kernel types:**

- **Computational kernels** are pattern-matched from ONNX operations during the `infer_computational_kernels` step (e.g., ONNX MatMul → MVAU, ONNX Softmax → Softmax)
- **Infrastructure kernels** are inserted by topology transforms during the `insert_infrastructure_kernels` step (e.g., DuplicateStreams for tensor fanout, FIFO for buffering)

Both types use the backends you specify in this section.

```yaml
kernels:
  # Use all available backends for a kernel
  - Thresholding                      # Thresholding layers

  # Specify particular backends
  - LayerNorm: LayerNorm_hls          # Use HLS backend only
  - MVAU: [MVAU_hls, MVAU_rtl]        # Use both HLS and RTL backends
```

Backend names must match the full registered name from the backend implementation. For example, if a backend is registered as:
```python
@backend(name="LayerNorm_hls", kernel="LayerNorm", language="hls")
class LayerNorm_hls(LayerNorm, HLSBackend):
    ...
```
Then use `LayerNorm_hls` in the blueprint, not just `hls`.

**Common computational kernels** (pattern-matched during `infer_computational_kernels`):
- `MVAU` - Matrix-Vector-Activation Unit (dense/linear layers)
- `Thresholding` - Quantized activation functions
- `LayerNorm` - Layer normalization
- `Softmax` - Softmax activation
- `ElementwiseBinaryOperation` - Element-wise operations
- `Pool` - Pooling layers

**Common infrastructure kernels** (inserted by topology transforms):
- `DuplicateStreams` - Stream duplication for tensor fanout (inserted by `insert_duplicate_streams` or `infer_dataflow_graph`)
- `StreamingFIFO` - FIFO buffering for timing closure
- `StreamingDataWidthConverter` - Data width conversion for stream width mismatches

### Steps

Steps define the transformation pipeline. They can be:
1. **Linear** - Applied unconditionally
2. **Branching** - Create alternative paths
3. **Optional** - Can be skipped

```yaml
steps:
  # Linear steps - always executed
  - "qonnx_to_finn"                   # Convert QONNX to FINN format
  - "tidy_up"                         # Clean up the model

  # Branching - creates multiple execution paths
  - ["streamline", "streamline_aggressive"]  # Try both approaches

  # Optional steps - creates paths with and without
  - ["minimize_bit_width", ~]         # ~ means skip this step

  # Dataflow graph construction - Option 1: Combined (backward compatible)
  - "build_dataflow_graph"            # Auto-splits kernels, runs both phases

  # Dataflow graph construction - Option 2: Split (finer control)
  # - "insert_infrastructure_kernels" # Phase 1: topology-based (DuplicateStreams, etc.)
  # - "infer_computational_kernels"   # Phase 2: pattern-matching (MVAU, LayerNorm, etc.)

  # Backend specialization
  - "specialize_kernel_backends"      # Partition + select HLS/RTL backends
  # Legacy name (deprecated): "build_hw_graph"

  # Parallelization
  - "apply_folding_config"            # Apply parallelization
  - "generate_estimate_reports"       # Generate resource estimates
```

**Step Validation**: Invalid step names will raise a `ValueError` with helpful suggestions. For example, if you misspell "streamline" as "steamline", you'll get an error suggesting the correct name.

**Branch Point Restrictions**:
- Branch points (lists) can only contain strings or skip indicators
- Nested lists are not allowed within branch points
- To insert a branch point via step operations, use double brackets: `with: [["option1", "option2"]]`

**Skip Value**: The following value indicates a step should be skipped:
- `~`

### Step Operations

Step operations allow you to modify the step list when inheriting from parent blueprints or organizing complex pipelines.


### Inheritance

Blueprint inheritance allows reusing and extending existing configurations.

```yaml
# parent.yaml
name: "Base FINN Pipeline"
clock_ns: 5.0
output: "estimates"

design_space:
  steps:
    - "qonnx_to_finn"
    - "tidy_up"
    - "streamline"

  kernels:
    - MVAU

# child.yaml - extends parent
extends: "parent.yaml"
name: "Extended Pipeline"
output: "rtl"                      # Override parent
board: "Pynq-Z1"                   # Add new field

design_space:
  steps:
    # Parent steps are inherited, then these operations applied:
    - after: "streamline"
      insert: "custom_optimization"
    - at_end:
      insert: "package_ip"

  # Kernels are replaced entirely (no merge)
  kernels:
    - MVAU
    - Thresholding                 # Add new kernel
```

#### Inheritance Rules

1. **Simple fields** (name, clock_ns, etc.) - Child overrides parent
2. **finn_config** - Deep merged (child fields override parent fields)
3. **steps** - Parent steps are inherited. Use step operations to modify them. If child specifies direct steps without operations, parent steps are replaced entirely.
4. **kernels** - Child replaces parent entirely (no merge). Note: If child blueprint has no `kernels` section at all, parent kernels are inherited. An empty `kernels: []` list explicitly clears parent kernels.

## Execution Semantics

### Execution Tree Structure

Brainsmith builds an execution tree from the blueprint where:
- **Nodes** represent execution segments (groups of sequential steps)
- **Branches** occur at variation points (lists in steps)
- **Leaves** represent complete execution paths

### Branch Expansion

Given these steps:
```yaml
steps:
  - "tidy_up"                      # Linear step
  - ["streamline", "streamline_aggressive"]  # Branch point (2 options)
  - "convert_to_hw"                # Linear step
  - ["fold_constants", ~]          # Branch point (2 options)
```

This creates 4 execution paths:
1. `tidy_up → streamline → convert_to_hw → fold_constants`
2. `tidy_up → streamline → convert_to_hw → (skip)`
3. `tidy_up → streamline_aggressive → convert_to_hw → fold_constants`
4. `tidy_up → streamline_aggressive → convert_to_hw → (skip)`

### Segment-Based Execution

To optimize performance, Brainsmith groups sequential steps into segments:
- Steps between branch points form a single segment
- Each segment is executed as one FINN build
- Artifacts are shared at branch points to avoid redundant computation

**Pre-Release Note**: Segment-based execution is functional but requires further testing and refinement for large design spaces.
