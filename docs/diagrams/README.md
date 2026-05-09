# RTL Architecture Diagrams

This directory contains comprehensive architecture diagrams for the Ushqyn RTL design.

## Diagram Files

### Source Files (DOT format)
- `system_architecture.dot` - Top-level system overview
- `module_hierarchy.dot` - Complete module hierarchy tree
- `execution_unit.dot` - Execution unit detailed architecture
- `gemv_pipeline.dot` - GEMV computation pipeline
- `memory_system.dot` - Memory subsystem and map
- `fsm_states.dot` - FSM state diagrams for all control units

### Generated Files (PNG format)
Run `./generate_diagrams.sh` to create:
- `system_architecture.png`
- `module_hierarchy.png`
- `execution_unit.png`
- `gemv_pipeline.png`
- `memory_system.png`
- `fsm_states.png`

## Prerequisites

Install Graphviz:
```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
```

## Usage

### Generate all diagrams:
```bash
./generate_diagrams.sh
```

### Generate individual diagram:
```bash
# PNG format
dot -Tpng system_architecture.dot -o system_architecture.png

# SVG format (scalable)
dot -Tsvg system_architecture.dot -o system_architecture.svg

# PDF format
dot -Tpdf system_architecture.dot -o system_architecture.pdf
```

## Diagram Descriptions

### 1. System Architecture
Shows the top-level structure with:
- Fetch Unit → Decoder → Execution Unit flow
- 4 separate memory instances
- Control signals (start/done)
- FSM states

### 2. Module Hierarchy
Complete tree showing all modules from top to bottom:
- tinyml_accelerator_top (220 lines)
  - fetch_unit, i_decoder
  - modular_execution_unit (482 lines)
    - buffer_controller, load/gemv/relu/store execution
    - 32 PEs, quantization pipeline
    - wallace_32x32, compressor_3to2

### 3. Execution Unit
Detailed view of modular_execution_unit:
- Main FSM controller
- Buffer management (vector & matrix)
- Operation modules (Load, GEMV, ReLU, Store)
- Data flow between buffers and operations

### 4. GEMV Pipeline
GEMV computation architecture:
- Processing Element array (32 PEs)
- Accumulation tree
- Bias addition
- 3-stage quantization (find max → compute scale → quantize)
- FSM control flow

### 5. Memory System
Memory subsystem details:
- 4 separate simple_memory instances
- Memory map (instructions, inputs, weights, biases, outputs)
- Access patterns (fetch, load, store)
- Synchronization requirements

### 6. FSM States
State machine diagrams for:
- Top-level FSM (7 states: IDLE → FETCH → DECODE → EXECUTE → DONE)
- Execution unit FSM (7 states: IDLE → DISPATCH → WAIT_* → COMPLETE)
- GEMV FSM (9 states: IDLE → process → QUANTIZE → DONE)

## Abstraction Levels

The diagrams show 4 levels of abstraction:

**Level 1: System**
- Input → Accelerator → Output
- Instructions (LOAD/GEMV/RELU/STORE)

**Level 2: Subsystem**
- Fetch & Decode → Execution Unit → Memory
- Buffers ↔ Compute

**Level 3: Module**
- Individual modules (fetch_unit, load_v, top_gemv, etc.)
- Buffer controllers, execution units

**Level 4: Component**
- PEs, multipliers, quantizers
- FSMs, registers, memory arrays

## Key Features Illustrated

1. **Modular Design**: Clear separation of concerns
2. **Hierarchical FSMs**: Multi-level control
3. **Memory System**: 4 separate instances requiring sync
4. **Tiled Computation**: Efficient memory bandwidth usage
5. **Pipelined Quantization**: Multi-stage processing
6. **Buffer Management**: Dual buffer files with tile access

## Documentation

For detailed module descriptions, signal flows, and implementation details, see:
- `../RTL_ARCHITECTURE.md` - Comprehensive architecture documentation
- `../../rtl/` - RTL source code
- `../../test/` - Testbenches and verification

## Notes

- All diagrams use consistent color coding:
  - Light blue: Top-level/control
  - Light yellow: Decode/FSM
  - Light pink: Execution/computation
  - Light green: I/O/completion
  - Light gray: Memory/storage
  - Orange: Critical paths/warnings

- DOT files are plain text and can be edited with any text editor
- Graphviz supports many output formats: PNG, SVG, PDF, PS, etc.
- SVG format is recommended for presentations (scalable, crisp)

## Revision History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | Dec 24, 2025 | Initial diagram suite creation |
