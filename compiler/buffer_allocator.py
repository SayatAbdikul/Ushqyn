"""Liveness-based linear-scan buffer allocator.

Pre-P4 (2026-05-09 era), `compile.py` used hand-rolled ping-pong:

    mat_buf            alternated 1 ↔ 2     (FC / Conv weights, LOAD_M)
    bias_vector_buf    alternated 3 ↔ 4     (biases, LOAD_V)
    gemv_buf           alternated 5 ↔ 6     (GEMV outputs)
    relu_buf           alternated 7 ↔ 8     (RELU outputs)
    conv_out_buf       alternated 10 ↔ 11   (CONV2D_RUN outputs)
    pool_out_buf       alternated 12 ↔ 13   (MAXPOOL outputs)

This works for strictly linear graphs (every tensor consumed by exactly
one downstream op) but breaks the moment three same-class tensors are
live simultaneously — residual connections, branching, skip-connections.
The double-buffering silently overwrites a still-live tensor.

(It also masked a separate latent bug: the matrix-buffer ping-pong used
IDs {1, 2} even though MATRIX_BUFFER_COUNT == 2, so RTL silently truncated
the buffer-ID field's high bit and aliased ID 2 → physical buffer 0. This
was harmless for double-buffering but is exactly the kind of "works by
coincidence" pattern P4 removes.)

Algorithm — liveness-based linear scan:

  1. Compute the topological order of nodes.
  2. Compute `last_use_idx` for every tensor name: the topo index of the
     last node that consumes it. Aliasing through Reshape / Flatten /
     BatchNormalization is propagated so a buffer stays live as long as
     ANY name in its alias chain is still being consumed.
  3. Walk topologically. At each step:
        a. Free buffer IDs whose tensor's `last_use_idx < current_idx`.
        b. For each new tensor introduced by the current node (initializer
           inputs, op outputs), allocate a free buffer ID from its class
           pool — vector for activations / biases, matrix for weights.

API surface used by compile.py:

    assignments = allocate_buffers(model, ordered_nodes,
                                   vector_count=16, matrix_count=2)

`assignments` is a {tensor_name: buffer_id} dict. Aliased tensors share
an ID. compile.py reads from this dict and never maintains its own
counter; this is the lock-in for the structural fix.
"""
from collections import defaultdict
import bisect
import numpy as np
from onnx import numpy_helper


# Tensor "buffer class" — vector buffers hold int8 vectors (inputs, biases,
# activations, FC outputs); matrix buffers hold int8 matrices (weights).
VECTOR = "vector"
MATRIX = "matrix"

# Convention: the model's primary input always lands in vector buffer 9.
# Some downstream code (cocotb test harnesses, `train_and_eval_cnn.py`'s
# manual `LOAD_V` prolog) hard-codes buffer 9 for the input. Reserving it
# here keeps that contract.
PRIMARY_INPUT_BUFFER = 9

# Matrix buffer IDs start at this offset to keep them disjoint from vector
# IDs in `golden_model`'s unified `buffers` dict. The RTL has separate
# matrix and vector buffer files with non-overlapping address spaces, so
# colliding IDs are fine in hardware (the buffer-ID field is wider than
# the per-pool index width and the upper bits truncate). The golden model,
# though, is a single namespace where two buffers with the same ID
# overwrite each other. This was the silent reason the pre-P4 ping-pong
# used IDs {1, 2} for matrices and {3, 4, …} for vectors — disjoint by
# accident. Codifying the disjointness here is correctness, not policy.
MATRIX_ID_OFFSET = 16

# Passthrough ops emit no instruction; their output aliases their input.
PASSTHROUGH_OPS = frozenset({"Reshape", "Flatten", "BatchNormalization"})


class BufferAllocator:
    """Linear-scan allocator over the accelerator's vector and matrix buffer pools.

    Caller provides each tensor's `last_use_idx` from a liveness pre-pass;
    `free_dead(idx)` releases buffers whose tensor died before `idx`.

    Per-class pools:
        vector  — IDs in [0, vector_count) minus PRIMARY_INPUT_BUFFER
        matrix  — IDs in [0, matrix_count)
    """

    def __init__(self, *, vector_count=16, matrix_count=2,
                 reserved_vector_ids=(PRIMARY_INPUT_BUFFER,)):
        if PRIMARY_INPUT_BUFFER >= vector_count:
            raise ValueError(
                f"PRIMARY_INPUT_BUFFER={PRIMARY_INPUT_BUFFER} exceeds "
                f"vector_count={vector_count}; no slot for the model input"
            )
        # Vector pool: 0..vector_count-1, minus reserved IDs (9 = primary input).
        # Matrix pool: MATRIX_ID_OFFSET..MATRIX_ID_OFFSET+matrix_count-1, kept
        # disjoint from the vector range so golden_model's unified buffer
        # dict never collides (see MATRIX_ID_OFFSET docstring).
        if MATRIX_ID_OFFSET < vector_count:
            raise ValueError(
                f"MATRIX_ID_OFFSET={MATRIX_ID_OFFSET} overlaps vector pool "
                f"[0, {vector_count})"
            )
        self._vec_free = sorted(set(range(vector_count)) - set(reserved_vector_ids))
        self._mat_free = list(range(MATRIX_ID_OFFSET, MATRIX_ID_OFFSET + matrix_count))
        self._vec_in_use = {}  # buffer_id → (tensor_name, last_use_idx)
        self._mat_in_use = {}
        self.assignments = {}  # tensor_name → buffer_id
        self.tensor_class = {}  # tensor_name → VECTOR or MATRIX

    # ── reservation / allocation ────────────────────────────────────────────

    def reserve_input(self, name: str, last_use_idx: int) -> int:
        """Pin the primary input tensor to PRIMARY_INPUT_BUFFER. Must be
        called before any other allocations so the reserved slot is never
        offered to a different tensor."""
        bid = PRIMARY_INPUT_BUFFER
        self._vec_in_use[bid] = (name, last_use_idx)
        self.assignments[name] = bid
        self.tensor_class[name] = VECTOR
        return bid

    def allocate(self, name: str, klass: str, last_use_idx: int) -> int:
        if klass == VECTOR:
            pool, in_use = self._vec_free, self._vec_in_use
        elif klass == MATRIX:
            pool, in_use = self._mat_free, self._mat_in_use
        else:
            raise ValueError(f"unknown buffer class: {klass!r}")

        if not pool:
            raise RuntimeError(
                f"out of {klass} buffers when allocating {name!r}; "
                f"live={list(in_use.values())}"
            )
        bid = pool.pop(0)
        in_use[bid] = (name, last_use_idx)
        self.assignments[name] = bid
        self.tensor_class[name] = klass
        return bid

    def alias(self, alias_name: str, source_name: str):
        """Map `alias_name` to whatever buffer `source_name` was assigned.
        Used for Reshape / Flatten / BatchNormalization passthroughs — no
        new buffer is allocated, but liveness was already extended in the
        pre-pass so the underlying buffer stays held until the alias's
        last consumer."""
        if source_name not in self.assignments:
            raise KeyError(
                f"cannot alias {alias_name!r} to {source_name!r}: "
                f"source has no assignment yet"
            )
        bid = self.assignments[source_name]
        self.assignments[alias_name] = bid
        self.tensor_class[alias_name] = self.tensor_class[source_name]
        return bid

    def free_dead(self, current_idx: int):
        """Release any buffer whose tensor's last_use_idx < current_idx.

        Frees both pools at once so allocators that switch class within
        one node don't see stale "in_use" entries from the previous node.
        """
        for in_use, free_pool in (
            (self._vec_in_use, self._vec_free),
            (self._mat_in_use, self._mat_free),
        ):
            dead = [bid for bid, (_, lu) in in_use.items() if lu < current_idx]
            for bid in dead:
                del in_use[bid]
                bisect.insort(free_pool, bid)


# ── liveness ────────────────────────────────────────────────────────────────

def compute_last_use(ordered_nodes, output_names, *, extra_aliases=None):
    """Compute `last_use_idx` for every tensor name referenced in the graph.

    Args:
        ordered_nodes:    nodes in topological order.
        output_names:     set of graph-output tensor names (live to program end).
        extra_aliases:    optional {alias_name: source_name} dict for in-place
                          ops that share a physical buffer with their input
                          (e.g. fused Relu on a CONV2D_RUN). Each alias's
                          last_use is folded into its source's last_use so
                          the underlying buffer stays held until the alias
                          itself is no longer consumed.

    Returns: dict mapping tensor name → topo idx of last consumer.
    Tensors that are graph outputs receive `last_use = len(ordered_nodes)`.
    """
    n_nodes = len(ordered_nodes)
    last_use: dict = {}

    # Forward pass: every input edge contributes a use.
    for idx, node in enumerate(ordered_nodes):
        for inp in node.input:
            if inp:
                last_use[inp] = max(last_use.get(inp, -1), idx)

    # Graph outputs are live to end of program.
    for name in output_names:
        last_use[name] = max(last_use.get(name, -1), n_nodes)

    # Reverse pass: a passthrough's output's last_use extends back to its
    # input's last_use, since they share a physical buffer. Walk reverse so
    # cascades (Reshape → Flatten → ...) propagate through the whole chain.
    for idx in range(n_nodes - 1, -1, -1):
        node = ordered_nodes[idx]
        if node.op_type in PASSTHROUGH_OPS and node.input and node.output:
            inp, out = node.input[0], node.output[0]
            if out in last_use:
                last_use[inp] = max(last_use.get(inp, -1), last_use[out])

    # Extra aliases (fused Relu, etc.): same propagation, but op-type-blind.
    if extra_aliases:
        for alias, source in extra_aliases.items():
            if alias in last_use:
                last_use[source] = max(last_use.get(source, -1),
                                       last_use[alias])

    return last_use


# ── top-level driver ────────────────────────────────────────────────────────

def _initializer_class(init):
    """An ONNX initializer is a matrix (weight) if it has 2+ dims; else a
    vector (bias / scalar). Mirrors the dram.save_all_initializers_to_dram
    classification."""
    arr = numpy_helper.to_array(init)
    return MATRIX if arr.ndim >= 2 else VECTOR


def allocate_buffers(graph, ordered_nodes, *,
                     vector_count=16, matrix_count=2,
                     extra_aliases=None):
    """Compute buffer-ID assignments for every tensor referenced in the graph.

    Args:
        graph:           onnx.GraphProto (the .graph attribute of the model).
        ordered_nodes:   nodes in topological order.
        vector_count:    physical vector-buffer count in the RTL.
        matrix_count:    physical matrix-buffer count in the RTL.
        extra_aliases:   optional {alias_name: source_name} for in-place ops
                         that share a buffer with their input (e.g. fused
                         Relu on a CONV2D_RUN, "big" standalone Relus that
                         exceed the 10-bit length field). Liveness for these
                         is propagated and the alias is materialised at its
                         producer node (no new buffer is allocated).

    Returns:
        dict {tensor_name: buffer_id}. Aliased tensors share an ID.

    Raises RuntimeError if the graph needs more concurrent buffers than the
    pools can supply — the failure is loud rather than silent corruption,
    which was the whole point of P4.
    """
    extra_aliases = extra_aliases or {}
    initializer_data = {init.name: init for init in graph.initializer}
    initializer_class = {
        name: _initializer_class(init) for name, init in initializer_data.items()
    }
    output_names = {o.name for o in graph.output}
    n_nodes = len(ordered_nodes)
    last_use = compute_last_use(ordered_nodes, output_names,
                                extra_aliases=extra_aliases)

    alloc = BufferAllocator(vector_count=vector_count, matrix_count=matrix_count)

    # Reserve primary input first — its slot (PRIMARY_INPUT_BUFFER) is held
    # by convention and downstream code hard-codes it.
    if graph.input:
        primary = graph.input[0].name
        alloc.reserve_input(primary, last_use.get(primary, n_nodes))

    for idx, node in enumerate(ordered_nodes):
        # 1. Release dead tensors before this node's allocations so freed
        #    IDs are available for re-use immediately.
        alloc.free_dead(idx)

        # 2. Allocate buffers for this node's initializer inputs (weights /
        #    biases). An initializer is consumed only by this node, so its
        #    last_use is exactly idx — it'll be freed at idx+1.
        for inp_name in node.input:
            if (inp_name in initializer_data
                    and inp_name not in alloc.assignments):
                klass = initializer_class[inp_name]
                alloc.allocate(inp_name, klass, idx)

        # 3. Op-type passthroughs (Reshape / Flatten / BatchNorm): alias
        #    output to input; emit nothing.
        if node.op_type in PASSTHROUGH_OPS:
            if node.input and node.output:
                src = node.input[0]
                dst = node.output[0]
                if src in alloc.assignments and dst not in alloc.assignments:
                    alloc.alias(dst, src)
            continue

        # 4. Compute nodes: allocate output buffers (vector class — every
        #    op output in this ISA is a flat int8 vector). Outputs flagged
        #    as extra-aliases (fused Relu, etc.) get aliased instead.
        for out_name in node.output:
            if not out_name or out_name in alloc.assignments:
                continue
            if out_name in extra_aliases:
                source = extra_aliases[out_name]
                if source in alloc.assignments:
                    alloc.alias(out_name, source)
                    continue
                # Source not yet allocated — this means the alias was
                # specified out of topo order; fall through to allocate
                # normally and rely on the dram walker / emitter to handle.
            lu = last_use.get(out_name, idx)
            alloc.allocate(out_name, VECTOR, lu)

    return alloc.assignments
