"""Tests for the liveness-based buffer allocator (P4, 2026-05-09).

Covers the parts the existing end-to-end tests don't catch directly:

  - liveness propagation through Reshape/Flatten/BatchNorm passthroughs
  - liveness propagation through fused-Relu / passthrough-Relu aliases
  - vector vs matrix pool disjointness (matrix IDs >= MATRIX_ID_OFFSET)
  - PRIMARY_INPUT_BUFFER reservation
  - allocator failure mode: out-of-buffers raises (loud failure, not silent)
  - branching graph: three same-class tensors live simultaneously
    (the exact scenario that broke the pre-P4 ping-pong)

The full SmallCNN end-to-end golden contract is exercised separately by
test_cnn_golden.py; the heavy MNIST cocotb test exercises end-to-end RTL.
This file isolates the allocator logic so a regression here surfaces
immediately rather than as a downstream byte mismatch.
"""
import pytest

import buffer_allocator
from buffer_allocator import (
    BufferAllocator,
    PRIMARY_INPUT_BUFFER,
    MATRIX_ID_OFFSET,
    VECTOR,
    MATRIX,
    compute_last_use,
    PASSTHROUGH_OPS,
)


# ── Stub graph machinery ─────────────────────────────────────────────────────

class _StubNode:
    """Minimal stand-in for an onnx.NodeProto. We don't pull onnx into this
    test file because the allocator only reads .op_type / .input / .output."""
    def __init__(self, op_type, input, output):
        self.op_type = op_type
        self.input = list(input)
        self.output = list(output)


# ── BufferAllocator unit tests ──────────────────────────────────────────────

def test_primary_input_reserved():
    a = BufferAllocator()
    a.reserve_input("x", last_use_idx=10)
    assert a.assignments["x"] == PRIMARY_INPUT_BUFFER
    # The reserved ID is no longer available for general allocation.
    a.allocate("y", VECTOR, last_use_idx=1)
    assert a.assignments["y"] != PRIMARY_INPUT_BUFFER


def test_matrix_ids_are_disjoint_from_vector_ids():
    """The ID spaces must not overlap — collisions corrupt golden_model's
    unified buffer dict (the silent reason pre-P4 used IDs {1,2} for
    matrix and {3,…} for vector)."""
    a = BufferAllocator()
    v_id = a.allocate("v", VECTOR, last_use_idx=1)
    m_id = a.allocate("m", MATRIX, last_use_idx=1)
    assert v_id < MATRIX_ID_OFFSET <= m_id
    assert v_id != m_id


def test_free_dead_returns_id_to_pool():
    a = BufferAllocator()
    bid = a.allocate("temp", VECTOR, last_use_idx=2)
    a.free_dead(current_idx=3)  # tensor's last_use=2 < 3 → freed
    bid2 = a.allocate("next", VECTOR, last_use_idx=5)
    assert bid2 == bid  # got back the same low-ID slot


def test_free_dead_keeps_live_buffer():
    a = BufferAllocator()
    a.allocate("alive", VECTOR, last_use_idx=10)
    free_before = list(a._vec_free)
    a.free_dead(current_idx=5)  # last_use=10 > 5 → still alive
    assert list(a._vec_free) == free_before


def test_alias_does_not_consume_pool_slot():
    a = BufferAllocator()
    a.allocate("source", VECTOR, last_use_idx=10)
    free_before = list(a._vec_free)
    a.alias("alias", "source")
    assert a.assignments["alias"] == a.assignments["source"]
    assert list(a._vec_free) == free_before


def test_out_of_buffers_raises_loud():
    """The whole point of P4: surface the "too many live tensors" failure
    rather than silently aliasing onto an in-use buffer."""
    # Default vector_count=16; reserve nothing so all 16 slots are free,
    # then drain them and assert the next allocation raises.
    a = BufferAllocator(reserved_vector_ids=())
    for i in range(16):
        a.allocate(f"v{i}", VECTOR, last_use_idx=100)
    with pytest.raises(RuntimeError, match="out of vector buffers"):
        a.allocate("v_overflow", VECTOR, last_use_idx=100)


def test_out_of_matrix_buffers_raises_loud():
    a = BufferAllocator(matrix_count=1)
    a.allocate("w0", MATRIX, last_use_idx=10)
    with pytest.raises(RuntimeError, match="out of matrix buffers"):
        a.allocate("w1", MATRIX, last_use_idx=10)


# ── Liveness with passthrough alias chains ───────────────────────────────────

def test_compute_last_use_propagates_through_passthrough():
    """A buffer must stay live through a Reshape chain so a downstream
    consumer of the reshape's output doesn't read garbage."""
    nodes = [
        _StubNode("Conv", ["x"], ["c"]),       # idx 0
        _StubNode("Reshape", ["c"], ["r"]),    # idx 1 — passthrough
        _StubNode("Gemm", ["r", "w", "b"], ["y"]),  # idx 2 — uses reshape output
    ]
    last_use = compute_last_use(nodes, output_names={"y"})
    # Without passthrough propagation, last_use["c"] would be 1 (the
    # Reshape). With propagation it extends to 2 (the Gemm).
    assert last_use["c"] == 2
    assert last_use["r"] == 2


def test_compute_last_use_propagates_through_fused_relu():
    """fused-Relu alias: the Conv output's last_use must extend through the
    Relu's consumers, even though there's no PASSTHROUGH op_type involved."""
    nodes = [
        _StubNode("Conv", ["x"], ["c"]),       # idx 0
        _StubNode("Relu", ["c"], ["r"]),       # idx 1 — fused, alias r→c
        _StubNode("MaxPool", ["r"], ["y"]),    # idx 2
    ]
    last_use = compute_last_use(
        nodes, output_names={"y"}, extra_aliases={"r": "c"}
    )
    assert last_use["c"] == 2  # extended through r→y consumption
    assert last_use["r"] == 2


def test_passthrough_set_includes_expected_ops():
    """Defensive: if anyone narrows PASSTHROUGH_OPS the allocator silently
    starts allocating new buffers for what should be aliases."""
    assert "Reshape" in PASSTHROUGH_OPS
    assert "Flatten" in PASSTHROUGH_OPS
    assert "BatchNormalization" in PASSTHROUGH_OPS


# ── Branching graph (three same-class tensors live simultaneously) ──────────

def test_three_concurrent_vector_tensors():
    """Pre-P4 ping-pong only had two slots per class. A graph that keeps
    three same-class tensors alive at once would silently overwrite. The
    allocator must instead pick three distinct IDs."""
    a = BufferAllocator(reserved_vector_ids=())  # no reservation, all free
    ids = []
    for i, name in enumerate(["t0", "t1", "t2"]):
        ids.append(a.allocate(name, VECTOR, last_use_idx=10))
    assert len(set(ids)) == 3, f"three live tensors collided: {ids}"


def test_residual_pattern_assigns_distinct_ids():
    """ResNet-style: a tensor is consumed by both an immediately-following
    op AND a much later add. Pre-P4 ping-pong overwrites the early branch.
    The allocator must keep the residual alive across the whole branch."""
    nodes = [
        _StubNode("Conv", ["x"], ["a"]),     # 0: produces a
        _StubNode("Conv", ["a"], ["b"]),     # 1: produces b  (consumes a)
        _StubNode("Conv", ["b"], ["c"]),     # 2: produces c
        # 3: residual add — consumes both `a` (from step 0) and `c`
        _StubNode("Add",  ["a", "c"], ["y"]),
    ]
    last_use = compute_last_use(nodes, output_names={"y"})
    assert last_use["a"] == 3  # alive across b, c
    assert last_use["b"] == 2
    assert last_use["c"] == 3

    # Build a tiny graph stub for allocate_buffers
    class _GraphStub:
        def __init__(self):
            self.input = []
            self.output = [type("", (), {"name": "y"})()]
            self.initializer = []
    a = buffer_allocator.allocate_buffers(_GraphStub(), nodes)
    assert a["a"] != a["b"]  # both alive at step 1
    assert a["a"] != a["c"]  # both alive at step 3
    assert a["b"] != a["c"]  # both alive at step 2


# ── Integration with allocate_buffers ────────────────────────────────────────

def test_passthrough_aliases_share_buffer_id():
    """A Flatten chain through three names must end up with all three
    sharing a single physical buffer ID."""
    nodes = [
        _StubNode("Conv", ["x"], ["c"]),
        _StubNode("Reshape", ["c"], ["r"]),
        _StubNode("Flatten", ["r"], ["f"]),
        _StubNode("Gemm", ["f"], ["y"]),
    ]
    class _GraphStub:
        def __init__(self):
            self.input = []
            self.output = [type("", (), {"name": "y"})()]
            self.initializer = []
    assignments = buffer_allocator.allocate_buffers(_GraphStub(), nodes)
    assert assignments["c"] == assignments["r"] == assignments["f"]
