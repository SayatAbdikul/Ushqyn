"""Independent hand-derived Phase 6 timeline/placement counterexamples."""
from dataclasses import replace
import itertools

import pytest

from scheduler.contract import Certificate, IllegalSchedule, Reservation
from scheduler.examples import disjoint_dma, quantized_chain
from scheduler.verify import verify


def test_fixed_quantized_chain_preserves_int32_liveness_and_reuses_memory():
    problem, cert = quantized_chain()
    report = verify(problem, cert)
    assert report["makespan_ticks"] == 9
    layout = {r["buffer"]: r for r in report["layout"]}
    assert (layout["acc32"]["first"], layout["acc32"]["last"], layout["acc32"]["size"]) == (0, 6, 32)
    assert (layout["q8"]["first"], layout["q8"]["last"]) == (4, 9)
    assert cert.offsets["output"] == cert.offsets["acc32"]


def test_disjoint_dma_can_overlap_only_when_actual_port_slots_do_not():
    problem, cert = disjoint_dma()
    assert verify(problem, cert)["makespan_ticks"] == 6
    cert = replace(cert, starts={"compute": 0, "load": 1})
    with pytest.raises(IllegalSchedule, match="resource conflict: sram_port"):
        verify(problem, cert)


def test_exhaustive_small_timeline_matches_hand_derived_conditions():
    # No generator or cost helper supplies the expected answer. On this fixture
    # the two SRAM access windows [c,c+2) and [d,d+2) must not intersect.
    problem, cert = disjoint_dma()
    for compute, dma in itertools.product(range(5), repeat=2):
        proposed = replace(cert, starts={"compute": compute, "load": dma})
        expected = compute + 2 <= dma or dma + 2 <= compute
        if expected:
            assert verify(problem, proposed)["makespan_ticks"] == max(compute + 6, dma + 2)
        else:
            with pytest.raises(IllegalSchedule, match="resource conflict"):
                verify(problem, proposed)


@pytest.mark.parametrize("start", [-1, 1.5, True])
def test_invalid_start_rejected(start):
    problem, cert = quantized_chain()
    with pytest.raises(IllegalSchedule, match="start time"):
        verify(problem, replace(cert, starts={**cert.starts, "mac": start}))


@pytest.mark.parametrize("offset", [-8, 1, 64, True])
def test_invalid_placement_rejected(offset):
    problem, cert = quantized_chain()
    with pytest.raises(IllegalSchedule, match="placement"):
        verify(problem, replace(cert, offsets={**cert.offsets, "input": offset}))


def test_int32_storage_cannot_be_reused_before_quantization_completes():
    problem, cert = quantized_chain()
    with pytest.raises(IllegalSchedule, match="live buffers overlap"):
        verify(problem, replace(cert, offsets={**cert.offsets, "q8": 16}))


def test_disjoint_resource_engines_do_not_waive_memory_ownership():
    problem, cert = disjoint_dma()
    with pytest.raises(IllegalSchedule, match="live buffers overlap"):
        verify(problem, replace(cert, offsets={**cert.offsets, "prefetch": 8}))


def test_missing_requantization_task_rejected():
    problem, cert = quantized_chain()
    with pytest.raises(IllegalSchedule, match="missing or extra scheduled task"):
        verify(problem, replace(cert, starts={"mac": 0, "next": 4}))


def test_changed_arithmetic_contract_rejects_old_certificate():
    problem, cert = quantized_chain()
    tasks = list(problem.tasks)
    tasks[1] = replace(tasks[1], semantic_id="different rounding or scale")
    with pytest.raises(IllegalSchedule, match="quantization identity"):
        verify(replace(problem, tasks=tuple(tasks)), cert)


def test_read_before_quantized_output_ready_rejected():
    problem, cert = quantized_chain()
    with pytest.raises(IllegalSchedule, match="read before producer completion: q8"):
        verify(problem, replace(cert, starts={**cert.starts, "next": 5}))


def test_dependency_cycle_rejected_even_with_matching_hash():
    problem, cert = quantized_chain()
    tasks = list(problem.tasks)
    tasks[0] = replace(tasks[0], reads=("q8",))
    problem = replace(problem, tasks=tuple(tasks))
    with pytest.raises(IllegalSchedule, match="dependency cycle"):
        verify(problem, replace(cert, problem_sha256=problem.digest()))


def test_multiple_writers_rejected():
    problem, cert = quantized_chain()
    tasks = list(problem.tasks)
    tasks[2] = replace(tasks[2], writes=("acc32", "output"))
    problem = replace(problem, tasks=tuple(tasks))
    with pytest.raises(IllegalSchedule, match="multiple/initial writer"):
        verify(problem, replace(cert, problem_sha256=problem.digest()))


def test_missing_producer_rejected():
    problem, cert = quantized_chain()
    buffers = list(problem.buffers)
    buffers[0] = replace(buffers[0], initial=False)
    problem = replace(problem, buffers=tuple(buffers))
    with pytest.raises(IllegalSchedule, match="no producer"):
        verify(problem, replace(cert, problem_sha256=problem.digest()))


def test_reservation_outside_task_rejected():
    problem, cert = disjoint_dma()
    tasks = list(problem.tasks)
    tasks[1] = replace(tasks[1], reservations=(Reservation("dma", 0, 3),))
    problem = replace(problem, tasks=tuple(tasks))
    with pytest.raises(IllegalSchedule, match="invalid reservation"):
        verify(problem, replace(cert, problem_sha256=problem.digest()))


def test_unused_retained_input_cannot_be_overwritten():
    problem, cert = quantized_chain()
    buffers = list(problem.buffers)
    buffers[0] = replace(buffers[0], retain=True)
    problem = replace(problem, buffers=tuple(buffers))
    with pytest.raises(IllegalSchedule, match="live buffers overlap"):
        verify(problem, replace(cert, problem_sha256=problem.digest()))
