"""Independent exhaustive oracles for the two declared Phase 6 search spaces."""
from dataclasses import replace
from itertools import product
import random

import pytest

from scheduler.contract import Buffer, Memory, Problem, Reservation, Resource, Task, IllegalSchedule
from scheduler.search import exact_search
from scheduler.catalogue import Catalogue, Choice, evaluate, search
from scheduler.verify import verify


def tiny_problem(capacity, shared_port, retain):
    return Problem((Memory('ram', capacity),), (Resource('port', 1), Resource('other', 1)),
                   (Buffer('a', 'ram', 2, retain=retain), Buffer('b', 'ram', 2)),
                   (Task('a', 'compute', 2, (), ('a',), 'a', (Reservation('port', 0, 1),)),
                    Task('b', 'dma', 2, (), ('b',), 'b',
                         (Reservation('port' if shared_port else 'other', 0, 1),))))


@pytest.mark.parametrize('capacity,shared,retain', product((2, 3, 4), (False, True), (False, True)))
def test_exact_timing_and_addresses_against_independent_enumeration(capacity, shared, retain):
    # This brute oracle uses neither the verifier nor the search helpers. Every
    # address and timeline is enumerated; a retained A lives until both finish.
    expected = []
    for a, b, pa, pb in product(range(3), range(3), range(capacity-1), range(capacity-1)):
        finish = max(a+2, b+2)
        simultaneous = max(a, b) < min(finish if retain else a+2, b+2)
        collision = max(pa, pb) < min(pa+2, pb+2)
        if not (shared and a == b) and not (simultaneous and collision):
            expected.append(finish)
    result = exact_search(tiny_problem(capacity, shared, retain), 4)
    assert result['objective'] == min(expected)
    assert result['optimal'] and not result['infeasible_within_horizon']
    assert verify(tiny_problem(capacity, shared, retain), result['certificate'])['makespan_ticks'] == min(expected)


def test_budget_retains_verified_fallback_without_false_optimality():
    problem = tiny_problem(4, False, False)
    result = exact_search(problem, 4, max_expansions=3)
    assert result['objective'] == 4
    assert not result['optimal'] and not result['infeasible_within_horizon']
    assert result['stop_reason'] == 'expansion_budget'
    assert exact_search(problem, 4, max_expansions=1)['certificate'] is None


def test_horizon_and_memory_infeasibility_are_separate_from_budget_failure():
    result = exact_search(tiny_problem(2, True, False), 2)
    assert result['infeasible_within_horizon'] and result['objective'] is None
    p = tiny_problem(2, False, True)
    # Both retained buffers coexist at the finish, at every possible placement.
    p = replace(p, buffers=tuple(replace(b, retain=True) for b in p.buffers))
    assert exact_search(p, 4)['infeasible_within_horizon']


def brute_paths(catalogue):
    # No dynamic programming, dominance, bound or evaluator from the solver.
    def walk(position, state, cost, path):
        if position == catalogue.length:
            return [(cost, path)] if state in catalogue.final_states else []
        results = []
        for c in catalogue.choices:
            if c.start == position and c.input_state == state:
                results += walk(c.stop, c.output_state, cost+c.cost, path+(c.name,))
        return results
    return sorted(walk(0, catalogue.initial_state, 0, ()))


def random_catalogue(seed):
    rng = random.Random(seed)
    choices = []
    for start in range(4):
        for stop in range(start+1, min(4, start+2)+1):
            for incoming, outgoing in product(('materialized', 'cached'), repeat=2):
                if rng.random() < .7:
                    choices.append(Choice(str(len(choices)), start, stop, incoming, outgoing, rng.randrange(20)))
    return Catalogue(4, tuple(choices))


@pytest.mark.parametrize('seed', range(30))
def test_exact_catalogue_and_beam_bounds_against_independent_paths(seed):
    catalogue = random_catalogue(seed)
    expected = brute_paths(catalogue)
    exact = search(catalogue)
    if not expected:
        assert exact['infeasible'] and exact['path'] is None
        return
    assert (exact['objective'], tuple(exact['path'])) == expected[0]
    assert exact['optimal_within_catalogue'] and exact['bound_gap_fraction'] == 0
    beam = search(catalogue, beam_width=1, fallback=expected[-1][1])
    assert beam['lower_bound'] <= expected[0][0] <= beam['objective']
    assert evaluate(catalogue, beam['path']) == beam['objective']
    if beam['optimal_within_catalogue']:
        assert beam['objective'] == expected[0][0]


def test_cheaper_prefix_cannot_dominate_a_different_live_frontier():
    c = Catalogue(2, (Choice('cheap', 0, 1, 'materialized', 'dead', 0),
                      Choice('live', 0, 1, 'materialized', 'useful', 2),
                      Choice('finish', 1, 2, 'useful', 'materialized', 3)))
    assert search(c)['objective'] == 5
    limited = search(c, beam_width=1, fallback=('live', 'finish'))
    assert limited['objective'] == 5 and limited['pruned_states'] == 1
    assert not limited['optimal_within_catalogue']


def test_deterministic_budget_and_feasible_fallback():
    c = random_catalogue(5)
    fallback = brute_paths(c)[-1][1]
    one = search(c, max_expansions=1, fallback=fallback)
    two = search(c, max_expansions=1, fallback=fallback)
    assert {k:v for k,v in one.items() if k != 'wall_seconds'} == {k:v for k,v in two.items() if k != 'wall_seconds'}
    assert one['stop_reason'] == 'expansion_budget' and not one['infeasible']
    with pytest.raises(IllegalSchedule, match='incomplete'):
        search(c, fallback=[])


def test_timeout_never_proves_infeasibility(monkeypatch):
    import scheduler.catalogue as module
    ticks = iter([0, 2, 3])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    r = search(Catalogue(1, (Choice('one', 0, 1, 'materialized', 'materialized', 2),)), seconds=1)
    assert r['stop_reason'] == 'wall_timeout' and not r['infeasible']
    assert not r['deterministic_expansion_budget']
