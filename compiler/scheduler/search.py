"""Bounded exact placement/timing search for small fixed-task problems.

No state is merged using a scalar memory estimate. Every integer start time
and aligned address in the declared horizon is retained unless a dependency,
resource or exact live-range constraint proves it impossible. Exhausting a
budget is never reported as optimality or infeasibility.
"""
from dataclasses import dataclass
import math
import time

from .contract import Certificate, IllegalSchedule, integer, require, validate_problem
from .verify import verify


class BudgetExhausted(Exception):
    pass


@dataclass
class Budget:
    max_expansions: int = 100000
    seconds: float = 60.0
    expansions: int = 0
    reason: str | None = None

    def __post_init__(self):
        require(integer(self.max_expansions, 1) and math.isfinite(self.seconds) and self.seconds > 0,
                "invalid search budget")
        self.deadline = time.monotonic() + self.seconds

    def step(self):
        if self.expansions >= self.max_expansions:
            self.reason = "expansion_budget"
        elif time.monotonic() >= self.deadline:
            self.reason = "wall_timeout"
        if self.reason:
            raise BudgetExhausted
        self.expansions += 1


def dependency_order(problem):
    _, _, _, tasks, producers = validate_problem(problem)
    dependencies = {t.name: {producers[b] for b in t.reads if b in producers} for t in problem.tasks}
    ordered = []
    while len(ordered) < len(tasks):
        ready = sorted(n for n, deps in dependencies.items() if n not in ordered and deps <= set(ordered))
        ordered.extend(ready)
    return ordered, dependencies


def lower_bound(problem):
    order, deps = dependency_order(problem)
    tasks = {t.name: t for t in problem.tasks}
    earliest_end = {}
    for name in order:
        earliest_end[name] = max((earliest_end[p] for p in deps[name]), default=0) + tasks[name].duration
    work_bounds = [math.ceil(sum((r.end-r.begin)*r.units for t in problem.tasks
                                 for r in t.reservations if r.resource == resource.name) / resource.capacity)
                   for resource in problem.resources]
    return max([max(earliest_end.values())] + work_bounds)


def _resource_legal(problem, starts):
    capacities = {r.name: r.capacity for r in problem.resources}
    events = {name: [] for name in capacities}
    for t in problem.tasks:
        if t.name in starts:
            for r in t.reservations:
                events[r.resource] += [(starts[t.name] + r.begin, r.units),
                                        (starts[t.name] + r.end, -r.units)]
    for name, entries in events.items():
        usage = 0
        # Ends sort before starts: touching half-open intervals are legal.
        for _, change in sorted(entries):
            usage += change
            if usage > capacities[name]:
                return False
    return True


def _lifetimes(problem, starts):
    tasks = {t.name: t for t in problem.tasks}
    writers = {b: t.name for t in problem.tasks for b in t.writes}
    horizon = max(starts[n] + t.duration for n, t in tasks.items())
    result = {}
    for b in problem.buffers:
        writer = writers.get(b.name)
        first = starts[writer] if writer else 0
        last = max([starts[t.name] + t.duration for t in problem.tasks if b.name in t.reads]
                   + [starts[writer] + tasks[writer].duration if writer else 0])
        result[b.name] = (first, horizon if b.retain else last)
    return result


def _placements(problem, starts, budget, greedy=False):
    capacities = {m.name: m.capacity for m in problem.memories}
    life = _lifetimes(problem, starts)
    ordered = sorted(problem.buffers, key=lambda b: (life[b.name][0], -b.size, b.name))
    assigned = {}

    def visit(i):
        if i == len(ordered):
            yield dict(assigned)
            return
        b = ordered[i]
        first, last = life[b.name]
        blockers = [other for other in ordered[:i] if other.memory == b.memory and
                    max(first, life[other.name][0]) < min(last, life[other.name][1])]
        if greedy:
            candidates = sorted({0} | {((assigned[o.name]+o.size+b.alignment-1)//b.alignment)*b.alignment
                                      for o in blockers})
        else:
            candidates = range(0, capacities[b.memory]-b.size+1, b.alignment)
        for offset in candidates:
            budget.step()
            if offset + b.size > capacities[b.memory]:
                continue
            if any(max(offset, assigned[o.name]) < min(offset+b.size, assigned[o.name]+o.size) for o in blockers):
                continue
            assigned[b.name] = offset
            yield from visit(i + 1)
            del assigned[b.name]
            if greedy:
                break
    yield from visit(0)


def serial_fallback(problem, budget=None):
    order, _ = dependency_order(problem)
    tasks = {t.name: t for t in problem.tasks}
    starts, cursor = {}, 0
    for name in order:
        starts[name] = cursor
        cursor += tasks[name].duration
    budget = budget or Budget(max_expansions=100000, seconds=10)
    try:
        offsets = next(_placements(problem, starts, budget, greedy=True))
    except (StopIteration, BudgetExhausted):
        return None
    certificate = Certificate(problem.digest(), starts, offsets)
    try:
        verify(problem, certificate)
    except IllegalSchedule:
        return None
    return certificate


def exact_search(problem, horizon, *, max_expansions=100000, seconds=60):
    require(integer(horizon, 1), "positive integer horizon required")
    order, dependencies = dependency_order(problem)
    tasks = {t.name: t for t in problem.tasks}
    bound = lower_bound(problem)
    budget = Budget(max_expansions, seconds)
    incumbent = serial_fallback(problem, budget)
    if incumbent and verify(problem, incumbent)["makespan_ticks"] > horizon:
        incumbent = None
    upper = min(horizon, verify(problem, incumbent)["makespan_ticks"] if incumbent else horizon)
    result, optimal = None, False
    starts = {}

    def timelines(index, deadline):
        if index == len(order):
            for offsets in _placements(problem, starts, budget):
                candidate = Certificate(problem.digest(), dict(starts), offsets)
                verify(problem, candidate)  # independent final acceptance, never bypassed
                yield candidate
            return
        name = order[index]
        first = max((starts[p] + tasks[p].duration for p in dependencies[name]), default=0)
        for tick in range(first, deadline - tasks[name].duration + 1):
            budget.step()
            starts[name] = tick
            if _resource_legal(problem, starts):
                yield from timelines(index + 1, deadline)
            del starts[name]

    try:
        for deadline in range(bound, upper + 1):
            result = next(timelines(0, deadline), None)
            if result:
                optimal = True
                break
    except BudgetExhausted:
        pass
    if result is None:
        result = incumbent
    cost = verify(problem, result)["makespan_ticks"] if result else None
    return {"certificate": result, "objective": cost, "lower_bound": bound,
            "optimal": optimal, "infeasible_within_horizon": result is None and budget.reason is None,
            "horizon": horizon, "expansions": budget.expansions, "stop_reason": budget.reason,
            "deterministic_expansion_budget": budget.reason != "wall_timeout"}
