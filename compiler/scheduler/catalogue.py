"""Exact and deterministic beam search over a declared segment catalogue.

The state string is the COMPLETE boundary contract (live placement, arithmetic,
layout, etc.). Only identical (position,state) frontiers can dominate each other.
Costs must be independent of earlier history once that state is known. This is
an explicit assumption checked by adapters, not a universal scheduler theorem.
"""
from dataclasses import dataclass
import math
import time

from .contract import integer, require


@dataclass(frozen=True)
class Choice:
    name: str
    start: int
    stop: int
    input_state: str
    output_state: str
    cost: int


@dataclass(frozen=True)
class Catalogue:
    length: int
    choices: tuple[Choice, ...]
    initial_state: str = "materialized"
    final_states: tuple[str, ...] = ("materialized",)


def validate(catalogue):
    require(integer(catalogue.length, 1), "invalid catalogue length")
    names = set()
    for c in catalogue.choices:
        require(isinstance(c.name, str) and c.name and c.name not in names, "duplicate/empty choice")
        require(integer(c.start) and integer(c.stop, 1) and c.start < c.stop <= catalogue.length,
                "invalid segment bounds")
        require(integer(c.cost) and isinstance(c.input_state, str) and c.input_state and
                isinstance(c.output_state, str) and c.output_state, "invalid cost/state")
        names.add(c.name)
    require(isinstance(catalogue.initial_state, str) and bool(catalogue.initial_state) and
            bool(catalogue.final_states) and all(isinstance(s, str) and s for s in catalogue.final_states),
            "empty/invalid boundary state")


def evaluate(catalogue, path):
    validate(catalogue)
    by_name = {c.name: c for c in catalogue.choices}
    position, state, total = 0, catalogue.initial_state, 0
    for name in path:
        require(name in by_name, "unknown candidate")
        choice = by_name[name]
        require((choice.start, choice.input_state) == (position, state), "incompatible frontier")
        position, state = choice.stop, choice.output_state
        total += choice.cost
    require(position == catalogue.length and state in catalogue.final_states, "incomplete path")
    return total


def _bounds(catalogue):
    # Relax boundary states; valid lower bound on this catalogue's additive cost.
    result = [math.inf] * (catalogue.length + 1)
    result[-1] = 0
    for position in reversed(range(catalogue.length)):
        result[position] = min((c.cost + result[c.stop] for c in catalogue.choices if c.start == position), default=math.inf)
    return result


def search(catalogue, *, beam_width=None, max_expansions=1000000, seconds=60, fallback=None):
    """DP if beam_width=None; bounded beam otherwise. Always validate fallback.

Expansion budgets/tie-breaking are deterministic. A wall timeout is an emergency
cutoff explicitly flagged as non-reproducible truncation. No optimality claim
survives beam pruning or a budget cutoff unless the certified bound is reached.
"""
    validate(catalogue)
    require(beam_width is None or integer(beam_width, 1), "invalid beam width")
    require(integer(max_expansions, 1) and math.isfinite(seconds) and seconds > 0, "invalid search budget")
    began = time.monotonic()
    best = (evaluate(catalogue, fallback), tuple(fallback)) if fallback is not None else None
    bounds = _bounds(catalogue)
    states = [dict() for _ in range(catalogue.length + 1)]
    states[0][catalogue.initial_state] = (0, ())
    outgoing = {}
    for choice in sorted(catalogue.choices, key=lambda c: c.name):
        outgoing.setdefault((choice.start, choice.input_state), []).append(choice)
    expansions, pruned, stop_reason = 0, 0, None
    for position in range(catalogue.length):
        candidates = sorted(states[position].items(), key=lambda entry: (entry[1][0]+bounds[position], entry[1][1], entry[0]))
        if beam_width is not None and len(candidates) > beam_width:
            pruned += len(candidates) - beam_width
            candidates = candidates[:beam_width]
        for state, (cost, path) in candidates:
            for choice in outgoing.get((position, state), []):
                if expansions >= max_expansions:
                    stop_reason = "expansion_budget"
                    break
                if time.monotonic() - began >= seconds:
                    stop_reason = "wall_timeout"
                    break
                expansions += 1
                proposed = (cost + choice.cost, path + (choice.name,))
                key = choice.output_state
                old = states[choice.stop].get(key)
                if old is None or proposed < old:
                    states[choice.stop][key] = proposed
                if choice.stop == catalogue.length and key in catalogue.final_states:
                    if best is None or proposed < best:
                        best = proposed
            if stop_reason:
                break
        if stop_reason:
            break
    if best is not None:
        require(evaluate(catalogue, best[1]) == best[0], "search output verification failed")
    exact = stop_reason is None and pruned == 0
    optimal = best is not None and (exact or best[0] == bounds[0])
    certified_bound = best[0] if best and exact else bounds[0]
    return {"path": list(best[1]) if best else None, "objective": best[0] if best else None,
            "relaxed_lower_bound": bounds[0] if math.isfinite(bounds[0]) else None,
            "lower_bound": certified_bound if math.isfinite(certified_bound) else None,
            "bound_gap_fraction": ((best[0]-certified_bound)/max(1, certified_bound)) if best and math.isfinite(certified_bound) else None,
            "optimal_within_catalogue": optimal, "infeasible": best is None and exact,
            "expansions": expansions, "pruned_states": pruned, "stop_reason": stop_reason,
            "beam_width": beam_width, "wall_seconds": time.monotonic()-began,
            "deterministic_expansion_budget": stop_reason != "wall_timeout"}
