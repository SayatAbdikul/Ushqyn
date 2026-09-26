"""Check a proposed placement/timeline without importing its scheduler.

Intervals are half-open. A produced buffer reserves storage from producer start
until every consumer completes; its bytes become readable at producer end.
This intentionally forbids in-place reuse and early/partial-output consumption.
"""
from .contract import integer, require, validate_problem


def verify(problem, certificate):
    memories, resources, buffers, tasks, producers = validate_problem(problem)
    require(certificate.problem_sha256 == problem.digest(), "problem/quantization identity mismatch")
    require(set(certificate.starts) == set(tasks), "missing or extra scheduled task")
    require(set(certificate.offsets) == set(buffers), "missing or extra buffer placement")
    for name, start in certificate.starts.items():
        require(integer(start), f"invalid start time: {name}")
    ends = {name: certificate.starts[name] + task.duration for name, task in tasks.items()}
    makespan = max(ends.values())
    users = {name: [] for name in buffers}
    for name, task in tasks.items():
        for operand in task.reads:
            users[operand].append(name)
            if operand in producers:
                require(ends[producers[operand]] <= certificate.starts[name],
                        f"read before producer completion: {operand}")
    layout = []
    for name, buffer in buffers.items():
        offset = certificate.offsets[name]
        require(integer(offset) and offset % buffer.alignment == 0 and
                offset + buffer.size <= memories[buffer.memory].capacity,
                f"out-of-bounds or unaligned placement: {name}")
        writer = producers.get(name)
        first = certificate.starts[writer] if writer is not None else 0
        last = max([ends[user] for user in users[name]] + [ends[writer] if writer is not None else 0])
        if buffer.retain:
            last = makespan
        layout.append({"buffer": name, "memory": buffer.memory, "offset": offset,
                       "size": buffer.size, "first": first, "last": last})
    for i, left in enumerate(layout):
        for right in layout[i + 1:]:
            if left["memory"] != right["memory"]:
                continue
            alive = max(left["first"], right["first"]) < min(left["last"], right["last"])
            alias = max(left["offset"], right["offset"]) < min(left["offset"] + left["size"], right["offset"] + right["size"])
            require(not (alive and alias), f"live buffers overlap: {left['buffer']}, {right['buffer']}")
    peaks = {}
    for name, resource in resources.items():
        events = {}
        for task in problem.tasks:
            for use in task.reservations:
                if use.resource == name:
                    first, last = certificate.starts[task.name] + use.begin, certificate.starts[task.name] + use.end
                    events[first] = events.get(first, 0) + use.units
                    events[last] = events.get(last, 0) - use.units
        live, peak = 0, 0
        for tick in sorted(events):
            live += events[tick]
            peak = max(peak, live)
            require(live <= resource.capacity, f"resource conflict: {name} at {tick}")
        peaks[name] = peak
    return {"status": "passed", "problem_sha256": problem.digest(),
            "makespan_ticks": makespan, "task_count": len(tasks), "layout": layout,
            "peak_resource_demand": peaks,
            "scope": "fixed-task model legality; not numerical equivalence, RTL lowering, or measured latency"}
