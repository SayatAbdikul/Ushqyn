"""Fixed-operation schedule contract, independent of any candidate generator.

Operations and buffer versions are immutable. Quantization is an explicit task
with an opaque arithmetic identity. Certificates only choose times/addresses;
they cannot remove or fuse tasks. Numerical equivalence and RTL lowering are
separate obligations. Port reservations must describe the actual access trace
before using this abstraction to certify a hardware schedule.
"""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class Memory:
    name: str
    capacity: int


@dataclass(frozen=True)
class Resource:
    name: str
    capacity: int


@dataclass(frozen=True)
class Buffer:
    name: str
    memory: str
    size: int
    alignment: int = 1
    initial: bool = False
    retain: bool = False


@dataclass(frozen=True)
class Reservation:
    resource: str
    begin: int
    end: int
    units: int = 1


@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    duration: int
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    semantic_id: str
    reservations: tuple[Reservation, ...]


@dataclass(frozen=True)
class Problem:
    memories: tuple[Memory, ...]
    resources: tuple[Resource, ...]
    buffers: tuple[Buffer, ...]
    tasks: tuple[Task, ...]
    schema: int = 1

    def digest(self):
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Certificate:
    problem_sha256: str
    starts: dict[str, int]
    offsets: dict[str, int]


class IllegalSchedule(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise IllegalSchedule(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def named(items, label):
    require(all(isinstance(x.name, str) and x.name for x in items), f"empty {label} name")
    result = {x.name: x for x in items}
    require(len(result) == len(items), f"duplicate {label} name")
    return result


def validate_problem(problem):
    require(type(problem.schema) is int and problem.schema == 1, "unsupported schema")
    memories = named(problem.memories, "memory")
    resources = named(problem.resources, "resource")
    buffers = named(problem.buffers, "buffer")
    tasks = named(problem.tasks, "task")
    require(bool(memories) and bool(resources) and bool(tasks), "empty scheduling problem")
    for item in (*problem.memories, *problem.resources):
        require(integer(item.capacity, 1), f"invalid capacity: {item.name}")
    for b in problem.buffers:
        require(b.memory in memories and integer(b.size, 1) and integer(b.alignment, 1),
                f"invalid buffer geometry: {b.name}")
        require(type(b.initial) is bool and type(b.retain) is bool, f"invalid buffer flags: {b.name}")
        require(b.size <= memories[b.memory].capacity, f"buffer exceeds memory: {b.name}")
    producers = {}
    for task in problem.tasks:
        require(task.kind in ("compute", "dma", "requantize"), f"unsupported task kind: {task.name}")
        require(integer(task.duration, 1) and isinstance(task.semantic_id, str) and
                bool(task.semantic_id.strip()), f"invalid duration/arithmetic identity: {task.name}")
        require(len(set(task.reads)) == len(task.reads) and
                len(set(task.writes)) == len(task.writes), f"duplicate operand: {task.name}")
        require(not set(task.reads) & set(task.writes), f"in-place operation unsupported: {task.name}")
        require(bool(task.writes), f"task has no result: {task.name}")
        for name in task.reads + task.writes:
            require(name in buffers, f"unknown buffer: {name}")
        for name in task.writes:
            require(not buffers[name].initial and name not in producers, f"multiple/initial writer: {name}")
            producers[name] = task.name
        require(bool(task.reservations), f"missing resource reservations: {task.name}")
        for reservation in task.reservations:
            require(reservation.resource in resources and integer(reservation.begin) and
                    integer(reservation.end, 1) and integer(reservation.units, 1) and
                    reservation.begin < reservation.end <= task.duration,
                    f"invalid reservation: {task.name}")
            require(reservation.units <= resources[reservation.resource].capacity,
                    f"reservation exceeds resource: {task.name}")
    for b in problem.buffers:
        require(b.initial or b.name in producers, f"buffer has no producer: {b.name}")
    dependencies = {t.name: {producers[b] for b in t.reads if b in producers}
                    for t in problem.tasks}
    visited = set()
    while len(visited) < len(tasks):
        ready = {name for name, deps in dependencies.items() if name not in visited and deps <= visited}
        require(bool(ready), "dependency cycle")
        visited.update(ready)
    return memories, resources, buffers, tasks, producers
