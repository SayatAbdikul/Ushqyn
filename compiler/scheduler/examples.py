"""Hand-derived fixtures for the boardless Phase 6 contract."""
from .contract import Buffer, Certificate, Memory, Problem, Reservation, Resource, Task


def quantized_chain():
    problem = Problem(
        memories=(Memory("scratch", 64),),
        resources=(Resource("engine", 1), Resource("sram_port", 1)),
        buffers=(Buffer("input", "scratch", 8, 8, initial=True),
                 Buffer("acc32", "scratch", 32, 8),
                 Buffer("q8", "scratch", 8, 8), Buffer("output", "scratch", 8, 8, retain=True)),
        tasks=(Task("mac", "compute", 4, ("input",), ("acc32",), "example:raw-int8-mac-int32-v1",
                    (Reservation("engine", 0, 4), Reservation("sram_port", 0, 4))),
               Task("quant", "requantize", 2, ("acc32",), ("q8",), "example:fixed-scale-round-saturate-v1",
                    (Reservation("engine", 0, 2), Reservation("sram_port", 0, 2))),
               Task("next", "compute", 3, ("q8",), ("output",), "example:next-fixed-int8-op-v1",
                    (Reservation("engine", 0, 3), Reservation("sram_port", 0, 3)))))
    # acc32 and output reuse bytes [8,40) after acc32's last read completes.
    cert = Certificate(problem.digest(), {"mac": 0, "quant": 4, "next": 6},
                       {"input": 0, "acc32": 8, "q8": 0, "output": 8})
    return problem, cert


def disjoint_dma():
    problem = Problem(
        memories=(Memory("scratch", 32), Memory("external", 16)),
        resources=(Resource("engine", 1), Resource("dma", 1), Resource("sram_port", 1)),
        buffers=(Buffer("input", "scratch", 8, 8, initial=True),
                 Buffer("output", "scratch", 8, 8, retain=True),
                 Buffer("source", "external", 8, 8, initial=True),
                 Buffer("prefetch", "scratch", 8, 8, retain=True)),
        tasks=(Task("compute", "compute", 6, ("input",), ("output",), "example:compute-v1",
                    (Reservation("engine", 0, 6), Reservation("sram_port", 0, 2))),
               Task("load", "dma", 2, ("source",), ("prefetch",), "example:byte-copy-v1",
                    (Reservation("dma", 0, 2), Reservation("sram_port", 0, 2)))))
    cert = Certificate(problem.digest(), {"compute": 0, "load": 2},
                       {"input": 0, "output": 8, "source": 0, "prefetch": 16})
    return problem, cert
