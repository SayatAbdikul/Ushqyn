# R05 decision: physical port awareness is established prior work

The tested v2 engine has a single synchronous SRAM port. A GEMM reduction group
of eight lanes requires at least one activation word and one weight word when
neither is retained in a register. The [service-bound trace](evidence/phase2/memory-service-bound.json)
shows that a scheduler assuming two simultaneous reads can service one group
per cycle, whereas this specific port can service at most one word per cycle:
two service cycles per group before accounting for descriptor, parameter,
compute and output cycles. The randomized memory-latency engine test holds each
request through backpressure and checks exact read/write counters. The board
hierarchy uses one inferred synchronous SRAM port; Gowin reports 16 SP BSRAM
blocks. These observations validate the constraint, not a new algorithm.

[DeFiNES](https://github.com/KULeuven-MICAS/DeFiNES) explicitly describes
memory-level physical ports and operand read/write direction. Properly adapted,
it is capable of observing this one-port restriction. [COSMA](https://arxiv.org/html/2311.18246v1)
already couples schedule, address allocation and tensor replacement, although its
published operator-level formulation differs from a tile-level execution graph.
Thus the simple bound does **not** supply the roadmap's proposed counterexample
to a strong B3 baseline, and claims of first port-aware fusion, first joint
memory/schedule optimization or first physical bank accounting are rejected.

The research decision is to treat phase 2 as an engineering platform. A future
algorithm paper must show, on the same eight-lane engine and real routed memory
modes, that an executable tile/live-state/bank certificate enables better legal
schedules than a tuned DeFiNES adaptation and tile-expanded COSMA. Use the same
model weights, arithmetic, clock, SRAM/SDRAM budget, host boundary and tuning
budget. The proposed 15% two-workload geometric-mean improvement remains a
prospective go/no-go threshold. If the gain comes only from supplying Gowin costs
to existing scheduling machinery, describe the result as a backend/methodology
contribution or pivot to a distinct mechanism. No baseline performance numbers
or SOTA ranking are claimed at R05.

For the routed MLP, a sampled final simulated job reports 10,112 useful MACs,
22,112 accepted SRAM read bytes and 98 written output bytes. Its 7,311 core
cycles split into 1,324 compute, 5,626 wait and 361 control. The wait counter
covers descriptor/parameter reads, synchronous response latency, reads of
activations and weights, and writes. It cannot be interpreted as a measurement of
stall cycles caused solely by memory contention, nor does it predict the latency
of a fused schedule. It does show that the board implementation exposes a
substantial service cost to optimize. The hardware was routed with 16 single-port
BSRAM blocks and one logical 64-bit request channel, so candidate schedules must
respect one accepted request per cycle at that channel.

The simple `2G` lower bound assumes each of the `G=ceil(K/8)` reduction groups
needs one activation word and one weight word from this channel and that neither
is already cached in the engine. At least `2G` acceptance slots are then needed,
while a hypothetical two-port interface could provide the same `2G` words in `G`
slots. A real layer also needs descriptor/parameter traffic and output writes.
Conversely, an implementation retaining activation words across output channels
may reduce these reads; adding a second bank, data reuse or prefetching changes
the premise. Any paper experiment must state the actual bank topology and count
traffic from the emitted schedule, rather than treating `2G` as a universal
latency floor.

A useful follow-up counterexample, if one exists, must hold arithmetic and the
same bank/port budget fixed while showing that a tuned published policy produces
an illegal or materially worse schedule after independent legality checking. A
straight cost-model correction is not enough. The next comparison should report:
model and calibration hashes, the exact runnable tile graph, allocator lifetimes,
per-bank address/port occupancy per cycle, transfer totals, corrected INT32 live
state, routed resource use, measured core cycles and host boundary. Baselines
should be given enough tuning time to choose their own legal cache/recomputation
policy. If they match the proposed method, the hypothesis fails and the project
should publish its reproducible tiny-FPGA implementation as engineering evidence
rather than claiming a SOTA scheduling method.
