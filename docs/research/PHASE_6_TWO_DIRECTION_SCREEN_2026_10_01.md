# Two architecture directions: bounded screens — 1 October 2026

The paired pointwise resource-placement and residual tile-ownership screens were run in parallel. **Neither currently establishes a new architecture result.** The physical probes reject three simple ways to reclaim logic for paired pointwise execution. On fixed residual graphs, runtime tile ownership has no advantage over a compiler's static last-use schedule. These conclusions are bounded to the mechanisms and workloads below; they do not prove that all resource-aware scheduling or residual architectures are exhausted.

## Physical resource placement for paired pointwise execution

The exact paired v3 RTL had already passed the KWS/VWW native correctness matrix and reduced fixed-memory native cycles by **14.114% KWS** and **10.268% VWW** relative to the selected co-issue control. Its Gowin build fails synthesis at **21,912/20,736 logic**. No paired result has been placed, routed, or measured on the board. The strongest selected *measured* board image is a different, cached-weight co-issue engine and must not be conflated with the paired native results. [Prior paired gate](PAIRED_POINTWISE_PORT_FABRIC_GATE_2026_09_29.md) · [selected board context](PARALLEL_ARCHITECTURE_EXPERIMENTS_2026_09_28.md)

Three isolated copies of paired v3 used the same 27 MHz Gowin recipe. Their changed RTL has not passed the native correctness matrix, so the builds are synthesis-only feasibility probes:

| Probe | Synthesized logic / 20,736 | Saved versus v3 | Physical decision |
| --- | ---: | ---: | --- |
| Weight cache requested as block RAM | 21,912 | 0 | Gowin rejects the RAM-style attribute for the asynchronous cache read; no improvement. |
| Activation tile requested as block RAM | 21,833 | 79 | Still 1,097 over the hard limit. |
| Weight cache hits disabled | 21,283 | 629 | Still 547 over the hard limit, with a predicted 30,720/39,212 extra 64-bit weight requests on KWS/VWW pointwise descriptors. |

The extra requests in the no-cache model target the on-chip scratchpad, where DMA has placed weight tiles; they are not a claim of extra SDRAM traffic or measured latency. None of these designs reached placement or timing. The source-pinned [physical screen](../../work/phase6/physical-resource-screen-v1/README.md) contains the raw Gowin reports, logs, input hashes, traffic calculation and memory cap. A new resource-aware engine would need a different port-aware state design, enough logic saving to leave placement margin, preserved exact native arithmetic and latency benefit, then a matched 27 MHz route and board campaign. The unused BSRAM capacity alone does not meet that gate.

Gowin's failed synthesis logs report whole-design resources, so the exact modules responsible for the remaining logic excess are not yet attributed. This limits the screen to the three source changes above.

## Exact residual tile ownership

The current KWS/VWW graphs are linear and the FPGA compiler does not lower residual Add. The screen therefore used four **synthetic residual blocks anchored to real frozen tensor shapes**, a 32 KiB scratchpad, a single shared 64-bit data port, and eight MAC lanes. It compared full-map liveness, ordinary blocked streaming, a resident-source in-place Add, static source overwrite at last use, and runtime tile ownership. It charged halos, intermediate quantized buffers, deferred outputs, copy reads/writes and runtime owner-token traffic.

For a fixed graph and tile order, the compiler knows each source tile's final halo consumer. It can schedule the same overwrite as a runtime token. The zero-token-cost ownership lower bound therefore has the **same storage and port work as static last-use for every tested tile choice**; actual token storage and updates make the runtime policy weakly worse. The strongest ordinary policies also outperform it in the optimistic MAC-plus-port proxy: **1,464,672 versus 1,651,220** for the VWW-shape basic block, and **1,298,112 versus 1,672,304** for the VWW-shape inverted block. These scores are neither device cycles nor a rigorous latency bound.

Small synthetic INT8 basic and inverted blocks matched the independent integer oracle on four tile sizes (196 output values per program per tiling). Nine Add edge cases passed ties-away rounding and saturation. There is no residual full-model accuracy, RTL Add implementation, routed hardware or board timing. The reproducible [residual screen](../../work/phase6/residual-ownership-screen-v1/README.md) and [model script](../../tools/phase6/residual_ownership_screen.py) record these limits.

**Research decision.** Stop runtime ownership as the central architecture claim for statically known graphs. Stop simple cache/BRAM style changes as a route to paired physical fit. A future resource-aware scheduling claim needs a new execution/storage mechanism that survives strong ordinary sharing and blocking controls across held-out shapes and physical budgets. The current evidence does not justify attributing a Q1-level novelty claim to either screened mechanism.
