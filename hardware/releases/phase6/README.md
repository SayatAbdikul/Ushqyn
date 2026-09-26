# Phase 6 boardless candidates

These gzip-compressed Gowin images pass synthesis, placement, routing and the
existing 20.25 MHz timing constraints. Neither has been tested on the physical
board. Do not replace an image while a Phase 5 campaign is running.

| File | Uncompressed SHA256 | Core Fmax |
|---|---|---:|
| `c256p1_candidate_750k.fs.gz` | `1dbc989708a4b24e342935a5b62ce1f98d055041801d9f4a98854eafb7fe2051` | 24.123 MHz |
| `spatial_candidate_750k.fs.gz` | `c20147ff9f35a677ebbc0e24281fc65ab14c35d33d807f49bc5eaf7092288500` | 20.798 MHz |

Target: GW2AR-LV18QN88C8/I7, device revision C, Tang Nano 20K. Both retain the
frozen Phase 5 host protocol, 750,000-baud UART, SDRAM controller, pin constraints,
32 KiB scratchpad and 20.25 MHz operating clock. The first adds a 256-entry
pointwise input cache and parameter reuse. The second also executes eight
neighboring pointwise pixels on the eight multipliers.

Use Python's standard `gzip` module or `gzip -dc` to extract an image to a new
file; verify its uncompressed SHA256 before future programming. Build sources,
reports, correctness evidence and the unrun physical plan are documented in
[the milestone report](../../../docs/research/PHASE_6_OPTIMIZATION.md).
The earlier spatial image that failed timing is not included here.
