# Producer-native padded channel planes: cost gate

The proposed change would make the preceding depthwise layer write each
channel into an 8-byte-aligned physical plane. Its following in-place ReLU
would need to traverse that physical layout, then the pointwise layer could
consume it directly without opcode 9 PACK. This is a reusable layout feature,
but it requires two descriptor/RTL extensions and SRAM lifetime changes.

The existing exact PACK candidate has four KWS and two VWW PACK operations.
Its 12 complete-model native checks passed, but the measured fixed/stalled
speedups were only 1.00643/1.00628 KWS and 1.00312/1.00306 VWW. A route was
not obtained because Gowin exited before synthesis in the tool environment.

We instrumented an isolated native Verilator bridge at the engine state level.
For both the fixed and stalled external RAM models, the four KWS PACK bodies
cost **19,524 host clock cycles**; the two VWW PACK bodies cost **3,842**.
The instrumented runs were output-exact and reproduced the selected and PACK
candidate total cycle counts. These counts include actual scratchpad
arbitration in the native simulator. They exclude descriptor fetch and DMA.

An optimistic producer-native proxy removes all PACK body cycles and all
measured extra DMA cycles, and assumes padded producer and ReLU addressing
cost zero cycles. It gives KWS **1.0212×** and VWW **1.0043×** fixed timing.
Giving an additional **2,048 cycles per removed PACK** for descriptor fetch
and transition raises those estimates to **1.0275× KWS, 1.0056× VWW**, with a
**1.0165×** geometric mean. The 2,048-cycle allowance is deliberately generous
and is a scenario assumption, not a measured descriptor cost.

To test whether padding every remaining VWW pointwise tile could change the
decision, we counted all remaining `PW_X2_REQ/PW_X2_WAIT` cycles specifically
while `spatial_pw` was active: **0 KWS** and **47,424 VWW**. The most aggressive
proxy also removes every one of those cycles, including tiles that currently
cannot fit padded SRAM. It reaches **1.0275× KWS, 1.0206× VWW**, or a
**1.0240× geometric mean** with fixed RAM; the stalled result is **1.0235×**.
This is still below the declared **1.03×** expansion trigger. It does not
model the cost of extra padded writes, ReLU mapping, retiling, or possible
loss of routing frequency.

**Decision:** stop this variant before implementing producer-native padded
output. The exact measured removal opportunity is too small on the current
audio and vision models. The broader pointwise read problem remains worth
addressing through a different architecture if a workload with larger
unaligned planes justifies it.

Reproduction: `python tools/phase6/profile_padded_cost.py`; the signed
machine-readable report is
[`evidence/phase6/producer-native-padding-cost-v1.json`](evidence/phase6/producer-native-padding-cost-v1.json).
