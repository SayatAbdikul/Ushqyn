# Static INT8 numerical contract, version 2

Status: implemented by `compiler/static_pipeline.py` and the software descriptor
VM. Phase 2 implements this arithmetic in the shared eight-lane FC/ReLU board RTL
through the separate 64-byte hardware descriptor ABI; software `.uq2` VM images
are lowered and checked before board loading, never flashed as-is. Historical
v1 RTL/golden files remain for regression reproduction. CNN/depthwise/pooling
hardware migration remains phase 3–4. This contract does not claim bit identity
with TFLite's implementation-specific requantization.

## Representation and calibration

Activations use signed INT8 `[-128,127]`, one positive finite scale `s` and integer
zero point `z` per tensor: `real = s*(q-z)`. Saved calibration includes the SHA256
of the canonical model, unique sample IDs, hashes of input contents, and observed
min/max for every runtime intermediate. Zero is included in the range. All-zero
ranges use scale 1, zero point -128. No input-dependent dynamic scale is permitted.
Weights use symmetric INT8 `[-127,127]`, zero point zero, and one scale per output
channel, initially `max(abs(w[c]))/127`, or 1 for an all-zero channel. Enlarge this
scale when necessary to fit the product-unit bias and representable requantization
ratio: reserve `(128+abs(zx))*127*reduction_length + 2` INT32 units, then floor the
weight scale at `abs(b)/(sx*remaining_budget)` and `(sy/sx)*2^-31`. This handles
near-zero BN-folded channels without wrapping or saturating bias. All final raw
and centered bounds are still checked after weight quantization. It can reduce
weight precision; whole-model quality must be evaluated independently.

Float-to-integer conversion rounds nearest with exact half ties **away from zero**:
`sign(x)*floor(abs(x)+0.5)`. Bias is `round(b/(sx*sw[c]))`, stored as signed
little-endian INT32. A missing bias becomes an explicit zero vector. Every scale
and source weight/bias must be finite. The software uses float64 for scale
construction; these scales and all fixed-point coefficients are preserved in the
image. Calibration ranges never serve as a proof that an integer cannot overflow.

## MAC and correction

The mathematical accumulator is `qb[c] + sum((qx-zx)*qw[c])`. To retain signed
8×8 multiplier inputs, the raw-MAC form uses
`corrected_bias[c] = qb[c] - zx*sum(qw[c])`. Compute correction with a wide compiler
intermediate and add it **once per complete output reduction**, not once per tile.
Padding contributes `qx=zx`, including products in the raw-MAC form. Omitting
padding products while applying the full correction is incorrect.

Before emitting parameters, prove both bounds per channel are at most `2^31-1`:

* `128*sum(abs(qw)) + abs(corrected_bias)` for raw products/partial sums;
* `max(abs(-128-zx),abs(127-zx))*sum(abs(qw)) + abs(qb)` for the centered oracle.

These conservative bounds prove all accumulation orders safe. Reject otherwise;
never silently wrap or saturate an intermediate. INT32 bias range is checked
before casting. Layer tiling and its hardware implementation remain later work.

## Requantization

Approximate `r=sx*sw[c]/sy` as `M/2^S`. Use `frexp(r)`, round its significand times
`2^31` to nearest, ties away, then normalize a rounded `2^31` multiplier. Require
`1<=M<=2^31-1` and `0<=S<=62`; reject unrepresentable ratios.

Compute signed 64-bit `p=acc*M`; round `abs(p)/2^S` to nearest, ties away; restore
the sign, add `zy`, and saturate only the output to `[-128,127]`. `S=0` performs no
rounding shift. Examples before zero point/clamp: `3/2 -> 2`, `-3/2 -> -2`,
`1/2 -> 1`, `-1/2 -> -1`. Accumulator and multiplier bounds keep both product and
rounding offset inside signed 64-bit. Python/NumPy scalar types are normalized to
Python integers before shifts to avoid narrow scalar overflow.

## Other operators and layouts

Conv uses batch-1 NCHW, constant OIHW weights, explicit independent padding,
stride, dilation and groups. Gemm is canonicalized to constant `[out,in]` weights
and `transB=1`; alpha/beta and legal scalar/channel bias broadcasts are folded.
`transA` and runtime weights reject. BN folds only into an exclusively consumed
Conv; standalone/training BN rejects. Exported channelwise Mul/Add affine forms
also fold into an exclusively consumed Conv/Gemm, including negative or zero
scales; observable/shared producers cannot be changed. No BN aliases a buffer.

ReLU clamps at the input zero point and requantizes. Clip encodes its finite scalar
bounds into the input quantization, clamps the integers, then requantizes; this
supports ReLU6 with the stated quantization error at its upper bound.
Reshape/Flatten/Identity and
Transpose preserve quantization metadata and every declared graph output.
MaxPool takes the maximum over valid input elements, then rescales. AveragePool
sums centered valid elements in INT32, applies a multiplier containing the divisor,
and rounds once; `count_include_pad` changes that divisor. Empty windows, ceil-mode,
and dilated pooling reject. GlobalAveragePool uses the complete spatial window.

Add broadcasts operands with ONNX semantics, scales each centered input into one
common fixed-point denominator, adds signed 64-bit products, then rounds once and
saturates. It must not round/saturate each operand separately. Its input centers
are at most 255 and coefficients at most `2^31-1`; their two-term sum fits INT64.

Unknown operators, custom domains, dynamic shapes and unsupported attributes
reject before output file creation. The compiler accepts a documented subset;
full benchmark import/quality is a separate gate and never inferred from unit tests.

## Image contract and verification

`USHQYN2\0`, a little-endian header length, JSON metadata and 16-byte aligned owned
segments form the software image. The header fixes target, image/numerical version,
capacity, graph inputs/outputs and typed tensor descriptors. Segments own tensors,
weights, INT32 bias/correction, scales, multipliers/shifts, descriptors and 64-bit
VM instructions. The low byte selects an operation; the upper 56 bits index its
descriptor. These are **software VM instructions**, not the hardware's 64-bit ISA.
Header and alignment padding are reserved bytes. SHA256 checksums detect corrupted
segments; they are integrity checks, not signatures. Bounds, overlap, length,
datatype, dependencies and target/version are checked on loading. Build fully and
validate in memory before atomically replacing an existing image.

Tests compare Python scalar integer convolution/dot/rounding oracles against the
image-executed program, and compare float BN/Gemm canonicalization against ONNX.
The image simulator shares the production operator implementation with graph
execution; graph-vs-image agreement alone is not an independent numerical oracle.
The scalar oracle is the independent check. RTL integration requires 100% exact
outputs and maximum error zero; classification accuracy is reported separately.

The ONNX 1.20 reference implementation of opset-9 BN (used by opset 13) was found
to apply running-statistic updates when its default momentum is populated. BN
parity tests therefore use explicit inference semantics under opset 14 and verify
the folding equation. Historical probe outputs are preserved verbatim, with this
oracle caveat; skipping BN is still semantically invalid.

## Classifier boundary and primary validation

`classifier_boundary.logits_model` explicitly removes one terminal rank-2,
class-axis Softmax and records host argmax (first-index ties). This preserves the
float classification decision; it does not provide quantized probabilities. The
ordinary compiler still rejects Softmax. Full software quality and independent
all-layer comparisons on selected real inputs are recorded in
[phase status](../research/PHASE_0_1_STATUS.md) and benchmark manifests. The VWW
quality split is the complete declared training-recipe validation partition,
not a certified MLPerf submission.
