"""Lower an ONNX graph to the accelerator's assembly language.

Walks the graph in topological order and emits one of the opcodes
defined in isa_spec.OPCODES (LOAD_V/LOAD_M/STORE/GEMV/RELU/CONV2D_CFG/
CONV2D_RUN/MAXPOOL) per supported op. Constant-folded ops (Reshape,
Flatten, BatchNormalization) carry buffer mappings forward without
emitting instructions.

DRAM addressing: post-P2 (2026-05-09) `generate_assembly` looks up every
initializer's DRAM address by name in the `(weight_map, bias_map,
conv_weight_map)` returned by `dram.save_all_initializers_to_dram`.

Buffer allocation: post-P4 (2026-05-09) buffer IDs come from the
liveness-based linear-scan allocator in `buffer_allocator.py`. The hand-
rolled ping-pong variables (mat_buf 1↔2, bias_vector_buf 3↔4, gemv_buf
5↔6, relu_buf 7↔8, conv_out_buf 10↔11, pool_out_buf 12↔13) are gone;
every weight / bias / activation buffer ID is read from
`assignments[tensor_name]`.

Caller contract:
    weight_map, bias_map, conv_weight_map = save_all_initializers_to_dram(model_path, dram_offsets)
    generate_assembly(model_path, asm_file, weight_map, bias_map, conv_weight_map)
"""
import onnx
from onnx import shape_inference
import numpy as np
from helper_functions import build_tensor_shape_map, build_initializer_map, topological_sort, tensor_size
from helper_functions import build_initializer_map_cnn
from accelerator_config import AcceleratorConfig
from buffer_allocator import allocate_buffers, PRIMARY_INPUT_BUFFER

# RELU instruction's length field is 10 bits — Relus on tensors larger than
# this can't be expressed standalone and are emitted as buffer-aliasing
# passthroughs (compiler emits no instruction; the allocator pre-folds the
# alias into the buffer-ID assignments).
RELU_MAX_LENGTH = (1 << 10) - 1


def get_node_attr(node, name, default=None):
    """Extract a named attribute from an ONNX node."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
    return default


def generate_assembly(model_path, output_file,
                      weight_map=None, bias_map=None, conv_weight_map=None):
    """Lower an ONNX graph to assembly.

    Args:
        model_path:         path to ONNX model
        output_file:        path to write assembly to
        weight_map:         {initializer_name: DRAM address} for FC/Gemm/MatMul weights
        bias_map:           {initializer_name: DRAM address} for ALL biases (conv + FC)
        conv_weight_map:    {initializer_name: DRAM address} for Conv weights

    The three maps are produced by `dram.save_all_initializers_to_dram` and
    are the single source of truth for initializer DRAM layout. Callers MUST
    run the DRAM walker first; convenience wrappers like the
    `if __name__ == "__main__"` block below do this internally.

    Raises ValueError if any required map is None — the previous behaviour of
    silently re-walking the graph with shadow counters is gone (it was the
    structural cause of Patches B and D).
    """
    if weight_map is None or bias_map is None or conv_weight_map is None:
        raise ValueError(
            "generate_assembly requires (weight_map, bias_map, conv_weight_map). "
            "Run `weight_map, bias_map, conv_weight_map = "
            "save_all_initializers_to_dram(model_path, dram_offsets)` first "
            "and pass the returned maps."
        )

    model = shape_inference.infer_shapes(onnx.load(model_path))
    graph = model.graph

    shape_map        = build_tensor_shape_map(model)
    initializer_map  = build_initializer_map(graph)       # MLP weights (2-D) / biases
    cnn_init_map     = build_initializer_map_cnn(graph)   # Conv weights (4-D)
    ordered_nodes    = topological_sort(graph)

    # Only inputs / outputs need address constants here — every initializer
    # address is looked up by name in the maps from save_all_initializers_to_dram.
    dram_addresses = {
        "inputs":  AcceleratorConfig.DRAM_ADDR_INPUTS,
        "outputs": AcceleratorConfig.DRAM_ADDR_OUTPUTS,
    }

    # ── Identify in-place / fused outputs (for the buffer allocator) ─────────
    # Two cases produce a Relu output that shares its input's buffer (no new
    # buffer allocated, no RELU instruction emitted):
    #   (A) Conv → Relu, where Relu is the FIRST consumer of the Conv's
    #       output → fused into CONV2D_RUN's relu_flag. We must restrict to
    #       the first consumer because topological_sort can interleave
    #       unrelated nodes (e.g. Constant operands of a later Reshape)
    #       between the Conv and its Relu — pre-Patch-D this caused silent
    #       activation drops (see git history for context).
    #   (B) Standalone Relu where the input tensor's element count exceeds
    #       RELU_MAX_LENGTH (10-bit field). The standalone RELU instruction
    #       can't address it; instead the buffer is aliased through.
    fused_relu_outputs = set()        # case A — fused into Conv
    passthrough_relu_outputs = set()  # case B — large standalone

    # Case A: walk Convs and find their first-consumer-Relu.
    pending_skip = set()
    for i, node in enumerate(ordered_nodes):
        if node.op_type != "Conv":
            continue
        for j in range(i + 1, len(ordered_nodes)):
            nxt = ordered_nodes[j]
            if nxt.output[0] in pending_skip:
                continue
            if node.output[0] not in nxt.input:
                continue
            if nxt.op_type == "Relu":
                fused_relu_outputs.add(nxt.output[0])
                pending_skip.add(nxt.output[0])
            break

    # Case B: large standalone Relus need shape info — derive from shape_map.
    def _shape_size(s):
        n = 1
        for d in s:
            n *= int(d) if isinstance(d, int) else 0
        return n
    for node in ordered_nodes:
        if node.op_type != "Relu" or node.output[0] in fused_relu_outputs:
            continue
        if _shape_size(shape_map.get(node.input[0], [])) > RELU_MAX_LENGTH:
            passthrough_relu_outputs.add(node.output[0])

    # Build alias map: relu_output → relu_input (for both fused / large cases).
    relu_aliases = {}
    for node in ordered_nodes:
        if node.op_type == "Relu" and (node.output[0] in fused_relu_outputs
                                       or node.output[0] in passthrough_relu_outputs):
            relu_aliases[node.output[0]] = node.input[0]

    # ── Run the buffer allocator ─────────────────────────────────────────────
    # The allocator owns every buffer-ID assignment. compile.py reads from
    # `assignments[name]` — there is no shadow allocator here.
    assignments = allocate_buffers(
        graph, ordered_nodes,
        vector_count=16, matrix_count=2,
        extra_aliases=relu_aliases,
    )

    asm_instructions = []
    tensor_size_map  = {}   # output element counts (for RELU length emission)
    skip_nodes       = set(fused_relu_outputs)  # fused Relus emit nothing

    # ── Emit LOAD_V for the model's primary input tensor ─────────────────────
    primary_input_name  = graph.input[0].name
    primary_input_shape = shape_map.get(primary_input_name, [])
    input_size = (int(np.prod(primary_input_shape[1:]))
                  if len(primary_input_shape) > 1 else 1)
    asm_instructions.append(
        f"LOAD_V {assignments[primary_input_name]}, "
        f"{hex(dram_addresses['inputs'])}, {input_size}"
    )

    # ── Process each node ────────────────────────────────────────────────────

    for i, node in enumerate(ordered_nodes):
        if node.output[0] in skip_nodes:
            continue

        # Reshape / Flatten / BatchNormalization: no instruction. Sizes
        # propagate so a later RELU can read the input tensor's count.
        if node.op_type in ("Reshape", "Flatten", "BatchNormalization"):
            src = node.input[0]
            dst = node.output[0]
            if src in tensor_size_map:
                tensor_size_map[dst] = tensor_size_map[src]
            continue

        # ── Initialiser loads (weights / biases) for this node ───────────────
        for arg_idx, input_name in enumerate(node.input):
            # Skip Conv weights AND bias here; both are emitted in the Conv
            # block below to keep CONV2D_CFG / CONV2D_RUN ordering tight.
            if node.op_type == "Conv" and arg_idx in (1, 2):
                continue
            if input_name not in initializer_map:
                continue
            tensor_data = initializer_map[input_name]
            tensor_type = tensor_data["type"]
            buf = assignments[input_name]

            if tensor_type == "weight":
                if len(tensor_data["shape"]) == 2:
                    rows, cols = tensor_data["shape"]
                else:
                    rows = int(np.prod(tensor_data["shape"][:-1]))
                    cols = tensor_data["shape"][-1]
                TILE_WIDTH  = AcceleratorConfig.TILE_ELEMS
                padded_cols = ((cols + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
                if input_name not in weight_map:
                    raise KeyError(
                        f"FC weight {input_name!r} not in weight_map; "
                        "did save_all_initializers_to_dram run on this model?")
                asm_instructions.append(
                    f"LOAD_M {buf}, {hex(weight_map[input_name])}, "
                    f"{rows}, {padded_cols}"
                )
            elif tensor_type == "bias":
                size = tensor_size(tensor_data["shape"])
                if input_name not in bias_map:
                    raise KeyError(
                        f"Bias {input_name!r} not in bias_map; "
                        "did save_all_initializers_to_dram run on this model?")
                asm_instructions.append(
                    f"LOAD_V {buf}, {hex(bias_map[input_name])}, {size}"
                )

        # ── Gemm / MatMul → GEMV ──────────────────────────────────────────────
        if node.op_type in ("Gemm", "MatMul"):
            in_buf  = assignments[node.input[0]]
            w_buf   = assignments[node.input[1]]
            b_buf   = assignments[node.input[2]] if len(node.input) > 2 else 0
            out_buf = assignments[node.output[0]]

            if node.input[1] in initializer_map:
                w_shape = initializer_map[node.input[1]]["shape"]
                rows, cols = (w_shape if len(w_shape) == 2
                              else (int(np.prod(w_shape[:-1])), w_shape[-1]))
            else:
                shape  = shape_map.get(node.input[1], ["?", "?"])
                rows, cols = shape[0], shape[1]

            asm_instructions.append(
                f"GEMV {out_buf}, {w_buf}, {in_buf}, {b_buf}, {rows}, {cols}"
            )
            tensor_size_map[node.output[0]] = rows

        # ── Add: absorbed into GEMV bias path ────────────────────────────────
        elif node.op_type == "Add":
            continue

        # ── Standalone Relu (small enough to address) ────────────────────────
        elif node.op_type == "Relu":
            if node.output[0] in passthrough_relu_outputs:
                # Big Relu — buffer was aliased through the allocator; emit
                # nothing, propagate size for downstream consumers.
                tensor_size_map[node.output[0]] = tensor_size_map.get(
                    node.input[0], 0)
                shape_map[node.output[0]] = shape_map.get(node.input[0], [])
                continue

            in_buf      = assignments[node.input[0]]
            out_buf     = assignments[node.output[0]]
            relu_length = tensor_size_map.get(node.input[0], 0)
            asm_instructions.append(f"RELU {out_buf}, {in_buf}, {relu_length}")
            tensor_size_map[node.output[0]] = relu_length
            shape_map[node.output[0]]       = shape_map.get(node.input[0], [])

        # ── Conv ─────────────────────────────────────────────────────────────
        elif node.op_type == "Conv":
            w_init_name = node.input[1] if len(node.input) > 1 else None
            b_init_name = node.input[2] if len(node.input) > 2 else None

            kernel_shape = get_node_attr(node, "kernel_shape", None)
            if kernel_shape is None:
                if w_init_name and w_init_name in cnn_init_map:
                    kernel_shape = cnn_init_map[w_init_name]["shape"][2:]
                else:
                    kernel_shape = [1, 1]

            strides = get_node_attr(node, "strides", [1, 1])
            pads    = get_node_attr(node, "pads",    [0, 0, 0, 0])
            kh, kw  = kernel_shape[0], kernel_shape[1]
            stride  = strides[0]
            pad     = pads[0]

            in_shape = shape_map.get(node.input[0], [])
            in_c   = int(in_shape[1]) if len(in_shape) >= 4 else 1
            fmap_h = int(in_shape[2]) if len(in_shape) >= 4 else 1
            fmap_w = int(in_shape[3]) if len(in_shape) >= 4 else 1

            # Conv weight rows=out_c, cols=in_c*kh*kw (logical/unpadded — see
            # golden_model.load_m for the FC-vs-Conv asymmetry).
            if w_init_name and w_init_name in cnn_init_map:
                out_c = cnn_init_map[w_init_name]["shape"][0]
            else:
                w_shape = shape_map.get(w_init_name, [1, 1, 1, 1])
                out_c   = int(w_shape[0])
            w_rows = out_c
            w_cols = in_c * kh * kw

            if w_init_name not in conv_weight_map:
                raise KeyError(
                    f"Conv weight {w_init_name!r} not in conv_weight_map; "
                    "did save_all_initializers_to_dram run on this model?")

            w_buf = assignments[w_init_name]
            asm_instructions.append(
                f"LOAD_M {w_buf}, {hex(conv_weight_map[w_init_name])}, "
                f"{w_rows}, {w_cols}"
            )

            # Bias load (if present)
            b_buf = 0
            if b_init_name and b_init_name in cnn_init_map:
                b_info = cnn_init_map[b_init_name]
                b_size = b_info["shape"][0]
                if b_init_name not in bias_map:
                    raise KeyError(
                        f"Conv bias {b_init_name!r} not in bias_map; "
                        "did save_all_initializers_to_dram run on this model?")
                b_buf = assignments[b_init_name]
                asm_instructions.append(
                    f"LOAD_V {b_buf}, {hex(bias_map[b_init_name])}, {b_size}"
                )

            in_buf  = assignments[node.input[0]]
            out_buf = assignments[node.output[0]]

            asm_instructions.append(
                f"CONV2D_CFG {out_buf}, {fmap_h}, {fmap_w}, {in_c}, {out_c}, "
                f"{kh}, {kw}, {stride}, {pad}"
            )

            # relu_flag is true iff the (already-pre-computed) fused-Relu
            # set contains this Conv's first consumer. fused_relu_outputs is
            # keyed by the Relu's OUTPUT name, so we need to peek at the next
            # consumer to find that name.
            relu_fused = False
            for j in range(i + 1, len(ordered_nodes)):
                nxt = ordered_nodes[j]
                if node.output[0] not in nxt.input:
                    continue
                if nxt.op_type == "Relu" and nxt.output[0] in fused_relu_outputs:
                    relu_fused = True
                break

            asm_instructions.append(
                f"CONV2D_RUN {out_buf}, {in_buf}, {w_buf}, {b_buf}, {int(relu_fused)}"
            )

            out_h = (fmap_h + 2 * pad - kh) // stride + 1
            out_w = (fmap_w + 2 * pad - kw) // stride + 1
            tensor_size_map[node.output[0]] = out_c * out_h * out_w
            shape_map[node.output[0]]       = [1, out_c, out_h, out_w]
            # If a Relu is fused into this Conv, propagate sizes/shape to it
            # so a downstream MaxPool reading the Relu's output gets the right
            # geometry without a separate Relu emission.
            if relu_fused:
                for j in range(i + 1, len(ordered_nodes)):
                    nxt = ordered_nodes[j]
                    if node.output[0] in nxt.input and nxt.op_type == "Relu":
                        tensor_size_map[nxt.output[0]] = tensor_size_map[node.output[0]]
                        shape_map[nxt.output[0]] = shape_map[node.output[0]]
                        break

        # ── MaxPool ──────────────────────────────────────────────────────────
        elif node.op_type == "MaxPool":
            kernel_shape = get_node_attr(node, "kernel_shape", [2, 2])
            strides      = get_node_attr(node, "strides",      [2, 2])
            pool_size    = kernel_shape[0]
            stride       = strides[0]

            in_buf   = assignments[node.input[0]]
            out_buf  = assignments[node.output[0]]
            in_shape = shape_map.get(node.input[0], [])
            channels = int(in_shape[1]) if len(in_shape) >= 4 else 1
            fmap_h   = int(in_shape[2]) if len(in_shape) >= 4 else 1
            fmap_w   = int(in_shape[3]) if len(in_shape) >= 4 else 1

            asm_instructions.append(
                f"MAXPOOL {out_buf}, {in_buf}, {fmap_h}, {fmap_w}, "
                f"{channels}, {pool_size}, {stride}"
            )

            out_h = (fmap_h - pool_size) // stride + 1
            out_w = (fmap_w - pool_size) // stride + 1
            tensor_size_map[node.output[0]] = channels * out_h * out_w
            shape_map[node.output[0]]       = [1, channels, out_h, out_w]

        # ── Final output STORE ────────────────────────────────────────────────
        if node.output[0] in {o.name for o in graph.output}:
            size    = tensor_size(shape_map.get(node.output[0], []))
            out_buf = assignments[node.output[0]]
            asm_instructions.append(
                f"STORE {out_buf}, {hex(dram_addresses['outputs'])}, {size}"
            )

    # ── Write assembly to file ────────────────────────────────────────────────
    with open(output_file, "w") as f:
        f.write("; Custom Architecture Assembly Code\n")
        f.write("; Generated from ONNX model\n\n")
        f.write("\n".join(asm_instructions))


if __name__ == "__main__":
    import sys
    from dram import save_all_initializers_to_dram
    model_path  = sys.argv[1] if len(sys.argv) > 1 else "mlp_model.onnx"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "assembly_code.asm"
    # Run the DRAM walker first — it owns initializer layout and returns the
    # name → address maps that generate_assembly looks up.
    dram_offsets = {
        "weights":      AcceleratorConfig.DRAM_ADDR_WEIGHTS,
        "biases":       AcceleratorConfig.DRAM_ADDR_BIASES,
        "conv_weights": AcceleratorConfig.DRAM_ADDR_CONV_WEIGHTS,
    }
    weight_map, bias_map, conv_weight_map = save_all_initializers_to_dram(
        model_path, dram_offsets)
    generate_assembly(model_path, output_file,
                      weight_map, bias_map, conv_weight_map)
    print(f"Assembly code generated and saved to {output_file}")