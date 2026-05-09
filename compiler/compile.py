"""Lower an ONNX graph to the accelerator's assembly language.

Walks the graph in topological order and emits one of the opcodes
defined in isa_spec.OPCODES (LOAD_V/LOAD_M/STORE/GEMV/RELU/CONV2D_CFG/
CONV2D_RUN/MAXPOOL) per supported op. Constant-folded ops (Reshape,
Flatten, BatchNormalization) carry buffer mappings forward without
emitting instructions.

DRAM addressing: post-P2 (2026-05-09) `generate_assembly` looks up every
initializer's DRAM address by name in the `(weight_map, bias_map,
conv_weight_map)` returned by `dram.save_all_initializers_to_dram`. The
old shadow counters (`bias_counter`, `weight_counter`, `conv_weight_counter`)
that had to advance in lockstep with the DRAM walker are gone — there is
now one walker, in `dram.py`, and the compiler is its consumer.

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
    initializer_map  = build_initializer_map(graph)       # Used for MLP weights (2-D)
    cnn_init_map     = build_initializer_map_cnn(graph)   # Used for conv weights (4-D)
    ordered_nodes    = topological_sort(graph)

    # Buffer ID layout
    # 0        : scratch / flatten passthrough
    # 1-2      : weight buffers (ping-pong for FC / conv weights)
    # 3-4      : bias buffers
    # 5-6      : GEMV output buffers
    # 7-8      : RELU output buffers
    # 9        : fixed input buffer (LOAD_V of the original input vector)
    # 10-11    : conv output feature-map buffers (ping-pong)
    # 12-13    : pool output buffers (ping-pong)
    mat_buf           = 1
    bias_vector_buf   = 3
    gemv_buf          = 5
    relu_buf          = 7
    input_buf         = 9   # always 9 for the primary input tensor
    conv_out_buf      = 10
    pool_out_buf      = 12
    tensor_buffer_map = {}
    tensor_size_map   = {}   # Track output element counts for RELU length
    asm_instructions  = []
    skip_nodes        = set()

    # Only inputs / outputs need address constants here — every initializer
    # address is looked up by name in the maps from save_all_initializers_to_dram.
    dram_addresses = {
        "inputs":  AcceleratorConfig.DRAM_ADDR_INPUTS,
        "outputs": AcceleratorConfig.DRAM_ADDR_OUTPUTS,
    }
    
    # ── Emit LOAD_V for the model's primary input tensor ──────────────────────
    # This is always the first graph input (e.g. the image tensor for CNNs).
    # For MLP models the Reshape node handler emits this for its input, but CNN
    # models start directly with a Conv node and need this prolog LOAD_V.
    primary_input_name = graph.input[0].name
    primary_input_shape = shape_map.get(primary_input_name, [])
    input_size = int(np.prod(primary_input_shape[1:])) if len(primary_input_shape) > 1 else 1
    asm_instructions.append(
        f"LOAD_V {input_buf}, {hex(dram_addresses['inputs'])}, {input_size}"
    )
    tensor_buffer_map[primary_input_name] = input_buf

    # ── Process each node ────────────────────────────────────────────────────

    for i, node in enumerate(ordered_nodes):
        if node.output[0] in skip_nodes:
            continue

        # ── Reshape: remap buffer without new instructions ────────────────────
        if node.op_type == "Reshape":
            input_name  = node.input[0]
            output_name = node.output[0]
            if input_name in tensor_buffer_map:
                tensor_buffer_map[output_name] = tensor_buffer_map[input_name]
            else:
                tensor_buffer_map[input_name]  = 0
                size = tensor_size(shape_map.get(input_name, []))
                asm_instructions.append(f"LOAD_V {input_buf}, {hex(dram_addresses['inputs'])}, {size}")
                tensor_buffer_map[output_name] = 0
            continue

        # ── Flatten: no instruction, just pass the buffer through ─────────────
        if node.op_type == "Flatten":
            src = node.input[0]
            dst = node.output[0]
            if src in tensor_buffer_map:
                tensor_buffer_map[dst] = tensor_buffer_map[src]
                if src in tensor_size_map:
                    tensor_size_map[dst] = tensor_size_map[src]
            continue

        # ── BatchNormalization: fold into trailing buffer (skip) ──────────────
        # For inference with pre-trained weights, BN params are folded into
        # the preceding conv weights during model export.  If they are still
        # present as separate nodes, we skip them; the compiler expects the
        # exporter to have fused or removed them beforehand.
        if node.op_type == "BatchNormalization":
            src = node.input[0]
            dst = node.output[0]
            if src in tensor_buffer_map:
                tensor_buffer_map[dst] = tensor_buffer_map[src]
                if src in tensor_size_map:
                    tensor_size_map[dst] = tensor_size_map[src]
            skip_nodes.add(node.output[0])
            continue

        # ── Process initialisers (weights / biases) for this node ─────────────
        for idx, input_name in enumerate(node.input):
            # Skip Conv weights AND bias; both are handled in the Conv block below.
            # idx==1: weight (LOAD_M emitted in Conv block)
            # idx==2: bias   (LOAD_V emitted in Conv block)
            # Letting either fall through here would emit a *duplicate* LOAD_V/M
            # for the same initializer (the historical pre-Patch-B regression).
            if node.op_type == "Conv" and idx in (1, 2):
                continue
                
            if input_name in initializer_map and input_name not in tensor_buffer_map:
                tensor_data = initializer_map[input_name]
                tensor_type = tensor_data["type"]

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

                    tensor_buffer_map[input_name] = mat_buf
                    asm_instructions.append(
                        f"LOAD_M {mat_buf}, {hex(weight_map[input_name])}, {rows}, {padded_cols}"
                    )
                    mat_buf = 2 if mat_buf == 1 else 1

                elif tensor_type == "bias":
                    size = tensor_size(tensor_data["shape"])
                    if input_name not in bias_map:
                        raise KeyError(
                            f"Bias {input_name!r} not in bias_map; "
                            "did save_all_initializers_to_dram run on this model?")
                    tensor_buffer_map[input_name] = bias_vector_buf
                    asm_instructions.append(
                        f"LOAD_V {bias_vector_buf}, {hex(bias_map[input_name])}, {size}"
                    )
                    bias_vector_buf = 4 if bias_vector_buf == 3 else 3

        # ── Gemm / MatMul → GEMV ──────────────────────────────────────────────
        if node.op_type in ["Gemm", "MatMul"]:
            in_buf   = tensor_buffer_map.get(node.input[0], "?")
            in_buf   = 9 if in_buf == 0 else in_buf
            w_buf    = tensor_buffer_map.get(node.input[1], "?")
            b_buf    = 4 if bias_vector_buf == 3 else 3

            if node.input[1] in initializer_map:
                w_shape = initializer_map[node.input[1]]["shape"]
                rows, cols = (w_shape if len(w_shape) == 2
                              else (int(np.prod(w_shape[:-1])), w_shape[-1]))
            else:
                shape  = shape_map.get(node.input[1], ["?", "?"])
                rows, cols = shape[0], shape[1]

            asm_instructions.append(f"GEMV {gemv_buf}, {w_buf}, {in_buf}, {b_buf}, {rows}, {cols}")
            tensor_buffer_map[node.output[0]] = gemv_buf
            tensor_size_map[node.output[0]]   = rows
            gemv_buf = 6 if gemv_buf == 5 else 5

        # ── Add: absorbed into GEMV bias path ────────────────────────────────
        elif node.op_type == "Add":
            continue

        # ── Relu ─────────────────────────────────────────────────────────────
        elif node.op_type == "Relu":
            in_buf      = tensor_buffer_map.get(node.input[0], "?")
            relu_length = tensor_size_map.get(node.input[0], 0)
            
            # The standalone RELU instruction has a 10-bit length limit (max 1023)
            # CNN ReLUs should be fused into CONV2D_RUN. If we see a large Relu,
            # we assume it was fused and just pass through the buffer mappings.
            if relu_length <= 1023:
                asm_instructions.append(f"RELU {relu_buf}, {in_buf}, {relu_length}")
                tensor_buffer_map[node.output[0]] = relu_buf
                tensor_size_map[node.output[0]]   = relu_length
                shape_map[node.output[0]]         = shape_map.get(node.input[0], [])
                relu_buf = 8 if relu_buf == 7 else 7
            else:
                # Fused Relu passthrough
                tensor_buffer_map[node.output[0]] = in_buf
                tensor_size_map[node.output[0]]   = relu_length
                shape_map[node.output[0]]         = shape_map.get(node.input[0], [])

        # ── Conv ─────────────────────────────────────────────────────────────
        elif node.op_type == "Conv":
            # Resolve weight from cnn_init_map (4-D: [out_C, in_C, kH, kW])
            w_init_name = node.input[1] if len(node.input) > 1 else None
            b_init_name = node.input[2] if len(node.input) > 2 else None

            # Read conv attributes
            kernel_shape = get_node_attr(node, "kernel_shape", None)
            if kernel_shape is None:
                if w_init_name and w_init_name in cnn_init_map:
                    kernel_shape = cnn_init_map[w_init_name]["shape"][2:]
                else:
                    # Fallback if both attribute and initializer are missing
                    kernel_shape = [1, 1]
                    
            strides      = get_node_attr(node, "strides",      [1, 1])
            pads         = get_node_attr(node, "pads",         [0, 0, 0, 0])
            kh, kw       = kernel_shape[0], kernel_shape[1]
            stride       = strides[0]          # assume square stride
            pad          = pads[0]             # assume symmetric padding

            # Resolve input feature-map shape [N, in_C, H, W]
            in_shape    = shape_map.get(node.input[0], [])
            in_c        = int(in_shape[1]) if len(in_shape) >= 4 else 1
            fmap_h      = int(in_shape[2]) if len(in_shape) >= 4 else 1
            fmap_w      = int(in_shape[3]) if len(in_shape) >= 4 else 1

            # Resolve conv weight shape (rows=out_c, cols=in_c*kh*kw); cols are
            # logical/unpadded — load_m emits with unpadded cols and load_m.sv
            # drops the padding (see golden_model.load_m docstring for the
            # FC-vs-Conv asymmetry). DRAM storage is padded; the address comes
            # from conv_weight_map which is computed by the same walker that
            # wrote the padded layout, so the asymmetry stays internal to load_m.
            if w_init_name and w_init_name in cnn_init_map:
                w_info = cnn_init_map[w_init_name]
                out_c  = w_info["shape"][0]
            else:
                w_shape = shape_map.get(w_init_name, [1, 1, 1, 1])
                out_c   = int(w_shape[0])
            w_rows = out_c
            w_cols = in_c * kh * kw

            if w_init_name not in conv_weight_map:
                raise KeyError(
                    f"Conv weight {w_init_name!r} not in conv_weight_map; "
                    "did save_all_initializers_to_dram run on this model?")
            w_addr = conv_weight_map[w_init_name]

            # ---- Emit weight load (LOAD_M with rows=out_c, cols=in_c*kh*kw) ----
            tensor_buffer_map[w_init_name] = mat_buf
            asm_instructions.append(
                f"LOAD_M {mat_buf}, {hex(w_addr)}, {w_rows}, {w_cols}"
            )
            cur_w_buf = mat_buf
            mat_buf = 2 if mat_buf == 1 else 1

            # ---- Emit bias load (if present) ----
            cur_b_buf = bias_vector_buf
            if b_init_name and b_init_name in cnn_init_map:
                b_info = cnn_init_map[b_init_name]
                b_size = b_info["shape"][0]
                if b_init_name not in bias_map:
                    raise KeyError(
                        f"Conv bias {b_init_name!r} not in bias_map; "
                        "did save_all_initializers_to_dram run on this model?")
                b_addr = bias_map[b_init_name]
                tensor_buffer_map[b_init_name] = bias_vector_buf
                asm_instructions.append(
                    f"LOAD_V {bias_vector_buf}, {hex(b_addr)}, {b_size}"
                )
                cur_b_buf = bias_vector_buf
                bias_vector_buf = 4 if bias_vector_buf == 3 else 3

            # Determine input buffer
            in_buf = tensor_buffer_map.get(node.input[0], input_buf)
            if in_buf == 0:
                in_buf = input_buf

            # ---- Emit CONV2D_CFG ----
            asm_instructions.append(
                f"CONV2D_CFG {conv_out_buf}, {fmap_h}, {fmap_w}, {in_c}, {out_c}, "
                f"{kh}, {kw}, {stride}, {pad}"
            )

            # ---- Emit CONV2D_RUN (with relu_flag=1 if next op is Relu) ----
            # Peek ahead: if the next node that *consumes this conv's output*
            # is a Relu, fuse the activation into this instruction.
            #
            # Why the consumer filter: topological_sort can interleave nodes
            # that don't depend on this conv (e.g. a Constant operand for a
            # later Reshape, or parallel branches) between this conv and its
            # actual Relu. A naive "first non-skip node" lookahead misses the
            # fusion in those cases — and because compile.py later treats a
            # standalone Relu with length > 1023 as a fused-passthrough, the
            # activation is then silently dropped. Skipping non-consumers
            # closes both halves of that bug.
            relu_fused = False
            for j in range(i + 1, len(ordered_nodes)):
                nxt = ordered_nodes[j]
                if nxt.output[0] in skip_nodes:
                    continue
                if node.output[0] not in nxt.input:
                    continue                       # not a consumer of this conv
                if nxt.op_type == "Relu":
                    relu_fused = True
                    skip_nodes.add(nxt.output[0])
                    tensor_buffer_map[nxt.output[0]] = conv_out_buf
                    out_h = (fmap_h + 2 * pad - kh) // stride + 1
                    out_w = (fmap_w + 2 * pad - kw) // stride + 1
                    tensor_size_map[nxt.output[0]] = out_c * out_h * out_w
                    shape_map[nxt.output[0]] = [1, out_c, out_h, out_w]
                break

            asm_instructions.append(
                f"CONV2D_RUN {conv_out_buf}, {in_buf}, {cur_w_buf}, {cur_b_buf}, {int(relu_fused)}"
            )

            out_h = (fmap_h + 2 * pad - kh) // stride + 1
            out_w = (fmap_w + 2 * pad - kw) // stride + 1
            tensor_buffer_map[node.output[0]] = conv_out_buf
            tensor_size_map[node.output[0]]   = out_c * out_h * out_w
            shape_map[node.output[0]]         = [1, out_c, out_h, out_w]
            conv_out_buf = 11 if conv_out_buf == 10 else 10

        # ── MaxPool ──────────────────────────────────────────────────────────
        elif node.op_type == "MaxPool":
            kernel_shape = get_node_attr(node, "kernel_shape", [2, 2])
            strides      = get_node_attr(node, "strides",      [2, 2])
            pool_size    = kernel_shape[0]
            stride        = strides[0]

            in_buf    = tensor_buffer_map.get(node.input[0], "?")
            in_shape  = shape_map.get(node.input[0], [])
            channels  = int(in_shape[1]) if len(in_shape) >= 4 else 1
            fmap_h    = int(in_shape[2]) if len(in_shape) >= 4 else 1
            fmap_w    = int(in_shape[3]) if len(in_shape) >= 4 else 1

            asm_instructions.append(
                f"MAXPOOL {pool_out_buf}, {in_buf}, {fmap_h}, {fmap_w}, {channels}, {pool_size}, {stride}"
            )

            out_h = (fmap_h - pool_size) // stride + 1
            out_w = (fmap_w - pool_size) // stride + 1
            tensor_buffer_map[node.output[0]] = pool_out_buf
            tensor_size_map[node.output[0]]   = channels * out_h * out_w
            shape_map[node.output[0]]         = [1, channels, out_h, out_w]
            pool_out_buf = 13 if pool_out_buf == 12 else 12

        # ── Final output STORE ────────────────────────────────────────────────
        if node.output[0] in [o.name for o in graph.output]:
            size    = tensor_size(shape_map.get(node.output[0], []))
            out_buf = tensor_buffer_map.get(node.output[0], "?")
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