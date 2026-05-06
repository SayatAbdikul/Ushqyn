// Conv2D Execution Module
// Performs full matrix convolutions with BSRAM-buffered accumulation
// for mathematically correct full-tensor INT8 quantization.

import accelerator_config_pkg::VECTOR_BUFFER_WIDTH;

module conv2d_execution #(
    parameter DATA_WIDTH = 8,
    parameter TILE_ELEMS = 32,
    parameter MAX_ROWS = 1024,
    parameter MAX_COLS = 1024,
    parameter ADDR_WIDTH = 24,
    // Per-output-element int32 accumulator depth. The conv unit needs
    // total_out_pixels × out_channels entries; oversized geometries
    // would silently overflow if this is too small. A SystemVerilog
    // assert in the FSM's IDLE state catches that loudly.
    parameter ACCUM_DEPTH = 4096,
    // Set DEBUG_CONV=1 at instantiation to enable per-element trace prints.
    parameter DEBUG_CONV = 0
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0] dest_buffer_id,   // Destination buffer for results
    input logic [4:0] w_buffer_id,      // Weight matrix buffer
    input logic [4:0] x_buffer_id,      // Input vector buffer
    input logic [4:0] b_buffer_id,      // Bias vector buffer
    
    // Geometry
    input logic [5:0] fmap_h, fmap_w,
    input logic [5:0] in_channels, out_channels,
    input logic [3:0] kernel_h, kernel_w,
    input logic [2:0] stride_val, pad_val,
    // Fused-ReLU flag from CONV2D_RUN.relu_flag (instr[25]).
    // When 1, post-quantization int8 outputs are clamped to >= 0 — matching
    // compiler/golden_model.py::conv2d(apply_relu=True) semantics.
    input logic       relu_flag,
    output logic done,
    
    // Random buffer access for input feature map
    output logic [4:0] vec_random_read_buffer_id,
    output logic [$clog2(VECTOR_BUFFER_WIDTH/DATA_WIDTH)-1:0] vec_random_read_addr,
    input logic signed [DATA_WIDTH-1:0] vec_random_read_data,
    
    // Sequential matrix access for weights
    output logic mat_read_enable,
    output logic [4:0] mat_read_buffer_id,
    input logic signed [DATA_WIDTH-1:0] mat_read_tile [0:TILE_ELEMS-1],
    input logic mat_read_valid,
    
    // Sequential vector write for results
    output logic vec_write_enable,
    output logic [4:0] vec_write_buffer_id,
    output logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1]
);

    // Internal logic and memory
    logic signed [31:0] accum_ram [0:ACCUM_DEPTH-1];


    // Geometry bounds
    logic [5:0] out_h, out_w;
    logic [9:0] total_patch_size;
    logic [9:0] total_out_pixels;
    
    // Loop counters
    logic [5:0] oh, ow, oc;
    logic [9:0] patch_element;
    logic [9:0] tile_offset;
    logic [4:0] patch_tile_idx;
        
    // MAC array connection
    logic signed [DATA_WIDTH-1:0] x_patch_reg [0:TILE_ELEMS-1];
    logic signed [DATA_WIDTH-1:0] w_tile_reg  [0:TILE_ELEMS-1];
    logic signed [DATA_WIDTH*2-1:0] pe_out [0:TILE_ELEMS-1];
    logic signed [31:0] mac_sum;
    
    generate
        for (genvar i = 0; i < TILE_ELEMS; i++) begin : pe_array
            pe #(.DATA_WIDTH(DATA_WIDTH)) pe_inst (
                .clk(clk), .rst(rst),
                .w(w_tile_reg[i]), .x(x_patch_reg[i]), .y(pe_out[i])
            );
        end
    endgenerate
    
    always_comb begin
        mac_sum = 0;
        for (int i = 0; i < TILE_ELEMS; i++) begin
            logic signed [31:0] extended;
            extended = {{(32-2*DATA_WIDTH){pe_out[i][2*DATA_WIDTH-1]}}, pe_out[i]};
            if (i <= patch_tile_idx) begin
                mac_sum += extended;
            end
        end
    end
    
    // -------------------------------------------------------------
    // Address generators
    // -------------------------------------------------------------
    logic [5:0] ih, iw;
    logic [3:0] cur_kh, cur_kw;
    logic [5:0] cur_ic;
    
    assign cur_kw = patch_element % kernel_w;
    assign cur_kh = (patch_element / kernel_w) % kernel_h;
    assign cur_ic = (patch_element / kernel_w) / kernel_h;
    
    assign ih = oh * stride_val + cur_kh - pad_val;
    assign iw = ow * stride_val + cur_kw - pad_val;
    
    logic valid_patch;
    /* verilator lint_off UNSIGNED */
    assign valid_patch = (ih < fmap_h && iw < fmap_w && ih[5] == 0 && iw[5] == 0); // sign bit check for negative wrap
    /* verilator lint_on UNSIGNED */
    
    // Dynamic buffer routing for asynchronous port
    assign vec_random_read_buffer_id = (state == STREAM_QUANT || state == WAIT_QUANT || state == MAX_PASS) ? b_buffer_id : x_buffer_id;
    
    // Address routing
    logic [$clog2(VECTOR_BUFFER_WIDTH/DATA_WIDTH)-1:0] x_addr_calc, b_addr_calc;
    assign x_addr_calc = cur_ic * fmap_h * fmap_w + ih * fmap_w + iw;
    
    // Bias address tracker. Iteration order is per-channel (NCHW), so the bias
    // index is just the current output channel; advanced once per total_out_pixels
    // elements (line in MAX_PASS / STREAM_QUANT).
    logic [5:0] quant_oc;
    assign b_addr_calc = quant_oc;

    assign vec_random_read_addr = (state == STREAM_QUANT || state == WAIT_QUANT || state == MAX_PASS) ? b_addr_calc : x_addr_calc;

    // Weight buffer is row-major [out_c, in_c*kh*kw], padded to TILE_ELEMS columns.
    // Each `mat_read_enable` pulse advances the buffer's internal tile cursor
    // by TILE_ELEMS, so we walk one complete out_c row before the next.
    
    // -------------------------------------------------------------
    // Quantization Pipeline connection
    // -------------------------------------------------------------
    logic signed [31:0] max_abs_reg;
    logic scale_ready, quant_valid_in, quant_valid_out;
    logic signed [31:0] reciprocal_scale;
    logic signed [31:0] q_int32_value;
    logic signed [7:0] q_int8_value;
    
    scale_calculator scale_inst (
        .clk(clk), .reset_n(~rst),
        .max_abs(max_abs_reg), .start(state == START_SCALE),
        .reciprocal_scale(reciprocal_scale), .ready(scale_ready)
    );
    
    quantizer_pipeline quant_inst (
        .clk(clk), .reset_n(~rst),
        .int32_value(q_int32_value), .reciprocal_scale(reciprocal_scale),
        .valid_in(quant_valid_in), .int8_value(q_int8_value), .valid_out(quant_valid_out)
    );

    // -------------------------------------------------------------
    // MAIN FSM
    // -------------------------------------------------------------
    typedef enum logic [4:0] {
        IDLE,
        INIT_CONV,
        LOOP_OC_INIT,
        LOOP_OH_OW_INIT,
        TILE_LOOP_INIT,
        FETCH_X_PIXEL,
        FETCH_W_TILE,
        WAIT_W_TILE,
        WAIT_PE,
        ACCUMULATE,
        TILE_LOOP_NEXT,
        STORE_ACCUM,
        LOOP_OH_OW_NEXT,
        LOOP_OC_NEXT,
        INIT_QUANT,
        MAX_PASS,
        START_SCALE,
        WAIT_SCALE,
        STREAM_QUANT,
        WAIT_QUANT,
        WRITE_DEST,
        DONE
    } state_t;
    
    state_t state;
    
    logic [11:0] quant_ptr_in, quant_ptr_out;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            mat_read_enable <= 0;
            vec_write_enable <= 0;
            max_abs_reg <= 0;
        end else begin
            mat_read_enable <= 0;
            vec_write_enable <= 0;
            done <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        out_h <= (fmap_h + 2 * pad_val - kernel_h) / stride_val + 1;
                        out_w <= (fmap_w + 2 * pad_val - kernel_w) / stride_val + 1;
                        total_patch_size <= kernel_h * kernel_w * in_channels;
                        state <= INIT_CONV;
                    end
                end
                
                INIT_CONV: begin
                    total_out_pixels <= out_h * out_w;
                    // Loud failure if the geometry would overflow accum_ram.
                    // Silent truncation would otherwise corrupt downstream
                    // results in subtle ways.
                    if (out_h * out_w * out_channels > ACCUM_DEPTH) begin
                        $error("conv2d_execution: out_h(%0d)*out_w(%0d)*out_channels(%0d)=%0d exceeds ACCUM_DEPTH=%0d",
                               out_h, out_w, out_channels,
                               out_h * out_w * out_channels, ACCUM_DEPTH);
                    end
                    max_abs_reg <= 0;
                    oc <= 0;
                    mat_read_buffer_id <= w_buffer_id; // assert weight buffer ID to start sequential reads
                    state <= LOOP_OC_INIT;
                end
                
                LOOP_OC_INIT: begin
                    if (oc >= out_channels) begin
                        state <= INIT_QUANT;
                    end else begin
                        tile_offset <= 0;
                        state <= TILE_LOOP_INIT;
                    end
                end
                
                TILE_LOOP_INIT: begin
                    if (tile_offset >= total_patch_size) begin
                        // All weight tiles for this `oc` channel have been processed for all patches
                        // meaning this output channel is complete across the whole feature map.
                        state <= LOOP_OC_NEXT;
                    end else begin
                        // Fetch the next 32-element W tile for the given channel
                        state <= FETCH_W_TILE;
                    end
                end
                
                FETCH_W_TILE: begin
                    mat_read_enable <= 1;
                    state <= WAIT_W_TILE;
                end
                
                WAIT_W_TILE: begin
                    if (mat_read_valid) begin
                        for (int i = 0; i < TILE_ELEMS; i++) w_tile_reg[i] <= mat_read_tile[i];
                        oh <= 0;
                        ow <= 0;
                        state <= LOOP_OH_OW_INIT;
                    end
                end
                
                LOOP_OH_OW_INIT: begin
                    if (oh >= out_h) begin
                        // Entire image processed for this tile_offset weighting
                        state <= TILE_LOOP_NEXT;
                    end else begin
                        patch_tile_idx <= 0;
                        patch_element <= tile_offset;
                        state <= FETCH_X_PIXEL;
                    end
                end
                
                FETCH_X_PIXEL: begin
                    if (patch_element < total_patch_size) begin
                        if (valid_patch) x_patch_reg[patch_tile_idx] <= vec_random_read_data;
                        else x_patch_reg[patch_tile_idx] <= 0;
                    end else begin
                        x_patch_reg[patch_tile_idx] <= 0;
                    end
                    
                    if (patch_tile_idx == TILE_ELEMS - 1 || patch_element == total_patch_size - 1) begin
                        state <= WAIT_PE;
                    end else begin
                        patch_tile_idx <= patch_tile_idx + 1;
                        patch_element <= patch_element + 1;
                    end
                end
                
                WAIT_PE: begin
                    // Wait one cycle for PE registered outputs to settle
                    state <= ACCUMULATE;
                end
                
                ACCUMULATE: begin
                    // Accumulate current mac array directly into RAM to allow tile_offset scaling
                    logic [11:0] linear_idx;
                    linear_idx = oc * total_out_pixels + oh * out_w + ow;
                    
                    if (DEBUG_CONV && oh == 9 && ow == 18 && oc == 2 && tile_offset == 0) begin
                        $display("[RTL_DBG] OH=9 OW=18: X_TILE=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                            x_patch_reg[0], x_patch_reg[1], x_patch_reg[2], x_patch_reg[3],
                            x_patch_reg[4], x_patch_reg[5], x_patch_reg[6], x_patch_reg[7]);
                        $display("[RTL_DBG] OH=9 OW=18: X_TILE_8=%0d", x_patch_reg[8]);
                        $display("[RTL_DBG] OH=9 OW=18: W_TILE=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                            $signed(w_tile_reg[0]), $signed(w_tile_reg[1]), $signed(w_tile_reg[2]), $signed(w_tile_reg[3]),
                            $signed(w_tile_reg[4]), $signed(w_tile_reg[5]), $signed(w_tile_reg[6]), $signed(w_tile_reg[7]));
                        $display("[RTL_DBG] OH=9 OW=18: W_TILE_8=%0d", w_tile_reg[8]);
                        $display("[RTL_DBG] OH=9 OW=18: MAC_SUM so far = %0d", mac_sum);
                    end

                    if (tile_offset == 0) begin
                        accum_ram[linear_idx] <= mac_sum;
                    end else begin
                        accum_ram[linear_idx] <= accum_ram[linear_idx] + mac_sum;
                        if (DEBUG_CONV && oh == 0 && ow == 0 && oc == 0) begin
                            $display("[RTL_DBG] OH=0 OW=0: MAC_SUM addition = %0d (Tile offset = %0d)", mac_sum, tile_offset);
                        end
                    end
                    
                    state <= STORE_ACCUM; // Reused simply as a routing jump name
                end
                
                STORE_ACCUM: begin
                    logic [11:0] linear_idx;
                    linear_idx = oc * total_out_pixels + oh * out_w + ow;
                    
                    // max_abs tracking moved to STREAM_QUANT (post-bias, matching golden model)
                    // Nothing to track here; just advance to the next spatial position.
                    
                    state <= LOOP_OH_OW_NEXT;
                end
                
                LOOP_OH_OW_NEXT: begin
                    if (ow == out_w - 1) begin
                        ow <= 0;
                        oh <= oh + 1;
                    end else begin
                        ow <= ow + 1;
                    end
                    state <= LOOP_OH_OW_INIT;
                end
                
                TILE_LOOP_NEXT: begin
                    tile_offset <= tile_offset + TILE_ELEMS;
                    state <= TILE_LOOP_INIT;
                end
                
                LOOP_OC_NEXT: begin
                    oc <= oc + 1;
                    state <= LOOP_OC_INIT;
                end
                
                INIT_QUANT: begin
                    max_abs_reg <= 0;
                    quant_ptr_in <= 0;
                    quant_oc <= 0;
                    state <= MAX_PASS;
                end
                
                MAX_PASS: begin
                    if (quant_ptr_in < total_out_pixels * out_channels) begin
                        // Bias: bias index is the current output channel (oc-major NCHW)
                        logic signed [31:0] b_ext;
                        logic signed [31:0] biased_val;
                        b_ext = {{(24){vec_random_read_data[7]}}, vec_random_read_data};
                        biased_val = accum_ram[quant_ptr_in] + b_ext;

                        if (DEBUG_CONV && (biased_val == 35720 || biased_val == -35720)) begin
                            $display("[RTL_DBG] FOUND MAX_ABS 35720 at quant_ptr_in=%0d", quant_ptr_in);
                        end

                        // Update max_abs_reg with the absolute value of the biased result
                        if (biased_val > max_abs_reg) max_abs_reg <= biased_val;
                        else if (-biased_val > max_abs_reg) max_abs_reg <= -biased_val;
                        
                        quant_ptr_in <= quant_ptr_in + 1;
                        
                        if ((quant_ptr_in + 1) % total_out_pixels == 0) begin
                            if (quant_oc >= out_channels - 1) quant_oc <= 0;
                            else quant_oc <= quant_oc + 1;
                        end
                    end else begin
                        state <= START_SCALE;
                    end
                end
                
                START_SCALE: begin
                    state <= WAIT_SCALE;
                end
                
                WAIT_SCALE: begin
                    // Wait until scale_calculator signals ready (it's pipelined).
                    if (scale_ready) begin
                        if (DEBUG_CONV) $display("[RTL_CONV] MAX_PASS finished, final max_abs_reg = %0d", max_abs_reg);
                        quant_ptr_in <= 0;
                        quant_ptr_out <= 0;
                        quant_oc <= 0;
                        state <= STREAM_QUANT;
                    end
                end
                
                STREAM_QUANT: begin
                    quant_valid_in <= 0;
                    
                    if (quant_ptr_in < total_out_pixels * out_channels) begin
                        // Bias: bias index is the current output channel (oc-major NCHW)
                        logic signed [31:0] b_ext;
                        b_ext = {{(24){vec_random_read_data[7]}}, vec_random_read_data};
                        q_int32_value <= accum_ram[quant_ptr_in] + b_ext;
                        
                        quant_valid_in <= 1;
                        quant_ptr_in   <= quant_ptr_in + 1;
                        
                        // quant_oc = which output channel are we quantizing right now?
                        // NCHW layout: channel changes every total_out_pixels elements
                        if ((quant_ptr_in + 1) % total_out_pixels == 0) begin
                            if (quant_oc >= out_channels - 1) quant_oc <= 0;
                            else quant_oc <= quant_oc + 1;
                        end
                    end
                    
                    if (quant_valid_out) begin
                        // Apply fused ReLU after quantization, matching the
                        // golden model's `if apply_relu: quantized = np.maximum(quantized, 0)`.
                        vec_write_tile[quant_ptr_out % TILE_ELEMS] <=
                            (relu_flag && q_int8_value < 0) ? 8'sd0 : q_int8_value;
                        quant_ptr_out <= quant_ptr_out + 1;
                        
                        if ((quant_ptr_out % TILE_ELEMS) == TILE_ELEMS - 1 || quant_ptr_out == (total_out_pixels * out_channels) - 1) begin
                            vec_write_enable <= 1;
                            vec_write_buffer_id <= dest_buffer_id;
                        end
                    end
                    
                    if (quant_ptr_out >= total_out_pixels * out_channels) begin
                        state <= DONE;
                    end
                end
                
                DONE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
