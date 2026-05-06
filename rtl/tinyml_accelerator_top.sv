// Top Module - tinyML Accelerator
// Fetches an instruction, decodes it, then executes it using the modular execution unit.
// 
// Architecture:
// - Fetch Unit: Reads 64-bit instructions from memory
// - Instruction Decoder: Decodes instruction fields (opcode, operands, addresses)
// - Modular Execution Unit: Executes operations with separated execution modules
//   * Buffer Controller: Manages vector and matrix buffer files
//   * Load Execution: Handles LOAD_V and LOAD_M from DRAM to buffers
//   * GEMV Execution: Matrix-vector multiplication with tiled computation
//   * ReLU Execution: Applies ReLU activation function
//   * Store Execution: Writes buffer data back to DRAM
// 
// FSM: IDLE -> FETCH -> WAIT_FETCH -> DECODE -> EXECUTE_START -> EXECUTE_WAIT -> DONE
// 
// The module pulses 'done' for one cycle when an instruction completes execution.
// Connect 'start' high for one cycle to begin processing a new instruction.
//
import accelerator_config_pkg::*;

module tinyml_accelerator_top #(
    parameter DATA_WIDTH = accelerator_config_pkg::DATA_WIDTH,
    parameter TILE_WIDTH = accelerator_config_pkg::TILE_WIDTH,
    parameter ADDR_WIDTH = accelerator_config_pkg::ADDR_WIDTH,
    parameter MAX_ROWS   = accelerator_config_pkg::MAX_ROWS,
    parameter MAX_COLS   = accelerator_config_pkg::MAX_COLS,
    parameter OUT_N      = accelerator_config_pkg::OUT_N,
    parameter INSTR_WIDTH= 64,
    parameter HEX_FILE   = "/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/dram.hex"
)(
    input  logic clk,
    input  logic rst,            // Active high
    input  logic start,          // Start a single instruction cycle
    output logic signed [DATA_WIDTH-1:0] y [0:OUT_N-1], // Truncated execution results
    output logic done            // High for 1 cycle when instruction fully executed
);
    // ------------------------------------------------------------
    // State Machine for top-level control
    // ------------------------------------------------------------
    typedef enum logic [2:0] { T_IDLE, T_FETCH, T_WAIT_FETCH, T_DECODE, T_EXECUTE_START, T_EXECUTE_WAIT, T_DONE } top_state_t;
    top_state_t t_state, t_state_n;

    // ------------------------------------------------------------
    // Shared Memory System
    // ------------------------------------------------------------
    logic [ADDR_WIDTH-1:0] mem_addr;
    logic [DATA_WIDTH-1:0] mem_rdata;
    logic [DATA_WIDTH-1:0] mem_wdata;
    logic                  mem_we;
    logic                  mem_req; // For info/arbitration logic if needed
    
    // Unified Memory Instance
    simple_memory #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .HEX_FILE  (HEX_FILE)
    ) main_memory (
        .clk(clk),
        .we(mem_we),
        .addr(mem_addr),
        .din(mem_wdata),
        .dout(mem_rdata),
        .dump(1'b0) // Control via top-level param/signal if needed
    );

    // ------------------------------------------------------------
    // Fetch Unit
    // ------------------------------------------------------------
    logic fetch_en;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [ADDR_WIDTH-1:0] pc; // will be needed for debugging and maybe branching in future
    /* verilator lint_on UNUSEDSIGNAL */
    logic [INSTR_WIDTH-1:0] instr;
    logic fetch_done;
    logic store_instr;
    logic fetch_mem_req;
    logic [ADDR_WIDTH-1:0] fetch_mem_addr;

    fetch_unit #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .INSTR_WIDTH(INSTR_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) fetch_u (
        .clk       (clk),
        .rst_n     (~rst),          // fetch_unit uses active-low reset
        .fetch_en_i(fetch_en),
        .pc_o      (pc),
        .instr_o   (instr),
        .done      (fetch_done),
        // Memory Interface
        .mem_req   (fetch_mem_req),
        .mem_addr  (fetch_mem_addr),
        .mem_rdata (mem_rdata), // Shared read data
        .mem_valid (1'b1)       // Assume always valid/ready for now (BSRAM latency handled by state machine)
    );

    // ------------------------------------------------------------
    // Decoder
    // ------------------------------------------------------------
    logic [4:0]  d_opcode;
    logic [4:0]  d_dest;
    logic [9:0]  d_length_or_cols;
    logic [9:0]  d_rows;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [23:0] d_addr;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [4:0]  d_b, d_x, d_w;

    logic [5:0]  d_fmap_h, d_fmap_w, d_in_channels, d_out_channels;
    logic [3:0]  d_kernel_h, d_kernel_w;
    logic [2:0]  d_stride_val, d_pad_val, d_pool_size;
    logic        d_relu_flag;

    i_decoder decoder_u (
        .instr (instr),
        .opcode(d_opcode),
        .dest  (d_dest),
        .length_or_cols(d_length_or_cols),
        .rows  (d_rows),
        .addr  (d_addr),
        .b     (d_b),
        .x     (d_x),
        .w     (d_w),
        .fmap_h(d_fmap_h),
        .fmap_w(d_fmap_w),
        .in_channels(d_in_channels),
        .out_channels(d_out_channels),
        .kernel_h(d_kernel_h),
        .kernel_w(d_kernel_w),
        .stride_val(d_stride_val),
        .pad_val(d_pad_val),
        .pool_size(d_pool_size),
        .relu_flag(d_relu_flag)
    );

    // Latched copies for stable drive during execution
    logic [4:0]  ex_opcode;
    logic [4:0]  ex_dest;
    logic [9:0]  ex_length_or_cols;
    logic [9:0]  ex_rows;
    logic [ADDR_WIDTH-1:0] ex_addr;
    logic [4:0]  ex_b, ex_x, ex_w;
    
    // CNN configuration latches
    logic [5:0]  ex_fmap_h, ex_fmap_w, ex_in_channels, ex_out_channels;
    logic [3:0]  ex_kernel_h, ex_kernel_w;
    logic [2:0]  ex_stride_val, ex_pad_val, ex_pool_size;
    logic        ex_relu_flag;

    // ------------------------------------------------------------
    // Execution Unit
    // ------------------------------------------------------------
    logic exec_start;
    logic exec_done;
    logic signed [DATA_WIDTH-1:0] exec_result [0:MAX_ROWS-1];

    logic exec_mem_req;
    logic exec_mem_we;
    logic [ADDR_WIDTH-1:0] exec_mem_addr;
    logic [DATA_WIDTH-1:0] exec_mem_wdata;

    modular_execution_unit #(
        .DATA_WIDTH (DATA_WIDTH),
        .TILE_WIDTH (TILE_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .MAX_ROWS   (MAX_ROWS),
        .MAX_COLS   (MAX_COLS)
    ) execution_u (
        .clk          (clk),
        .rst          (rst),
        .start        (exec_start),
        .opcode       (ex_opcode),
        .dest         (ex_dest),
        .length_or_cols(ex_length_or_cols),
        .rows         (ex_rows),
        .addr         (ex_addr),
        .b_id         (ex_b),
        .x_id         (ex_x),
        .w_id         (ex_w),
        .fmap_h       (ex_fmap_h),
        .fmap_w       (ex_fmap_w),
        .in_channels  (ex_in_channels),
        .out_channels (ex_out_channels),
        .kernel_h     (ex_kernel_h),
        .kernel_w     (ex_kernel_w),
        .stride_val   (ex_stride_val),
        .pad_val      (ex_pad_val),
        .pool_size    (ex_pool_size),
        .relu_flag    (ex_relu_flag),
        .result       (exec_result),
        .done         (exec_done),
        // Memory Interface
        .mem_req      (exec_mem_req),
        .mem_we       (exec_mem_we),
        .mem_addr     (exec_mem_addr),
        .mem_wdata    (exec_mem_wdata),
        .mem_rdata    (mem_rdata), // Shared read data
        .mem_valid    (1'b1)       // Assume simple_memory is always ready
    );

    // ------------------------------------------------------------
    // Top-level FSM next-state logic
    // ------------------------------------------------------------
    // Track if we hit a zero instruction (program end)
    /* verilator lint_off UNUSEDSIGNAL */
    logic program_done;
    /* verilator lint_on UNUSEDSIGNAL */
    
    // Memory Arbitration Logic
    // Since T_FETCH/T_WAIT_FETCH and T_EXECUTE_* are mutually exclusive in this FSM,
    // we can use a simple priority mux or state-based mux.
    always_comb begin
        if (t_state == T_FETCH || t_state == T_WAIT_FETCH) begin
            // Fetch Unit has access
            mem_addr  = fetch_mem_addr;
            mem_req   = fetch_mem_req; // Might need to drive this based on logic, but fetch_u provides it
            mem_we    = 1'b0; // Fetch is read-only
            mem_wdata = '0;
        end else begin
            // Execution Unit has access (Load/Store)
            mem_addr  = exec_mem_addr;
            mem_req   = exec_mem_req;
            mem_we    = exec_mem_we;
            mem_wdata = exec_mem_wdata;
        end
    end
    
    always_comb begin
        t_state_n = t_state;
        fetch_en  = 1'b0;
        exec_start = 1'b0;
        done = 1'b0;
        program_done = 1'b0;
        
        case (t_state)
            T_IDLE: begin
                // Wait for start signal before beginning execution
                if (start) begin
                    t_state_n = T_FETCH;
                end
            end
            T_FETCH: begin
                fetch_en   = 1'b1;          // Pulse enable to begin fetch
                t_state_n  = T_WAIT_FETCH;
            end
            T_WAIT_FETCH: begin
                if (fetch_done) begin
                    if (instr == '0) begin
                        // Zero instruction = program end, signal done and go to IDLE
                        program_done = 1'b1;
                        done = 1'b1;
                        t_state_n = T_IDLE;
                    end else begin
                        t_state_n = T_DECODE;
                    end
                end
            end
            T_DECODE: begin
                // Latch decoded fields next cycle
                if (d_opcode == 5'h06) begin
                    t_state_n = T_DONE; // Configuration instruction: setup complete
                end else begin
                    t_state_n = T_EXECUTE_START;
                end
            end
            T_EXECUTE_START: begin
                exec_start = 1'b1;          // One-cycle start pulse
                t_state_n  = T_EXECUTE_WAIT;
            end
            T_EXECUTE_WAIT: begin
                if (exec_done) begin
                    t_state_n = T_DONE;
                end
            end
            T_DONE: begin
                // Instruction complete, automatically fetch next instruction
                // (don't pulse done here - only pulse when program ends)
                t_state_n = T_FETCH;
            end
            default: t_state_n = T_IDLE;
        endcase
    end

    // ------------------------------------------------------------
    // Sequential section: state & latching decoded fields
    // ------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            t_state <= T_IDLE;
            ex_opcode <= '0; ex_dest <= '0; ex_length_or_cols <= '0; ex_rows <= '0; ex_addr <= '0; ex_b <= '0; ex_x <= '0; ex_w <= '0;
            ex_fmap_h <= 0; ex_fmap_w <= 0; ex_in_channels <= 0; ex_out_channels <= 0;
            ex_kernel_h <= 0; ex_kernel_w <= 0; ex_stride_val <= 0; ex_pad_val <= 0; ex_pool_size <= 0;
            ex_relu_flag <= 0;
            store_instr <= 0;
        end else begin
            t_state <= t_state_n;
            if (t_state == T_DECODE) begin
                ex_opcode        <= d_opcode;
                ex_dest          <= d_dest;
                ex_length_or_cols<= d_length_or_cols;
                ex_rows          <= d_rows;
                ex_addr          <= d_addr;
                ex_b             <= d_b;
                ex_x             <= d_x;
                ex_w             <= d_w;
                // Latch relu_flag only on CONV2D_RUN; default to 0 otherwise so a
                // stale flag from a prior CONV2D_RUN doesn't bleed into anything.
                ex_relu_flag     <= (d_opcode == 5'h07) ? d_relu_flag : 1'b0;
                // Configuration latches (updated exclusively when provided)
                if (d_opcode == 5'h06) begin
                    ex_fmap_h <= d_fmap_h;
                    ex_fmap_w <= d_fmap_w;
                    ex_in_channels <= d_in_channels;
                    ex_out_channels <= d_out_channels;
                    ex_kernel_h <= d_kernel_h;
                    ex_kernel_w <= d_kernel_w;
                    ex_stride_val <= d_stride_val;
                    ex_pad_val <= d_pad_val;
                end else if (d_opcode == 5'h08) begin
                    // MaxPool configures its own pool size and stride over 0x08 payload
                    ex_pool_size <= d_pool_size;
                    ex_stride_val <= d_stride_val;
                    ex_fmap_h <= d_fmap_h;
                    ex_fmap_w <= d_fmap_w;
                    ex_in_channels <= d_in_channels;
                end

                // Track if this is a store instruction for potential future use
                store_instr <= (d_opcode == 5'b00011);
            end
        end
    end

    // ------------------------------------------------------------
    // Output assignment (truncate execution results to OUT_N)
    // ------------------------------------------------------------
    generate
        genvar gi;
        for (gi = 0; gi < OUT_N; gi++) begin : OUT_COPY
            always_comb begin
                if (gi < MAX_ROWS && !store_instr) y[gi] = exec_result[gi]; else y[gi] = '1;
            end
        end
    endgenerate

endmodule
