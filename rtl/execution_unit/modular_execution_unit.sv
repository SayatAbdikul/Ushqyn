// Modular Execution Unit - Top Coordinator
// Refactored execution unit with separated concerns for better maintainability
// 
// Architecture:
// - Buffer Controller: Manages all buffer file I/O
// - Load Execution: Handles LOAD_V and LOAD_M operations
// - GEMV Execution: Orchestrates matrix-vector multiplication
// - ReLU Execution: Applies activation functions
// - Store Execution: Handles memory writes (placeholder)
//
// This module provides the same interface as the original execution_unit
// but with cleaner separation of concerns and fixes for critical bugs:
// 1. ReLU now correctly reads from source buffer (not dest)
// 2. GEMV writes results back to buffer for subsequent operations
// 3. Proper length handling for ReLU operations

module modular_execution_unit #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter ADDR_WIDTH = 24,
    parameter MAX_ROWS = 1024,
    parameter MAX_COLS = 1024
)(
    input logic clk,
    input logic rst,
    
    // Control interface (same as original execution_unit)
    input logic start,
    input logic [4:0] opcode,
    input logic [4:0] dest,
    input logic [9:0] length_or_cols,
    input logic [9:0] rows,
    input logic [ADDR_WIDTH-1:0] addr,
    input logic [4:0] b_id, x_id, w_id,
    
    // CNN configuration parameters
    input logic [5:0] fmap_h, fmap_w, in_channels, out_channels,
    input logic [3:0] kernel_h, kernel_w,
    input logic [2:0] stride_val, pad_val, pool_size,
    input logic       relu_flag,        // CONV2D_RUN fused-ReLU bit (latched in top)

    // Results (same as original execution_unit)
    output logic signed [DATA_WIDTH-1:0] result [0:MAX_ROWS-1],
    output logic done,

    // Unified Memory Interface
    output logic                        mem_req,
    output logic                        mem_we,
    output logic [ADDR_WIDTH-1:0]       mem_addr,
    output logic [DATA_WIDTH-1:0]       mem_wdata,
    input  logic [DATA_WIDTH-1:0]       mem_rdata,
    input  logic                        mem_valid
);

    localparam TILE_ELEMS = TILE_WIDTH / DATA_WIDTH;
    
    // Main FSM states
    typedef enum logic [3:0] {
        IDLE,
        DISPATCH,
        WAIT_LOAD,
        WAIT_GEMV,
        WAIT_RELU,
        WAIT_STORE,
        WAIT_CONV2D,
        WAIT_MAXPOOL,
        COMPLETE
    } main_state_t;
    
    main_state_t state;
    
    // ========================================================================
    // Buffer Controller Signals
    // ========================================================================
    
    // Vector buffer write interface
    logic buf_vec_write_enable;
    logic [4:0] buf_vec_write_buffer_id;
    logic signed [DATA_WIDTH-1:0] buf_vec_write_tile [0:TILE_ELEMS-1];
    
    // Vector buffer read interface
    logic buf_vec_read_enable;
    logic [4:0] buf_vec_read_buffer_id;
    logic signed [DATA_WIDTH-1:0] buf_vec_read_tile [0:TILE_ELEMS-1];
    logic buf_vec_read_valid;
    
    // Matrix buffer write interface
    logic buf_mat_write_enable;
    logic [4:0] buf_mat_write_buffer_id;
    logic [TILE_WIDTH-1:0] buf_mat_write_tile;
    
    // Matrix buffer read interface
    logic buf_mat_read_enable;
    logic [4:0] buf_mat_read_buffer_id;
    logic signed [DATA_WIDTH-1:0] buf_mat_read_tile [0:TILE_ELEMS-1];
    logic buf_mat_read_valid;
    
    // ========================================================================
    // Load Execution Signals
    // ========================================================================
    
    logic load_start, load_done;
    logic load_vec_write_enable;
    logic [4:0] load_vec_write_buffer_id;
    logic signed [DATA_WIDTH-1:0] load_vec_write_tile [0:TILE_ELEMS-1];
    logic load_mat_write_enable;
    logic [4:0] load_mat_write_buffer_id;
    logic [TILE_WIDTH-1:0] load_mat_write_tile;
    
    // Load Memory signals
    logic                        load_mem_req;
    logic [ADDR_WIDTH-1:0]       load_mem_addr;
    
    // ========================================================================
    // GEMV Execution Signals
    // ========================================================================
    
    logic gemv_start, gemv_done;
    logic gemv_vec_read_enable;
    logic [4:0] gemv_vec_read_buffer_id;
    logic gemv_mat_read_enable;
    logic [4:0] gemv_mat_read_buffer_id;
    logic gemv_vec_write_enable;
    logic [4:0] gemv_vec_write_buffer_id;
    logic signed [DATA_WIDTH-1:0] gemv_vec_write_tile [0:TILE_ELEMS-1];
    logic signed [DATA_WIDTH-1:0] gemv_result [0:MAX_ROWS-1];
    
    // ========================================================================
    // ReLU Execution Signals
    // ========================================================================
    
    logic relu_start, relu_done;
    logic relu_vec_read_enable;
    logic [4:0] relu_vec_read_buffer_id;
    logic relu_vec_write_enable;
    logic [4:0] relu_vec_write_buffer_id;
    logic signed [DATA_WIDTH-1:0] relu_vec_write_tile [0:TILE_ELEMS-1];
    logic signed [DATA_WIDTH-1:0] relu_result [0:1023];
    
    // ========================================================================
    // Conv2D Execution Signals
    // ========================================================================
    
    logic conv2d_start, conv2d_done;
    logic [4:0] conv2d_vec_random_read_buffer_id;
    logic [$clog2(8192/DATA_WIDTH)-1:0] conv2d_vec_random_read_addr;
    logic conv2d_mat_read_enable;
    logic [4:0] conv2d_mat_read_buffer_id;
    logic conv2d_vec_write_enable;
    logic [4:0] conv2d_vec_write_buffer_id;
    logic signed [DATA_WIDTH-1:0] conv2d_vec_write_tile [0:TILE_ELEMS-1];
    
    // ========================================================================
    // MaxPool Execution Signals
    // ========================================================================
    
    logic maxpool_start, maxpool_done;
    logic [4:0] maxpool_vec_random_read_buffer_id;
    logic [$clog2(8192/DATA_WIDTH)-1:0] maxpool_vec_random_read_addr;
    logic maxpool_vec_write_enable;
    logic [4:0] maxpool_vec_write_buffer_id;
    logic signed [DATA_WIDTH-1:0] maxpool_vec_write_tile [0:TILE_ELEMS-1];
    
    // ========================================================================
    // Store Execution Signals
    // ========================================================================
    
    logic store_start, store_done;
    logic store_vec_read_enable;
    logic [4:0] store_vec_read_buffer_id;

    // Store Memory Signals
    logic                        store_mem_req;
    logic                        store_mem_we;
    logic [ADDR_WIDTH-1:0]       store_mem_addr;
    logic [DATA_WIDTH-1:0]       store_mem_wdata;
    
    // ========================================================================
    // Module Instantiations
    // ========================================================================
    
    // Buffer Controller
    buffer_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WIDTH(TILE_WIDTH),
        .TILE_ELEMS(TILE_ELEMS)
    ) buffer_ctrl (
        .clk(clk),
        .rst(rst),
        .vec_write_enable(buf_vec_write_enable),
        .vec_write_buffer_id(buf_vec_write_buffer_id),
        .vec_write_tile(buf_vec_write_tile),
        .vec_read_enable(buf_vec_read_enable),
        .vec_read_buffer_id(buf_vec_read_buffer_id),
        .vec_read_tile(buf_vec_read_tile),
        .vec_read_valid(buf_vec_read_valid),
        .vec_random_read_buffer_id(buf_vec_random_read_buffer_id),
        .vec_random_read_addr(buf_vec_random_read_addr),
        .vec_random_read_data(buf_vec_random_read_data),
        .mat_write_enable(buf_mat_write_enable),
        .mat_write_buffer_id(buf_mat_write_buffer_id),
        .mat_write_tile(buf_mat_write_tile),
        .mat_read_enable(buf_mat_read_enable),
        .mat_read_buffer_id(buf_mat_read_buffer_id),
        .mat_read_tile(buf_mat_read_tile),
        .mat_read_valid(buf_mat_read_valid),
        /* verilator lint_off PINCONNECTEMPTY */
        .vec_write_done(),
        .vec_read_done(),
        .mat_write_done(),
        .mat_read_done()
        /* verilator lint_on PINCONNECTEMPTY */
    );
    
    // Assign generic Combinational Vector Read port based on state 
    // Wait, Buffer Controller already exposes `vec_random_read_*` ports! We forgot to declare them here.
    // We added them to buffer_controller in standard interface.
    logic [4:0] buf_vec_random_read_buffer_id;
    logic [$clog2(8192/DATA_WIDTH)-1:0] buf_vec_random_read_addr;
    logic signed [DATA_WIDTH-1:0] buf_vec_random_read_data;
    
    assign buf_vec_random_read_buffer_id = (state == WAIT_CONV2D) ? conv2d_vec_random_read_buffer_id : 
                                           (state == WAIT_MAXPOOL) ? maxpool_vec_random_read_buffer_id : 5'd0;
    assign buf_vec_random_read_addr      = (state == WAIT_CONV2D) ? conv2d_vec_random_read_addr : 
                                           (state == WAIT_MAXPOOL) ? maxpool_vec_random_read_addr : 0;
    
    // Load Execution Module
    load_execution #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WIDTH(TILE_WIDTH),
        .TILE_ELEMS(TILE_ELEMS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) load_exec (
        .clk(clk),
        .rst(rst),
        .start(load_start),
        .opcode(opcode),
        .dest_buffer_id(dest),
        .length_or_cols(length_or_cols),
        .rows(rows),
        .addr(addr),
        .done(load_done),
        .vec_write_enable(load_vec_write_enable),
        .vec_write_buffer_id(load_vec_write_buffer_id),
        .vec_write_tile(load_vec_write_tile),
        .mat_write_enable(load_mat_write_enable),
        .mat_write_buffer_id(load_mat_write_buffer_id),
        .mat_write_tile(load_mat_write_tile),
        // Memory Interface
        .mem_req(load_mem_req),
        .mem_addr(load_mem_addr),
        .mem_rdata(mem_rdata),
        .mem_valid(mem_valid)
    );
    
    // GEMV Execution Module
    gemv_execution #(
        .DATA_WIDTH(DATA_WIDTH),
        // .TILE_WIDTH(TILE_WIDTH),
        .TILE_ELEMS(TILE_ELEMS),
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLS(MAX_COLS)
    ) gemv_exec (
        .clk(clk),
        .rst(rst),
        .start(gemv_start),
        .dest_buffer_id(dest),
        .w_buffer_id(w_id),
        .x_buffer_id(x_id),
        .b_buffer_id(b_id),
        .cols(length_or_cols),
        .rows(rows),
        .done(gemv_done),
        .vec_read_enable(gemv_vec_read_enable),
        .vec_read_buffer_id(gemv_vec_read_buffer_id),
        .vec_read_tile(buf_vec_read_tile),
        .vec_read_valid(buf_vec_read_valid),
        .mat_read_enable(gemv_mat_read_enable),
        .mat_read_buffer_id(gemv_mat_read_buffer_id),
        .mat_read_tile(buf_mat_read_tile),
        .mat_read_valid(buf_mat_read_valid),
        .vec_write_enable(gemv_vec_write_enable),
        .vec_write_buffer_id(gemv_vec_write_buffer_id),
        .vec_write_tile(gemv_vec_write_tile),
        .result(gemv_result)
    );
    
    // ReLU Execution Module
    relu_execution #(
        .DATA_WIDTH(DATA_WIDTH),
        // .TILE_WIDTH(TILE_WIDTH),
        .TILE_ELEMS(TILE_ELEMS)
    ) relu_exec (
        .clk(clk),
        .rst(rst),
        .start(relu_start),
        .dest_buffer_id(dest),
        .x_buffer_id(x_id),  // CRITICAL FIX: ReLU reads from x_id, not dest
        .length(length_or_cols),
        .done(relu_done),
        .vec_read_enable(relu_vec_read_enable),
        .vec_read_buffer_id(relu_vec_read_buffer_id),
        .vec_read_tile(buf_vec_read_tile),
        .vec_read_valid(buf_vec_read_valid),
        .vec_write_enable(relu_vec_write_enable),
        .vec_write_buffer_id(relu_vec_write_buffer_id),
        .vec_write_tile(relu_vec_write_tile),
        // Result Output
        .result(relu_result)
    );
    
    // Conv2D Execution Module
    conv2d_execution #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_ELEMS(TILE_ELEMS),
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLS(MAX_COLS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) conv2d_exec (
        .clk(clk),
        .rst(rst),
        .start(conv2d_start),
        .dest_buffer_id(dest),
        .w_buffer_id(w_id),
        .x_buffer_id(x_id),
        .b_buffer_id(b_id),
        .fmap_h(fmap_h),
        .fmap_w(fmap_w),
        .in_channels(in_channels),
        .out_channels(out_channels),
        .kernel_h(kernel_h),
        .kernel_w(kernel_w),
        .stride_val(stride_val),
        .pad_val(pad_val),
        .relu_flag(relu_flag),
        .done(conv2d_done),
        .vec_random_read_buffer_id(conv2d_vec_random_read_buffer_id),
        .vec_random_read_addr(conv2d_vec_random_read_addr),
        .vec_random_read_data(buf_vec_random_read_data),
        .mat_read_enable(conv2d_mat_read_enable),
        .mat_read_buffer_id(conv2d_mat_read_buffer_id),
        .mat_read_tile(buf_mat_read_tile),
        .mat_read_valid(buf_mat_read_valid),
        .vec_write_enable(conv2d_vec_write_enable),
        .vec_write_buffer_id(conv2d_vec_write_buffer_id),
        .vec_write_tile(conv2d_vec_write_tile)
    );
    
    // MaxPool Execution Module
    maxpool_execution #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_ELEMS(TILE_ELEMS)
    ) maxpool_exec (
        .clk(clk),
        .rst(rst),
        .start(maxpool_start),
        .dest_buffer_id(dest),
        .x_buffer_id(x_id),
        .fmap_h(fmap_h),
        .fmap_w(fmap_w),
        .in_channels(in_channels),
        .pool_size(pool_size),
        .stride_val(stride_val),
        .done(maxpool_done),
        .vec_random_read_buffer_id(maxpool_vec_random_read_buffer_id),
        .vec_random_read_addr(maxpool_vec_random_read_addr),
        .vec_random_read_data(buf_vec_random_read_data),
        .vec_write_enable(maxpool_vec_write_enable),
        .vec_write_buffer_id(maxpool_vec_write_buffer_id),
        .vec_write_tile(maxpool_vec_write_tile)
    );
    
    // Store Execution Module (placeholder)
    store_execution #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WIDTH(TILE_WIDTH),
        .TILE_ELEMS(TILE_ELEMS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) store_exec (
        .clk(clk),
        .rst(rst),
        .start(store_start),
        .src_buffer_id(dest),
        .length(length_or_cols),
        .addr(addr),
        .done(store_done),
        .vec_read_enable(store_vec_read_enable),
        .vec_read_buffer_id(store_vec_read_buffer_id),
        .vec_read_tile(buf_vec_read_tile),
        .vec_read_valid(buf_vec_read_valid),
        // Memory Interface
        .mem_req(store_mem_req),
        .mem_we(store_mem_we),
        .mem_addr(store_mem_addr),
        .mem_wdata(store_mem_wdata),
        .mem_ready(1'b1) // Assuming always ready for now or connected to mem_valid if ack needed
    );
    
    // ========================================================================
    // Buffer Controller Multiplexing
    // ========================================================================
    
    always_comb begin
        // Default: all interfaces idle
        buf_vec_write_enable = 0;
        buf_vec_write_buffer_id = 0;
        buf_vec_write_tile = '{default: 0};
        buf_vec_read_enable = 0;
        buf_vec_read_buffer_id = 0;
        buf_mat_write_enable = 0;
        buf_mat_write_buffer_id = 0;
        buf_mat_write_tile = 0;
        buf_mat_read_enable = 0;
        buf_mat_read_buffer_id = 0;

        // Default Memory Mux
        mem_req   = 0;
        mem_we    = 0;
        mem_addr  = 0;
        mem_wdata = 0;
        
        // Route signals based on active operation
        case (state)
            DISPATCH: begin
                // During DISPATCH, we're about to start an operation
                // Route signals based on opcode so the first request isn't lost
                case (opcode)
                    5'h01, 5'h02: begin // LOAD_V or LOAD_M about to start
                        buf_vec_write_enable = load_vec_write_enable;
                        buf_vec_write_buffer_id = load_vec_write_buffer_id;
                        buf_vec_write_tile = load_vec_write_tile;
                        buf_mat_write_enable = load_mat_write_enable;
                        buf_mat_write_buffer_id = load_mat_write_buffer_id;
                        buf_mat_write_tile = load_mat_write_tile;
                        // Memory Mux
                        mem_req  = load_mem_req;
                        mem_addr = load_mem_addr;
                        mem_we   = 0;
                    end
                    5'h04: begin // GEMV about to start
                        buf_vec_read_enable = gemv_vec_read_enable;
                        buf_vec_read_buffer_id = gemv_vec_read_buffer_id;
                        buf_mat_read_enable = gemv_mat_read_enable;
                        buf_mat_read_buffer_id = gemv_mat_read_buffer_id;
                        buf_vec_write_enable = gemv_vec_write_enable;
                        buf_vec_write_buffer_id = gemv_vec_write_buffer_id;
                        buf_vec_write_tile = gemv_vec_write_tile;
                    end
                    5'h05: begin // RELU about to start
                        buf_vec_read_enable = relu_vec_read_enable;
                        buf_vec_read_buffer_id = relu_vec_read_buffer_id;
                        buf_vec_write_enable = relu_vec_write_enable;
                        buf_vec_write_buffer_id = relu_vec_write_buffer_id;
                        buf_vec_write_tile = relu_vec_write_tile;
                    end
                    5'h07: begin // CONV2D about to start
                        buf_mat_read_enable = conv2d_mat_read_enable;
                        buf_mat_read_buffer_id = conv2d_mat_read_buffer_id;
                        buf_vec_write_enable = conv2d_vec_write_enable;
                        buf_vec_write_buffer_id = conv2d_vec_write_buffer_id;
                        buf_vec_write_tile = conv2d_vec_write_tile;
                    end
                    5'h08: begin // MAXPOOL about to start
                        buf_vec_write_enable = maxpool_vec_write_enable;
                        buf_vec_write_buffer_id = maxpool_vec_write_buffer_id;
                        buf_vec_write_tile = maxpool_vec_write_tile;
                    end
                    5'h03: begin // STORE about to start
                        buf_vec_read_enable = store_vec_read_enable;
                        buf_vec_read_buffer_id = store_vec_read_buffer_id;
                        // Memory Mux
                        mem_req   = store_mem_req;
                        mem_we    = store_mem_we;
                        mem_addr  = store_mem_addr;
                        mem_wdata = store_mem_wdata;
                    end
                    default: begin
                        // NOP or unknown - all idle
                    end
                endcase
            end
            
            WAIT_LOAD: begin
                // Load module has priority during load operations
                buf_vec_write_enable = load_vec_write_enable;
                buf_vec_write_buffer_id = load_vec_write_buffer_id;
                buf_vec_write_tile = load_vec_write_tile;
                buf_mat_write_enable = load_mat_write_enable;
                buf_mat_write_buffer_id = load_mat_write_buffer_id;
                buf_mat_write_tile = load_mat_write_tile;
                // Memory Mux
                mem_req  = load_mem_req;
                mem_addr = load_mem_addr;
                mem_we   = 0;
            end
            
            WAIT_GEMV: begin
                // GEMV reads from buffers and writes results
                buf_vec_read_enable = gemv_vec_read_enable;
                buf_vec_read_buffer_id = gemv_vec_read_buffer_id;
                buf_mat_read_enable = gemv_mat_read_enable;
                buf_mat_read_buffer_id = gemv_mat_read_buffer_id;
                buf_vec_write_enable = gemv_vec_write_enable;
                buf_vec_write_buffer_id = gemv_vec_write_buffer_id;
                buf_vec_write_tile = gemv_vec_write_tile;
            end
            
            WAIT_RELU: begin
                // ReLU reads and writes vector buffers
                buf_vec_read_enable = relu_vec_read_enable;
                buf_vec_read_buffer_id = relu_vec_read_buffer_id;
                buf_vec_write_enable = relu_vec_write_enable;
                buf_vec_write_buffer_id = relu_vec_write_buffer_id;
                buf_vec_write_tile = relu_vec_write_tile;
            end

            WAIT_CONV2D: begin
                // Conv2D reads from buffers and writes results
                buf_mat_read_enable = conv2d_mat_read_enable;
                buf_mat_read_buffer_id = conv2d_mat_read_buffer_id;
                buf_vec_write_enable = conv2d_vec_write_enable;
                buf_vec_write_buffer_id = conv2d_vec_write_buffer_id;
                buf_vec_write_tile = conv2d_vec_write_tile;
            end
            
            WAIT_MAXPOOL: begin
                // MaxPool reads sequentially and randomly, writes serially mapped to block
                buf_vec_write_enable = maxpool_vec_write_enable;
                buf_vec_write_buffer_id = maxpool_vec_write_buffer_id;
                buf_vec_write_tile = maxpool_vec_write_tile;
            end
            
            WAIT_STORE: begin
                // Store reads from vector buffer
                buf_vec_read_enable = store_vec_read_enable;
                buf_vec_read_buffer_id = store_vec_read_buffer_id;
                // Memory Mux
                mem_req   = store_mem_req;
                mem_we    = store_mem_we;
                mem_addr  = store_mem_addr;
                mem_wdata = store_mem_wdata;
            end
            
            default: begin
                // All idle
            end
        endcase
    end
    
    // ========================================================================
    // Main Control FSM
    // ========================================================================
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            load_start <= 0;
            gemv_start <= 0;
            relu_start <= 0;
            conv2d_start <= 0;
            store_start <= 0;
            
            for (int i = 0; i < MAX_ROWS; i++) begin
                result[i] <= 0;
            end
        end else begin
            // Default signal values
            done <= 0;
            load_start <= 0;
            gemv_start <= 0;
            relu_start <= 0;
            conv2d_start <= 0;
            maxpool_start <= 0;
            store_start <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        //$display("[MODULAR_EXEC] Received opcode 0x%h", opcode);
                        state <= DISPATCH;
                    end
                end
                
                DISPATCH: begin
                    // Route to appropriate execution module based on opcode
                    case (opcode)
                        5'h00: begin // NOP
                            //$display("[MODULAR_EXEC] NOP operation");
                            state <= COMPLETE;
                        end
                        
                        5'h01, 5'h02: begin // LOAD_V or LOAD_M
                            load_start <= 1;
                            state <= WAIT_LOAD;
                        end
                        
                        5'h03: begin // STORE
                            store_start <= 1;
                            state <= WAIT_STORE;
                        end
                        
                        5'h04: begin // GEMV
                            gemv_start <= 1;
                            state <= WAIT_GEMV;
                        end
                        
                        5'h05: begin // RELU
                            relu_start <= 1;
                            state <= WAIT_RELU;
                        end

                        5'h07: begin // CONV2D
                            conv2d_start <= 1;
                            state <= WAIT_CONV2D;
                        end
                        
                        5'h08: begin // MAXPOOL
                            maxpool_start <= 1;
                            state <= WAIT_MAXPOOL;
                        end
                        
                        default: begin
                            //$display("[MODULAR_EXEC] Unknown opcode: 0x%h", opcode);
                            state <= COMPLETE;
                        end
                    endcase
                end
                
                WAIT_LOAD: begin
                    if (load_done) begin
                        //$display("[MODULAR_EXEC] Load operation complete");
                        state <= COMPLETE;
                    end
                end
                
                WAIT_GEMV: begin
                    if (gemv_done) begin
                        //$display("[MODULAR_EXEC] GEMV operation complete");
                        // Copy GEMV results to result register
                        for (int i = 0; i < MAX_ROWS; i++) begin
                            result[i] <= gemv_result[i];
                        end
                        state <= COMPLETE;
                    end
                end
                
                WAIT_RELU: begin
                    if (relu_done) begin
                        //$display("[MODULAR_EXEC] ReLU operation complete");
                        // Copy ReLU results to result register
                        for (int i = 0; i < MAX_ROWS; i++) begin
                            result[i] <= relu_result[i];
                        end
                        state <= COMPLETE;
                    end
                end

                WAIT_CONV2D: begin
                    if (conv2d_done) begin
                        //$display("[MODULAR_EXEC] Conv2D operation complete");
                        state <= COMPLETE;
                    end
                end
                
                WAIT_MAXPOOL: begin
                    if (maxpool_done) begin
                        state <= COMPLETE;
                    end
                end
                
                WAIT_STORE: begin
                    if (store_done) begin
                        //$display("[MODULAR_EXEC] Store operation complete");
                        state <= COMPLETE;
                    end
                end
                
                COMPLETE: begin
                    //$display("[MODULAR_EXEC] Operation complete");
                    done <= 1;
                    state <= IDLE;
                    
                    // NOTE: Result output is handled by reading from buffers
                    // In the modular design, results stay in buffers for subsequent ops
                    // For compatibility with original interface, we could add
                    // a final read-back stage here if needed
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
