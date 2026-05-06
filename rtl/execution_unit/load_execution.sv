// Load Execution Module
// Handles LOAD_V (load vector) and LOAD_M (load matrix) operations
// Coordinates load_v and load_m modules with buffer controller
//
// Operation Flow:
// 1. Receives start signal with opcode (LOAD_V or LOAD_M)
// 2. Activates appropriate load module (load_v or load_m)
// 3. Receives tiles from load module
// 4. Writes tiles to buffer controller
// 5. Signals done when complete

module load_execution #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter TILE_ELEMS = TILE_WIDTH / DATA_WIDTH,
    parameter ADDR_WIDTH = 24
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0]  opcode,           // 0x01=LOAD_V, 0x02=LOAD_M
    input logic [4:0]  dest_buffer_id,   // Target buffer to write to
    // 18-bit length for LOAD_V (matches assembler/golden_model contract);
    // LOAD_M only uses the lower 10 bits (cols field).
    input logic [17:0] length_or_cols,
    input logic [9:0]  rows,             // Matrix rows (unused for LOAD_V)
    input logic [ADDR_WIDTH-1:0] addr,  // DRAM address to load from
    output logic done,
    
    // Buffer controller interface - vector writes
    output logic vec_write_enable,
    output logic [4:0] vec_write_buffer_id,
    output logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1],
    
    // Buffer controller interface - matrix writes
    output logic mat_write_enable,
    output logic [4:0] mat_write_buffer_id,
    output logic [TILE_WIDTH-1:0] mat_write_tile,
    
    // Unified Memory Interface
    output logic mem_req,
    output logic [ADDR_WIDTH-1:0] mem_addr,
    input  logic [DATA_WIDTH-1:0] mem_rdata,
    input  logic mem_valid
);

    // FSM states
    typedef enum logic [1:0] {
        IDLE,
        LOADING_VECTOR,
        LOADING_MATRIX,
        COMPLETE
    } load_state_t;
    
    load_state_t state;
    
    // Load_v module signals
    logic load_v_start, load_v_done, load_v_tile_ready;
    logic signed [DATA_WIDTH-1:0] load_v_tile [0:TILE_ELEMS-1];
    
    // Load_m module signals
    logic load_m_start, load_m_done, load_m_tile_ready;
    logic [TILE_WIDTH-1:0] load_m_tile;
    
    // Tile counters for tracking progress
    logic [9:0] tile_count;

    // Memory Logic
    logic load_v_mem_req;
    logic [ADDR_WIDTH-1:0] load_v_mem_addr;
    logic load_m_mem_req;
    logic [ADDR_WIDTH-1:0] load_m_mem_addr;

    // MUX Memory Requests based on State/Opcode
    always_comb begin
        if (state == LOADING_MATRIX || opcode == 5'h02) begin
            mem_req  = load_m_mem_req;
            mem_addr = load_m_mem_addr;
        end else begin
            // Default to vector load (safe even in IDLE)
            mem_req  = load_v_mem_req;
            mem_addr = load_v_mem_addr;
        end
    end
    
    // Load_v module instantiation
    // Loads vector data from memory and outputs tiles
    load_v #(
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) load_v_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(load_v_start),
        .dram_addr(addr),
        .length(length_or_cols),
        .data_out(load_v_tile),
        .tile_out(load_v_tile_ready),
        .valid_out(load_v_done),
        // Memory Interface
        .mem_req(load_v_mem_req),
        .mem_addr(load_v_mem_addr),
        .mem_rdata(mem_rdata), // Shared input
        .mem_valid(mem_valid)  // Shared input
    );
    
    // Load_m module instantiation
    // Loads matrix data from memory and outputs tiles
    // Now row-aware: stores each row aligned to tile boundaries
    load_m #(
        .TILE_WIDTH(TILE_WIDTH)
    ) load_m_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(load_m_start),
        .dram_addr(addr),
        .rows(rows),
        .cols(length_or_cols[9:0]),    // LOAD_M cols stays 10-bit per ISA
        .data_out(load_m_tile),
        .tile_out(load_m_tile_ready),
        .valid_out(load_m_done),
        // Memory Interface
        .mem_req(load_m_mem_req),
        .mem_addr(load_m_mem_addr),
        .mem_rdata(mem_rdata), // Shared input
        .mem_valid(mem_valid)  // Shared input
    );
    
    // Control logic for buffer writes
    // Vector writes: triggered by load_v tile ready
    always_comb begin
        vec_write_enable = load_v_tile_ready;
        vec_write_buffer_id = dest_buffer_id;
        vec_write_tile = load_v_tile;
    end
    
    // Debug: Log what load_v is outputting
    always @(posedge clk) begin
        if (load_v_tile_ready && dest_buffer_id == 9 && tile_count == 0) begin
            //$display("[LOAD_EXEC_DBG] load_v outputting tile 0 to buffer 9, first 4 values: %h %h %h %h",
            //         load_v_tile[0], load_v_tile[1], load_v_tile[2], load_v_tile[3]);
        end
    end
    
    // Matrix writes: triggered by load_m tile ready
    always_comb begin
        mat_write_enable = load_m_tile_ready;
        mat_write_buffer_id = dest_buffer_id;
        mat_write_tile = load_m_tile;
    end
    
    // Main FSM for load execution
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            load_v_start <= 0;
            load_m_start <= 0;
            tile_count <= 0;
        end else begin
            // Default signal values
            done <= 0;
            load_v_start <= 0;
            load_m_start <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        tile_count <= 0;
                        
                        case (opcode)
                            5'h01: begin // LOAD_V
                                load_v_start <= 1;
                                state <= LOADING_VECTOR;
                                //$display("[LOAD_EXEC] Starting LOAD_V: dest=%0d, length=%0d, addr=0x%h",
                                //         dest_buffer_id, length_or_cols, addr);
                            end
                            
                            5'h02: begin // LOAD_M
                                load_m_start <= 1;
                                state <= LOADING_MATRIX;
                                //$display("[LOAD_EXEC] Starting LOAD_M: dest=%0d, rows=%0d, cols=%0d, total_elements=%0d, addr=0x%h",
                                //         dest_buffer_id, rows, length_or_cols, rows * length_or_cols, addr);
                            end
                            
                            default: begin
                                // Invalid opcode for this module
                                state <= COMPLETE;
                            end
                        endcase
                    end
                end
                
                LOADING_VECTOR: begin
                    // Track tiles being written
                    if (load_v_tile_ready) begin
                        tile_count <= tile_count + 1;
                        //$display("[LOAD_EXEC] Vector tile %0d written to buffer %0d",
                        //         tile_count + 1, dest_buffer_id);
                    end
                    
                    // Complete when load_v signals done
                    if (load_v_done) begin
                        // Account for tile_ready and done happening in same cycle
                        //$display("[LOAD_EXEC] LOAD_V complete: %0d tiles written", 
                        //         load_v_tile_ready ? tile_count + 1 : tile_count);
                        state <= COMPLETE;
                    end
                end
                
                LOADING_MATRIX: begin
                    // Track tiles being written
                    if (load_m_tile_ready) begin
                        tile_count <= tile_count + 1;
                        //$display("[LOAD_EXEC] Matrix tile %0d written to buffer %0d",
                        //         tile_count + 1, dest_buffer_id);
                    end
                    
                    // Complete when load_m signals done
                    if (load_m_done) begin
                        // Account for tile_ready and done happening in same cycle
                        //$display("[LOAD_EXEC] LOAD_M complete: %0d tiles written", 
                        //         load_m_tile_ready ? tile_count + 1 : tile_count);
                        state <= COMPLETE;
                    end
                end
                
                COMPLETE: begin
                    done <= 1;
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
