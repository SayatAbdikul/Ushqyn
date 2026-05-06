// Store Execution Module
// Handles STORE operations - writes vector data from buffer to memory
// Uses the store.sv module for actual DRAM writes
//
// Operation Flow:
// 1. Start the store module with source buffer and DRAM address
// 2. Store module reads tiles from buffer and writes to DRAM
// 3. Signal done when store completes

module store_execution #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter TILE_ELEMS = TILE_WIDTH / DATA_WIDTH,
    parameter ADDR_WIDTH = 24
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0]  src_buffer_id,    // Source buffer to read from
    input logic [17:0] length,           // 18-bit per ISA (STORE shares LOAD_V's length encoding)
    input logic [ADDR_WIDTH-1:0] addr,   // DRAM address to write to
    output logic done,
    
    // Buffer controller interface - vector reads
    // Buffer controller interface - vector reads
    output logic vec_read_enable,
    output logic [4:0] vec_read_buffer_id,
    input logic signed [DATA_WIDTH-1:0] vec_read_tile [0:TILE_ELEMS-1],
    input logic vec_read_valid,
    
    // Memory Interface
    output logic                        mem_req,
    output logic                        mem_we,
    output logic [ADDR_WIDTH-1:0]       mem_addr,
    output logic [DATA_WIDTH-1:0]       mem_wdata,
    input  logic                        mem_ready
);

    // Store module control signals
    logic store_start;
    logic store_done;
    
    // Connect vec_read_tile to unsigned array for store module
    logic [DATA_WIDTH-1:0] buf_read_data [0:TILE_ELEMS-1];
    
    // Convert signed to unsigned (reinterpret bits)
    always_comb begin
        for (int i = 0; i < TILE_ELEMS; i++) begin
            buf_read_data[i] = vec_read_tile[i];
        end
    end
    
    // Instantiate store module
    store #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WIDTH(TILE_WIDTH),
        .TILE_ELEMS(TILE_ELEMS)
    ) store_inst (
        .clk(clk),
        .rst(rst),
        .start(store_start),
        .dram_addr(addr),
        .length(length),
        .buf_id(src_buffer_id),
        .buf_read_en(vec_read_enable),
        .buf_read_id(vec_read_buffer_id),
        .buf_read_data(buf_read_data),
        .buf_read_done(vec_read_valid),
        .done(store_done),
        // Memory Interface
        .mem_req(mem_req),
        .mem_we(mem_we),
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_ready(mem_ready)
    );
    
    // Simple control FSM
    typedef enum logic [1:0] {
        IDLE,
        STORING,
        COMPLETE
    } store_state_t;
    
    store_state_t state;
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            store_start <= 0;
        end else begin
            // Default signal values
            done <= 0;
            store_start <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        store_start <= 1;
                        state <= STORING;
                        //$display("[STORE_EXEC] Starting STORE: src_buf=%0d, length=%0d, addr=0x%h",
                        //         src_buffer_id, length, addr);
                    end
                end
                
                STORING: begin
                    if (store_done) begin
                        state <= COMPLETE;
                    end
                end
                
                COMPLETE: begin
                    //$display("[STORE_EXEC] STORE execution complete");
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
