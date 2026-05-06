module buffer_file #(
    parameter DATA_WIDTH = 8,
    parameter BUFFER_WIDTH = 8192, // Default to vector size, override for matrix
    parameter BUFFER_COUNT = 2,
    parameter TILE_WIDTH = 256,
    parameter TILE_SIZE = 32
)(
    input clk,
    input reset_n,
    input write_enable,
    input read_enable,
    input [TILE_WIDTH-1:0] write_data,
    input [$clog2(BUFFER_COUNT)-1:0] write_buffer,
    input [$clog2(BUFFER_COUNT)-1:0] read_buffer,
    output reg [DATA_WIDTH-1:0] read_data [0:TILE_SIZE-1],
    output reg writing_done,
    output reg reading_done,
    // New: reset tile indices for a specific buffer
    input reset_indices_enable,
    input [$clog2(BUFFER_COUNT)-1:0] reset_indices_buffer,
    
    // New: Random access combinational read port
    input [$clog2(BUFFER_COUNT)-1:0] random_read_buffer,
    input [$clog2(BUFFER_WIDTH/DATA_WIDTH)-1:0] random_read_addr,
    output logic [DATA_WIDTH-1:0] random_read_data
);

// Calculate buffer parameters
localparam TILE_COUNT = BUFFER_WIDTH / TILE_WIDTH;
localparam TILE_INDEX_WIDTH = (TILE_COUNT == 1) ? 1 : $clog2(TILE_COUNT);

// Memory buffers
(* ram_style = "block", syn_ramstyle = "block_ram" *)
logic [BUFFER_WIDTH-1:0] buffers [0:BUFFER_COUNT-1];

// Tile indices - separate for each buffer
logic [TILE_INDEX_WIDTH-1:0] w_tile_index [0:BUFFER_COUNT-1];
logic [TILE_INDEX_WIDTH-1:0] r_tile_index [0:BUFFER_COUNT-1];

// Random access combinational logic
assign random_read_data = buffers[random_read_buffer][random_read_addr * DATA_WIDTH +: DATA_WIDTH];


// Edge detection for read_enable (only read on rising edge)
logic read_enable_prev;

// Reset and operation logic
integer i;
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        read_enable_prev <= 0;
        // Reset buffers and indices
        for (i = 0; i < BUFFER_COUNT; i++) begin
            // verilator lint_off WIDTHCONCAT
            buffers[i] <= {BUFFER_WIDTH{1'b0}};
            // verilator lint_on WIDTHCONCAT
            w_tile_index[i] <= 0;
            r_tile_index[i] <= 0;
        end
        for (i = 0; i < TILE_SIZE; i++) begin
            read_data[i] <= 0;
        end
        writing_done <= 0;
        reading_done <= 0;
    end else begin
        // Default done signals
        writing_done <= 0;
        reading_done <= 0;
        
        // Update edge detection
        read_enable_prev <= read_enable;

        // Write operation
        if (write_enable) begin
            // Use index 0 if resetting this buffer, otherwise use current index
            logic [TILE_INDEX_WIDTH-1:0] effective_w_index;
            effective_w_index = (reset_indices_enable && reset_indices_buffer == write_buffer) ? 
                                0 : w_tile_index[write_buffer];
            
            buffers[write_buffer][(effective_w_index * TILE_WIDTH) +: TILE_WIDTH] <= write_data;
            //$display("[BUFFER] Write to buffer %0d tile %0d", write_buffer, effective_w_index);
            
            if ({ {(32-TILE_INDEX_WIDTH){1'b0}}, effective_w_index } == TILE_COUNT - 1) begin
                w_tile_index[write_buffer] <= 0;
                writing_done <= 1;
            end else begin
                w_tile_index[write_buffer] <= effective_w_index + 1;
            end
        end 
        // Read operation - only on rising edge of read_enable
        if (read_enable && !read_enable_prev) begin
            // Use index 0 if resetting this buffer, otherwise use current index
            logic [TILE_INDEX_WIDTH-1:0] effective_r_index;
            effective_r_index = (reset_indices_enable && reset_indices_buffer == read_buffer) ?
                                0 : r_tile_index[read_buffer];
            
            for (i = 0; i < TILE_SIZE; i++) begin
                read_data[i] <= buffers[read_buffer][(effective_r_index * TILE_WIDTH + i * 8) +: DATA_WIDTH];
            end
            //$display("[BUFFER] Read from buffer %0d tile %0d", read_buffer, effective_r_index);
            
            if ({ {(32-TILE_INDEX_WIDTH){1'b0}}, effective_r_index } == TILE_COUNT - 1) begin
                r_tile_index[read_buffer] <= 0;
                reading_done <= 1;
            end else begin
                r_tile_index[read_buffer] <= effective_r_index + 1;
            end
        end
    end
end

endmodule

