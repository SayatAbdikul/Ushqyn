module maxpool_execution_tb_wrapper #(
    parameter DATA_WIDTH = 8,
    parameter TILE_ELEMS = 32,
    parameter ADDR_WIDTH = 24
)(
    input logic clk,
    input logic rst,
    
    // Control interface triggering execution
    input logic start,
    input logic [4:0] dest_buffer_id,
    input logic [4:0] x_buffer_id,
    
    // Geometry bounds
    input logic [5:0] fmap_h, fmap_w, in_channels,
    input logic [2:0] pool_size, stride_val,
    
    output logic done,
    
    // Bus tap for C++ buffer preloading
    input logic preload_en,
    input logic [4:0] preload_buf_id,
    input logic signed [DATA_WIDTH-1:0] preload_data [0:TILE_ELEMS-1]
);

    // Wires between executor and buffers
    logic [4:0] vec_random_read_buffer_id;
    logic [$clog2(8192/DATA_WIDTH)-1:0] vec_random_read_addr;
    logic signed [DATA_WIDTH-1:0] vec_random_read_data;
    
    logic vec_write_enable;
    logic [4:0] vec_write_buffer_id;
    logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1];

    logic vec_write_enable_mux;
    logic [4:0] vec_write_buffer_id_mux;
    logic signed [DATA_WIDTH-1:0] vec_write_tile_mux [0:TILE_ELEMS-1];
    
    always_comb begin
        if (preload_en) begin
            vec_write_enable_mux = 1'b1;
            vec_write_buffer_id_mux = preload_buf_id;
            for (int i = 0; i < TILE_ELEMS; i++) vec_write_tile_mux[i] = preload_data[i];
        end else begin
            vec_write_enable_mux = vec_write_enable;
            vec_write_buffer_id_mux = vec_write_buffer_id;
            for (int i = 0; i < TILE_ELEMS; i++) vec_write_tile_mux[i] = vec_write_tile[i];
        end
    end
    
    buffer_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_ELEMS(TILE_ELEMS)
    ) buffers (
        .clk(clk),
        .rst(rst),
        .vec_write_enable(vec_write_enable_mux),
        .vec_write_buffer_id(vec_write_buffer_id_mux),
        .vec_write_tile(vec_write_tile_mux),
        .vec_read_enable(1'b0),
        .vec_read_buffer_id(5'd0),
        /* verilator lint_off PINCONNECTEMPTY */
        .vec_read_tile(),
        .vec_read_valid(),
        /* verilator lint_on PINCONNECTEMPTY */
        .vec_random_read_buffer_id(vec_random_read_buffer_id),
        .vec_random_read_addr(vec_random_read_addr),
        .vec_random_read_data(vec_random_read_data),
        
        .mat_write_enable(1'b0),
        .mat_write_buffer_id(5'd0),
        .mat_write_tile('0), 
        
        .mat_read_enable(1'b0),
        .mat_read_buffer_id(5'd0),
        /* verilator lint_off PINCONNECTEMPTY */
        .mat_read_tile(),
        .mat_read_valid(),
        
        .vec_write_done(),
        .vec_read_done(),
        .mat_write_done(),
        .mat_read_done()
        /* verilator lint_on PINCONNECTEMPTY */
    );

    maxpool_execution #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_ELEMS(TILE_ELEMS)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .dest_buffer_id(dest_buffer_id),
        .x_buffer_id(x_buffer_id),
        .fmap_h(fmap_h),
        .fmap_w(fmap_w),
        .in_channels(in_channels),
        .pool_size(pool_size),
        .stride_val(stride_val),
        .done(done),
        .vec_random_read_buffer_id(vec_random_read_buffer_id),
        .vec_random_read_addr(vec_random_read_addr),
        .vec_random_read_data(vec_random_read_data),
        .vec_write_enable(vec_write_enable),
        .vec_write_buffer_id(vec_write_buffer_id),
        .vec_write_tile(vec_write_tile)
    );

endmodule
