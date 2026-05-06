module conv2d_execution_tb_wrapper #(
    parameter DATA_WIDTH = 8,
    parameter TILE_ELEMS = 32,
    parameter MAX_ROWS = 1024,
    parameter MAX_COLS = 1024,
    parameter ADDR_WIDTH = 24
)(
    input logic clk,
    input logic rst,
    
    // Control interface triggering execution
    input logic start,
    input logic [4:0] dest_buffer_id,
    input logic [4:0] w_buffer_id,
    input logic [4:0] x_buffer_id,
    input logic [4:0] b_buffer_id,
    
    // Geometry bounds
    input logic [5:0] fmap_h, fmap_w, in_channels, out_channels,
    input logic [3:0] kernel_h, kernel_w,
    input logic [2:0] stride_val, pad_val,
    input logic       relu_flag,

    output logic done,
    
    // Bus tap for C++ buffer preloading
    input logic preload_en,
    input logic is_matrix,
    input logic [4:0] preload_buf_id,
    input logic signed [DATA_WIDTH-1:0] preload_data [0:TILE_ELEMS-1]
);

    // Wires between executor and buffers
    logic [4:0] vec_random_read_buffer_id;
    logic [$clog2(8192/DATA_WIDTH)-1:0] vec_random_read_addr;
    logic signed [DATA_WIDTH-1:0] vec_random_read_data;
    
    logic mat_read_enable;
    logic [4:0] mat_read_buffer_id;
    logic signed [DATA_WIDTH-1:0] mat_read_tile [0:TILE_ELEMS-1];
    logic mat_read_valid;
    
    logic vec_write_enable;
    logic [4:0] vec_write_buffer_id;
    logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1];

    logic [256-1:0] packed_matrix_data;
    logic vec_write_enable_mux;
    logic [4:0] vec_write_buffer_id_mux;
    logic signed [DATA_WIDTH-1:0] vec_write_tile_mux [0:TILE_ELEMS-1];
    
    always_comb begin
        if (preload_en && !is_matrix) begin
            vec_write_enable_mux = 1'b1;
            vec_write_buffer_id_mux = preload_buf_id;
            for (int i = 0; i < TILE_ELEMS; i++) vec_write_tile_mux[i] = preload_data[i];
        end else begin
            vec_write_enable_mux = vec_write_enable;
            vec_write_buffer_id_mux = vec_write_buffer_id;
            for (int i = 0; i < TILE_ELEMS; i++) vec_write_tile_mux[i] = vec_write_tile[i];
        end
        
        for (int i = 0; i < TILE_ELEMS; i++) begin
            packed_matrix_data[i*DATA_WIDTH +: DATA_WIDTH] = preload_data[i];
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
        
        .mat_write_enable(preload_en && is_matrix ? 1'b1 : 1'b0),
        .mat_write_buffer_id(preload_buf_id),
        .mat_write_tile(packed_matrix_data), 
        
        .mat_read_enable(mat_read_enable),
        .mat_read_buffer_id(mat_read_buffer_id),
        .mat_read_tile(mat_read_tile),
        .mat_read_valid(mat_read_valid),
        
        /* verilator lint_off PINCONNECTEMPTY */
        .vec_write_done(),
        .vec_read_done(),
        .mat_write_done(),
        .mat_read_done()
        /* verilator lint_on PINCONNECTEMPTY */
    );

    conv2d_execution #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_ELEMS(TILE_ELEMS),
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLS(MAX_COLS),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .dest_buffer_id(dest_buffer_id),
        .w_buffer_id(w_buffer_id),
        .x_buffer_id(x_buffer_id),
        .b_buffer_id(b_buffer_id),
        .fmap_h(fmap_h),
        .fmap_w(fmap_w),
        .in_channels(in_channels),
        .out_channels(out_channels),
        .kernel_h(kernel_h),
        .kernel_w(kernel_w),
        .stride_val(stride_val),
        .pad_val(pad_val),
        .relu_flag(relu_flag),
        .done(done),
        .vec_random_read_buffer_id(vec_random_read_buffer_id),
        .vec_random_read_addr(vec_random_read_addr),
        .vec_random_read_data(vec_random_read_data),
        .mat_read_enable(mat_read_enable),
        .mat_read_buffer_id(mat_read_buffer_id),
        .mat_read_tile(mat_read_tile),
        .mat_read_valid(mat_read_valid),
        .vec_write_enable(vec_write_enable),
        .vec_write_buffer_id(vec_write_buffer_id),
        .vec_write_tile(vec_write_tile)
    );

endmodule
