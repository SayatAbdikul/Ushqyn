// Buffer Controller Module
// Encapsulates vector and matrix buffer files, providing a unified interface
// for reading and writing tiles. Manages addressing and tile-based I/O operations.
//
// Features:
// - Dual buffer system: separate vector and matrix buffers
// - Tile-based read/write operations
// - Automatic tile indexing
// - Support for multiple concurrent buffer IDs

module buffer_controller #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter TILE_ELEMS = 32,
    parameter VECTOR_BUFFER_WIDTH = 8192,
    parameter MATRIX_BUFFER_WIDTH = 131072,
    parameter VECTOR_BUFFER_COUNT = 16,                     // Number of logical buffers
    parameter MATRIX_BUFFER_COUNT = 2                     // Number of logical buffers
)(
    input logic clk,
    input logic rst,
    
    // Vector buffer write interface
    input logic vec_write_enable,
    input logic [4:0] vec_write_buffer_id,
    input logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1],
    
    // Vector buffer read interface
    input logic vec_read_enable,
    input logic [4:0] vec_read_buffer_id,
    output logic signed [DATA_WIDTH-1:0] vec_read_tile [0:TILE_ELEMS-1],
    output logic vec_read_valid,
    
    // Vector buffer random-access read interface (combinational)
    input logic [4:0] vec_random_read_buffer_id,
    input logic [$clog2(VECTOR_BUFFER_WIDTH/DATA_WIDTH)-1:0] vec_random_read_addr,
    output logic signed [DATA_WIDTH-1:0] vec_random_read_data,
    
    // Matrix buffer write interface
    input logic mat_write_enable,
    input logic [4:0] mat_write_buffer_id,
    input logic [TILE_WIDTH-1:0] mat_write_tile,
    
    // Matrix buffer read interface
    input logic mat_read_enable,
    input logic [4:0] mat_read_buffer_id,
    output logic signed [DATA_WIDTH-1:0] mat_read_tile [0:TILE_ELEMS-1],
    output logic mat_read_valid,
    
    // Status signals
    output logic vec_write_done,
    output logic vec_read_done,
    output logic mat_write_done,
    output logic mat_read_done
);

    // Packed tile for vector writes
    logic [TILE_WIDTH-1:0] vec_write_tile_packed;
    
    // Delayed read enable signals for valid generation (buffer has 1-cycle read latency)
    logic vec_read_enable_d;
    logic mat_read_enable_d;
    
    // Pack vector tile data from unpacked array to packed bus
    always_comb begin
        vec_write_tile_packed = '0;
        for (int i = 0; i < TILE_ELEMS; i++) begin
            vec_write_tile_packed[i*DATA_WIDTH +: DATA_WIDTH] = vec_write_tile[i];
        end
    end
    
    // Track last accessed buffer and operation type for each buffer
    logic [4:0] vec_last_buffer_id, mat_last_buffer_id;
    logic vec_last_was_read, mat_last_was_read;
    logic vec_reset_indices, mat_reset_indices;
    
    // Detect when we switch buffers OR switch between read/write operations
    // Reset ensures we start from the correct tile index
    always_comb begin
        vec_reset_indices = 0;
        mat_reset_indices = 0;
        
        if (vec_read_enable) begin
            // Reset if different buffer OR switching from write to read on same buffer
            vec_reset_indices = (vec_read_buffer_id != vec_last_buffer_id) || 
                                (vec_read_buffer_id == vec_last_buffer_id && !vec_last_was_read);
        end else if (vec_write_enable) begin
            // Reset if different buffer OR switching from read to write on same buffer
            vec_reset_indices = (vec_write_buffer_id != vec_last_buffer_id) ||
                                (vec_write_buffer_id == vec_last_buffer_id && vec_last_was_read);
        end
        
        if (mat_read_enable) begin
            mat_reset_indices = (mat_read_buffer_id != mat_last_buffer_id) ||
                                (mat_read_buffer_id == mat_last_buffer_id && !mat_last_was_read);
        end else if (mat_write_enable) begin
            mat_reset_indices = (mat_write_buffer_id != mat_last_buffer_id) ||
                                (mat_write_buffer_id == mat_last_buffer_id && mat_last_was_read);
        end
    end
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            vec_last_buffer_id <= 5'h1F;  // Initialize to invalid value
            mat_last_buffer_id <= 5'h1F;
            vec_last_was_read <= 0;
            mat_last_was_read <= 0;
        end else begin
            // Track last operation
            if (vec_read_enable) begin
                vec_last_buffer_id <= vec_read_buffer_id;
                vec_last_was_read <= 1;
            end else if (vec_write_enable) begin
                vec_last_buffer_id <= vec_write_buffer_id;
                vec_last_was_read <= 0;
            end
            
            if (mat_read_enable) begin
                mat_last_buffer_id <= mat_read_buffer_id;
                mat_last_was_read <= 1;
            end else if (mat_write_enable) begin
                mat_last_buffer_id <= mat_write_buffer_id;
                mat_last_was_read <= 0;
            end
        end
    end
    
    // Select which buffer to reset (prioritize read operations)
    /* verilator lint_off UNUSEDSIGNAL */
    logic [4:0] vec_reset_buffer_id, mat_reset_buffer_id;
    /* verilator lint_on UNUSEDSIGNAL */
    assign vec_reset_buffer_id = vec_read_enable ? vec_read_buffer_id : vec_write_buffer_id;
    assign mat_reset_buffer_id = mat_read_enable ? mat_read_buffer_id : mat_write_buffer_id;
    
    // Vector Buffer File Instance
    // Optimized for smaller vectors (x, bias, intermediate activations)
    /* verilator lint_off WIDTHTRUNC */
    buffer_file #(
        .BUFFER_WIDTH(VECTOR_BUFFER_WIDTH),
        .BUFFER_COUNT(VECTOR_BUFFER_COUNT),
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_ELEMS)
    ) vector_buffer_inst (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(vec_write_enable),
        .read_enable(vec_read_enable),
        .write_data(vec_write_tile_packed),
        .write_buffer(vec_write_buffer_id),
        .read_buffer(vec_read_buffer_id),
        .read_data(vec_read_tile),
        .writing_done(vec_write_done),
        .reading_done(vec_read_done),
        .reset_indices_enable(vec_reset_indices),
        .reset_indices_buffer(vec_reset_buffer_id),
        .random_read_buffer(vec_random_read_buffer_id),
        .random_read_addr(vec_random_read_addr),
        .random_read_data(vec_random_read_data)
    );
    /* verilator lint_on WIDTHTRUNC */
    
    // Matrix Buffer File Instance
    // Optimized for larger weight matrices
    /* verilator lint_off WIDTHTRUNC */
    buffer_file #(
        .BUFFER_WIDTH(MATRIX_BUFFER_WIDTH),
        .BUFFER_COUNT(MATRIX_BUFFER_COUNT),
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_ELEMS)
    ) matrix_buffer_inst (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(mat_write_enable),
        .read_enable(mat_read_enable),
        .write_data(mat_write_tile),
        .write_buffer(mat_write_buffer_id),
        .read_buffer(mat_read_buffer_id),
        .read_data(mat_read_tile),
        .writing_done(mat_write_done),
        .reading_done(mat_read_done),
        .reset_indices_enable(mat_reset_indices),
        .reset_indices_buffer(mat_reset_buffer_id)
    );
    /* verilator lint_on WIDTHTRUNC */
    
    // Generate read valid signals (1 cycle after read enable)
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            vec_read_enable_d <= 0;
            mat_read_enable_d <= 0;
            vec_read_valid <= 0;
            mat_read_valid <= 0;
        end else begin
            vec_read_enable_d <= vec_read_enable;
            mat_read_enable_d <= mat_read_enable;
            vec_read_valid <= vec_read_enable_d;
            mat_read_valid <= mat_read_enable_d;
        end
    end

endmodule
