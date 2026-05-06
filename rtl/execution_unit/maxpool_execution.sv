// MaxPool Execution Module
// Slides a downsampling spatial window over a dynamic feature map geometry,
// accumulating explicit 8-bit maximums entirely utilizing the random access memory bus 
// to bypass complex systolic routing blockades without needing explicit tile packing FSMs.

module maxpool_execution #(
    parameter DATA_WIDTH = 8,
    parameter TILE_ELEMS = 32
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0] dest_buffer_id,
    input logic [4:0] x_buffer_id,
    
    // Geometry bounds
    input logic [5:0] fmap_h, fmap_w, in_channels,
    input logic [2:0] pool_size, stride_val,
    output logic done,
    
    // Random access for sequential unrolling of 2D geometry matrices
    output logic [4:0] vec_random_read_buffer_id,
    output logic [$clog2(8192/DATA_WIDTH)-1:0] vec_random_read_addr,
    input logic signed [DATA_WIDTH-1:0] vec_random_read_data,
    
    // Standard stream emitter bus
    output logic vec_write_enable,
    output logic [4:0] vec_write_buffer_id,
    output logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1]
);

    logic [5:0] out_h, out_w;
    logic [9:0] total_out_pixels;
    
    logic [5:0] oh, ow, c;
    logic [2:0] kh, kw;
    
    // State machine definition
    typedef enum logic [3:0] {
        IDLE,
        INIT,
        LOOP_INIT,
        WINDOW_FETCH,
        WINDOW_EVAL,
        EMIT_PIXEL,
        DONE
    } state_t;
    
    state_t state;
    
    logic signed [DATA_WIDTH-1:0] max_val;
    logic [11:0] quant_ptr_out; // Number of flushed elements 
    
    assign vec_random_read_buffer_id = x_buffer_id;
    
    // Math address coordinate mapping (NCHW: channel outermost, matches
    // conv2d_execution's output layout and golden_model.maxpool's input).
    logic [5:0] ih, iw;
    assign ih = oh * stride_val + kh;
    assign iw = ow * stride_val + kw;

    assign vec_random_read_addr = c * fmap_h * fmap_w + ih * fmap_w + iw;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            vec_write_enable <= 0;
            for (int i = 0; i < TILE_ELEMS; i++) vec_write_tile[i] <= 0;
        end else begin
            vec_write_enable <= 0;
            done <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        // Using truncated div down behavior mimicking valid padding bounds
                        out_h <= (fmap_h - pool_size) / stride_val + 1;
                        out_w <= (fmap_w - pool_size) / stride_val + 1;
                        state <= INIT;
                    end
                end
                
                INIT: begin
                    total_out_pixels <= out_h * out_w;
                    oh <= 0;
                    ow <= 0;
                    c <= 0;
                    quant_ptr_out <= 0;
                    state <= LOOP_INIT;
                end
                
                LOOP_INIT: begin
                    // Outermost counter is `c` (NCHW). When it rolls past
                    // in_channels, every (c, oh, ow) has been emitted.
                    if (c >= in_channels) begin
                        state <= DONE;
                    end else begin
                        kh <= 0;
                        kw <= 0;
                        max_val <= -128; // smallest signed int8
                        state <= WINDOW_FETCH;
                    end
                end
                
                WINDOW_FETCH: begin
                    // Request initiated combinationally via assign logic.
                    // Wait one cycle for eval if needed, though combinational block RAM port 
                    // is available instantaneously. To respect FSM clocking margins we 
                    // fetch into eval.
                    state <= WINDOW_EVAL;
                end
                
                WINDOW_EVAL: begin
                    if (vec_random_read_data > max_val) max_val <= vec_random_read_data;
                    
                    if (kw == pool_size - 1) begin
                        kw <= 0;
                        if (kh == pool_size - 1) begin
                            state <= EMIT_PIXEL;
                        end else begin
                            kh <= kh + 1;
                            state <= WINDOW_FETCH;
                        end
                    end else begin
                        kw <= kw + 1;
                        state <= WINDOW_FETCH;
                    end
                end
                
                EMIT_PIXEL: begin
                    vec_write_tile[quant_ptr_out % TILE_ELEMS] <= max_val;
                    quant_ptr_out <= quant_ptr_out + 1;
                    
                    if ((quant_ptr_out % TILE_ELEMS) == TILE_ELEMS - 1 || 
                         quant_ptr_out == (total_out_pixels * in_channels) - 1) begin
                        vec_write_enable <= 1;
                        vec_write_buffer_id <= dest_buffer_id;
                    end
                    
                    // Increment 3D traversal loops in NCHW order: ow inner,
                    // oh middle, c outer. Channel changes only after a full
                    // out_h × out_w plane has been emitted, matching the
                    // [c, oh, ow] layout consumers expect.
                    if (ow == out_w - 1) begin
                        ow <= 0;
                        if (oh == out_h - 1) begin
                            oh <= 0;
                            c  <= c + 1;
                        end else begin
                            oh <= oh + 1;
                        end
                    end else begin
                        ow <= ow + 1;
                    end

                    state <= LOOP_INIT;
                end
                
                DONE: begin
                    done <= 1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
            
        end
    end
endmodule
