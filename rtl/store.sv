module store #(
    parameter ADDR_WIDTH = 24,
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter TILE_ELEMS = TILE_WIDTH / DATA_WIDTH
)(
    input  logic                        clk,
    input  logic                        rst,
    input  logic                        start,
    input  logic [ADDR_WIDTH-1:0]       dram_addr,
    input  logic [17:0]                 length,      // 18-bit per ISA (STORE shares LOAD_V's length encoding)
    input  logic [4:0]                  buf_id,      // which vector buffer to read from

    // Connect to the shared vector buffer file for reads
    output logic                        buf_read_en,
    output logic [4:0]                  buf_read_id,
    input  logic [DATA_WIDTH-1:0]       buf_read_data [0:TILE_ELEMS-1],
    input  logic                        buf_read_done,

    output logic                        done,
    
    // Memory Interface
    output logic                        mem_req,
    output logic                        mem_we,
    output logic [ADDR_WIDTH-1:0]       mem_addr,
    output logic [DATA_WIDTH-1:0]       mem_wdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                        mem_ready // Ack/busy
    /* verilator lint_on UNUSEDSIGNAL */
);

    // Simple memory interface
    // logic                               mem_we; -> Output
    // logic [ADDR_WIDTH-1:0]              mem_addr; -> Output
    // logic [DATA_WIDTH-1:0]              mem_din; -> Output as mem_wdata
    /* verilator lint_off UNUSEDSIGNAL */
    logic                               mem_dump;
    /* verilator lint_on UNUSEDSIGNAL */

    typedef enum logic [2:0] { S_IDLE, S_REQ_TILE, S_WRITE_TILE, S_ADVANCE, S_FINISH } s_state_t;
    s_state_t state;

    logic [ADDR_WIDTH-1:0] base_addr;
    // 18-bit length × 8 bits/elem = up to 21 bits; 24 bits gives headroom.
    logic [23:0]           written_bits;
    // Use exact index width for TILE_ELEMS entries
    logic [$clog2(TILE_ELEMS)-1:0] elem_idx;
    logic read_requested;

    // Constant mapping of buffer id
    assign buf_read_id = buf_id;

    // Helper: zero-extend elem_idx to address width
    wire [ADDR_WIDTH-1:0] elem_idx_ext = {{(ADDR_WIDTH-$bits(elem_idx)){1'b0}}, elem_idx};

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state        <= S_IDLE;
            done         <= 1'b0;
            base_addr    <= '0;
            written_bits <= '0;
            elem_idx     <= '0;
            buf_read_en  <= 1'b0;
            read_requested <= 1'b0;
            mem_we       <= 1'b0;
            mem_dump     <= 1'b0;
            mem_addr     <= '0;
            mem_wdata    <= '0;
        end else begin
            // Defaults each cycle
            done        <= 1'b0;
            buf_read_en <= 1'b0;
            mem_we      <= 1'b0;
            mem_dump    <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        base_addr    <= dram_addr;
                        written_bits <= 24'd0;
                        elem_idx     <= '0;
                        read_requested <= 1'b0; // Reset read_requested when starting a new operation
                        state        <= S_REQ_TILE;
                    end
                end

                S_REQ_TILE: begin
                    // Request a tile from buffer - pulse read for exactly 1 cycle
                    if (!read_requested) begin
                        buf_read_en <= 1'b1;
                        read_requested <= 1'b1;
                    end
                    if (buf_read_done) begin
                        elem_idx <= '0;
                        read_requested <= 1'b0;
                        state    <= S_WRITE_TILE;
                    end
                end

                S_WRITE_TILE: begin
                    // Write elements of current tile until either tile done or length satisfied
                    if (written_bits < length*DATA_WIDTH) begin
                        mem_addr <= base_addr + elem_idx_ext;
                        mem_wdata  <= buf_read_data[elem_idx];
                        mem_we   <= 1'b1;
                        //$display("[STORE_DBG] TileWrite: base=0x%h, idx=%0d, wdata=0x%h (from buf[%0d])", 
                        //         base_addr, elem_idx, buf_read_data[elem_idx], elem_idx);
                        mem_dump <= 1'b1;
                        written_bits <= written_bits + DATA_WIDTH;
                        if (int'(elem_idx) == TILE_ELEMS-1) begin
                            state <= S_ADVANCE;
                        end
                        elem_idx <= elem_idx + 1'b1;
                    end else begin
                        state <= S_FINISH;
                    end
                end

                S_ADVANCE: begin
                    // Move to next tile if more to go
                    if (written_bits < length*DATA_WIDTH) begin
                        base_addr <= base_addr + TILE_ELEMS[ADDR_WIDTH-1:0];
                        state     <= S_REQ_TILE;
                    end else begin
                        state     <= S_FINISH;
                    end
                end

                S_FINISH: begin
                    // Trigger memory dump to update dram.hex
                    mem_dump <= 1'b1;
                    done     <= 1'b1;
                    state    <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    assign mem_req = mem_we; 

endmodule
