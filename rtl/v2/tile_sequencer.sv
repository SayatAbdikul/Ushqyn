// Autonomous DMA/compute command list. Program: 2048 x 16 bytes in BSRAM.
// Words: {opcode, flags, reserved16}, arg0, arg1, arg2 (little endian).
// 0 HALT; 1 DMA (flags[0]=to SRAM, ext, SRAM, bytes);
// 2 RUN (PC, {live_end16,live_base16}, 0); 3 WAIT (flags: engine=1,DMA=2).
module v2_tile_sequencer (
    input logic clk, rst_n, start, abort_run,
    input logic [10:0] start_index,
    output logic busy, abort_units,
    output logic [7:0] error_code,
    output logic [31:0] elapsed, engine_cycles, dma_cycles, overlap_cycles,
    output logic [11:0] command_index,
    input logic host_req, host_wr,
    input logic [23:0] host_addr,
    input logic [63:0] host_wdata,
    input logic [7:0] host_wstrb,
    output logic host_ready, host_rvalid,
    output logic [63:0] host_rdata,
    output logic engine_start,
    output logic [23:0] engine_pc,
    input logic engine_busy,
    input logic [7:0] engine_error,
    output logic dma_start, dma_to_sram,
    output logic [23:0] dma_ext, dma_sram,
    output logic [31:0] dma_length,
    input logic dma_busy,
    input logic [7:0] dma_error
);
    typedef enum logic [3:0] {IDLE, FETCH0, READ0, FETCH1, READ1,
                              EXECUTE, ADVANCE, DRAIN} state_t;
    state_t state;
    logic [63:0] low_word, high_word;
    logic [15:0] live_base, live_end;
    logic memory_req, memory_ready, memory_rvalid;
    logic [63:0] memory_rdata;
    wire fetching = state == FETCH0 || state == FETCH1;
    wire [7:0] op = low_word[7:0];
    wire [7:0] flags = low_word[15:8];
    wire [31:0] arg0 = low_word[63:32];
    wire [31:0] arg1 = high_word[31:0];
    wire [31:0] arg2 = high_word[63:32];
    wire [32:0] dma_end = {1'b0,arg1} + {1'b0,arg2};
    assign memory_req = busy ? fetching : host_req;
    assign host_ready = !busy && memory_ready;
    assign host_rvalid = !busy && memory_rvalid;
    assign host_rdata = memory_rdata;
    v2_scratchpad #(.MEM_BYTES(32768)) program_memory (
        .clk(clk), .rst_n(rst_n), .req(memory_req),
        .wr(!busy && host_wr),
        .addr(busy ? {9'b0,command_index[10:0],(state==FETCH1),3'b0} :
                       {9'b0,host_addr[14:0]}),
        .wdata(host_wdata), .wstrb(host_wstrb),
        .ready(memory_ready), .rvalid(memory_rvalid), .rdata(memory_rdata)
    );
    task automatic fail(input logic [7:0] code);
        begin error_code <= code; abort_units <= 1; state <= DRAIN; end
    endtask
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=IDLE;busy<=0;abort_units<=0;error_code<=0;
            elapsed<=0;engine_cycles<=0;dma_cycles<=0;overlap_cycles<=0;
            command_index<=0;low_word<=0;high_word<=0;live_base<=0;live_end<=32768;
            engine_start<=0;engine_pc<=0;dma_start<=0;dma_to_sram<=0;
            dma_ext<=0;dma_sram<=0;dma_length<=0;
        end else begin
            engine_start<=0;dma_start<=0;abort_units<=0;
            if (busy) begin
                elapsed<=elapsed+1;
                if (engine_busy) engine_cycles<=engine_cycles+1;
                if (dma_busy) dma_cycles<=dma_cycles+1;
                if (engine_busy && dma_busy) overlap_cycles<=overlap_cycles+1;
            end
            if (abort_run) begin
                if (busy) fail(8'd6);
                else error_code<=0;
            end else if (busy && state!=DRAIN && elapsed==32'h7fffffff)
                fail(8'd7);
            else case (state)
                IDLE: if (start) begin
                    busy<=1;error_code<=0;elapsed<=0;engine_cycles<=0;
                    dma_cycles<=0;overlap_cycles<=0;command_index<={1'b0,start_index};
                    state<=FETCH0;
                end
                FETCH0: if (memory_ready) state<=READ0;
                READ0: if (memory_rvalid) begin low_word<=memory_rdata;state<=FETCH1;end
                FETCH1: if (memory_ready) state<=READ1;
                READ1: if (memory_rvalid) begin high_word<=memory_rdata;state<=EXECUTE;end
                EXECUTE: begin
                    if (low_word[31:16]!=0) fail(8'd10);
                    else case (op)
                        0: if (!engine_busy && !dma_busy) begin
                            if (engine_error!=0 || dma_error!=0)
                                fail(engine_error!=0 ? engine_error : dma_error);
                            else begin busy<=0;state<=IDLE;end
                        end
                        1: if (!dma_busy) begin
                            if (flags>1 || arg0>=8388608 || arg0[2:0]!=0 ||
                                arg1[2:0]!=0 || arg2==0 || dma_end>32768 ||
                                {1'b0,arg0}+{1'b0,arg2}>8388608) fail(8'd1);
                            else if (engine_busy && arg1<{16'b0,live_end} &&
                                     dma_end>{17'b0,live_base}) fail(8'd9);
                            else begin
                                dma_to_sram<=flags[0];dma_ext<=arg0[23:0];
                                dma_sram<=arg1[23:0];dma_length<=arg2;
                                dma_start<=1;state<=ADVANCE;
                            end
                        end
                        2: if (!engine_busy && !dma_busy) begin
                            if (arg0[5:0]!=0 || arg0>32704 ||
                                arg1[15:0]>=arg1[31:16] || arg1[31:16]>32768 ||
                                arg0<{16'b0,arg1[15:0]} || arg0+64>{16'b0,arg1[31:16]})
                                fail(8'd1);
                            else begin
                                live_base<=arg1[15:0];live_end<=arg1[31:16];
                                engine_pc<=arg0[23:0];engine_start<=1;state<=ADVANCE;
                            end
                        end
                        3: if ((!flags[0] || !engine_busy) && (!flags[1] || !dma_busy)) begin
                            if ((flags[0] && engine_error!=0) || (flags[1] && dma_error!=0))
                                fail(flags[0] && engine_error!=0 ? engine_error : dma_error);
                            else state<=ADVANCE;
                        end
                        default: fail(8'd10);
                    endcase
                end
                ADVANCE: if (command_index==2047) fail(8'd7);
                         else begin command_index<=command_index+1;state<=FETCH0;end
                DRAIN: if (!engine_busy && !dma_busy) begin busy<=0;state<=IDLE;end
                default: fail(8'd10);
            endcase
        end
    end
endmodule
